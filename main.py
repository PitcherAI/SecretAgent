import os
import json
import base64
import httpx
import re
import io
import csv
import statistics
import asyncio
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

app = FastAPI()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# 1. Grok 4.1 Fast (New Free Tier - Excellent Reasoning)
# 2. Llama 3.3 70B (Reliable Generalist)
# 3. DeepSeek V3 (High Performance Backup)
# 4. Gemini 2.0 Flash (Experimental, use as last resort due to 429s)
MODEL_PRIORITY = [
    "x-ai/grok-4.1-fast:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat:free",
    "google/gemini-2.0-flash-exp:free"
]

# Specialized model just for Vision tasks (Grok 4.1 Fast can be text-only)
VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"

client = AsyncOpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL")
STUDENT_SECRET = os.environ.get("STUDENT_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ------------------------------------------------------------------
# HELPER: ROBUST LLM CALLER
# ------------------------------------------------------------------
async def query_llm(messages, response_format=None, force_model=None):
    """
    Tries models in order. If one is rate-limited, it waits or switches.
    """
    # If a specific model is forced (e.g. for vision), use only that one
    priority_list = [force_model] if force_model else MODEL_PRIORITY

    for model in priority_list:
        retries = 3
        delay = 5
        
        for attempt in range(retries):
            try:
                print(f"🧠 Asking {model} (Attempt {attempt+1})...")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format
                )
                return response 
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "upstream" in error_msg:
                    print(f"⏳ Rate Limited on {model}. Waiting {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2 
                else:
                    print(f"⚠️ API Error on {model}: {e}")
                    break 
        
        print(f"⏭️ Skipping {model}, trying next fallback...")
    
    raise Exception("❌ All models failed.")

# ------------------------------------------------------------------
# HELPER: FETCH EXTERNAL DATA
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    print(f"📥 Fetching: {url}")
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(url, headers=headers, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            if is_binary:
                return resp.content
            return resp.text
    except Exception as e:
        print(f"⚠️ Fetch Error: {e}")
        return None

# ------------------------------------------------------------------
# HELPER: MATH ENGINE
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction, metric="sum"):
    try:
        numbers = []
        try:
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                for cell in row:
                    clean_cell = cell.strip().replace(',', '')
                    if re.match(r'^-?\d+(\.\d+)?$', clean_cell):
                        numbers.append(float(clean_cell))
        except:
            pass

        if not numbers:
            lines = [l.strip().replace(',', '') for l in content.split('\n') if l.strip()]
            for line in lines:
                if re.match(r'^-?\d+(\.\d+)?$', line):
                    numbers.append(float(line))
        
        if not numbers: return None

        print(f"🧮 Math Engine: {len(numbers)} nums. Filter: {direction} {cutoff_val}. Metric: {metric}")
        cutoff = float(cutoff_val)
        
        if direction in ["<", "below", "less"]: filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "at most", "up to"]: filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above", "more"]: filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least"]: filtered = [n for n in numbers if n >= cutoff]
        elif direction in ["=", "equal"]: filtered = [n for n in numbers if n == cutoff]
        else: filtered = numbers 

        if not filtered: return 0

        metric = metric.lower()
        if metric == "count": result = len(filtered)
        elif metric in ["mean", "average"]: result = statistics.mean(filtered)
        elif metric in ["max", "maximum"]: result = max(filtered)
        elif metric in ["min", "minimum"]: result = min(filtered)
        else: result = sum(filtered)

        if isinstance(result, float) and result.is_integer(): result = int(result)
        elif isinstance(result, float): result = round(result, 4)

        print(f"✅ Calculation Result: {result}")
        return result
    except Exception as e:
        print(f"Math Error: {e}")
        return None

# ------------------------------------------------------------------
# CORE AGENT LOGIC
# ------------------------------------------------------------------
async def solve_quiz(start_url: str):
    print(f"🚀 Starting background task for: {start_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context()
        page = await context.new_page()
        
        current_url = start_url
        
        while current_url:
            print(f"🔗 Navigating to: {current_url}")
            try:
                await page.goto(current_url, timeout=45000)
                await page.wait_for_load_state("networkidle")
                
                if "congratulations" in page.url.lower():
                    print("🎉 Quiz Completed!")
                    break

                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                # --- MAIN REASONING STEP (Using Grok 4.1) ---
                # We separate text/image here. We send Text to Grok.
                # If Grok needs to see the image, we rely on its vision capabilities 
                # OR fallback to the specific vision model in the 'scrape' phase.
                
                system_prompt = """
                You are an autonomous agent.
                1. Analyze the HTML. 
                2. If there is a question requiring visual analysis, return {"action": "visual_analyze"}.
                3. If you need to download/scrape, return {"action": "scrape", "scrape_url": "...", "submit_url": "..."}.
                4. If you have the answer, return {"action": "submit", "answer": <value>, "submit_url": "..."}.
                5. If math is needed: {"action": "scrape", "scrape_url": "...", "math_filter": {"cutoff": 10, "direction": "<=", "metric": "sum"}}.
                6. Default submit_url: "/submit".
                """
                
                user_content = [{"type": "text", "text": f"HTML Content:\n{html_content}"}]

                # Try attaching image to Grok, but be prepared for it to ignore it if text-only
                # user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})

                response = await query_llm(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"}
                )
                
                llm_output = json.loads(response.choices[0].message.content)
                print(f"🤖 Action Plan: {llm_output}")

                answer = None
                raw_submit_url = llm_output.get("submit_url")

                # SPECIAL VISUAL CHECK (If Grok says "I need to see the image")
                if llm_output.get("action") == "visual_analyze":
                    print("🖼️ Grok requested visual analysis. Switching to Vision Model...")
                    vision_resp = await query_llm(
                        messages=[
                            {"role": "system", "content": "Analyze image. Return JSON: {\"answer\": <value>}"},
                            {"role": "user", "content": [
                                {"type": "text", "text": "What is the answer based on this screenshot?"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                            ]}
                        ],
                        response_format={"type": "json_object"},
                        force_model=VISION_MODEL 
                    )
                    answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                # HANDLE SCRAPE
                elif llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    headers = llm_output.get("headers", None)
                    target_url = urljoin(current_url, raw_scrape_url)
                    
                    print(f"🔎 Scraping: {target_url}")
                    path = urlparse(target_url).path.lower()
                    
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                        if img_bytes:
                            b64_scraped = base64.b64encode(img_bytes).decode('utf-8')
                            # Force Vision Model for Scraped Images
                            vision_resp = await query_llm(
                                messages=[
                                    {"role": "system", "content": "Analyze image. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Question?"},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scraped}"}}
                                    ]}
                                ],
                                response_format={"type": "json_object"},
                                force_model=VISION_MODEL
                            )
                            answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                    elif path.endswith(('.csv', '.txt', '.json', '.xml')):
                        scraped_data = await fetch_external_content(target_url, headers=headers)
                        
                        math_req = llm_output.get("math_filter")
                        if math_req and scraped_data:
                            answer = perform_filtered_math(
                                scraped_data, 
                                math_req.get("cutoff"), 
                                math_req.get("direction"),
                                math_req.get("metric", "sum")
                            )

                        if answer is None and scraped_data:
                            truncated_data = scraped_data[:50000] 
                            follow_up = await query_llm(
                                messages=[
                                    {"role": "system", "content": "Analyze data. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": f"Data:\n{truncated_data}"}
                                ],
                                response_format={"type": "json_object"}
                            )
                            answer = json.loads(follow_up.choices[0].message.content).get("answer")

                    else:
                        page2 = await context.new_page()
                        await page2.goto(target_url, timeout=30000)
                        scraped_html = await page2.content()
                        await page2.close()
                        
                        follow_up = await query_llm(
                            messages=[
                                {"role": "system", "content": "Analyze page. Return JSON: {\"answer\": <value>}"},
                                {"role": "user", "content": f"Main HTML: {html_content}\n\nScraped HTML:\n{scraped_html}"}
                            ],
                            response_format={"type": "json_object"}
                        )
                        answer = json.loads(follow_up.choices[0].message.content).get("answer")

                else:
                    answer = llm_output.get("answer")

                if answer is None:
                    # Fallback: If Grok couldn't decide, maybe it missed visual context.
                    # Try one last hail mary with vision model on the screenshot
                    print("⚠️ No answer from text analysis. Trying Visual Analysis fallback...")
                    vision_resp = await query_llm(
                        messages=[
                            {"role": "system", "content": "Return JSON: {\"answer\": <value>}"},
                            {"role": "user", "content": [
                                {"type": "text", "text": "What is the answer to the question on this screen?"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                            ]}
                        ],
                        response_format={"type": "json_object"},
                        force_model=VISION_MODEL
                    )
                    answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                if isinstance(answer, str) and "<svg" in answer:
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                if raw_submit_url:
                     parsed_sub = urlparse(raw_submit_url)
                     if not parsed_sub.path or parsed_sub.path == "/":
                         submit_url = urljoin(current_url, "/submit")
                     else:
                         submit_url = urljoin(current_url, raw_submit_url)
                else:
                    submit_url = urljoin(current_url, "/submit")

                if isinstance(answer, dict):
                    candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url', 'answer']]
                    answer = candidates[0] if candidates else json.dumps(answer)

                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"📤 Submitting to {submit_url}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    try:
                        resp_data = resp.json()
                    except:
                        print(f"🔥 Server Error: {resp.text[:200]}")
                        break
                    
                print(f"✅ Result: {resp_data}")
                
                if resp_data.get("correct"):
                    current_url = resp_data.get("url")
                else:
                    print(f"⛔ Incorrect: {resp_data.get('reason')}")
                    break
                    
            except Exception as e:
                print(f"🔥 Critical Error: {e}")
                break
        
        await browser.close()
        print("🏁 Task Finished.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
