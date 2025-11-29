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
from openai import AsyncOpenAI, RateLimitError, APIError

app = FastAPI()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# We use a list of models. If the first fails (429), we try the next.
# 1. Gemini 2.0 Flash (Fastest, Smartest)
# 2. Llama 3.3 70B (Very Stable Free Tier)
MODEL_PRIORITY = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat:free"
]

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
# HELPER: ROBUST LLM CALLER (Fixes 429 Errors)
# ------------------------------------------------------------------
async def query_llm(messages, response_format=None):
    """
    Tries models in order. If one is rate-limited, it waits or switches.
    """
    for model in MODEL_PRIORITY:
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
                return response # Success!
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check for Rate Limits (429) or Overloaded server
                if "429" in error_msg or "rate limit" in error_msg or "upstream" in error_msg:
                    print(f"⏳ Rate Limited on {model}. Waiting {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2 # Exponential backoff (5s -> 10s -> 20s)
                else:
                    print(f"⚠️ API Error on {model}: {e}")
                    break # Fatal error for this model, try next one
        
        print(f"⏭️ Skipping {model}, trying next fallback...")
    
    raise Exception("❌ All models failed or rate limited.")

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
                
                # Check if we landed on a "completed" or "correct" static page (stop condition)
                if "congratulations" in page.url.lower():
                    print("🎉 Quiz Completed!")
                    break

                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                user_content = [
                    {"type": "text", "text": f"HTML Content:\n{html_content}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]

                # AUDIO
                audio_element = await page.query_selector("audio source, a[href$='.mp3'], a[href$='.wav']")
                if audio_element:
                    src = await audio_element.get_attribute("src") or await audio_element.get_attribute("href")
                    if src:
                        audio_url = urljoin(current_url, src)
                        print(f"🎤 Found Audio: {audio_url}")
                        audio_bytes = await fetch_external_content(audio_url, is_binary=True)
                        if audio_bytes:
                            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                            # Note: Only Gemini supports direct audio. If fallback to Llama happens,
                            # Llama will ignore this or error. For stability, we might need transcription
                            # but let's trust Gemini works most of the time.
                            user_content.append({
                                "type": "image_url", 
                                "image_url": {"url": f"data:audio/mp3;base64,{b64_audio}"}
                            })

                print("👀 Context acquired. Planning action...")

                system_prompt = """
                You are an autonomous data extraction agent.
                1. Analyze HTML/Screenshot.
                2. Return JSON.
                3. ACTIONS:
                   - {"action": "scrape", "scrape_url": "<url>", "headers": {"key": "val"}, "submit_url": "<url>"}
                   - {"action": "scrape", "scrape_url": "<url>", "math_filter": {"cutoff": 10, "direction": "<=", "metric": "sum"}}
                   - {"action": "submit", "answer": <value>, "submit_url": "<url>"}
                4. For Charts: Generate SVG code.
                5. Default submit_url: "/submit"
                """
                
                # --- NEW CALL METHOD ---
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

                # HANDLE SCRAPE
                if llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    headers = llm_output.get("headers", None)
                    target_url = urljoin(current_url, raw_scrape_url)
                    
                    print(f"🔎 Scraping: {target_url}")
                    path = urlparse(target_url).path.lower()
                    
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                        if img_bytes:
                            b64_scraped = base64.b64encode(img_bytes).decode('utf-8')
                            vision_resp = await query_llm(
                                messages=[
                                    {"role": "system", "content": "Analyze image. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Question?"},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scraped}"}}
                                    ]}
                                ],
                                response_format={"type": "json_object"}
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
                        # Webpage scraping
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

                if answer is None or not raw_submit_url:
                    print("❌ Failure: No answer found.")
                    break

                # SVG / Dictionary cleanup
                if isinstance(answer, str) and "<svg" in answer:
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                parsed_sub = urlparse(raw_submit_url)
                if not parsed_sub.path or parsed_sub.path == "/":
                    submit_url = urljoin(current_url, "/submit")
                else:
                    submit_url = urljoin(current_url, raw_submit_url)

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
                # Don't break immediately on critical error, maybe retry the loop?
                # For now, we break to avoid infinite loops on fatal errors.
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
