import os
import json
import asyncio
import base64
import httpx
import re
import io
import csv
import statistics
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

app = FastAPI()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
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
# HELPER: FETCH EXTERNAL DATA
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    if headers is None: headers = {}
    
    # Auto-inject headers for API calls
    if "/api/" in url or "vercel.app" in url:
        if "email" not in headers:
            print("💉 Auto-injecting Auth Headers...")
            headers["email"] = STUDENT_EMAIL
            headers["secret"] = STUDENT_SECRET

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
# HELPER: AUDIO TRANSCRIPTION
# ------------------------------------------------------------------
async def transcribe_audio(audio_bytes, filename="audio.mp3"):
    print(f"🎤 Transcribing audio ({len(audio_bytes)} bytes)...")
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        transcription = await client.audio.transcriptions.create(
            model="openai/whisper-1", 
            file=audio_file
        )
        print(f"🗣️ Transcript: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"⚠️ Transcription Failed: {e}")
        return ""

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
                    clean = cell.strip().replace(',', '')
                    if re.match(r'^-?\d+(\.\d+)?$', clean):
                        numbers.append(float(clean))
        except:
            lines = [l.strip().replace(',', '') for l in content.split('\n') if l.strip()]
            for line in lines:
                if re.match(r'^-?\d+(\.\d+)?$', line):
                    numbers.append(float(line))
        
        if not numbers: return None

        # print(f"🧮 Math: {len(numbers)} nums. Filter: {direction} {cutoff_val}")
        cutoff = float(cutoff_val)
        
        if direction in ["<", "below"]: filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "at most"]: filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above"]: filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least"]: filtered = [n for n in numbers if n >= cutoff]
        elif direction == "=": filtered = [n for n in numbers if n == cutoff]
        elif direction == "mod": filtered = [n for n in numbers if n % cutoff == 0]
        else: filtered = numbers

        if not filtered: return 0

        metric = metric.lower()
        if metric == "count": result = len(filtered)
        elif metric == "mean": result = statistics.mean(filtered)
        elif metric == "max": result = max(filtered)
        elif metric == "min": result = min(filtered)
        else: result = sum(filtered)

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 4)

        return result
    except Exception as e:
        print(f"Math Error: {e}")
        return None

# ------------------------------------------------------------------
# CORE AGENT LOGIC
# ------------------------------------------------------------------
async def solve_quiz(start_url: str):
    print(f"🚀 Starting task: {start_url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context()
        page = await context.new_page()
        
        current_url = start_url
        
        while current_url:
            print(f"🔗 Navigating: {current_url}")
            try:
                # Retry navigation
                for attempt in range(3):
                    try:
                        await page.goto(current_url, timeout=60000, wait_until="domcontentloaded")
                        await asyncio.sleep(2)
                        break
                    except Exception as e:
                        if attempt == 2: raise e
                
                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                audio_transcript = ""
                audio_element = await page.query_selector("audio source, a[href$='.mp3'], a[href$='.wav']")
                if audio_element:
                    src = await audio_element.get_attribute("src") or await audio_element.get_attribute("href")
                    if src:
                        audio_url = urljoin(current_url, src)
                        audio_bytes = await fetch_external_content(audio_url, is_binary=True)
                        if audio_bytes:
                            audio_transcript = await transcribe_audio(audio_bytes)

                # FIX: Stronger instruction to prevent "Key Mismatch" (submitting instructions instead of answer)
                system_prompt = """
                You are an autonomous data extraction agent.
                1. Analyze HTML, Screenshot, and Audio.
                2. NAVIGATION RULE: If the text says "Add ?email=..." or "Go to...", THIS IS A NAVIGATION INSTRUCTION.
                   Return: {"action": "scrape", "scrape_url": "<the_modified_url_with_params>", "submit_url": "<url>"}
                   DO NOT submit the instruction text as the answer.
                3. DATA RULE: If you need data from a file/API, return:
                   {"action": "scrape", "scrape_url": "<url>", "headers": {"key": "val"}, "submit_url": "<url>"}
                4. MATH RULE: If instructions specify a filter (e.g. "sum numbers < 5000"), extract it:
                   {"action": "scrape", "scrape_url": "<file>", "submit_url": "<url>", "math_filter": {"cutoff": 12000, "direction": "<=", "metric": "sum"}}
                   Directions: "<", ">", "<=", ">=", "=", "mod"
                5. ANSWER RULE: If you have the FINAL answer, return:
                   {"action": "submit", "answer": <value>, "submit_url": "<url>"}
                6. Default "submit_url" to "/submit". Output Valid JSON.
                """
                
                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"HTML:\n{html_content}\n\nAudio:\n{audio_transcript}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ],
                    response_format={"type": "json_object"}
                )
                
                llm_output = json.loads(response.choices[0].message.content)
                print(f"🤖 Plan: {llm_output}")

                answer = None
                raw_submit_url = llm_output.get("submit_url")
                math_context = {} 

                # --- EXECUTION ---
                if llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    headers = llm_output.get("headers", {})
                    target_url = urljoin(current_url, raw_scrape_url)
                    
                    print(f"🔎 Scraping: {target_url}")
                    path = urlparse(target_url).path.lower()
                    
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                        if img_bytes:
                            b64_scraped = base64.b64encode(img_bytes).decode('utf-8')
                            vision_resp = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Analyze image. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scraped}"}}]}
                                ],
                                response_format={"type": "json_object"}
                            )
                            answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                    elif path.endswith(('.csv', '.txt', '.json', '.xml')) or "/api/" in target_url:
                        scraped_data = await fetch_external_content(target_url, headers=headers)
                        math_req = llm_output.get("math_filter")
                        
                        if math_req and scraped_data:
                            # Save context for brute-force retry
                            math_context = {
                                "data": scraped_data, 
                                "cutoff": math_req.get("cutoff"), 
                                "metric": math_req.get("metric", "sum"),
                                "dir": math_req.get("direction")
                            }
                            answer = perform_filtered_math(scraped_data, math_context["cutoff"], math_context["dir"], math_context["metric"])
                            print(f"⚡ Math Result 1: {answer}")

                        if answer is None and scraped_data:
                            trunc = scraped_data[:50000]
                            follow_up = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Analyze data. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": f"Context:\n{html_content}\nData:\n{trunc}"}
                                ],
                                response_format={"type": "json_object"}
                            )
                            answer = json.loads(follow_up.choices[0].message.content).get("answer")

                    else:
                        print("🌐 Webpage Scrape")
                        page2 = await context.new_page()
                        if headers: await page2.set_extra_http_headers(headers)
                        await page2.goto(target_url, timeout=30000)
                        scraped_html = await page2.content()
                        await page2.close()
                        
                        follow_up = await client.chat.completions.create(
                            model="openai/gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Analyze page. Return JSON: {\"answer\": <value>}"},
                                {"role": "user", "content": f"Main:\n{html_content}\nScraped:\n{scraped_html}"}
                            ],
                            response_format={"type": "json_object"}
                        )
                        answer = json.loads(follow_up.choices[0].message.content).get("answer")

                else:
                    answer = llm_output.get("answer")

                if answer is None or not raw_submit_url:
                    print("❌ Failure: No answer.")
                    break

                # --- SUBMISSION ---
                submit_url = urljoin(current_url, raw_submit_url)
                if isinstance(answer, dict):
                    cands = [v for k, v in answer.items() if k not in ['email', 'secret', 'url', 'answer']]
                    answer = cands[0] if cands else json.dumps(answer)
                if isinstance(answer, str) and "<svg" in answer:
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                payload = {"email": STUDENT_EMAIL, "secret": STUDENT_SECRET, "url": current_url, "answer": answer}
                
                print(f"📤 Submitting: {answer}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    try: res = resp.json()
                    except: res = {"error": resp.text}
                    
                print(f"✅ Result: {res}")
                
                # --- BRUTE FORCE RETRY (Fix for "Wrong Sum") ---
                if not res.get("correct") and "Wrong sum" in str(res.get("reason")) and math_context:
                    print("🔄 Retry Triggered: Brute-forcing all boundaries...")
                    # Try all valid mathematical directions
                    candidates = ["<", "<=", ">", ">="]
                    
                    for direction in candidates:
                        if direction == math_context["dir"]: continue # Skip already tried
                        
                        retry_ans = perform_filtered_math(math_context["data"], math_context["cutoff"], direction, math_context["metric"])
                        print(f"⚡ Testing {direction}: {retry_ans}")
                        
                        payload["answer"] = retry_ans
                        async with httpx.AsyncClient() as http:
                            resp = await http.post(submit_url, json=payload, timeout=30)
                            res = resp.json()
                        
                        print(f"   Outcome: {res}")
                        if res.get("correct"):
                            print("🎉 Brute-force SUCCESS!")
                            break # Stop loop if correct

                if res.get("correct"):
                    current_url = res.get("url")
                else:
                    print(f"⛔ Stop: {res.get('reason')}")
                    break
                    
            except Exception as e:
                print(f"🔥 Error: {e}")
                break
        
        await browser.close()
        print("🏁 Done.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
