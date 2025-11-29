import os
import json
import asyncio
import base64
import httpx
import re
import io
import csv
import statistics
import sys
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

# Try importing standard data libs
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

try:
    import pypdf
except ImportError:
    pypdf = None

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
            headers["email"] = STUDENT_EMAIL
            headers["secret"] = STUDENT_SECRET

    print(f"📥 Fetching: {url} (Headers: {list(headers.keys())})")
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
# HELPER: PDF PARSING
# ------------------------------------------------------------------
def read_pdf(pdf_bytes):
    if not pypdf: return "Error: pypdf not installed. Cannot read PDF."
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join([page.extract_text() for page in reader.pages[:10]])
        return text[:20000]
    except Exception as e:
        return f"PDF Error: {e}"

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
# HELPER: UNIVERSAL PYTHON EXECUTOR
# ------------------------------------------------------------------
def execute_python(code, context_data=""):
    print("🐍 Executing Python Logic...")
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    local_scope = {
        "data": context_data,
        "csv": csv,
        "json": json,
        "statistics": statistics,
        "re": re,
        "math": __import__("math"),
        "pd": pd,
        "np": np
    }
    
    try:
        exec(code, {}, local_scope)
        sys.stdout = old_stdout
        result = redirected_output.getvalue()
        if not result.strip() and "answer" in local_scope:
            result = str(local_scope["answer"])
        print(f"🐍 Output: {result.strip()[:100]}...")
        return result.strip()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Python Error: {e}"

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

        print(f"🧮 Math: {len(numbers)} nums. Filter: {direction} {cutoff_val}. Metric: {metric}")
        cutoff = float(cutoff_val)
        
        if direction in ["<", "below", "less"]:
            filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "at most", "up to", "max", "maximum"]:
            filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above", "more", "greater"]:
            filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least", "min", "minimum"]:
            filtered = [n for n in numbers if n >= cutoff]
        elif direction in ["=", "=="]:
            filtered = [n for n in numbers if n == cutoff]
        elif direction in ["%", "mod"]:
            filtered = [n for n in numbers if n % cutoff == 0]
        else:
            filtered = numbers

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
async def solve_quiz(start_url: str, user_email: str, user_secret: str):
    print(f"🚀 Starting task: {start_url} for {user_email}")
    
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
                for attempt in range(3):
                    try:
                        await page.goto(current_url, timeout=60000, wait_until="domcontentloaded")
                        await asyncio.sleep(2)
                        break
                    except Exception as e:
                        if attempt == 2: raise e
                
                # --- CONTEXT ---
                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                # --- AUDIO DETECT ---
                audio_transcript = ""
                audio_element = await page.query_selector("audio source, a[href$='.mp3'], a[href$='.wav']")
                if audio_element:
                    src = await audio_element.get_attribute("src") or await audio_element.get_attribute("href")
                    if src:
                        audio_url = urljoin(current_url, src)
                        audio_bytes = await fetch_external_content(audio_url, is_binary=True)
                        if audio_bytes:
                            audio_transcript = await transcribe_audio(audio_bytes)

                # --- RESILIENCE LOOP (Retry on same page) ---
                feedback_history = []
                success = False
                
                # Try up to 5 times on the same URL before giving up
                for attempt_idx in range(5):
                    if attempt_idx > 0:
                        print(f"🔄 Retry Attempt {attempt_idx + 1}/5 on same page...")

                    # --- PLANNING ---
                    system_prompt = f"""
                    You are an autonomous data extraction agent.
                    YOUR EMAIL: {user_email}
                    
                    1. Analyze HTML, Screenshot, and Audio.
                    2. PREVIOUS MISTAKES: {json.dumps(feedback_history)}
                       *If previous attempts failed, TRY A DIFFERENT APPROACH (e.g., scrape a DIFFERENT link, flip math logic, or change scraping target).*
                    
                    3. STARTING: If the page asks to start/begin or for an email, the answer is "{user_email}".
                       Return: {{"action": "submit", "answer": "{user_email}", "submit_url": "<url>"}}
                    
                    4. COMMANDS: 
                       - If page says "submit the command string" or "not the output", SUBMIT the full command text (e.g. `uv http get...`).
                       - If page says "run this command", SCRAPE the URL in the command.
                    
                    5. DATA SCRAPING: If you need data from a file/API/PDF, return:
                       {{"action": "scrape", "scrape_url": "<url>", "headers": {{"key": "val"}}, "submit_url": "<url>"}}
                       *DO NOT SCRAPE the submit URL.*
                    
                    6. MATH: If instructions specify a filter, extract it:
                       {{"action": "scrape", "scrape_url": "<file>", "submit_url": "<url>", "math_filter": {{"cutoff": 12000, "direction": "<=", "metric": "sum"}}}}
                    
                    7. ANSWER: If you have the answer, return:
                       {{"action": "submit", "answer": <value>, "submit_url": "<url>"}}
                    
                    8. Output valid JSON.
                    """
                    
                    response = await client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": f"HTML:\n{html_content[:30000]}\n\nAudio:\n{audio_transcript}"},
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
                    
                    # ACTION: PYTHON
                    if llm_output.get("action") == "python":
                        code = llm_output.get("code")
                        answer = execute_python(code, html_content)

                    # ACTION: SCRAPE
                    elif llm_output.get("action") == "scrape":
                        raw_scrape_url = llm_output.get("scrape_url")
                        
                        if "/submit" in raw_scrape_url or "submit" == raw_scrape_url.strip("/"):
                            print(f"⚠️ Correction: Setting answer to email.")
                            answer = user_email
                            raw_submit_url = "/submit"
                        else:
                            headers = llm_output.get("headers", {})
                            if "/api/" in raw_scrape_url or "vercel.app" in raw_scrape_url:
                                if "email" not in headers:
                                    headers["email"] = user_email
                                    headers["secret"] = user_secret

                            target_url = urljoin(current_url, raw_scrape_url)
                            print(f"🔎 Scraping: {target_url}")
                            path = urlparse(target_url).path.lower()
                            
                            # 1. Image
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

                            # 2. PDF
                            elif path.endswith('.pdf'):
                                pdf_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                                pdf_text = read_pdf(pdf_bytes)
                                follow_up = await client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Analyze PDF text. Return JSON: {\"answer\": <value>}"},
                                        {"role": "user", "content": f"PDF Content:\n{pdf_text}"}
                                    ],
                                    response_format={"type": "json_object"}
                                )
                                answer = json.loads(follow_up.choices[0].message.content).get("answer")

                            # 3. Data File / API
                            elif path.endswith(('.csv', '.txt', '.json', '.xml')) or "/api/" in target_url or ".json" in target_url:
                                scraped_data = await fetch_external_content(target_url, headers=headers)
                                
                                # Check Math
                                math_req = llm_output.get("math_filter")
                                if math_req and scraped_data:
                                    math_context = {
                                        "data": scraped_data, 
                                        "cutoff": math_req.get("cutoff"), 
                                        "metric": math_req.get("metric", "sum"),
                                        "dir": math_req.get("direction")
                                    }
                                    answer = perform_filtered_math(scraped_data, math_context["cutoff"], math_context["dir"], math_context["metric"])
                                    print(f"⚡ Math Result 1: {answer}")

                                # Fallback: Python Execution
                                if answer is None and scraped_data:
                                    print("Asking LLM to analyze/process data...")
                                    follow_up = await client.chat.completions.create(
                                        model="openai/gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "Analyze the data. You can write Python code. Return JSON: {\"answer\": <val>} OR {\"python_code\": <code>}. IMPORTANT: Assign result to variable 'answer'."},
                                            {"role": "user", "content": f"Data:\n{scraped_data[:50000]}"}
                                        ],
                                        response_format={"type": "json_object"}
                                    )
                                    res_json = json.loads(follow_up.choices[0].message.content)
                                    if "python_code" in res_json:
                                        answer = execute_python(res_json["python_code"], scraped_data)
                                    else:
                                        answer = res_json.get("answer")

                            # 4. Webpage
                            else:
                                print("🌐 Webpage Scrape")
                                page2 = await context.new_page()
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

                    # --- INTERCEPTOR (UV/CURL) ---
                    if isinstance(answer, str) and (answer.strip().startswith("uv ") or answer.strip().startswith("curl ")):
                        if "not the output" in html_content.lower() or "exact command string" in html_content.lower():
                            print(f"⚠️ Passing command string AS IS (per instructions): {answer}")
                        else:
                            print(f"⚠️ Intercepted Command: '{answer}'. Executing...")
                            url_match = re.search(r'(https?://[^\s]+)', answer)
                            if url_match:
                                cmd_url = url_match.group(1)
                                cmd_headers = {}
                                if "json" in answer: cmd_headers["Accept"] = "application/json"
                                fetched_data = await fetch_external_content(cmd_url, headers=cmd_headers)
                                if fetched_data:
                                    try:
                                        answer = json.loads(fetched_data)
                                    except:
                                        answer = fetched_data

                    if answer is None:
                        feedback_history.append(f"Plan {llm_output} produced NO answer. Try something else.")
                        continue

                    # --- SUBMISSION ---
                    if not raw_submit_url: raw_submit_url = "/submit"
                    submit_url = urljoin(current_url, raw_submit_url)
                    if urlparse(submit_url).path == urlparse(current_url).path:
                        submit_url = urljoin(current_url, "/submit")

                    if isinstance(answer, dict):
                        candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url']]
                        if candidates: answer = candidates[0]
                        else: answer = json.dumps(answer)

                    if isinstance(answer, str) and "<svg" in answer:
                        answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                    payload = {"email": user_email, "secret": user_secret, "url": current_url, "answer": answer}
                    
                    print(f"📤 Submitting: {answer}")
                    async with httpx.AsyncClient() as http:
                        resp = await http.post(submit_url, json=payload, timeout=30)
                        try: res = resp.json()
                        except: res = {"error": resp.text}
                        
                    print(f"✅ Result: {res}")
                    
                    # --- AUTO-RETRY LOGIC (Math Flip) ---
                    if not res.get("correct") and math_context and "Wrong sum" in str(res):
                        print("🔄 Fast Math Retry: Flipping logic...")
                        old_dir = math_context["dir"]
                        new_dir = "<" if old_dir == "<=" else ("<=" if old_dir == "<" else old_dir)
                        if new_dir == old_dir:
                            new_dir = ">" if old_dir == ">=" else (">=" if old_dir == ">" else old_dir)

                        if new_dir != old_dir:
                            retry_ans = perform_filtered_math(math_context["data"], math_context["cutoff"], new_dir, math_context["metric"])
                            print(f"⚡ Retry Math: {retry_ans}")
                            payload["answer"] = retry_ans
                            async with httpx.AsyncClient() as http:
                                resp = await http.post(submit_url, json=payload, timeout=20)
                                res = resp.json()
                            print(f"✅ Fast Retry Result: {res}")

                    if res.get("correct"):
                        current_url = res.get("url")
                        success = True
                        break # Break Retry Loop, Go to Next Page
                    else:
                        print(f"⛔ Attempt failed: {res.get('reason')}")
                        feedback_history.append(f"Answer '{answer}' for plan {llm_output} failed. Reason: {res.get('reason')}")
                        # Loop continues to next attempt attempt_idx

                if not success:
                    print("❌ All retries failed for this URL. Stopping.")
                    break
                    
            except Exception as e:
                print(f"🔥 Error: {e}")
                break
        
        await browser.close()
        print("🏁 Done.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url, request.email, request.secret)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
