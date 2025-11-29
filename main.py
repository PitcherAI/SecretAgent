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

# Try importing pypdf
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

# Pre-compile regex for speed
NUMBER_PATTERN = re.compile(r'^-?\d+(\.\d+)?$')

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ------------------------------------------------------------------
# HELPER: FAST FETCH
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    if headers is None: headers = {}
    
    # Auto-inject headers
    if "/api/" in url or "vercel.app" in url:
        if "email" not in headers:
            headers["email"] = STUDENT_EMAIL
            headers["secret"] = STUDENT_SECRET

    print(f"📥 Fetching: {url}")
    try:
        async with httpx.AsyncClient() as http_client:
            # Reduced timeout for fail-fast
            resp = await http_client.get(url, headers=headers, timeout=20, follow_redirects=True)
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
    if not pypdf: return "pypdf missing."
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        # Limit to first 10 pages for speed
        text = "\n".join([page.extract_text() for page in reader.pages[:10]])
        return text[:20000] 
    except Exception as e:
        return f"PDF Error: {e}"

# ------------------------------------------------------------------
# HELPER: AUDIO TRANSCRIPTION
# ------------------------------------------------------------------
async def transcribe_audio(audio_bytes, filename="audio.mp3"):
    print(f"🎤 Transcribing ({len(audio_bytes)} bytes)...")
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        # Whisper-1 is fast enough
        transcription = await client.audio.transcriptions.create(
            model="openai/whisper-1", 
            file=audio_file
        )
        return transcription.text
    except Exception:
        return ""

# ------------------------------------------------------------------
# HELPER: UNIVERSAL PYTHON EXECUTOR
# ------------------------------------------------------------------
def execute_python(code, context_data=""):
    print("🐍 Executing Python...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    local_scope = {
        "data": context_data,
        "csv": csv,
        "json": json,
        "statistics": statistics,
        "re": re,
        "math": __import__("math")
    }
    
    try:
        exec(code, {}, local_scope)
        result = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        if not result.strip() and "answer" in local_scope:
            result = str(local_scope["answer"])
        return result.strip()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Python Error: {e}"

# ------------------------------------------------------------------
# HELPER: MATH ENGINE (Optimized)
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction, metric="sum"):
    try:
        numbers = []
        # Fast path: Split and check regex
        lines = [l.strip().replace(',', '') for l in content.split('\n') if l.strip()]
        
        for line in lines:
            if NUMBER_PATTERN.match(line):
                numbers.append(float(line))
        
        if not numbers: return None

        print(f"🧮 Math: {len(numbers)} nums. Filter: {direction} {cutoff_val}")
        cutoff = float(cutoff_val)
        
        # Optimize filtering using list comprehensions
        if direction in ["<", "below"]: filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "at most", "up to", "max"]: filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above"]: filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least", "min"]: filtered = [n for n in numbers if n >= cutoff]
        elif direction in ["=", "=="]: filtered = [n for n in numbers if n == cutoff]
        elif direction in ["%", "mod"]: filtered = [n for n in numbers if n % cutoff == 0]
        else: filtered = numbers

        if not filtered: return 0

        metric = metric.lower()
        if metric == "count": result = len(filtered)
        elif metric == "mean": result = statistics.mean(filtered)
        elif metric == "max": result = max(filtered)
        elif metric == "min": result = min(filtered)
        elif metric == "median": result = statistics.median(filtered)
        else: result = sum(filtered)

        if isinstance(result, float) and result.is_integer():
            return int(result)
        elif isinstance(result, float):
            return round(result, 4)
        return result

    except Exception:
        return None

# ------------------------------------------------------------------
# CORE AGENT LOGIC
# ------------------------------------------------------------------
async def solve_quiz(start_url: str, user_email: str, user_secret: str):
    print(f"🚀 Start: {start_url}")
    
    async with async_playwright() as p:
        # Optimized Args for Render Free Tier
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", 
                "--disable-dev-shm-usage", 
                "--disable-gpu", 
                "--disable-extensions", # Saves memory
                "--mute-audio" # Don't process audio stream
            ]
        )
        context = await browser.new_context()
        page = await context.new_page()
        
        current_url = start_url
        
        while current_url:
            print(f"🔗 Nav: {current_url}")
            try:
                # Fast Navigation: Wait for DOM, not Network Idle
                for attempt in range(2):
                    try:
                        await page.goto(current_url, timeout=45000, wait_until="domcontentloaded")
                        break
                    except Exception:
                        if attempt == 1: pass 
                
                # --- CONTEXT ---
                html_content = await page.content() 
                
                # Optimization: JPEG Quality 60 is much smaller/faster than PNG
                screenshot = await page.screenshot(type="jpeg", quality=60, full_page=True)
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                # --- AUDIO ---
                audio_transcript = ""
                # Fast selector check
                if await page.locator("audio, a[href$='.mp3'], a[href$='.wav']").count() > 0:
                    audio_element = await page.query_selector("audio source, a[href$='.mp3'], a[href$='.wav']")
                    if audio_element:
                        src = await audio_element.get_attribute("src") or await audio_element.get_attribute("href")
                        if src:
                            audio_url = urljoin(current_url, src)
                            audio_bytes = await fetch_external_content(audio_url, is_binary=True)
                            if audio_bytes:
                                audio_transcript = await transcribe_audio(audio_bytes)

                # --- PLANNING ---
                # Shortened prompt to save input tokens and speed up inference
                system_prompt = """
                You are a data agent. Analyze Inputs.
                1. NAV: If text says "Add ?email=...", return: {"action": "scrape", "scrape_url": "<mod_url>", "submit_url": "<url>"}
                2. DATA: If file/API, return: {"action": "scrape", "scrape_url": "<url>", "headers": {"key": "val"}, "submit_url": "<url>"}
                3. MATH: If filter (e.g. "sum < 5000"), extract: {"action": "scrape", "scrape_url": "<file>", "submit_url": "<url>", "math_filter": {"cutoff": 12000, "direction": "<=", "metric": "sum"}}
                4. COMPLEX: If Geo/Net/Sorting logic needed, return: {"action": "python", "code": "<code>", "submit_url": "<url>"}
                5. ANSWER: {"action": "submit", "answer": <value>, "submit_url": "<url>"}
                6. VISUAL: Generate SVG if asked.
                Default submit_url: "/submit". Output JSON.
                """
                
                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini", # Fast model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"HTML:{html_content[:30000]}\nAudio:{audio_transcript}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
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
                action = llm_output.get("action")
                
                if action == "python":
                    answer = execute_python(llm_output.get("code"), html_content)

                elif action == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    headers = llm_output.get("headers", {})
                    target_url = urljoin(current_url, raw_scrape_url)
                    
                    print(f"🔎 Scrape: {target_url}")
                    path = urlparse(target_url).path.lower()
                    
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                        if img_bytes:
                            b64_scraped = base64.b64encode(img_bytes).decode('utf-8')
                            vision_resp = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Return JSON: {\"answer\": <val>}"},
                                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_scraped}"}}]}
                                ],
                                response_format={"type": "json_object"}
                            )
                            answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                    elif path.endswith('.pdf'):
                        pdf_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                        pdf_text = read_pdf(pdf_bytes)
                        follow_up = await client.chat.completions.create(
                            model="openai/gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Analyze PDF. JSON: {\"answer\": <val>}"},
                                {"role": "user", "content": pdf_text}
                            ],
                            response_format={"type": "json_object"}
                        )
                        answer = json.loads(follow_up.choices[0].message.content).get("answer")

                    elif path.endswith(('.csv', '.txt', '.json', '.xml')) or "/api/" in target_url:
                        scraped_data = await fetch_external_content(target_url, headers=headers)
                        math_req = llm_output.get("math_filter")
                        
                        if math_req and scraped_data:
                            math_context = {
                                "data": scraped_data, 
                                "cutoff": math_req.get("cutoff"), 
                                "metric": math_req.get("metric", "sum"),
                                "dir": math_req.get("direction")
                            }
                            answer = perform_filtered_math(scraped_data, math_context["cutoff"], math_context["dir"], math_context["metric"])
                            print(f"⚡ Math: {answer}")

                        if answer is None and scraped_data:
                            # If no simple math, maybe Python is needed
                            follow_up = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Analyze data. Return JSON: {\"answer\": <val>} OR {\"python_code\": <code>}"},
                                    {"role": "user", "content": f"Data:\n{scraped_data[:40000]}"}
                                ],
                                response_format={"type": "json_object"}
                            )
                            res = json.loads(follow_up.choices[0].message.content)
                            if "python_code" in res:
                                answer = execute_python(res["python_code"], scraped_data)
                            else:
                                answer = res.get("answer")

                    else:
                        print("🌐 Webpage Scrape")
                        page2 = await context.new_page()
                        if headers: await page2.set_extra_http_headers(headers)
                        await page2.goto(target_url, timeout=30000, wait_until="domcontentloaded")
                        scraped_html = await page2.content()
                        await page2.close()
                        
                        follow_up = await client.chat.completions.create(
                            model="openai/gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Analyze. JSON: {\"answer\": <val>}"},
                                {"role": "user", "content": f"Main:\n{html_content[:10000]}\nScraped:\n{scraped_html[:20000]}"}
                            ],
                            response_format={"type": "json_object"}
                        )
                        answer = json.loads(follow_up.choices[0].message.content).get("answer")

                else:
                    answer = llm_output.get("answer")

                if answer is None: break

                # --- SUBMISSION ---
                submit_url = urljoin(current_url, raw_submit_url)
                
                # Unwrap Dicts
                if isinstance(answer, dict):
                    cands = [v for k, v in answer.items() if k not in ['email', 'secret', 'url', 'answer']]
                    answer = cands[0] if cands else json.dumps(answer)

                if isinstance(answer, str) and "<svg" in answer:
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                payload = {"email": user_email, "secret": user_secret, "url": current_url, "answer": answer}
                
                print(f"📤 Submitting: {answer}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=20)
                    try: res = resp.json()
                    except: res = {"error": resp.text}
                    
                print(f"✅ Result: {res}")
                
                # --- RETRY LOGIC ---
                if not res.get("correct") and math_context and "Wrong sum" in str(res):
                    print("🔄 Retry: Flipping logic...")
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
                        print(f"✅ Retry Result: {res}")

                if res.get("correct"):
                    current_url = res.get("url")
                else:
                    break
                    
            except Exception as e:
                print(f"🔥 Error: {e}")
                break
        
        await browser.close()
        print("🏁 Done.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    # Pass credentials directly
    background_tasks.add_task(solve_quiz, request.url, request.email, request.secret)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
