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

# We still keep these for the initial auth check
EXPECTED_SECRET = os.environ.get("STUDENT_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ------------------------------------------------------------------
# HELPER: FETCH EXTERNAL DATA
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    if headers is None: headers = {}
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

                # --- PLANNING ---
                # FIX: Injected user_email and added "Start" logic
                system_prompt = f"""
                You are an autonomous data extraction agent.
                YOUR EMAIL: {user_email}
                
                1. Analyze HTML, Screenshot, and Audio.
                2. STARTING: If the page asks to start/begin or for an email, the answer is "{user_email}".
                   Return: {{"action": "submit", "answer": "{user_email}", "submit_url": "<url>"}}
                3. NAVIGATION: If text says "Add ?email=..." or "Go to...", return: 
                   {{"action": "scrape", "scrape_url": "<modified_url>", "submit_url": "<url>"}}
                4. DATA SCRAPING: If you need data from a file/API/PDF, return:
                   {{"action": "scrape", "scrape_url": "<url>", "headers": {{"key": "val"}}, "submit_url": "<url>"}}
                   *DO NOT SCRAPE the submit URL (e.g. /submit).*
                5. ANALYSIS:
                   - For simple math filters ("sum < 5000"), use "math_filter" key.
                   - For COMPLEX analysis (Geo, Network, ML, Sorting, Transformation), write Python code.
                     Return: {{"action": "python", "code": "<python_code>", "submit_url": "<url>"}}
                     *Note: In your code, the file content is available in variable `data`.*
                6. VISUALIZATION: If asked for a chart, generate SVG code in the answer.
                7. ANSWER: If you have the answer, return:
                   {{"action": "submit", "answer": <value>, "submit_url": "<url>"}}
                8. Output valid JSON.
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
                
                # --- EXECUTION ---
                
                # ACTION: PYTHON
                if llm_output.get("action") == "python":
                    code = llm_output.get("code")
                    answer = execute_python(code, html_content)

                # ACTION: SCRAPE
                elif llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    
                    # --- CRITICAL FIX: SUBMIT SCRAPE GUARD ---
                    # If Agent tries to scrape /submit, intervene and set answer to email
                    if "/submit" in raw_scrape_url or "submit" == raw_scrape_url.strip("/"):
                        print(f"⚠️ Agent tried to scrape submit URL '{raw_scrape_url}'. Correcting to Email Submission...")
                        answer = user_email
                    else:
                        headers = llm_output.get("headers", {})
                        
                        # Auto-Inject headers for API calls using credentials from REQUEST
                        if "/api/" in raw_scrape_url or "vercel.app" in raw_scrape_url:
                            if "email" not in headers:
                                print("💉 Auto-injecting Request Credentials into Headers...")
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
                            print("📄 Detected PDF.")
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
                        elif path.endswith(('.csv', '.txt', '.json', '.xml')) or "/api/" in target_url:
                            scraped_data = await fetch_external_content(target_url, headers=headers)
                            
                            # Fallback: Python Execution for Analysis
                            print("Asking LLM to analyze/process data...")
                            follow_up = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Analyze the data. You can write Python code. Return JSON: {\"answer\": <val>} OR {\"python_code\": <code>}. IMPORTANT: Assign result to variable 'answer' in python code."},
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

                if answer is None:
                    print("❌ Failure: No answer found.")
                    break

                # --- SUBMISSION ---
                submit_url = urljoin(current_url, raw_submit_url)
                
                # BUG FIX: Allow 'answer' key to be extracted
                if isinstance(answer, dict):
                    # Filter out metadata keys, but KEEP 'answer'
                    candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url']]
                    if candidates:
                        answer = candidates[0]
                    else:
                        answer = json.dumps(answer)

                if isinstance(answer, str) and "<svg" in answer:
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode('utf-8')).decode('utf-8')

                payload = {"email": user_email, "secret": user_secret, "url": current_url, "answer": answer}
                
                print(f"📤 Submitting: {answer}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    try: res = resp.json()
                    except: res = {"error": resp.text}
                    
                print(f"✅ Result: {res}")
                
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
    # Verify the secret matches what we expect (Auth Check)
    if request.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    
    # Pass the request credentials specifically to the worker
    background_tasks.add_task(solve_quiz, request.url, request.email, request.secret)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
