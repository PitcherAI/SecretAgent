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
                system_prompt = f"""
                You are an autonomous data extraction and problem-solving agent.
                YOUR EMAIL: {user_email}

                1. Analyze the page (HTML + screenshot + audio transcript).
                2. If the page asks for email or to start, submit the email.
                3. If a URL modification is needed (e.g. ?email=...), return scrape action to the modified URL.
                4. COMMANDS:
                   - If the page says "submit the command string", "submit the exact command", or "not the output", submit the raw command (e.g. uv http get ...).
                   - If it says "run the command" or "execute", scrape the URL inside the command.
                5. If data must be fetched/analyzed (JSON, CSV, PDF, image, API), use scrape action.
                6. For complex analysis, use python action with code that sets variable "answer".
                7. Final answer → submit action.
                8. Always output valid JSON only.
                """

                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"HTML:\n{html_content}\n\nAudio transcript:\n{audio_transcript}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ],
                    response_format={"type": "json_object"}
                )
               
                llm_output = json.loads(response.choices[0].message.content)
                print(f"🤖 Plan: {llm_output}")

                answer = None
                raw_submit_url = llm_output.get("submit_url", "/submit")

                # --- EXECUTION ---
                if llm_output.get("action") == "python":
                    code = llm_output.get("code")
                    answer = execute_python(code, html_content)

                elif llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    if "/submit" in raw_scrape_url.lower():
                        answer = user_email
                    else:
                        headers = llm_output.get("headers", {})
                        target_url = urljoin(current_url, raw_scrape_url)
                        print(f"🔎 Scraping: {target_url}")

                        path = urlparse(target_url).path.lower()

                        if path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                            if img_bytes:
                                b64 = base64.b64encode(img_bytes).decode()
                                vision = await client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Describe what you see and extract the answer. Return JSON: {\"answer\": ...}"},
                                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}
                                    ],
                                    response_format={"type": "json_object"}
                                )
                                answer = json.loads(vision.choices[0].message.content).get("answer")

                        elif path.endswith('.pdf'):
                            pdf_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                            if pdf_bytes:
                                pdf_text = read_pdf(pdf_bytes)
                                follow = await client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Extract the answer from PDF. Return JSON: {\"answer\": ...}"},
                                        {"role": "user", "content": pdf_text}
                                    ],
                                    response_format={"type": "json_object"}
                                )
                                answer = json.loads(follow.choices[0].message.content).get("answer")

                        else:
                            data = await fetch_external_content(target_url, headers=headers)
                            if data:
                                follow = await client.chat.completions.create(
                                    model="openai/gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Analyze the data and return JSON: {\"answer\": ...} or {\"python_code\": \"...\"} (set variable answer)"},
                                        {"role": "user", "content": data[:50000]}
                                    ],
                                    response_format={"type": "json_object"}
                                )
                                res = json.loads(follow.choices[0].message.content)
                                answer = execute_python(res["python_code"], data) if "python_code" in res else res.get("answer")
                else:
                    answer = llm_output.get("answer")

                # --- COMMAND INTERCEPTOR (uv / curl) ---
                if isinstance(answer, str) and (answer.strip().startswith("uv ") or answer.strip().startswith("curl ")):
                    if ("not the output" in html_content.lower() or 
                        "exact command" in html_content.lower() or 
                        "command string" in html_content.lower()):
                        print(f"⚠️ Passing command string AS IS: {answer}")
                    else:
                        print(f"⚠️ Executing command instead: {answer}")
                        url_match = re.search(r'(https?://[^\s\'\"]+)', answer)
                        if url_match:
                            cmd_url = url_match.group(1)
                            cmd_headers = {"Accept": "application/json"} if "json" in answer.lower() else {}
                            fetched = await fetch_external_content(cmd_url, headers=cmd_headers)
                            if fetched:
                                try:
                                    answer = json.loads(fetched)
                                except:
                                    answer = fetched

                if answer is None:
                    print("❌ No answer produced")
                    break

                # --- SUBMISSION ---
                submit_url = urljoin(current_url, raw_submit_url)

                if isinstance(answer, dict):
                    answer = json.dumps(answer)
                if isinstance(answer, str) and answer.strip().startswith("<svg"):
                    answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode()).decode()

                payload = {
                    "email": user_email,
                    "secret": user_secret,
                    "url": current_url,
                    "answer": answer
                }

                print(f"📤 Submitting to {submit_url}: {answer}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    try:
                        res = resp.json()
                    except:
                        res = {"error": resp.text or "non-json response"}

                print(f"✅ Result: {res}")

                # ─── ROBUST SUCCESS DETECTION (critical fix) ───
                correct_val = res.get("correct")
                error_val = res.get("error")
                reason = res.get("reason") or error_val

                is_success = (
                    correct_val in (True, "true", "True") or
                    error_val in ("", None, False, "false", "None")
                )

                next_url_raw = res.get("url") or res.get("next_url") or res.get("next")

                if next_url_raw:
                    current_url = urljoin(current_url, next_url_raw)
                    print(f"✅ Correct → proceeding to {current_url}")
                elif is_success:
                    print("✅ Correct → final step! Challenge completed.")
                    current_url = None
                else:
                    print(f"⛔ Incorrect / stopped: {reason or res}")
                    current_url = None

            except Exception as e:
                print(f"🔥 Fatal error: {e}")
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
