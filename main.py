import os
import json
import asyncio
import base64
import httpx
import re
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
async def fetch_external_content(url, is_binary=False):
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(url, timeout=30)
            resp.raise_for_status()
            if is_binary:
                return resp.content # Bytes for Audio/Images
            return resp.text # String for CSV/Text
    except Exception as e:
        print(f"Fetch Error ({url}): {e}")
        return None

# ------------------------------------------------------------------
# HELPER: AUDIO TRANSCRIPTION
# ------------------------------------------------------------------
async def transcribe_audio(audio_bytes, filename="audio.mp3"):
    """
    Sends audio to Whisper via AI Pipe to get the instructions.
    """
    print(f"🎤 Transcribing audio ({len(audio_bytes)} bytes)...")
    try:
        # Create a mock file-like object with a name (required by OpenAI API)
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        # AI Pipe supports OpenAI's audio endpoints
        # We use a separate call or model if needed, but 'openai/whisper-1' is standard
        # Note: We might need to temporarily switch base_url if AI Pipe routes audio differently,
        # but usually it proxies standard endpoints.
        
        # NOTE: For some proxies, we might need to use the standard OpenAI URL with the pipe key.
        # But let's try the configured client first.
        transcription = await client.audio.transcriptions.create(
            model="openai/whisper-1", 
            file=audio_file
        )
        print(f"🗣️ Transcript: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"⚠️ Transcription Failed: {e}")
        return "[Audio Transcription Failed]"

# ------------------------------------------------------------------
# HELPER: MATH ENGINE
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction):
    """
    Executes math logic extracted from the Audio/Text.
    """
    try:
        # Clean and parse numbers
        lines = [l.strip().replace(',', '') for l in content.split('\n') if l.strip()]
        numbers = []
        for line in lines:
            # Robust number parsing
            if re.match(r'^-?\d+(\.\d+)?$', line):
                numbers.append(float(line))
        
        if not numbers:
            return None

        print(f"🧮 Math Engine: Processing {len(numbers)} numbers. Filter: {direction} {cutoff_val}")

        filtered = []
        cutoff = float(cutoff_val)

        # Logic Map
        if direction in ["<", "below", "less", "smaller"]:
            filtered = [n for n in numbers if n < cutoff]
        elif direction in [">", "above", "more", "greater", "larger"]:
            filtered = [n for n in numbers if n > cutoff]
        elif direction in ["=", "equal", "=="]:
            filtered = [n for n in numbers if n == cutoff]
        elif direction in ["!=", "not", "different"]:
            filtered = [n for n in numbers if n != cutoff]
        # Modulo logic (e.g. "divisible by 7")
        elif direction in ["%", "divisible", "mod"]:
            filtered = [n for n in numbers if n % cutoff == 0]
        else:
            # Fallback: Return Total Sum if direction is unclear
            print("⚠️ Unknown direction, returning total sum.")
            return int(sum(numbers))

        result = int(sum(filtered))
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
                
                # Extract Data
                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                # --- AUDIO DETECTION ---
                # Check for audio files in the HTML
                audio_transcript = ""
                audio_element = await page.query_selector("audio source, a[href$='.mp3'], a[href$='.wav']")
                if audio_element:
                    src = await audio_element.get_attribute("src") or await audio_element.get_attribute("href")
                    if src:
                        audio_url = urljoin(current_url, src)
                        print(f"🎵 Found Audio: {audio_url}")
                        audio_bytes = await fetch_external_content(audio_url, is_binary=True)
                        if audio_bytes:
                            audio_transcript = await transcribe_audio(audio_bytes)

                print("👀 Page content extracted. Consulting LLM...")

                # 3. Ask LLM (Step 1: Identify Action)
                system_prompt = """
                You are an expert data extraction agent.
                1. Analyze the HTML, Screenshot, and Audio Transcript (if any).
                2. If the answer requires downloading a file (CSV/JSON/TXT), return valid JSON:
                   {"action": "scrape", "scrape_url": "<url_to_scrape>", "submit_url": "<url_to_submit>"}
                3. If the instructions specify a MATH FILTER (e.g. "sum numbers < 5000", "cutoff 12000"), extract it:
                   {"action": "scrape", "scrape_url": "<file_url>", "submit_url": "<submit_url>", "math_filter": {"cutoff": 12000, "direction": "<"}}
                   Directions: "<", ">", "=", "divisible"
                4. If you have the answer directly, return:
                   {"action": "submit", "answer": <value>, "submit_url": "<url_found>"}
                5. AUDIO PRIORITY: The Audio Transcript often contains the filtering rule (e.g. "The cutoff is..."). Trust it.
                6. Default "submit_url" to "/submit".
                """
                
                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"HTML Content:\n{html_content}\n\nAudio Transcript:\n{audio_transcript}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ],
                    response_format={"type": "json_object"}
                )
                
                llm_output = json.loads(response.choices[0].message.content)
                print(f"🤖 LLM Action: {llm_output}")

                # --- HANDLE ACTIONS ---
                answer = None
                raw_submit_url = llm_output.get("submit_url")

                if llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    target_url = urljoin(current_url, raw_scrape_url)
                    print(f"🔎 Agent requested scraping: {target_url}")
                    
                    path = urlparse(target_url).path
                    if path.endswith(('.csv', '.txt', '.json', '.xml')):
                        print("📂 Detected file. Using HTTPX.")
                        scraped_data = await fetch_external_content(target_url)
                        
                        # Check Math
                        math_req = llm_output.get("math_filter")
                        if math_req and scraped_data:
                            cutoff = math_req.get("cutoff")
                            direction = math_req.get("direction")
                            answer = perform_filtered_math(scraped_data, cutoff, direction)
                        
                        # Fallback to LLM
                        if answer is None and scraped_data:
                            print("Asking LLM to analyze file content...")
                            truncated_data = scraped_data[:50000]
                            follow_up = await client.chat.completions.create(
                                model="openai/gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "Analyze the file. Solve the question. Return JSON: {\"answer\": <value>}"},
                                    {"role": "user", "content": f"Context HTML: {html_content}\nAudio Transcript: {audio_transcript}\nFile Content:\n{truncated_data}"}
                                ],
                                response_format={"type": "json_object"}
                            )
                            final_data = json.loads(follow_up.choices[0].message.content)
                            answer = final_data.get("answer")
                    else:
                        # Webpage fallback
                        print("🌐 Detected webpage. Using Playwright.")
                        page2 = await context.new_page()
                        await page2.goto(target_url, timeout=30000)
                        scraped_data = await page2.content()
                        await page2.close()
                
                else:
                    answer = llm_output.get("answer")

                # --- SUBMISSION ---
                if answer is None or not raw_submit_url:
                    print("❌ Missing answer or submit URL. Retrying loop...")
                    break

                # URL Cleanup
                parsed_sub = urlparse(raw_submit_url)
                if not parsed_sub.path or parsed_sub.path == "/":
                    submit_url = urljoin(current_url, "/submit")
                else:
                    submit_url = urljoin(current_url, raw_submit_url)

                # Dict Safety
                if isinstance(answer, dict):
                    candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url']]
                    answer = candidates[0] if candidates else json.dumps(answer)

                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"📤 Submitting '{answer}' to {submit_url}")
                
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    try:
                        resp_data = resp.json()
                    except:
                        print(f"🔥 Invalid Server Response: {resp.text[:200]}")
                        break
                    
                print(f"✅ Result: {resp_data}")
                
                if resp_data.get("correct"):
                    current_url = resp_data.get("url")
                else:
                    print(f"⛔ Incorrect. Reason: {resp_data.get('reason')}")
                    break
                    
            except Exception as e:
                print(f"🔥 Crash: {e}")
                break
        
        await browser.close()
        print("🏁 Finished.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
