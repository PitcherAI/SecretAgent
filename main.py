import os
import json
import asyncio
import base64
import httpx
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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
# HELPER: FETCH EXTERNAL DATA (CSV/JSON/TEXT)
# ------------------------------------------------------------------
async def fetch_external_content(url):
    """
    Uses Python requests (httpx) to grab data from a URL.
    Better than a browser for CSVs or raw text files.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=20)
            resp.raise_for_status()
            # Return text (limit to 100k chars to save tokens)
            return resp.text[:100000] 
    except Exception as e:
        return f"Error fetching content: {str(e)}"

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
                # 1. Navigate to the Quiz Page
                await page.goto(current_url, timeout=45000)
                await page.wait_for_load_state("networkidle")
                
                # 2. Extract Data (Full HTML + Screenshot)
                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                print("👀 Page content extracted. Consulting LLM...")

                # 3. Ask LLM to solve it
                system_prompt = """
                You are an expert data extraction agent.
                1. Analyze the HTML and screenshot.
                2. If the answer requires data from a DIFFERENT link (like a CSV, JSON, or text file linked in the page), return valid JSON:
                   {"action": "scrape", "scrape_url": "<url_to_scrape>"}
                3. If you have the final answer, return valid JSON:
                   {"action": "submit", "answer": <extracted_value_only>, "submit_url": "<url_found>"}
                4. AUDIO/FILES: If there is an audio file, look for a text link nearby (like "Download Data" or a .csv link) and request to SCRAPE that URL.
                5. Do NOT output the instruction text as the answer.
                6. The "answer" field must be a SINGLE value (string or number). 
                7. For "submit_url", if explicitly mentioned use it. If vague, default to "/submit".
                8. Your output must be a valid JSON object.
                """
                
                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"HTML Content:\n{html_content}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ],
                    response_format={"type": "json_object"}
                )
                
                llm_output = json.loads(response.choices[0].message.content)
                print(f"🤖 LLM Action: {llm_output}")

                # --- HANDLE SCRAPING REQUESTS ---
                if llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    target_url = urljoin(current_url, raw_scrape_url)
                    print(f"🔎 Agent requested scraping: {target_url}")
                    
                    # SMART SCRAPING SWITCHER
                    # If it's a file (CSV/TXT), use HTTPX (faster, no download dialogs)
                    # If it's a Webpage, use Playwright (renders JS)
                    path = urlparse(target_url).path
                    if path.endswith(('.csv', '.txt', '.json', '.xml')):
                        print("📂 Detected file download. Using HTTPX.")
                        scraped_data = await fetch_external_content(target_url)
                    else:
                        print("🌐 Detected webpage. Using Playwright.")
                        page2 = await context.new_page()
                        await page2.goto(target_url, timeout=30000)
                        await page2.wait_for_load_state("networkidle")
                        scraped_data = await page2.content()
                        await page2.close()
                    
                    print(f"✅ Scraped data retrieved (First 500 chars): {scraped_data[:500]}")
                    print("Asking LLM to analyze scraped data...")
                    
                    follow_up_response = await client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Analyze the scraped content below. Solve the user's question (e.g., sum a column, find a code). Return valid JSON: {\"answer\": <value>, \"submit_url\": <from_previous_step>}"},
                            {"role": "user", "content": f"Original Page HTML: {html_content}\n\nScraped Content:\n{scraped_data}"}
                        ],
                        response_format={"type": "json_object"}
                    )
                    llm_output = json.loads(follow_up_response.choices[0].message.content)
                    print(f"🤖 LLM Final Answer: {llm_output}")

                # --- SUBMISSION LOGIC ---
                answer = llm_output.get("answer")
                raw_submit_url = llm_output.get("submit_url")
                
                # Unwrap accidentally nested dictionaries
                if isinstance(answer, dict):
                    candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url']]
                    if candidates:
                        answer = candidates[0]
                    else:
                        answer = json.dumps(answer)

                if not answer or not raw_submit_url:
                    print("❌ Failed to find answer or submit URL")
                    break

                # --- URL FIX: Prevent submitting to root domain ---
                parsed_url = urlparse(raw_submit_url)
                if not parsed_url.path or parsed_url.path == "/":
                    print(f"⚠️ Warning: LLM suggested root URL '{raw_submit_url}'. Defaulting to '/submit'.")
                    submit_url = urljoin(current_url, "/submit")
                else:
                    submit_url = urljoin(current_url, raw_submit_url)

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
                    except json.JSONDecodeError:
                        print(f"🔥 Error: Server returned non-JSON response (Status: {resp.status_code})")
                        print(f"Response Text: {resp.text[:200]}...") 
                        break
                    
                print(f"✅ Result: {resp_data}")
                
                if resp_data.get("correct"):
                    current_url = resp_data.get("url")
                else:
                    print(f"⛔ Answer incorrect. Reason: {resp_data.get('reason')}")
                    break
                    
            except Exception as e:
                print(f"🔥 Error: {e}")
                break
        
        await browser.close()
        print("🏁 Finished.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
