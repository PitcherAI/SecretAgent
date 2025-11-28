import os
import json
import asyncio
import base64
import httpx
from urllib.parse import urljoin
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
                # FIX: Added strict instructions to prevent nested JSON objects as answers
                system_prompt = """
                You are an expert data extraction agent.
                1. Analyze the HTML and screenshot.
                2. If the user asks to "scrape" or "download" a DIFFERENT link to get the answer, return valid JSON:
                   {"action": "scrape", "scrape_url": "<url_to_scrape>"}
                3. If you have the answer, return valid JSON:
                   {"action": "submit", "answer": <extracted_value_only>, "submit_url": "<url_found>"}
                4. Look inside HTML tags/scripts for hidden secrets.
                5. Do NOT output the instruction text as the answer.
                6. The "answer" field must be a SINGLE value (string or number). 
                   Do NOT return a dictionary/JSON object as the "answer". 
                   If the page shows a JSON example, extract ONLY the value requested (e.g., the 'cutoff' or 'sum').
                7. Your output must be a valid JSON object.
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
                    
                    # Open a new tab
                    page2 = await context.new_page()
                    await page2.goto(target_url, timeout=30000)
                    await page2.wait_for_load_state("networkidle")
                    scraped_content = await page2.content()
                    await page2.close()
                    
                    print("✅ Scraped data retrieved. Asking LLM again...")
                    
                    follow_up_response = await client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Extract the answer from the scraped content below. Return valid JSON: {\"answer\": <value>, \"submit_url\": <from_previous_step>}"},
                            {"role": "user", "content": f"Original Page HTML: {html_content}\n\nScraped External Content: {scraped_content}"}
                        ],
                        response_format={"type": "json_object"}
                    )
                    llm_output = json.loads(follow_up_response.choices[0].message.content)
                    print(f"🤖 LLM Final Answer: {llm_output}")

                # --- SUBMISSION LOGIC ---
                answer = llm_output.get("answer")
                raw_submit_url = llm_output.get("submit_url")
                
                # --- SAFETY FIX: Unwrap accidentally nested dictionaries ---
                if isinstance(answer, dict):
                    print(f"⚠️ Warning: LLM returned a dict as answer. Attempting to extract value...")
                    # Try to find a key that is NOT standard metadata
                    candidates = [v for k, v in answer.items() if k not in ['email', 'secret', 'url']]
                    if candidates:
                        answer = candidates[0] # Pick the first unique value
                        print(f"🔧 Fixed Answer: {answer}")
                    else:
                        answer = json.dumps(answer) # Fallback to string

                if not answer or not raw_submit_url:
                    print("❌ Failed to find answer or submit URL")
                    break

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
                    resp_data = resp.json()
                    
                print(f"✅ Result: {resp_data}")
                
                if resp_data.get("correct"):
                    current_url = resp_data.get("url")
                else:
                    print("⛔ Answer incorrect.")
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
