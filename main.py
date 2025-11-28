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
# Connect to AI Pipe (OpenRouter) using the Token from Env Vars
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
        # Launch options optimized for Render/Docker environment
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", 
                "--disable-dev-shm-usage", # Critical for memory limits
                "--disable-gpu"
            ]
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
                # FIX: We use page.content() to get raw HTML (tags, attributes, scripts)
                # instead of page.inner_text() which misses hidden data like <div id="secret">
                html_content = await page.content() 
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode('utf-8')
                
                print("👀 Page content extracted (HTML + Screenshot). Consulting LLM...")

                # 3. Ask LLM to solve it
                # We explicitly ask for JSON to make parsing reliable
                system_prompt = """
                You are an expert data extraction agent.
                1. Analyze the provided HTML and screenshot.
                2. If the user asks for a specific value (like a "sum", "code", or "secret"), extract the ACTUAL value.
                3. Do NOT simply repeat the instruction text. 
                   (Example: If asked for a secret, return "X83-99", NOT "the secret code").
                4. Look inside HTML tags (like <span id="secret">, <script>, or data-attributes) if the answer is hidden.
                5. Identify the SUBMISSION URL (it might be relative like '/submit').
                6. Return strict JSON: {"answer": <value>, "submit_url": "<url_found>"}
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
                
                # Parse LLM Response
                llm_content = response.choices[0].message.content
                print(f"🤖 LLM Raw Output: {llm_content}")
                
                try:
                    result = json.loads(llm_content)
                except json.JSONDecodeError:
                    print("❌ Failed to parse JSON from LLM")
                    break

                answer = result.get("answer")
                raw_submit_url = result.get("submit_url")
                
                if not answer or not raw_submit_url:
                    print("❌ LLM could not find answer or submit_url")
                    break

                # 4. Fix Relative URLs (The Crash Fix)
                # If LLM gives "/submit", this converts it to "https://example.com/submit"
                submit_url = urljoin(current_url, raw_submit_url)

                # 5. Submit the Answer
                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"📤 Submitting answer '{answer}' to {submit_url}")
                
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    resp_data = resp.json()
                    
                print(f"✅ Server Response: {resp_data}")
                
                # 6. Check if there is a next question
                if resp_data.get("correct"):
                    next_url = resp_data.get("url")
                    if next_url:
                        current_url = next_url # Loop continues with new URL
                    else:
                        print("🎉 Quiz Completed! No new URL provided.")
                        break
                else:
                    print("⛔ Answer was incorrect. Stopping.")
                    break
                    
            except Exception as e:
                print(f"🔥 Error encountered: {e}")
                break
        
        await browser.close()
        print("🏁 Background task finished.")

# ------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------
@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    # Verify Secret
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    
    # Start the "Agent" in the background so we can return 200 OK immediately
    background_tasks.add_task(solve_quiz, request.url)
    
    return {"message": "Quiz processing started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "LLM Quiz Agent"}
