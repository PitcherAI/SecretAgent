import os
import json
import asyncio
import base64
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

app = FastAPI()

# --- CONFIGURATION: AI PIPE ---
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

async def solve_quiz(start_url: str):
    print(f"Starting quiz: {start_url}")
    async with async_playwright() as p:
        # --- CRITICAL RENDER FLAGS ---
        # --disable-dev-shm-usage: Prevents memory crash in Docker
        # --no-sandbox: Required for root user in Docker
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", 
                "--disable-dev-shm-usage", 
                "--disable-gpu"
            ]
        )
        context = await browser.new_context()
        page = await context.new_page()
        
        current_url = start_url
        
        while current_url:
            print(f"Navigating to {current_url}")
            try:
                await page.goto(current_url, timeout=45000) # 45s timeout
                await page.wait_for_load_state("networkidle")
                
                # Scrape
                text = await page.inner_text("body")
                screenshot = await page.screenshot(type="png")
                b64_img = base64.b64encode(screenshot).decode()
                
                # LLM Analysis
                system_prompt = """
                You are a data agent.
                1. Solve the question in the text/image.
                2. Extract the submission URL.
                3. JSON Response: {"answer": <value>, "submit_url": "<url>"}
                """
                
                response = await client.chat.completions.create(
                    model="openai/gpt-4o-mini", # Provider/Model format for AI Pipe
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Page Text: {text}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                answer = result.get("answer")
                submit_url = result.get("submit_url")
                
                if not answer or not submit_url:
                    print("Error: Missing answer or url")
                    break

                # Submit
                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"Submitting: {answer}")
                async with httpx.AsyncClient() as http:
                    resp = await http.post(submit_url, json=payload, timeout=30)
                    data = resp.json()
                    
                if data.get("correct"):
                    current_url = data.get("url")
                else:
                    print(f"Wrong answer: {data}")
                    break
                    
            except Exception as e:
                print(f"Crash: {e}")
                break
        
        await browser.close()

@app.post("/run")
async def start(req: QuizRequest, tasks: BackgroundTasks):
    if req.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    tasks.add_task(solve_quiz, req.url)
    return {"message": "Started"}

@app.get("/")
async def health():

    return {"status": "200 ok"}

