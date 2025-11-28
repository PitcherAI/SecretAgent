import os
import json
import asyncio
import base64
import httpx
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
async def fetch_external_content(url):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# ------------------------------------------------------------------
# HELPER: LOCAL STATS (Robust Fix for Sum/Max/Min/Avg)
# ------------------------------------------------------------------
def analyze_data_locally(content):
    """
    Calculates basic stats for numeric data.
    Returns a text summary to help the LLM.
    """
    try:
        # Clean data
        lines = [l.strip().replace(',', '') for l in content.split('\n') if l.strip()]
        numbers = []
        for line in lines:
            if line.replace('.', '', 1).lstrip('-').isdigit():
                numbers.append(float(line))
        
        # If dataset is numeric, return useful stats
        if len(numbers) > 0 and len(numbers) > len(lines) * 0.5:
            stats = {
                "sum": int(sum(numbers)), # Int is safer for this quiz
                "count": len(numbers),
                "max": max(numbers),
                "min": min(numbers),
                "average": sum(numbers) / len(numbers)
            }
            print(f"🧮 Local Stats Calculated: {stats}")
            return json.dumps(stats)
            
    except Exception as e:
        print(f"Math check failed: {e}")
    
    return "No numeric stats available."

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
                
                print("👀 Page content extracted. Consulting LLM...")

                # 3. Ask LLM (Step 1: Identify Action)
                system_prompt = """
                You are an expert data extraction agent.
                1. Analyze the HTML and screenshot.
                2. If the answer requires data from a link (CSV, JSON, TXT), return valid JSON:
                   {"action": "scrape", "scrape_url": "<url_to_scrape>", "submit_url": "<url_to_submit_answer>"}
                3. If you have the answer directly, return valid JSON:
                   {"action": "submit", "answer": <value>, "submit_url": "<url_found>"}
                4. AUDIO/FILES: If there is an audio file, look for a text link nearby (like "Download Data" or a .csv link) and scrape THAT.
                5. Do NOT output the instruction text as the answer.
                6. For "submit_url", default to "/submit" if not found.
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

                # --- HANDLE SCRAPING ---
                answer = None
                raw_submit_url = llm_output.get("submit_url")

                if llm_output.get("action") == "scrape":
                    raw_scrape_url = llm_output.get("scrape_url")
                    target_url = urljoin(current_url, raw_scrape_url)
                    print(f"🔎 Agent requested scraping: {target_url}")
                    
                    # 1. Download Data
                    # Handle Files vs Webpages
                    path = urlparse(target_url).path
                    if path.endswith(('.csv', '.txt', '.json', '.xml')):
                        print("📂 Detected file. Using HTTPX.")
                        scraped_data = await fetch_external_content(target_url)
                        # Calculate Stats (Sum/Max/Etc) Locally
                        stats_info = analyze_data_locally(scraped_data)
                    else:
                        print("🌐 Detected webpage. Using Playwright.")
                        page2 = await context.new_page()
                        await page2.goto(target_url, timeout=30000)
                        await page2.wait_for_load_state("networkidle")
                        scraped_data = await page2.content()
                        await page2.close()
                        stats_info = "No stats."

                    # 2. Ask LLM to Analyze with Stats Support
                    print("Asking LLM to analyze (with local stats)...")
                    truncated_data = scraped_data[:6000] # Fit in context
                    
                    follow_up = await client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": """
                             Analyze the file content and the 'Local Stats' provided.
                             Solve the user's question (e.g., 'sum', 'max', 'count').
                             If the question asks for a calculation (sum/max/etc), PREFER the values in 'Local Stats' as they are exact.
                             Return valid JSON: {"answer": <value>}
                             """},
                            {"role": "user", "content": f"Local Stats (Calculated by Python): {stats_info}\n\nOriginal Page HTML: {html_content}\n\nFile Content (Truncated):\n{truncated_data}"}
                        ],
                        response_format={"type": "json_object"}
                    )
                    final_data = json.loads(follow_up.choices[0].message.content)
                    answer = final_data.get("answer")
                
                else:
                    # Direct answer found on page
                    answer = llm_output.get("answer")

                # --- SUBMISSION ---
                if not answer or not raw_submit_url:
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
