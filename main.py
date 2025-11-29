import os
import json
import base64
import httpx
import re
import io
import csv
import statistics
import asyncio
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

app = FastAPI()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_PRIORITY = [
    "x-ai/grok-4.1-fast:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat:free",
    "google/gemini-2.0-flash-exp:free"
]

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
# HELPER: ROBUST LLM CALLER
# ------------------------------------------------------------------
async def query_llm(messages, response_format=None, force_model=None):
    priority_list = [force_model] if force_model else MODEL_PRIORITY
    for model in priority_list:
        retries = 5
        delay = 4
        for attempt in range(retries):
            try:
                print(f"🧠 Asking {model} (Attempt {attempt+1})...")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.0,
                    max_tokens=1500,
                )
                print(f"✅ Success with {model}")
                return response
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "upstream" in error_msg:
                    print(f"⏳ Rate limited on {model}. Waiting {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print(f"⚠️ Error on {model}: {e}")
                    break
        print(f"⏭️ Skipping {model}...")
    raise Exception("❌ All models failed.")

# ------------------------------------------------------------------
# FETCH EXTERNAL DATA
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    print(f"📥 Fetching: {url}")
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as http_client:
        resp = await http_client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content if is_binary else resp.text

# ------------------------------------------------------------------
# MATH ENGINE (FULL, EXACT ORIGINAL + FIXES)
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction, metric="sum"):
    try:
        numbers = []
        # Try CSV first
        try:
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                for cell in row:
                    clean_cell = cell.strip().replace(',', '').replace(' ', '')
                    if re.match(r'^-?\d+(\.\d+)?$', clean_cell):
                        numbers.append(float(clean_cell))
        except:
            pass

        # Fallback: line by line
        if not numbers:
            lines = [l.strip().replace(',', '').replace(' ', '') for l in content.split('\n') if l.strip()]
            for line in lines:
                if re.match(r'^-?\d+(\.\d+)?$', line):
                    numbers.append(float(line))

        if not numbers:
            return None

        print(f"🧮 Math Engine: {len(numbers)} numbers found. Filter: {direction} {cutoff_val}. Metric: {metric}")
        cutoff = float(cutoff_val)

        if direction in ["<", "below", "less"]: filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "at most", "up to", "<="]: filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above", "more"]: filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least"]: filtered = [n for n in numbers if n >= cutoff]
        elif direction in ["=", "equal"]: filtered = [n for n in numbers if n == cutoff]
        else: filtered = numbers

        if not filtered:
            return 0

        metric = metric.lower()
        if metric == "count": result = len(filtered)
        elif metric in ["mean", "average"]: result = statistics.mean(filtered)
        elif metric in ["max", "maximum"]: result = max(filtered)
        elif metric in ["min", "minimum"]: result = min(filtered)
        else: result = sum(filtered)

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 6)

        print(f"✅ Calculation Result: {result}")
        return result
    except Exception as e:
        print(f"Math Error: {e}")
        return None

# ------------------------------------------------------------------
# CORE AGENT LOGIC — FINAL, BATTLE-TESTED VERSION
# ------------------------------------------------------------------
async def solve_quiz(start_url: str):
    print(f"🚀 Starting background task for: {start_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-features=IsolateOrigins,site-per-process"]
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()
        current_url = start_url

        while current_url:
            print(f"🔗 Navigating to: {current_url}")
            try:
                await page.goto(current_url, wait_until="networkidle", timeout=90000)
            except:
                await page.goto(current_url, wait_until="load", timeout=90000)

            # Respect delay if server asks for it
            if 'delay' in locals():
                print(f"😴 Respecting server delay: {delay}s")
                await asyncio.sleep(delay)

            if any(word in page.url.lower() for word in ["congrat", "complete", "finish", "success"]):
                print("🎉 Quiz Completed!")
                break

            html_content = await page.content()
            screenshot_bytes = await page.screenshot(full_page=True, type="png")
            b64_img = base64.b64encode(screenshot_bytes).decode('utf-8')

            system_prompt = """
You are the world's best autonomous agent for solving TDS, P2, and all LLM evaluation quizzes.
You are given the full HTML + a complete full-page screenshot (use the image for plots, charts, canvas, LaTeX, images, anything not perfectly rendered in HTML).

Return strict JSON only. Possible actions:

1. You have the answer → {"action": "submit", "answer": "exact value (number/string)", "submit_url": "/submit" or full if different}
2. Need to scrape a resource → {"action": "scrape", "scrape_url": "relative or absolute URL", "math_filter": {"cutoff": 12345, "direction": "<=", "metric": "sum"} if it's a data file}

Never explain, never add extra fields. Be extremely precise.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Current URL: {current_url}\n\nHTML:\n{html_content[:50000]}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ]

            try:
                response = await query_llm(messages, response_format={"type": "json_object"})
                llm_output = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"LLM parsing failed: {e}. Falling back to vision-only.")
                fallback = await query_llm([
                    {"role": "system", "content": "Look only at the screenshot and return the exact answer in JSON."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the answer?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ], response_format={"type": "json_object"})
                llm_output = json.loads(fallback.choices[0].message.content)
                llm_output = {"action": "submit", "answer": llm_output.get("answer", llm_output)}

            print(f"🤖 Action Plan: {llm_output}")

            answer = None
            submit_url = urljoin(current_url, llm_output.get("submit_url", "/submit"))

            if llm_output.get("action") == "scrape":
                scrape_url = urljoin(current_url, llm_output["scrape_url"])
                print(f"🔎 Scraping: {scrape_url}")
                path = urlparse(scrape_url).path.lower()

                if path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
                    img_bytes = await fetch_external_content(scrape_url, is_binary=True)
                    b64_scraped = base64.b64encode(img_bytes).decode()
                    vision_resp = await query_llm([
                        {"role": "system", "content": "You have seen the quiz page. Now analyze this scraped image and give the final answer."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scraped}"}}
                        ]}
                    ], response_format={"type": "json_object"})
                    answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                elif path.endswith(('.csv', '.txt', '.json', '.data')):
                    data = await fetch_external_content(scrape_url)
                    math_req = llm_output.get("math_filter")
                    if math_req:
                        answer = perform_filtered_math(data, math_req["cutoff"], math_req["direction"], math_req.get("metric", "sum"))
                    if answer is None:
                        truncated = data[:100000]
                        follow_up = await query_llm([
                            {"role": "system", "content": "Extract the exact answer from this data."},
                            {"role": "user", "content": f"Data:\n{truncated}"}
                        ], response_format={"type": "json_object"})
                        answer = json.loads(follow_up.choices[0].message.content)["answer"]

                else:
                    page2 = await context.new_page()
                    await page2.goto(scrape_url, wait_until="networkidle", timeout=60000)
                    scraped_html = await page2.content()
                    await page2.close()
                    follow_up = await query_llm([
                        {"role": "system", "content": "Combine both pages and return the answer."},
                        {"role": "user", "content": f"Main HTML:\n{html_content[:30000]}\n\nScraped HTML:\n{scraped_html[:30000]}"}
                    ], response_format={"type": "json_object"})
                    answer = json.loads(follow_up.choices[0].message.content)["answer"]

            else:
                answer = llm_output.get("answer")

            if answer is None:
                print("⚠️ Final vision fallback")
                final = await query_llm([
                    {"role": "system", "content": "Just read the screenshot and give the exact answer."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Answer?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ], response_format={"type": "json_object"})
                answer = json.loads(final.choices[0].message.content).get("answer", final.choices[0].message.content)

            if isinstance(answer, str) and "<svg" in answer:
                answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode()).decode()

            payload = {
                "email": STUDENT_EMAIL,
                "secret": STUDENT_SECRET,
                "url": current_url,
                "answer": answer
            }

            print(f"📤 Submitting → {answer} to {submit_url}")
            async with httpx.AsyncClient() as http:
                resp = await http.post(submit_url, json=payload, timeout=45)
                try:
                    resp_data = resp.json()
                except:
                    resp_data = {"text": resp.text[:500]}

            print(f"✅ Server: {resp_data}")

            delay = resp_data.get("delay", 0)
            if resp_data.get("correct") in [True, "True"]:
                current_url = resp_data.get("url")
            else:
                if "url" in resp_data:
                    current_url = resp_data["url"]
                    print("Wrong but continuing (quiz allows it)")
                else:
                    print("⛔ Stopped — incorrect")
                    break

        await browser.close()
        print("🏁 Task Finished.")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
