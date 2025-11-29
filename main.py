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
# CONFIGURATION — FINAL NOV 29 2025 VERSION
# ------------------------------------------------------------------
MODEL_PRIORITY = [
    "x-ai/grok-4.1-fast:free",      # Has vision + best reasoning
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat:free",
    "google/gemini-2.0-flash-exp:free"  # Also has vision
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
# ROBUST LLM CALLER
# ------------------------------------------------------------------
async def query_llm(messages, response_format=None):
    for model in MODEL_PRIORITY:
        for attempt in range(5):
            try:
                print(f"🧠 Asking {model} (Attempt {attempt+1})")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.0,
                    max_tokens=1500,
                )
                print(f"✅ {model} succeeded")
                return response
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate limit" in err:
                    await asyncio.sleep(8 * (2 ** attempt))
                else:
                    print(f"⚠️ {model} error: {e}")
                    break
        print(f"⏭️ Skipping {model}")
    raise Exception("All models failed")

# ------------------------------------------------------------------
# FETCH
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    print(f"📥 Fetching: {url}")
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content if is_binary else r.text

# ------------------------------------------------------------------
# MATH ENGINE — ENHANCED
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction, metric="sum"):
    try:
        numbers = []
        # CSV first
        try:
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                for cell in row:
                    clean = cell.strip().replace(',', '')
                    if re.match(r'^-?\d+(\.\d+)?$', clean):
                        numbers.append(float(clean))
        except:
            pass
        # Fallback line-by-line
        if not numbers:
            for line in content.split('\n'):
                clean = line.strip().replace(',', '')
                if re.match(r'^-?\d+(\.\d+)?$', clean):
                    numbers.append(float(clean))
        if not numbers:
            return None

        cutoff = float(cutoff_val)
        if direction in ["<", "below", "less than"]: filtered = [n for n in numbers if n < cutoff]
        elif direction in ["<=", "up to", "at most"]: filtered = [n for n in numbers if n <= cutoff]
        elif direction in [">", "above", "greater than"]: filtered = [n for n in numbers if n > cutoff]
        elif direction in [">=", "at least"]: filtered = [n for n in numbers if n >= cutoff]
        elif direction in ["=", "exactly"]: filtered = [n for n in numbers if n == cutoff]
        else: filtered = numbers

        if metric == "count": result = len(filtered)
        elif metric in ["mean", "avg", "average"]: result = statistics.mean(filtered)
        elif metric == "max": result = max(filtered)
        elif metric == "min": result = min(filtered)
        else: result = sum(filtered)

        return int(result) if isinstance(result, float) and result.is_integer() else round(result, 6)
    except:
        return None

# ------------------------------------------------------------------
# CORE SOLVER — FINAL BATTLE-TESTED VERSION (NO VISION CRASH EVER)
# ------------------------------------------------------------------
async def solve_quiz(start_url: str):
    print(f"🚀 Starting: {start_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        current_url = start_url

        while current_url:
            print(f"🔗 Navigating: {current_url}")
            await page.goto(current_url, wait_until="networkidle", timeout=90000)

            if any(x in page.url.lower() for x in ["congrat", "success", "finish", "complete"]):
                print("🎉 Quiz completed!")
                break

            html = await page.content()
            screenshot = await page.screenshot(full_page=True, type="png")
            b64_img = base64.b64encode(screenshot).decode()

            system_prompt = """
You are the ultimate autonomous agent for TDS, P2, and all LLM quizzes.
You have the full HTML + full-page screenshot (use image for charts, plots, canvas, LaTeX, images).

Return ONLY valid JSON:

{"action": "submit", "answer": "exact answer", "submit_url": "/submit" or other}
OR
{"action": "scrape", "scrape_url": "relative or full", "math_filter": {"cutoff": 123, "direction": "<=", "metric": "sum"} if data file}

No explanations. Be 100% accurate.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"URL: {current_url}\nHTML (truncated if long):\n{html[:50000]}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ]

            try:
                resp = await query_llm(messages, {"type": "json_object"})
                plan = json.loads(resp.choices[0].message.content)
            except:
                # Extreme fallback — pure vision
                resp = await query_llm([
                    {"role": "system", "content": "Just look at the screenshot and return the exact answer in JSON."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]}
                ], {"type": "json_object"})
                plan = {"action": "submit", "answer": json.loads(resp.choices[0].message.content).get("answer", "42")}

            print(f"🤖 Plan: {plan}")

            answer = None
            submit_url = urljoin(current_url, plan.get("submit_url", "/submit"))

            if plan.get("action") == "scrape":
                scrape_url = urljoin(current_url, plan["scrape_url"])
                print(f"🔎 Scraping: {scrape_url}")
                path = urlparse(scrape_url).path.lower()

                if path.endswith(('.csv', '.txt', '.json', '.data')):
                    data = await fetch_external_content(scrape_url)
                    mf = plan.get("math_filter")
                    if mf:
                        answer = perform_filtered_math(data, mf["cutoff"], mf["direction"], mf.get("metric", "sum"))
                    if answer is None:
                        truncated = data[:100000]
                        follow = await query_llm([
                            {"role": "system", "content": "Extract exact answer from this data."},
                            {"role": "user", "content": truncated}
                        ], {"type": "json_object"})
                        answer = json.loads(follow.choices[0].message.content)["answer"]
                elif path.endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif')):
                    img_bytes = await fetch_external_content(scrape_url, is_binary=True)
                    b64 = base64.b64encode(img_bytes).decode()
                    vision = await query_llm([
                        {"role": "system", "content": "You have seen the quiz. Now analyze this scraped image and give final answer."},
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}
                    ], {"type": "json_object"})
                    answer = json.loads(vision.choices[0].message.content)["answer"]
                else:
                    page2 = await context.new_page()
                    await page2.goto(scrape_url, wait_until="networkidle")
                    scraped_html = await page2.content()
                    await page2.close()
                    combo = await query_llm([
                        {"role": "system", "content": "Combine both pages."},
                        {"role": "user", "content": f"Main: {html[:30000]}\nScraped: {scraped_html[:30000]}"}
                    ], {"type": "json_object"})
                    answer = json.loads(combo.choices[0].message.content)["answer"]
            else:
                answer = plan.get("answer")

            if answer is None:
                final = await query_llm([
                    {"role": "system", "content": "Just the answer from the screenshot."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]}
                ], {"type": "json_object"})
                answer = json.loads(final.choices[0].message.content).get("answer", answer)

            if isinstance(answer, str) and "<svg" in answer:
                answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode()).decode()

            payload = {
                "email": STUDENT_EMAIL,
                "secret": STUDENT_SECRET,
                "url": current_url,
                "answer": answer
            }

            print(f"📤 Submitting: {answer}")
            async with httpx.AsyncClient() as client:
                r = await client.post(submit_url, json=payload, timeout=60)
                try:
                    result = r.json()
                except:
                    result = {"text": r.text[:500]}

            print(f"✅ Response: {result}")

            if result.get("correct") in [True, "True"]:
                current_url = result.get("url")
            elif "url" in result:
                current_url = result["url"]
                print("Continuing despite wrong (quiz allows)")
            else:
                print("⛔ Stopped — wrong and no next URL")
                break

            if (delay := result.get("delay")):
                print(f"😴 Sleeping {delay}s")
                await asyncio.sleep(delay)

        await browser.close()
        print("🏁 Finished")

@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def root():
    return {"status": "ok"}
