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

# No longer needed — we always send the screenshot to the main model chain
# VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"

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
        retries = 3
        delay = 5
        for attempt in range(retries):
            try:
                print(f"🧠 Asking {model} (Attempt {attempt+1})...")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.3,
                    max_tokens=1024,
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
# HELPER: FETCH EXTERNAL DATA
# ------------------------------------------------------------------
async def fetch_external_content(url, headers=None, is_binary=False):
    print(f"📥 Fetching: {url}")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http_client:
        resp = await http_client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content if is_binary else resp.text

# ------------------------------------------------------------------
# MATH ENGINE (unchanged, works perfectly)
# ------------------------------------------------------------------
def perform_filtered_math(content, cutoff_val, direction, metric="sum"):
    # ... (exact same as your original) ...

# ------------------------------------------------------------------
# CORE AGENT LOGIC — COMPLETELY REWRITTEN FOR RELIABILITY
# ------------------------------------------------------------------
async def solve_quiz(start_url: str):
    print(f"🚀 Starting background task for: {start_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context()
        page = await context.new_page()
        current_url = start_url

        while current_url:
            print(f"🔗 Navigating to: {current_url}")
            await page.goto(current_url, timeout=60000)
            await page.wait_for_load_state("networkidle")

            if "congratulations" in page.url.lower() or "complete" in page.url.lower():
                print("🎉 Quiz Completed!")
                break

            html_content = await page.content()
            # FULL PAGE screenshot — critical fix
            screenshot_bytes = await page.screenshot(full_page=True, type="png")
            b64_img = base64.b64encode(screenshot_bytes).decode('utf-8')

            # =================================================================
            # MAIN REASONING — ALWAYS send HTML + full screenshot
            # =================================================================
            system_prompt = """
You are an expert autonomous agent solving extremely hard LLM evaluation quizzes (TDS, P2, etc.).
You are given the complete HTML and a full-page screenshot of the current quiz page.

Use the HTML for text and links.
Use the screenshot for plots, charts, images, math rendering, canvas, JS-generated content, or anything not perfectly in HTML.

Your goal: find the exact answer and submit it, or scrape a resource if needed.

Output strict JSON only. Possible actions:

1. You know the answer → {"action": "submit", "answer": "the exact answer", "submit_url": "/submit" or other if visible}
2. Need external resource (CSV, image, API, etc.) → {"action": "scrape", "scrape_url": "relative or full URL", "math_filter": {"cutoff": 12345, "direction": "<=", "metric": "sum"} if applicable}

Be precise. Answers are usually numbers or short strings. For math/data questions, scrape first if a file/link is mentioned.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"HTML:\n{html_content}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ]

            response = await query_llm(messages, response_format={"type": "json_object"})
            llm_output = json.loads(response.choices[0].message.content)
            print(f"🤖 Action Plan: {llm_output}")

            answer = None
            submit_url = urljoin(current_url, llm_output.get("submit_url", "/submit"))

            # =================================================================
            # HANDLE SCRAPE
            # =================================================================
            if llm_output.get("action") == "scrape":
                raw_scrape_url = llm_output["scrape_url"]
                target_url = urljoin(current_url, raw_scrape_url)
                headers = llm_output.get("headers")

                path = urlparse(target_url).path.lower()
                print(f"🔎 Scraping: {target_url}")

                if path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    img_bytes = await fetch_external_content(target_url, headers=headers, is_binary=True)
                    b64_scraped = base64.b64encode(img_bytes).decode()
                    # Use main model chain (Grok → Gemini, both have vision)
                    vision_resp = await query_llm([
                        {"role": "system", "content": "Analyze the scraped image together with the quiz page (HTML + screenshot already seen) and return the exact answer."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Quiz HTML and screenshot were provided earlier. Now here is the linked/scraped image. What is the final answer?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_scraped}"}}
                        ]}
                    ], response_format={"type": "json_object"})
                    answer = json.loads(vision_resp.choices[0].message.content).get("answer")

                elif path.endswith(('.csv', '.txt', '.json', '.xml', '.data')):
                    data = await fetch_external_content(target_url, headers=headers)
                    math_req = llm_output.get("math_filter")
                    if math_req and data:
                        answer = perform_filtered_math(data, math_req["cutoff"], math_req["direction"], math_req.get("metric", "sum"))
                    if answer is None:
                        # Let LLM analyze raw data if no math_filter or math failed
                        truncated = data[:60000]
                        follow_up = await query_llm([
                            {"role": "system", "content": "Extract the exact answer from this data file."},
                            {"role": "user", "content": f"Data:\n{truncated}\n\nReturn JSON: {{\"answer\": value}}"}
                        ], response_format={"type": "json_object"})
                        answer = json.loads(follow_up.choices[0].message.content)["answer"]

                else:
                    # Regular page scrape
                    page2 = await context.new_page()
                    await page2.goto(target_url, timeout=30000)
                    scraped_html = await page2.content()
                    await page2.close()
                    follow_up = await query_llm([
                        {"role": "system", "content": "Combine both pages and return the answer."},
                        {"role": "user", "content": f"Main page HTML:\n{html_content}\n\nScraped page HTML:\n{scraped_html}\n\nReturn JSON: {{\"answer\": value}}"}
                    ], response_format={"type": "json_object"})
                    answer = json.loads(follow_up.choices[0].message.content)["answer"]

            else:
                answer = llm_output.get("answer")

            # Final fallback (very rare now)
            if answer is None:
                print("⚠️ Final vision fallback")
                fallback = await query_llm([
                    {"role": "system", "content": "Look at the screenshot and give the exact answer."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the answer?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ], response_format={"type": "json_object"})
                answer = json.loads(fallback.choices[0].message.content)["answer"]

            # SVG handling
            if isinstance(answer, str) and "<svg" in answer:
                answer = "data:image/svg+xml;base64," + base64.b64encode(answer.encode()).decode()

            # =================================================================
            # SUBMIT
            # =================================================================
            payload = {
                "email": STUDENT_EMAIL,
                "secret": STUDENT_SECRET,
                "url": current_url,
                "answer": answer
            }

            print(f"📤 Submitting answer → {answer} to {submit_url}")
            async with httpx.AsyncClient() as http:
                resp = await http.post(submit_url, json=payload, timeout=30)
                try:
                    resp_data = resp.json()
                except:
                    resp_data = {"text": resp.text[:500]}

            print(f"✅ Server response: {resp_data}")

            if resp_data.get("correct") is True:
                next_url = resp_data.get("url")
                if next_url:
                    current_url = next_url
                else:
                    print("🎉 Finished (no next URL)")
                    break
            else:
                # Some quizzes (like P2) let you continue even if wrong
                if "url" in resp_data:
                    current_url = resp_data["url"]
                    print("Continuing despite wrong answer (quiz allows it)")
                else:
                    print("⛔ Stopped — incorrect and no next URL")
                    break

        await browser.close()
        print("🏁 Task Finished.")

# =================================================================
# FASTAPI ENDPOINTS
# =================================================================
@app.post("/run")
async def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Bad Secret")
    background_tasks.add_task(solve_quiz, request.url)
    return {"message": "Started", "status": "ok"}

@app.get("/")
async def health_check():
    return {"status": "ok"}
