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

MODEL_PRIORITY = [
    "x-ai/grok-4.1-fast:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-chat:free"
]

client = AsyncOpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openrouter/v1"
)

STUDENT_EMAIL = os.environ["STUDENT_EMAIL"]
STUDENT_SECRET = os.environ["STUDENT_SECRET"]

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

async def query_llm(messages, response_format=None):
    for model in MODEL_PRIORITY:
        for _ in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.0,
                    max_tokens=1200
                )
                return resp
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    await asyncio.sleep(10)
                else:
                    break
    raise Exception("All models failed")

async def fetch(url, binary=False):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content if binary else r.text

def math_engine(data, cutoff, direction, metric="sum"):
    nums = []
    for line in data.splitlines():
        clean = line.strip().replace(',', '')
        if re.match(r'^-?\d+(\.\d+)?$', clean):
            nums.append(float(clean))
    if not nums:
        return None
    c = float(cutoff)
    if direction in ["<=", "up to", "at most", "<="]: filtered = [n for n in nums if n <= c]
    elif direction in [">=", "at least"]: filtered = [n for n in nums if n >= c]
    elif direction == "<": filtered = [n for n in nums if n < c]
    elif direction == ">": filtered = [n for n in nums if n > c]
    else: filtered = nums
    if metric == "count": return len(filtered)
    elif metric == "mean": return round(statistics.mean(filtered), 4)
    elif metric == "max": return max(filtered)
    elif metric == "min": return min(filtered)
    else: return sum(filtered) if not isinstance(sum(filtered), float) or not sum(filtered).is_integer() else int(sum(filtered))

async def solve_quiz(start_url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        url = start_url

        while url:
            await page.goto(url, wait_until="networkidle", timeout=90000)
            html = await page.content()
            screenshot = await page.screenshot(full_page=True)
            b64 = base64.b64encode(screenshot).decode()

            messages = [
                {"role": "system", "content": "You are the best LLM quiz solver ever. You get full HTML + full screenshot (use image for plots/charts/canvas). Return ONLY JSON: {\"action\": \"submit\", \"answer\": value, \"submit_url\": \"...\"} or {\"action\": \"scrape\", \"scrape_url\": \"...\", \"math_filter\": {...}}"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"URL: {url}\nHTML:\n{html[:45000]}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ]

            plan = {"action": "submit", "answer": "42"}
            try:
                plan = json.loads((await query_llm(messages, {"type": "json_object"})).choices[0].message.content)
            except:
                plan = {"action": "submit", "answer": json.loads((await query_llm([{"role": "system", "content": "Answer from image only"}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]} ], {"type": "json_object"})).choices[0].message.content).get("answer", "unknown")}

            submit_url = urljoin(url, plan.get("submit_url", "/submit"))
            answer = plan.get("answer")

            if plan.get("action") == "scrape":
                su = urljoin(url, plan["scrape_url"])
                if su.lower().endswith(('.csv', '.txt', '.data')):
                    data = await fetch(su)
                    mf = plan.get("math_filter")
                    answer = math_engine(data, mf["cutoff"], mf["direction"], mf.get("metric", "sum")) if mf else None
                    if answer is None:
                        answer = json.loads((await query_llm([{"role": "system", "content": "Extract answer"}, {"role": "user", "content": data[:80000]}])).choices[0].message.content).get("answer")
                elif su.lower().endswith(('.png','.jpg','.jpeg','.svg')):
                    img = base64.b64encode(await fetch(su, binary=True)).decode()
                    answer = json.loads((await query_llm([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}])).choices[0].message.content).get("answer")

            payload = {"email": STUDENT_EMAIL, "secret": STUDENT_SECRET, "url": url, "answer": answer}
            async with httpx.AsyncClient() as c:
                r = await c.post(submit_url, json=payload)
                result = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}

            print(f"Submitted {answer} → {result}")

            if result.get("correct") in [True, "True"]:
                url = result.get("url")
            elif result.get("url"):
                url = result["url"]
            else:
                break

            if delay := result.get("delay"):
                await asyncio.sleep(delay)

@app.post("/run")
async def run(req: QuizRequest, bg: BackgroundTasks):
    if req.secret != STUDENT_SECRET:
        raise HTTPException(403)
    bg.add_task(solve_quiz, req.url)
    return {"status": "started"}

@app.get("/")
def health():
    return {"status": "ok"}
