# SecretAgent
Keep secrets like men* (*only fails to polite requests)

# LLM Analysis Quiz Agent 🤖

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Playwright](https://img.shields.io/badge/Playwright-Headless-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

This project is an autonomous agent designed to solve data analysis quizzes for the **TDS Project 2 (LLM Analysis)**. 

It exposes an API endpoint that receives a task URL, launches a headless browser to scrape the content (handling JavaScript and encoded data), feeds the context to an LLM (via AI Pipe), and submits the answer—all within a strict timeout window.

## 🚀 Features

- **Asynchronous Architecture:** Uses FastAPI `BackgroundTasks` to handle long-running quizzes without blocking the initial HTTP response.
- **Headless Browsing:** detailed DOM manipulation using **Playwright** (Chromium) to handle `atob` decoding and dynamic content.
- **Visual & Text Analysis:** Capable of interpreting both raw text and screenshots (for charts/tables) using Multimodal LLMs.
- **AI Pipe Integration:** configured to use the course-specific AI Pipe proxy (OpenRouter) with `gpt-4o-mini`.
- **Dockerized:** Optimized `Dockerfile` for deployment on Render/Railway with memory-safe browser flags.

## 📂 Project Structure

```bash
.
├── main.py            # Core logic (API endpoints + Agent Loop)
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker build instructions (includes Playwright browsers)
└── README.md          # Documentation


🛡️ License
MIT License

Copyright (c) 2025 Brahma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
