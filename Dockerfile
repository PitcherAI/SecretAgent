# 1. Use official Python slim image
FROM python:3.11-slim-bookworm

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies required for Playwright/Chromium
# We avoid installing the full browser here to keep image small
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Install ONLY Chromium (saves space vs full playwright install)
RUN playwright install chromium --with-deps

# 6. Copy application code
COPY . .

# 7. Start the app
# Render automatically sets the PORT environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]