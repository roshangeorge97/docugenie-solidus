# Stage 1: Install dependencies
FROM python:3.10-slim AS dependencies
WORKDIR /app
COPY requirements.txt .

# Enable logging during the installation process
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --progress-bar='on' --timeout 70 -r requirements.txt

# Stage 2: Build the application
FROM python:3.10-slim
WORKDIR /app

# Copy the installed dependencies from the previous stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000

CMD ["python", "-u", "main.py"]

