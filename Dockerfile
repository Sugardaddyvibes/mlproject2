FROM python:3.9-slim-bookworm

WORKDIR /app

# 1. Install system dependencies + compiler for XGBoost
RUN apt-get update -y && \
    apt-get install -y awscli build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the application
COPY . .

CMD ["python3", "application.py"]