FROM python:3.11-slim-bullseye

LABEL maintainer="Vectis Machining"
LABEL version="2.0.0"

WORKDIR /app

# Install system dependencies for pythonocc
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install pythonocc-core first (takes longest)
RUN pip install --no-cache-dir pythonocc-core==7.7.2

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Create models directory
RUN mkdir -p models

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
