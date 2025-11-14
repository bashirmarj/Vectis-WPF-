# Use Debian-based Miniconda (has apt-get AND conda)
FROM continuumio/miniconda3:24.1.2-0

LABEL maintainer="Vectis Machining"
LABEL version="2.0.0"

WORKDIR /app

# Install system dependencies (Debian-based, so apt-get works)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pythonocc-core via conda (ONLY way to get it)
RUN conda install -y -c conda-forge python=3.10 pythonocc-core=7.7.2 && \
    conda clean --all --yes

# Copy requirements and install Python dependencies via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
