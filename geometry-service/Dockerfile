FROM continuumio/miniconda3:24.1.2-0

LABEL maintainer="Vectis Machining"
LABEL version="2.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pythonocc-core via conda
RUN conda install -y -c conda-forge python=3.10 pythonocc-core=7.7.2 numpy && \
    conda clean --all --yes

# Install PyTorch CPU-only (before requirements.txt to use cache)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy ALL application code FIRST (includes occwl folder)
COPY . .

# Install Python dependencies (occwl removed from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Add occwl to Python path (so imports work)
ENV PYTHONPATH=/app:$PYTHONPATH

# Create models directory
RUN mkdir -p models

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
