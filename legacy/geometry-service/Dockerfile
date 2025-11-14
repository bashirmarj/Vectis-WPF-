# Dockerfile - Production Geometry Service with BRepNet
# Base: pythonocc-core pre-built image
# Target: DigitalOcean App Platform (4 vCPU, 16GB RAM)

FROM conda/miniconda3:latest

LABEL maintainer="Vectis Machining"
LABEL version="2.0.0"
LABEL description="CAD Feature Recognition Service with BRepNet"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with pythonocc-core
RUN conda create -n vectis python=3.10 -y && \
    conda install -n vectis -c conda-forge pythonocc-core=7.7.0 -y && \
    conda clean -a -y

# Activate conda environment
SHELL ["conda", "run", "-n", "vectis", "/bin/bash", "-c"]

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY brepnet_wrapper.py .
COPY geometric_fallback.py .

# Create models directory
RUN mkdir -p models

# Download BRepNet pre-trained model (run during build)
# NOTE: In production, this should be downloaded separately and added to image
# or mounted as a volume
# RUN wget -O models/brepnet_pretrained.onnx \
#     https://github.com/AutodeskAILab/BRepNet/releases/download/v1.0/model.onnx

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Run with gunicorn (production WSGI server)
CMD ["conda", "run", "--no-capture-output", "-n", "vectis", \
     "gunicorn", "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
