# ---------- CAD Geometry Service with AAGNet Support ----------
FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pythonocc-core FIRST from conda-forge (not available on PyPI)
RUN conda install -y -c conda-forge python=3.10 pythonocc-core=7.7.2 numpy && \
    conda clean --all --yes

# Clone AAGNet repository (external dependency)
RUN git clone https://github.com/whjdark/AAGNet.git AAGNet-main

# Install occwl (Autodesk's OpenCascade wrapper, needed by AAGNet)
RUN git clone https://github.com/AutodeskAILab/occwl.git && \
    cd occwl && \
    python setup.py install && \
    cd .. && \
    rm -rf occwl

# Copy application files
COPY requirements.txt .
COPY app.py .
COPY aagnet_recognizer.py .
COPY ml_inference.py .
COPY feature_grouping.py .
COPY util.py .
COPY routing_selector_industrial.py .
COPY model_optimization_COMPLETE.py .
COPY validation_utils_COMPLETE.py .
COPY machining_estimator.py .

# Copy AAGNet model components (UV-Net)
COPY activations.py .
COPY decoders.py .
COPY encoders.py .
COPY inst_segmentors.py .
COPY layers.py .
COPY loss.py .
COPY misc.py .
COPY pnaconv.py .

# Copy checkpoints (UV-Net model weights)
COPY checkpoints/ ./checkpoints/

# Install Python dependencies (pythonocc already installed via conda)
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Download AAGNet model weights (170MB)
# Uncomment if you have weights hosted somewhere:
# RUN wget https://your-storage-url/weight_on_MFInstseg.pth -O AAGNet-main/weights/weight_on_MFInstseg.pth

# Create weights directory (weights can be mounted or downloaded at runtime)
RUN mkdir -p AAGNet-main/weights

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
