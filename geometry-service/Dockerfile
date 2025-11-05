# ---------- CAD Geometry Service with AAGNet (Render + OCC binaries) ----------
FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Copy all requirements and application files
COPY requirements.txt .
COPY app.py .
COPY ml_inference.py .
COPY feature_grouping.py .
COPY util.py .
COPY routing_selector_industrial.py .
COPY model_optimization_COMPLETE.py .
COPY validation_utils_COMPLETE.py .
COPY machining_estimator.py .

# Copy AAGNet model components
COPY activations.py .
COPY decoders.py .
COPY encoders.py .
COPY inst_segmentors.py .
COPY layers.py .
COPY loss.py .
COPY misc.py .
COPY pnaconv.py .

# Copy optional dependencies (if needed)
COPY occwl/ ./occwl/
COPY checkpoints/ ./checkpoints/

# Install pythonocc-core and dependencies from conda-forge (precompiled)
RUN conda install -y -c conda-forge pythonocc-core=7.7.2 numpy && \
    pip install --no-cache-dir -r requirements.txt && \
    conda clean --all --yes

EXPOSE 5000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run application using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "60", "--workers", "2", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
