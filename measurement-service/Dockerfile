# Dockerfile for Measurement Service
# Standalone Flask application for CAD measurement calculations

FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with OpenCascade
RUN conda install -c conda-forge pythonocc-core=7.7.0 -y && \
    conda clean -afy

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir flask flask-cors numpy gunicorn requests

# Copy application code
COPY app.py .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MEASUREMENT_SERVICE_PORT=8081

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--workers", "2", "--timeout", "120", "app:app"]
