# ---------- CAD Geometry Service (Render + OCC binaries) ----------
FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies WITHOUT cleaning conda cache during build
RUN conda install -y -c conda-forge pythonocc-core=7.7.2 numpy && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py machining_estimator.py routing_selector_industrial.py ./

EXPOSE 5000
CMD ["python", "app.py"]
