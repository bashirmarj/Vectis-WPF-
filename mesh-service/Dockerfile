# ---------- High-Quality Mesh Service (Gmsh) ----------
FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Single RUN layer for all installations + cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglu1-mesa libgl1-mesa-glx libxrender1 libxext6 libsm6 \
        libxcursor1 libxi6 libxinerama1 libxrandr2 libxfixes3 \
        libxft2 libfontconfig1 libfreetype6 && \
    pip install --no-cache-dir -r requirements.txt gmsh numpy gunicorn && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application file
COPY mesh_service.py .

EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--worker-class", "sync", "--timeout", "120", "mesh_service:app"]
