# ---------- High-Quality Mesh Service (Gmsh) ----------
FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app
COPY requirements.txt .
COPY mesh_service.py .

# Install system dependencies for Gmsh (OpenGL, X11, and font rendering libraries)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglu1-mesa \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxfixes3 \
    libxft2 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gmsh numpy gunicorn

EXPOSE 5001
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--worker-class", "sync", "--timeout", "120", "mesh_service:app"]
