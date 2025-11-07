FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Install pythonocc-core via conda
RUN conda install -y -c conda-forge python=3.10 pythonocc-core=7.7.2 numpy && \
    conda clean --all --yes

# Copy all application files (AAGNet files already in repo!)
COPY . .

# Remove any Python cache files to ensure fresh deployment
RUN find . -type d -name __pycache__ -exec rm -r {} + || true && \
    find . -type f -name '*.pyc' -delete || true

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]
