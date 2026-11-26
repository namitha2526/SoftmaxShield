# Multi-Service Production Dockerfile for SoftmaxShield
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib_cache

WORKDIR /app

# Copy dependency definition
COPY requirements.txt setup.py pyproject.toml ./

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Copy application source code and benchmark data
COPY softmaxshield/ ./softmaxshield/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY app.py demo_hallucinated.py README.md ./

# Expose ports: 8000 (FastAPI REST Service), 8501 (Streamlit Web Dashboard)
EXPOSE 8000 8501

# Default entrypoint starts the FastAPI service
CMD ["uvicorn", "softmaxshield.api.service:app", "--host", "0.0.0.0", "--port", "8000"]

