FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements files
COPY api/requirements.txt ./api_requirements.txt
COPY worker/requirements.txt ./worker_requirements.txt
COPY frontend/requirements.txt ./frontend_requirements.txt
COPY omtra_requirements.txt ./omtra_requirements.txt

# Install core dependencies first
RUN pip install --no-cache-dir -r api_requirements.txt && \
    pip install --no-cache-dir -r worker_requirements.txt && \
    pip install --no-cache-dir -r frontend_requirements.txt

# Install OMTRA dependencies (optional, may fail on some platforms)
RUN pip install --no-cache-dir -r omtra_requirements.txt || echo "Some OMTRA dependencies failed to install - using stub mode"

# Copy shared module
COPY shared ./shared

# Copy all application code
COPY api ./api
COPY worker ./worker
COPY frontend ./frontend

# Copy OMTRA source code from parent directory
COPY ../omtra ./omtra
COPY ../routines ./routines

# Create directories
RUN mkdir -p /srv/app/jobs /srv/app/models

# Set permissions
RUN chmod -R 755 /app
RUN chmod -R 777 /srv/app/jobs

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "worker/worker.py"]
