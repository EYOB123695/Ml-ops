# Base lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/ /app/data/

# Expose FastAPI port
EXPOSE 8000

# Healthcheck to ensure container is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI serving app
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
