# Line 1: Use an official lightweight Debian-based Python 3.11 slim image
# This minimizes container attack surface and keeps the image size under 500MB
FROM python:3.11-slim

# Line 2: Prevent Python from writing .pyc bytecode files to disk (reduces disk bloat)
ENV PYTHONDONTWRITEBYTECODE=1

# Line 3: Force Python stdout and stderr streams to be unbuffered
# Ensures real-time container log streaming into docker logs / Kubernetes logs
ENV PYTHONUNBUFFERED=1

# Line 4: Set the default container execution directory to /app
WORKDIR /app

# Line 5: Install essential operating system dependencies
# - build-essential: Provides GCC compilers if packages need C-extension compilation
# - curl: Required for Docker container health check probes
# - rm -rf /var/lib/apt/lists/*: Cleans up apt caches to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Line 6: Copy ONLY requirements.txt first to leverage Docker layer caching
# If application code changes, Docker reuses this cached layer without reinstalling dependencies
COPY requirements.txt /app/

# Line 7: Install PyTorch CPU first (lightweight, avoids CUDA bloat) followed by requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Line 8: Copy the source code directory (serving engine, pipelines, monitoring) into /app/src/
COPY src/ /app/src/

# Line 9: Copy experiment configurations (YAML files) into /app/configs/
COPY configs/ /app/configs/

# Line 10: Copy Cora dataset files and disk caches into /app/data/
COPY data/ /app/data/

# Line 11: Document that container listens on HTTP port 8000 for incoming API traffic
EXPOSE 8000

# Line 12: Automated Docker Healthcheck
# Pings http://localhost:8000/health every 30 seconds; flags container as UNHEALTHY if curl fails
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Line 13: Container startup command
# Launches production Uvicorn ASGI server binding to 0.0.0.0 on port 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
