FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Docker group access for managing sub-containers
ARG DOCKER_GID=999
RUN groupadd -g ${DOCKER_GID} docker 2>/dev/null || true && \
    useradd -m -s /bin/bash service && \
    usermod -aG docker service && \
    mkdir -p /app /data && \
    chown -R service:service /app /data

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/ ./service/
COPY mcp_local/ ./mcp_local/
COPY config/ ./config/

VOLUME /data

EXPOSE 11434

# Run as root for Docker socket access (container manager needs it)
# USER service

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:11434/health || exit 1

CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "11434"]
