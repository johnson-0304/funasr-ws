# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY scripts/download-models.sh scripts/download-models.sh
RUN bash scripts/download-models.sh /models

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models \
    HOST=0.0.0.0 \
    PORT=10095
RUN apt-get update && apt-get install -y --no-install-recommends curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=models /models /models
COPY funasr_ws ./funasr_ws
USER app
EXPOSE 10095
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
CMD ["python", "-m", "funasr_ws"]
