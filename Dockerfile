FROM python:3.13.7-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --prefix=/install .


FROM python:3.13.7-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 fraudapp \
    && useradd --system --uid 10001 --gid fraudapp --home-dir /nonexistent fraudapp

WORKDIR /app

COPY --from=builder /install /usr/local
COPY configs ./configs

RUN mkdir -p /app/models /app/data/processed/features \
    && chown -R fraudapp:fraudapp /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2); raise SystemExit(0 if response.status == 200 else 1)"]

CMD ["uvicorn", "fraud_detection.serving.api:create_scored_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
