# UAIS-V production API image
FROM python:3.11-slim-bookworm@sha256:8dca233de9f3d9bb410665f00a4da6dd06f331083137e0e98ccf227236fcc438

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements-api.txt /app/requirements-api.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements-api.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin uais

COPY deploy /app/deploy
COPY src /app/src
COPY configs /app/configs

RUN mkdir -p /app/models /app/experiments \
    && chown -R uais:uais /app

USER uais

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()" || exit 1

CMD ["uvicorn", "deploy.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
