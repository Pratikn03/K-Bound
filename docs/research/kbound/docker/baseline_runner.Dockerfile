# Pinned reproducible runner for TTA baselines (Kim et al. 2024, Baek et al. 2022, POEM, AETTA)
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements_baseline.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_baseline.txt

# Entrypoint for running reproducible baseline validations
CMD ["python3", "-c", "import torch, timm; print(f'TTA Baseline Environment ready: PyTorch {torch.__version__}, timm {timm.__version__}')"]
