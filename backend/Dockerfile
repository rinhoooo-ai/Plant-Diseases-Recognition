# backend/Dockerfile
# Tối ưu cho RunPod A100 GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir faiss-gpu-cu12
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .

# FAISS index sẽ được mount hoặc download lúc startup
# ENV vars để configure
ENV INDEX_DIR=/data/faiss_index
ENV VLM_MODEL_ID=Qwen/Qwen3-VL-7B-Instruct
ENV TOP_K=10
ENV IMG_WEIGHT=0.6
ENV TXT_WEIGHT=0.4

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
