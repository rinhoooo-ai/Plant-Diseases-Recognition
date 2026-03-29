# backend/Dockerfile — HF Spaces GPU (T4)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces yêu cầu user non-root
RUN useradd -m -u 1000 user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /home/user/app

# Install Python deps
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir faiss-gpu-cu12
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=user main.py .
COPY --chown=user faiss_index/ ./faiss_index/

USER user

ENV INDEX_DIR=/home/user/app/faiss_index
ENV VLM_MODEL_ID=Qwen/Qwen3-VL-7B-Instruct
ENV TOP_K=10
ENV IMG_WEIGHT=0.6
ENV TXT_WEIGHT=0.4

# HF Spaces dùng port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
