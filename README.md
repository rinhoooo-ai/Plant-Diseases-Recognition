# 🌿 Plant Disease Recognition AI

> CLIP ViT-L/14 + Qwen3-VL + FAISS dual-index retrieval pipeline  
> Dataset: PlantVillageVQA (55k images, 38 disease classes)

## Architecture

```
Input Image
    │
    ├─► CLIP ViT-L/14 ──────────────────► Image Embedding (768d)
    │                                              │
    └─► Qwen3-VL-7B ──► Symptom Text ─► SBERT ──► Text Embedding (384d)
                         Description               │
                              │           FAISS Dual Index Search
                              │           (Image Index + Text Index)
                              │                    │
                              └────────────────────┘
                                         │
                                   Weighted Voting
                                (img_w=0.6, txt_w=0.4)
                                         │
                                  Top-5 Candidates
                                  + Confidence Scores
```

## Model Selection (A100 80GB)

| Model | VRAM FP16 | Recommended |
|---|---|---|
| Qwen3-VL-2B-Instruct | ~5GB | Dev/testing |
| Qwen3-VL-7B-Instruct | ~16GB | ✅ Production (cân bằng) |
| **Qwen3-VL-32B-Instruct** | ~65GB | ✅ Best quality trên A100 80GB |
| Qwen3-VL-235B (MoE) | ~470GB | ❌ Cần 6x A100 |

## Setup

### Bước 1: Build FAISS Index

```bash
# Download dataset từ HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('SyedNazmusSakib/PlantVillageVQA')
ds.save_to_disk('./hf_data')
# Lưu CSV
import pandas as pd
df = ds['train'].to_pandas()
df.to_csv('./PlantVillageVQA.csv', index=False)
"

# Extract images
# (images đã có trong dataset hoặc download riêng)

# Build index
python build_faiss_index.py \
    --images_dir ./images \
    --csv_path ./PlantVillageVQA.csv \
    --output_dir ./faiss_index \
    --batch_size 64 \
    --split train
```

### Bước 2: Backend

```bash
cd backend

# Cài dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu-cu12
pip install -r requirements.txt

# Cài Qwen3-VL support (nếu transformers < 4.57)
pip install git+https://github.com/huggingface/transformers

# Set env vars
export INDEX_DIR=/path/to/faiss_index
export VLM_MODEL_ID=Qwen/Qwen3-VL-7B-Instruct   # hoặc 32B

# Chạy server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Bước 3: Frontend

```bash
cd frontend

npm install

# Dev local
cp .env.example .env.local
# Sửa VITE_API_URL trong .env.local

npm run dev
```

### Deploy Frontend lên Vercel

```bash
# Vercel CLI
npm i -g vercel
vercel --prod

# Thêm env var trong Vercel dashboard:
# VITE_API_URL = https://your-runpod-endpoint.proxy.runpod.net
```

### Deploy Backend lên RunPod

1. Upload Docker image hoặc dùng RunPod template
2. Mount `/data/faiss_index` volume
3. Set env vars:
   - `INDEX_DIR=/data/faiss_index`
   - `VLM_MODEL_ID=Qwen/Qwen3-VL-7B-Instruct`
4. Expose port 8000

## API

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf.jpg"
```

Response:
```json
{
  "disease": "Tomato Late Blight",
  "confidence": 0.72,
  "reasoning_trace": "The leaf shows large, dark brown irregular lesions with water-soaked margins, yellowing of surrounding tissue, and white mold-like growth on the underside...",
  "top_candidates": [
    {"disease": "Tomato Late Blight", "score": 1.23, "confidence_pct": 42.1},
    {"disease": "Tomato Early Blight", "score": 0.87, "confidence_pct": 29.8},
    ...
  ],
  "latency_ms": 2340.5
}
```

### GET /health

```json
{"status": "ok", "device": "cuda", "vlm_model": "Qwen/Qwen3-VL-7B-Instruct", "index_size": 45231}
```

## File Structure

```
plant_disease_app/
├── build_faiss_index.py    # Bước 1: Build FAISS index từ PlantVillageVQA
├── backend/
│   ├── main.py             # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile          # Deploy lên RunPod
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main React component
    │   ├── main.jsx
    │   └── index.css
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── vercel.json
    └── .env.example
```
