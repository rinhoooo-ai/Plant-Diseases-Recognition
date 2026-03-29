"""
backend/main.py
===============
FastAPI backend cho Plant Disease Recognition app.

Pipeline per request:
  1. Nhận ảnh upload
  2. CLIP ViT-L/14 → image embedding
  3. Qwen3-VL-7B (hoặc 32B) → mô tả triệu chứng bằng ngôn ngữ tự nhiên
  4. SentenceTransformer → text embedding của symptom description
  5. FAISS dual retrieval: top-K từ image index + top-K từ text index
  6. Weighted voting → bệnh + confidence score
  7. Trả về JSON: disease, confidence, reasoning_trace, top_candidates

Usage:
    # Cài deps
    pip install fastapi uvicorn[standard] torch transformers faiss-gpu
    pip install sentence-transformers Pillow python-multipart bitsandbytes accelerate

    # Chạy
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Environment variables (optional):
    VLM_MODEL_ID    : HF model id cho VLM (default: Qwen/Qwen3-VL-7B-Instruct)
    INDEX_DIR       : Đường dẫn thư mục FAISS index (default: ./faiss_index)
    TOP_K           : Số kết quả retrieve mỗi index (default: 10)
    IMG_WEIGHT      : Trọng số image retrieval (default: 0.6)
    TXT_WEIGHT      : Trọng số text retrieval (default: 0.4)
"""

import os
import io
import pickle
import base64
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import faiss
from PIL import Image
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# CLIP
from transformers import CLIPProcessor, CLIPModel

# Qwen3-VL
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor as Qwen3Processor,
    BitsAndBytesConfig,
)

# Sentence Transformer
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config từ env
# ─────────────────────────────────────────────
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-7B-Instruct")
INDEX_DIR    = os.getenv("INDEX_DIR", "./faiss_index")
TOP_K        = int(os.getenv("TOP_K", "10"))
IMG_WEIGHT   = float(os.getenv("IMG_WEIGHT", "0.6"))
TXT_WEIGHT   = float(os.getenv("TXT_WEIGHT", "0.4"))
CLIP_MODEL   = "openai/clip-vit-large-patch14"
SBERT_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE} | VLM: {VLM_MODEL_ID}")


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────
class PredictionResult(BaseModel):
    disease: str
    confidence: float          # 0.0 – 1.0
    reasoning_trace: str       # Symptom description từ Qwen3-VL
    top_candidates: List[Dict[str, Any]]  # [{disease, score, votes}]
    latency_ms: float


# ─────────────────────────────────────────────
# Global model holders (loaded once at startup)
# ─────────────────────────────────────────────
class ModelStore:
    clip_model: Optional[CLIPModel] = None
    clip_processor: Optional[CLIPProcessor] = None
    vlm_model: Optional[Any] = None
    vlm_processor: Optional[Any] = None
    sbert: Optional[SentenceTransformer] = None
    image_index: Optional[faiss.Index] = None
    text_index: Optional[faiss.Index] = None
    metadata: Optional[pd.DataFrame] = None

store = ModelStore()


# ─────────────────────────────────────────────
# Startup: load tất cả models
# ─────────────────────────────────────────────
def load_clip():
    """Load CLIP ViT-L/14."""
    logger.info(f"Loading CLIP: {CLIP_MODEL}")
    store.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    store.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    store.clip_model.eval()
    logger.info("CLIP loaded ✓")


def load_vlm():
    """
    Load Qwen3-VL với 4-bit quantization để tiết kiệm VRAM.
    Nếu dùng 32B và có A100 80GB full, bỏ quantization_config.
    """
    logger.info(f"Loading VLM: {VLM_MODEL_ID}")

    # 4-bit quantization config (BitsAndBytes NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    store.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",          # tự phân bổ lên GPU/CPU
        torch_dtype=torch.bfloat16,
    )
    store.vlm_processor = Qwen3Processor.from_pretrained(VLM_MODEL_ID)
    store.vlm_model.eval()
    logger.info("VLM loaded ✓")


def load_sbert():
    """Load Sentence Transformer."""
    logger.info(f"Loading SBERT: {SBERT_MODEL}")
    store.sbert = SentenceTransformer(SBERT_MODEL)
    logger.info("SBERT loaded ✓")


def load_faiss():
    """Load FAISS indexes và metadata từ disk."""
    index_dir = Path(INDEX_DIR)
    if not index_dir.exists():
        raise RuntimeError(f"Index directory không tồn tại: {INDEX_DIR}. "
                           "Chạy build_faiss_index.py trước!")

    store.image_index = faiss.read_index(
        str(index_dir / "image_index.faiss"))
    store.text_index = faiss.read_index(
        str(index_dir / "text_index.faiss"))

    # Load lên GPU nếu có
    if DEVICE == "cuda" and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        store.image_index = faiss.index_cpu_to_gpu(res, 0, store.image_index)
        store.text_index  = faiss.index_cpu_to_gpu(res, 0, store.text_index)

    with open(index_dir / "metadata.pkl", "rb") as f:
        store.metadata = pickle.load(f)

    logger.info(f"FAISS loaded ✓  ({store.image_index.ntotal} vectors)")


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────
@torch.no_grad()
def embed_image_clip(pil_image: Image.Image) -> np.ndarray:
    """Extract và normalize CLIP embedding từ PIL image."""
    inputs = store.clip_processor(
        images=pil_image, return_tensors="pt"
    ).to(DEVICE)
    feat = store.clip_model.get_image_features(**inputs)  # (1, 768)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()  # (1, 768)


@torch.no_grad()
def describe_symptoms_qwen(pil_image: Image.Image) -> str:
    """
    Dùng Qwen3-VL verbalize các triệu chứng nhìn thấy trên lá cây.
    Prompt được thiết kế để output ngắn gọn, súc tích — phù hợp làm
    text query cho FAISS retrieval.
    """
    # Convert PIL → bytes để pass vào processor
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)

    # Prompt hướng dẫn mô tả triệu chứng bệnh
    prompt_text = (
        "You are a plant pathology expert. "
        "Carefully examine this leaf image and describe ONLY the visible disease symptoms. "
        "Focus on: leaf color changes, spots/lesions (shape, color, size, location), "
        "texture changes, wilting, necrosis, or any abnormal patterns. "
        "Be concise and specific. If the leaf appears healthy, say 'Healthy leaf, no symptoms.' "
        "Do NOT diagnose yet — only describe what you see."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # Qwen3-VL nhận base64 encoded image
                    "image": "data:image/jpeg;base64," +
                             base64.b64encode(buf.read()).decode()
                },
                {"type": "text", "text": prompt_text}
            ],
        }
    ]

    inputs = store.vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Di chuyển tensor lên device của model
    inputs = {k: v.to(store.vlm_model.device)
              for k, v in inputs.items()
              if isinstance(v, torch.Tensor)}

    generated_ids = store.vlm_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,        # greedy cho reproducibility
        temperature=None,
        top_p=None,
    )

    # Trim phần prompt
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    description = store.vlm_processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return description.strip()


def embed_text_sbert(text: str) -> np.ndarray:
    """Encode symptom description thành SBERT embedding."""
    emb = store.sbert.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return emb.astype(np.float32)  # (1, 384)


def retrieve_candidates(image_embed: np.ndarray,
                         text_embed: np.ndarray,
                         top_k: int) -> Dict[str, float]:
    """
    Dual FAISS retrieval + weighted voting.

    image_embed: (1, 768) float32
    text_embed:  (1, 384) float32

    Returns dict: {disease_label: aggregated_score}
    """
    meta = store.metadata

    # --- Image index search ---
    img_scores, img_ids = store.image_index.search(image_embed, top_k)
    img_scores = img_scores[0]   # (top_k,)
    img_ids    = img_ids[0]      # (top_k,)

    # --- Text index search ---
    txt_scores, txt_ids = store.text_index.search(text_embed, top_k)
    txt_scores = txt_scores[0]
    txt_ids    = txt_ids[0]

    # --- Weighted vote aggregation ---
    vote_dict: Dict[str, float] = {}

    for idx, score in zip(img_ids, img_scores):
        if idx < 0 or idx >= len(meta):
            continue
        label = meta.iloc[idx]["disease_label"]
        vote_dict[label] = vote_dict.get(label, 0.0) + IMG_WEIGHT * float(score)

    for idx, score in zip(txt_ids, txt_scores):
        if idx < 0 or idx >= len(meta):
            continue
        label = meta.iloc[idx]["disease_label"]
        vote_dict[label] = vote_dict.get(label, 0.0) + TXT_WEIGHT * float(score)

    return vote_dict


def format_top_candidates(vote_dict: Dict[str, float],
                           n: int = 5) -> List[Dict]:
    """Sắp xếp và normalize thành top-N candidates."""
    sorted_items = sorted(vote_dict.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:n]

    total = sum(s for _, s in top) or 1.0
    return [
        {
            "disease": label,
            "score": round(float(score), 4),
            "confidence_pct": round(float(score) / total * 100, 1)
        }
        for label, score in top
    ]


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Recognition API",
    description="CLIP + Qwen3-VL + FAISS dual retrieval pipeline",
    version="1.0.0"
)

# CORS — cho phép React frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Thay bằng domain cụ thể khi production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load tất cả models khi server khởi động."""
    logger.info("=== Server startup: loading models ===")
    load_clip()
    load_vlm()
    load_sbert()
    load_faiss()
    logger.info("=== Tất cả models đã load xong ✓ ===")


@app.get("/health")
async def health_check():
    """Kiểm tra server có sẵn sàng không."""
    return {
        "status": "ok",
        "device": DEVICE,
        "vlm_model": VLM_MODEL_ID,
        "index_size": store.image_index.ntotal if store.image_index else 0,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh lá cây, trả về dự đoán bệnh + reasoning trace.

    Request: multipart/form-data với field 'file' là ảnh JPG/PNG.
    Response: JSON PredictionResult.
    """
    t0 = time.perf_counter()

    # ── 1. Đọc và validate ảnh ──────────────────
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh (JPG/PNG/WebP).")

    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Không đọc được ảnh: {e}")

    logger.info(f"Received image: {pil_image.size} | {file.filename}")

    # ── 2. CLIP image embedding ──────────────────
    image_embed = embed_image_clip(pil_image)   # (1, 768)

    # ── 3. Qwen3-VL symptom description ─────────
    reasoning_trace = describe_symptoms_qwen(pil_image)
    logger.info(f"Symptoms: {reasoning_trace[:120]}...")

    # ── 4. SBERT text embedding ──────────────────
    text_embed = embed_text_sbert(reasoning_trace)  # (1, 384)

    # ── 5. FAISS dual retrieval + voting ─────────
    vote_dict = retrieve_candidates(image_embed, text_embed, TOP_K)

    if not vote_dict:
        raise HTTPException(500, "Không tìm thấy kết quả phù hợp.")

    # ── 6. Format kết quả ────────────────────────
    top_candidates = format_top_candidates(vote_dict, n=5)
    best = top_candidates[0]
    total_score = sum(c["score"] for c in top_candidates)
    confidence = round(best["score"] / total_score, 4) if total_score > 0 else 0.0

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Prediction: {best['disease']} ({confidence:.2%}) | {latency_ms}ms")

    return PredictionResult(
        disease=best["disease"],
        confidence=confidence,
        reasoning_trace=reasoning_trace,
        top_candidates=top_candidates,
        latency_ms=latency_ms,
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                workers=1, reload=False)
