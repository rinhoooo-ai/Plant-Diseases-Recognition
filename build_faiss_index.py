"""
build_faiss_index.py
====================
Bước 1: Đọc PlantVillageVQA dataset từ HuggingFace, extract CLIP image embeddings
và Sentence Transformer text embeddings, lưu vào FAISS dual index.

Chạy một lần trước khi khởi động backend.

Usage:
    python build_faiss_index.py \
        --images_dir /data/images \
        --csv_path /data/PlantVillageVQA.csv \
        --output_dir /data/faiss_index \
        --batch_size 64 \
        --split train

Requirements:
    pip install torch torchvision transformers faiss-gpu sentence-transformers
    pip install datasets pandas Pillow tqdm
"""

import os
import argparse
import numpy as np
import pandas as pd
import faiss
import torch
import pickle
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional

# CLIP
from transformers import CLIPProcessor, CLIPModel

# Sentence Transformer
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Config mặc định
# ─────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"          # ViT-L/14
SBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" # nhẹ, nhanh, đủ tốt
IMAGE_EMBED_DIM = 768   # CLIP ViT-L/14 output dim
TEXT_EMBED_DIM = 384    # all-MiniLM-L6-v2 output dim


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for PlantVillage")
    parser.add_argument("--images_dir", type=str, default="./images",
                        help="Thư mục chứa ảnh JPEGs")
    parser.add_argument("--csv_path", type=str, default="./PlantVillageVQA.csv",
                        help="File CSV annotation")
    parser.add_argument("--output_dir", type=str, default="./faiss_index",
                        help="Thư mục lưu FAISS index và metadata")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size khi extract embeddings")
    parser.add_argument("--split", type=str, default="train",
                        help="Chỉ index split này (train/val/test/all)")
    parser.add_argument("--question_type", type=str,
                        default="Specific Disease Identification",
                        help="Lọc theo question_type để lấy disease label")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Giới hạn số mẫu (debug)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
def load_clip(device: str):
    """Load CLIP ViT-L/14 lên GPU."""
    print(f"[CLIP] Loading {CLIP_MODEL_ID} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


def load_sbert():
    """Load Sentence Transformer (CPU đủ dùng vì text nhỏ)."""
    print(f"[SBERT] Loading {SBERT_MODEL_ID} ...")
    return SentenceTransformer(SBERT_MODEL_ID)


# ─────────────────────────────────────────────
# Prepare dataset
# ─────────────────────────────────────────────
def prepare_records(csv_path: str, images_dir: str,
                    split: str, question_type: str,
                    max_samples: Optional[int]) -> pd.DataFrame:
    """
    Đọc CSV, lọc theo split + question_type, ghép image_path tuyệt đối.
    Trả về DataFrame với các cột: image_id, image_path, disease_label, answer.
    """
    print(f"[DATA] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Lọc split
    if split != "all":
        df = df[df["split"] == split]

    # Lọc question type để lấy disease name
    df_disease = df[df["question_type"] == question_type].copy()

    # Mỗi image chỉ cần 1 record
    df_disease = df_disease.drop_duplicates(subset="image_id")

    if max_samples:
        df_disease = df_disease.head(max_samples)

    # Ghép đường dẫn tuyệt đối
    df_disease["abs_image_path"] = df_disease["image_path"].apply(
        lambda p: os.path.join(images_dir, os.path.basename(p))
    )

    # Lọc file tồn tại
    exists_mask = df_disease["abs_image_path"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"[WARN] {missing} ảnh không tìm thấy, bỏ qua.")
    df_disease = df_disease[exists_mask].reset_index(drop=True)

    print(f"[DATA] Tổng records: {len(df_disease)} | "
          f"Classes: {df_disease['answer'].nunique()}")
    return df_disease


# ─────────────────────────────────────────────
# Extract embeddings
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_clip_embeddings(df: pd.DataFrame,
                             clip_model, clip_processor,
                             device: str, batch_size: int) -> np.ndarray:
    """
    Extract CLIP ViT-L/14 image embeddings theo batch.
    Returns: float32 array shape (N, 768), đã L2-normalize.
    """
    all_embeds = []
    paths = df["abs_image_path"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP embed"):
        batch_paths = paths[i: i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                # Ảnh lỗi → thay bằng ảnh trắng 224x224
                print(f"[WARN] Không đọc được {p}: {e}")
                images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

        inputs = clip_processor(images=images, return_tensors="pt",
                                padding=True).to(device)
        feats = clip_model.get_image_features(**inputs)  # (B, 768)
        # L2 normalize
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu().float().numpy())

    return np.vstack(all_embeds)  # (N, 768)


def extract_text_embeddings(df: pd.DataFrame,
                             sbert_model,
                             batch_size: int) -> np.ndarray:
    """
    Encode disease label (answer) bằng Sentence Transformer.
    Returns: float32 array shape (N, 384), đã L2-normalize.
    """
    texts = df["answer"].tolist()
    print("[SBERT] Encoding disease labels ...")
    embeds = sbert_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize tự động
        convert_to_numpy=True
    )
    return embeds.astype(np.float32)


# ─────────────────────────────────────────────
# Build FAISS indexes
# ─────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray, dim: int,
                      use_gpu: bool = True) -> faiss.Index:
    """
    Tạo FAISS IndexFlatIP (Inner Product = cosine similarity sau normalize).
    Dùng GPU nếu có.
    """
    index = faiss.IndexFlatIP(dim)

    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"[FAISS] Sử dụng GPU (có {faiss.get_num_gpus()} GPU)")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    print(f"[FAISS] Index built: {index.ntotal} vectors, dim={dim}")
    return index


# ─────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────
def save_outputs(output_dir: str,
                 image_index: faiss.Index,
                 text_index: faiss.Index,
                 metadata: pd.DataFrame,
                 image_embeds: np.ndarray,
                 text_embeds: np.ndarray):
    """Lưu FAISS indexes + metadata + embeddings arrays."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert GPU index về CPU trước khi save
    def to_cpu(idx):
        try:
            return faiss.index_gpu_to_cpu(idx)
        except Exception:
            return idx

    faiss.write_index(to_cpu(image_index),
                      os.path.join(output_dir, "image_index.faiss"))
    faiss.write_index(to_cpu(text_index),
                      os.path.join(output_dir, "text_index.faiss"))

    # Lưu numpy arrays để inspect sau
    np.save(os.path.join(output_dir, "image_embeds.npy"), image_embeds)
    np.save(os.path.join(output_dir, "text_embeds.npy"), text_embeds)

    # Metadata: image_id, disease_label, image_path
    meta = metadata[["image_id", "answer", "abs_image_path"]].copy()
    meta.rename(columns={"answer": "disease_label"}, inplace=True)
    meta.to_pickle(os.path.join(output_dir, "metadata.pkl"))
    meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    # Lưu config
    config = {
        "clip_model": CLIP_MODEL_ID,
        "sbert_model": SBERT_MODEL_ID,
        "image_embed_dim": IMAGE_EMBED_DIM,
        "text_embed_dim": TEXT_EMBED_DIM,
        "n_records": len(metadata),
    }
    with open(os.path.join(output_dir, "index_config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print(f"[SAVE] Đã lưu tất cả vào {output_dir}/")
    print(f"  image_index.faiss  ({image_embeds.shape})")
    print(f"  text_index.faiss   ({text_embeds.shape})")
    print(f"  metadata.pkl       ({len(meta)} rows)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CONFIG] Device: {device}")

    # 1. Load models
    clip_model, clip_processor = load_clip(device)
    sbert_model = load_sbert()

    # 2. Prepare data
    df = prepare_records(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        split=args.split,
        question_type=args.question_type,
        max_samples=args.max_samples
    )

    # 3. Extract embeddings
    image_embeds = extract_clip_embeddings(
        df, clip_model, clip_processor, device, args.batch_size
    )
    text_embeds = extract_text_embeddings(df, sbert_model, args.batch_size)

    # 4. Build FAISS indexes
    use_gpu = (device == "cuda")
    image_index = build_faiss_index(image_embeds, IMAGE_EMBED_DIM, use_gpu)
    text_index = build_faiss_index(text_embeds, TEXT_EMBED_DIM, use_gpu)

    # 5. Save
    save_outputs(
        output_dir=args.output_dir,
        image_index=image_index,
        text_index=text_index,
        metadata=df,
        image_embeds=image_embeds,
        text_embeds=text_embeds
    )

    print("\n✅ Index building hoàn tất!")
    print(f"   Tổng ảnh indexed: {len(df)}")
    print(f"   Số disease classes: {df['answer'].nunique()}")


if __name__ == "__main__":
    main()
