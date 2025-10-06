# %%
import os
import base64
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import open_clip

# =============================
# Config
# =============================
ANNOTS_CSV = "../archive/annots_arrs/annot_arrs_train.csv"
IMG_ROOT = "../archive/img_arrs"
BATCH_SIZE = 16
TOP_K = 10
CKPT_STEPS = 3189   # your saved SAE checkpoint step

# Global cache files (encode each unique image exactly once)
GLOBAL_EMBS_NPY = "all_clip_embs_fp16.npy"
GLOBAL_INDEX_CSV = "all_clip_index.csv"

# =============================
# SAE import
# =============================
import sys
sys.path.append("../")
from model import SAEConfig, SAE  # noqa: E402

# =============================
# Helpers
# =============================

def device_str() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def ckpt_path(steps: int) -> Tuple[str, str]:
    return f"../ckpt/{steps}.pt", f"../ckpt/{steps}.optim.pt"


def emb_url(embedding: np.ndarray) -> str:
    meme_search_url = "https://nooscope.osmarks.net/?page=advanced&e="
    return meme_search_url + base64.urlsafe_b64encode(
        embedding.astype(np.float16).tobytes()
    ).decode("utf-8")


def feature_vector(model: SAE, j: int) -> np.ndarray:
    """Return decoder column j (feature direction in embedding space)."""
    W = model.down_proj.weight.detach().cpu().numpy()  # (d_emb, d_hidden)
    return W[:, j]


class ImageDataset(Dataset):
    def __init__(self, images_paths, root=IMG_ROOT, transform=None):
        self.paths = [str(p) for p in images_paths if isinstance(p, str) and len(p)]
        self.root = root
        self.transform = transform

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
            arr = arr.astype(np.uint8)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return Image.fromarray(arr, mode="RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        path = os.path.join(self.root, rel)
        if rel.lower().endswith(".npy"):
            arr = np.load(path, allow_pickle=False)
            img = self._to_pil(arr)
        else:
            with Image.open(path) as im:
                img = im.convert("RGB")
        return self.transform(img) if self.transform is not None else img

@dataclass
class TrainConfig:
    model: SAEConfig
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    compile: bool

@dataclass
class RunOutputs:
    name: str
    embs_path: str
    mean_mag: torch.Tensor
    freq: torch.Tensor
    top_freq_idx: np.ndarray
    top_mag_idx: np.ndarray


# =============================
# Model init (OpenCLIP + SAE)
# =============================
print("Setting up models…")
DEV = device_str()
model_name = "ViT-SO400M-14-SigLIP-384"
pretrained_tag = dict(open_clip.list_pretrained())[model_name]
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    device=DEV,
    pretrained=pretrained_tag,
    precision="fp32",
)
clip_model.eval()

print("Loading SAE checkpoint…")
model_path, _ = ckpt_path(CKPT_STEPS)
state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
sae = SAE(state_dict["config"].model)
sae.load_state_dict(state_dict["model"], strict=False)
sae.to(DEV).eval()

# =============================
# Core pipeline (legacy encode_images retained for ad-hoc use)
# =============================

def encode_images(paths, name: str) -> str:
    """(Legacy) Encode a *specific list* of images and save to {name}_embs.npy.
    Kept for backwards-compat, but one-vs-all now uses the global cache below.
    """
    embs_npy = f"{name.lower()}_embs.npy"
    if os.path.exists(embs_npy):
        print(f"Reusing cached embeddings: {embs_npy}")
        return embs_npy
    ds = ImageDataset(paths, transform=preprocess)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    all_embs = []
    with torch.inference_mode():
        for batch_imgs in tqdm(loader, desc=f"Encoding {name}"):
            batch_imgs = batch_imgs.to(DEV, non_blocking=False)
            feats = clip_model.encode_image(batch_imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().numpy().astype(np.float16))
    embs = np.concatenate(all_embs, axis=0)
    np.save(embs_npy, embs)
    print(f"Saved {embs.shape} to {embs_npy}")
    return embs_npy


def run_sae_on_embs(embs_npy: str) -> Tuple[torch.Tensor, torch.Tensor]:
    embs = np.load(embs_npy)
    return run_sae_on_embs_array(embs)


def run_sae_on_embs_array(embs_np: np.ndarray, batch: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure dtype float32 for SAE input
    if embs_np.dtype != np.float32:
        embs_np = embs_np.astype(np.float32, copy=False)
    with torch.inference_mode():
        with sae.activation_logging():
            for i in tqdm(range(0, len(embs_np), batch), desc="SAE pass (array)"):
                x = torch.from_numpy(embs_np[i:i+batch]).to(DEV)
                _ = sae(x)
    mean_mag, freq, n, _ = sae.get_activation_stats()
    print(f"samples seen: {n}")
    return mean_mag, freq


def summarize(name: str, mean_mag: torch.Tensor, freq: torch.Tensor) -> Dict:
    top_freq = torch.topk(freq, k=TOP_K)
    top_mag = torch.topk(mean_mag, k=TOP_K)
    top_freq_idx = top_freq.indices.cpu().numpy()
    top_mag_idx = top_mag.indices.cpu().numpy()
    print(f"\nTop-{TOP_K} features by activation FREQUENCY for {name}:")
    for r, (j, f) in enumerate(zip(top_freq_idx, top_freq.values.cpu().numpy()), start=1):
        print(f"{r:2d}. feat {int(j):6d} | freq={f:.6f}")
    print(f"\nTop-{TOP_K} features by MEAN MAGNITUDE for {name}:")
    for r, (j, m) in enumerate(zip(top_mag_idx, top_mag.values.cpu().numpy()), start=1):
        print(f"{r:2d}. feat {int(j):6d} | mean_mag={m:.6f}")
    return {"top_freq_idx": top_freq_idx, "top_mag_idx": top_mag_idx}


def compare_sets(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, kind: str):
    A, B = set(a.tolist()), set(b.tolist())
    inter = A & B
    union = A | B
    jacc = len(inter) / max(1, len(union))
    print(f"\nOverlap ({kind}) between {label_a} and {label_b}: {len(inter)}/{len(union)} | Jaccard={jacc:.3f}")
    if inter:
        print("Shared feature IDs:", sorted(inter))

# =============================
# GLOBAL CACHE (encode each unique image once)
# =============================

def _valid_paths(df: pd.DataFrame) -> pd.Series:
    return df['Arr_name'].notnull() & (df['Arr_name'].astype(str).str.len() > 0)


def build_global_cache(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Build (or load) a global cache of CLIP embeddings for ALL unique images.
    Returns a memory-mappable embeddings array and a path->row index map.
    """
    paths = df.loc[_valid_paths(df), 'Arr_name'].astype(str)
    uniq_paths = np.unique(paths.values)

    if os.path.exists(GLOBAL_EMBS_NPY) and os.path.exists(GLOBAL_INDEX_CSV):
        print("Loading global cache from disk…")
        idx_df = pd.read_csv(GLOBAL_INDEX_CSV)
        path2idx = {p: int(i) for p, i in zip(idx_df['path'], idx_df['idx'])}
        embs = np.load(GLOBAL_EMBS_NPY, mmap_mode='r')  # float16 memmap
        # Basic sanity: ensure shapes match
        if len(path2idx) == embs.shape[0]:
            return embs, path2idx
        else:
            print("Global cache shape mismatch; rebuilding…")

    print("Encoding ALL unique images once → global cache…")
    ds = ImageDataset(uniq_paths, transform=preprocess)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    # We'll accumulate in float16 (after L2 norm) to save disk/RAM
    all_embs = []
    with torch.inference_mode():
        for batch_imgs in tqdm(loader, desc="Encoding ALL images (global cache)"):
            batch_imgs = batch_imgs.to(DEV, non_blocking=False)
            feats = clip_model.encode_image(batch_imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().numpy().astype(np.float16))
    embs = np.concatenate(all_embs, axis=0)

    # Persist
    np.save(GLOBAL_EMBS_NPY, embs)
    path2idx = {p: i for i, p in enumerate(uniq_paths)}
    pd.DataFrame({"path": uniq_paths, "idx": np.arange(len(uniq_paths), dtype=int)}).to_csv(GLOBAL_INDEX_CSV, index=False)
    print(f"Saved global cache: {embs.shape} → {GLOBAL_EMBS_NPY}, index → {GLOBAL_INDEX_CSV}")
    # Reload as memmap for safer downstream slicing
    embs = np.load(GLOBAL_EMBS_NPY, mmap_mode='r')
    return embs, path2idx


def subset_embeddings(paths_iterable, path2idx: Dict[str, int], embs: np.ndarray) -> np.ndarray:
    idxs = [path2idx[p] for p in (str(x) for x in paths_iterable) if p in path2idx]
    if len(idxs) == 0:
        raise RuntimeError("No paths from subset were found in the global cache.")
    idxs = np.asarray(idxs, dtype=np.int64)
    # Slice from memmap (float16) and upcast once for SAE
    sub = np.asarray(embs[idxs], dtype=np.float32)
    return sub

# =============================
# One-vs-All Δ-frequency per class (Top-5)
# =============================

CLASS_LABELS = [
    "Peace","Affection","Esteem","Anticipation","Engagement","Confidence",
    "Happiness","Pleasure","Excitement","Surprise","Sympathy","Doubt/Confusion",
    "Disconnection","Fatigue","Embarrassment","Yearning","Disapproval","Aversion",
    "Annoyance","Anger","Sensitivity","Sadness","Disquietment","Fear","Pain","Suffering"
]


def build_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if label not in df.columns:
        raise KeyError(f"Label '{label}' not found in annotations CSV columns: {list(df.columns)}")
    sub = df[df[label] > 0]
    sub = sub[_valid_paths(sub)]
    return sub


def build_neg_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if label not in df.columns:
        raise KeyError(f"Label '{label}' not found in annotations CSV columns: {list(df.columns)}")
    sub = df[df[label] <= 0]
    sub = sub[_valid_paths(sub)]
    return sub


def one_vs_all_topk(df: pd.DataFrame, labels: list[str], topk: int = 5, min_stability: float = 0.02,
                    all_embs: np.ndarray | None = None, path2idx: Dict[str, int] | None = None) -> pd.DataFrame:
    """For each label L, compute Δfreq = freq(L) - freq(not L), return topk rows per class.
    Uses the *global embedding cache* to avoid redundant CLIP work.
    """
    if all_embs is None or path2idx is None:
        raise ValueError("Global cache not initialized. Call build_global_cache(df) first and pass (all_embs, path2idx).")

    records = []

    for label in labels:
        try:
            pos_df = build_subset(df, label)
            neg_df = build_neg_subset(df, label)

            if len(pos_df) == 0 or len(neg_df) == 0:
                print(f"[{label}] skipped (pos={len(pos_df)}, neg={len(neg_df)})")
                continue

            # Gather embeddings via the global cache (no CLIP re-encoding)
            embs_pos = subset_embeddings(pos_df['Arr_name'].astype(str), path2idx, all_embs)
            embs_neg = subset_embeddings(neg_df['Arr_name'].astype(str), path2idx, all_embs)

            mean_pos, freq_pos = run_sae_on_embs_array(embs_pos)
            mean_neg, freq_neg = run_sae_on_embs_array(embs_neg)

            k = min(len(freq_pos), len(freq_neg))
            fa = freq_pos[:k].cpu().numpy()
            fb = freq_neg[:k].cpu().numpy()
            ma = mean_pos[:k].cpu().numpy()
            mb = mean_neg[:k].cpu().numpy()

            df_feat = pd.DataFrame({
                "feat": np.arange(k, dtype=int),
                "freq_pos": fa, "freq_neg": fb,
                "mag_pos":  ma, "mag_neg":  mb,
            })
            df_feat["Δfreq(pos-neg)"] = df_feat["freq_pos"] - df_feat["freq_neg"]
            df_feat["Δmag(pos-neg)"]  = df_feat["mag_pos"]  - df_feat["mag_neg"]

            # Keep features that fire sometimes in either group (stability filter)
            stable = df_feat[(df_feat["freq_pos"] > min_stability) | (df_feat["freq_neg"] > min_stability)]
            top = stable.sort_values(["Δfreq(pos-neg)", "Δmag(pos-neg)"], ascending=False).head(topk).copy()
            top.insert(0, "label", label)

            # Print a compact per-class report
            print(f"\n[{label}] Top-{topk} one-vs-all by Δfreq (pos - neg):")
            print(top[["feat","Δfreq(pos-neg)","Δmag(pos-neg)","freq_pos","freq_neg"]].to_string(index=False))

            # Optional: decoder-direction URLs for the first few features
            for j in top["feat"].astype(int).tolist()[:5]:
                try:
                    fv = feature_vector(sae, j)
                    print(f"  feat {j:6d} → {emb_url(fv)}")
                except Exception as _e:
                    print(f"  feat {j:6d} (URL skipped: {_e})")

            records.append(top)
        except Exception as e:
            print(f"[{label}] analysis error: {e}")

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    # Save a flat CSV summary per class
    out_csv = "one_vs_all_top5_delta_freq.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved per-class Top-{topk} Δ-frequency summary → {out_csv}")
    return result

# ---- run when executed directly ----
if __name__ == "__main__":
    df = pd.read_csv(ANNOTS_CSV)
    # Build/load the global cache once
    ALL_EMBS, PATH2IDX = build_global_cache(df)
    # (You already set up CLIP+SAE above; we just invoke the loop)
    _ = one_vs_all_topk(df, CLASS_LABELS, topk=5, min_stability=0.02, all_embs=ALL_EMBS, path2idx=PATH2IDX)
