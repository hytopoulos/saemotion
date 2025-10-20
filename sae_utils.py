"""
Shared utilities for SAEmotion analysis scripts.
Contains common code for model loading, data processing, and caching.
"""
import os
import base64
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import open_clip

from model import SAEConfig, SAE

CLASS_LABELS = [
    # demographic
    "Kid", "Teenager", "Adult", "Male", "Female",
    # unknown
    "Valence","Arousal","Dominance",
    # emotions
    "Peace","Affection","Esteem","Anticipation","Engagement","Confidence",
    "Happiness","Pleasure","Excitement","Surprise","Sympathy","Doubt/Confusion",
    "Disconnection","Fatigue","Embarrassment","Yearning","Disapproval","Aversion",
    "Annoyance","Anger","Sensitivity","Sadness","Disquietment","Fear","Pain","Suffering"
]


# =============================
# Helpers
# =============================

def device_str() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def ckpt_path(steps: int, base_dir: str = "ckpt") -> Tuple[str, str]:
    """Return paths to model and optimizer checkpoints."""
    return f"{base_dir}/{steps}.pt", f"{base_dir}/{steps}.optim.pt"


def emb_url(embedding: np.ndarray) -> str:
    """Generate nooscope search URL for an embedding."""
    meme_search_url = "https://nooscope.osmarks.net/?page=advanced&e="
    return meme_search_url + base64.urlsafe_b64encode(
        embedding.astype(np.float16).tobytes()
    ).decode("utf-8")


def feature_vector(model: SAE, j: int) -> np.ndarray:
    """Return decoder column j (feature direction in embedding space)."""
    W = model.down_proj.weight.detach().cpu().numpy()  # (d_emb, d_hidden)
    return W[:, j]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize a vector."""
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return x
    return x / norm


def build_weighted_vector(sae_model: SAE, feature_weights: list) -> np.ndarray:
    if not feature_weights:
        raise RuntimeError("No feature weights provided.")

    first_fv = feature_vector(sae_model, feature_weights[0][0])
    vec_sum = np.zeros_like(first_fv, dtype=np.float32)

    for feature_id, weight in feature_weights:
        fv = feature_vector(sae_model, feature_id)
        vec_sum += fv * float(weight)

    return vec_sum.astype(np.float32)


# =============================
# Dataset
# =============================

class ImageDataset(Dataset):
    """Dataset for loading images from numpy arrays or image files."""
    
    def __init__(self, images_paths, root: str, transform=None):
        self.paths = [str(p) for p in images_paths if isinstance(p, str) and len(p)]
        self.root = root
        self.transform = transform

    def _to_pil(self, arr: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
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


# =============================
# Dataclasses
# =============================

class TrainConfig:
    """Placeholder for unpickling legacy checkpoints."""
    pass

class RunOutputs:
    name: str
    embs_path: str
    mean_mag: torch.Tensor
    freq: torch.Tensor
    top_freq_idx: np.ndarray
    top_mag_idx: np.ndarray


# =============================
# Model Loading
# =============================

def load_clip_model(model_name: str = "ViT-SO400M-14-SigLIP-384", device: str = None):
    """Load and return CLIP model, tokenizer, and preprocessing transform."""
    if device is None:
        device = device_str()
    
    pretrained_tag = dict(open_clip.list_pretrained())[model_name]
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        device=device,
        pretrained=pretrained_tag,
        precision="fp32",
    )
    clip_model.eval()
    return clip_model, preprocess


def load_sae_model(ckpt_steps: int, ckpt_dir: str = "ckpt", device: str = None):
    """Load and return SAE model from checkpoint."""
    if device is None:
        device = device_str()
    
    model_path, _ = ckpt_path(ckpt_steps, ckpt_dir)
    state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict["config"].model.device = device
    sae = SAE(state_dict["config"].model)
    sae.load_state_dict(state_dict["model"], strict=False)
    sae.to(device).eval()
    return sae


# =============================
# SAE Processing
# =============================

def run_sae_on_embs(sae: SAE, embs_npy: str, device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SAE on embeddings from a numpy file."""
    embs = np.load(embs_npy)
    return run_sae_on_embs_array(sae, embs, device=device)


def run_sae_on_embs_array(sae: SAE, embs_np: np.ndarray, batch: int = 16, device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run SAE on embeddings array and return mean magnitude and frequency."""
    if device is None:
        device = device_str()
    
    # Ensure dtype float32 for SAE input
    if embs_np.dtype != np.float32:
        embs_np = embs_np.astype(np.float32, copy=False)
    
    with torch.inference_mode():
        with sae.activation_logging():
            for i in tqdm(range(0, len(embs_np), batch), desc="SAE pass (array)"):
                x = torch.from_numpy(embs_np[i:i+batch]).to(device)
                _ = sae(x)
    
    mean_mag, freq, n, _ = sae.get_activation_stats()
    print(f"samples seen: {n}")
    return mean_mag, freq


# =============================
# Caching
# =============================

def _valid_paths(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for valid image paths in dataframe."""
    return df['Arr_name'].notnull() & (df['Arr_name'].astype(str).str.len() > 0)


def build_global_cache(
    df: pd.DataFrame,
    clip_model,
    preprocess,
    img_root: str,
    cache_npy: str = "all_clip_embs_fp16.npy",
    cache_csv: str = "all_clip_index.csv",
    batch_size: int = 16,
    device: str = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build (or load) a global cache of CLIP embeddings for ALL unique images.
    Returns a memory-mappable embeddings array and a path->row index map.
    """
    if device is None:
        device = device_str()
    
    paths = df.loc[_valid_paths(df), 'Arr_name'].astype(str)
    uniq_paths = np.unique(paths.values)

    if os.path.exists(cache_npy) and os.path.exists(cache_csv):
        print("Loading global cache from disk…")
        idx_df = pd.read_csv(cache_csv)
        path2idx = {p: int(i) for p, i in zip(idx_df['path'], idx_df['idx'])}
        embs = np.load(cache_npy, mmap_mode='r')  # float16 memmap
        # Basic sanity: ensure shapes match
        if len(path2idx) == embs.shape[0]:
            return embs, path2idx
        else:
            print("Global cache shape mismatch; rebuilding…")

    print("Encoding ALL unique images once → global cache…")
    ds = ImageDataset(uniq_paths, root=img_root, transform=preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    # We'll accumulate in float16 (after L2 norm) to save disk/RAM
    all_embs = []
    with torch.inference_mode():
        for batch_imgs in tqdm(loader, desc="Encoding ALL images (global cache)"):
            batch_imgs = batch_imgs.to(device, non_blocking=False)
            feats = clip_model.encode_image(batch_imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().numpy().astype(np.float16))
    embs = np.concatenate(all_embs, axis=0)

    # Persist
    np.save(cache_npy, embs)
    path2idx = {p: i for i, p in enumerate(uniq_paths)}
    pd.DataFrame({"path": uniq_paths, "idx": np.arange(len(uniq_paths), dtype=int)}).to_csv(cache_csv, index=False)
    print(f"Saved global cache: {embs.shape} → {cache_npy}, index → {cache_csv}")
    # Reload as memmap for safer downstream slicing
    embs = np.load(cache_npy, mmap_mode='r')
    return embs, path2idx


def subset_embeddings(paths_iterable, path2idx: Dict[str, int], embs: np.ndarray) -> np.ndarray:
    """Extract subset of embeddings for given paths."""
    idxs = [path2idx[p] for p in (str(x) for x in paths_iterable) if p in path2idx]
    if len(idxs) == 0:
        raise RuntimeError("No paths from subset were found in the global cache.")
    idxs = np.asarray(idxs, dtype=np.int64)
    # Slice from memmap (float16) and upcast once for SAE
    sub = np.asarray(embs[idxs], dtype=np.float32)
    return sub

def build_feature_emotion_mean_var_matrix(df: pd.DataFrame,
                                          class_labels: list,
                                          all_embs: np.ndarray,
                                          path2idx: dict,
                                          sae: SAE,
                                          device: str = None) -> pd.DataFrame:

    if device is None:
        device = device_str()
    
    # --- helper: create mask from series ---
    def _create_mask(series):
        """Convert a series to a boolean mask, handling bool/numeric types."""
        if series.dtype == bool:
            return series
        # For numeric: try > 0, fallback to >= 0.5, then != 0
        mask = series > 0
        if not mask.any():
            try:
                mask = series >= 0.5
            except Exception:
                mask = series != 0
        return mask

    # Map labels to actual columns - prefer exact match, then case-insensitive
    label_to_col = {}
    df_cols = set(df.columns)
    
    for lbl in class_labels:
        # Try exact "<Label>_logit" first
        if f"{lbl}_logit" in df_cols:
            label_to_col[lbl] = f"{lbl}_logit"
        # Try exact label match
        elif lbl in df_cols:
            label_to_col[lbl] = lbl
        # Try case-insensitive match
        else:
            lower_lbl = lbl.lower()
            for col in df.columns:
                if col.lower() == f"{lower_lbl}_logit" or col.lower() == lower_lbl:
                    label_to_col[lbl] = col
                    break

    if not label_to_col:
        available = ", ".join(list(df.columns)[:40])
        raise KeyError(
            f"Could not match any CLASS_LABELS to dataframe columns. "
            f"Available columns include: {available}..."
        )

    # Helper for computing mean and variance vectors for a boolean mask over rows
    def _mean_var_for_subset(mask):
        sub_df = df[mask]
        if len(sub_df) == 0:
            return None, None
        embs = subset_embeddings(sub_df['Arr_name'].astype(str), path2idx, all_embs)
        mean_mag, freq = run_sae_on_embs_array(sae, embs, device=device)
        return mean_mag.detach().cpu().numpy(), freq.detach().cpu().numpy()

    # Determine number of SAE features by probing the first mapped class with data
    n_features = None
    for lbl, col in label_to_col.items():
        mask = _create_mask(df[col])
        if mask.sum() > 0:
            probe_mean, probe_freq = _mean_var_for_subset(mask)
            if probe_freq is not None:
                n_features = probe_freq.shape[0]
                break
    
    if n_features is None:
        raise ValueError("No samples found for any mapped class to determine feature dimension.")

    # Initialize matrices for mean and variance
    mean_mat = np.zeros((n_features, len(class_labels)), dtype=np.float32)
    var_mat = np.zeros((n_features, len(class_labels)), dtype=np.float32)

    # Fill columns
    for j, lbl in enumerate(class_labels):
        col = label_to_col.get(lbl)
        if col is None:
            continue
        
        mask = _create_mask(df[col])
        if not mask.any():
            continue
            
        mean_mag, freq = _mean_var_for_subset(mask)
        if freq is None or mean_mag is None:
            continue
            
        k = min(n_features, freq.shape[0])
        mean_mat[:k, j] = mean_mag[:k]
        var_mat[:k, j] = freq[:k]

    # Build output DataFrame with interleaved mean and var columns
    out_df = pd.DataFrame({"feat": np.arange(n_features, dtype=int)})
    for lbl in class_labels:
        idx = class_labels.index(lbl)
        out_df[f"{lbl}_mean"] = mean_mat[:, idx]
        out_df[f"{lbl}_var"] = var_mat[:, idx]
    
    return out_df


def save_feature_emotion_mean_var_matrix(df: pd.DataFrame,
                                         class_labels: list,
                                         all_embs: np.ndarray,
                                         path2idx: dict,
                                         sae: SAE,
                                         device: str = None,
                                         out_csv: str = "feature_by_emotion_mean_var.csv") -> pd.DataFrame:
    """
    builds the mean/variance matrix and saves as CSV.
    """
    mat_df = build_feature_emotion_mean_var_matrix(df, class_labels, all_embs, path2idx, sae, device)
    mat_df.to_csv(out_csv, index=False)
    print(f"Saved feature x emotion mean/var matrix → {out_csv}")
    return mat_df
