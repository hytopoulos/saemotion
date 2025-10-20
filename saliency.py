"""Saliency map generation for SAE features.

This script generates saliency maps showing which image regions activate specific
SAE features. It supports both standard gradients and SmoothGrad.
"""
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

import sae_utils

# ==== CONFIGURATION ====
CSV_PATH = "Top_20_images_per_feature.csv"
OUTDIR = "saliency_outputs"
TIMEOUT = 15 # image download timeout
SMOOTHGRAD = True
N_SAMPLES = 25 # smoothgrad number of samples
SIGMA = 1 # smoothgrad noise level

# Visualization parameters
BLUR_SIGMA = 6.0 # heatmap blur
LOW_Q = 65 # heatmap low quantile
HIGH_Q = 99.8 # heatmap high quantile
GAMMA = 0.6 # heatmap gamma
ALPHA_MAX = 0.55
CONTOUR_QS = []
CMAP = "inferno"
SAVE_HEAT_ONLY = True

def saliency_from_direction(
    img_pil,
    w_np,
    l2_normalize_embed=True,
    smoothgrad=False,
    n=25,
    sigma=0.08,
    device="mps",
):
    """Compute saliency map for a given direction in embedding space.
    
    Args:
        img_pil: PIL Image to analyze
        w_np: Direction vector in embedding space (numpy array)
        l2_normalize_embed: Whether to L2 normalize embeddings and direction
        smoothgrad: Whether to use SmoothGrad (averages over noisy samples)
        n: Number of samples for SmoothGrad
        sigma: Noise level for SmoothGrad
        device: Device to run computation on
        
    Returns:
        Normalized saliency map as numpy array [H, W]
    """
    # Preprocess image
    base = emov5.preprocess(img_pil).unsqueeze(0).to(device)
    pixel_values = base.clone().requires_grad_(True)

    def forward(px):
        """Forward pass computing projection onto direction vector."""
        img_feats = emov5.clip_model.encode_image(px)
        w = torch.as_tensor(w_np, device=img_feats.device, dtype=img_feats.dtype)
        if l2_normalize_embed:
            img_feats = F.normalize(img_feats, dim=-1)
            w = F.normalize(w, dim=0)
        return (img_feats * w).sum(-1).mean()

    # Compute gradients with respect to pixels
    if not smoothgrad:
        if pixel_values.grad is not None:
            pixel_values.grad.zero_()
        score = forward(pixel_values)
        score.backward()
        sal = pixel_values.grad.detach().abs().mean(1, keepdim=False)[0]
    else:
        grads = None
        for _ in range(n):
            noisy = base + torch.randn_like(base) * sigma * base.std()
            noisy = noisy.detach().requires_grad_(True)
            if noisy.grad is not None:
                noisy.grad.zero_()
            score = forward(noisy)
            score.backward()
            g = noisy.grad.detach().abs().mean(1, keepdim=False)[0]
            grads = g if grads is None else (grads + g)
        sal = grads / float(n)

    # Normalize to [0, 1]
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal.float().cpu().numpy()

def main():
    """Generate saliency maps for top-activating images of each SAE feature.
    
    Reads a CSV with columns: feature_index, rank, image_url
    For each row, downloads the image, computes saliency map, and saves
    both overlay and heat-only visualizations.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Validate required columns
    required_cols = {"feature_index", "rank", "image_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Cache feature vectors so we compute each feature once
    fv_cache: dict[int, np.ndarray] = {}

    # Progress counters for resume behavior
    skip_count = 0
    saved_count = 0
    download_fail_count = 0
    saliency_error_count = 0

    for i, row in df.iterrows():
        j = int(row["feature_index"])
        r = int(row["rank"])
        url = str(row["image_url"])

        # Organize outputs in per-feature subfolders
        feat_dir = os.path.join(OUTDIR, f"feature_{j:05d}")
        os.makedirs(feat_dir, exist_ok=True)
        out_png = os.path.join(feat_dir, f"feat{j:05d}_rank{r:02d}.png")

        # Skip if already exists (enables resuming)
        if os.path.exists(out_png):
            print(f"[skip] {out_png}")
            skip_count += 1
            continue

        # Compute feature vector (cached)
        if j not in fv_cache:
            try:
                fv_cache[j] = emov5.feature_vector(emov5.sae, j)
            except Exception as e:
                print(f"[feature_vector error] j={j} -> {e}")
                continue

        fv = fv_cache[j]

        # Download image
        try:
            resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "saliency-batch/1.0"})
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[download fail] rank={r} j={j} url={url} -> {e}")
            download_fail_count += 1
            continue

        # Compute saliency
        try:
            sal = saliency_from_direction(img, fv, smoothgrad=SMOOTHGRAD, n=N_SAMPLES, sigma=SIGMA)
        except Exception as e:
            print(f"[saliency error] rank={r} j={j} -> {e}")
            saliency_error_count += 1
            continue

        # Overlay smoothed heatmap on the image
        try:
            # Build display RGB from preprocessed image (invert normalization)
            mean = torch.tensor(getattr(emov5, "image_mean", [0.48145466, 0.4578275, 0.40821073]))
            std  = torch.tensor(getattr(emov5, "image_std",  [0.26862954, 0.26130258, 0.27577711]))

            tmp = emov5.preprocess(img).cpu()
            tmp = (tmp * std[:, None, None] + mean[:, None, None]).clamp(0, 1)
            rgb = (tmp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Hp, Wp = rgb.shape[:2]

            # Resize saliency to match display grid
            if sal.shape != (Hp, Wp):
                sal_img = Image.fromarray((np.clip(sal, 0, None) * 255).astype(np.uint8))
                sal_rs = np.array(sal_img.resize((Wp, Hp), resample=Image.LANCZOS)) / 255.0
            else:
                sal_rs = sal.astype(np.float32, copy=False)
            sal_rs = np.clip(sal_rs, 0.0, None)

            # Smooth in display space
            sal_s = gaussian_filter(sal_rs, sigma=BLUR_SIGMA).astype(np.float32)

            # Robust normalization with soft masking + gamma
            low  = np.percentile(sal_s, LOW_Q)
            high = max(np.percentile(sal_s, HIGH_Q), low + 1e-6)
            sal_n = (sal_s - low) / (high - low)
            sal_n = np.clip(sal_n, 0.0, 1.0)
            if GAMMA != 1.0:
                sal_n = np.power(sal_n, GAMMA)

            # Alpha ramps with saliency
            alpha = (sal_n ** 1.0) * ALPHA_MAX

            # Plot overlay
            fig_w = 8.0
            fig_h = fig_w * (Hp / Wp)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
            ax.imshow(rgb)
            ax.imshow(sal_n, cmap=CMAP, alpha=alpha, interpolation="bilinear")

            # Optional contour lines
            if CONTOUR_QS:
                levels = [np.percentile(sal_s, q) for q in CONTOUR_QS]
                levels_n = [(lv - low) / (high - low) for lv in levels]
                levels_n = [lv for lv in levels_n if 0.0 < lv < 1.0]
                if levels_n:
                    ax.contour(sal_n, levels=levels_n, linewidths=0.7,
                              colors="white", alpha=0.7, antialiased=True)

            ax.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_png.replace(".png", "_overlay.png"),
                       bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close(fig)

            # Save heat-only visualization
            if SAVE_HEAT_ONLY:
                fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
                ax2.imshow(sal_n, cmap=CMAP, interpolation="bilinear")
                ax2.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=200)
                plt.close(fig2)

            print(f"[saved] {out_png.replace('.png', '_overlay.png')}")
            if SAVE_HEAT_ONLY:
                print(f"[saved] {out_png}")
            saved_count += 1

        except Exception as e:
            print(f"[save fail] {out_png} -> {e}")

    # Summary
    total = len(df)
    print("\n==== Summary ====")
    print(f"total rows: {total}")
    print(f"saved: {saved_count}")
    print(f"skipped (already existed): {skip_count}")
    print(f"download failures: {download_fail_count}")
    print(f"saliency errors: {saliency_error_count}")

if __name__ == "__main__":
    main()
