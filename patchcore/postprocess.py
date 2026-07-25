"""把 tile 的粗網格分數 → 上採樣 → 拼回原圖 → 高斯模糊 → 平滑 heatmap。"""
from typing import List, Dict
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import Config


def stitch_heatmap(tile_scores: torch.Tensor, tiles: List[Dict], cfg: Config) -> np.ndarray:
    """tile_scores (B, gh, gw) → input_size×input_size 的 heatmap。

    每個 tile 的分數上採樣回 tile 尺寸,依 (y0,x0) 貼回畫布;重疊區取平均(同 Part B)。
    """
    L = cfg.input_size
    error_map = np.zeros((L, L), dtype=np.float32)
    count_map = np.zeros((L, L), dtype=np.float32)

    scores = tile_scores.detach().cpu().float()
    for i, d in enumerate(tiles):
        s = scores[i][None, None]                              # (1,1,gh,gw)
        up = F.interpolate(s, size=(d["h"], d["w"]), mode="bilinear", align_corners=False)
        up = up.squeeze().numpy()
        y0, x0, h, w = d["y0"], d["x0"], d["h"], d["w"]
        error_map[y0:y0 + h, x0:x0 + w] += up
        count_map[y0:y0 + h, x0:x0 + w] += 1.0

    count_map[count_map == 0] = 1.0
    error_map /= count_map
    return error_map


def smooth(heatmap: np.ndarray, cfg: Config) -> np.ndarray:
    """高斯模糊,抹掉粗網格的塊狀假影。"""
    if cfg.gaussian_sigma <= 0:
        return heatmap
    return cv2.GaussianBlur(heatmap, (0, 0), sigmaX=cfg.gaussian_sigma)


def save_overlay(orig: np.ndarray, heatmap: np.ndarray, out_path, threshold=None):
    """存一張三聯圖:原圖 / error map / overlay(+ 可選 defect mask)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-8)

    n = 3 if threshold is None else 4
    plt.figure(figsize=(4 * n, 4))
    plt.subplot(1, n, 1); plt.title("Original"); plt.imshow(orig, cmap="gray"); plt.axis("off")
    plt.subplot(1, n, 2); plt.title("Error map"); plt.imshow(hm, cmap="inferno"); plt.axis("off")
    plt.subplot(1, n, 3); plt.title("Overlay"); plt.imshow(orig, cmap="gray")
    plt.imshow(hm, cmap="inferno", alpha=0.5); plt.axis("off")
    if threshold is not None:
        plt.subplot(1, n, 4); plt.title("Defect mask")
        plt.imshow(heatmap >= threshold, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] 已存 {out_path}")
