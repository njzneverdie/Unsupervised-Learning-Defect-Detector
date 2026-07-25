"""影像 I/O、tiling、送進 backbone 前的張量前處理,以及合成缺陷注入(評估用)。"""
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
import torch

from config import Config

IMG_EXT = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]

# ImageNet 預訓練模型的正規化常數(必用,否則特徵分布不對)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_images(data_root) -> List[Path]:
    root = Path(data_root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def load_image(path, cfg: Config) -> np.ndarray:
    """讀灰階 → resize 成 input_size 正方形 → normalize 到 [0,1] 的 float32。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def make_tiles(img: np.ndarray, cfg: Config) -> List[Dict]:
    """把 input_size 影像切成 grid 個 tile(含重疊)。

    回傳每個 tile 的 dict:tile(灰階 [0,1])、y0/x0(左上角)、h/w(邊長)。
    y0/x0 之後用來把 heatmap 拼回原圖(等同 notebook 的 meta 座標)。
    """
    t = cfg.tile_size
    tiles = []
    for (y0, x0) in cfg.tile_positions():
        tiles.append({"tile": img[y0:y0 + t, x0:x0 + t], "y0": y0, "x0": x0, "h": t, "w": t})
    return tiles


def tiles_to_tensor(tiles: List[Dict], device) -> torch.Tensor:
    """一批 tile → (B, 3, t, t) 已正規化張量(灰階複製成 3 通道)。"""
    arr = np.stack([d["tile"] for d in tiles])            # (B, t, t)
    x = torch.from_numpy(arr).unsqueeze(1)                # (B, 1, t, t)
    x = x.repeat(1, 3, 1, 1)                              # (B, 3, t, t)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def inject_defects(img: np.ndarray, cfg: Config, rng=None):
    """在正常影像上隨機貼橢圓缺陷(暗孔洞 / 亮夾雜物),回傳 (被汙染影像, GT mask)。

    因為資料集沒有缺陷標註,用這個產生已知 ground-truth 來做量化評估。
    取得真實標註後,把這個函式換成讀真實 mask 即可,其餘程式不動。
    """
    rng = rng or np.random.default_rng(cfg.random_state)
    out = img.copy()
    mask = np.zeros(img.shape, dtype=np.uint8)
    H, W = img.shape
    for _ in range(cfg.synth_defects_per_image):
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        ax, ay = int(rng.integers(cfg.synth_min, cfg.synth_max)), int(rng.integers(cfg.synth_min, cfg.synth_max))
        angle = int(rng.integers(0, 180))
        val = 0.0 if rng.random() < 0.5 else 1.0          # 暗孔洞 / 亮夾雜物
        blob = np.zeros(img.shape, dtype=np.uint8)
        cv2.ellipse(blob, (cx, cy), (ax, ay), angle, 0, 360, 1, -1)
        noisy = np.clip(val + rng.normal(0, 0.05, img.shape).astype(np.float32), 0, 1)
        out[blob == 1] = noisy[blob == 1]
        mask[blob == 1] = 1
    return out, mask.astype(bool)
