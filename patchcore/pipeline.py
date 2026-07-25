"""把「一張圖 → heatmap + 影像級分數」的完整推論串起來,給 detect / evaluate 共用。"""
import numpy as np
import torch

from config import Config
from data import make_tiles, tiles_to_tensor
from backbone import FeatureExtractor
from patch_features import aggregate, grid_to_patch_features
from memory_bank import build_memory
from scorer import score_patches, image_level_score
from postprocess import stitch_heatmap, smooth


@torch.no_grad()
def image_to_features(img: np.ndarray, extractor: FeatureExtractor, cfg: Config):
    """一張 input_size 影像 → (patch 特徵 (P,D), (B,gh,gw), tiles)。"""
    tiles = make_tiles(img, cfg)
    batch = tiles_to_tensor(tiles, extractor.device)
    l2, l3 = extractor.extract(batch)
    grid = aggregate(l2, l3, cfg)
    feats, shape = grid_to_patch_features(grid)
    return feats, shape, tiles


@torch.no_grad()
def build_memory_from_images(images, extractor: FeatureExtractor, cfg: Config, device):
    """掃一批正常影像 → 蒐集所有 patch 特徵 → coreset → 記憶庫。images 為 [np.ndarray]。"""
    collected = []
    for i, img in enumerate(images):
        feats, _, _ = image_to_features(img, extractor, cfg)
        collected.append(feats.cpu())                    # 搬回 CPU 避免 GPU 記憶體累積
        if (i + 1) % 20 == 0:
            print(f"[memory] 已處理 {i + 1}/{len(images)} 張")
    all_feats = torch.cat(collected, dim=0)
    return build_memory(all_feats, cfg, device)


@torch.no_grad()
def detect_image(img: np.ndarray, extractor: FeatureExtractor, memory: torch.Tensor, cfg: Config):
    """一張影像 → (heatmap input_size×input_size, 影像級分數)。"""
    feats, (B, gh, gw), tiles = image_to_features(img, extractor, cfg)
    min_dist, argmin_idx = score_patches(feats, memory, cfg)
    img_score = image_level_score(min_dist, argmin_idx, feats, memory, cfg)

    tile_scores = min_dist.reshape(B, gh, gw)
    heatmap = stitch_heatmap(tile_scores, tiles, cfg)
    heatmap = smooth(heatmap, cfg)
    return heatmap, img_score
