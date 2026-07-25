"""把 layer2 / layer3 特徵圖 → 局部聚合 + 多層 concat → patch 特徵網格。"""
import torch
import torch.nn.functional as F

from config import Config


def aggregate(layer2: torch.Tensor, layer3: torch.Tensor, cfg: Config) -> torch.Tensor:
    """機制④⑤:
    1. 每個位置改成其 3x3 鄰域平均(局部感知 + 對微小位移 robust)。
    2. layer3 上採樣到 layer2 的解析度後,沿通道 concat。

    回傳 (B, 1536, gh, gw):gh=gw=t/8,每個位置一條 1536 維 patch 特徵。
    """
    k = cfg.neighborhood
    pad = k // 2
    l2 = F.avg_pool2d(layer2, kernel_size=k, stride=1, padding=pad)
    l3 = F.avg_pool2d(layer3, kernel_size=k, stride=1, padding=pad)

    # layer3 (t/16) → 上採樣到 layer2 (t/8)
    l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)

    return torch.cat([l2, l3], dim=1)          # 512 + 1024 = 1536


def grid_to_patch_features(grid: torch.Tensor):
    """(B, D, gh, gw) → 攤平成 (B*gh*gw, D) 的 patch 特徵,並回傳 (B, gh, gw) 供之後 reshape。"""
    B, D, gh, gw = grid.shape
    feats = grid.permute(0, 2, 3, 1).reshape(B * gh * gw, D).contiguous()
    return feats, (B, gh, gw)
