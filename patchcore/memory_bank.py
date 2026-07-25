"""記憶庫:蒐集正常特徵 + greedy k-center coreset 子抽樣 + 存/讀。"""
import torch

from config import Config


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, m: int, cfg: Config, device) -> torch.Tensor:
    """貪婪 k-center coreset:從 features 選出 m 條最能覆蓋整體幾何的向量。

    目標(minimax):min_C max_x min_{c in C} ||x - c||  → 每步補「目前覆蓋最差」的點。
    用隨機投影(Johnson–Lindenstrauss)在低維算距離加速選點,但最後回傳「原始維度」向量。
    """
    N, D = features.shape
    m = min(m, N)
    g = torch.Generator(device="cpu").manual_seed(cfg.random_state)

    # 隨機投影到 proj_dim(距離近似保持,選點更快)
    proj = torch.randn(D, cfg.proj_dim, generator=g) / (cfg.proj_dim ** 0.5)
    f = (features.to(device) @ proj.to(device))            # (N, proj_dim)

    selected = torch.empty(m, dtype=torch.long)
    min_dist = torch.full((N,), float("inf"), device=device)

    last = torch.randint(0, N, (1,), generator=g).item()   # 隨機起點
    for i in range(m):
        selected[i] = last
        d = torch.cdist(f[last:last + 1], f).squeeze(0)     # 新點到所有點的距離
        min_dist = torch.minimum(min_dist, d)               # 更新各點到已選集的最近距離
        min_dist[last] = -1.0                               # 不重選
        last = int(torch.argmax(min_dist).item())           # 選覆蓋最差的點

    return features[selected.cpu()]


@torch.no_grad()
def build_memory(all_features: torch.Tensor, cfg: Config, device) -> torch.Tensor:
    """把蒐集到的所有正常 patch 特徵 → (可選)隨機預抽 → coreset → 記憶庫。"""
    N = all_features.shape[0]
    g = torch.Generator(device="cpu").manual_seed(cfg.random_state)

    if N > cfg.presample_max:                               # 先隨機下採,控制 coreset 計算量
        idx = torch.randperm(N, generator=g)[:cfg.presample_max]
        all_features = all_features[idx]
        N = cfg.presample_max

    m = max(1, int(N * cfg.coreset_ratio))
    memory = greedy_coreset(all_features, m, cfg, device)
    print(f"[memory] 蒐集 {N} 條 → coreset 保留 {memory.shape[0]} 條(維度 {memory.shape[1]})")
    return memory


def save_memory(memory: torch.Tensor, cfg: Config, path):
    torch.save({"memory": memory.cpu(), "config": cfg.__dict__}, path)
    print(f"[memory] 已存到 {path}")


def load_memory(path, device):
    obj = torch.load(path, map_location="cpu")
    return obj["memory"].to(device), obj.get("config", {})
