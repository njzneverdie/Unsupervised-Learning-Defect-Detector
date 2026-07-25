"""最近鄰打分:每條測試特徵到記憶庫的最近距離 = 異常分數;影像級分數含 reweighting。"""
import torch

from config import Config


@torch.no_grad()
def score_patches(features: torch.Tensor, memory: torch.Tensor, cfg: Config):
    """features (P, D)、memory (M, D) → (min_dist (P,), argmin_idx (P,))。

    分塊計算距離矩陣以控制記憶體;對每條特徵取到記憶庫的最小距離(1-NN)。
    """
    mins, args = [], []
    for chunk in features.split(cfg.score_chunk, dim=0):
        d = torch.cdist(chunk, memory)                  # (c, M)
        mn, ai = torch.min(d, dim=1)                    # 1-NN 距離與索引
        mins.append(mn)
        args.append(ai)
    return torch.cat(mins), torch.cat(args)


@torch.no_grad()
def image_level_score(min_dist, argmin_idx, features, memory, cfg: Config) -> float:
    """影像級分數 = 最異常 patch 的距離 s*,再乘上 PatchCore 的 reweighting 權重 w。

    w = 1 - exp(||t*-m*||) / Σ_{m in N_b(m*)} exp(||t*-m||)
    直覺:若 m* 位於記憶庫稀疏區(罕見正常),維持較高分;位於密集區則略降權。
    """
    top = int(torch.argmax(min_dist).item())
    s_star = min_dist[top]
    t_star = features[top:top + 1]                      # (1, D)
    m_star_idx = int(argmin_idx[top].item())
    m_star = memory[m_star_idx:m_star_idx + 1]          # (1, D)

    # m* 在記憶庫裡的 b 個最近鄰(含自己)
    d_mem = torch.cdist(m_star, memory).squeeze(0)      # (M,)
    b = min(cfg.reweight_b + 1, memory.shape[0])
    nn_idx = torch.topk(d_mem, b, largest=False).indices
    neighbors = memory[nn_idx]                          # (b, D)

    d_t = torch.cdist(t_star, neighbors).squeeze(0)     # t* 到這些鄰居的距離 (b,)
    ratio = torch.softmax(d_t, dim=0)[0]                # 對應 m*(最近)那一項;softmax 內含數值穩定
    w = 1.0 - ratio
    return float((w * s_star).item())
