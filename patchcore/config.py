"""集中所有可調參數。改這裡就好,不要把數字散落在各檔案。"""
from dataclasses import dataclass, field
from typing import Tuple
import torch


@dataclass
class Config:
    # ---- 影像 / tiling ----
    input_size: int = 512            # 每張圖先 resize 成正方形這個尺寸
    tile_size: int = 288            # 2x2 tile,每塊的邊長(含重疊)
    grid: Tuple[int, int] = (2, 2)  # tile 網格(列, 行)

    # ---- backbone / 特徵 ----
    neighborhood: int = 3           # 局部聚合的鄰域大小(3x3 平均)
    # WRN50 的 layer2=512ch、layer3=1024ch,concat 後 = 1536 維

    # ---- 記憶庫 / coreset ----
    presample_max: int = 40000      # coreset 前先隨機下採到多少條(控制計算量)
    coreset_ratio: float = 0.1      # coreset 保留比例
    proj_dim: int = 128             # 隨機投影維度(加速 coreset 選點)

    # ---- 打分 ----
    score_chunk: int = 1024         # 最近鄰查詢的分塊大小(控記憶體)
    reweight_b: int = 5             # 影像級 reweighting 用的鄰居數

    # ---- 後處理 ----
    gaussian_sigma: float = 4.0     # heatmap 高斯模糊
    global_quantile: float = 0.90   # 由正常圖誤差分布定的全域門檻分位數

    # ---- 評估(合成缺陷) ----
    eval_images: int = 20
    synth_defects_per_image: int = 3
    synth_min: int = 8              # 合成缺陷半徑下限(px)
    synth_max: int = 22             # 合成缺陷半徑上限(px)

    # ---- 其他 ----
    random_state: int = 42
    device: str = "auto"            # auto / mps / cuda / cpu

    def resolve_device(self) -> torch.device:
        """auto:有 Apple GPU(MPS)就用,其次 CUDA,否則 CPU。"""
        if self.device != "auto":
            return torch.device(self.device)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def tile_positions(self):
        """回傳 2x2(或其他 grid)每個 tile 的左上角座標 [(y0, x0), ...]。

        n 個 tile 沿一軸鋪滿長度 L,tile 邊長 t:
        - n==1 → [0]
        - 否則平均分布,自動產生重疊 = t - (L-t)/(n-1)
        """
        L, t = self.input_size, self.tile_size
        rows, cols = self.grid

        def axis(n):
            if n == 1:
                return [0]
            return [round(i * (L - t) / (n - 1)) for i in range(n)]

        ys, xs = axis(rows), axis(cols)
        return [(y, x) for y in ys for x in xs]
