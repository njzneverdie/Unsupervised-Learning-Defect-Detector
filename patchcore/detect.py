"""進入點②:載記憶庫 → 對一張新影像出 heatmap 與影像級分數。

用法:
    python patchcore/detect.py --image test.jpg --memory memory.pt --out result.png
"""
import argparse

from config import Config
from data import load_image
from backbone import FeatureExtractor
from memory_bank import load_memory
from pipeline import detect_image
from postprocess import save_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="要檢測的影像")
    ap.add_argument("--memory", default="memory.pt", help="記憶庫路徑")
    ap.add_argument("--out", default="result.png", help="輸出視覺化 png")
    ap.add_argument("--quantile", type=float, default=None,
                    help="二值化門檻分位數(0~1);不給則不畫 mask")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = Config(device=args.device)
    device = cfg.resolve_device()
    print(f"[device] {device}")

    extractor = FeatureExtractor(cfg, device)
    memory, _ = load_memory(args.memory, device)

    img = load_image(args.image, cfg)
    heatmap, score = detect_image(img, extractor, memory, cfg)
    print(f"[score] 影像級異常分數 = {score:.6f}")

    thr = None
    if args.quantile is not None:
        import numpy as np
        thr = float(np.quantile(heatmap, args.quantile))
    save_overlay(img, heatmap, args.out, threshold=thr)


if __name__ == "__main__":
    main()
