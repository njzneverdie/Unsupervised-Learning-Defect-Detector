"""進入點①:掃正常影像 → 建記憶庫 → 存檔。

用法:
    python patchcore/build_memory.py --data <影像資料夾> --out memory.pt
    python patchcore/build_memory.py --data <資料夾> --limit 200   # 只用前 200 張
"""
import argparse

from config import Config
from data import list_images, load_image
from backbone import FeatureExtractor
from memory_bank import save_memory
from pipeline import build_memory_from_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="正常(defect-free)影像資料夾")
    ap.add_argument("--out", default="memory.pt", help="記憶庫輸出路徑")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 張(0=全部)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = Config(device=args.device)
    device = cfg.resolve_device()
    print(f"[device] {device}")

    paths = list_images(args.data)
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"找不到影像:{args.data}")
    print(f"[data] 使用 {len(paths)} 張影像建庫")

    extractor = FeatureExtractor(cfg, device)
    images = [load_image(p, cfg) for p in paths]
    memory = build_memory_from_images(images, extractor, cfg, device)
    save_memory(memory, cfg, args.out)


if __name__ == "__main__":
    main()
