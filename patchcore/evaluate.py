"""進入點③:用合成缺陷做量化評估 → 像素級 ROC-AUC / IoU、影像級 ROC-AUC。

與 notebook Part C 用同一套指標與 inject_defects,方便和 AE 公平對比。

用法:
    python patchcore/evaluate.py --data <影像資料夾> --memory memory.pt
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from config import Config
from data import list_images, load_image, inject_defects
from backbone import FeatureExtractor
from memory_bank import load_memory
from pipeline import detect_image


def iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / union if union > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="評估用影像資料夾")
    ap.add_argument("--memory", default="memory.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = Config(device=args.device)
    device = cfg.resolve_device()
    print(f"[device] {device}")

    extractor = FeatureExtractor(cfg, device)
    memory, _ = load_memory(args.memory, device)

    paths = list_images(args.data)
    rng = np.random.default_rng(cfg.random_state)
    eval_ids = rng.choice(len(paths), size=min(cfg.eval_images, len(paths)), replace=False)

    # 第一輪:算乾淨圖的 heatmap 與分數,並蒐集用來定全域門檻
    clean_heats, clean_scores, dirties = [], [], []
    for idx in eval_ids:
        clean = load_image(paths[int(idx)], cfg)
        dirty, gt = inject_defects(clean, cfg, rng=rng)
        hc, sc = detect_image(clean, extractor, memory, cfg)
        clean_heats.append(hc)
        clean_scores.append(sc)
        dirties.append((dirty, gt))

    global_thr = float(np.quantile(np.concatenate([h.ravel() for h in clean_heats]),
                                   cfg.global_quantile))

    # 第二輪:算注入缺陷圖,對照 GT
    pixel_aucs, ious, defect_scores = [], [], []
    for (dirty, gt) in dirties:
        hd, sd = detect_image(dirty, extractor, memory, cfg)
        if gt.any() and (~gt).any():
            pixel_aucs.append(roc_auc_score(gt.ravel(), hd.ravel()))
            ious.append(iou(hd >= global_thr, gt))
        defect_scores.append(sd)

    # 影像級 AUC:正常(0) vs 注入缺陷(1)
    y_true = np.r_[np.zeros(len(clean_scores)), np.ones(len(defect_scores))]
    y_score = np.r_[clean_scores, defect_scores]
    image_auc = roc_auc_score(y_true, y_score)

    print("\n===== PatchCore 評估結果 =====")
    print(f"評估影像數      : {len(eval_ids)}")
    print(f"全域門檻(q={cfg.global_quantile}): {global_thr:.6f}")
    print(f"像素級 ROC-AUC  : {np.mean(pixel_aucs):.3f}  (±{np.std(pixel_aucs):.3f})")
    print(f"像素級 IoU@門檻 : {np.nanmean(ious):.3f}")
    print(f"影像級 ROC-AUC  : {image_auc:.3f}")


if __name__ == "__main__":
    main()
