# PatchCore Defect Detector（`.py` 版,可在 VS Code 直接跑）

用「ImageNet 預訓練特徵 + 記憶庫 + 最近鄰」做 SEM 晶粒影像的無監督缺陷偵測。
不用訓練 backbone,適合筆電(含 Apple Silicon MPS)。

## 原理一句話
把所有**正常**影像的局部特徵記進一個記憶庫;測試時每個局部去問「離最近的正常樣本多遠」,越遠 = 越異常。詳見 repo 根目錄 `PROJECT_REVIEW.md` 的討論。

## 安裝

```bash
cd patchcore
python -m venv .venv && source .venv/bin/activate   # 建議用虛擬環境
pip install -r requirements.txt
```

> 裝置自動偵測:Apple Silicon(M 系列)用 **MPS**、Colab/有 NVIDIA 卡用 **CUDA**、否則 CPU。

### 在 Colab 跑
打開 [`run_on_colab.ipynb`](run_on_colab.ipynb)(GitHub 上點 "Open in Colab" 或上傳到 Colab),
先在 `Runtime → Change runtime type → GPU` 開 GPU,再由上往下逐格執行:會自動 clone repo、
下載資料集、跑建庫/評估/偵測三步。

## 三步流程

```bash
# ① 用正常影像建記憶庫(先用少量張數跑通,例如 --limit 100)
python build_memory.py --data /path/to/grainsize_dataset --out memory.pt --limit 100

# ② 對單張影像出 heatmap + 影像級分數(--quantile 0.95 會多畫二值 mask)
python detect.py --image /path/to/test.jpg --memory memory.pt --out result.png --quantile 0.95

# ③ 量化評估(合成缺陷 → 像素/影像級 ROC-AUC、IoU)
python evaluate.py --data /path/to/grainsize_dataset --memory memory.pt
```

> 從 repo 根目錄跑的話,把 `build_memory.py` 換成 `patchcore/build_memory.py` 等。

## 模組結構

| 檔案 | 職責 |
|------|------|
| `config.py` | 集中所有參數 + 裝置解析 + tile 座標 |
| `data.py` | 讀圖 / tiling / 正規化張量 / 合成缺陷 |
| `backbone.py` | 凍結 WRN50 + forward hook 抓 layer2/3 |
| `patch_features.py` | 3×3 鄰域聚合 + 多層 concat → patch 特徵 |
| `memory_bank.py` | greedy k-center coreset + 存讀 |
| `scorer.py` | 最近鄰打分 + 影像級 reweighting |
| `postprocess.py` | 上採樣 + 拼回 + 高斯模糊 + 視覺化 |
| `pipeline.py` | 串起建庫 / 偵測的完整流程 |
| `build_memory.py` / `detect.py` / `evaluate.py` | 三個 CLI 進入點 |

## 調參重點(都在 `config.py`)
- `tile_size` / `input_size`：小缺陷偵測解析度。
- `presample_max` / `coreset_ratio`：記憶庫大小 ↔ 速度/精度。
- `gaussian_sigma`：heatmap 平滑度。
- `global_quantile`：二值化門檻鬆緊。

## 注意事項
- **記憶庫只能用正常(defect-free)影像**。若混入缺陷,同類缺陷會被當正常而漏檢。
  第一版可假設整批≈正常;之後可用第一版模型剔除高分可疑圖,再重建更乾淨的庫。
- 合成缺陷評估屬**相對驗證與調參**,分布未必等同真實缺陷;取得真實標註後把
  `data.inject_defects` 換成讀真實 mask 即可,其餘不動。
- 本套件為初版骨架,尚未在真實資料上實跑驗證(見 repo `CHANGELOG.md`)。
