# CHANGELOG — 整理版變更紀錄

**日期**：2026-07-24
**原始檔**：`unsupervised_learning_defect_detector.ipynb`（52 cells,保留不動）
**整理版**：`unsupervised_learning_defect_detector_cleaned.ipynb`（36 cells,新檔）

整理目標：**清理重複 cell、統一成單一套分群、將設定抽成 CONFIG**。原始 notebook 未被修改,可隨時對照。

---

## 摘要

| 項目 | 原始 | 整理版 |
|------|------|--------|
| Cell 數 | 52 | 48（36 整理 + 12 Part C 新增） |
| `preprocess_image` 定義次數 | 3（預設值不一致） | 1 |
| `extract_patches` 定義次數 | 3 | 1 |
| `image_paths` 重定義次數 | 4 | 1 |
| PCA/K-means 套數 | 2（矛盾、變數污染） | 1 |
| 參數散落 | 各 cell 硬編碼 | 集中於 CONFIG cell |
| 隨機種子 | 部分 | 統一 `set_seed` |

---

## 1. 抽出 CONFIG（新增）

新增「0. 全域設定」區塊,把原本散落各處的硬編碼參數集中:

- `RESIZE_TO=(512,512)`、`PATCH_SIZE=64`、`STRIDE=32`、`MAX_PATCHES_TOTAL=80000`
- `PCA_MAX_SAMPLES=30000`、`PCA_COMPONENTS=50`、`N_CLUSTERS=5`
- `AE_MAX_TRAIN=20000`、`AE_EPOCHS=15`、`AE_BATCH=256`、`AE_LR=1e-3`
- `ANOMALY_QUANTILE=0.95`、`RANDOM_STATE=42`

**對應原始**：這些值原本分別寫死在 cell 23（`MAX_PATCHES_TOTAL`）、cell 25（`MAX_SAMPLES_PCA`）、
cell 29/31（`K=5`、`n_components=50`）、cell 41（`MAX_TRAIN`）、cell 44（`epochs=15`、`batch_size=256`）、
cell 51（`0.95`）等。

## 2. 統一隨機種子（新增）

新增 `set_seed` cell,一次設定 `random` / `numpy` / `PYTHONHASHSEED` / `tensorflow`。
**對應 Review 第 8 點**：原本 `np.random.choice`（原 cell 33/37）與 TF 權重初始化沒有種子,結果不可完全重現。

## 3. 合併重複的環境/下載 cell

- 原 cell 5、8、10（`pip install` / `kaggle download` / `unzip`）→ 合併為 **1 個 cell**。

## 4. 合併重複的 import 與函式定義（重點）

- 原 cell 11、19、20、21、22、39 各自 `import cv2/numpy` 並重新掃描 `image_paths` → 統一為 **單一** 「共用函式」區塊。
- `preprocess_image` 原本定義 **3 次**,且 `resize_to` 預設值不一致（cell 15=`(512,512)`、cell 21=`None`、cell 22=`(512,512)`）→ 統一為 **1 個**,預設值取自 `CONFIG.RESIZE_TO`。
  **對應 Review 第 9 點**（「同名函式不同行為」的溫床）。
- `extract_patches` 原本定義 **3 次** → 統一為 **1 個**。

## 5. 統一成單一套 PCA + K-means（重點,修正邏輯 bug）

**這是最重要的修正。** 原始 notebook 有兩套分群:

- 原 cell 27–29：`StandardScaler → PCA(50) → KMeans`,結果變數 `clusters`。
- 原 cell 31：**沒有 StandardScaler** 的 `PCA(50) → KMeans`,結果變數 `cluster_labels`。

而下游混用兩者:代表 patch（原 cell 33/35）用 `cluster_labels`,cluster map（原 cell 39）用 `clusters`
→ **兩張圖的群編號其實對不上,會誤導。**

**整理版做法**：
- 只保留 **有 StandardScaler 的版本**（較正確）。
- 唯一的分群結果一律命名 `clusters`;刪除 `cluster_labels` 與重複的 PCA/KMeans cell。
- 代表 patch（`show_cluster_examples`）與 cluster map 全部改用同一套 `clusters`。
- 另外印出 `pca_50.explained_variance_ratio_.sum()`,讓 PCA 維度選擇有依據（**對應 Review 第 7 點**）。

**對應 Review 第 2 點。**

## 6. 消除變數覆寫 `patches`

原始 notebook 中 `patches` 一名多用:迴圈內單張圖的 patch、迴圈外的全資料集、cell 25 的切片,語意混亂。

**整理版命名規則**：
- 全資料集 → `dataset_patches`
- 迴圈內單張圖 → `img_patches`
- 分群抽樣 → `X`

**對應 Review 第 1 點。**

## 7. `compute_recon_error_for_image` 改為自足函式

原始函式偷用全域 `image_paths`（該變數被重定義 4 次,行為不穩定）。
整理版把 `image_paths` 加為 **參數**,並改用共用的 `preprocess_image`（取代函式內重複的 `cv2.imread + resize + /255`）。
**對應 Review 第 4 點。**

## 8. Demo 影像 ID 改為動態挑選(避免截斷造成的例外)

- Part A cluster map:原本寫死 `IMG_ID=0`。整理版改成 `valid_ids_A = np.unique(meta[:N,0]); IMG_ID = valid_ids_A[0]`。
- Part B heatmap:原本寫死 `IMG_ID=50`,但 patch 在 80k 上限被截斷,`img_id=50` 不一定存在。
  整理版改成從 `np.unique(meta[:,0])` 取中間一張。
  **對應 Review 第 5 點。**

## 9. 整理開發過程留言

移除/改寫像 `# 降成 float16 省記憶體 爆掉了`、`#60000改20000(加快速度)`、`# 分5群結果最適合`
等開發過程註解,改為說明性註解（**對應 Review 第 12 點**）。float16 的用途改以正式註解說明。

---

## 10. 新增 Part C：跨圖全域門檻 & 量化評估（第二次補強）

**日期**：2026-07-24（同日,整理後追加）。在整理版 notebook 末尾新增「Part C」章節,共 12 個 cell,
並在 CONFIG cell 追加 Part C 專用參數。

### 10.1 CONFIG 追加參數
`GLOBAL_BASELINE_IMAGES`、`GLOBAL_QUANTILE`、`IMAGE_SCORE_QUANTILE`、`IMAGE_FLAG_QUANTILE`、
`EVAL_IMAGES`、`SYNTH_DEFECTS_PER_IMAGE`、`SYNTH_DEFECT_MIN/MAX`。

### 10.2 跨圖全域門檻（對應 Review 第 6 點,已實作）
- 新增 `error_map_from_image(img, model)`:對**任意**影像(含合成影像)切 patch → 重建 → error map + per-patch 誤差。
- 取 `GLOBAL_BASELINE_IMAGES` 張影像,匯集所有 per-patch 誤差 → `GLOBAL_THRESHOLD`(跨圖共用的像素級門檻)。
- 每張圖給一個異常分數(其 patch 誤差的 `IMAGE_SCORE_QUANTILE` 分位數)→ 排序 + `IMAGE_LEVEL_THRESHOLD` 判定「正常 / 可疑」。
- 現在可跨圖比較「哪張更異常」,並用同一門檻畫 defect mask。

### 10.3 量化評估 ROC-AUC / IoU（對應 Review 第 11 點,已實作）
- 因資料集無標註,採**合成缺陷注入** `inject_defects(img)`:在正常圖上貼隨機橢圓(暗孔洞/亮夾雜物)並回傳 GT mask。
- 指標:
  - **像素級 ROC-AUC**:誤差值當分數 vs GT mask。
  - **像素級 IoU**:`error_map >= GLOBAL_THRESHOLD` 的預測區 vs GT mask。
  - **影像級 ROC-AUC**:正常圖 vs 注入缺陷圖的異常分數。
- 已附判讀說明與限制(合成缺陷分布 ≠ 真實缺陷;取得真實標註後把 `inject_defects` 換掉即可,其餘不變)。

---

## 仍未處理（留待後續）

- **Review 第 10 點**:float16/float32 精度取捨,維持原策略（記憶體考量）。
- 真實標註資料上的評估:目前用合成缺陷驗證,屬相對驗證;需少量真實 mask 才能得到真實表現數據。

---

## 檔案清單

| 檔案 | 說明 |
|------|------|
| `unsupervised_learning_defect_detector.ipynb` | 原始 notebook（未改動） |
| `unsupervised_learning_defect_detector_cleaned.ipynb` | **整理版（本次產出）** |
| `PROJECT_REVIEW.md` | 專案說明 + code review |
| `NOTEBOOK.md` | 原始 notebook 的 markdown 匯出 |
| `CHANGELOG.md` | 本變更紀錄 |
