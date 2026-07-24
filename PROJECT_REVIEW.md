# Unsupervised Learning Defect Detector — 專案說明與 Code Review

本文件說明 `unsupervised_learning_defect_detector.ipynb` 這個專案的目標、流程、重要程式碼，並附上 code review。
完整 notebook 內容另存於 [`NOTEBOOK.md`](NOTEBOOK.md)。

---

## 1. 這個專案在做什麼？

**一句話**：在**沒有任何缺陷標註**的情況下，對 SEM（掃描式電子顯微鏡）晶粒微結構影像做「異常/缺陷偵測」與「微結構分群」。

- **輸入**：一張 SEM 顯微影像
- **輸出**：
  - 一張 **anomaly heatmap**，標出哪裡可能是孔洞、夾雜物、異常晶粒或加工缺陷
  - 或把整張圖判定為「正常微結構」還是「有可疑缺陷」

之所以用**無監督學習**，是因為缺陷資料通常沒有人工標註、且缺陷樣本稀少。專案用兩條互補的路線來解決：

| 路線 | 方法 | 用途 |
|------|------|------|
| **Part A** | PCA + K-means 分群 | 發現微結構的不同「型態」（晶界、平滑區、可疑區），並畫回原圖成 cluster map |
| **Part B**（主角） | CNN Autoencoder + 重建誤差 | 只學「正常」長什麼樣；重建誤差高的地方 → 可能是缺陷。輸出 heatmap 與二值化缺陷遮罩 |

**核心假設（Part B）**：Autoencoder 在大量（大多為正常）patch 上學會壓縮再還原。正常紋理能被還原得很好（誤差低）；沒學好的異常紋理還原誤差高，因此**重建誤差 = 異常分數**。

---

## 2. 整體資料流程（Pipeline）

```
SEM 影像 (任意尺寸)
   │  preprocess_image: 灰階 → resize 512×512 → /255 normalize
   ▼
512×512 float 影像
   │  extract_patches: 64×64 視窗、stride 32 滑動切塊
   ▼
patches (N, 64, 64, 1)  +  meta (N, 3) = [img_id, y, x]
   ├──────────────► Part A: reshape→4096維 → StandardScaler → PCA(50) → K-means(K=5)
   │                        → 代表 patch 視覺化 + 用 meta 貼回原圖成 cluster map
   │
   └──────────────► Part B: 訓練 Conv Autoencoder（只用正常 patch，無標籤）
                            → 逐 patch 算 MSE 重建誤差
                            → 用 meta 把誤差貼回像素座標、重疊區取平均 → error heatmap
                            → 取 95 百分位當門檻 → binary defect mask
```

`meta` 這個 `(N,3)` 陣列（`img_id, y, x`）是把 patch 級結果「攤回」影像座標的關鍵。

---

## 3. 重要程式碼介紹

### 3.1 前處理與切 patch（Part A/B 共用地基）

```python
def preprocess_image(path, resize_to=(512, 512)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def extract_patches(img, patch_size=64, stride=32):
    H, W = img.shape
    patches, coords = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append(img[y:y+patch_size, x:x+patch_size])
            coords.append((y, x))
    return np.stack(patches), np.array(coords)
```

- **為什麼切 patch**：缺陷偵測是「局部」問題。整張圖丟進模型會被大量正常區域稀釋掉小缺陷；切成 64×64 小塊能定位到具體位置。
- **stride 32 < patch 64**：相鄰 patch 有 50% 重疊，讓 heatmap 更平滑、避免格子邊界斷裂。重疊也是後面 heatmap 要「取平均」的原因。
- 每個 patch 都記下左上角座標 `(y, x)`，之後才能貼回原圖。

### 3.2 建立全資料集的 patch 庫 + meta（cell 23）

```python
PATCH_SIZE, STRIDE, MAX_PATCHES_TOTAL = 64, 32, 80000
all_patches, meta_info, total_patches = [], [], 0

for img_id, path in enumerate(image_paths):
    img = preprocess_image(path, resize_to=(512, 512))
    patches, coords = extract_patches(img, PATCH_SIZE, STRIDE)
    # ...達到上限就截斷...
    patches = patches.astype(np.float16)          # 省記憶體
    all_patches.append(patches)
    img_ids = np.full((coords.shape[0], 1), img_id, dtype=np.int32)
    meta_info.append(np.hstack([img_ids, coords]))  # [img_id, y, x]
    total_patches += patches.shape[0]

patches = np.concatenate(all_patches)[..., np.newaxis]  # (N, 64, 64, 1)
meta    = np.concatenate(meta_info)                     # (N, 3)
```

- `MAX_PATCHES_TOTAL = 80000` 是**記憶體上限保護**（Colab RAM 有限）。
- 存成 `float16` 省一半記憶體；後面做 PCA / AE 推論時再轉回 `float32`。
- **`meta` 是全專案樞紐**：讓每個 patch 都能回答「我來自哪張圖、哪個座標」。

### 3.3 Part A：PCA + K-means（cell 25–31）

```python
patches_flat = patches.reshape(patches.shape[0], -1)      # (N, 4096)
X_scaled = StandardScaler().fit_transform(X_flat)         # 標準化
X_pca50  = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
clusters = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_pca50)
```

- 64×64 = **4096 維**太高，直接分群又慢又受雜訊影響。PCA 先壓到 50 維保留主要變異。
- 另做一個 `PCA(2)` 只為了 2D 散點視覺化。
- K=5 是人工挑的（README 說「分 5 群最適合」），把微結構分成幾種型態。

### 3.4 Part A：把 cluster 貼回原圖（cell 39）

```python
meta_used = meta[:N]                        # 只有前 N 個 patch 有 cluster 標籤
mask_img = (meta_used[:, 0] == IMG_ID)      # 挑出屬於某張圖的 patch
coords_img   = meta_used[mask_img][:, 1:3]
clusters_img = clusters[mask_img]

cluster_map = -1 * np.ones((H, W), dtype=int)
for (y, x), c in zip(coords_img, clusters_img):
    cluster_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = c       # 後蓋前
```

用 `meta` 反查座標，把每個 patch 的群編號塗回一張 512×512 的圖，得到「微結構分布圖」。

### 3.5 Part B：CNN Autoencoder（cell 43，主角）

```python
inp = layers.Input(shape=(64, 64, 1))
# Encoder：逐步下採樣、增加通道
x = layers.Conv2D(32,  (3,3), padding='same', activation='relu')(inp)
x = layers.MaxPool2D((2,2))(x)              # 32×32
x = layers.Conv2D(64,  (3,3), padding='same', activation='relu')(x)
x = layers.MaxPool2D((2,2))(x)              # 16×16
x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
encoded = layers.MaxPool2D((2,2))(x)        # 8×8×128 ← 瓶頸(latent)

# Decoder：逐步上採樣還原
x = layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu')(encoded) # 16×16
x = layers.Conv2DTranspose(64,  (3,3), strides=2, padding='same', activation='relu')(x)        # 32×32
x = layers.Conv2DTranspose(32,  (3,3), strides=2, padding='same', activation='relu')(x)        # 64×64
decoded = layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')(x)                     # 還原到 [0,1]

autoencoder = models.Model(inp, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
```

- **對稱的 encoder–decoder**：把 64×64 壓成 8×8×128 的瓶頸再還原。瓶頸強迫模型只能記住「常見紋理」，記不住罕見缺陷。
- 最後 `sigmoid` 讓輸出落在 `[0,1]`，對齊輸入的正規化範圍；loss 用 `MSE`（即像素重建誤差）。
- 訓練用 `EarlyStopping` + `ReduceLROnPlateau`，`X_train` 同時當輸入和目標（自我重建）。

### 3.6 Part B：重建誤差 heatmap（cell 47，最關鍵的推論邏輯）

```python
def compute_recon_error_for_image(img_id, patches_all, meta_all, model, patch_size=64):
    mask = (meta_all[:, 0] == img_id)
    coords = meta_all[mask][:, 1:3]
    patches_img = patches_all[mask]
    recon = model.predict(patches_img, batch_size=256, verbose=0)
    errors = np.mean((patches_img - recon)**2, axis=(1,2,3))   # 每個 patch 一個 MSE

    orig = cv2.resize(cv2.imread(...)/255., (512, 512))
    error_map = np.zeros((H, W)); count_map = np.zeros((H, W))
    for (y, x), e in zip(coords, errors):
        error_map[y:y+patch_size, x:x+patch_size] += e         # 累加
        count_map[y:y+patch_size, x:x+patch_size] += 1.0       # 計數
    count_map[count_map == 0] = 1.0
    error_map /= count_map                                     # 重疊區取平均
    return orig, error_map
```

這是 Part B 的心臟：
1. 對該圖的每個 patch 算 MSE 重建誤差（一個純量）。
2. 把誤差「塗」回像素座標；因為 stride 有重疊，用 `count_map` 對重疊區**取平均**，避免格線假影。
3. 得到與原圖同尺寸的 `error_map`（= 異常熱圖）。

### 3.7 Part B：二值化缺陷遮罩（cell 51）

```python
threshold = np.quantile(error_map, 0.95)   # 取誤差前 5% 當 anomaly
defect_mask = (error_map >= threshold)
```

用**分位數**當門檻（相對閾值），把誤差最高的 5% 區域標為缺陷，得到二值遮罩。

---

## 4. Code Review

整體評語：**流程正確、概念清晰、對記憶體/速度有實務考量**，適合作為教學/研究原型。但作為「工程專案」有不少可維護性、正確性與嚴謹度的問題。以下依嚴重度排序。

### 🔴 需要修正（正確性 / 邏輯問題）

1. **`patches` 這個變數名被重複覆寫，語意混亂且易出錯**
   - cell 23 迴圈內的區域變數叫 `patches`（單張圖的 patch），迴圈外又把「全資料集」也命名為 `patches`。
   - cell 25 又出現 `X = patches[:N]`，cell 31 再對整包 `patches` 做 PCA。同名不同物，讀者很難追蹤形狀。
   - **建議**：全域用 `all_patches` / `dataset_patches`，單張用 `img_patches`，避免覆寫。

2. **Part A 有兩套彼此矛盾的 PCA/K-means，變數還互相污染**
   - cell 27–29 用 `StandardScaler → PCA(50) → KMeans`，結果變數叫 `clusters`、資料叫 `X`/`X_pca2`。
   - cell 31 又做一次 **沒有 StandardScaler** 的 `PCA(50) → KMeans`，結果叫 `cluster_labels`。
   - 後面 cell 33/35 用 `cluster_labels`，cell 37/39 又用 `clusters`——**兩套分群結果混用**。cluster map（cell 39）用的是「有標準化」的 `clusters`，代表 patch（cell 33）用的是「沒標準化」的 `cluster_labels`，兩張圖的群編號並不對應，容易誤導。
   - **建議**：只保留一套（建議保留有 `StandardScaler` 的版本），刪掉重複 cell。

3. **`train_test_split` 洗牌後，`X_train` 的順序與 `meta` 不再對應**（潛在陷阱）
   - Part B 訓練用 `X_sub = X_all[:N_train_use]` 再 `train_test_split(shuffle 預設 True)`。這對「訓練」沒問題。
   - 但要注意：**推論 heatmap（cell 47/49）是直接用原始 `patches` + `meta` 依 `img_id` 過濾**，沒有用到打亂後的 `X_train`，所以目前碰巧沒 bug。不過這種「有時用原陣列、有時用切片」的寫法很脆弱，建議統一資料入口。

4. **`compute_recon_error_for_image` 依賴外部全域變數 `image_paths`、`np`、`H/W`**
   - 函式簽章收了 `patches_all, meta_all, model`，卻又偷用全域 `image_paths`。若 `image_paths` 在別的 cell 被重新定義（cell 11/20/22/39 都重定義過），結果會悄悄改變。
   - **建議**：把 `image_paths` 也當參數傳入，讓函式自足（pure function）。

5. **`IMG_ID` 可能沒有對應 patch 就丟例外**
   - 因為 patch 在 80k 上限處被截斷，`img_id=50`（cell 49）不一定有 patch。cell 47 有 `raise ValueError`，但 demo 直接寫死 `IMG_ID=50`，重跑不同資料量時可能中斷。
   - **建議**：demo 前先 `valid_ids = np.unique(meta[:,0])`，從中挑一個。

### 🟡 建議改善（穩健性 / 品質）

6. **`error_map` 正規化與門檻用同一張圖的 min/max/quantile**——每張圖的閾值都不同，無法跨圖比較「這張比那張更異常」。若要判斷「整張圖正常 vs 異常」，需要一個**全域**的誤差基準（例如用一批正常圖的誤差分布定門檻）。目前只能做「同張圖內的相對熱區」。
   → ✅ **已於整理版 Part C 補上**：用一批影像建立全域 per-patch 誤差門檻 `GLOBAL_THRESHOLD`，並替每張圖算異常分數做正常/可疑判定與排序。

7. **K=5 與 PCA=50 是硬編碼、缺乏依據**。可補上 elbow / silhouette 分數，或 PCA 的 `explained_variance_ratio_` 累積曲線來佐證選擇。

8. **沒有固定所有隨機源**。`KMeans`/`PCA`/`train_test_split` 有給 `random_state`，但 `np.random.choice`（cell 33/37）、TensorFlow 權重初始化沒有設種子，結果不完全可重現。建議統一 `set_seed()`。

9. **重複的 import 與函式定義**散落在 cell 19–22（`preprocess_image`/`extract_patches` 定義了 3 次，且 `resize_to` 預設值不一致：cell 15 是 `(512,512)`、cell 21 是 `None`）。這正是「同名函式不同行為」的溫床，應合併成單一 cell。

10. **`float16 → float32` 來回轉換**雖省記憶體，但 `float16` 精度低，patch 值在 `[0,1]` 尚可接受；不過 PCA 的 `StandardScaler` 在低精度上可能放大數值誤差。若記憶體允許，資料前處理階段建議維持 `float32`。

### 🟢 小建議 / 加分項

11. **可加量化評估**：目前全靠肉眼看 heatmap。若能取得少量標註（哪怕 10 張），就能算 ROC-AUC / IoU，讓「缺陷偵測有效」有數據支撐。
    → ✅ **已於整理版 Part C 補上**：用合成缺陷注入產生 GT mask，計算像素級 ROC-AUC / IoU 與影像級 ROC-AUC（取得真實標註後把 `inject_defects` 換掉即可）。
12. **註解可再工程化**：像 `# 降成 float16 省記憶體 爆掉了`、`#60000改20000(加快速度)` 這類開發過程留言，正式版可整理成參數說明。
13. **可重構為 `.py` 模組 + 參數設定區塊**：把 `PATCH_SIZE / STRIDE / K / MAX_PATCHES` 等集中成一個 config，方便調參與復現。
14. **安全性做得好**👍：`kaggle.json` 用上傳而非寫死金鑰（cell 3），README 也特別提醒，符合最佳實務。

---

## 5. 一句話總結

這是一個**架構正確、概念完整**的無監督缺陷偵測原型：Part A 用 PCA+K-means 做微結構分群、Part B 用 Autoencoder 重建誤差做 patch 級異常定位，並靠 `meta` 座標把結果貼回原圖。主要待改進處集中在**變數命名/重複 cell 造成的混亂**、**兩套分群結果混用**，以及**缺乏量化評估與跨圖統一門檻**。清理重複程式碼並補上評估指標後，會是一個相當扎實的專案。
