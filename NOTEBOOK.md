# Unsupervised Learning Defect Detector — Notebook (Markdown 匯出)

> 由 unsupervised_learning_defect_detector.ipynb 自動轉出，共 52 個 cell。

1. 經典無監督 (PCA + K-means / GMM) 做 microstructure clustering + 異常分數

2. CNN Autoencoder 做 patch-level anomaly detection（主角）

3. Self-supervised 表徵學習 + One-Class SVM / Isolation Forest

$$整個計畫的目標$$
輸入: 一張 SEM 影像

輸出:

*   一張 anomaly heatmap 顯示哪裡可能是孔洞、夾雜、異常晶粒、加工缺縣


*   或標示這張圖是「正常 microstructure」還是「有可疑缺陷」。

$$在 Colab 自動建立 kaggle.json$$

```python
import json, os

# 請將你的 kaggle.json 上傳到 Colab，勿將 key 直接寫在 notebook
# kaggle.json 可從 https://www.kaggle.com/settings -> API -> Create New Token 下載
kaggle_json_path = "/root/.kaggle/kaggle.json"
if not os.path.exists(kaggle_json_path):
    from google.colab import files
    print("請上傳你的 kaggle.json")
    uploaded = files.upload()
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open(kaggle_json_path, "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(kaggle_json_path, 0o600)
print("Kaggle credentials ready.")
```

$$安裝 Kaggle CLI$$

```python
!pip install kaggle
```

https://www.kaggle.com/datasets/dragonzhang/grainsize-train?resource=download

$$下載指定 dataset$$

```python
!kaggle datasets download -d dragonzhang/grainsize-train
```

$$解壓縮 Dataset$$

```python
!unzip -q grainsize-train.zip -d grainsize_dataset
```

```python
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = Path("/content/grainsize_dataset")
IMG_EXT = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]

image_paths = [p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT]
print("共有影像張數：", len(image_paths))
image_paths[:5]
```

簡單看一張圖確認：

```python
img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
plt.imshow(img0, cmap='gray')
plt.axis('off')
```

$1. Preprocessing：灰階 + Normalize +縮放$

```python
def preprocess_image(path, resize_to=(512, 512)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

    # 轉成 float32 然後做 normalize
    img = img.astype(np.float32) / 255.0

    return img

# 測試
img = preprocess_image(image_paths[0], resize_to=None)
print("shape:", img.shape, "min:", img.min(), "max:", img.max())
plt.imshow(img, cmap='gray')
plt.axis('off')
```

2. 切 Patch 的函式

```python
def extract_patches(img, patch_size=64, stride=32):

    H, W = img.shape
    patches = []
    coords = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

    patches = np.stack(patches, axis=0)
    coords = np.array(coords)
    return patches, coords

# test
patches, coords = extract_patches(img, patch_size=64, stride=32)
print("圖被切成幾個 patch :", patches.shape[0])
print("每個 patch 尺寸:", patches.shape[1:])

# 看幾個 patch
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(patches[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

3. 切整個 Dataset 的 patch

```python
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
```

```python
DATA_ROOT = Path("/content/grainsize_dataset")  # 你的 dataset 資料夾路徑
IMG_EXT = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]

image_paths = [p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT]

print("找到影像數量:", len(image_paths))
image_paths[:5]
```

```python
def preprocess_image(path, resize_to=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = img.astype(np.float32) / 255.0
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    return img


def extract_patches(img, patch_size=64, stride=32):
    H, W = img.shape
    patches = []
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    return np.stack(patches), np.array(coords)
```

```python
import cv2
import numpy as np
from pathlib import Path

DATA_ROOT = Path("/content/grainsize_dataset")
IMG_EXT = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]

image_paths = [p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT]
print("找到影像數量:", len(image_paths))

def preprocess_image(path, resize_to=(512, 512)):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def extract_patches(img, patch_size=64, stride=32):
    H, W = img.shape
    patches = []
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    return np.stack(patches), np.array(coords)
```

```python
PATCH_SIZE = 64
STRIDE = 32
MAX_PATCHES_TOTAL = 80000

all_patches = []
meta_info = []

total_patches = 0

for img_id, path in enumerate(image_paths):
    img = preprocess_image(path, resize_to=(512, 512))
    patches, coords = extract_patches(img, patch_size=PATCH_SIZE, stride=STRIDE)

    if total_patches + patches.shape[0] > MAX_PATCHES_TOTAL:
        remain = MAX_PATCHES_TOTAL - total_patches
        if remain <= 0:
            break
        patches = patches[:remain]
        coords = coords[:remain]

    # 降成 float16 省記憶體 爆掉了
    patches = patches.astype(np.float16)

    all_patches.append(patches)
    img_ids = np.full((coords.shape[0], 1), img_id, dtype=np.int32)
    meta = np.hstack([img_ids, coords])
    meta_info.append(meta)

    total_patches += patches.shape[0]

    print(f"處理到第 {img_id} 張圖，累積 patch 數: {total_patches}")

    if total_patches >= MAX_PATCHES_TOTAL:
        break

all_patches = np.concatenate(all_patches, axis=0)   # (N, 64, 64)
meta_info  = np.concatenate(meta_info, axis=0)      # (N, 3)

print("總共 patch 數量:", all_patches.shape[0])
print("meta_info 形狀:", meta_info.shape)

# 加 channel 維度
patches = all_patches[..., np.newaxis]  # (N, 64, 64, 1)
meta    = meta_info
print("patches 形狀:", patches.shape)
```

PCA + K-means 無監督分群
 準備資料（攤平 + 抽樣）

為了省 RAM，我們只用最多 30,000 個 patch 來做 PCA + clustering。

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

MAX_SAMPLES_PCA = 30000
N = min(patches.shape[0], MAX_SAMPLES_PCA)

X = patches[:N]               # (N, 64, 64, 1)
X_flat = X.reshape(N, -1)     # (N, 4096)

print("PCA 使用 patch 數:", N, "，每個維度:", X_flat.shape[1])
```

標準化 + PCA 降維

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# 降到 50 維
pca_50 = PCA(n_components=50, random_state=42)
X_pca50 = pca_50.fit_transform(X_scaled)

# 降到 2 維
pca_2 = PCA(n_components=2, random_state=42)
X_pca2 = pca_2.fit_transform(X_scaled)

print("PCA50:", X_pca50.shape, "  PCA2:", X_pca2.shape)
```

分群 + visualize

```python
import matplotlib.pyplot as plt

K = 5  # 分5群結果最適合
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca50)

print("各群大小:", np.bincount(clusters))

plt.figure(figsize=(6,5))
for k in range(K):
    idx = (clusters == k)
    plt.scatter(X_pca2[idx, 0], X_pca2[idx, 1], s=3, alpha=0.5, label=f"C{k}")
plt.legend(markerscale=3)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D + K-means Clusters")
plt.show()
```

Step 1 — flatten + PCA + K-means

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# 將 patch (N, 64, 64, 1) → (N, 4096)
patches_flat = patches.reshape(patches.shape[0], -1)

# PCA 降維
pca = PCA(n_components=50, random_state=42)
patches_pca = pca.fit_transform(patches_flat)

# 分群
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(patches_pca)

print("分群完成")
print("各群數量：", np.bincount(cluster_labels))
```

Step 2 — 抽取每群 25 個 patch 作為代表

```python
import random

samples_per_cluster = 25
cluster_samples = {}

for c in range(5):
    idx = np.where(cluster_labels == c)[0]
    chosen = np.random.choice(idx, samples_per_cluster, replace=False)
    cluster_samples[c] = patches[chosen]
```

Step 3 — 畫出 5*5 的代表圖（共五張圖，每張代表一群）

```python
import matplotlib.pyplot as plt

for c in range(5):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    fig.suptitle(f"Cluster {c} Representative Patches", fontsize=16)

    for i in range(25):
        ax = axes[i//5, i%5]
        ax.imshow(cluster_samples[c][i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.show()
```

A4. 看每個 cluster 的代表 patch

```python
def show_cluster_examples(cluster_id, num_examples=8):
    idx = np.where(clusters == cluster_id)[0]
    if len(idx) == 0:
        print(f"Cluster {cluster_id} 沒有樣本")
        return
    choose = np.random.choice(idx, size=min(num_examples, len(idx)), replace=False)
    imgs = X[choose, :, :, 0]

    cols = 4
    rows = int(np.ceil(len(imgs) / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id} examples")
    plt.show()

for k in range(K):
    show_cluster_examples(k)
```

A5. 把 cluster 畫回某張 SEM 圖（cluster map）

```python
import cv2
from pathlib import Path

DATA_ROOT = Path("/content/grainsize_dataset")
IMG_EXT = [".jpg", ".png", ".jpeg", ".tif", ".tiff"]
image_paths = [p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in IMG_EXT]

# 只對前 N 個 patch 有 cluster 標籤
meta_used = meta[:N]

IMG_ID = 0

mask_img = (meta_used[:, 0] == IMG_ID)
coords_img = meta_used[mask_img][:, 1:3]   # (M, 2)
clusters_img = clusters[mask_img]          # (M,)

print("這張圖的 patch 數:", len(clusters_img))

# 讀並 resize 成 512x512 (重要)
orig = cv2.imread(str(image_paths[IMG_ID]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
orig = cv2.resize(orig, (512, 512), interpolation=cv2.INTER_AREA)
H, W = orig.shape

cluster_map = -1 * np.ones((H, W), dtype=int)
PATCH_SIZE = 64
STRIDE = 32

for (y, x), c in zip(coords_img, clusters_img):
    cluster_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = c

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original (resized)")
plt.imshow(orig, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Cluster Map")
plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
plt.axis('off')
plt.show()
```

Part B：Autoencoder + Reconstruction Error Heatmap
B1. 準備訓練資料（轉回 float32）

我們用全部 80k patch 訓練也 OK，如果你覺得太慢，可以只用 60k 左右。

```python
from sklearn.model_selection import train_test_split

X_all = patches.astype(np.float32)

MAX_TRAIN = 20000  #60000改20000(加快速度)
N_train_use = min(X_all.shape[0], MAX_TRAIN)
X_sub = X_all[:N_train_use]

X_train, X_val = train_test_split(X_sub, test_size=0.1, random_state=42)
print("Train:", X_train.shape, " Val:", X_val.shape)
```

B2. 建立 Autoencoder 模型並訓練

```python
import tensorflow as tf
from tensorflow.keras import layers, models

input_shape = (64, 64, 1)

inp = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
x = layers.MaxPool2D((2,2))(x)              # 32x32
x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPool2D((2,2))(x)              # 16x16
x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
encoded = layers.MaxPool2D((2,2))(x)        # 8x8x128

x = layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu')(encoded) # 16x16
x = layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')(x)        # 32x32
x = layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')(x)        # 64x64
decoded = layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')(x)

autoencoder = models.Model(inp, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
autoencoder.summary()
```

```python
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
]

history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=15,#100改15
    batch_size=256,
    shuffle=True,
    callbacks=callbacks
)
```

```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.title('Autoencoder training')
plt.show()
```

B3. 對某張圖做 reconstruction error heatmap

```python
import cv2

def compute_recon_error_for_image(img_id, patches_all, meta_all, model,
                                  patch_size=64):
    # 找出屬於這張圖的 patch
    mask = (meta_all[:, 0] == img_id)
    coords = meta_all[mask][:, 1:3]
    patches_img = patches_all[mask]   # (M, 64, 64, 1)
    if patches_img.shape[0] == 0:
        raise ValueError(f"img_id {img_id} 沒有對應的 patch（因為前面截斷在 80k）")

    # AE 重建
    recon = model.predict(patches_img, batch_size=256, verbose=0)
    errors = np.mean((patches_img - recon)**2, axis=(1,2,3))  # (M,)

    # 讀原圖（resize 成 512x512，跟當初切 patch 一致）
    orig = cv2.imread(str(image_paths[img_id]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    orig = cv2.resize(orig, (512, 512), interpolation=cv2.INTER_AREA)
    H, W = orig.shape

    error_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for (y, x), e in zip(coords, errors):
        error_map[y:y+patch_size, x:x+patch_size] += e
        count_map[y:y+patch_size, x:x+patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    error_map /= count_map

    return orig, error_map
```

B4. 畫 heatmap + 疊圖 + binary defect mask

```python
IMG_ID = 50  # 想看哪一張換這裡

orig, error_map = compute_recon_error_for_image(
    img_id=IMG_ID,
    patches_all=patches.astype(np.float32),  # AE 推論用 float32
    meta_all=meta,
    model=autoencoder,
    patch_size=64
)

# 正規化到 0-1
em_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(orig, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Error Map")
plt.imshow(em_norm, cmap='inferno')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(orig, cmap='gray')
plt.imshow(em_norm, cmap='inferno', alpha=0.5)
plt.axis('off')

plt.tight_layout()
plt.show()
```

如果想看「二值化缺陷區」：

```python
threshold = np.quantile(error_map, 0.95)  # 取 error 前 5% 當 anomaly
defect_mask = (error_map >= threshold)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Defect Mask (top 5% error)")
plt.imshow(defect_mask, cmap='gray')
plt.axis('off')
plt.show()
```
