# Unsupervised Learning Defect Detector

Microstructure clustering and anomaly detection on SEM grain images — no labels required.  
Two complementary approaches: **PCA + K-means** for pattern discovery, and a **CNN Autoencoder** for patch-level defect localization.

## Demo

- **Live notebook**: https://njzneverdie.github.io/Unsupervised-Learning-Defect-Detector/notebook.html
- **Presentation**: [`presentation.pdf`](presentation.pdf)

## Dataset

[Grain Size Train](https://www.kaggle.com/datasets/dragonzhang/grainsize-train) — SEM grain microstructure images.

## Approach

### Part A — PCA + K-means Clustering

1. **Preprocessing**: grayscale → normalize → extract 64×64 patches (stride 32)
2. **Dimensionality reduction**: PCA to 50 components (`StandardScaler` first)
3. **Clustering**: K-means with K=5, visualize representative patches per cluster
4. **Cluster map**: project cluster labels back onto original image coordinates

Goal: identify distinct microstructure patterns (e.g. grain boundaries, smooth regions, defect-prone zones) without any annotation.

### Part B — CNN Autoencoder + Reconstruction Error Heatmap

1. **Training**: Autoencoder trained on 64×64 patches (unsupervised, no defect labels)
2. **Inference**: reconstruct each patch; high reconstruction error → likely anomaly
3. **Heatmap**: per-pixel error map overlaid on original image
4. **Defect mask**: threshold at 95th percentile of error values → binary anomaly mask

```
Input patch → Encoder → Latent vector → Decoder → Reconstructed patch
                                                          ↓
                                               MSE vs original → error heatmap
```

## Setup

```bash
pip install tensorflow scikit-learn opencv-python matplotlib kaggle
```

To download the dataset in Colab, upload your `kaggle.json` when prompted (do **not** hardcode credentials).

## Results

- Part A produces interpretable cluster maps showing spatial distribution of microstructure types
- Part B generates anomaly heatmaps highlighting potential defect regions without any labeled training data

## Project Structure

```
├── unsupervised_learning_defect_detector.ipynb  # Notebook (API key removed)
├── presentation.pdf                              # Project slides
└── docs/                                         # GitHub Pages
    ├── index.html
    └── notebook.html
```
