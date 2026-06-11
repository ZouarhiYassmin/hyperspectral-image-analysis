# hyperspectral-image-analysis

Features


Convolutional Autoencoder (CNN-AE) for lossy image compression
Reconstruction quality metrics: PSNR, SSIM
Spectral Angle Mapper (SAM) for comparing spectral signatures
Pearson correlation across spectral bands
PCA-based dimensionality reduction for spectral visualization
Spectral loss analysis: how much spectral information is lost after compression

hyperspectral-image-analysis/
├── src/
│   ├── autoencoder.py        # CNN autoencoder architecture (PyTorch)
│   ├── train.py              # Training loop with early stopping
│   ├── metrics.py            # PSNR, SSIM, SAM, Pearson
│   ├── spectral_analysis.py  # PCA, signature comparison, visualization
│   └── dataset.py            # Dataset loader (synthetic + real .npy/.mat)
├── notebooks/
│   └── demo.ipynb            # End-to-end demo notebook
├── tests/
│   └── test_metrics.py       # Unit tests for metrics
├── results/                  # Output figures and model checkpoints
├── requirements.txt
└── README.md
