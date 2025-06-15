# Anomaly_Detection_Deakin_FTP  
**Unsupervised Defect Segmentation** using multi‑level ResNet50 features + 1×1‑Conv Autoencoder

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/pytorch-1.12%2B-red)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A complete research & codebase for unsupervised industrial anomaly segmentation on MVTec AD and VisA datasets, combining:
1. **Hierarchical feature extraction** from frozen ResNet50 (layers 2 & 3 hooks)  
2. **Multi‑scale feature fusion** (adaptive pooling + concatenation)  
3. **Deep feature reconstruction** via a 1×1 convolutional autoencoder  
4. **Anomaly scoring & localization** (reconstruction‐error heatmaps, top‑k image‐level scoring, thresholding, up‑sampling for pixel‐level localization)

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/FTP_Anomaly_Detection.git
cd FTP_Anomaly_Detection

# 2. Create and activate a virtualenv (optional but recommended)
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Jupyter Notebook  
jupyter lab Autoencoders_using_resnet_backbone_multilevel.ipynb



FTP_Anomaly_Detection/
├── Deakin_FTP_Report.pdf  
├── Autoencoders_using_resnet_backbone_multilevel.ipynb  
├── requirements.txt  
├── models/  
│   ├── mvtec_carpet_ae.pth  
│   ├── mvtec_bottle_ae.pth  
│   └── ...           # Trained CAE checkpoints per category
└── results/  
    ├── mvtec_metrics.csv  
    └── visa_metrics.csv  
