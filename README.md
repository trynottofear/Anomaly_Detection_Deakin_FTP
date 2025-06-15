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
git clone https://github.com/trynottofear/Anomaly_Detection_Deakin_FTP/FTP_Anomaly_Detection.git
cd FTP_Anomaly_Detection

# 2. Create and activate a virtualenv (optional but recommended)
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt (If present)

# 4. Launch the Jupyter Notebook  
jupyter lab Autoencoders_using_resnet_backbone_multilevel.ipynb
```

---

## 📁 Repository Structure

```
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
```

- **`Deakin_FTP_Report.pdf`**  
  The full technical report detailing methodology, experiments, and results.

- **`Autoencoders_using_resnet_backbone_multilevel.ipynb`**  
  End‑to‑end Jupyter Notebook that:
  1. Loads MVTec AD & VisA datasets  
  2. Extracts multi‑level features via ResNet50 hooks  
  3. Defines & trains the `FeatCAE` autoencoder per category  
  4. Computes anomaly‐score heatmaps, AUROC/F1, confusion matrices  
  5. Summarizes and plots per‐category metrics

- **`models/`**  
  Saved PyTorch state dicts for each object/category. Use these for inference or further experiments.

- **`results/`**  
  CSV files containing per‐category AUROC, AUPRC, F1‐score, threshold values, and summary tables for both MVTec AD and VisA.

---

## ⚙️ Installation & Dependencies

Create a virtual environment and install via:

```bash
pip install -r requirements.txt
```

**`requirements.txt`** should include at least:

```text
torch>=1.12
torchvision>=0.13
numpy
scikit-learn
matplotlib
tqdm
pandas
Pillow
jupyterlab
```

---

## 📝 Usage

### 1. Data Preparation  
Download MVTec AD and VisA datasets and arrange folder structure:
```
data/
├── mvtec_ad/
│   ├── carpet/
│   ├── bottle/
│   └── …  
└── visa/
    ├── toothbrush/
    └── …
```

Edit the notebook’s `DATA_ROOT` variable to point to your `data/dataset` directory.

### 2. Feature Extraction & CAE Training  
Run the notebook cells under:
- **Section 2:** Dataset loader & transforms  
- **Section 3:** `resnet_feature_extractor` definition  
- **Section 6–9:** Define, train, and save `FeatCAE` for each category  

Model checkpoints will be saved to `models/<category>_ae.pth`.

### 3. Inference & Evaluation  
Proceed to notebook sections:
- **Section 10–15:**  
  - Generate anomaly heatmaps  
  - Compute image‐level scores (top‑k mean)  
  - Plot ROC curves & confusion matrices  
  - Export metrics to `results/*.csv`

---

## 🎯 Key Highlights

- **Modular Design:** Separate ResNet backbone, feature fusion, CAE, and scoring.  
- **Data‑Driven Bottleneck:** Optionally set latent dim \(d\) via PCA (95% variance) as described in Section 3.5 of the report.  
- **Configurable Scoring:** Easily switch top‑k, threshold strategy (mean+3σ or percentile).  
- **Comprehensive Visuals:** Inline plots for feature maps, loss curves, ROC, and heatmaps.  
- **Reproducible Results:** All per‐category metrics are logged in CSV for easy comparison.

---

## 📊 Results

Metrics summary tables are under `results/`. 

Load and visualize with:
```python
import pandas as pd
df = pd.read_csv('results/mvtec_metrics.csv')
```

---

## 📚 Technical Report

See **`Deakin_FTP_Report.pdf`** for:
- Problem statement & related work  
- Detailed methodology with equations  
- Experimental protocol & hyperparameter choices  
- In‑depth analysis of results  

---

## 🤝 Contributing

Feel free to:
- Add support for other backbones (e.g. EfficientNet)  
- Plug in different scoring strategies (e.g. SSIM‐based loss)  
- Extend to semi‐supervised or few‐shot anomaly detection  

Please open issues or pull requests on GitHub.

---

## 📄 License

Distributed under the [MIT License](LICENSE).  

---

## ⚠️ Code Attribution

This repository includes components inspired by or adapted from:

- Some GitHub notebooks from earlier MVTec anomaly detection work
- Open-source repositories

If you are the author of any reused component and would like credit added or content removed, feel free to contact me.

---

*Happy anomaly hunting!*  
