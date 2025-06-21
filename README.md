# [MICCAI 2025] CSAL-3D

**CSAL-3D: Cold-start Active Learning for 3D Medical Image Segmentation via SSL-driven Uncertainty-Reinforced Diversity Sampling**, for MICCAI 2025 (Early Accept).
## 📌 Overall Framework

The overall CSAL-3D pipeline consists of:
- A **CSAL-adapted Self-Supervised Learning (SSL)** framework for both 3D-aware feature extraction and uncertainty estimation.
- An **Ensemble-based Uncertainty Estimation** strategy to generate sample-level uncertainty scores.
- A **URDS (Uncertainty-Reinforced Diversity Sampling)** method that hierarchically combines diversity and uncertainty for one-shot sample selection.

## 📁 Dataset Download
We evaluate our method on three publicly available 3D medical image segmentation datasets from the **Medical Segmentation Decathlon (MSD)**:
- **Brain Tumor (Task01_BrainTumour)** [MRI, multi-modality]
- **Heart (Task02_Heart)** [MRI]
- **Spleen (Task09_Spleen)** [CT]
- Datasets can be downloaded from the official MSD website:
👉 [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

### Environment Setup

We recommend using Python 3.8+ with PyTorch 2.3.1 and MONAI 1.3+.

## 🙏 Acknowledgement 
Our codebase is built upon the excellent COLosSAL project (https://github.com/han-liu/COLosSAL), which provides a benchmark for Cold-Start Active Learning in 3D medical image segmentation.
