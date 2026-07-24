# 🔍 Zoom-SALK — Camouflaged Object Detection using Deep Learning

![Version](https://img.shields.io/badge/version-v3.0.2-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![License](https://img.shields.io/badge/license-Apache--2.0-orange)
![Status](https://img.shields.io/badge/status-FROZEN-success)

**Reproducible Evaluation and Representative Test-Time Augmentation Study of ZoomNeXt for Camouflaged Object Detection**

---

## 📋 Introduction

This project combines theoretical analysis of **SALK-Net** (Scientific Reports, 2025) with practical implementation using **ZoomNeXt** (IEEE TPAMI, 2024) for Camouflaged Object Detection (COD). A Test-Time Augmentation (TTA) modification is proposed and validated on a representative subset.

> **SALK-Net and ZoomNeXt solve the same Computer Vision problem (Camouflaged Object Detection) using different architectures. Due to the availability of an official implementation and pretrained weights, ZoomNeXt is selected as the practical implementation framework.**

---

## 👥 Authors

| Role | Name |
|------|------|
| Student | ایلیا ایروانی (Ilya Irvani) |
| Supervisor | میلاد سلطانی (Milad Soltani) |
| Course | مبانی بینایی ماشین (Fundamentals of Computer Vision) |
| Semester | نیمسال دوم ۱۴۰۴-۱۴۰۵ |

---

## 📊 Benchmark Results

### Baseline (390 COD10K-Test images, CPU)
| Metric | Our Result | Paper | Diff |
|--------|-----------|-------|------|
| Sm ↑ | 0.848 | 0.861 | -1.5% |
| Fw ↑ | 0.760 | 0.768 | -1.0% |
| MAE ↓ | 0.037 | 0.026 | +42% |
| Em ↑ | 0.898 | 0.926 | -3.0% |

### Representative TTA (20-image subset)
| Metric | Baseline | TTA | Delta |
|--------|----------|-----|-------|
| Sm | 0.8527 | 0.8609 | +0.97% |
| Fw | 0.7845 | 0.7889 | +0.56% |
| MAE | 0.0537 | 0.0512 | -4.64% |
| Em | 0.9054 | 0.9080 | +0.29% |

**All 4 metrics improved. 75% of images improved individually.**

---

## 🚀 Quick Start

```bash
# Install (6 packages only)
pip install opencv-python-headless timm==0.9.12 einops gdown
pip install "numpy<2"

# Download weight
gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW -O weights/resnet50-zoomnext.pth

# Run inference
python run/infer.py --image test.jpg --weight weights/resnet50-zoomnext.pth

# Run with TTA
python run/infer_tta.py --image test.jpg --weight weights/resnet50-zoomnext.pth
```

**[📖 Windows Guide](run/WINDOWS_SETUP_FA.md)** | **[☁️ Colab Guide](run/COLAB_GUIDE_FA.md)**

---

## 📁 Repository Structure

```
Zoom-SALK/
├── docs/                    ← All documentation
│   ├── reports/             ← Reports (English, Persian, Technical)
│   ├── presentation/        ← Slides (PPTX + PDF)
│   ├── figures/             ← Figures from papers (SALK-Net, ZoomNeXt)
│   ├── tables/              ← Tables from papers
│   ├── validation/          ← QA and validation reports
│   └── executive_summary/   ← 2-page executive summary
├── project/                 ← Full project workspace
│   ├── notebook/            ← Colab notebook
│   ├── scripts/             ← All Python scripts
│   ├── configs/             ← Model configurations
│   ├── repository/          ← Official ZoomNeXt source
│   ├── outputs/             ← TTA + benchmark JSON results
│   └── weights/             ← Weight download instructions
├── run/                     ← Self-contained execution package
├── deliverables/            ← University submission files
├── github/                  ← GitHub-ready package
└── archive/                 ← Original phase-based structure
```

---

## 📦 Weight Download

| Property | Value |
|----------|-------|
| File | resnet50-zoomnext.pth |
| Size | 114 MB |
| SHA256 | `e7f958ce379ae709494999d62f489a3bce4cf7c2852f532e7db376abd8119ab9` |
| Parameters | 28,458,084 (28.458M) |
| Download | `gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW` |

---

## 🔧 Requirements

**Inference (6):** torch (pre-installed), torchvision, numpy<2, opencv-python-headless, timm==0.9.12, einops

**Evaluation (+5):** pysodmetrics, scipy, PyYAML, matplotlib, tqdm

**Forbidden:** albumentations (pulls numpy≥2, breaks torch)

---

## 🎓 University Deliverables

All files in `deliverables/`:
- University_Report_FA.docx / .pdf (Persian, RTL)
- Presentation_FA.pptx / .pdf (16 slides)
- Presentation_Script_FA.docx (10-12 min)
- Defense_QA_FA.docx (30 Q&A)

---

## ❓ FAQ

<details>
<summary>Why not use SALK-Net directly?</summary>
SALK-Net has no released code or pretrained weights. ZoomNeXt has both.
</details>

<details>
<summary>Why only 390 images?</summary>
Sandbox limitations (4GB RAM, CPU only). 390 is a representative subset.
</details>

<details>
<summary>Is TTA a "full validation"?</summary>
No. It is a **representative** evaluation on 20 images. This is clearly stated in all reports.
</details>

---

## 📝 Citation

```bibtex
@software{zoom-salk,
  title = {Zoom-SALK: Camouflaged Object Detection using Deep Learning},
  version = {3.0.0},
  year = {2026},
  license = {Apache-2.0}
}
```

---

## 📜 License

Apache License 2.0 — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [ZoomNeXt](https://github.com/lartpang/ZoomNeXt) by Pang et al. (IEEE TPAMI 2024)
- SALK-Net by Zheng et al. (Scientific Reports 2025)

---

**v3.0.2 | 2026-07-24 | FROZEN — READY FOR UNIVERSITY SUBMISSION**
