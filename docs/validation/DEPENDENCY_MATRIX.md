# DEPENDENCY MATRIX — Zoom-SALK

## Inference (ESSENTIAL — 6 packages)
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.2.2+cpu | Model forward pass |
| torchvision | 0.17.2+cpu | timm dependency |
| numpy | 1.26.4 (<2) | Arrays — MUST be <2 |
| opencv-python-headless | latest | Image I/O |
| timm | 0.9.12 | ResNet50 backbone |
| einops | 0.8.1 | Tensor rearrange |

## Evaluation (5 packages)
| Package | Version | Purpose |
|---------|---------|---------|
| pysodmetrics | latest | Sm, Fw, MAE, Em |
| scipy | 1.14.1 | pysodmetrics dep |
| PyYAML | 6.0.3 | dataset.yaml |
| matplotlib | 3.9.2 | Visualization |
| tqdm | 4.67.1 | Progress bars |

## Forbidden
| Package | Reason |
|---------|--------|
| albumentations | Pulls numpy>=2, breaks torch |
| numpy>=2 | Incompatible with torch ABI |

## Install
```bash
pip install opencv-python-headless timm==0.9.12 einops gdown
pip install "numpy<2"
pip install pysodmetrics PyYAML matplotlib tqdm
```
