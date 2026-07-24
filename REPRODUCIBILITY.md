# Reproducibility — Zoom-SALK v3.0.1

## Weight Verification
- **File:** resnet50-zoomnext.pth
- **Size:** 114,362,834 bytes (114.36 MB)
- **SHA256:** `e7f958ce379ae709494999d62f489a3bce4cf7c2852f532e7db376abd8119ab9`
- **Parameters:** 28,458,084 (28.458M) — matches paper exactly
- **Weight keys:** 799 (all matched with strict=True, 0 missing, 0 unexpected)

## Environment Lock
- **Python:** 3.12.13
- **PyTorch:** 2.2.2+cpu (sandbox) / 2.5.1+cu121 (Colab)
- **NumPy:** 1.26.4 (<2 — REQUIRED for torch ABI compatibility)
- **timm:** 0.9.12
- **einops:** 0.8.1
- **opencv:** 4.11.0
- **pysodmetrics:** latest (for evaluation)

## Model Construction
```python
import methods as model_zoo
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False)
# Parameters: 28,458,084 — matches paper
```

## Weight Loading
```python
from utils.io import load_weight
load_weight(model=model, load_path="resnet50-zoomnext.pth", strict=True)
# 799 keys, all matched, 0 missing, 0 unexpected
```

## Inference Pipeline (Verified)
```
Image → ms_resize(0.5x, 1.0x, 1.5x) → div(255) → permute(2,0,1) → unsqueeze(0)
  → RN50_ZoomNeXt forward → sigmoid → minmax normalize → resize to original → save
```
- Interpolation: cv2.INTER_LINEAR (bilinear)
- No padding, no cropping, no artifacts
- Output shape matches input shape

## TTA Pipeline (infer_tta.py)
```
P_original = sigmoid(model(I))
P_flipped = sigmoid(model(flip(I)))  → flip back
P_final = (P_original + P_flipped) / 2
```
- 2 forward passes (1.96× cost)
- 0 additional parameters
- No architecture change

## Checkpoint/Resume
- Inference script saves progress every 30 images
- Can resume from last checkpoint if interrupted

## Dependency Verification
```bash
python run/dependency_check.py
# Expected: ALL DEPENDENCIES OK — SAFE TO RUN BENCHMARK
```

## Forbidden Packages
- **albumentations** — pulls numpy≥2, breaks torch ABI
- **numpy≥2** — incompatible with torch compiled against numpy 1.x

## Dataset Integrity
- COD10K-Test: 2,026 images, 2,026 GT masks, 0 corrupt, 100% binary GT
- Validated with `validate_dataset.py`
