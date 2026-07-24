# Dependency Audit — Zoom-SALK v2.4.0

**Audit Date:** 2026-07-23

## Inference Dependencies (REQUIRED)

| Package | Version | Status | Notes |
|---------|---------|--------|-------|
| torch | 2.2.2+cpu | ✅ PASS | Do NOT reinstall on Colab |
| torchvision | 0.17.2+cpu | ✅ PASS | Matches torch |
| numpy | 1.26.4 | ✅ PASS | MUST be <2 (torch ABI) |
| opencv-python-headless | 4.11.0 | ✅ PASS | Unpinned (py3.12 compatible) |
| timm | 0.9.12 | ✅ PASS | ResNet50 backbone creation |
| einops | 0.8.1 | ✅ PASS | Tensor rearrange in MHSIU/RGPU |

## Evaluation Dependencies (REQUIRED for metrics)

| Package | Version | Status | Notes |
|---------|---------|--------|-------|
| pysodmetrics | installed | ✅ PASS | Sm, Fw, MAE, Em computation |
| scipy | 1.14.1 | ✅ PASS | pysodmetrics dependency |
| PyYAML | 6.0.3 | ✅ PASS | dataset.yaml parsing |
| matplotlib | 3.9.2 | ✅ PASS | Visualization (Colab pre-installs) |
| tqdm | 4.67.1 | ✅ PASS | Progress bars |

## Forbidden Packages

| Package | Status | Reason |
|---------|--------|--------|
| albumentations | ✅ NOT INSTALLED | Pulls numpy>=2, breaks torch ABI |
| numpy>=2 | ✅ NOT PRESENT | Incompatible with torch compiled against numpy 1.x |

## Colab Pre-installed Packages (IGNORE)

The following packages are pre-installed on Google Colab but are NOT used by ZoomNeXt.
Their version conflicts (if any) are irrelevant and should be ignored:

- jax, jaxlib (requires numpy>=2, but we force numpy<2 — jax breaks, which is fine)
- cupy, rasterio, shap (unrelated to COD)
- opencv version warnings (we use opencv-python-headless, not opencv-python)

## Install Commands (Minimal)

```bash
# Inference only (6 packages):
pip install opencv-python-headless timm==0.9.12 einops gdown
pip install "numpy<2"

# Evaluation (5 additional packages):
pip install pysodmetrics PyYAML matplotlib tqdm
pip install "numpy<2"  # re-force after any package that may pull numpy 2
```

## Verification

```bash
python cpu_demo/dependency_check.py
```

Expected: `ALL DEPENDENCIES OK — SAFE TO RUN BENCHMARK`

## Verdict

**Dependency Audit: PASS** — all required packages present, no forbidden packages, numpy<2 enforced.
