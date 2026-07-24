# Release Notes — Zoom-SALK v2.4.0

**Release Date:** 2026-07-24
**Status:** Final Release — FROZEN

## Highlights

- Complete COD research project: SALK-Net theory + ZoomNeXt implementation + TTA extension
- Baseline benchmark within 1-3% of paper (3 of 4 metrics)
- TTA improved all 4 metrics on representative 20-image subset (+0.97% Sm, -4.64% MAE)
- Full university submission package in Persian (RTL) and English
- Self-contained execution package (Run_Project/)
- SHA256-verified pretrained weights

## Repository Statistics

| Metric | Value |
|--------|-------|
| Total files | 180+ |
| Reports | 16 (9 DOCX + 1 PDF + 6 MD) |
| Scripts | 11 Python |
| Notebook | 1 Colab (21 cells) |
| Model | RN50_ZoomNeXt (28,458,084 params) |
| Dataset | COD10K-Test (390 evaluated of 2,026) |

## Benchmarks

| Metric | Our Result | Paper |
|--------|-----------|-------|
| Sm | 0.848 | 0.861 |
| Fw | 0.760 | 0.768 |
| MAE | 0.037 | 0.026 |
| Em | 0.898 | 0.926 |

## Known Limitations

- 390-image subset (not full 2,026)
- TTA on 20 images (representative, not full)
- CPU evaluation only
- SALK-Net: no code/weights for direct comparison

## Download

```
Zoom-SALK_v2.4.0.zip (19 MB, without weight)
Weight: gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW (114 MB, separate download)
```

## Requirements

Python 3.12 | PyTorch 2.2+ | numpy<2 | timm==0.9.12 | einops | opencv-python-headless

## License

Apache-2.0

## Citation

See CITATION.cff

## Future Work

Full 2,026-image TTA on GPU, multi-scale TTA, evaluation on other datasets, backbone comparison.
