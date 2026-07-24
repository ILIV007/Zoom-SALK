# Results & Sample Outputs

This folder contains benchmark results and sample prediction outputs.

## Benchmark Results (Frozen)

### Baseline (390 COD10K-Test images, CPU)
| Metric | Our Result | Paper | Diff |
|--------|-----------|-------|------|
| Sm ↑   | 0.8484    | 0.861 | -1.5% |
| Fw ↑   | 0.7597    | 0.768 | -1.0% |
| MAE ↓  | 0.0369    | 0.026 | +42%  |
| Em ↑   | 0.8976    | 0.926 | -3.0% |

### Representative TTA (20-image subset)
| Metric | Baseline | TTA    | Delta   | %       |
|--------|----------|--------|---------|---------|
| Sm ↑   | 0.8527   | 0.8609 | +0.0082 | +0.97%  |
| Fw ↑   | 0.7845   | 0.7889 | +0.0044 | +0.56%  |
| MAE ↓  | 0.0537   | 0.0512 | -0.0025 | -4.64%  |
| Em ↑   | 0.9054   | 0.9080 | +0.0026 | +0.29%  |

Per-image outcome: 15 improved (75%), 1 unchanged (5%), 4 worse (20%)

## JSON Data
- `project/outputs/phase451_env.json` — Environment lock
- `project/outputs/phase452_baseline.json` — Baseline statistics (390 images)
- `project/outputs/tta_comparison/tta_full_validation.json` — TTA per-image results
- `project/outputs/tta_comparison/statistical_summary.json` — TTA aggregate
- `project/outputs/failure_analysis.json` — Failure case data

## Sample Predictions
Sample prediction images can be found in the Phase 3.1 and Phase 3.2 reports (DOCX files in `docs/reports/technical/`).
