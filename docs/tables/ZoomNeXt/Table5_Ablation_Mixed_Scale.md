# Table 5 — Mixed-scale Input Scheme Ablation

**Source:** ZoomNeXt paper, Page 12 (Table 5)

All models based on the baseline. Tests single-scale vs. mixed-scale input combinations.

| Input Scale | Strategy | Params (M) | GFLOPs | COD10K Sm | COD10K Fω | COD10K MAE | NC4K Sm | NC4K Fω | NC4K MAE | ARG |
|-------------|----------|-----------|--------|-----------|-----------|------------|---------|---------|----------|-----|
| 1.0× only | — | 25.216 | 41.000 | 0.826 | 0.677 | 0.035 | 0.850 | 0.756 | 0.049 | 0.00% |
| 0.5× only | — | 25.216 | 41.369 | 0.761 | 0.552 | 0.049 | 0.810 | 0.686 | 0.061 | −13.63% |
| 1.5× only | — | 25.216 | 104.972 | 0.850 | 0.717 | 0.031 | 0.856 | 0.769 | 0.048 | +3.36% |
| 0.5×,1.0× | Addition | 25.216 | 48.987 | 0.827 | 0.672 | 0.034 | 0.858 | 0.765 | 0.046 | +1.23% |
| 0.5×,1.0× | MHSIU | 25.952 | 61.581 | 0.833 | 0.689 | 0.033 | 0.857 | 0.768 | 0.046 | +2.07% |
| 0.5×,1.5× | MHSIU | 26.137 | 141.342 | 0.853 | 0.719 | 0.031 | 0.862 | 0.774 | 0.046 | +4.07% |
| 1.0×,1.5× | MHSIU | 25.952 | 132.459 | 0.856 | 0.722 | 0.030 | 0.866 | 0.780 | 0.045 | +4.96% |
| 0.5×,1.0×,1.5× | Addition | 25.216 | 120.571 | 0.849 | 0.719 | 0.030 | 0.863 | 0.782 | 0.043 | +5.15% |
| **0.5×,1.0×,1.5×** | **MHSIU** | **26.530** | **149.100** | **0.856** | **0.728** | **0.029** | **0.870** | **0.789** | **0.043** | **+6.03%** |
| 1.5×,1.5×,1.5× | MHSIU | 26.530 | 295.659 | 0.853 | 0.719 | 0.031 | 0.864 | 0.779 | 0.045 | +4.47% |

**Key observations:**
- The mixed 3-scale scheme (0.5×, 1.0×, 1.5×) with MHSIU gives the best result (+6.03%).
- MHSIU is consistently better than simple addition for multi-scale fusion.
- 1.5× scale alone is better than 1.0× alone (larger scale captures more structure).
- 0.5× alone is worst (too small — loses detail).
- Using 3× 1.5× is worse than mixed scales — the diversity of scales matters, not just larger inputs.
- This validates the "zoom in and out" design philosophy.
