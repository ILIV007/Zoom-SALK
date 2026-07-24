# Table 3 — Ablation Study: Effectiveness of Proposed Modules (Image COD)

**Source:** ZoomNeXt paper, Page 11 (Table 3)

RGPU = Rich Granularity Perception Unit; MHSIU = Multi-Head Scale Integration Unit; L_UAL = Uncertainty Awareness Loss; ARG = Average Relative Gain.

| No. | Model | Params (M) | GFLOPs | COD10K Sm↑ | COD10K Fω↑ | COD10K MAE↓ | COD10K Fβ↑ | COD10K Em↑ | NC4K Sm↑ | NC4K Fω↑ | NC4K MAE↓ | NC4K Fβ↑ | NC4K Em↑ | ARG |
|-----|-------|-----------|--------|------------|------------|-------------|------------|------------|----------|----------|------------|----------|----------|-----|
| 0 | Baseline | 25.216 | 41.000 | 0.826 | 0.677 | 0.035 | 0.724 | 0.903 | 0.850 | 0.756 | 0.049 | 0.800 | 0.912 | 0.00% |
| 1.1 | +RGPU(G=2) | 25.583 | 47.801 | 0.836 | 0.691 | 0.033 | 0.739 | 0.913 | 0.859 | 0.769 | 0.047 | 0.808 | 0.916 | +2.05% |
| 1.3 | +RGPU(G=6) | 27.145 | 77.694 | 0.843 | 0.705 | 0.031 | 0.749 | 0.914 | 0.863 | 0.778 | 0.045 | 0.816 | 0.919 | +3.77% |
| 2.3 | +MH-SIU(M=4) | 26.530 | 149.100 | 0.856 | 0.728 | 0.029 | 0.770 | 0.923 | 0.870 | 0.789 | 0.043 | 0.825 | 0.924 | +6.03% |
| 3 | +RGPU(G=6)+MHSIU(M=4) | 28.458 | 185.794 | 0.861 | 0.734 | 0.029 | 0.773 | 0.924 | 0.875 | 0.792 | 0.042 | 0.826 | 0.925 | +6.56% |
| **4** | **No.3 + L_UAL** | **28.458** | **185.794** | **0.861** | **0.768** | **0.026** | **0.801** | **0.925** | **0.874** | **0.816** | **0.037** | **0.846** | **0.928** | **+9.94%** |

**Key observations:**
- Each component contributes positively: RGPU (+3.77%), MHSIU (+6.03%), combined (+6.56%), +UAL (+9.94%).
- RGPU best at G=6 groups; MHSIU best at M=4 heads.
- UAL provides the largest single-component boost (from +6.56% to +9.94%), especially on Fω (0.734 → 0.768) and MAE (0.029 → 0.026).
- The final model (No.4) is RN50_ZoomNeXt with 28.458M params and 185.794 GFLOPs.
