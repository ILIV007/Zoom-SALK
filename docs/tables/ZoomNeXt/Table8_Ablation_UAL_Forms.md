# Table 8 — Different Forms of UAL (Uncertainty Awareness Loss)

**Source:** ZoomNeXt paper, Page 13 (Table 8)

Form 0 = model without UAL. Form 1.5 (power, α=2) is the default. "—" = unable to converge.

| No. | Form | α | COD10K Sm↑ | COD10K Fω↑ | COD10K MAE↓ | COD10K Fβ↑ | COD10K Em↑ | NC4K Sm↑ | NC4K Fω↑ | NC4K MAE↓ | NC4K Fβ↑ | NC4K Em↑ |
|-----|------|---|------------|------------|-------------|------------|------------|----------|----------|------------|----------|----------|
| 0 | ✗ (no UAL) | — | 0.861 | 0.734 | 0.029 | 0.773 | 0.924 | 0.875 | 0.792 | 0.042 | 0.826 | 0.925 |
| 1.1–1.3 | pow, α=1/8, 1/4, 1/2 | — | — | — | — | — | — | — | — | — | — | — |
| 1.4 | pow, α=1 | 1 | 0.859 | 0.762 | 0.026 | 0.795 | 0.923 | 0.870 | 0.810 | 0.039 | 0.841 | 0.926 |
| **1.5** | **pow, α=2** | **2** | **0.861** | **0.768** | **0.026** | **0.801** | **0.925** | **0.874** | **0.816** | **0.037** | **0.846** | **0.928** |
| 1.6 | pow, α=4 | 4 | 0.857 | 0.768 | 0.027 | 0.802 | 0.923 | 0.868 | 0.815 | 0.038 | 0.844 | 0.925 |
| 1.7 | pow, α=8 | 8 | 0.859 | 0.767 | 0.026 | 0.802 | 0.924 | 0.867 | 0.814 | 0.038 | 0.843 | 0.923 |
| 2.4 | exp, α=1 | 1 | 0.863 | 0.753 | 0.026 | 0.790 | 0.927 | 0.873 | 0.802 | 0.039 | 0.834 | 0.925 |
| 2.5 | exp, α=2 | 2 | 0.862 | 0.762 | 0.026 | 0.797 | 0.927 | 0.873 | 0.810 | 0.038 | 0.841 | 0.928 |
| 3 | BCE w/ ω=1+Φ²pow(x) | 2 | 0.861 | 0.737 | 0.028 | 0.777 | 0.928 | 0.873 | 0.791 | 0.041 | 0.825 | 0.926 |

**UAL formula:** L_UAL = 1 − Δ = 1 − |2p − 1|²  (where Δ = certainty, p = prediction)

**Key observations:**
- Without UAL (Form 0): Fω = 0.734, MAE = 0.029 on COD10K.
- With UAL Form 1.5 (power, α=2): Fω = 0.768, MAE = 0.026 — a significant improvement.
- The power form (1.5) outperforms the exponential form (2.5) on Fω.
- α=2 is the sweet spot; higher α (4, 8) slightly degrades.
- Using UAL as a BCE weight (Form 3) is worse than as an auxiliary loss — the separate loss formulation is better.
- Several forms (1.1–1.3 with small α) fail to converge.
