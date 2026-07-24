# Table 8 — Ablation: Loss Constraints (COD10K)

**Source:** SALK-Net paper, Page 15 (Table 8)

| No. | Loss combination | Sm ↑ | Fω ↑ | M ↓ | Em ↑ |
|-----|------------------|------|------|-----|------|
| 1 | L_dwbcel | 0.854 | 0.757 | 0.027 | 0.913 |
| 2 | L_bcel + L_dwbioul | 0.856 | 0.762 | 0.027 | 0.914 |
| 3 | L_dwbcel + L_dwioul | 0.858 | 0.767 | 0.025 | 0.918 |
| 4 | L_dwbcel + L_dwioul + L_apl | **0.861** | **0.772** | **0.025** | **0.927** |

**Loss components:**
- **DWBCEL** — Dynamic Weighted Binary Cross-Entropy Loss: assigns larger weights to boundary/structure pixels.
- **DWIOUL** — Dynamic Weighted Intersection-over-Union Loss: explores global structural information, improves boundary accuracy and region continuity.
- **APL** — Auxiliary Prior Loss: penalizes predictions near 0.5 (uncertain pixels), forcing decision confidence.

**Key observations:**
- Adding both DWBCEL and DWIOUL (row 3) improves all metrics over either alone, confirming the value of **differentiated dynamic weighting**.
- Adding APL (row 4) gives the final boost, especially on Em (+0.009), demonstrating that **prior knowledge** strongly helps with blurred/challenging pixels (visualized in Fig. 12).
- This full loss (row 4) is the configuration used in the final SALK-Net model.
