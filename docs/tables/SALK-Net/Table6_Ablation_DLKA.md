# Table 6 — Ablation: DLKA Effectiveness (COD10K)

**Source:** SALK-Net paper, Page 15 (Table 6)

DLKA = Dynamic Large Kernel Attention. Applied in EASPP and/or SFAM.

| No. | DLKA in ASPP | DLKA in SFAM | Sm ↑ | Fω ↑ | M ↓ | Em ↑ |
|-----|--------------|--------------|------|------|-----|------|
| I | | | 0.821 | 0.688 | 0.041 | 0.878 |
| II | ✓ | | 0.827 | 0.712 | 0.036 | 0.877 |
| III | | ✓ | 0.832 | 0.728 | 0.035 | 0.887 |
| IV | ✓ | ✓ | **0.837** | **0.729** | **0.032** | **0.894** |

**Key observations (from the paper text):**
- Both II and III improve over the baseline (I); using DLKA in SFAM (III) gives a slightly larger gain than in ASPP (II).
- Combining DLKA in both modules (IV) gives the best result: Sm +1.6%, Fω +4.1%, Em +1.6%, M −0.9% relative to baseline.
- This validates the paper's claim that **DLKA's receptive field expansion** significantly improves model performance.
