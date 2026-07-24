# Table 5 — Ablation: Input Scale (COD10K)

**Source:** SALK-Net paper, Page 14 (Table 5)

| No. | 0.5× | 1.0× | 1.5× | Sm ↑ | Fω ↑ | M ↓ | Em ↑ |
|-----|------|------|------|------|------|-----|------|
| (1) | ✓ | | | 0.756 | 0.560 | 0.074 | 0.836 |
| (2) | | ✓ | | 0.802 | 0.651 | 0.061 | 0.862 |
| (3) | | | ✓ | 0.812 | 0.678 | 0.047 | 0.871 |
| (4) | ✓ | ✓ | ✓ | **0.821** | **0.688** | **0.041** | **0.878** |

**Key observations:**
- The mixed multi-scale scheme (all three scales) outperforms any single-scale scheme on every metric.
- The 1.5× scale alone (row 3) is better than 0.5× (row 1) and 1.0× (row 2), suggesting larger-scale/global structure is valuable.
- Per the paper text + Fig. 11: the 0.5× scale focuses on **texture details**, while the 1.5× scale focuses on **overall structure**; combined they provide **complementary** information to the primary 1.0× scale.
- This table is the empirical justification for the **ternary encoding** input design.
