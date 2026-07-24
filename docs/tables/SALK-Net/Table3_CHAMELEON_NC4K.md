# Table 3 — Comparison with 12 SOTA methods on CHAMELEON and NC4K

**Source:** SALK-Net paper, Page 11 (Table 3)

**Note:** CHAMELEON and NC4K are **untrained** datasets (used only for testing → generalization evaluation).

| Module | Params | CHAMELEON Sm ↑ | CHAMELEON Fω ↑ | CHAMELEON M ↓ | CHAMELEON Em ↑ | NC4K Sm ↑ | NC4K Fω ↑ | NC4K M ↓ | NC4K Em ↑ |
|--------|--------|----------------|----------------|---------------|----------------|-----------|-----------|----------|------------|
| ITSD | 17.082M | 0.814 | 0.662 | 0.030 | 0.898 | 0.811 | 0.679 | 0.064 | 0.855 |
| LSR | 50.935M | 0.890 | 0.831 | 0.025 | 0.936 | 0.840 | 0.766 | 0.048 | 0.904 |
| PFNet | 46.498M | 0.882 | 0.810 | 0.033 | 0.942 | 0.829 | 0.745 | 0.053 | 0.894 |
| PraNet | 30.528M | 0.860 | 0.763 | 0.044 | 0.943 | 0.822 | 0.724 | 0.059 | 0.875 |
| SINet | 48.947M | 0.871 | 0.804 | 0.034 | 0.937 | 0.808 | 0.723 | 0.058 | 0.883 |
| MGL-R | 63.595M | 0.893 | 0.812 | 0.031 | 0.941 | 0.833 | 0.739 | 0.053 | 0.893 |
| UJSC | 217.982M | 0.891 | 0.833 | 0.030 | 0.943 | 0.842 | 0.771 | 0.047 | 0.906 |
| UGTR | 48.868M | 0.888 | 0.824 | 0.027 | 0.950 | 0.848 | 0.776 | 0.044 | 0.901 |
| SINet-V2 | 24.932M | 0.888 | 0.816 | 0.030 | 0.929 | 0.847 | 0.769 | 0.048 | 0.904 |
| FEDER | 44.129M | 0.894 | 0.834 | 0.028 | 0.942 | 0.846 | 0.788 | 0.045 | 0.912 |
| EAMNet | 26.824M | 0.901 | 0.847 | 0.024 | 0.955 | 0.862 | 0.801 | 0.040 | 0.916 |
| ZoomNext | 28.458M | 0.912 | 0.863 | 0.020 | 0.969 | 0.869 | 0.808 | 0.038 | 0.925 |
| **SALK-Net (Ours)** | **34.882M** | **0.906** | **0.858** | **0.021** | **0.962** | **0.872** | **0.817** | **0.037** | **0.926** |

**Key observations:**
- On NC4K (largest untrained test set), SALK-Net is best on Sm (0.872), Fω (0.817), Em (0.926), and M (0.037).
- These NC4K numbers correspond to the abstract's claim: "0.872 and 0.926 respectively in the untrained large dataset".
- On CHAMELEON, ZoomNext slightly edges SALK-Net on a few metrics (Sm 0.912 vs 0.906, Fω 0.863 vs 0.858, M 0.020 vs 0.021, Em 0.969 vs 0.962), showing SALK-Net is not strictly dominant on the smallest dataset, but it leads on the large NC4K benchmark which is more statistically reliable.
