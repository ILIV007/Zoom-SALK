# Table 2 — Comparison with 12 SOTA methods on CAMO and COD10K

**Source:** SALK-Net paper, Page 11 (Table 2)

**Metrics:** ↑ = higher is better, ↓ = lower is better. Bold = best.

| Module | Params | CAMO Sm ↑ | CAMO Fω ↑ | CAMO M ↓ | CAMO Em ↑ | COD10K Sm ↑ | COD10K Fω ↑ | COD10K M ↓ | COD10K Em ↑ |
|--------|--------|-----------|-----------|----------|-----------|-------------|-------------|------------|--------------|
| ITSD | 17.082M | 0.750 | 0.610 | 0.102 | 0.830 | 0.767 | 0.557 | 0.051 | 0.787 |
| LSR | 50.935M | 0.787 | 0.696 | 0.080 | 0.859 | 0.804 | 0.673 | 0.037 | 0.883 |
| PFNet | 46.498M | 0.782 | 0.695 | 0.085 | 0.855 | 0.800 | 0.660 | 0.040 | 0.868 |
| PraNet | 30.528M | 0.769 | 0.663 | 0.094 | 0.835 | 0.789 | 0.629 | 0.045 | 0.840 |
| SINet | 48.947M | 0.747 | 0.646 | 0.091 | 0.825 | 0.776 | 0.631 | 0.043 | 0.867 |
| MGL-R | 63.595M | 0.775 | 0.673 | 0.088 | 0.842 | 0.814 | 0.666 | 0.035 | 0.890 |
| UJSC | 217.982M | 0.800 | 0.728 | 0.073 | 0.872 | 0.809 | 0.684 | 0.035 | 0.882 |
| UGTR | 48.868M | 0.808 | 0.734 | 0.070 | 0.880 | 0.827 | 0.708 | 0.032 | 0.881 |
| SINet-V2 | 24.932M | 0.820 | 0.743 | 0.070 | 0.885 | 0.815 | 0.680 | 0.037 | 0.864 |
| FEDER | 44.129M | 0.807 | 0.737 | 0.069 | 0.876 | 0.823 | 0.715 | 0.037 | 0.901 |
| EAMNet | 26.824M | 0.831 | 0.763 | 0.064 | 0.890 | 0.839 | 0.733 | 0.029 | 0.907 |
| ZoomNext | 28.458M | 0.822 | 0.760 | 0.069 | 0.885 | 0.855 | 0.758 | 0.026 | 0.926 |
| **SALK-Net (Ours)** | **34.882M** | **0.849** | **0.796** | **0.055** | **0.907** | **0.861** | **0.772** | **0.025** | **0.927** |

**Key observations:**
- SALK-Net achieves the **best result on every metric** for both CAMO and COD10K.
- On COD10K: Sm = 0.861, Fω = 0.772, M = 0.025, Em = 0.927.
- These are the numbers explicitly quoted in the paper abstract: "structural similarity measure and enhanced alignment measure in a large trained dataset reached 0.861 and 0.927 respectively".
