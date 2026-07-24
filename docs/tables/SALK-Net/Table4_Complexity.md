# Table 4 — Model Complexity and Inference Performance (COD10K)

**Source:** SALK-Net paper, Page 12 (Table 4)

| Method | SALK-Net (Ours) | LSR | UJSC | PFNet | UGTR | ZoomNext |
|--------|-----------------|-----|------|-------|------|----------|
| Params | 34.882M | 50.935M | 217.982M | 46.498M | 48.868M | 28.458M |
| FLOPs | 91.801G | 66.625G | 112.341G | 53.222G | 1.007T | 185.794G |
| FPS | 27.689 | 38.184 | 29.222 | 29.916 | 11.108 | 22.108 |
| Fω ↑ | **0.772** | 0.673 | 0.684 | 0.660 | 0.728 | 0.758 |

**Key observations (quoted / derived from the paper):**
- SALK-Net's FLOPs (91.801G) are **only about half** of ZoomNext's (185.794G), while achieving a higher Fω (0.772 vs 0.758).
- Compared to the Transformer-based UGTR, SALK-Net's FLOPs are **less than one-sixth** (91.801G vs 1.007T), with better Params, FPS, and Fω.
- LSR has the highest FPS (38.184) but a much lower Fω (0.673).
- SALK-Net therefore offers a strong accuracy/efficiency trade-off (near real-time inference at ~27.7 FPS).

**Note for the practical project:** FPS was measured on an NVIDIA RTX 3090 GPU. CPU inference will be substantially slower; this is a key constraint for the laptop demo (Phase 3+).
