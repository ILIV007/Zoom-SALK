# Table 2 — Video COD: Comparisons with SOTA

**Source:** ZoomNeXt paper, Page 7 (Table 2)

"T" = number of frames in the video clip. T=1 = model without temporal difference (dynamic routing not activated). T=5 = final model for video COD.

| Model | Backbone | FPS | Params | CAD Sm | CAD Fω | CAD MAE | CAD mDice | CAD mIoU | MoCA-TE Sm | MoCA-TE Fω | MoCA-TE MAE | MoCA-TE mDice | MoCA-TE mIoU |
|-------|----------|-----|--------|--------|--------|---------|-----------|----------|------------|------------|-------------|---------------|---------------|
| SINet (2020) | ResNet-50 | 56.509 | 48.947M | 0.636 | 0.346 | 0.041 | 0.381 | 0.283 | 0.598 | 0.231 | 0.028 | 0.256 | 0.202 |
| SINet-V2 (2021) | Res2Net-50 | 38.307 | 26.976M | 0.653 | 0.382 | 0.039 | 0.413 | 0.318 | 0.588 | 0.204 | 0.031 | 0.229 | 0.180 |
| STL-Net-LT (2022) | PVTv2-B5 | 35.646 | 82.303M | 0.696 | 0.481 | 0.030 | 0.524 | 0.402 | 0.631 | 0.311 | 0.027 | 0.331 | 0.272 |
| STL-Net-ST (2022) | PVTv2-B5 | 6.488 | 82.383M | 0.696 | 0.471 | 0.031 | 0.515 | 0.392 | 0.637 | 0.304 | 0.027 | 0.328 | 0.271 |
| **ZoomNeXt (T=1)** | **PVTv2-B5** | **11.741** | **84.774M** | **0.721** | **0.525** | **0.024** | **0.567** | **0.436** | **0.690** | **0.395** | **0.017** | **0.424** | **0.353** |
| **ZoomNeXt (T=3)** | **PVTv2-B5** | **23.207** | **84.775M** | **0.741** | **0.565** | **0.021** | **0.598** | **0.487** | **0.703** | **0.425** | **0.010** | **0.444** | **0.378** |
| **ZoomNeXt (T=5)** | **PVTv2-B5** | **26.454** | **84.776M** | **0.757** | **0.593** | **0.020** | **0.631** | **0.510** | **0.734** | **0.476** | **0.010** | **0.497** | **0.422** |
| ZoomNeXt (T=7) | PVTv2-B5 | 30.292 | 84.778M | 0.751 | 0.583 | 0.020 | 0.618 | 0.505 | 0.706 | 0.428 | 0.011 | 0.444 | 0.382 |

**Key observations:**
- T=5 is the best configuration for video COD — performance improves from T=1 → T=5, then slightly degrades at T=7.
- ZoomNeXt (T=5) significantly outperforms STL-Net (the previous SOTA for video COD) on all metrics.
- The temporal conditional computation (difference-aware routing) is key: T=1 (no temporal) is already strong, but T=5 adds meaningful motion cues.
