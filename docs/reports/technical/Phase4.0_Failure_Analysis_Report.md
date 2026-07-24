# Phase 4.0 — Failure Analysis Report

## Overview

Based on the Phase 3.2.5 benchmark (390 COD10K-Test images, RN50_ZoomNeXt), the model achieves:
- **S-measure (Sm): 0.8484** (mean), std=0.1434
- **Weighted F-measure (Fw): 0.7597**
- **MAE: 0.0369**
- **E-measure (Em): 0.8976**

**Success rate: 95.1%** (371/390 images with Sm ≥ 0.5)

## Failure Cases (Sm < 0.5): 19 images (4.9%)

### Categorization

| Category | Count | % of failures | Key animals |
|----------|-------|---------------|-------------|
| **Complex background** | 10 | 52.6% | SeaHorse (5), Crab (2), ScorpionFish (1), Shrimp (1), Fish (1) |
| **Thin object** | 5 | 26.3% | Pipefish (5) — thin, elongated structures |
| **Low contrast** | 3 | 15.8% | Stingaree (2), Turtle (1) — buried in sand |
| **Heavy camouflage** | 1 | 5.3% | Octopus (1) — extreme texture match |

### Root Cause Analysis

1. **Complex background (52.6%)** — The dominant failure mode. The model gets confused when the background has texture similar to the object. SeaHorse is the worst: 5 failures out of 55 images (9.1% failure rate). The SeaHorse's curled body shape and tail create complex boundary structures that the 384×384 resolution cannot resolve.

2. **Thin objects (26.3%)** — All 5 failures are Pipefish. At 384×384 base resolution, a thin pipefish (often <10px wide after resizing) is too narrow for the model's receptive field to detect reliably. The model either misses the entire fish or produces fragmented predictions.

3. **Low contrast (15.8%)** — Stingaree and Turtle buried in sand. The object-sand boundary is so subtle that the model's prediction collapses to near-zero. Stingaree-1198 has Sm=0.285 (the worst single image).

4. **Heavy camouflage (5.3%)** — Only Octopus-454. The octopus texture matches the reef so perfectly that even humans struggle. This is a genuinely hard sample, not a model deficiency.

### Per-Animal Performance (worst to best)

| Animal | Count | Avg Sm | Min Sm | Failure Rate |
|--------|-------|--------|--------|--------------|
| Pagurian | 6 | 0.714 | 0.501 | 0% (but lowest avg) |
| LeafySeaDragon | 5 | 0.737 | 0.596 | 0% |
| Stingaree | 9 | 0.740 | 0.285 | 22.2% |
| Turtle | 4 | 0.786 | 0.325 | 25.0% |
| Shrimp | 12 | 0.808 | 0.492 | 8.3% |
| Pipefish | 141 | 0.840 | 0.432 | 3.5% |
| SeaHorse | 55 | 0.855 | 0.425 | 9.1% |
| Octopus | 21 | 0.896 | 0.428 | 4.8% |
| BatFish | 4 | 0.925 | 0.892 | 0% |

### Key Findings

1. **Boundary precision is the #1 weakness**: most failures involve objects with complex or thin boundaries (SeaHorse, Pipefish, Crab), not objects that are completely invisible.

2. **Prediction noise**: many failure cases have near-zero Fw (e.g., Pipefish-487 has Fw=0.0001), meaning the model's confidence is very low — it's not confidently wrong, it's uncertain.

3. **Orientation sensitivity**: thin objects (Pipefish) and asymmetric objects (SeaHorse) may benefit from test-time augmentation with horizontal flips.

4. **Resolution limitation**: 384×384 base resolution is insufficient for thin structures. However, increasing resolution would require retraining (out of scope).

### Implications for Phase 4.1 Modification

The failure analysis suggests that a **Test-Time Augmentation (TTA)** approach could help:
- Horizontal flip + average would reduce prediction noise (helps with low-confidence cases)
- Multi-scale TTA could help with thin objects (different scales capture different details)
- TTA is a standard, non-architectural, no-retraining technique — perfect for this project's constraints

---

*Phase 4.0 — Failure Analysis Report | 2026-07-22*
