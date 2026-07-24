# Phase 4.1 — Modification Proposal

## Title: Test-Time Augmentation (TTA) for ZoomNeXt Inference

## 1. Motivation (from Phase 4.0 Failure Analysis)

The failure analysis of 390 COD10K-Test images revealed:
- **19 failure cases (4.9%)**, dominated by **complex background (52.6%)** and **thin objects (26.3%)**
- Root cause: **boundary precision** and **prediction noise** (many failures have near-zero Fw, meaning low confidence rather than confident errors)
- The model is not architecturally broken — it's uncertain on hard cases

**Key insight**: if the model's predictions are noisy (low confidence) rather than systematically wrong, averaging multiple predictions (from augmented inputs) should reduce noise and improve boundary precision.

## 2. Proposed Modification

### Name: Test-Time Augmentation with Horizontal Flip

### Description:
At inference time, instead of running the model once on the input image, run it **twice**:
1. **Forward pass on original image** → prediction P₁
2. **Forward pass on horizontally-flipped image** → prediction P₂, then flip P₂ back horizontally → P₂'
3. **Final prediction** = average(P₁, P₂')

### Why horizontal flip?
- COD10K images are natural scenes where left-right symmetry is common (animals don't have a preferred orientation in the dataset)
- Horizontal flip is the safest augmentation (no geometric distortion, no color change)
- It's a standard TTA technique used in saliency detection and segmentation literature

## 3. Relationship to ZoomNeXt Paper

The ZoomNeXt paper does NOT use TTA at inference time. The paper's reported metrics (Sm=0.861, Fw=0.768, MAE=0.026, Em=0.926) are from single-pass inference without augmentation.

This modification is **additive** — it does not change the model architecture, weights, or training. It only changes the inference procedure. This is explicitly allowed by the project rules:
- ✅ No architecture change
- ✅ No weight modification  
- ✅ No retraining
- ✅ Original code preserved (new script only)

TTA is a well-established technique in computer vision, used by:
- **PSPNet** (Zhao et al., CVPR 2017) — multi-scale + flip TTA
- **DeepLabv3+** (Chen et al., 2018) — flip TTA for semantic segmentation
- **SINet-V2** (Fan et al., 2021) — the COD baseline itself uses multi-scale testing in some experiments

## 4. Implementation Plan

### Structure:
```
ZoomNeXt_Project/
├── repository/ZoomNeXt/          # Original code (untouched)
├── cpu_demo/
│   ├── infer.py                  # Original single-pass inference (baseline)
│   └── infer_tta.py              # NEW: TTA inference (2-pass, horizontal flip)
```

### infer_tta.py key changes from infer.py:
```python
# Original (infer.py):
logits = model(data={"image_s": s, "image_m": m, "image_l": l})
probs = logits.sigmoid()

# Modified (infer_tta.py):
# Pass 1: original
logits1 = model(data={"image_s": s, "image_m": m, "image_l": l})
probs1 = logits1.sigmoid()

# Pass 2: horizontal flip
s_flip = torch.flip(s, dims=[3])
m_flip = torch.flip(m, dims=[3])
l_flip = torch.flip(l, dims=[3])
logits2 = model(data={"image_s": s_flip, "image_m": m_flip, "image_l": l_flip})
probs2 = logits2.sigmoid()
probs2 = torch.flip(probs2, dims=[3])  # flip back

# Average
probs = (probs1 + probs2) / 2.0
```

### Cost:
- **2× inference time** (two forward passes per image)
- **No additional memory** (same model, sequential passes)
- **No additional parameters** (0 new parameters)

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Slower inference (2×) | LOW | Acceptable for research/demo; GPU mitigates this |
| No improvement | LOW | TTA is well-proven in literature; even 0.5% improvement is reportable |
| Worse on some images | VERY LOW | Averaging never makes predictions worse in expectation; worst case = no change |
| Breaks original code | ZERO | New file only; original infer.py untouched |

## 6. Expected Metric Improvement

Based on TTA literature and the failure analysis:

| Metric | Baseline (390 img) | Expected with TTA | Expected Δ |
|--------|-------------------|-------------------|------------|
| S-measure (Sm) | 0.8484 | 0.850–0.855 | +0.2–0.7% |
| Weighted F-measure (Fw) | 0.7597 | 0.765–0.770 | +0.5–1.0% |
| MAE | 0.0369 | 0.034–0.036 | -3–8% |
| E-measure (Em) | 0.8976 | 0.900–0.903 | +0.2–0.5% |

**Rationale for expectations:**
- Sm and Em are structural measures — TTA helps with boundary precision
- Fw is most sensitive to prediction confidence — averaging increases confidence
- MAE should decrease because averaging reduces over-prediction noise
- The improvement will be larger on the 19 failure cases (where noise reduction matters most)

## 7. Evaluation Plan

1. Run `infer_tta.py` on the same 390 images (or subset) used for baseline
2. Compute Sm, Fw, MAE, Em with the same pysodmetrics
3. Compare against baseline (Phase 3.2.5 results)
4. Analyze: which failure cases improved? Did any get worse?
5. Report: per-image delta + aggregate delta

## 8. Scientific Defensibility

This modification is scientifically defensible because:
- **Rooted in failure analysis**: the proposal directly addresses the identified weakness (prediction noise on hard cases)
- **Standard technique**: TTA is widely used and well-documented in CV literature
- **Controlled experiment**: single-variable change (TTA on/off), same model, same weights, same data
- **Measurable**: before/after metrics provide clear quantitative evidence
- **Reversible**: just use infer.py instead of infer_tta.py to revert

---

## Approval Request

This proposal is submitted for approval before implementation (Phase 4.2). 

**If approved**, the implementation will:
1. Create `cpu_demo/infer_tta.py` (new file, original code untouched)
2. Run on 50-image subset for quick verification
3. Run on full available set for final metrics
4. Generate Phase 4.3 Evaluation Report + Phase 4.4 Ablation Study

---

*Phase 4.1 — Modification Proposal | 2026-07-22*
