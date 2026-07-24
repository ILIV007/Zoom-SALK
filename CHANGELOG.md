# Changelog

## v3.0.4 — ۱۴۰۴/۰۵/۰۳

### Fixed — Notebook (numpy 2.x support)
- **Cell 1 (Install)**: removed the `numpy==1.26.4` pin. Modern torch (2.11+ with CUDA 12.8)
  supports numpy 2.x natively. Forcing numpy<2 was breaking the Colab scientific stack.
  Now only installs timm, einops, gdown — lets numpy stay at Colab default (2.x).
- **Cell 10 (Optional Eval)**: fixed `numpy.dtype size changed` binary incompatibility error.
  Root cause: scikit-image 0.22.0 was pinned but compiled against numpy 1.x, then numpy 2.x
  was active → Cython binary mismatch. Fix: install LATEST scipy/scikit-image/scikit-learn
  (all numpy-2 compatible). pysodmetrics installed with `--no-deps` to avoid opencv 5.x
  version conflict (the code works fine with opencv 5.x).

### Changed — Deliverables (font + figures + layout)
- **Font migration**: Tahoma → Vazirmatn (modern open-source Persian font). Tahoma was not
  installed on the system, causing LibreOffice to fall back to DejaVu Sans (poor Persian
  glyphs, disconnected letters). Vazirmatn is now installed system-wide and used in all
  docx + pptx files. Persian letters now render with correct cursive joining.
- **Embedded SALK-Net figures** in University_Report_FA.docx (6 figures):
  - Fig 1 (Attention Effects) — section 2 (problem definition)
  - Fig 3 (Overall Framework) — section 4 (SALK-Net architecture)
  - Fig 5 (DLKA Structure) — section 4.2
  - Fig 8 (CIEM) — section 4.4
  - Fig 9 (PR Curves) — section 9 (benchmark results)
  - Fig 10 (Visual Comparisons) — section 13 (qualitative analysis)
- **Slide 5 redesigned**: compact innovation cards (TE/DLKA/APL) on top half +
  SALK-Net architecture figure (Fig 3) on bottom half.
- All content reviewed for accuracy and SALK-Net focus.

## v3.0.3 — ۱۴۰۴/۰۵/۰۳
- Initial SALK-Net focus (removed ZoomNeXt from deliverables), RTL bidi tags, 10-slide pptx
- Notebook Cell 1 + Cell 10 attempted fix with numpy pinning (INCORRECT — see v3.0.4)

## v3.0.2 — ۱۴۰۴/۰۵/۰۲
- Purpose-based structure

## v3.0.1 — ۱۴۰۴/۰۵/۰۲
- Reproducibility docs

## v3.0.0 — ۱۴۰۴/۰۵/۰۱
- Final frozen results + TTA
