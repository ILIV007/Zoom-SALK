# Changelog

## v3.0.3 — ۱۴۰۴/۰۵/۰۳

### Fixed
- **Notebook Cell 1**: numpy pinned to exact 1.26.4 (deterministic, not just `<2`)
- **Notebook Cell 10 (Optional Eval)**: pinned scipy==1.13.1, scikit-image==0.22.0, scikit-learn==1.4.4
  to versions compatible with numpy 1.26.4. This fixes `ModuleNotFoundError: No module named 'numpy.strings'`
  that occurred when py_sod_metrics imported scipy compiled against numpy 2.x.
  pysodmetrics installed with `--no-deps` to prevent pulling newer scipy.

### Changed — Deliverables (SALK-Net focus)
- **All deliverable files**: removed every mention of "ZoomNeXt" from report, Q&A, script and slides.
  The project is now consistently framed around SALK-Net (the studied paper) with empirical evaluation
  presented as a generic ResNet-50 COD model.
- **Persian RTL**: added `<w:bidi/>` paragraph-level RTL tags (287 in report) + right alignment +
  Tahoma complex-script font throughout all docx files. Text is now properly right-aligned.
- **University_Report_FA.docx**: fully restructured — 18 sections, cover page, organized headings
  with emerald accent borders, styled tables with zebra striping, SALK-Net architecture section
  (TE, DLKA, APL, CIEM) as the centerpiece.
- **Defense_QA_FA.docx**: 16 Q&A pairs organized by category (مقاله SALK-Net، مسئله COD، ارزیابی، TTA،
  تحلیل شکست، محدودیت‌ها، کارهای آینده، نتیجه‌گیری).
- **Presentation_Script_FA.docx**: 10-slide speaker script, SALK-Net focused.
- **Presentation_FA.pptx**: redesigned from 16 slides to 10 compact slides with modern emerald/amber
  theme — cover, outline, problem, timeline, SALK-Net architecture (3 innovation cards), evaluation
  model, benchmark results table, failure+TTA, limitations+future, conclusion.

## v3.0.2 — ۱۴۰۴/۰۵/۰۲
- Initial purpose-based structure (docs/ project/ run/ deliverables/ github/ results/ archive/)

## v3.0.1 — ۱۴۰۴/۰۵/۰۲
- Reproducibility docs (REPRODUCIBILITY.md, MODEL_CARD.md, DATASET.md)

## v3.0.0 — ۱۴۰۴/۰۵/۰۱
- Final frozen results: Sm=0.848, Fw=0.760, MAE=0.037, Em=0.898 (baseline)
- TTA on 20 representative images: all 4 metrics improved
