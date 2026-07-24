"""
Phase 3.2.1 — Dataset Validation Script
Checks COD10K-Test integrity BEFORE running heavy inference.
No processing — only integrity check.
"""
import os, sys, json, time
import cv2

DATASET_ROOT = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset"
DATASET_NAME = "COD10K"
IMG_DIR = os.path.join(DATASET_ROOT, DATASET_NAME, "Imgs")
GT_DIR = os.path.join(DATASET_ROOT, DATASET_NAME, "GT")

print("=" * 60)
print("PHASE 3.2.1 — COD10K-Test Dataset Validation")
print("=" * 60)

t0 = time.perf_counter()

# 1. Check directories exist
print("\n[1] Directory structure check:")
for d, label in [(IMG_DIR, "Imgs"), (GT_DIR, "GT")]:
    exists = os.path.isdir(d)
    print(f"  {label}: {d} — {'OK' if exists else 'MISSING'}")
    assert exists, f"Directory missing: {d}"

# 2. List all files
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".png")])
print(f"\n[2] File counts:")
print(f"  Images (.jpg): {len(img_files)}")
print(f"  GT masks (.png): {len(gt_files)}")

# 3. Name matching (image stem should match GT stem)
img_stems = set(os.path.splitext(f)[0] for f in img_files)
gt_stems = set(os.path.splitext(f)[0] for f in gt_files)
matched = img_stems & gt_stems
img_only = img_stems - gt_stems
gt_only = gt_stems - img_stems
print(f"\n[3] Name matching:")
print(f"  Matched pairs: {len(matched)}")
print(f"  Images without GT: {len(img_only)}")
print(f"  GT without images: {len(gt_only)}")
if img_only:
    print(f"    Examples (img only): {sorted(img_only)[:5]}")
if gt_only:
    print(f"    Examples (gt only): {sorted(gt_only)[:5]}")

# 4. Check for corrupt/unreadable files (sample first 50 + all names)
print(f"\n[4] File readability check (sampling 50 + checking all sizes):")
corrupt = []
zero_size = []
for i, fname in enumerate(img_files):
    img_path = os.path.join(IMG_DIR, fname)
    gt_path = os.path.join(GT_DIR, fname.replace(".jpg", ".png"))
    # Size check (all files)
    if os.path.getsize(img_path) < 100:
        zero_size.append(("image", fname))
    if os.path.exists(gt_path) and os.path.getsize(gt_path) < 100:
        zero_size.append(("gt", fname))
    # Read check (sample 50 spread across the list)
    if i % max(1, len(img_files) // 50) == 0:
        img = cv2.imread(img_path)
        if img is None:
            corrupt.append(("image", fname))
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            corrupt.append(("gt", fname))

print(f"  Checked {len(img_files)} files (sizes) + 50 sampled reads")
print(f"  Zero-size files: {len(zero_size)}")
print(f"  Corrupt (unreadable) files: {len(corrupt)}")
if zero_size:
    print(f"    Zero-size: {zero_size[:5]}")
if corrupt:
    print(f"    Corrupt: {corrupt[:5]}")

# 5. Resolution statistics (sample 20)
print(f"\n[5] Resolution statistics (sample 20):")
resolutions = []
for i in range(0, len(img_files), max(1, len(img_files) // 20)):
    img = cv2.imread(os.path.join(IMG_DIR, img_files[i]))
    if img is not None:
        h, w = img.shape[:2]
        resolutions.append((w, h))
if resolutions:
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    print(f"  Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")

# 6. GT mask value check (binary?)
print(f"\n[6] GT mask format check (sample 10):")
gt_binary = 0
gt_checked = 0
for i in range(0, len(gt_files), max(1, len(gt_files) // 10)):
    gt = cv2.imread(os.path.join(GT_DIR, gt_files[i]), cv2.IMREAD_GRAYSCALE)
    if gt is not None:
        unique = set(np.unique(gt).tolist()) if (np := __import__('numpy')) else set()
        gt_checked += 1
        if unique <= {0, 255}:
            gt_binary += 1
print(f"  Checked {gt_checked} GT masks")
print(f"  Binary (0/255 only): {gt_binary}/{gt_checked}")

t1 = time.perf_counter()
print(f"\n[7] Validation time: {t1-t0:.2f}s")

# 7. Final verdict
usable = len(matched) - len(corrupt)
print(f"\n{'='*60}")
print(f"VALIDATION RESULT")
print(f"{'='*60}")
print(f"  Total images:    {len(img_files)}")
print(f"  Total GT:        {len(gt_files)}")
print(f"  Matched pairs:   {len(matched)}")
print(f"  Corrupt files:   {len(corrupt)}")
print(f"  Zero-size files: {len(zero_size)}")
print(f"  USABLE IMAGES:   {usable}")
print(f"  VERDICT: {'PASS — safe to run full inference' if usable == len(matched) and len(corrupt) == 0 else 'ISSUES FOUND — review above'}")
print(f"{'='*60}")

# Save validation report
report = {
    "dataset": "COD10K-Test",
    "validation_time": t1 - t0,
    "total_images": len(img_files),
    "total_gt": len(gt_files),
    "matched_pairs": len(matched),
    "images_without_gt": len(img_only),
    "gt_without_images": len(gt_only),
    "corrupt_files": len(corrupt),
    "zero_size_files": len(zero_size),
    "usable_images": usable,
    "verdict": "PASS" if usable == len(matched) and len(corrupt) == 0 else "ISSUES",
    "resolution_sample": {"min_w": min(widths) if resolutions else 0, "max_w": max(widths) if resolutions else 0},
}
out_path = "/home/z/my-project/research-project/Phase3.2_Output/dataset_validation.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nValidation report saved: {out_path}")
