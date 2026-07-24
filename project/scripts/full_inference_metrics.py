"""
Phase 3.2.2+3.2.3 — Full COD10K Inference + Metrics (combined, checkpointed)
Optimized: no per-image PNG saving (metrics computed on-the-fly).
"""
import sys, os, time, json
sys.path.insert(0, "/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")
os.chdir("/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")

import torch, cv2, numpy as np
import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize
import py_sod_metrics

WEIGHT_PATH = "/home/z/my-project/research-project/ZoomNeXt_Project/weights/resnet50-zoomnext.pth"
DATASET_ROOT = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset"
IMG_DIR = os.path.join(DATASET_ROOT, "COD10K", "Imgs")
GT_DIR = os.path.join(DATASET_ROOT, "COD10K", "GT")
OUTPUT_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CKPT_FILE = os.path.join(LOG_DIR, "full_metrics_checkpoint.json")
PER_IMAGE_FILE = os.path.join(LOG_DIR, "per_image_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(LOG_DIR, exist_ok=True)

# === Load checkpoint ===
all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
start_idx = 0
per_image_results = []

if os.path.exists(CKPT_FILE):
    with open(CKPT_FILE) as f:
        ckpt = json.load(f)
    start_idx = ckpt.get("next_idx", 0)
    per_image_results = ckpt.get("per_image_results", [])
    print(f"Resuming from index {start_idx} ({len(per_image_results)} done)")

if start_idx >= len(all_images):
    print("Already complete. Run metrics summary.")
    sys.exit(0)

# === Build model ===
print(f"Device: {DEVICE}")
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False, use_checkpoint=False)
load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
model = model.to(DEVICE).eval()
print("Model ready.")

# === Metrics (cumulative for Sm, MAE, Em; per-image list for wfm) ===
sm_metric = py_sod_metrics.Smeasure()
mae_metric = py_sod_metrics.MAE()
em_metric = py_sod_metrics.Emeasure()
wfm_values = []

# === Inference loop ===
batch_start = time.perf_counter()
todo = all_images[start_idx:]
print(f"Processing {len(todo)} images (index {start_idx} to {len(all_images)-1})...")

for i, img_name in enumerate(todo):
    idx = start_idx + i
    img_path = os.path.join(IMG_DIR, img_name)
    gt_path = os.path.join(GT_DIR, img_name.replace(".jpg", ".png"))

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_rgb.shape[:2]

    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=384, base_w=384)
    image_s = torch.from_numpy(images[0]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    image_m = torch.from_numpy(images[1]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    image_l = torch.from_numpy(images[2]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data={"image_s": image_s, "image_m": image_m, "image_l": image_l})
    t1 = time.perf_counter()
    infer_time = t1 - t0

    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = (probs - probs.min()) / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    pred_uint8 = (pred * 255).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8) * 255

    # Metrics
    sm_metric.step(pred=pred_uint8, gt=gt_bin)
    mae_metric.step(pred=pred_uint8, gt=gt_bin)
    em_metric.step(pred=pred_uint8, gt=gt_bin)
    w_single = py_sod_metrics.WeightedFmeasure()
    w_single.step(pred=pred_uint8, gt=gt_bin)
    wfm_val = float(w_single.get_results()["wfm"])
    if not np.isnan(wfm_val):
        wfm_values.append(wfm_val)

    per_image_results.append({
        "image": img_name,
        "infer_time": round(infer_time, 3),
        "wfm": round(wfm_val, 4) if not np.isnan(wfm_val) else None,
        "pred_mean": round(float(pred.mean()), 4),
    })

    # Checkpoint every 50 images
    if (i + 1) % 10 == 0:
        with open(CKPT_FILE, "w") as f:
            json.dump({"next_idx": idx + 1, "per_image_results": per_image_results}, f)
        elapsed = time.perf_counter() - batch_start
        total_done = len(per_image_results)
        remaining = len(all_images) - total_done
        rate = (i + 1) / elapsed
        eta = remaining / rate if rate > 0 else 0
        print(f"  [{total_done}/{len(all_images)}] {img_name} | {infer_time:.3f}s | rate {rate:.1f} img/s | ETA {eta:.0f}s ({eta/60:.1f}m)")

# === Final checkpoint + save ===
with open(CKPT_FILE, "w") as f:
    json.dump({"next_idx": len(all_images), "per_image_results": per_image_results}, f)

# Compute final aggregate metrics
sm = sm_metric.get_results()["sm"]
mae = mae_metric.get_results()["mae"]
em = em_metric.get_results()["em"]["adp"]
wfm = np.mean(wfm_values) if wfm_values else 0.0

total_elapsed = time.perf_counter() - batch_start
avg_time = sum(r["infer_time"] for r in per_image_results) / len(per_image_results)

# Save per-image results
with open(PER_IMAGE_FILE, "w") as f:
    json.dump(per_image_results, f, indent=2)

# Save summary
summary = {
    "dataset": "COD10K-Test",
    "total_images": len(per_image_results),
    "device": DEVICE,
    "total_inference_time": sum(r["infer_time"] for r in per_image_results),
    "avg_inference_time": avg_time,
    "batch_elapsed_time": total_elapsed,
    "metrics": {"sm": sm, "wfm": wfm, "mae": mae, "em": em},
    "paper_metrics": {"sm": 0.861, "wfm": 0.768, "mae": 0.026, "em": 0.926},
}
with open(os.path.join(LOG_DIR, "metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"FULL COD10K-Test INFERENCE + METRICS COMPLETE")
print(f"{'='*60}")
print(f"  Images: {len(per_image_results)}")
print(f"  Avg inference: {avg_time:.3f}s/image")
print(f"  S-measure:     {sm:.4f}  (paper: 0.861)")
print(f"  Weighted F:    {wfm:.4f}  (paper: 0.768)")
print(f"  MAE:           {mae:.4f}  (paper: 0.026)")
print(f"  E-measure:     {em:.4f}  (paper: 0.926)")
print(f"{'='*60}")
