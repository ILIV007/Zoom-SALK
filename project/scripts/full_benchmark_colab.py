"""
Phase 3.2 — Full COD10K Benchmark (Colab GPU version)
Run this on Google Colab with T4 GPU to evaluate ALL 2026 COD10K-Test images.

Expected runtime: ~80 seconds on T4 GPU (vs ~5.5 hours on CPU).

Upload this script to Colab or paste into a cell.
"""
import sys, os, time, json
sys.path.insert(0, "/content/ZoomNeXt")
os.chdir("/content/ZoomNeXt")

import torch, cv2, numpy as np
import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize
import py_sod_metrics

# === Config ===
WEIGHT_PATH = "/content/weights/resnet50-zoomnext.pth"
IMG_DIR = "/content/COD-TestDataset/COD10K/Imgs"
GT_DIR = "/content/COD-TestDataset/COD10K/GT"
OUTPUT_DIR = "/content/outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Build model ===
print(f"Device: {DEVICE}")
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False)
load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
model = model.to(DEVICE).eval()
print(f"Parameters: {sum(v.numel() for v in model.parameters()):,}")

# === Get all images ===
all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"Total COD10K-Test images: {len(all_images)}")

# === Metrics ===
sm_metric = py_sod_metrics.Smeasure()
wfm_metric = py_sod_metrics.WeightedFmeasure()
mae_metric = py_sod_metrics.MAE()
em_metric = py_sod_metrics.Emeasure()

per_image = []
t_start = time.perf_counter()

for i, img_name in enumerate(all_images):
    image = cv2.imread(os.path.join(IMG_DIR, img_name))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(GT_DIR, img_name.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_rgb.shape[:2]

    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=384, base_w=384)
    is_ = torch.from_numpy(images[0]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    im_ = torch.from_numpy(images[1]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    il_ = torch.from_numpy(images[2]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data={"image_s": is_, "image_m": im_, "image_l": il_})
    t1 = time.perf_counter()

    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = (probs - probs.min()) / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    pred_uint8 = (pred * 255).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8) * 255

    sm_metric.step(pred=pred_uint8, gt=gt_bin)
    wfm_metric.step(pred=pred_uint8, gt=gt_bin)
    mae_metric.step(pred=pred_uint8, gt=gt_bin)
    em_metric.step(pred=pred_uint8, gt=gt_bin)

    per_image.append({"image": img_name, "t": round(t1-t0, 3), "pm": round(float(pred.mean()),4)})

    if (i+1) % 200 == 0:
        elapsed = time.perf_counter() - t_start
        rate = (i+1) / elapsed
        eta = (len(all_images) - i - 1) / rate
        print(f"  [{i+1}/{len(all_images)}] {rate:.1f} img/s | ETA {eta:.0f}s")

# === Final metrics ===
sm = sm_metric.get_results()["sm"]
wfm = wfm_metric.get_results()["wfm"]
mae = mae_metric.get_results()["mae"]
em = em_metric.get_results()["em"]["adp"]
if isinstance(wfm, float) and wfm != wfm: wfm = 0.0

total_elapsed = time.perf_counter() - t_start
avg_t = np.mean([r["t"] for r in per_image])

summary = {
    "total_images": len(all_images),
    "device": DEVICE,
    "total_elapsed": total_elapsed,
    "avg_inference_time": float(avg_t),
    "metrics": {"sm": float(sm), "wfm": float(wfm), "mae": float(mae), "em": float(em)},
    "paper": {"sm": 0.861, "wfm": 0.768, "mae": 0.026, "em": 0.926},
}
with open(os.path.join(OUTPUT_DIR, "full_benchmark_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "full_per_image.json"), "w") as f:
    json.dump(per_image, f)

print(f"\n{'='*60}")
print(f"FULL COD10K-Test BENCHMARK COMPLETE ({DEVICE})")
print(f"{'='*60}")
print(f"  Images: {len(all_images)}")
print(f"  Total time: {total_elapsed:.1f}s")
print(f"  Avg/image: {avg_t:.3f}s")
print(f"  S-measure:  {sm:.4f}  (paper: 0.861)")
print(f"  Weighted F: {wfm:.4f}  (paper: 0.768)")
print(f"  MAE:        {mae:.4f}  (paper: 0.026)")
print(f"  E-measure:  {em:.4f}  (paper: 0.926)")
print(f"{'='*60}")
