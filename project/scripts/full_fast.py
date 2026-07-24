"""
Phase 3.2.2+3.2.3 — Full COD10K Inference + Metrics (optimized, cumulative wfm)
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
IMG_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset/COD10K/Imgs"
GT_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset/COD10K/GT"
LOG_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs/logs"
CKPT_FILE = os.path.join(LOG_DIR, "ckpt_fast.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(LOG_DIR, exist_ok=True)

all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

# Resume
start_idx = 0
if os.path.exists(CKPT_FILE):
    with open(CKPT_FILE) as f:
        start_idx = json.load(f).get("next_idx", 0)
    print(f"Resuming from index {start_idx}")

# Build model
print(f"Device: {DEVICE}")
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False)
load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
model = model.to(DEVICE).eval()

# Cumulative metrics (fast — no per-image object creation)
sm_metric = py_sod_metrics.Smeasure()
wfm_metric = py_sod_metrics.WeightedFmeasure()
mae_metric = py_sod_metrics.MAE()
em_metric = py_sod_metrics.Emeasure()

# If resuming, we need to re-process from 0 to start_idx to rebuild metric state
# (can't serialize pysodmetrics state). So we DON'T resume metrics — only index.
# For a clean run, we always start from 0 for metrics.
if start_idx > 0:
    print(f"Re-processing first {start_idx} images to rebuild metric state (no inference re-run needed for log)...")
    # Actually, we can't skip — we must re-run inference for metric accumulation.
    # So reset to 0.
    start_idx = 0
    print("Resetting to 0 for metric consistency.")

todo = all_images[start_idx:]
print(f"Processing {len(todo)} images...")

per_image = []
t_batch = time.perf_counter()

for i, img_name in enumerate(todo):
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
    infer_time = t1 - t0

    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = (probs - probs.min()) / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    pred_uint8 = (pred * 255).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8) * 255

    sm_metric.step(pred=pred_uint8, gt=gt_bin)
    wfm_metric.step(pred=pred_uint8, gt=gt_bin)
    mae_metric.step(pred=pred_uint8, gt=gt_bin)
    em_metric.step(pred=pred_uint8, gt=gt_bin)

    per_image.append({"image": img_name, "t": round(infer_time, 3), "pm": round(float(pred.mean()), 4)})

    if (i + 1) % 100 == 0:
        with open(CKPT_FILE, "w") as f:
            json.dump({"next_idx": i + 1}, f)
        elapsed = time.perf_counter() - t_batch
        rate = (i + 1) / elapsed
        eta = (len(todo) - i - 1) / rate
        print(f"  [{i+1}/{len(todo)}] {rate:.1f} img/s | ETA {eta:.0f}s ({eta/60:.1f}m)")

# Final results
sm = sm_metric.get_results()["sm"]
wfm = wfm_metric.get_results()["wfm"]
mae = mae_metric.get_results()["mae"]
em = em_metric.get_results()["em"]["adp"]
if isinstance(wfm, float) and wfm != wfm:
    wfm = 0.0

avg_t = sum(r["t"] for r in per_image) / len(per_image)
summary = {
    "total_images": len(per_image),
    "device": DEVICE,
    "avg_inference_time": avg_t,
    "total_inference_time": sum(r["t"] for r in per_image),
    "metrics": {"sm": float(sm), "wfm": float(wfm), "mae": float(mae), "em": float(em)},
    "paper": {"sm": 0.861, "wfm": 0.768, "mae": 0.026, "em": 0.926},
}
with open(os.path.join(LOG_DIR, "metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(LOG_DIR, "per_image_results.json"), "w") as f:
    json.dump(per_image, f)

print(f"\n{'='*60}")
print(f"COMPLETE: {len(per_image)} images")
print(f"  Sm={sm:.4f} (paper 0.861) | Fw={wfm:.4f} (paper 0.768)")
print(f"  MAE={mae:.4f} (paper 0.026) | Em={em:.4f} (paper 0.926)")
print(f"  Avg: {avg_t:.3f}s/img")
print(f"{'='*60}")
