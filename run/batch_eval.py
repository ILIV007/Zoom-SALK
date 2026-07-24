"""
Phase 3.1 — Batch Inference + Real Metric Computation on COD10K-Test

Runs RN50_ZoomNeXt on a subset of COD10K-Test, computes real metrics
(Sm, Fw, MAE, Em), and saves visualizations (Input | GT | Prediction).
"""
import sys, os, time, json
sys.path.insert(0, "/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")
os.chdir("/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")

import torch, cv2, numpy as np
import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize
import py_sod_metrics

# === Config ===
WEIGHT_PATH = "/home/z/my-project/research-project/ZoomNeXt_Project/weights/resnet50-zoomnext.pth"
DATASET_ROOT = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset"
OUTPUT_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/images/phase3.1"
NUM_IMAGES = 50  # subset for quick evaluation (full 2026 would take ~80s on GPU, ~2min on CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Build model + load weight ===
print(f"Device: {DEVICE}")
print("Building RN50_ZoomNeXt...")
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False, use_checkpoint=False)
n_params = sum(v.numel() for v in model.parameters())
print(f"Parameters: {n_params:,} ({n_params/1e6:.3f}M)")
load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
print("Weight loaded.")
model = model.to(DEVICE).eval()

# === Get image list from COD10K-Test ===
img_dir = os.path.join(DATASET_ROOT, "COD10K", "Imgs")
gt_dir = os.path.join(DATASET_ROOT, "COD10K", "GT")
image_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])[:NUM_IMAGES]
print(f"\nEvaluating on {len(image_names)} COD10K-Test images...")

# === Metric containers ===
sm_metric = py_sod_metrics.Smeasure()
mae_metric = py_sod_metrics.MAE()
em_metric = py_sod_metrics.Emeasure()
wfm_values = []  # compute per-image to avoid NaN contamination

# === Run batch inference ===
total_time = 0
results_log = []

for idx, img_name in enumerate(image_names):
    img_path = os.path.join(img_dir, img_name)
    gt_path = os.path.join(gt_dir, img_name.replace(".jpg", ".png"))

    # Read image + GT
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = image_rgb.shape[:2]

    # Preprocess (multi-scale)
    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=384, base_w=384)
    image_s = torch.from_numpy(images[0]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    image_m = torch.from_numpy(images[1]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    image_l = torch.from_numpy(images[2]).div(255).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data={"image_s": image_s, "image_m": image_m, "image_l": image_l})
    t1 = time.perf_counter()
    infer_time = t1 - t0
    total_time += infer_time

    # Post-process
    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = (probs - probs.min()) / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    pred_uint8 = (pred * 255).astype(np.uint8)

    # Binarize GT
    gt_bin = (gt > 127).astype(np.uint8) * 255

    # Update metrics
    sm_metric.step(pred=pred_uint8, gt=gt_bin)
    mae_metric.step(pred=pred_uint8, gt=gt_bin)
    em_metric.step(pred=pred_uint8, gt=gt_bin)
    # Compute wfm per-image to avoid NaN contamination from degenerate predictions
    wfm_single = py_sod_metrics.WeightedFmeasure()
    wfm_single.step(pred=pred_uint8, gt=gt_bin)
    wfm_val = float(wfm_single.get_results()["wfm"])
    if not np.isnan(wfm_val):
        wfm_values.append(wfm_val)

    results_log.append({
        "image": img_name,
        "inference_time": infer_time,
        "pred_mean": float(pred.mean()),
    })

    # Save visualization for first 8 images
    if idx < 8:
        pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
        input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gt_color = cv2.cvtColor(gt_bin, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([input_bgr, gt_color, pred_color])
        stem = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_vis3.png"), vis)

    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1}/{len(image_names)}] {img_name} | {infer_time:.3f}s")

# === Compute final metrics ===
print("\n" + "="*60)
print("REAL METRICS — RN50_ZoomNeXt on COD10K-Test (subset)")
print("="*60)
sm = sm_metric.get_results()["sm"]
wfm = np.mean(wfm_values) if wfm_values else 0.0
mae = mae_metric.get_results()["mae"]
em = em_metric.get_results()["em"]["adp"]
print(f"  Images evaluated:    {len(image_names)}")
print(f"  Total inference time: {total_time:.2f}s")
print(f"  Avg inference/image:  {total_time/len(image_names):.3f}s")
print(f"  Device:               {DEVICE}")
print(f"  Parameters:           {n_params:,} ({n_params/1e6:.3f}M)")
print(f"")
print(f"  S-measure (Sm):       {sm:.4f}   (paper: 0.861)")
print(f"  Weighted F-measure:   {wfm:.4f}   (paper: 0.768)")
print(f"  MAE:                  {mae:.4f}   (paper: 0.026)")
print(f"  E-measure (Em):       {em:.4f}   (paper: 0.926)")
print(f"")
print(f"  Paper values (COD10K, RN50_ZoomNeXt, 2026 images):")
print(f"    Sm=0.861, Fw=0.768, MAE=0.026, Em=0.926")
print(f"  Our values ({len(image_names)} image subset):")
print(f"    Sm={sm:.4f}, Fw={wfm:.4f}, MAE={mae:.4f}, Em={em:.4f}")
print("="*60)

# === Save results JSON ===
results = {
    "dataset": "COD10K-Test",
    "num_images": len(image_names),
    "device": DEVICE,
    "model": "RN50_ZoomNeXt",
    "parameters": n_params,
    "total_inference_time": total_time,
    "avg_inference_time": total_time / len(image_names),
    "metrics": {
        "sm": sm,
        "wfm": wfm,
        "mae": mae,
        "em": em,
    },
    "paper_metrics": {"sm": 0.861, "wfm": 0.768, "mae": 0.026, "em": 0.926},
    "per_image": results_log,
}
with open(os.path.join(OUTPUT_DIR, "metrics_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {OUTPUT_DIR}/metrics_results.json")
print(f"Visualizations saved: {len(os.listdir(OUTPUT_DIR)) - 1} files")
