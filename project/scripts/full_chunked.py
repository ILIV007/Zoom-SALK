"""
Phase 3.2.2+3.2.3 — Chunked full COD10K inference with per-image metrics.
Each run processes as many images as possible, saves results, exits.
Next run resumes from where it left off.
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
RESULTS_FILE = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs/logs/per_image_metrics.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

# Load existing results
results = []
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f"Resuming: {len(results)}/{len(all_images)} already done")

if len(results) >= len(all_images):
    print("Already complete!")
else:
    # Build model
    print(f"Device: {DEVICE}")
    model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False)
    load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
    model = model.to(DEVICE).eval()

    todo = all_images[len(results):]
    print(f"Processing {len(todo)} images (index {len(results)} to {len(all_images)-1})...")

    t_start = time.perf_counter()
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

        probs = logits.sigmoid().squeeze().cpu().numpy()
        probs = (probs - probs.min()) / (probs.max() + 1e-8)
        pred = resize(probs, height=orig_h, width=orig_w)
        pred_uint8 = (pred * 255).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8) * 255

        # Per-image metrics
        s = py_sod_metrics.Smeasure(); s.step(pred=pred_uint8, gt=gt_bin)
        w = py_sod_metrics.WeightedFmeasure(); w.step(pred=pred_uint8, gt=gt_bin)
        m = py_sod_metrics.MAE(); m.step(pred=pred_uint8, gt=gt_bin)
        e = py_sod_metrics.Emeasure(); e.step(pred=pred_uint8, gt=gt_bin)

        sm_val = float(s.get_results()["sm"])
        wfm_val = float(w.get_results()["wfm"])
        mae_val = float(m.get_results()["mae"])
        em_val = float(e.get_results()["em"]["adp"])
        if np.isnan(wfm_val): wfm_val = 0.0

        results.append({
            "image": img_name,
            "t": round(t1 - t0, 3),
            "sm": round(sm_val, 4),
            "wfm": round(wfm_val, 4),
            "mae": round(mae_val, 4),
            "em": round(em_val, 4),
            "pm": round(float(pred.mean()), 4),
        })

        # Save every 30 images
        if (i + 1) % 30 == 0:
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f)
            elapsed = time.perf_counter() - t_start
            done = len(results)
            rate = (i + 1) / elapsed
            remaining = len(all_images) - done
            eta = remaining / rate
            print(f"  [{done}/{len(all_images)}] {rate:.1f}/s | ETA {eta:.0f}s ({eta/60:.1f}m)")

    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)

# Compute aggregate
if len(results) >= len(all_images):
    sm_avg = np.mean([r["sm"] for r in results])
    wfm_avg = np.mean([r["wfm"] for r in results])
    mae_avg = np.mean([r["mae"] for r in results])
    em_avg = np.mean([r["em"] for r in results])
    avg_t = np.mean([r["t"] for r in results])

    summary = {
        "total_images": len(results),
        "device": DEVICE,
        "avg_inference_time": float(avg_t),
        "metrics": {"sm": float(sm_avg), "wfm": float(wfm_avg), "mae": float(mae_avg), "em": float(em_avg)},
        "paper": {"sm": 0.861, "wfm": 0.768, "mae": 0.026, "em": 0.926},
    }
    with open(os.path.join(os.path.dirname(RESULTS_FILE), "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} images")
    print(f"  Sm={sm_avg:.4f} (paper 0.861)")
    print(f"  Fw={wfm_avg:.4f} (paper 0.768)")
    print(f"  MAE={mae_avg:.4f} (paper 0.026)")
    print(f"  Em={em_avg:.4f} (paper 0.926)")
    print(f"  Avg: {avg_t:.3f}s/img")
    print(f"{'='*60}")
else:
    print(f"\nPartial: {len(results)}/{len(all_images)} done. Run again to continue.")
