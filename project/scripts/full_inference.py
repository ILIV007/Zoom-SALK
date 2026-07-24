"""
Phase 3.2.2 — Full COD10K-Test Inference Pipeline (with checkpoint/resume)

Runs RN50_ZoomNeXt on ALL 2026 COD10K-Test images.
- Saves prediction masks to outputs/predictions/
- Saves a checkpoint file so it can resume if interrupted
- Logs timing + progress to outputs/logs/
- Does NOT save visualizations for all (only metrics + a few viz later)

Usage:
  python full_inference.py              # runs/resumes
  python full_inference.py --reset      # clears checkpoint, starts over
"""
import sys, os, time, json, argparse
sys.path.insert(0, "/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")
os.chdir("/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")

import torch, cv2, numpy as np
import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize

# === Config ===
WEIGHT_PATH = "/home/z/my-project/research-project/ZoomNeXt_Project/weights/resnet50-zoomnext.pth"
DATASET_ROOT = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset"
IMG_DIR = os.path.join(DATASET_ROOT, "COD10K", "Imgs")
GT_DIR = os.path.join(DATASET_ROOT, "COD10K", "GT")
OUTPUT_BASE = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs"
PRED_DIR = os.path.join(OUTPUT_BASE, "predictions")
LOG_DIR = os.path.join(OUTPUT_BASE, "logs")
CHECKPOINT = os.path.join(LOG_DIR, "checkpoint.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start over")
parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for testing)")
args = parser.parse_args()

# === Load checkpoint ===
if args.reset and os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
    print("Checkpoint cleared.")

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    processed = set(ckpt.get("processed", []))
    print(f"Resuming: {len(processed)} images already processed.")
else:
    processed = set()

# === Build model ===
print(f"Device: {DEVICE}")
print("Building RN50_ZoomNeXt...")
model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False, use_checkpoint=False)
load_weight(model=model, load_path=WEIGHT_PATH, strict=True)
model = model.to(DEVICE).eval()
print("Model ready.")

# === Get image list ===
all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
if args.limit:
    all_images = all_images[:args.limit]
# Also skip if prediction file already exists (more robust than checkpoint alone)
def pred_exists(fname):
    stem = os.path.splitext(fname)[0]
    return os.path.exists(os.path.join(PRED_DIR, f"{stem}.png"))

todo = [f for f in all_images if f not in processed and not pred_exists(f)]
# Reconcile processed set with actual files on disk
for f in all_images:
    if pred_exists(f):
        processed.add(f)
print(f"Total: {len(all_images)} | Already done: {len(processed)} | To do: {len(todo)}")

# === Inference loop ===
total_time = 0.0
times_log = []
start_time = time.perf_counter()

for idx, img_name in enumerate(todo):
    img_path = os.path.join(IMG_DIR, img_name)
    gt_path = os.path.join(GT_DIR, img_name.replace(".jpg", ".png"))
    stem = os.path.splitext(img_name)[0]

    # Read image
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # Multi-scale preprocess
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
    times_log.append({"image": img_name, "time": infer_time})

    # Post-process
    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = (probs - probs.min()) / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    pred_uint8 = (pred * 255).astype(np.uint8)

    # Save prediction
    cv2.imwrite(os.path.join(PRED_DIR, f"{stem}.png"), pred_uint8)

    # Update checkpoint
    processed.add(img_name)
    if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
        with open(CHECKPOINT, "w") as f:
            json.dump({"processed": sorted(processed), "count": len(processed)}, f)
        elapsed = time.perf_counter() - start_time
        done = len(processed)
        remaining = len(all_images) - done
        eta = (elapsed / (done - (len(processed) - len(todo)))) * remaining if done > 0 else 0
        print(f"  [{done}/{len(all_images)}] {img_name} | {infer_time:.3f}s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

# === Final log ===
total_elapsed = time.perf_counter() - start_time
avg_time = total_time / len(todo) if todo else 0
final_log = {
    "total_images": len(all_images),
    "processed": len(processed),
    "device": DEVICE,
    "total_inference_time": total_time,
    "total_elapsed_time": total_elapsed,
    "avg_inference_time": avg_time,
    "per_image_times": times_log[-100:],  # last 100
}
with open(os.path.join(LOG_DIR, "inference_log.json"), "w") as f:
    json.dump(final_log, f, indent=2)

print(f"\n{'='*60}")
print(f"INFERENCE COMPLETE")
print(f"{'='*60}")
print(f"  Images processed: {len(processed)}/{len(all_images)}")
print(f"  Total inference time: {total_time:.1f}s")
print(f"  Total elapsed time: {total_elapsed:.1f}s")
print(f"  Avg inference/image: {avg_time:.3f}s")
print(f"  Device: {DEVICE}")
print(f"  Predictions saved: {PRED_DIR}")
print(f"  Log saved: {LOG_DIR}/inference_log.json")
print(f"{'='*60}")
