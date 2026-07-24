"""
Phase 3.0 — Standalone Inference Script for RN50_ZoomNeXt
Run: python infer.py --image <path> --weight <path>

This is a minimal, dependency-light script that:
- Auto-detects CPU/GPU
- Loads the RN50_ZoomNeXt model + pretrained weight
- Runs inference on a single image
- Saves the prediction mask + a side-by-side visualization

Note: This is a Phase 3.0 verification script. A more complete version
(infer_single.py with timing instrumentation) will be developed in Phase 4.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# Add the ZoomNeXt repo to path (must be first so its methods/utils take priority)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.join(PROJECT_ROOT, "repository", "ZoomNeXt")
sys.path.insert(0, REPO_DIR)
# Change to repo dir so relative imports inside the repo work
os.chdir(REPO_DIR)

import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize


def main():
    parser = argparse.ArgumentParser(description="ZoomNeXt single-image inference")
    parser.add_argument("--image", required=True, help="Path to input image (.jpg or .png)")
    parser.add_argument("--weight", required=True, help="Path to pretrained weight (.pth)")
    parser.add_argument("--model-name", default="RN50_ZoomNeXt", help="Model name (default: RN50_ZoomNeXt)")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "images", "output"),
                        help="Output directory (default: <project>/images/output)")
    parser.add_argument("--device", default=None, help="Force device (default: auto: cuda if available else cpu)")
    args = parser.parse_args()

    # Resolve relative paths against the project root (PROJECT_ROOT defined at module level)
    if not os.path.isabs(args.image):
        args.image = os.path.join(PROJECT_ROOT, args.image)
    if not os.path.isabs(args.weight):
        args.weight = os.path.join(PROJECT_ROOT, args.weight)

    # 1. Device auto-detection (solves Risk R1)
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    # 2. Build model (pretrained=False to avoid backbone download)
    print(f"[INFO] Building model: {args.model_name}")
    model = model_zoo.__dict__[args.model_name](
        num_frames=1, pretrained=False, use_checkpoint=False
    )
    n_params = sum(v.numel() for v in model.parameters())
    print(f"[INFO] Parameters: {n_params:,} ({n_params/1e6:.3f}M)")

    # 3. Load weight (map_location="cpu" is CPU-safe)
    print(f"[INFO] Loading weight: {args.weight}")
    assert os.path.exists(args.weight), f"Weight not found: {args.weight}"
    load_weight(model=model, load_path=args.weight, strict=True)
    print("[INFO] Weight loaded successfully (strict=True).")

    model = model.to(device).eval()

    # 4. Read + preprocess image
    assert args.image.endswith((".jpg", ".png")), f"Image must be .jpg or .png: {args.image}"
    image = cv2.imread(args.image)
    assert image is not None, f"Cannot read image: {args.image}"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    print(f"[INFO] Input image: {args.image} ({orig_w}x{orig_h})")

    base_h, base_w = 384, 384
    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
    image_s = torch.from_numpy(images[0]).div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    image_m = torch.from_numpy(images[1]).div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    image_l = torch.from_numpy(images[2]).div(255).permute(2, 0, 1).unsqueeze(0).to(device)

    # 5. Inference + timing
    print("[INFO] Running inference...")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data={"image_s": image_s, "image_m": image_m, "image_l": image_l})
    t1 = time.perf_counter()
    inference_time = t1 - t0
    print(f"[INFO] Inference time: {inference_time:.3f} s")

    # 6. Post-process (same as Evaluator.eval in main_for_image.py)
    probs = logits.sigmoid().squeeze().cpu().numpy()
    probs = probs - probs.min()
    probs = probs / (probs.max() + 1e-8)
    pred = resize(probs, height=orig_h, width=orig_w)
    print(f"[INFO] Prediction range: [{pred.min():.4f}, {pred.max():.4f}], mean: {pred.mean():.4f}")

    # 7. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    pred_uint8 = (pred * 255).astype(np.uint8)
    pred_path = os.path.join(args.output_dir, f"{stem}_pred.png")
    cv2.imwrite(pred_path, pred_uint8)
    print(f"[INFO] Saved prediction: {pred_path}")

    # Visualization: input | heatmap prediction
    pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis = np.hstack([input_bgr, pred_color])
    vis_path = os.path.join(args.output_dir, f"{stem}_vis.png")
    cv2.imwrite(vis_path, vis)
    print(f"[INFO] Saved visualization: {vis_path}")

    print(f"\n[DONE] Device={device} | Params={n_params/1e6:.3f}M | Time={inference_time:.3f}s")


if __name__ == "__main__":
    main()
