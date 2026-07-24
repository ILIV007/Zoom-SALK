"""
Phase 4.2 — Test-Time Augmentation (TTA) Inference for RN50_ZoomNeXt

Modification: horizontal flip + averaging (2-pass inference).
Original infer.py is NOT modified. This is a new file.

Usage:
  python infer_tta.py --image <path> --weight <path>
  python infer_tta.py --batch <img_dir> --weight <path> --gt-dir <gt_dir>
"""
import argparse, os, sys, time, json
import torch, cv2, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.join(PROJECT_ROOT, "repository", "ZoomNeXt")
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize

def run_tta_inference(model, image_rgb, device, base_h=384, base_w=384):
    """Run TTA inference: original + horizontal flip, averaged."""
    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
    s = torch.from_numpy(images[0]).div(255).permute(2,0,1).unsqueeze(0).to(device)
    m = torch.from_numpy(images[1]).div(255).permute(2,0,1).unsqueeze(0).to(device)
    l = torch.from_numpy(images[2]).div(255).permute(2,0,1).unsqueeze(0).to(device)

    # Pass 1: original
    with torch.no_grad():
        logits1 = model(data={"image_s": s, "image_m": m, "image_l": l})
    probs1 = logits1.sigmoid()

    # Pass 2: horizontal flip
    s_f = torch.flip(s, dims=[3])
    m_f = torch.flip(m, dims=[3])
    l_f = torch.flip(l, dims=[3])
    with torch.no_grad():
        logits2 = model(data={"image_s": s_f, "image_m": m_f, "image_l": l_f})
    probs2 = logits2.sigmoid()
    probs2 = torch.flip(probs2, dims=[3])  # flip prediction back

    # Average
    probs = (probs1 + probs2) / 2.0
    return probs.squeeze().cpu().numpy()

def run_single_inference(model, image_rgb, device, base_h=384, base_w=384):
    """Original single-pass inference (baseline)."""
    images = ms_resize(image_rgb, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)
    s = torch.from_numpy(images[0]).div(255).permute(2,0,1).unsqueeze(0).to(device)
    m = torch.from_numpy(images[1]).div(255).permute(2,0,1).unsqueeze(0).to(device)
    l = torch.from_numpy(images[2]).div(255).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(data={"image_s": s, "image_m": m, "image_l": l})
    return logits.sigmoid().squeeze().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="ZoomNeXt TTA inference")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--batch", help="Directory of images for batch processing")
    parser.add_argument("--gt-dir", help="GT mask directory (for batch metrics)")
    parser.add_argument("--weight", required=True, help="Weight file path")
    parser.add_argument("--model-name", default="RN50_ZoomNeXt")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "images", "output_tta"))
    parser.add_argument("--limit", type=int, default=None, help="Limit batch size")
    parser.add_argument("--mode", choices=["tta", "baseline", "both"], default="both",
                        help="tta=TTA only, baseline=original only, both=run both for comparison")
    args = parser.parse_args()

    if not os.path.isabs(args.weight):
        args.weight = os.path.join(PROJECT_ROOT, args.weight)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Mode: {args.mode}")

    # Build model
    model = model_zoo.__dict__[args.model_name](num_frames=1, pretrained=False, use_checkpoint=False)
    n_params = sum(v.numel() for v in model.parameters())
    print(f"[INFO] Parameters: {n_params:,} ({n_params/1e6:.3f}M)")
    load_weight(model=model, load_path=args.weight, strict=True)
    model = model.to(device).eval()

    # Single image mode
    if args.image:
        if not os.path.isabs(args.image):
            args.image = os.path.join(PROJECT_ROOT, args.image)
        image = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        if args.mode in ("tta", "both"):
            t0 = time.perf_counter()
            probs_tta = run_tta_inference(model, image_rgb, device)
            tta_time = time.perf_counter() - t0
            probs_tta = (probs_tta - probs_tta.min()) / (probs_tta.max() + 1e-8)
            pred_tta = resize(probs_tta, height=orig_h, width=orig_w)
            print(f"[TTA] Inference: {tta_time:.3f}s | range [{pred_tta.min():.4f}, {pred_tta.max():.4f}]")

        if args.mode in ("baseline", "both"):
            t0 = time.perf_counter()
            probs_base = run_single_inference(model, image_rgb, device)
            base_time = time.perf_counter() - t0
            probs_base = (probs_base - probs_base.min()) / (probs_base.max() + 1e-8)
            pred_base = resize(probs_base, height=orig_h, width=orig_w)
            print(f"[BASE] Inference: {base_time:.3f}s | range [{pred_base.min():.4f}, {pred_base.max():.4f}]")

        # Save
        stem = os.path.splitext(os.path.basename(args.image))[0]
        if args.mode in ("tta", "both"):
            cv2.imwrite(os.path.join(args.output_dir, f"{stem}_tta_pred.png"), (pred_tta * 255).astype(np.uint8))
        if args.mode in ("baseline", "both"):
            cv2.imwrite(os.path.join(args.output_dir, f"{stem}_base_pred.png"), (pred_base * 255).astype(np.uint8))
        print(f"[DONE] Outputs saved to {args.output_dir}")
        return

    # Batch mode
    if args.batch:
        import py_sod_metrics
        if not os.path.isabs(args.batch):
            args.batch = os.path.join(PROJECT_ROOT, args.batch)
        img_names = sorted([f for f in os.listdir(args.batch) if f.endswith((".jpg", ".png"))])
        if args.limit:
            img_names = img_names[:args.limit]
        print(f"[INFO] Processing {len(img_names)} images...")

        results = []
        sm_base = py_sod_metrics.Smeasure() if args.mode in ("baseline", "both") else None
        wfm_base = py_sod_metrics.WeightedFmeasure() if args.mode in ("baseline", "both") else None
        mae_base = py_sod_metrics.MAE() if args.mode in ("baseline", "both") else None
        em_base = py_sod_metrics.Emeasure() if args.mode in ("baseline", "both") else None
        sm_tta = py_sod_metrics.Smeasure() if args.mode in ("tta", "both") else None
        wfm_tta = py_sod_metrics.WeightedFmeasure() if args.mode in ("tta", "both") else None
        mae_tta = py_sod_metrics.MAE() if args.mode in ("tta", "both") else None
        em_tta = py_sod_metrics.Emeasure() if args.mode in ("tta", "both") else None

        t_start = time.perf_counter()
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(args.batch, img_name)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image_rgb.shape[:2]

            # Baseline
            if args.mode in ("baseline", "both"):
                t0 = time.perf_counter()
                probs_b = run_single_inference(model, image_rgb, device)
                base_time = time.perf_counter() - t0
                probs_b = (probs_b - probs_b.min()) / (probs_b.max() + 1e-8)
                pred_b = resize(probs_b, height=orig_h, width=orig_w)
                pred_b_u8 = (pred_b * 255).astype(np.uint8)
            else:
                base_time = 0

            # TTA
            if args.mode in ("tta", "both"):
                t0 = time.perf_counter()
                probs_t = run_tta_inference(model, image_rgb, device)
                tta_time = time.perf_counter() - t0
                probs_t = (probs_t - probs_t.min()) / (probs_t.max() + 1e-8)
                pred_t = resize(probs_t, height=orig_h, width=orig_w)
                pred_t_u8 = (pred_t * 255).astype(np.uint8)
            else:
                tta_time = 0

            # Metrics
            entry = {"image": img_name, "base_time": round(base_time, 3), "tta_time": round(tta_time, 3)}
            if args.gt_dir:
                gt_path = os.path.join(args.gt_dir, img_name.replace(".jpg", ".png"))
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt_bin = (gt > 127).astype(np.uint8) * 255
                    if args.mode in ("baseline", "both"):
                        s = py_sod_metrics.Smeasure(); s.step(pred=pred_b_u8, gt=gt_bin)
                        w = py_sod_metrics.WeightedFmeasure(); w.step(pred=pred_b_u8, gt=gt_bin)
                        m = py_sod_metrics.MAE(); m.step(pred=pred_b_u8, gt=gt_bin)
                        e = py_sod_metrics.Emeasure(); e.step(pred=pred_b_u8, gt=gt_bin)
                        entry["base_sm"] = round(float(s.get_results()["sm"]), 4)
                        entry["base_wfm"] = round(float(w.get_results()["wfm"]), 4)
                        entry["base_mae"] = round(float(m.get_results()["mae"]), 4)
                        entry["base_em"] = round(float(e.get_results()["em"]["adp"]), 4)
                        # Also cumulative
                        sm_base.step(pred=pred_b_u8, gt=gt_bin)
                        wfm_base.step(pred=pred_b_u8, gt=gt_bin)
                        mae_base.step(pred=pred_b_u8, gt=gt_bin)
                        em_base.step(pred=pred_b_u8, gt=gt_bin)
                    if args.mode in ("tta", "both"):
                        s = py_sod_metrics.Smeasure(); s.step(pred=pred_t_u8, gt=gt_bin)
                        w = py_sod_metrics.WeightedFmeasure(); w.step(pred=pred_t_u8, gt=gt_bin)
                        m = py_sod_metrics.MAE(); m.step(pred=pred_t_u8, gt=gt_bin)
                        e = py_sod_metrics.Emeasure(); e.step(pred=pred_t_u8, gt=gt_bin)
                        entry["tta_sm"] = round(float(s.get_results()["sm"]), 4)
                        entry["tta_wfm"] = round(float(w.get_results()["wfm"]), 4)
                        entry["tta_mae"] = round(float(m.get_results()["mae"]), 4)
                        entry["entry_tta_em"] = round(float(e.get_results()["em"]["adp"]), 4)
                        sm_tta.step(pred=pred_t_u8, gt=gt_bin)
                        wfm_tta.step(pred=pred_t_u8, gt=gt_bin)
                        mae_tta.step(pred=pred_t_u8, gt=gt_bin)
                        em_tta.step(pred=pred_t_u8, gt=gt_bin)

            results.append(entry)
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  [{i+1}/{len(img_names)}] {elapsed:.0f}s elapsed", flush=True)

        # Final aggregate metrics
        summary = {"images": len(results), "device": device, "mode": args.mode}
        if args.gt_dir:
            if args.mode in ("baseline", "both"):
                summary["baseline"] = {
                    "Sm": float(sm_base.get_results()["sm"]),
                    "Fw": float(wfm_base.get_results()["wfm"]),
                    "MAE": float(mae_base.get_results()["mae"]),
                    "Em": float(em_base.get_results()["em"]["adp"]),
                }
            if args.mode in ("tta", "both"):
                summary["tta"] = {
                    "Sm": float(sm_tta.get_results()["sm"]),
                    "Fw": float(wfm_tta.get_results()["wfm"]),
                    "MAE": float(mae_tta.get_results()["mae"]),
                    "Em": float(em_tta.get_results()["em"]["adp"]),
                }

        # Save
        out_json = os.path.join(args.output_dir, "tta_comparison_results.json")
        with open(out_json, "w") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)

        print(f"\n{'='*60}")
        print(f"TTA COMPARISON COMPLETE ({len(results)} images)")
        print(f"{'='*60}")
        if "baseline" in summary and "tta" in summary:
            b = summary["baseline"]; t = summary["tta"]
            print(f"  {'Metric':10s} {'Baseline':>10s} {'TTA':>10s} {'Delta':>10s}")
            for m in ["Sm", "Fw", "MAE", "Em"]:
                d = t[m] - b[m]
                print(f"  {m:10s} {b[m]:10.4f} {t[m]:10.4f} {d:+10.4f}")
        elif "baseline" in summary:
            b = summary["baseline"]
            print(f"  Baseline: Sm={b['Sm']:.4f} Fw={b['Fw']:.4f} MAE={b['MAE']:.4f} Em={b['Em']:.4f}")
        elif "tta" in summary:
            t = summary["tta"]
            print(f"  TTA: Sm={t['Sm']:.4f} Fw={t['Fw']:.4f} MAE={t['MAE']:.4f} Em={t['Em']:.4f}")
        print(f"\nResults: {out_json}")

if __name__ == "__main__":
    main()
