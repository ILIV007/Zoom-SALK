"""Phase 3.2.5.2 — Full COD10K-Test Inference with Cumulative Metrics"""
import sys, os, time, json
sys.path.insert(0, "/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")
os.chdir("/home/z/my-project/research-project/ZoomNeXt_Project/repository/ZoomNeXt")
import torch, cv2, numpy as np
import methods as model_zoo
from utils.io import load_weight
from utils.ops import ms_resize, resize
import py_sod_metrics

WEIGHT = "/home/z/my-project/research-project/ZoomNeXt_Project/weights/resnet50-zoomnext.pth"
IMG_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset/COD10K/Imgs"
GT_DIR = "/home/z/my-project/research-project/ZoomNeXt_Project/dataset/COD-TestDataset/COD10K/GT"
OUT = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs/logs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
STATUS = os.path.join(OUT, "run_status.json")
FINAL = os.path.join(OUT, "metrics_final.json")

imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
print(f"Device: {DEV}, Images: {len(imgs)}", flush=True)

model = model_zoo.__dict__["RN50_ZoomNeXt"](num_frames=1, pretrained=False)
load_weight(model=model, load_path=WEIGHT, strict=True)
model = model.to(DEV).eval()

sm_m = py_sod_metrics.Smeasure()
wfm_m = py_sod_metrics.WeightedFmeasure()
mae_m = py_sod_metrics.MAE()
em_m = py_sod_metrics.Emeasure()
times = []
t0_all = time.perf_counter()

for i, name in enumerate(imgs):
    img = cv2.imread(os.path.join(IMG_DIR, name))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(os.path.join(GT_DIR, name.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
    h, w = rgb.shape[:2]
    ms = ms_resize(rgb, scales=(0.5, 1.0, 1.5), base_h=384, base_w=384)
    s = torch.from_numpy(ms[0]).div(255).permute(2,0,1).unsqueeze(0).to(DEV)
    m = torch.from_numpy(ms[1]).div(255).permute(2,0,1).unsqueeze(0).to(DEV)
    l = torch.from_numpy(ms[2]).div(255).permute(2,0,1).unsqueeze(0).to(DEV)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(data={"image_s": s, "image_m": m, "image_l": l})
    times.append(time.perf_counter() - t0)
    p = logits.sigmoid().squeeze().cpu().numpy()
    p = (p - p.min()) / (p.max() + 1e-8)
    p = resize(p, height=h, width=w)
    pu = (p * 255).astype(np.uint8)
    gb = (gt > 127).astype(np.uint8) * 255
    sm_m.step(pred=pu, gt=gb); wfm_m.step(pred=pu, gt=gb)
    mae_m.step(pred=pu, gt=gb); em_m.step(pred=pu, gt=gb)
    if (i+1) % 50 == 0:
        el = time.perf_counter() - t0_all
        rate = (i+1)/el; eta = (len(imgs)-i-1)/rate
        with open(STATUS, "w") as f: json.dump({"done": i+1, "total": len(imgs), "eta": eta, "rate": rate}, f)
        print(f"  [{i+1}/{len(imgs)}] {rate:.1f}/s ETA {eta:.0f}s", flush=True)

sm = sm_m.get_results()["sm"]; wfm = wfm_m.get_results()["wfm"]
mae = mae_m.get_results()["mae"]; em = em_m.get_results()["em"]["adp"]
if isinstance(wfm, float) and wfm != wfm: wfm = 0.0
el = time.perf_counter() - t0_all
final = {"Sm": float(sm), "Fw": float(wfm), "MAE": float(mae), "Em": float(em),
         "images": len(imgs), "device": DEV, "total_time": el, "avg_time": float(np.mean(times)),
         "paper": {"Sm": 0.861, "Fw": 0.768, "MAE": 0.026, "Em": 0.926}}
with open(FINAL, "w") as f: json.dump(final, f, indent=2)
with open(STATUS, "w") as f: json.dump({"done": len(imgs), "total": len(imgs), "eta": 0, "complete": True}, f)
print(f"\nDONE: {len(imgs)} imgs in {el:.0f}s")
print(f"Sm={sm:.4f} Fw={wfm:.4f} MAE={mae:.4f} Em={em:.4f}")
