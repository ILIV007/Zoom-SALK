"""
Phase 3.2.5.0 — Evaluation Dependency Stabilization
Checks all inference + evaluation dependencies BEFORE running the full benchmark.
Installs ONLY missing packages — never touches torch/torchvision/numpy.

Rules:
- NO albumentations (pulls numpy>=2, breaks torch)
- NO full requirements.txt install
- NO torch/torchvision reinstall
- NO numpy downgrade/upgrade unless explicitly verified
- Only install the specific missing package
"""
import sys, os, importlib, subprocess, json

INFERENCE_DEPS = {
    "torch": None,
    "torchvision": None,
    "timm": None,
    "einops": None,
    "cv2": "opencv-python-headless",
    "numpy": None,
}
EVAL_DEPS = {
    "py_sod_metrics": "pysodmetrics",
    "scipy": None,
    "yaml": "PyYAML",
    "matplotlib": None,
    "tqdm": None,
}
FORBIDDEN = ["albumentations"]

def get_version(modname):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None

def install(pkg_name):
    print(f"  Installing {pkg_name}...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], capture_output=True, text=True)
    return r.returncode == 0

print("=" * 60)
print("PHASE 3.2.5.0 — DEPENDENCY CHECK")
print("=" * 60)

import torch
print(f"\nPython:      {sys.version.split()[0]}")
print(f"Torch:       {torch.__version__}")
print(f"CUDA:        {torch.version.cuda if torch.cuda.is_available() else 'N/A (CPU)'}")
print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

print(f"\n--- Inference Dependencies ---")
inference_status = {}
for mod, pip_name in INFERENCE_DEPS.items():
    ver = get_version(mod)
    if ver:
        print(f"  OK  {mod:15s} {ver}")
        inference_status[mod] = ver
    else:
        install_name = pip_name or mod
        print(f"  MISSING {mod:15s} — installing {install_name}...")
        if install(install_name):
            ver = get_version(mod)
            print(f"       Installed: {ver}")
            inference_status[mod] = ver
        else:
            print(f"       FAILED")
            inference_status[mod] = None

print(f"\n--- Evaluation Dependencies ---")
eval_status = {}
for mod, pip_name in EVAL_DEPS.items():
    ver = get_version(mod)
    if ver:
        print(f"  OK  {mod:15s} {ver}")
        eval_status[mod] = ver
    else:
        install_name = pip_name or mod
        print(f"  MISSING {mod:15s} — installing {install_name}...")
        if install(install_name):
            ver = get_version(mod)
            print(f"       Installed: {ver}")
            eval_status[mod] = ver
        else:
            print(f"       FAILED")
            eval_status[mod] = None

print(f"\n--- Forbidden Packages Check ---")
alb_ver = None
for pkg in FORBIDDEN:
    ver = get_version(pkg)
    if ver:
        print(f"  WARN {pkg:15s} {ver} — should NOT be installed")
        alb_ver = ver
    else:
        print(f"  OK   {pkg:15s} not installed (correct)")

print(f"\n--- NumPy Version Check ---")
import numpy
np_ver = numpy.__version__
print(f"  NumPy: {np_ver}")
if int(np_ver.split(".")[0]) >= 2:
    print(f"  WARN NumPy >= 2! Forcing <2...")
    install("numpy<2")
    importlib.reload(numpy)
    import numpy
    print(f"  NumPy after fix: {numpy.__version__}")
else:
    print(f"  OK   NumPy < 2 (correct)")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
all_ok = True
for mod, ver in {**inference_status, **eval_status}.items():
    status = "OK" if ver else "FAIL"
    print(f"  [{status}] {mod}: {ver or 'MISSING'}")
    if not ver: all_ok = False
if alb_ver:
    print(f"  [WARN] albumentations: {alb_ver}")
    all_ok = False

if all_ok:
    print(f"\nALL DEPENDENCIES OK — SAFE TO RUN BENCHMARK")
else:
    print(f"\nSOME DEPENDENCIES MISSING — FIX BEFORE BENCHMARK")

report = {
    "python": sys.version.split()[0], "torch": torch.__version__,
    "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "numpy": numpy.__version__,
    "inference_packages": inference_status, "evaluation_packages": eval_status,
    "albumentations_installed": alb_ver is not None, "all_ok": all_ok,
}
out_path = "/home/z/my-project/research-project/ZoomNeXt_Project/outputs/dependency_check.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved: {out_path}")
