"""
verify_env.py
=============
Run this in a Kaggle notebook cell BEFORE starting the backend server.
It checks all required secrets/env vars and tells you exactly what is
missing and why.

Usage (in a Kaggle notebook cell):
    %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/verify_env.py

Blur GPU install order (required on Kaggle T4):
    1. Notebook Accel = GPU T4 x2
    2. pip install -r requirements.txt -r requirements-blur.txt
    3. pip uninstall -y onnxruntime onnxruntime-gpu
    4. pip install onnxruntime-gpu
    5. export BLUR_DEVICE=cuda:0  (and BLUR_REQUIRE_GPU=1)
    6. This script must show CUDAExecutionProvider + torch.cuda True
    7. uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path

REQUIRED = {
    "ASSEMBLYAI_API_KEY": "AssemblyAI transcription (get at assemblyai.com)",
    "SUPABASE_URL": "Supabase project URL",
    "SUPABASE_SERVICE_ROLE_KEY": "Supabase service-role key (Settings > API)",
}

OPTIONAL = {
    "GOOGLE_API_KEY": "Google AI (used for LLM summaries)",
    "VITE_SUPABASE_ANON_KEY": "Supabase anon key (for frontend)",
}

BLUR_OPTIONAL = {
    "BLUR_YOLO_WEIGHTS": "Path to yolov8 .pt (default: auto-download yolov8n.pt)",
    "BLUR_FACE_WEIGHTS": "Optional dedicated face detector .pt (reference fallback)",
    "BLUR_DEVICE": "cuda:0 or cpu (default: auto; use cuda:0 on Kaggle)",
    "BLUR_REQUIRE_GPU": "1 to fail-fast without CUDA/ORT GPU (default on Kaggle)",
    "BLUR_TRACKER": "Ultralytics tracker yaml (default: botsort.yaml)",
    "BLUR_FACE_DETECT": "insightface|heuristic (default: insightface SCRFD on person crops)",
    "BLUR_MIN_FACE_PX": "Min face box side in px (default: 28)",
    "BLUR_MATCH_THRESHOLD": "Cosine match threshold (default: 0.78)",
    "BLUR_MAX_THROTTLE": "Max auto detect throttle (default: 4)",
    "BLUR_OPERATOR": "gaussian|mosaic (default: gaussian)",
    "BLUR_INSIGHTFACE_MODEL": "InsightFace pack name (default: buffalo_l)",
}


def _on_kaggle() -> bool:
    return bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or Path("/kaggle").exists()
    )


print("=" * 60)
print("  NeuroClip Environment Verification")
print("=" * 60)

# ── 1. Kaggle Secrets ──────────────────────────────────────────
print("\n[1] Kaggle Secrets:")
kaggle_ok = []
kaggle_fail = []
try:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    all_keys = list(REQUIRED.keys()) + list(OPTIONAL.keys())
    for key in all_keys:
        try:
            val = client.get_secret(key)
            if val:
                kaggle_ok.append(key)
                print(f"    ✓  {key}  (len={len(val)})")
            else:
                kaggle_fail.append(key)
                print(f"    ✗  {key}  — secret exists but is EMPTY")
        except Exception as e:
            kaggle_fail.append(key)
            print(f"    ✗  {key}  — {e}")
except ImportError:
    print("    [not on Kaggle — kaggle_secrets not available]")
except Exception as e:
    print(f"    ERROR initialising UserSecretsClient: {e}")

# ── 2. Current os.environ ─────────────────────────────────────
print("\n[2] Current environment variables (os.environ):")
for key in list(REQUIRED.keys()) + list(OPTIONAL.keys()):
    val = os.environ.get(key, "")
    if val:
        print(f"    ✓  {key}  (len={len(val)})")
    else:
        print(f"    ✗  {key}  — NOT SET")

# ── 3. Blur / GPU ─────────────────────────────────────────────
print("\n[3] Blur module & GPU:")
cuda_ok = False
ort_cuda_ok = False
blur_deps_ok = False

try:
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    print(f"    torch.cuda.is_available() = {cuda_ok}")
    if cuda_ok:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    ✗  CUDA not available — set Notebook Accel = GPU T4 x2")
except Exception as e:
    print(f"    torch not available: {e}")

try:
    import onnxruntime as ort

    providers = list(ort.get_available_providers())
    print(f"    ort providers: {providers}")
    ort_cuda_ok = "CUDAExecutionProvider" in providers
    if ort_cuda_ok:
        print("    ✓  CUDAExecutionProvider present")
    else:
        print(
            "    ✗  CUDAExecutionProvider MISSING — "
            "pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
        )
except Exception as e:
    print(f"    onnxruntime not available: {e}")

for key, hint in BLUR_OPTIONAL.items():
    val = os.getenv(key, "")
    if val:
        print(f"    ✓  {key} = {val}")
    else:
        print(f"    ·  {key}  — not set ({hint})")

blur_weights = os.getenv(
    "BLUR_YOLO_WEIGHTS",
    "/kaggle/input/neuroclip-blur-weights/yolov8n.pt",
)
if Path(blur_weights).exists():
    print(f"    ✓  BLUR weights file exists: {blur_weights}")
else:
    print(f"    ·  BLUR weights not at {blur_weights} (ultralytics will try yolov8n.pt)")

try:
    from ultralytics import YOLO  # noqa: F401

    print("    ✓  ultralytics import OK")
    blur_deps_ok = True
except ImportError:
    print("    ✗  ultralytics not installed — run: pip install -r requirements-blur.txt")

try:
    import insightface  # noqa: F401

    print("    ✓  insightface import OK")
    blur_deps_ok = True
except ImportError:
    print("    ·  insightface not installed (needed for SCRFD/ArcFace blur)")

# Hard fail on Kaggle when blur deps are present but GPU EP is missing
gpu_fail = False
if _on_kaggle() and blur_deps_ok:
    if not cuda_ok:
        gpu_fail = True
        print(
            "\n    ❌  FAIL: On Kaggle with blur deps but torch.cuda is False. "
            "Set Accel=GPU T4 x2 and restart kernel."
        )
    if not ort_cuda_ok:
        gpu_fail = True
        print(
            "\n    ❌  FAIL: On Kaggle with blur deps but CUDAExecutionProvider absent. "
            "pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu"
        )
    if not gpu_fail:
        print("\n    ✅  Kaggle GPU path OK for blur (torch CUDA + ORT CUDA EP)")

# ── 4. .env files ─────────────────────────────────────────────
print("\n[4] .env files found on disk:")
candidates = [
    Path("/kaggle/working/NeuroClip/backend/Semantic-search-app/.env"),
    Path("/kaggle/working/NeuroClip/backend/.env"),
    Path("/kaggle/working/.env"),
]
found_any = False
for p in candidates:
    if p.exists():
        found_any = True
        print(f"    ✓  {p}  ({p.stat().st_size} bytes)")
        for line in p.read_text(errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k = s.split("=", 1)[0].strip()
                print(f"         key: {k}")
    else:
        print(f"    ✗  {p}  — not found")

# ── 5. Verdict ────────────────────────────────────────────────
print("\n" + "=" * 60)
missing = [k for k in REQUIRED if not os.environ.get(k)]
fatal = bool(missing) or gpu_fail
if missing:
    print("  ❌  MISSING required vars — backend WILL fail:")
    for k in missing:
        print(f"       {k}  ({REQUIRED[k]})")
    print("\n  Fix options:")
    print("   A) Add to Kaggle Secrets (Notebook sidebar → Add-ons → Secrets)")
    print("      then RESTART the kernel so secrets become available.")
    print()
    print("   B) Create /kaggle/working/NeuroClip/backend/Semantic-search-app/.env")
    print("      with the missing keys and restart.")
elif gpu_fail:
    print("  ❌  GPU FAIL — blur will be unusable (~1h/clip on CPU). Fix [3] above.")
else:
    print("  ✅  All required vars are SET — backend should work.")
print("=" * 60)

if fatal:
    sys.exit(1)
