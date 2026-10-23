"""
kaggle_setup_cell.py
====================
One-shot Kaggle setup AFTER the repo is already cloned to /kaggle/working/NeuroClip.

Usage (in a Kaggle GPU T4 notebook cell):

    %cd /kaggle/working/NeuroClip/backend/Semantic-search-app/backend
    %run kaggle_setup_cell.py

What it does:
  1. pip install requirements.txt + requirements-blur.txt
  2. Force-reinstall onnxruntime-gpu==1.19.2 (CUDA 12) — LAST, so InsightFace
     cannot leave you on CPU onnxruntime / CUDA-13 wheels
  3. Set BLUR_DEVICE / BLUR_REQUIRE_GPU
  4. Print provider probe (must include CUDAExecutionProvider)

Then RESTART KERNEL and start uvicorn + ngrok (see kaggle_deploy.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ORT_GPU_VERSION = "1.19.2"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    py = sys.executable
    os.chdir(BACKEND)

    print("=" * 60)
    print("  NeuroClip Kaggle blur setup")
    print("=" * 60)

    _run([py, "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")])
    _run([py, "-m", "pip", "install", "-r", str(BACKEND / "requirements-blur.txt")])

    # Always last: insightface / other deps often reinstall CPU onnxruntime
    _run([py, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"])
    _run([py, "-m", "pip", "install", f"onnxruntime-gpu=={ORT_GPU_VERSION}"])

    os.environ["BLUR_DEVICE"] = "cuda:0"
    os.environ["BLUR_REQUIRE_GPU"] = "1"

    probe = r"""
import os, torch, onnxruntime as ort
os.environ.setdefault("BLUR_DEVICE", "cuda:0")
try:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception as e:
    print("preload_dlls:", e)
print("torch.cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("ort:", ort.__version__, ort.get_available_providers())
raise SystemExit(0 if "CUDAExecutionProvider" in ort.get_available_providers() else 2)
"""
    rc = subprocess.call([py, "-c", probe])
    if rc != 0:
        print(
            "\n❌ CUDAExecutionProvider missing. Accel=GPU T4 x2? "
            f"Need onnxruntime-gpu=={ORT_GPU_VERSION} (CUDA 12), not 1.27+ (CUDA 13)."
        )
        return 1

    print(
        "\n✅ Setup OK.\n"
        "  1) Restart kernel\n"
        "  2) os.environ['BLUR_DEVICE']='cuda:0'; os.environ['BLUR_REQUIRE_GPU']='1'\n"
        "  3) Start: uvicorn main:app --host 0.0.0.0 --port 8000\n"
        "  4) Then ngrok on 8000 (set NGROK_AUTH_TOKEN via Kaggle Secrets)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
