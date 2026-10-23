"""
install_kaggle_blur_gpu.py
==========================
Run in a Kaggle GPU notebook AFTER Accel=GPU T4 x2, BEFORE uvicorn.

  %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/install_kaggle_blur_gpu.py

Why this exists:
  Plain `pip install onnxruntime-gpu` pulls 1.27+ built for CUDA 13.
  Kaggle T4 stacks are CUDA 12.x → CUDAExecutionProvider never appears → InsightFace stays CPU.
  We pin a CUDA-12-compatible wheel and verify providers.
"""

from __future__ import annotations

import subprocess
import sys


# CUDA 12.x compatible (Kaggle T4). Do NOT use 1.27+ from default PyPI (CUDA 13).
ORT_GPU_VERSION = "1.19.2"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    py = sys.executable
    print("=" * 60)
    print(f"  Installing onnxruntime-gpu=={ORT_GPU_VERSION} (CUDA 12 / Kaggle T4)")
    print("=" * 60)

    _run([py, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"])
    _run([py, "-m", "pip", "install", f"onnxruntime-gpu=={ORT_GPU_VERSION}"])

    # Fresh import in a subprocess so we don't see a stale module
    probe = r"""
import onnxruntime as ort
try:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
except Exception as e:
    print("preload_dlls:", e)
providers = ort.get_available_providers()
print("ort.__version__=", ort.__version__)
print("providers=", providers)
ok = "CUDAExecutionProvider" in providers
raise SystemExit(0 if ok else 2)
"""
    print("\n[probe] onnxruntime providers after install:")
    rc = subprocess.call([py, "-c", probe])
    if rc != 0:
        print(
            "\n❌ FAIL: CUDAExecutionProvider still missing.\n"
            "  1) Confirm Notebook Accel = GPU T4 x2 (not CPU)\n"
            "  2) Restart kernel, re-run this script\n"
            "  3) Then: import torch; print(torch.cuda.is_available())\n"
        )
        return 1

    print(
        "\n✅ CUDAExecutionProvider OK.\n"
        "  NEXT: Restart the kernel (recommended), set BLUR_DEVICE=cuda:0,\n"
        "  run verify_env.py, then start uvicorn on port 8000.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
