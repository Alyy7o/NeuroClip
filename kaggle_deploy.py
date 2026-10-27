# Kaggle Deployment Script for NeuroClip Backend
# Accel MUST be: GPU T4 x2 (CPU blur ≈ 1h for a 17s clip).
#
# =============================================================================
# PASTE THIS AS YOUR SETUP CELL (order matters — ORT pin LAST)
# =============================================================================
#
# import os, shutil
#
# # 0) Clean + clone
# repo_path = "/kaggle/working/NeuroClip"
# if os.path.exists(repo_path):
#     shutil.rmtree(repo_path)
# !git clone https://github.com/Alyy7o/NeuroClip.git
#
# # 1) Deps (insightface may pull CPU onnxruntime — that is OK for now)
# %cd /kaggle/working/NeuroClip/backend/Semantic-search-app/backend
# !pip install -r requirements.txt
# !pip install -r requirements-blur.txt
#
# # 2) CRITICAL — pin CUDA-12 ORT AFTER requirements (Kaggle ≠ CUDA 13)
# #    Bare `pip install onnxruntime-gpu` → 1.31 / CUDA13 → no CUDAExecutionProvider
# %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/install_kaggle_blur_gpu.py
# # (same as: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.19.2)
#
# # 3) Env for blur GPU fail-fast + perf knobs
# os.environ["BLUR_DEVICE"] = "cuda:0"
# os.environ["BLUR_REQUIRE_GPU"] = "1"
# # Sparse locked path (Phase 1) + pipeline (Phase 2) + dual-T4 chunks (Phase 3)
# # os.environ["BLUR_TRACK_INTERVAL"] = "3"
# # os.environ["BLUR_FACE_INTERVAL"] = "5"
# # os.environ["BLUR_LOCKED_IMGSZ"] = "416"
# # os.environ["BLUR_LOCKED_DET_SIZE"] = "320"
# # os.environ["BLUR_PIPELINE"] = "1"
# # os.environ["BLUR_CHUNK_PARALLEL"] = "1"  # auto on T4x2 for clips >= 3 min
# # License-plate blur (Phase-1 blur-all) — REQUIRED before toggling plates in UI:
# %run install_plate_weights.py
# # (downloads weights/yolov8n-license-plate.pt from Hugging Face)
# os.environ["BLUR_PLATE_WEIGHTS"] = "/kaggle/working/NeuroClip/backend/Semantic-search-app/backend/weights/yolov8n-license-plate.pt"
# # os.environ["BLUR_PLATE_INTERVAL"] = "2"
# # optional person YOLO weights dataset:
# # os.environ["BLUR_YOLO_WEIGHTS"] = "/kaggle/input/neuroclip-blur-weights/yolov8n.pt"
#
# # 4) Verify (must print CUDAExecutionProvider)
# import torch, onnxruntime as ort
# print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
# print("ort:", ort.__version__, ort.get_available_providers())
# !nvidia-smi
#
# # 5) RESTART KERNEL now, then run the SERVER cell below (do not skip restart)
#
# =============================================================================
# SERVER CELL (after kernel restart — set env again, then uvicorn + ngrok)
# =============================================================================
#
# import os
# os.environ["BLUR_DEVICE"] = "cuda:0"
# os.environ["BLUR_REQUIRE_GPU"] = "1"
# # os.environ["BLUR_CHUNK_PARALLEL"] = "1"  # dual-T4 for long clips
# %cd /kaggle/working/NeuroClip/backend/Semantic-search-app/backend
# %run install_plate_weights.py
# os.environ["BLUR_PLATE_WEIGHTS"] = str(__import__("pathlib").Path("weights/yolov8n-license-plate.pt").resolve())

#
# # Start API in background, then tunnel:
# import subprocess, time
# subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
# time.sleep(5)
#
# from kaggle_secrets import UserSecretsClient  # preferred
# # token = UserSecretsClient().get_secret("NGROK_AUTH_TOKEN")
# # Or: export NGROK_AUTH_TOKEN in the cell env — do NOT hardcode tokens in git
#
# from pyngrok import ngrok
# ngrok.kill()
# ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])
# public_url = ngrok.connect(8000).public_url
# print("Backend URL:", public_url)
# print("Set frontend .env: VITE_API_BASE_URL=" + public_url)
#
# =============================================================================

import os

try:
    from pyngrok import ngrok
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install pyngrok first") from exc

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "").strip()


def start_tunnel(port: int = 8000):
    if not NGROK_AUTH_TOKEN or NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTH_TOKEN":
        raise RuntimeError(
            "Set NGROK_AUTH_TOKEN env (Kaggle Secrets → NGROK_AUTH_TOKEN). "
            "Do not hardcode the token in the notebook if you can avoid it."
        )
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
        ngrok.kill()
        print(" * Closed existing Ngrok sessions.")
    except Exception:
        pass

    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(port).public_url
    print("\n" + "=" * 50)
    print(f" * Backend is live at: {public_url}")
    print(f" * Tunnel target: http://localhost:{port}  (uvicorn :8000, NOT Flask :5000)")
    print(" * Set frontend .env: VITE_API_BASE_URL=" + public_url)
    print(" * Endpoints: /upload-and-search, /compress-video, /anonymize-video")
    print(" * /health should show insightface_on_gpu after first blur engine load")
    print("=" * 50 + "\n")
    return public_url


if __name__ == "__main__":
    start_tunnel()
