# Kaggle Deployment Script for NeuroClip Backend (summarization + compression + blur)
# MUST use a GPU notebook: Accel = GPU T4 x2 (CPU takes ~1h for a ~17s blur job).
#
# --- Cell 0: Confirm GPU ---
# import torch, onnxruntime as ort
# print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
# print("ort providers:", ort.get_available_providers())
# # Expect: cuda True + 'CUDAExecutionProvider' in providers AFTER Cell 1
#
# --- Cell 1: Install dependencies (order matters for InsightFace GPU) ---
# !pip install -r /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/requirements.txt
# !pip install -r /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/requirements-blur.txt
# !pip uninstall -y onnxruntime onnxruntime-gpu
# !pip install onnxruntime-gpu
# import os
# os.environ["BLUR_DEVICE"] = "cuda:0"
# # On Kaggle, BLUR_REQUIRE_GPU defaults on — fail fast if still CPU-only
# os.environ["BLUR_REQUIRE_GPU"] = "1"
# !pip install pyngrok
#
# --- Cell 2: Verify environment ---
# %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/verify_env.py
# # Must show CUDAExecutionProvider and torch.cuda True, else blur will be unusable
#
# --- Cell 3: Blur model weights (optional dataset) ---
# import os
# os.environ["BLUR_YOLO_WEIGHTS"] = "/kaggle/input/neuroclip-blur-weights/yolov8n.pt"
# os.environ["BLUR_DEVICE"] = "cuda:0"
#
# --- Cell 4: Start FastAPI (port 8000 — NOT Flask :5000) ---
# IMPORTANT: Restart kernel after pulling new code or /anonymize-video will 404.
# %cd /kaggle/working/NeuroClip/backend/Semantic-search-app/backend
# !uvicorn main:app --host 0.0.0.0 --port 8000
# Verify: curl -H "ngrok-skip-browser-warning: true" $NGROK_URL/health
#   must show has_anonymize_endpoint=true AND insightface_on_gpu=true (after first blur load)
#
# --- Cell 5: Ngrok tunnel (point at 8000) ---
import os
from pyngrok import ngrok

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "YOUR_NGROK_AUTH_TOKEN")


def start_tunnel(port: int = 8000):
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
    print(f" * Tunnel target: http://localhost:{port}  (use 8000 for uvicorn, NOT 5000 Flask)")
    print(" * Set frontend .env: VITE_API_BASE_URL=" + public_url)
    print(" * Endpoints: /upload-and-search, /compress-video, /anonymize-video")
    print(" * Health should report insightface_on_gpu after engine load")
    print("=" * 50 + "\n")
    return public_url


if __name__ == "__main__":
    start_tunnel()

# --- Cell 6 (optional): smoke-test blur ---
# curl -X POST "$NGROK_URL/anonymize-video" -H "ngrok-skip-browser-warning: true" \
#   -F "file=@test.mp4" -F "reference_images=@person.jpg"
