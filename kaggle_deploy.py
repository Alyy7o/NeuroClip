# Kaggle Deployment Script for NeuroClip Backend (summarization + compression + blur)
# Use a GPU notebook (T4 x2). Run cells in order.

# --- Cell 1: Install dependencies ---
# !pip install -r /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/requirements.txt
# !pip install -r /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/requirements-blur.txt
# !pip install pyngrok

# --- Cell 2: Verify environment ---
# %run /kaggle/working/NeuroClip/backend/Semantic-search-app/backend/verify_env.py

# --- Cell 3: Blur model weights (add Kaggle Dataset "neuroclip-blur-weights") ---
# import os
# os.environ["BLUR_YOLO_WEIGHTS"] = "/kaggle/input/neuroclip-blur-weights/yolov8n.pt"
# os.environ["BLUR_DEVICE"] = "cuda:0"

# --- Cell 4: Start FastAPI (one process for all modules) ---
# IMPORTANT: Restart kernel after pulling new code or /anonymize-video will 404.
# %cd /kaggle/working/NeuroClip/backend/Semantic-search-app/backend
# !uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Verify: curl -H "ngrok-skip-browser-warning: true" $NGROK_URL/health
#   must show "build_id": "2026-05-blur-v1" and "has_anonymize_endpoint": true

# --- Cell 5: Ngrok tunnel ---
import os
from pyngrok import ngrok

# Replace with your Ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
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
    print(" * Set frontend .env: VITE_API_BASE_URL=" + public_url)
    print(" * Endpoints: /upload-and-search, /compress-video, /anonymize-video")
    print("=" * 50 + "\n")
    return public_url


if __name__ == "__main__":
    start_tunnel()

# --- Cell 6 (optional): smoke-test blur ---
# curl -X POST "$NGROK_URL/anonymize-video" -H "ngrok-skip-browser-warning: true" \
#   -F "file=@test.mp4" -F "reference_images=@person.jpg"
