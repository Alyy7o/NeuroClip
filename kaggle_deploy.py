# Kaggle Deployment Script for NeuroClip Backend
# Paste these commands into separate cells in a Kaggle Notebook (with GPU T4 x2 enabled)

# --- Cell 1: Install dependencies ---
# !pip install -r /kaggle/input/neuroclip/backend/Semantic-search-app/backend/requirements.txt
# !pip install pyngrok

# --- Cell 2: Setup Ngrok and Run ---
import os
from pyngrok import ngrok, conf

# Replace with your actual Ngrok Auth Token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"

def start_tunnel():
    # 1. Kill any existing tunnels to avoid "Too many sessions" error (ERR_NGROK_324)
    try:
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            ngrok.disconnect(t.public_url)
        ngrok.kill()
        print(" * Closed existing Ngrok sessions.")
    except Exception:
        pass

    # 2. Set auth token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Open a HTTP tunnel on port 8000
    public_url = ngrok.connect(8000).public_url
    print(f"\n" + "="*50)
    print(f" * Backend is live at: {public_url}")
    print(f" * Update your frontend .env VITE_API_BASE_URL to this URL.")
    print(f"="*50 + "\n")
    return public_url

if __name__ == "__main__":
    url = start_tunnel()
    print("Starting Uvicorn in the background...")
    # Using '!' shell command is much more stable in Kaggle notebooks
    # Run this in a new cell:
    # !uvicorn main:app --host 0.0.0.0 --port 8000
