# Kaggle Deployment Script for NeuroClip Backend
# Paste these commands into separate cells in a Kaggle Notebook (with GPU T4 x2 enabled)

# --- Cell 1: Install dependencies ---
# !pip install -r /kaggle/input/neuroclip/backend/Semantic-search-app/backend/requirements.txt
# !pip install pyngrok

# --- Cell 2: Setup Ngrok and Run ---
import os
from pyngrok import ngrok, conf
import uvicorn
import asyncio

# Replace with your actual Ngrok Auth Token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"

def start_tunnel():
    conf.get_default().auth_token = NGROK_AUTH_TOKEN
    
    # Open a HTTP tunnel on port 8000
    public_url = ngrok.connect(8000).public_url
    print(f" * Backend is live at: {public_url}")
    print(f" * Update your frontend .env VITE_API_BASE_URL to this URL.")
    return public_url

# To run this in Kaggle, you would typically run the FastAPI app in a background thread 
# or use a library like 'nest_asyncio' if running inside a notebook cell.
if __name__ == "__main__":
    url = start_tunnel()
    # Import your app here
    # from main import app
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Run the server now!")
