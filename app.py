import os
import sys
from pathlib import Path
import gradio as gr

# 1. Add your backend directory to the Python path
# Locate main.py relative to this file
backend_path = Path(__file__).parent / "backend" / "Semantic-search-app" / "backend"
sys.path.append(str(backend_path))

# 2. Import your FastAPI app
try:
    from main import app as fastapi_app
except ImportError:
    print(f"Error: Could not find main.py at {backend_path}")
    sys.exit(1)

# 3. Mount FastAPI into Gradio
# This allows your React frontend to keep talking to the same API endpoints
# while Hugging Face provides ZeroGPU access.
app = gr.mount_gradio_app(fastapi_app, gr.Blocks(), path="/")

# Note: On Hugging Face, the server runs on port 7860 by default
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
