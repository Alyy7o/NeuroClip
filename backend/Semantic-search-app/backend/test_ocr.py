import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from ocr_utils import load_ocr_model, extract_high_value_frames, run_ocr_on_frames

def test_ocr_logic():
    print("--- Starting OCR Logic Test ---")
    
    # Paths
    backend_dir = Path(__file__).resolve().parent
    repo_root = backend_dir.parents[1]
    model_path = repo_root / "custom_model_training" / "neuroclip_v1.pth"
    
    # 1. Load Model
    print(f"Testing model loading from: {model_path}")
    model = load_ocr_model(str(model_path))
    if model:
        print("SUCCESS: Model loaded correctly.")
    else:
        print("RESULT: Model file not found (expected if running without .pth).")

    # 2. Check Dependencies
    print("\nChecking dependencies...")
    try:
        import easyocr
        import cv2
        import torch
        print(f"SUCCESS: torch version {torch.__version__}")
        print(f"SUCCESS: cv2 version {cv2.__version__}")
        print("SUCCESS: easyocr is installed.")
    except ImportError as e:
        print(f"FAILURE: Missing dependency: {e}")

    print("\n--- OCR Logic Test Complete ---")

if __name__ == "__main__":
    test_ocr_logic()
