"""
Download YOLOv8 license-plate detector weights for NeuroClip Phase-1 plate blur.

Run from the backend directory (same place as main.py / uvicorn):

  python install_plate_weights.py

On Kaggle (after %cd to backend):

  %run install_plate_weights.py

Then set (optional if default path exists):

  os.environ["BLUR_PLATE_WEIGHTS"] = "weights/yolov8n-license-plate.pt"

Restart the API process so BlurEngine can load the new file (or rely on lazy load
on the next anonymize job).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "weights" / "yolov8n-license-plate.pt"


def main() -> int:
    url = os.getenv("BLUR_PLATE_WEIGHTS_URL", DEFAULT_URL).strip() or DEFAULT_URL
    out = Path(os.getenv("BLUR_PLATE_WEIGHTS", str(DEFAULT_OUT))).expanduser()
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[plate-weights] already present: {out} ({out.stat().st_size} bytes)")
        print(f'[plate-weights] export BLUR_PLATE_WEIGHTS="{out.as_posix()}"')
        return 0

    print(f"[plate-weights] downloading\n  {url}\n→ {out}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as exc:
        print(f"[plate-weights] download failed: {exc}", file=sys.stderr)
        return 1

    size = out.stat().st_size if out.exists() else 0
    if size < 1_000_000:
        print(f"[plate-weights] file looks too small ({size} bytes)", file=sys.stderr)
        return 1

    print(f"[plate-weights] OK {out} ({size} bytes)")
    print(f'[plate-weights] export BLUR_PLATE_WEIGHTS="{out.as_posix()}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
