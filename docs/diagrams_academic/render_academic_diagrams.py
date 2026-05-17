#!/usr/bin/env python
"""Entry point that renders every NeuroClip Chapter 4 academic-style diagram.

Run from the repo root or from this folder:

    python docs/diagrams_academic/render_academic_diagrams.py

It will write a matching ``.png`` and ``.svg`` for every figure in this folder.
Required dependency: ``matplotlib`` (already available in the project venv).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagrams import render_all  # noqa: E402  (path setup must come first)


def main() -> int:
    print("Rendering NeuroClip Chapter 4 academic diagrams ...")
    render_all()
    print(f"Outputs written to: {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
