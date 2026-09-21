#!/usr/bin/env python3
"""Compatibility entry point for the publication R/TikZ figure workflow."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    pipeline = Path(__file__).resolve().parent / "r_tikz" / "build.py"
    raise SystemExit(subprocess.call([sys.executable, str(pipeline), *sys.argv[1:]]))
