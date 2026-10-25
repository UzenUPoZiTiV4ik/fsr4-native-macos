#!/usr/bin/env python3
"""Compatibility entry point for the FSR 4.1.2 Lite compiler."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "fsr412" / "compile_fsr412.py"), run_name="__main__")
