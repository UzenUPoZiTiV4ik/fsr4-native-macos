#!/usr/bin/env python3
"""Compatibility entry point for the local FSR 4.1.2 model importer."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'fsr412_import_model.py'), run_name='__main__')
