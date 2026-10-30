#!/usr/bin/env python3
"""Prove an exact FSR 4.1.2 source transform changes no bits.

Builds a second root whose tools/fsr412/optimization.py replaces the named
text transform with the identity, gives it a private build directory, and
runs build/tests/fsr412_runtime_ab_test against the real root on both APIs.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disable', required=True, help='optimization.py text transform to replace with identity')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fsr412-ab-') as temp:
        root, build = Path(temp) / 'root', Path(temp) / 'root/build'
        root.mkdir()
        for entry in ROOT.iterdir():
            if entry.name not in ('tools', 'build'):
                (root / entry.name).symlink_to(entry)
        shutil.copytree(ROOT / 'tools', root / 'tools', ignore=shutil.ignore_patterns('__pycache__'), symlinks=True)
        optimization = root / 'tools/fsr412/optimization.py'
        text = optimization.read_text()
        if f'def {args.disable}(text' not in text:
            raise SystemExit(f'unknown transform {args.disable}')
        optimization.write_text(text + f'\n\ndef {args.disable}(text, *unused, **unused_named):\n    return text\n')
        build.mkdir()
        for entry in (ROOT / 'build').iterdir():
            if entry.name == 'fsr412_model' or not entry.name.startswith('fsr412_'):
                (build / entry.name).symlink_to(entry)
        test = ROOT / 'build/tests/fsr412_runtime_ab_test'
        for api in ([], ['mtl4']):
            subprocess.run([str(test), str(ROOT), str(ROOT / 'build'), str(root), str(build)] + api, check=True)


if __name__ == '__main__':
    main()
