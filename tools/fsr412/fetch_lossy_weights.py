#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch and statically decode pinned FSR 4.1.2 Lite Lossy data into build/.

The Windows DLL is never executed or redistributed. Trained data belongs to
upstream GPL-2.0-or-later; this MIT repository contains only the independent
loader/decoder. The receipt records the source's conditional border/tap biases.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from fsr412.lossy_weights import recover_primary_training,DLL_SHA256,LOSSY_INITIALIZER_SHA256,COMMON_SHA256
from fsr412.model_data import validate_lossy_data
URL='https://github.com/zgauthier2000/fsr4-shader-performance-overrides/releases/download/dll-2026-11-01/fsr4.1.2-cyboman-rx7000-2026-11-01.7z'
ARCHIVE_SHA256='743adcd599dee3cbf6035ed08ceeced987b75178f74a22b945a421d030d8d34e'
MEMBER='4.1.2l-lossy/amd_fidelityfx_upscaler_dx12.dll'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def atomic(path,data):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--dll',type=Path)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--force',action='store_true',help='redecode and refresh the receipt even if already present')
    parser.add_argument('--url',default=URL)
    parser.add_argument('--dxc',type=Path,default=ROOT/'tools/bin/dxc')
    args=parser.parse_args()
    out=args.outdir or args.root/'build/fsr412_lossy'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.import.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data_path=out/'initializers-lite-lossy.bin'
        receipt_path=out/'import.json'
        if not args.force and data_path.is_file() and receipt_path.is_file():
            try:
                validate_lossy_data(data_path.read_bytes(),json.loads(receipt_path.read_text()))
            except (OSError,ValueError,KeyError,TypeError):
                pass
            else:
                print('Already verified:',data_path)
                return
        base=args.root/'build/fsr412_model/initializers-lite.bin'
        if not base.is_file() or digest(base)!=COMMON_SHA256:
            raise ValueError('Import the official AMD Lite model first; its pinned base initializer is required')
        with tempfile.TemporaryDirectory(prefix='fsr412-lite-lossy-') as work:
            work=Path(work)
            dll=args.dll
            if dll is None:
                archive=args.archive
                if archive is None:
                    archive=work/'release.7z'
                    with urllib.request.urlopen(args.url,timeout=120) as response,archive.open('wb') as stream:
                        shutil.copyfileobj(response,stream)
                if digest(archive)!=ARCHIVE_SHA256:
                    raise ValueError('Lite Lossy release archive hash mismatch')
                tar=shutil.which('bsdtar') or shutil.which('tar')
                if not tar:raise ValueError('bsdtar is required to unpack the pinned archive')
                subprocess.run([tar,'-xf',str(archive),'-C',str(work),MEMBER],check=True)
                dll=work/MEMBER
            if digest(dll)!=DLL_SHA256:
                raise ValueError('Lite Lossy DLL hash mismatch')
            data,receipt=recover_primary_training(dll,args.dxc,base)
            if hashlib.sha256(data).hexdigest()!=LOSSY_INITIALIZER_SHA256:
                raise ValueError('Recovered Lite Lossy initializer hash mismatch')
            validate_lossy_data(data,receipt)
            receipt.update(release='dll-2026-11-01',archive_url=URL,archive_sha256=ARCHIVE_SHA256,
                license='GPL-2.0-or-later upstream trained data; decoder/adapter MIT',
                windows_dll_executed=False,upstream_dll_parity=False)
            atomic(data_path,data)
            atomic(receipt_path,(json.dumps(receipt,indent=2)+'\n').encode())
        print('Verified Lite Lossy data:',data_path)
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        raise SystemExit('Lite Lossy import failed: '+str(error))
