#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add uniform, pre-barrier tile gates to the existing integer Metal entries.

Only the experimental Lite Lossy session calls this generator. Ordinary
libraries, lane-layout probes, model weights and the PassParams ABI remain
unchanged. The host supplies an extra 32-byte constant block at buffer 5.
"""
from __future__ import annotations
import argparse
import hashlib
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REVISION=1
BEGIN='// fsr412-neural-tiling-v1-begin\n'
END='// fsr412-neural-tiling-v1-end\n'


def close_parenthesis(text,start):
    depth=0
    for i in range(start,len(text)):
        if text[i]=='(':depth+=1
        elif text[i]==')':
            depth-=1
            if depth==0:return i
    raise ValueError('unterminated native Metal kernel signature')


def apply_neural_tiling(source: str,header: str|None=None) -> str:
    if BEGIN in source:
        if source.count(BEGIN)!=1 or source.count(END)!=1:
            raise ValueError('incomplete Lite neural tile transform')
        return source
    if 'fsr412TileControl' in source or 'fsr412TileGroup' in source:
        raise ValueError('native Metal tile identifier collision')
    include='#include "fsr411_model.h"'
    if source.count(include)!=1:
        raise ValueError('unknown native integer Metal source')
    header=(ROOT/'src/fsr412_neural_tile.h').read_text() if header is None else header
    header=header.replace('#pragma once\n','').replace(include+'\n','')
    if 'struct Control' not in header or 'inline bool active(' not in header:
        raise ValueError('unknown tile constant ABI')
    changes=[]
    count=0
    for match in re.finditer(r'\bkernel\s+void\s+(\w+)\s*\(',source):
        name=match[1]
        if name in ('fsr411_mpp_chain_probe','fsr411_simd_probe'):
            continue
        start=match.end()-1
        end=close_parenthesis(source,start)
        signature=source[start+1:end]
        if '[[buffer(2)]]' not in signature and 'FSR411_SIMD_ARGS' not in signature:
            raise ValueError('unknown integer kernel arguments: '+name)
        if '[[buffer(5)]]' in signature:
            raise ValueError('native buffer 5 is occupied: '+name)
        position=end+1
        while position<len(source) and source[position].isspace():position+=1
        if source[position:position+1]!='{':
            raise ValueError('unknown kernel body: '+name)
        group=re.search(r'uint[23]\s+(\w+)\s*\[\[threadgroup_position_in_grid\]\]',signature)
        # SIMD_ARGS is one macro shared by all SIMD entries. Add its bindings
        # once, rather than break every macro invocation's declaration.
        if 'FSR411_SIMD_ARGS' in signature:
            group_name='group'
            additions=''
        else:
            group_name=group[1] if group else 'fsr412TileGroup'
            additions=', constant fsr412_tiling::Control& fsr412TileControl [[buffer(5)]]'
            if not group:additions+=', uint2 fsr412TileGroup [[threadgroup_position_in_grid]]'
        if additions:changes.append((end,end,additions))
        # Do not insert a newline: macro-generated MPP/SIMD entry bodies use
        # trailing backslashes. This statement remains on their existing line.
        gate=f' if (!fsr412_tiling::active(fsr412TileControl, {group_name}.x, {group_name}.y)) return; '
        changes.append((position+1,position+1,gate))
        count+=1
    if not count:
        raise ValueError('no native integer kernels found for tile gating')
    if 'FSR411_SIMD_ARGS' in source:
        marker='#define FSR411_SIMD_ARGS'
        start=source.find(marker)
        if start<0 or source.find(marker,start+1)>=0:
            raise ValueError('unknown SIMD argument macro')
        end=source.find('\n\n',start)
        if end<0:raise ValueError('unterminated SIMD argument macro')
        macro=source[start:end]
        if '[[threadgroup_position_in_grid]]' not in macro or '[[buffer(5)]]' in macro:
            raise ValueError('unknown SIMD dispatch binding')
        changes.append((end,end,', constant fsr412_tiling::Control& fsr412TileControl [[buffer(5)]]'))
    inserted=BEGIN+header+'\n'+END
    end=source.index(include)+len(include)
    changes.append((end,end,'\n'+inserted))
    for start,end,text in sorted(changes,reverse=True):source=source[:start]+text+source[end:]
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    text=apply_neural_tiling(args.source.read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if not args.out.exists() or args.out.read_text()!=text:args.out.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())
if __name__=='__main__':main()
