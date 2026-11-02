#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add the original GPU skip-marker gate to independent integer operators.

Only the opt-in original Lite Lossy adapter uses this source. The 212-byte
PassParams ABI, integer arithmetic and ordinary libraries remain unchanged.
The marker is read before any workgroup barrier, so all lanes return together.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re

BEGIN = '// fsr412-original-neural-gate-v1-begin\n'
END = '// fsr412-original-neural-gate-v1-end\n'
HELPER = r'''
inline bool fsr412OriginalSkipped(device const uchar *scratch) {
#if FSR411_MAXH <= 1080
    constexpr uint plane = (FSR411_MAXW + 2u) * (FSR411_MAXH + 2u) * 16u;
    const device uint *marker = reinterpret_cast<const device uint *>(scratch + plane + 128u);
    return marker[0] == 1511506142u && marker[1] == 168889943u;
#else
    // Original 4320 PRE has no skip path.
    (void)scratch;
    return false;
#endif
}
'''


def closing_parenthesis(source, start):
    depth = 0
    for index in range(start, len(source)):
        depth += (source[index] == '(') - (source[index] == ')')
        if depth == 0:
            return index
    raise ValueError('unterminated original integer kernel signature')


def apply_original_gate(source):
    if BEGIN in source:
        if source.count(BEGIN) != 1 or source.count(END) != 1:
            raise ValueError('incomplete original neural gate')
        return source
    if 'fsr412OriginalSkipped' in source:
        raise ValueError('original neural gate identifier collision')
    include = '#include "fsr411_model.h"'
    if source.count(include) != 1:
        raise ValueError('unknown independent integer operator header')
    edits = []
    kernels = 0
    for match in re.finditer(r'\bkernel\s+void\s+(\w+)\s*\(', source):
        if match[1] in ('fsr411_mpp_chain_probe', 'fsr411_simd_probe'):
            continue
        end = closing_parenthesis(source, match.end() - 1)
        arguments = source[match.end():end]
        if '[[buffer(2)]]' not in arguments and 'FSR411_SIMD_ARGS' not in arguments:
            raise ValueError('unknown original integer kernel: ' + match[1])
        brace = end + 1
        while brace < len(source) and source[brace].isspace():
            brace += 1
        if source[brace:brace + 1] != '{':
            raise ValueError('unknown integer kernel body: ' + match[1])
        # Keep macro continuation lines intact in MPP and SIMD declarations.
        edits.append((brace + 1, ' if (fsr412OriginalSkipped(scratch)) return; '))
        kernels += 1
    if not kernels:
        raise ValueError('no original integer operators found')
    edits.append((source.index(include) + len(include), '\n' + BEGIN + HELPER + END))
    for offset, replacement in sorted(edits, reverse=True):
        source = source[:offset] + replacement + source[offset:]
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = apply_original_gate(args.source.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.is_file() or args.out.read_text() != source:
        args.out.write_text(source)


if __name__ == '__main__':
    main()
