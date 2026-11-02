#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate external Lite Lossy data and patch only POST's final CT tables.

GPL-trained data is loaded from build/, never embedded in this MIT source.
Every lookup is anchored against the hash-verified AMD initializer and the
matching translated POST. Unknown table layouts are rejected, not guessed.
"""
from __future__ import annotations
import hashlib
import json
import re
import struct
from pathlib import Path

DLL_SHA256='d9b0a1f5781c2d633a327272f6f1c8607057df7cafd0233cf38c7395c8f7dbc5'
SCHEMA='fsr412_lite_lossy_import_v1'
FILE='initializers-lite-lossy.bin'
BIAS_MATRIX_SHA256='ff0ef7ff06cc065d00d9a3db8371327c202cb46c6f9a78ef4adae0fbfd4ea089'


def validate_lossy_data(data,receipt):
    from fsr412.lossy_weights import LOSSY_INITIALIZER_SHA256
    if (len(data)!=27392 or not isinstance(receipt,dict) or receipt.get('schema')!=SCHEMA or
        receipt.get('source_dll_sha256')!=DLL_SHA256 or not isinstance(receipt.get('initializers'),dict) or
        receipt['initializers'].get(FILE)!=LOSSY_INITIALIZER_SHA256 or sha(data)!=LOSSY_INITIALIZER_SHA256):
        raise ValueError('Lite Lossy initializer/receipt does not match the pinned release')
    matrix=receipt.get('post_ct_biases_by_tap')
    if (not isinstance(matrix,list) or len(matrix)!=4 or
        any(not isinstance(row,list) or len(row)!=8 or
            any(type(value) is not int or not -(1<<31)<=value<(1<<31) for value in row) for row in matrix)):
        raise ValueError('invalid Lite Lossy subpixel bias matrix')
    if sha(json.dumps(matrix,separators=(',',':')).encode())!=BIAS_MATRIX_SHA256:
        raise ValueError('Lite Lossy subpixel biases do not match the pinned release')
    if tuple(matrix[0])!=struct.unpack_from('<8i',data,27264):
        raise ValueError('Lite Lossy CT head order does not match the AMD initializer')
    return matrix


def sha(data):return hashlib.sha256(data).hexdigest()


def load_lossy_initializer(root):
    directory=Path(root)/'build/fsr412_lossy'
    path=directory/FILE
    try:
        data=path.read_bytes()
        receipt=json.loads((directory/'import.json').read_text())
        validate_lossy_data(data,receipt)
    except (OSError,ValueError,ImportError,KeyError,TypeError) as error:
        raise ValueError('verified Lite Lossy data is missing or invalid; run make lite-lossy-weights: '+str(error)) from error
    return path,data,receipt


def initializer_for(root,tier,preset,profile):
    root=Path(root)
    if profile not in ('int8-lite','int8-lite-lossy'):
        raise ValueError('unknown Lite model profile')
    if profile=='int8-lite-lossy':
        path,data,receipt=load_lossy_initializer(root)
        if preset=='normal' and tier<=2160:
            return path,data,True
    name='initializers-lite-ultraperformance.bin' if preset=='ultraperformance' else 'initializers-lite.bin'
    path=root/'build/fsr412_model'/name
    return path,path.read_bytes(),False


def _unique_offset(data,needle,label):
    start=data.find(needle)
    if start<0 or data.find(needle,start+1)>=0:
        raise ValueError('unknown/nonunique AMD Lite '+label+' initializer layout')
    return start


def patch_post_tail(text,base_initializer,trained_initializer):
    if len(base_initializer)!=27392 or len(trained_initializer)!=27392:
        raise ValueError('Lite POST needs complete 27392-byte initializers')
    matches=list(re.finditer(r'constant spvUnsafeArray<uint, 128> (\w+) = spvUnsafeArray<uint, 128>\(\{(.*?)\}\);',text))
    if len(matches)!=1:
        raise ValueError('unknown Lite POST final CT literal table')
    table=matches[0]
    words=[int(x) for x in re.findall(r'(\d+)u',table[2])]
    if len(words)!=128:raise ValueError('incomplete Lite POST CT literal table')
    offset=_unique_offset(base_initializer,struct.pack('<128I',*words),'CT weights')
    trained_words=struct.unpack_from('<128I',trained_initializer,offset)
    values=re.findall(r'= int4\((-?[\d, -]+)\);',text)
    bias_vectors=[tuple(int(x.strip()) for x in row.split(',')) for row in values]
    if len(bias_vectors)!=2 or any(len(row)!=4 for row in bias_vectors):
        raise ValueError('unknown Lite POST final CT bias vectors')
    edits=[]
    for row in bias_vectors:
        bias_offset=_unique_offset(base_initializer,struct.pack('<4i',*row),'CT biases')
        replacement=struct.unpack_from('<4i',trained_initializer,bias_offset)
        before='= int4('+', '.join(str(x) for x in row)+');'
        if text.count(before)!=1:raise ValueError('nonunique Lite POST CT bias expression')
        edits.append((before,'= int4('+', '.join(str(x) for x in replacement)+');'))
    after='constant spvUnsafeArray<uint, 128> '+table[1]+' = spvUnsafeArray<uint, 128>({ '+', '.join(str(x)+'u' for x in trained_words)+' });'
    text=text[:table.start()]+after+text[table.end():]
    for before,after in edits:text=text.replace(before,after,1)
    return text


def patch_post_biases_by_tap(text,initializer,receipt):
    """Apply the source's four CT bias heads AFTER the Lossy cache transform.

    AMD stores filter first (bytes 27264..27279), recurrent second. The
    external receipt preserves all four subpixel heads instead of averaging
    or incorrectly using tap 0 for the other phases.
    """
    matrix=validate_lossy_data(initializer,receipt)
    if 'Fsr412TapBias' in text:raise ValueError('Lite CT subpixel biases already transformed')
    loops=list(re.finditer(r'for \(int (_\d+) = 0; \1 < 4; \1\+\+\)',text))
    if len(loops)!=1:raise ValueError('unknown Lite POST subpixel loop')
    phase=loops[0][1]
    arrays=[]
    for head,begin in (('Filter',0),('Recurrent',4)):
        row=matrix[0][begin:begin+4]
        before='= int4('+', '.join(str(x) for x in row)+');'
        if text.count(before)!=1:raise ValueError('unknown Lite POST primary '+head+' bias')
        name='Fsr412TapBias'+head
        text=text.replace(before,'= '+name+'['+phase+'];',1)
        values=', '.join('int4('+', '.join(str(x) for x in entry[begin:begin+4])+')' for entry in matrix)
        arrays.append('constant int4 '+name+'[4] = { '+values+' };\n')
    kernel=re.search(r'kernel\s+void\s+\w+\s*\(',text)
    if not kernel:raise ValueError('missing Lite POST kernel')
    return text[:kernel.start()]+''.join(arrays)+'\n'+text[kernel.start():]


def main():
    import argparse,sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--verify",action="store_true",required=True)
    args=parser.parse_args()
    path,data,receipt=load_lossy_initializer(args.root)
    print(json.dumps({"schema":SCHEMA,"initializer":str(path),"sha256":sha(data)}))

if __name__=="__main__":main()
