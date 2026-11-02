# SPDX-License-Identifier: MIT
"""Controlled IO/dispatch variants of external original POST translations.

Oracle adaptations, not literal DLL scheduling or Windows evidence. Ordered
changes group scheduling only. Isolated changes five cache-address bases only;
explicit host copy-in/back preserves the original logical scratch state.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
from upstream_oracle import ACTIVE_2160, external_output, require, sha256, scratch_geometry

POST=ACTIVE_2160['post']
OLD_BASE=49941264
LOGICAL_BYTES=99820992
PLANE_BYTES=33273664
NEW_BASE=(LOGICAL_BYTES+255)&~255
PHYSICAL_BYTES=NEW_BASE+PLANE_BYTES


def ordered(source,offset_slot=4):
    require('oracleGroupYOffset' not in source,'POST already ordered')
    global_token='uint3 gl_GlobalInvocationID [[thread_position_in_grid]]'
    has_global='gl_GlobalInvocationID' in source
    if has_global:require(source.count(global_token)==1,'unknown global invocation coordinate signature')
    entry=re.search(r'kernel void main0\(',source);require(entry is not None,'POST entry changed')
    source=source[:entry.end()]+f'constant uint& oracleGroupYOffset [[buffer({offset_slot})]], '+source[entry.end():]
    token='uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]';require(source.count(token)==1,'POST group coordinate signature changed')
    source=source.replace(token,'uint3 oracleRawGroupID [[threadgroup_position_in_grid]]')
    if has_global:source=source.replace(global_token,'uint3 oracleRawGlobalID [[thread_position_in_grid]]')
    brace=source.index('\n{',entry.start())+2
    coordinates='\n    uint3 gl_WorkGroupID=oracleRawGroupID+uint3(0u,oracleGroupYOffset,0u);'
    # Every selected original POST uses numthreads(256,1,1), so shifting
    # group Y by one shifts global invocation Y by exactly one as well.
    if has_global:coordinates+='\n    uint3 gl_GlobalInvocationID=oracleRawGlobalID+uint3(0u,oracleGroupYOffset,0u);'
    return source[:brace]+coordinates+source[brace:]


def isolated(source,geometry=None):
    geometry=geometry or scratch_geometry(2160);old=geometry['saved_cnb_byte_offset'];logical=geometry['allocation_bytes'];plane=geometry['half_plane_bytes'];row=geometry['half_row_bytes']
    require(old is not None,'this POST tier has no cache/skip path')
    new=(logical+255)&~255;physical=new+plane
    # The invalid-store sentinel remains outside the adapted allocations for
    # cache-using 1080/2160 tiers. 4320 has no savedCNB path and is not adapted.
    require(physical<134217728,'IO sentinel must remain outside adapted scratch extent')
    pattern=re.compile(r'(uint _\d+ = \(\(\(_\d+ \* '+str(row)+r'u\) \+ )'+str(old)+r'u(\) \+ \(_\d+ << 4u\)\) >> 2u;)')
    require(len(list(pattern.finditer(source)))==5 and source.count(str(old)+'u')==5,'cache address site shape/count changed')
    source=pattern.sub(lambda m:m[1]+str(new)+'u'+m[2],source)
    source=re.sub(r', '+str(logical//4)+r'u\)',', '+str(physical//4)+'u)',source)
    return source


def build_selected(output,identity,geometry,compile_metal=True):
    output=external_output(Path(output));receipt=json.loads((output/(identity+'.translation.json')).read_text());literal=(output/(identity+'.metal')).read_text()
    require(sha256(literal.encode())==receipt['metal_sha256'],'literal POST source identity mismatch')
    require(receipt['reflection']['entryPoints'][0]['workgroup_size']==[256,1,1],'unknown original POST thread geometry')
    if geometry['saved_cnb_byte_offset'] is None:return {}
    offset_slot=max(b['metal_slot'] for b in receipt['bindings'] if b['metal_kind']=='buffer')+1
    logical=geometry['allocation_bytes'];plane=geometry['half_plane_bytes'];new=(logical+255)&~255;physical=new+plane
    results={}
    for mode,source in [('ordered',ordered(literal,offset_slot)),('isolated',isolated(literal,geometry))]:
        stem=identity+'.'+mode;msl=output/(stem+'.metal');msl.write_text(source)
        record=dict(receipt);record['schema']='fsr412_original_post_variant_v1';record['oracle_post_mode']=mode
        record['literal_metal_sha256']=receipt['metal_sha256'];record['metal_sha256']=sha256(source.encode());record['library']=stem+'.metallib'
        if mode=='ordered':
            record['group_y_offset_buffer_slot']=offset_slot;record['dispatch_policy']='one group-Y row at a time, top to bottom, completion between rows'
        else:
            record['cache_original_byte_offset']=geometry['saved_cnb_byte_offset'];record['cache_relocated_byte_offset']=new;record['cache_copy_bytes']=plane;record['physical_scratch_bytes']=physical
            record['cache_copy_policy']='copy original cache span to appended cache immediately before POST; copy back immediately after POST; no concurrent dispatches'
            record['io_guard']=dict(receipt['io_guard']);record['io_guard']['buffer_word_limits']={name:(physical//4 if name.startswith('oracle_u11_') else count) for name,count in receipt['io_guard']['buffer_word_limits'].items()}
        record['adaptation_limit']='controlled original-instruction oracle adaptation; not literal dispatch ordering/alias behavior and not Windows runtime parity'
        if compile_metal:
            air=output/(stem+'.air');library=output/(stem+'.metallib');command=['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-c',str(msl),'-o',str(air)]
            subprocess.run(command,check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
            record['compile_command']=command;record['library_sha256']=sha256(library.read_bytes())
        (output/(stem+'.translation.json')).write_text(json.dumps(record,indent=2)+'\n');results[mode]=record
    return results


def build(output,compile_metal=True):return build_selected(output,POST,scratch_geometry(2160),compile_metal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('build/fsr412_upstream'));parser.add_argument('--no-compile',action='store_true')
    args=parser.parse_args();result=build(args.out,not args.no_compile);print(json.dumps({mode:record['metal_sha256'] for mode,record in result.items()}))
