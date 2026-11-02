# SPDX-License-Identifier: MIT
"""Reach only the original fused P1 DS-border branch, without copying its policy.

Force original GroupID=(0,2,0). Preserve the original entry skip-marker gate:
for 1080/2160 this helper is a no-op when the original neural frame is skipped.
Shader bytes/source remain external. No DLL/GPU execution by this compiler.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
from upstream_oracle import ACTIVE_2160, external_output, require, sha256


def force_border_group(source):
    group='uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]'
    require(source.count(group)==1,'original P1 group signature changed')
    source=source.replace(group,'uint3 oracleUnusedGroupID [[threadgroup_position_in_grid]]')
    global_id='uint3 gl_GlobalInvocationID [[thread_position_in_grid]]'
    has_global='gl_GlobalInvocationID' in source
    if has_global:
        require(source.count(global_id)==1 and 'gl_LocalInvocationID' in source,'cannot reconstruct original global/local coordinates')
        source=source.replace(global_id,'uint3 oracleUnusedGlobalID [[thread_position_in_grid]]')
    entry=source.index('kernel void ');brace=source.index('\n{',entry)+2
    coordinates='\n    const uint3 gl_WorkGroupID=uint3(0u,2u,0u);'
    if has_global:coordinates+='\n    const uint3 gl_GlobalInvocationID=uint3(0u,4u,0u)+gl_LocalInvocationID;'
    return source[:brace]+coordinates+source[brace:]


def build(output=Path('build/fsr412_upstream'),identity=None,compile_metal=True):
    output=external_output(output);identity=identity or ACTIVE_2160['pass1'];receipt=json.loads((output/(identity+'.translation.json')).read_text())
    previous_path=output/(identity+'.border-only.translation.json')
    previous=json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    source=(output/(identity+'.metal')).read_text();require(sha256(source.encode())==receipt['metal_sha256'],'original P1 source identity mismatch')
    require(receipt['reflection']['entryPoints'][0]['workgroup_size']==[64,2,1],'unknown original P1 thread geometry')
    text=(output/(identity+'.ll')).read_text();skip='1511506142' in text
    if skip:
        require(re.search(r'br i1 %fs.skip, label %fs.ret, label %fs.cont',text) is not None and 'fs.ret:' in text,'P1 skip gate shape changed')
    source=force_border_group(source);stem=identity+'.border-only';msl=output/(stem+'.metal');msl.write_text(source)
    record=dict(receipt);record['schema']='fsr412_original_p1_border_variant_v1';record['role']='pass1_border';record['library']=stem+'.metallib'
    record['literal_metal_sha256']=receipt['metal_sha256'];record['metal_sha256']=sha256(source.encode());record['dispatch_groups']=[1,1,1];record['wgsize']=[64,2,1]
    record['forced_original_group']=[0,2,0];record['skip_marker_gate_preserved']=skip
    record['execute_policy']='safe to call every frame; original marker gate makes this a no-op on skipped 1080/2160 frames; 4320 runs every frame'
    record['transformation']='group/global coordinate substitution only; original border stores and early skip return unchanged'
    if compile_metal:
        air=output/(stem+'.air');library=output/(stem+'.metallib');command=['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-c',str(msl),'-o',str(air)]
        subprocess.run(command,check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
        record['compile_command']=command;record['library_sha256']=sha256(library.read_bytes())
    else:
        library=output/(stem+'.metallib')
        if library.is_file() and previous.get('metal_sha256')==record['metal_sha256'] and previous.get('library_sha256')==sha256(library.read_bytes()):
            record['library_sha256']=previous['library_sha256'];record['compile_command']=previous['compile_command']
    (output/(stem+'.translation.json')).write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('build/fsr412_upstream'));parser.add_argument('--shader-sha256');parser.add_argument('--no-compile',action='store_true')
    args=parser.parse_args();print(json.dumps(build(args.out,args.shader_sha256,not args.no_compile)))
