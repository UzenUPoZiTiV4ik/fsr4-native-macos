# SPDX-License-Identifier: MIT
"""Translate pinned P2 with native packed-dot SPIR-V as a diagnostic variant.

The baseline asset is preserved. This tool compiles but never runs the GPU.
It uses the caller's existing converter, not a downloaded duplicate build.
"""
import argparse
import json
from pathlib import Path
import subprocess
from upstream_oracle import ACTIVE_2160, external_output, metal_descriptor_abi, require, sha256
from oracle_io import guard_scalar_io


def build(converter,output=Path('build/fsr412_upstream'),cross=Path('tools/bin/spirv-cross')):
    output=external_output(output);identity=ACTIVE_2160['pass2'];stem=identity+'.dot4'
    raw=output/(stem+'.raw.spv');spv=output/(stem+'.spv');msl=output/(stem+'.metal')
    command=[str(converter),str(output/(identity+'.dxil')),'--output',str(raw),'--ssbo-uav','--ssbo-srv','--ssbo-scalar','--no-bda','--use-reflection-names','--enable-shader-i8-dot']
    subprocess.run(command,check=True);patched,bindings=metal_descriptor_abi(raw.read_bytes());spv.write_bytes(patched)
    subprocess.run([str(cross),str(spv),'--msl','--msl-version','30100','--msl-decoration-binding','--output',str(msl)],check=True)
    unguarded=msl.read_text();(output/(stem+'.unguarded.metal')).write_text(unguarded)
    source,io=guard_scalar_io(unguarded,bindings,{'t18':131072,'u11':99820992});msl.write_text(source)
    air=output/(stem+'.air');library=output/(stem+'.metallib')
    subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-c',str(msl),'-o',str(air)],check=True)
    subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
    receipt={'schema':'fsr412_p2_native_dot_diagnostic_v1','dxil_sha256':identity,'entry':'main0','bindings':bindings,'io_guard':io,'converter_command':command,'converter_sha256':sha256(converter.read_bytes()),'spirv_sha256':sha256(patched),'metal_sha256':sha256(source.encode()),'library':library.name,'library_sha256':sha256(library.read_bytes()),'executed':False,'equivalence':'unverified diagnostic; compare against original integer DXIL CPU SSA'}
    (output/(stem+'.translation.json')).write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--converter',type=Path,required=True);args=parser.parse_args();print(json.dumps(build(args.converter)))
