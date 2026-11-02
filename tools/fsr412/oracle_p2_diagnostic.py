# SPDX-License-Identifier: MIT
"""Add bounded diagnostic writes to an external original P2 MSL translation.

No shader code is embedded or vendored. Four A packed words, 64 B accumulator
words, 32 C accumulator words and b0 dimensions are written by invocation 0.
The extra buffer is Metal slot 5. This tool compiles but never runs the GPU.
"""
from pathlib import Path
import json
import re
import subprocess
from upstream_oracle import ACTIVE_2160, require, external_output, sha256


def build(output=Path('build/fsr412_upstream')):
    output=external_output(output);identity=ACTIVE_2160['pass2'];source=(output/(identity+'.metal')).read_text()
    literal=json.loads((output/(identity+'.translation.json')).read_text())
    require(sha256(source.encode())==literal['metal_sha256'],'P2 source identity mismatch')
    entry='kernel void main0('
    require(source.count(entry)==1,'unexpected entry')
    source=source.replace(entry,entry+'device uint *oracleP2Trace [[buffer(5)]], ')
    probes=[('uint _4848 = (_721 + 74889792u) >> 2u;', '_47',4,0),
            ('uint _9301 = _722 >> 2u;','_56',64,4),
            ('uint4 _12979 = as_type<uint4>(oracle_b0_buffer4._m0[4u]);','_61',32,68)]
    for anchor,array,count,offset in probes:
        require(source.count(anchor)==1,'diagnostic SSA anchor changed')
        writes=f'if (gl_GlobalInvocationID.x==0u && gl_GlobalInvocationID.y==0u && gl_GlobalInvocationID.z==0u) {{ for(uint i=0;i<{count}u;++i) oracleP2Trace[{offset}u+i]={array}[i]; }}\n            '
        source=source.replace(anchor,writes+anchor)
    anchor=probes[-1][0]
    source=source.replace(anchor,anchor+'\n            if (gl_GlobalInvocationID.x==0u && gl_GlobalInvocationID.y==0u) { oracleP2Trace[100]=_12979.x; oracleP2Trace[101]=_12979.y; }')
    stem=identity+'.probe';msl=output/(stem+'.metal');air=output/(stem+'.air');library=output/(stem+'.metallib')
    msl.write_text(source);subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-c',str(msl),'-o',str(air)],check=True)
    subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
    receipt={'schema':'fsr412_p2_ssa_diagnostic_v1','original_dxil_sha256':identity,'literal_metal_sha256':literal['metal_sha256'],'probe_metal_sha256':sha256(source.encode()),'library':library.name,'library_sha256':sha256(library.read_bytes()),'extra_buffer_slot':5,'extra_buffer_bytes':408,'layout':{'A_packed_words':[0,4],'B_accumulator_words':[4,64],'C_accumulator_words':[68,32],'b0_width_height':[100,2]},'executed':False}
    (output/(stem+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':print(json.dumps(build()))
