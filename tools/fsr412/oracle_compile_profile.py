# SPDX-License-Identifier: MIT
"""Compile a pinned external original shader with a correctness profile.

No shader arithmetic/source transformations. -O0 isolates Metal optimizer
failures from DXIL/SPIR-V translation failures. The baseline is preserved.
"""
import argparse
import json
from pathlib import Path
import subprocess
from upstream_oracle import ACTIVE_2160, external_output, require, sha256


def build(stage='pass2',output=Path('build/fsr412_upstream')):
    output=external_output(output);require(stage in ACTIVE_2160,'unknown original stage');identity=ACTIVE_2160[stage]
    receipt=json.loads((output/(identity+'.translation.json')).read_text());source=(output/(identity+'.metal')).read_bytes()
    require(sha256(source)==receipt['metal_sha256'],'original translated source changed')
    stem=identity+'.o0';msl=output/(stem+'.metal');air=output/(stem+'.air');library=output/(stem+'.metallib');msl.write_bytes(source)
    command=['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-O0','-c',str(msl),'-o',str(air)]
    subprocess.run(command,check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
    record=dict(receipt);record['schema']='fsr412_original_correctness_compile_v1';record['stage']=stage
    record['optimization_profile']='-O0';record['compile_command']=command;record['library']=library.name;record['library_sha256']=sha256(library.read_bytes());record['source_unchanged']=True
    record['equivalence']='requires original integer DXIL CPU SSA/GPU validation, not implied by compiler success'
    (output/(stem+'.translation.json')).write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',default='pass2');args=parser.parse_args();print(json.dumps(build(args.stage)))
