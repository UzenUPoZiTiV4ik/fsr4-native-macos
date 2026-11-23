# SPDX-License-Identifier: MIT
"""Compile a pinned external original shader with a correctness profile.

No shader arithmetic/source transformations. -O0 isolates Metal optimizer
failures from DXIL/SPIR-V translation failures. The baseline is preserved.
"""
import argparse
import json
from pathlib import Path
from upstream_oracle import ACTIVE_2160, build_metallib, external_output, require, verified_translation


def build(stage='pass2',output=Path('build/fsr412_upstream')):
    output=external_output(output);require(stage in ACTIVE_2160,'unknown original stage');identity=ACTIVE_2160[stage]
    receipt,text=verified_translation(output,identity,'original translated source changed')
    stem=identity+'.o0';msl=output/(stem+'.metal');msl.write_text(text)
    record=dict(receipt);record['schema']='fsr412_original_correctness_compile_v1';record['stage']=stage
    record['optimization_profile']='-O0';record['compile_command'],record['library_sha256']=build_metallib(output,stem,'-O0')
    record['library']=stem+'.metallib';record['source_unchanged']=True
    record['equivalence']='requires original integer DXIL CPU SSA/GPU validation, not implied by compiler success'
    (output/(stem+'.translation.json')).write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',default='pass2');args=parser.parse_args();print(json.dumps(build(args.stage)))
