# SPDX-License-Identifier: MIT
"""Independent static permutation classifier for pinned original Lite DXIL.

Flags are inferred from typed resource handles, used cbuffer lanes, depth
comparisons, transfer-function coefficients, tensor pitches and actual baked
matrix/quantizer evidence. Names select only the old table role, never flags.
No DLL/GPU execution or upstream classifier code is used.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import struct
from upstream_oracle import ENTRY_PREFIX, LOSSY_SHA256, require, scratch_geometry, sha256
from lossy_weights import _DOT, _cnb_weights, _literal_store_blocks, _grouped_matrix, _table_map, _quantizers


def body(text):
    start=text.index('define void @');start=text.index('{',start)+1
    return text[start:text.index('\n}',start)]


def handles(text):
    result={}
    legacy=re.compile(r'(%[\w.]+) = call %dx.types.Handle @dx.op.createHandle\(i32 57, i8 (\d+), i32 \d+, i32 (\d+),')
    bound=re.compile(r'(%[\w.]+) = call %dx.types.Handle @dx.op.createHandleFromBinding\(i32 217, %dx.types.ResBind \{ i32 (\d+), i32 \d+, i32 (\d+), i8 (\d+) \}')
    for match in legacy.finditer(text):result[match[1]]=(int(match[2]),0,int(match[3]))
    for match in bound.finditer(text):result[match[1]]=(int(match[4]),int(match[3]),int(match[2]))
    annotations=re.findall(r'(%[\w.]+) = call %dx.types.Handle @dx.op.annotateHandle\(i32 216, %dx.types.Handle (%[\w.]+),',text)
    for _ in range(4):
        for dst,src in annotations:
            if src in result:result[dst]=result[src]
    require(result,'no resource handles resolved')
    return result


def float_constants(text):
    constants=set()
    for token in re.findall(r'0x[0-9A-F]{16}|-?\d+\.\d+e[+-]\d+',body(text)):
        constants.add(struct.unpack('>d',bytes.fromhex(token[2:]))[0] if token.startswith('0x') else float(token))
    return constants


def colorspace(text,resources):
    constants=float_constants(text)
    def has(value):return any(abs(candidate-value)<=max(1e-9,abs(value)*1e-6) for candidate in constants)
    pq=has(2610/16384) or has(2413/128) and has(2392/128)
    srgb=has(2.4) and (has(12.92) or has(.04045))
    linear=has(150.0) and not pq and not srgb
    require(sum((pq,srgb,linear))<=1,'ambiguous transfer signature')
    value=3 if pq else 2 if srgb else 0 if linear else 1
    if value==0:require(any(r['hlsl_binding'] in ('t6','t17') for r in resources),'linear exposure binding missing')
    if value==1:require(not has(150.0),'nonlinear shader has unexpected Mu-law transfer')
    return value,{'pq_m1_or_c2_c3':pq,'srgb_2p4_and_threshold':srgb,'mu_law_150':has(150.0),'constant_count':len(constants)}


def pre_flags(text,resources):
    resolved=handles(text);low=any(r['hlsl_binding']=='t2' for r in resources)
    depth_results=set()
    for match in re.finditer(r'(%[\w.]+) = call %dx.types.ResRet.f32 @dx.op.textureLoad.f32\(i32 66, %dx.types.Handle (%[\w.]+),',text):
        if resolved.get(match[2])==(0,0,2):depth_results.add(match[1])
    depth_scalars=set()
    for match in re.finditer(r'(%[\w.]+) = extractvalue %dx.types.ResRet.f32 (%[\w.]+), 0',text):
        if match[2] in depth_results:depth_scalars.add(match[1])
    comparisons=Counter()
    for match in re.finditer(r'fcmp(?: fast)? (ogt|olt) float (%[\w.]+), (%[\w.]+)',text):
        if match[2] in depth_scalars:comparisons[match[1]]+=1
    if low:
        require(comparisons in (Counter(ogt=8),Counter(olt=8)),'depth selection is not eight observed min/max comparisons')
        inverted=int(comparisons['ogt']==8)
    else:
        require(not depth_results and not comparisons,'full-MV shader unexpectedly consumes depth')
        inverted=None
    lanes=set()
    for match in re.finditer(r'(%[\w.]+) = call %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32\(i32 59, %dx.types.Handle (%[\w.]+), i32 3\)',text):
        if resolved.get(match[2])!=(2,0,1):continue
        for lane in re.findall(r'extractvalue %dx.types.CBufRet.f32 '+re.escape(match[1])+r', ([23])\b',text):lanes.add(int(lane))
    require(lanes in (set(),{2,3}),'partial jitter cancellation signature')
    cs,transfer=colorspace(text,resources)
    input_exposure=any(r['hlsl_binding']=='t6' and r['name']=='r_input_exposure' for r in resources)
    # AMD LoadExposureValue(): FFX_MLSR_AUTOEXPOSURE_ENABLED=1 reads only
    # r_auto_exposure_texture; 0 reads the caller's r_input_exposure (t6).
    # PRE exposure is unused for perceptual colour spaces. Both AE flags
    # select byte-identical PRE containers there, not an assumed fallback.
    ae=int(not input_exposure) if cs==0 else None
    return {'low_res_mv':int(low),'depth_inverted':inverted,'jittered_mv':int(bool(lanes)),
            'colorspace':cs,'auto_exposure':ae}, {'depth_comparisons':dict(comparisons),'jitter_cbuffer_3_lanes':sorted(lanes),'transfer':transfer,'input_exposure_binding':input_exposure}


def tier(stage,text):
    if stage=='border':
        values=[level for level in (1080,2160,4320) if re.search(r'\b'+str(scratch_geometry(level)['quarter_plane_bytes'])+r'\b',body(text))]
        require(len(values)==1,'border plane size is not unique');return values[0]
    rows={int(v) for v in re.findall(r'\b(7712|15392|30752|61472)\b',body(text))}
    mapping={7712:1080,15392:2160,30752:4320} if stage=='pass2' else {15392:1080,30752:2160,61472:4320}
    require(rows and max(rows) in mapping,'unknown tensor pitch signature')
    return mapping[max(rows)]


def preset(stage,text,bases):
    if stage in ('pre','border'):return None,{'initializer_selection':'runtime t18; identical/shared container across presets'}
    if stage in ('pass1','post'):
        dots=list(_DOT.finditer(text));A=_cnb_weights(dots[:1408])['A']
    elif stage=='pass2':
        blocks=_literal_store_blocks(text);A=_grouped_matrix([v for _,chunk in blocks[:36] for v in chunk],16,9,True)
    else:
        shifts=Counter(re.findall(r'\bashr i32 [^,]+, (\d+)',text))
        value='normal' if shifts.get('6')==4 else 'ultraperformance'
        _quantizers('pass3',value,text)
        return value,{'quantizer_evidence':'complete pass3 integer-shift multiset verified'}
    candidates=[name for name,data in bases.items() if _table_map('post' if stage=='post' else stage,data)['A']==A]
    require(len(candidates)==1,'baked A does not select exactly one verified Lite preset')
    return candidates[0],{'unchanged_baked_A_matches_full_t18_preset':candidates[0]}


def classify(assets):
    assets=Path(assets);manifest=json.loads((assets/'inventory.json').read_text())
    require(manifest['dll_sha256']==LOSSY_SHA256,'unrecognized target')
    bases={}
    for name,info in manifest['initializers'].items():
        data=(assets/info['path']).read_bytes();require(len(data)==131072 and sha256(data)==info['sha256'],'full initializer identity changed')
        bases[name]=data[:27392]
    roles={'prepass':'pre','postpass':'post','pass1':'pass1','pass2':'pass2','pass3':'border','pass4':'pass3'}
    cache={};records=[]
    for record in manifest['records']:
        if record['reference_program_version']!=0x50066:continue
        old=record['reference_entry'].removeprefix(ENTRY_PREFIX)
        if old not in roles or not record['changed']:continue
        stage=roles[old];reference=record['reference_sha256'];target=record['lossy']['sha256'];key=(reference,target,stage)
        if key not in cache:
            ref=(assets/(reference+'.ll')).read_text();text=(assets/(target+'.ll')).read_text();resources=manifest['shaders'][reference]['resources']
            require(sha256(ref.encode())==manifest['shaders'][reference]['listing_sha256'] and sha256(text.encode())==manifest['shaders'][target]['listing_sha256'],'DXC listing identity changed')
            flags,evidence=pre_flags(ref,resources) if stage=='pre' else ({},{})
            if stage=='post':
                cs,transfer=colorspace(ref,resources);flags={'colorspace':cs,'auto_exposure':int(not any(r['hlsl_binding']=='t6' and r['name']=='r_input_exposure' for r in resources))};evidence={'transfer':transfer}
            value,weights=preset(stage,text,bases)
            cache[key]={'stage':stage,'tier':tier(stage,text),'preset':value,'flags':flags,'evidence':{**evidence,**weights},'dxil_sha256':target,'workgroup':manifest['shaders'][target]['workgroup'],'resources':manifest['shaders'][target]['resources']}
        records.append({**cache[key],'table_offset':record['table_offset'],'reference_header_hash':record['reference_header_hash'],'reference_sha256':reference})
    counts=Counter(r['stage'] for r in records)
    require(counts==Counter(pre=180,post=48,pass1=6,pass2=6,border=6,pass3=6),'active original table coverage changed: '+str(counts))
    result={'schema':'fsr412_original_permutation_index_v1','dll_sha256':LOSSY_SHA256,'authority':'static typed DXIL/resource/initializer evidence','records':records,'table_counts':dict(counts),'limits':['No Windows runtime selector/allocation observation.','AE is a shared/ignored PRE permutation for perceptual colour spaces, established by equal shader bytes.','Full-MV PRE does not consume depth, so inversion is an equivalent ignored flag.']}
    (assets/'permutation-index.json').write_text(json.dumps(result,indent=2)+'\n');return result


def select(index,options):
    require(options['tier'] in (1080,2160,4320),'unsupported tier')
    require(options['preset'] in ('normal','ultraperformance'),'unsupported preset')
    for name in ('depth_inverted','low_res_mv','jittered_mv','auto_exposure'):require(options[name] in (0,1),'invalid flag '+name)
    require(options['colorspace'] in (0,1,2,3),'invalid colorspace')
    result={}
    for stage in ('pre','pass1','pass2','border','pass3','post'):
        candidates=[r for r in index['records'] if r['stage']==stage and r['tier']==options['tier'] and r['preset'] in (None,options['preset']) and all(value is None or options[key]==value for key,value in r['flags'].items())]
        hashes=set(r['dxil_sha256'] for r in candidates)
        require(len(hashes)==1,f'{stage}: expected one unique/equivalent shader, got {len(hashes)}')
        first=candidates[0];result[stage]={**first,'table_offsets':[r['table_offset'] for r in candidates],'equivalent_records':len(candidates)}
    return {'schema':'fsr412_original_selected_permutation_v1','options':options,'source_dll_sha256':LOSSY_SHA256,'kernels':result,'geometry':scratch_geometry(options['tier']),'initializer_file':'initializers-'+options['preset']+'.bin','flags_selected_from':'typed resource handles/used cbuffer lanes/transfer constants/baked initializer A, not filenames'}
