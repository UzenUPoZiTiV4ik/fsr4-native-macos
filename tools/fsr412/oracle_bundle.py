# SPDX-License-Identifier: MIT
"""Build an external, production-like bundle for one statically selected shader set.

This independent loader source copies/compiles only external artifact data.
It does not execute the DLL/GPU or claim untested Windows/runtime equivalence.
"""
from pathlib import Path
import json
import subprocess
from upstream_oracle import external_output, require, sha256, translate, NOOP_SHA256
from oracle_variants import build_selected
from oracle_pass1_border import build as build_pass1_border
from lossy_weights import decode_post
import struct


def resource_abi(selected,inventory):
    resources={}
    for kernel in selected['kernels'].values():
        for row in inventory['shaders'][kernel['reference_sha256']]['resources']:
            register=row['hlsl_binding'].replace('cb','b')
            if register not in resources:resources[register]={'name':row['name'],'reflection_type':row['type'],'reflection_format':row['format'],'reflection_dimension':row['dimension']}
    options=selected['options'];geometry=selected['geometry']
    for register,row in resources.items():
        if register=='t18':row.update(kind='buffer',bytes=131072,file=selected['initializer_file'])
        elif register=='u11':row.update(kind='buffer',bytes=geometry['allocation_bytes'],persistent=True)
        elif register=='b0':row.update(kind='buffer',bytes=144,shape='uint4[9]',slot4='outputW/4,outputH/4,0,0')
        elif register=='b1':row.update(kind='buffer',bytes=104,metal_allocation_bytes=112)
        elif register.startswith('s'):row.update(kind='sampler',fixture='linear clamp-to-edge normalized; Windows state unobserved')
        else:
            size='output' if register in ('t0','t4','t9','u1','u2','u3','u6') else ('render' if options['low_res_mv'] else 'output') if register=='t1' else [1,1] if register=='t6' else [2,1] if register in ('t17','u9') else 'render'
            row.update(kind='texture',size=size)
            if register in ('t4','u6'):row['fixture_storage']='RGBA8Unorm; Windows physical format unobserved'
            if register=='t9':row['alias']='u3'
    return resources


def compile_source(bundle,stem,optimization=None):
    msl=bundle/(stem+'.metal');air=bundle/(stem+'.air');library=bundle/(stem+'.metallib')
    command=['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math']
    if optimization:command.append(optimization)
    command+=['-c',str(msl),'-o',str(air)];subprocess.run(command,check=True)
    subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)],check=True)
    return {'compile_command':command,'optimization_profile':optimization or 'default','library_sha256':sha256(library.read_bytes())}


def build_bundle(assets,bundle,selected,converter,cross,compile_metal=False,post_mode='isolated'):
    assets=Path(assets).resolve();bundle=external_output(Path(bundle));bundle.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((assets/'inventory.json').read_text());geometry=selected['geometry'];resources=resource_abi(selected,inventory)
    previous=json.loads((bundle/'bindings.json').read_text()) if (bundle/'bindings.json').is_file() else {}
    initializer=(assets/selected['initializer_file']).read_bytes();require(len(initializer)==131072,'full t18 initializer required')
    (bundle/selected['initializer_file']).write_bytes(initializer)
    resources['t18']['sha256']=sha256(initializer);kernels={};variants={}
    for stage,kernel in selected['kernels'].items():
        identity=kernel['dxil_sha256']
        for suffix in ('.dxil','.ll'):
            data=(assets/(identity+suffix)).read_bytes()
            if suffix=='.dxil':require(sha256(data)==identity,'selected DXIL source identity mismatch')
            (bundle/(identity+suffix)).write_bytes(data)
        receipt=translate(bundle,identity,converter,cross,geometry['allocation_bytes'])
        stem=identity
        if stage=='post' and geometry['saved_cnb_byte_offset'] is not None:
            variants=build_selected(bundle,identity,geometry,False)
            if post_mode!='literal':receipt=variants[post_mode];stem=identity+'.'+post_mode
        elif stage=='post':post_mode='literal-no-skip-cache'
        optimization='-O0' if stage in ('pass2','pass3') else None
        row={'entry':receipt['entry'],'library':stem+'.metallib','source':stem+'.metal','dxil_sha256':identity,
             'metal_sha256':receipt['metal_sha256'],'wgsize':kernel['workgroup'],'buffers':[],'textures':[],'samplers':[],
             'typed_selection_evidence':kernel['evidence'],'table_offsets':kernel['table_offsets'],
             'optimization_profile':optimization or 'default','IO_guard':receipt['io_guard']}
        for binding in receipt['bindings']:
            register=binding['d3d_register'];require(register in resources,'actual target binding has no original reflection role: '+register)
            bucket={'buffer':'buffers','texture':'textures','sampler':'samplers'}[binding['metal_kind']]
            row[bucket].append([binding['metal_slot'],resources[register]['name']])
        if stage=='post' and post_mode!='literal-no-skip-cache':
            for key in ('group_y_offset_buffer_slot','cache_original_byte_offset','cache_relocated_byte_offset','cache_copy_bytes','physical_scratch_bytes','cache_copy_policy','dispatch_policy'):
                if key in receipt:row[key]=receipt[key]
            if post_mode=='isolated':resources['u11']['physical_allocation_bytes']=receipt['physical_scratch_bytes']
        if compile_metal:row.update(compile_source(bundle,stem,optimization))
        else:
            cached=previous.get('kernels',{}).get(stage,{})
            library=bundle/row['library']
            if library.is_file() and cached.get('metal_sha256')==row['metal_sha256'] and cached.get('optimization_profile')==row['optimization_profile'] and cached.get('library_sha256')==sha256(library.read_bytes()):
                for key in ('compile_command','library_sha256'):row[key]=cached[key]
        row['compiled']=bool(row.get('library_sha256'))
        kernels[stage]=row
        (bundle/(stem+'.translation.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    helper=build_pass1_border(bundle,selected['kernels']['pass1']['dxil_sha256'],compile_metal)
    helper_row={'entry':helper['entry'],'library':helper['library'],'source':selected['kernels']['pass1']['dxil_sha256']+'.border-only.metal',
                'dxil_sha256':helper['dxil_sha256'],'metal_sha256':helper['metal_sha256'],'wgsize':helper['wgsize'],
                'dispatch_groups':[1,1,1],'execute_policy':helper['execute_policy'],'skip_marker_gate_preserved':helper['skip_marker_gate_preserved'],
                'buffers':[],'textures':[],'samplers':[]}
    for binding in helper['bindings']:
        helper_row['buffers'].append([binding['metal_slot'],resources[binding['d3d_register']]['name']])
    if 'library_sha256' in helper:helper_row['library_sha256']=helper['library_sha256']
    helper_row['compiled']=bool(helper_row.get('library_sha256'))
    kernels['pass1_border']=helper_row
    old_entries=['prepass','pass0_post']
    for i in range(1,13):old_entries.extend([f'pass{i}',f'pass{i}_post'])
    old_entries.append('postpass');roles={'prepass':'pre','pass1':'pass1','pass2':'pass2','pass3':'border','pass4':'pass3','postpass':'post'}
    mapping=[{'old_dispatch_index':i,'old_entry':entry,'replacement':roles.get(entry,'noop'),'sha256':kernels[roles[entry]]['dxil_sha256'] if entry in roles else NOOP_SHA256} for i,entry in enumerate(old_entries)]
    result={'schema':'fsr412_original_selected_bindings_v1','source_dll_sha256':selected['source_dll_sha256'],
            'options':{**selected['options'],'ultra_performance':int(selected['options']['preset']=='ultraperformance')},
            'shader_selection_verified':True,'windows_runtime_parity_verified':False,'post_mode':post_mode,
            'initializers':selected['initializer_file'],'initializers_sha256':sha256(initializer),'initializer_bytes':131072,
            'scratch_bytes':geometry['allocation_bytes'],'scratch_allocation_bytes':resources['u11'].get('physical_allocation_bytes',geometry['allocation_bytes']),
            'geometry':geometry,'resources_by_register':resources,'kernels':kernels,'active_order':['pre','pass1','pass2','border','pass3','post'],'old27_mapping':mapping,
            'dispatch':{'pre':'ceil(W/16),ceil(H/16),1','pass1':'ceil((W/2)/64),H/2,1 including original border-only group','pass2':'provider-compatible ceil((W/2)/64),H/2,1; minimal quarter ROI only with preserved sentinel effects','border':'1,1,1 every frame','pass3':'ceil((W/4)/64),H/4,1','post':'ceil(W/32),ceil(H/32),1; obey selected ordered/isolated control'},
            'invalid_store_sentinel':{'byte_offset':134217728,'inside_original_allocation':geometry['allocation_bytes']>134217728,'P2_invalid_output_value':'zero uint4','optimization_constraint':'4320 partial-ROI provider overdispatch may write zero uint4 inside residual plane P; dropping those threads requires preserving this store effect'},
            'native_replacement_order':['pre','native_pass1_cnb','native_pass1_ds','pass1_border','native_pass2','border','native_pass3','post'],
            'native_initializer_contract':{'bytes':27392,'profile':'primary-trained' if selected['options']['preset']=='normal' and selected['options']['tier']<=2160 else 'base-only',
                                           'sha256':'c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc' if selected['options']['preset']=='normal' and selected['options']['tier']<=2160 else sha256(initializer[:27392]),
                                           'not_runtime_t18':True},
            'limits':['All 384 option combinations select unique/equivalent original shader bytes, but GPU quality was tested only on the documented subset.',
                      'Original class-sized raw tensor halo must be preserved; clipping neural convolution input to a smaller viewport changes boundary pixels.',
                      'P2/P3 correctness references require -O0; legacy optimized scalar P2 Metal compilation failed original DXIL CPU SSA checks.',
                      'Texture formats/sampler state are explicit integration choices until original Windows allocation state is observed.',
                      'No GPL shader source or artifact bytes are stored outside external build/temp paths by this loader.']}
    post_identity=selected['kernels']['post']['dxil_sha256']
    post_tables,conditional=decode_post((assets/(post_identity+'.ll')).read_text(),initializer[:27392])
    for bias in conditional:
        if bias['layer']=='A':
            bias['half_width']='b1.width / 2 (original integer sdiv)';bias['half_height']='b1.height / 2 (original integer sdiv)'
    ct_taps=[list(struct.unpack('<8i',post_tables['bCT'])) for _ in range(4)]
    for bias in conditional:
        if bias['layer']=='CT':ct_taps[bias['tap']][bias['channel']]=bias['conditional_bias']
    result['conditional_post_biases']=conditional;result['post_ct_biases_by_tap']=ct_taps
    (bundle/'selection.json').write_text(json.dumps(selected,indent=2)+'\n');(bundle/'bindings.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
