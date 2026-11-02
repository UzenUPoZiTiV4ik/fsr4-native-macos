# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/fsr412'))
import upstream_oracle as oracle
from oracle_io import guard_scalar_io
import oracle_variants
from oracle_selection import classify, select
from oracle_dxil_probe import Probe, signed
from oracle_pass1_border import force_border_group
import itertools
from collections import Counter


def dxbc(payload=b'x'):
    blob=bytearray(44+len(payload));blob[:4]=b'DXBC'
    struct.pack_into('<III',blob,20,1,len(blob),1);struct.pack_into('<I',blob,32,36)
    blob[36:40]=b'DXIL';struct.pack_into('<I',blob,40,len(payload));blob[44:]=payload
    return bytes(blob)


def pe_fixture(blob):
    data=bytearray(4096);data[:2]=b'MZ';struct.pack_into('<I',data,60,128);data[128:132]=b'PE\0\0'
    struct.pack_into('<H',data,134,2);struct.pack_into('<H',data,148,240);opt=152
    struct.pack_into('<H',data,opt,0x20b);struct.pack_into('<Q',data,opt+24,0x180000000)
    for i,(name,rva,off,size) in enumerate([(b'.rdata',4096,1024,2048),(b'.data',8192,3072,1024)]):
        p=opt+240+40*i;data[p:p+len(name)]=name;struct.pack_into('<IIII',data,p+8,size,rva,size,off)
    data[1024:1024+len(blob)]=blob;struct.pack_into('<QQ',data,3072,len(blob),0x180001000)
    return bytes(data)


class ExtractorTests(unittest.TestCase):
    def test_chunk_bounds_and_overlap_are_checked(self):
        blob=dxbc();self.assertTrue(oracle.valid_dxbc(blob))
        bad=bytearray(blob);struct.pack_into('<I',bad,32,40);self.assertFalse(oracle.valid_dxbc(bad))
        bad=bytearray(blob);struct.pack_into('<I',bad,24,len(blob)+1);self.assertFalse(oracle.valid_dxbc(bad))

    def test_table_follows_file_backed_pointer_not_neighbor_guess(self):
        blob=dxbc(b'\x66\x00\x05\x00'+b'fsr4_model_v07_fp8_no_scale_pass3\0')
        pe=oracle.PE(pe_fixture(blob));records=oracle.shader_records(pe)
        self.assertEqual(len(records),1);self.assertEqual(records[0]['table_offset'],3072)
        self.assertEqual(pe.shader_at(3072),(1024,blob));self.assertEqual(records[0]['reference_program_version'],0x50066)
        with self.assertRaises(ValueError):pe.offset(0x180003000)

    def test_virtual_only_memory_is_not_materialized(self):
        data=bytearray(pe_fixture(dxbc()));struct.pack_into('<I',data,152+240+8,4096)
        pe=oracle.PE(bytes(data))
        with self.assertRaises(ValueError):pe.offset(0x180001fff,2)

    def test_stripped_reflection_names_do_not_hide_bindings(self):
        text='''; Resource Bindings:
;                                   texture    byte r/o T0 t18 1
; ScratchBuffer UAV byte r/w U0 u11 1
target datalayout = ""
define void @main() {
 ret void
}
; NumThreads=(64,1,1)
'''
        value=oracle.listing_metadata(text);self.assertEqual([r['hlsl_binding'] for r in value['resources']],['t18','u11'])
        self.assertEqual(value['workgroup'],[64,1,1]);self.assertTrue(value['noop'])
        self.assertFalse(oracle.listing_metadata(text.replace('ret void','%v = call i32 @dx.op.groupId.i32(i32 94, i32 0)\n ret void'))['noop'])

    def test_real_tensor_extents_and_marker_are_not_native_tight_layout(self):
        expected={1080:(25027392,8342592),2160:(99820992,33273792),4320:(398707392,None)}
        for tier,(size,marker) in expected.items():
            geometry=oracle.scratch_geometry(tier);self.assertEqual(geometry['allocation_bytes'],size);self.assertEqual(geometry['skip_marker_byte_offset'],marker)
        self.assertEqual(oracle.scratch_geometry(2160)['saved_cnb_byte_offset'],49941264)
        with self.assertRaises(ValueError):oracle.scratch_geometry(1440)

    def test_assets_cannot_be_published_into_third_party(self):
        with self.assertRaises(ValueError):oracle.external_output(ROOT/'third_party/fsr412/oracle')
        self.assertEqual(oracle.external_output(ROOT/'build/fsr412_upstream'),(ROOT/'build/fsr412_upstream').resolve())

    def test_readonly_uav_keeps_original_d3d_register_class(self):
        instructions=[(71,[3,33,11]),(71,[3,34,0]),(71,[1,2]),(72,[1,0,24]),(30,[1,4]),(32,[2,12,1]),(59,[2,3,12])]
        words=[0x07230203,0x10000,0,20,0]
        for op,args in instructions:words += [((len(args)+1)<<16)|op]+args
        data=struct.pack('<%dI'%len(words),*words)
        _,inferred=oracle.metal_descriptor_abi(data);_,typed=oracle.metal_descriptor_abi(data,{11:1})
        self.assertEqual(inferred[0]['d3d_register'],'t11');self.assertEqual(typed[0]['d3d_register'],'u11')

    def test_descriptor_namespace_patch_does_not_touch_compute_operations(self):
        instructions=[(71,[3,33,11]),(71,[3,34,0]),(71,[1,2]),(30,[1,4]),(32,[2,12,1]),(59,[2,3,12]),(128,[4,7,5,6])]
        words=[0x07230203,0x10000,0,20,0]
        for op,args in instructions:words += [((len(args)+1)<<16)|op]+args
        transformed,bindings=oracle.metal_descriptor_abi(struct.pack('<%dI'%len(words),*words))
        _,ops=oracle._spv_instructions(transformed)
        self.assertIn((128,[4,7,5,6]),ops);self.assertEqual(bindings[0]['d3d_register'],'u11')
        self.assertIn((71,[3,34,1]),ops);self.assertIn((71,[3,33,0]),ops)


class GuardTests(unittest.TestCase):
    def source(self,body,view='uint'):
        return f'''#include <metal_stdlib>
struct S {{ {view} _m0[1]; }};
kernel void main0(device S& oracle_u11_buffer0 [[buffer(0)]], texture2d<float> oracle_t3_texture0 [[texture(0)]])
{{
{body}
}}
'''
    bindings=[{'name':'oracle_u11_buffer0','d3d_register':'u11','metal_kind':'buffer'},
              {'name':'oracle_t3_texture0','d3d_register':'t3','metal_kind':'texture'}]
    def guard(self,body,view='uint'):
        return guard_scalar_io(self.source(body,view),self.bindings,{'u11':64})[0]
    def test_invalid_stores_discard_instead_of_writing_guard_memory(self):
        source=self.guard('oracle_u11_buffer0._m0[33554432u] = value;')
        self.assertIn('OracleStore(oracle_u11_buffer0._m0, uint(33554432u), value, 16u);',source)
        self.assertIn('if (i < n) p[i] = v',source)
    def test_loaded_rhs_is_guarded_before_store_without_overlapping_edits(self):
        source=self.guard('oracle_u11_buffer0._m0[4u] = oracle_u11_buffer0._m0[5u] + 7u;')
        self.assertIn('OracleLoad(oracle_u11_buffer0._m0, uint(5u), 16u) + 7u',source)
    def test_vector_aliases_are_checked_per_physical_word(self):
        source=self.guard('uint4 v = oracle_u11_buffer0._m0[3u];\noracle_u11_buffer0._m0[4u] = v;','uint4')
        self.assertIn('OracleLoad(words,i*N+c,n)',source);self.assertIn('OracleStore(words,i*N+c,v[c],n)',source)
    def test_texture_only_integer_io_is_guarded_sampling_left_intact(self):
        source=self.guard('float4 v=oracle_t3_texture0.read(uint2(1u,2u));\nfloat4 q=oracle_t3_texture0.sample(s,float2(0.5f));')
        self.assertIn('OracleTextureRead(oracle_t3_texture0, uint2(1u,2u))',source)
        self.assertIn('oracle_t3_texture0.sample(s,float2(0.5f))',source)
    def test_atomic_oob_uses_same_word_bound(self):
        source=self.guard('uint old=atomic_fetch_max_explicit((device atomic_uint*)&oracle_u11_buffer0._m0[i], v, memory_order_relaxed);')
        self.assertIn('OracleAtomicMax(oracle_u11_buffer0._m0, uint(i), v, 16u)',source)
    def test_double_guard_rejected(self):
        source=self.guard('uint v=oracle_u11_buffer0._m0[0u];')
        with self.assertRaises(ValueError):guard_scalar_io(source,self.bindings,{'u11':64})


class ControlledVariantTests(unittest.TestCase):
    def test_ordered_changes_coordinates_not_shader_math(self):
        source='''kernel void main0(device S& p [[buffer(0)]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
float value=fma(x,y,z);
uint row=gl_WorkGroupID.y << 4u;
}
'''
        result=oracle_variants.ordered(source)
        self.assertIn('float value=fma(x,y,z);',result)
        self.assertIn('gl_WorkGroupID=oracleRawGroupID+uint3(0u,oracleGroupYOffset,0u)',result)
        self.assertIn('oracleGroupYOffset [[buffer(4)]]',result)

    def test_pass1_border_coordinates_do_not_override_skip_return(self):
        source='''kernel void main0(uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
if (skipMarker) return;
uint y=gl_WorkGroupID.y;
}
'''
        result=force_border_group(source)
        self.assertIn('if (skipMarker) return;',result)
        self.assertIn('const uint3 gl_WorkGroupID=uint3(0u,2u,0u)',result)
        self.assertIn('uint y=gl_WorkGroupID.y;',result)

    def test_ordered_ae_variant_offsets_global_y_as_well(self):
        source='''kernel void main0(uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
uint y=gl_GlobalInvocationID.y;
}
'''
        result=oracle_variants.ordered(source)
        self.assertIn('gl_GlobalInvocationID=oracleRawGlobalID+uint3(0u,oracleGroupYOffset,0u)',result)
        self.assertIn('uint y=gl_GlobalInvocationID.y;',result)

    def test_isolation_only_changes_five_cache_bases_and_io_extent(self):
        source='\n'.join(f'uint _{i} = (((_1 * 30752u) + 49941264u) + (_2 << 4u)) >> 2u;' for i in range(5))+'\nfloat value=fma(x,y,z);\nuint constantWeight=24955248u;\nuint value2=OracleLoad(p, i, 24955248u);'
        result=oracle_variants.isolated(source)
        self.assertEqual(result.count(str(oracle_variants.NEW_BASE)+'u'),5)
        self.assertIn('float value=fma(x,y,z);',result)
        self.assertIn(str(oracle_variants.PHYSICAL_BYTES//4)+'u',result)
        self.assertIn('uint constantWeight=24955248u;',result)
        self.assertLess(oracle_variants.PHYSICAL_BYTES,134217728)
        with self.assertRaises(ValueError):oracle_variants.isolated(source.replace('49941264u','123u',1))


class IntegerSSAProbeTests(unittest.TestCase):
    def test_signed_dot_and_round_shift_are_exact(self):
        p=Probe.__new__(Probe);p.env={};p.counts=Counter()
        a=struct.unpack('<I',struct.pack('<4b',-1,2,-3,4))[0];b=struct.unpack('<I',struct.pack('<4b',5,-6,7,-8))[0]
        value=p.call('dot4AddPacked.i32',['i32 163','i32 3','i32 '+str(a),'i32 '+str(b)])
        self.assertEqual(signed(value),-67);p.env['%x']=(-257)&0xffffffff
        self.assertEqual(signed(p.evaluate('ashr i32 %x, 8')),-2)


@unittest.skipUnless((ROOT/'build/fsr412_upstream/permutation-index.json').is_file(),'external permutation inventory not available')
class PermutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.index=classify(ROOT/'build/fsr412_upstream')
    def test_all_384_options_have_unique_or_byte_equivalent_originals(self):
        count=0
        for tier,preset,di,lm,jm,cs,ae in itertools.product((1080,2160,4320),('normal','ultraperformance'),(0,1),(0,1),(0,1),(0,1,2,3),(0,1)):
            value=select(self.index,dict(tier=tier,preset=preset,depth_inverted=di,low_res_mv=lm,jittered_mv=jm,colorspace=cs,auto_exposure=ae))
            self.assertEqual(set(value['kernels']),set(oracle.ACTIVE_2160));count+=1
        self.assertEqual(count,384)
    def test_known_selector_reproduces_original_pinned_six(self):
        # The pinned six read only r_auto_exposure_texture: AMD AUTOEXPOSURE_ENABLED=1.
        value=select(self.index,dict(tier=2160,preset='normal',depth_inverted=1,low_res_mv=1,jittered_mv=0,colorspace=0,auto_exposure=1))
        self.assertEqual({k:r['dxil_sha256'] for k,r in value['kernels'].items()},oracle.ACTIVE_2160)
    def test_auto_exposure_matches_amd_load_exposure_value(self):
        inventory=json.loads((ROOT/'build/fsr412_upstream/inventory.json').read_text())
        for ae in (0,1):
            value=select(self.index,dict(tier=2160,preset='normal',depth_inverted=1,low_res_mv=1,jittered_mv=0,colorspace=0,auto_exposure=ae))
            for stage in ('pre','post'):
                reference=value['kernels'][stage]['reference_sha256']
                names={r['name'] for r in inventory['shaders'][reference]['resources']}
                self.assertEqual('r_input_exposure' in names,ae==0,(stage,ae))
    def test_ignored_pre_flags_are_proven_shared_not_post_fallback(self):
        options=dict(tier=2160,preset='normal',depth_inverted=0,low_res_mv=0,jittered_mv=0,colorspace=1,auto_exposure=0)
        a=select(self.index,options);b=select(self.index,{**options,'depth_inverted':1,'auto_exposure':1})
        self.assertEqual(a['kernels']['pre']['dxil_sha256'],b['kernels']['pre']['dxil_sha256'])
        self.assertNotEqual(a['kernels']['post']['dxil_sha256'],b['kernels']['post']['dxil_sha256'])
    def test_invalid_options_fail_closed(self):
        options=dict(tier=2160,preset='normal',depth_inverted=0,low_res_mv=1,jittered_mv=0,colorspace=7,auto_exposure=0)
        with self.assertRaises(ValueError):select(self.index,options)


@unittest.skipUnless((ROOT/'build/fsr412_upstream-oracle/u8-smoke-literal/receipt.json').is_file(),'original captured DS inputs unavailable')
class CapturedSSATests(unittest.TestCase):
    def test_original_dxil_p2_not_legacy_optimized_gpu_values(self):
        assets=ROOT/'build/fsr412_upstream';p=Probe((assets/(oracle.ACTIVE_2160['pass2']+'.ll')).read_text(),(assets/'initializers-normal.bin').read_bytes(),ROOT/'build/fsr412_upstream-oracle/u8-smoke-literal')
        result=p.run();self.assertEqual(result['output'][:8],[20,-11,-12,-22,-18,38,-9,-39]);self.assertEqual(result['operations']['dot4AddPacked.i32'],1600)


@unittest.skipUnless((ROOT/'build/fsr412_upstream/contract.json').is_file(),'external pinned artifacts have not been imported')
class ArtifactContractTests(unittest.TestCase):
    def test_all_old_noops_and_border_are_bound_to_actual_hashes(self):
        contract=json.loads((ROOT/'build/fsr412_upstream/contract.json').read_text())
        stages=contract['old27_mapping'];self.assertEqual(len(stages),27)
        noops=[s for s in stages if s['replacement']=='noop'];self.assertEqual(len(noops),21)
        self.assertTrue(all(s['replacement_sha256']==[oracle.NOOP_SHA256] for s in noops))
        self.assertEqual(stages[6]['replacement'],'border');self.assertEqual(stages[8]['replacement'],'pass3')
    def test_post_conditional_biases_are_not_flattened_away(self):
        contract=json.loads((ROOT/'build/fsr412_upstream/contract.json').read_text())
        edge=[b for b in contract['conditional_post_biases'] if b['layer']=='A'];self.assertEqual(len(edge),16)
        self.assertEqual(edge[0]['primary_bias'],11);self.assertEqual(edge[0]['edge_bias'],12)
        self.assertEqual(contract['post_ct_biases_by_tap'][0],[120,-303,-156,-92,-113,-152,-400,-149])
        self.assertNotEqual(contract['post_ct_biases_by_tap'][0],contract['post_ct_biases_by_tap'][1])

    def test_all_six_guarded_shaders_have_explicit_aliases_and_full_initializer(self):
        for identity in oracle.ACTIVE_2160.values():
            receipt=json.loads((ROOT/'build/fsr412_upstream'/f'{identity}.translation.json').read_text())
            self.assertEqual(receipt['dxil_sha256'],identity);self.assertGreater(receipt['io_guard']['access_sites'],0)
            self.assertTrue(all(r['same_buffer_alias']==r['d3d_register'] for r in receipt['bindings'] if r['metal_kind']=='buffer'))
        data=(ROOT/'build/fsr412_upstream/initializers-normal.bin').read_bytes();self.assertEqual(len(data),131072)
        self.assertEqual(oracle.sha256(data),'621a99e22c4259c3200097513947d072633f67634eaf95365fb4123b4c56ac10')


if __name__=='__main__':unittest.main()
