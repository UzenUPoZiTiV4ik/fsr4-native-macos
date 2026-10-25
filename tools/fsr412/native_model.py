#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pack verified AMD FSR 4.1.2 INT8 Lite weights for the native Metal ABI.

No 4.1.1 model coefficients are used. The reusable Metal kernels implement
integer convolution operators; every weight, bias and quantizer below comes
from the supplied 4.1.2 Lite initializer and its independently inspected DXIL
and SPIR-V. CNB+downsample is split into two dispatches while preserving the
three-stage Lite graph. Native output parity is checked separately on GPU.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, struct, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON_SHA = "4b9898da719dc2dca5dc45fd71dec5f80bdb9886a580fe3038af59efe637d55e"
ULTRA_SHA = "4c57292d9963e82dec86134060eef0c2b66214cd1a31a9e9bc73739b5fe2d1d7"
NONE=0xffffffff

def digest(data): return hashlib.sha256(data).hexdigest()
def signed(x): return x if x < 128 else x-256

def atomic(path,data):
    if path.exists() and path.read_bytes()==data:return
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()

def model(tier, preset, initializer):
    expected=ULTRA_SHA if preset=='ultraperformance' else COMMON_SHA
    if len(initializer)!=27392 or digest(initializer)!=expected:
        raise ValueError('FSR 4.1.2 Lite initializer identity mismatch')
    ultra=preset=='ultraperformance'
    w=tier*8//9;h=tier//2
    row=(w+2)*16;plane=row*(h+2)
    qw=w//2;qh=h//2;qrow=(qw+2)*16;bank=qrow*(qh+2)
    def tensor(level,base,channels):
        width,height,pitch,b=(w,h,row,plane) if level==0 else (qw,qh,qrow,bank)
        return [base+pitch+16,16,pitch,width,height,channels,b if channels>16 else 0]
    def get(off,n): return list(struct.unpack_from('<'+str(n)+'i',initializer,off))
    def conv(off,cin=16,cout=16,taps=9):
        return [signed(initializer[off+tap*cin*cout+oc*cin+ic])
                for tap in range(taps) for ic in range(cin) for oc in range(cout)]
    def pw(off,cin,cout):
        return [signed(initializer[off+oc*cin+ic]) for ic in range(cin) for oc in range(cout)]
    def ct(off,cin,cout):
        return [signed(initializer[off+tap*cin*cout+oc*cin+ic])
                for ic in range(cin) for tap in range(4) for oc in range(cout)]
    z=[0,0,0,0]
    def cnb(name,off,bias,shifts,tail,inp,outp,islot,oslot):
        return dict(kind='cnb',kernel='fsr411_cnb_coop_chain_32x1',tile=[32,1],threads=32,
            input=inp,output=outp,input_slot=islot,output_slot=oslot,source=name,
            matrices=[conv(off[0]),pw(off[1],16,32),pw(off[2],32,16)],
            bias_values=[get(bias[0],16),get(bias[1],32),get(bias[2],16)],
            initializer_weight_offsets=off,initializer_bias_offsets=bias,
            q=[[0,0,shifts[0],0],[0,0,shifts[1],1],tail,tail,z,z])
    # PRE A -> CNB B. The fused driver then downsamples B into C. Separate
    # dispatches use the same disjoint physical tensor ranges.
    p1=cnb('AMD Lite pass1 CNB',[1152,3584,4224],[3456,4096,4736],
        [8,8] if ultra else [7,9],[0,5,6,0],tensor(0,0,16),tensor(0,plane,16),0,2)
    p4=dict(kind='ds',kernel='fsr411_ds16',tile=[64,1],threads=128,
        input=tensor(0,plane,16),output=tensor(1,2*plane,32),input_slot=2,output_slot=3,
        source='AMD Lite pass1 downsample',matrices=[conv(4864,16,32,4)],
        bias_values=[get(6912,32)],initializer_weight_offsets=[4864],initializer_bias_offsets=[6912],
        q=[z,z,z,z,[0,0,7 if ultra else 6,0],[0,0,8 if ultra else 7,0]])
    p2=dict(kind='fnb',kernel='fsr411_fnb32_coop_chain_16x1',tile=[16,1],threads=32,
        input=tensor(1,2*plane,32),output=tensor(1,0,32),input_slot=3,output_slot=4,
        source='AMD Lite pass2 FNB32',matrices=[conv(7040),pw(9472,32,64),pw(11776,64,32)],
        bias_values=[get(9344,16),get(11520,64),get(13824,32)],
        initializer_weight_offsets=[7040,9472,11776],initializer_bias_offsets=[9344,11520,13824],
        q=[[0,0,8,0],[0,0,8,1],[0,5,6,0],[0,6,7,0],z,z])
    p3=dict(kind='ct',kernel='fsr411_ct32_coop_chain_16x1',tile=[16,1],threads=32,
        input=tensor(1,0,32),output=tensor(0,2*plane,16),skip=tensor(0,plane,16),
        input_slot=4,output_slot=5,skip_slot=2,
        source='AMD Lite pass3 FNB32 CT2D',matrices=[conv(13952),pw(16384,32,64),pw(18688,64,32),ct(20864,32,16)],
        bias_values=[get(16256,16),get(18432,64),get(20736,32),get(22912,16)],
        initializer_weight_offsets=[13952,16384,18688,20864],initializer_bias_offsets=[16256,18432,20736,22912],
        q=[[0,0,9 if ultra else 8,0],[0,0,7,1],
           [0,6,7,0] if ultra else [0,6,6,0], [0,7,7,0] if ultra else [0,7,6,0],
           [0,7,8,0] if ultra else [0,8,8,0],[0,7,8,0] if ultra else [0,8,8,0]])
    # Ultra POST rounds its 3x3 CNB by 2^7 and resolves residuals as
    # rint((x/16 + acc/1024)*8); normal uses 2^8 and rint((x/32 + acc/1024)*16).
    post=cnb('AMD Lite POST CNB',[23040,25472,26112],[25344,25984,26624],
        [7,8] if ultra else [8,8],[0,6,7,0] if ultra else [0,5,6,0],tensor(0,2*plane,16),
        [0,16,w*16,w,h,16,0],5,5)
    post.update(before_stage=9,output_offset_base='scratch_append',scratch_append_alignment=256,scratch_append_bytes=w*h*16)
    return {1:p1,4:p4,2:p2,3:p3},post

def pack(tier,preset,initializer,out):
    passes,post=model(tier,preset,initializer);blob=bytearray();params=bytearray()
    def append(data):
        blob.extend(bytes((-len(blob))%64));off=len(blob);blob.extend(data);return off
    def emit(p):
        matrices=p.pop('matrices');biases=p.pop('bias_values')
        ws=[append(bytes(x&255 for x in m)) for m in matrices]
        bs=[append(struct.pack('<'+str(len(b))+'i',*b)) for b in biases]
        ws += [NONE]*(4-len(ws));bs += [NONE]*(4-len(bs))
        words=p['input']+p['output']+p.get('skip',[0]*7)+ws+bs+sum(p['q'],[])
        off=len(params);params.extend(struct.pack('<53I',*words));params.extend(bytes(256-212))
        nf=p['input'][5];kind=p['kind'];tile=[32 if nf==16 or (nf==32 and kind!='ds') else 16,4]
        e={**p,'parameter_offset':off,'weights':ws,'biases':bs,'function':p['kernel'],
            'dispatch_space':'input' if kind=='ct' else 'output',
            'reference':{'function':'fsr411_ref_'+kind,'tile':[32,1],'threads':32},
            'simd':{'function':'fsr411_simd_'+(kind if kind=='cnb' else kind+str(nf)),'tile':tile,'threads':128},
            'source_sha256':digest(initializer),'accumulator_bounds':[],
            'coefficient_provenance':'verified AMD FSR4.1.2 Lite initializer, no 4.1.1 coefficients'}
        return e
    entries={str(k):emit(v) for k,v in passes.items()};pe=emit(post)
    m={'schema':'fsr411_native_v1','abi_version':1,'parameter_size':212,'parameter_stride':256,
       'tier':tier,'model':preset,'fsr_version':'4.1.2','profile':'int8-lite','passes':entries,'auxiliary_passes':{'post_cnb':pe},
       'initializer_sha256':digest(initializer),'initializer_bytes':len(initializer),
       'kernel_source':'src/fsr411_mpp.metal','header_source':'src/fsr411_model.h',
       'weights_bytes':len(blob),'weights_sha256':digest(blob),'params_bytes':len(params),'params_sha256':digest(params)}
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.fsr412_native.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        atomic(out/'fsr411_native_weights.bin',blob);atomic(out/'fsr411_native_params.bin',params)
        atomic(out/'fsr411_native_manifest.json',(json.dumps(m,indent=2)+'\n').encode())
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tier',type=int,choices=(1080,2160,4320),default=1080)
    ap.add_argument('--preset',choices=('normal','ultraperformance'),default='normal');ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--initializer',type=Path)
    a=ap.parse_args();init=a.initializer or ROOT/'build/fsr412_model'/('initializers-lite-ultraperformance.bin' if a.preset=='ultraperformance' else 'initializers-lite.bin')
    m=pack(a.tier,a.preset,init.read_bytes(),a.outdir.resolve())
    print(json.dumps({'model':'FSR4.1.2 INT8 Lite','outdir':str(a.outdir),'weights_bytes':m['weights_bytes'],'params_bytes':m['params_bytes']}))
if __name__=='__main__':main()
