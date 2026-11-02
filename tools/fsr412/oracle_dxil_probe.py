#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded CPU evaluation of original integer P2 DXIL SSA for one invocation.

Reads a DXC listing and captured resources as data. Does not load a DLL, launch
a shader/GPU, use upstream code or infer a model from a filename. Unsupported
LLVM/DXIL operations fail closed. Intended for contract/converter diagnostics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
from upstream_oracle import ACTIVE_2160, require

MASK=(1<<32)-1


def signed(value):
    value &= MASK
    return value-(1<<32) if value & (1<<31) else value


class Probe:
    def __init__(self,listing,initializer,capture,x=0,y=0):
        self.initializer=initializer;self.capture=Path(capture);self.x,self.y=x,y
        receipt=json.loads((self.capture/'receipt.json').read_text());frame=receipt['frames'][0]
        tensors=next(s['tensors'] for s in frame['stages'] if s['stage']=='pass1')
        self.ds=[(tensors[name],(self.capture/tensors[name]['path']).read_bytes()) for name in ('ds0','ds1')]
        self.b0=(self.capture/receipt['tensor_sizes']['path']).read_bytes()
        self.env={};self.outputs={};self.counts=Counter();self.traces={};self.steps=0
        body=listing[listing.index('define void @'):];body=body[body.index('{')+1:body.index('\n}')]
        self.blocks={'0':[]};current='0'
        for line in body.splitlines():
            label=re.match(r'; <label>:(\d+)',line) or re.match(r'([\w.]+):',line)
            if label:current=label[1];self.blocks[current]=[];continue
            line=line.split(';',1)[0].strip()
            if line:self.blocks[current].append(line)

    def value(self,token):
        token=token.strip()
        if token.startswith('%'):return self.env[token]
        if token=='true':return 1
        if token=='false':return 0
        require(token!='undef','attempt to evaluate undefined operand')
        return int(token)&MASK

    def operand(self,typed):
        return self.value(typed.strip().rsplit(' ',1)[-1])

    def raw_read(self,handle,address):
        if handle=='initializer':
            require(0<=address<=len(self.initializer)-4,'initializer load out of bounds')
            return struct.unpack_from('<I',self.initializer,address)[0]
        require(handle=='scratch','unexpected raw resource')
        if address in self.outputs:return self.outputs[address]
        if address in (33273792,33273796):return 0
        for descriptor,data in self.ds:
            relative=address-descriptor['scratch_byte_offset']
            if relative<0:continue
            row,offset=divmod(relative,descriptor['scratch_row_bytes'])
            if row<descriptor['height_cells'] and offset+4<=descriptor['width_cells']*16:
                return struct.unpack_from('<I',data,row*descriptor['width_cells']*16+offset)[0]
        raise ValueError(f'uncaptured scratch read {address}')

    def call(self,operation,args):
        self.counts[operation]+=1
        if operation=='createHandle':
            cls,register=self.operand(args[1]),self.operand(args[3])
            return 'scratch' if cls==1 and register==11 else 'initializer' if cls==0 and register==18 else 'b0' if cls==2 and register==0 else None
        if operation=='threadId.i32':return (self.x,self.y,0)[self.operand(args[1])]
        if operation=='rawBufferLoad.i32':
            handle,address,mask=self.operand(args[1]),self.operand(args[2]),self.operand(args[4])
            return tuple(self.raw_read(handle,address+4*i) if mask&(1<<i) else 0 for i in range(4))+(0,)
        if operation=='rawBufferStore.i32':
            handle,address,mask=self.operand(args[1]),self.operand(args[2]),self.operand(args[8])
            require(handle=='scratch','unexpected store resource')
            for i in range(4):
                if mask&(1<<i):self.outputs[address+4*i]=self.operand(args[4+i])
            return None
        if operation=='cbufferLoadLegacy.i32':
            require(self.operand(args[1])=='b0','unexpected constant buffer')
            return struct.unpack_from('<4I',self.b0,self.operand(args[2])*16)
        if operation=='tertiary.i32':return (self.operand(args[1])*self.operand(args[2])+self.operand(args[3]))&MASK
        if operation=='binary.i32':
            op=self.operand(args[0]);a,b=signed(self.operand(args[1])),signed(self.operand(args[2]))
            require(op in (37,38),'unsupported integer binary opcode')
            return (max(a,b) if op==37 else min(a,b))&MASK
        if operation=='dot4AddPacked.i32':
            require(self.operand(args[0])==163,'non-signed dot')
            acc,a,b=map(self.operand,args[1:]);aa=struct.unpack('<4b',struct.pack('<I',a));bb=struct.unpack('<4b',struct.pack('<I',b))
            return (acc+sum(x*y for x,y in zip(aa,bb)))&MASK
        raise ValueError('unsupported DXIL operation '+operation)

    def evaluate(self,expression):
        op=expression.split(' ',1)[0]
        if op=='alloca':return [None]*int(re.search(r'\[(\d+) x i32\]',expression)[1])
        if op=='getelementptr':
            match=re.search(r'\* (%[\w.]+), i32 0, i32 ([\w.%+-]+)',expression)
            require(match is not None,'unsupported local pointer shape')
            return (self.env[match[1]],self.value(match[2]))
        if op=='load':
            match=re.search(r'i32\* (%[\w.]+)',expression);array,index=self.env[match[1]]
            require(array[index] is not None,'uninitialized private array read')
            return array[index]
        if op=='extractvalue':
            match=re.search(r' (%[\w.]+), (\d+)$',expression);return self.env[match[1]][int(match[2])]
        if op=='call':
            match=re.search(r'@dx\.op\.([\w.]+)\((.*)\)',expression)
            require(match is not None,'unexpected function call')
            return self.call(match[1],match[2].split(','))
        if op=='select':
            args=expression[7:].split(',');return self.operand(args[1] if self.operand(args[0]) else args[2])
        if op=='icmp':
            match=re.fullmatch(r'icmp (\w+) i(?:1|32) ([^,]+), (.+)',expression);pred,a,b=match[1],self.value(match[2]),self.value(match[3])
            if pred.startswith('s'):a,b=signed(a),signed(b)
            compare={'eq':lambda:a==b,'ne':lambda:a!=b,'ult':lambda:a<b,'ule':lambda:a<=b,'ugt':lambda:a>b,'uge':lambda:a>=b,'slt':lambda:a<b,'sle':lambda:a<=b,'sgt':lambda:a>b,'sge':lambda:a>=b}
            return int(compare[pred]())
        match=re.fullmatch(r'(add|mul|shl|lshr|ashr|and|or|xor)(?: (?:nsw|nuw|exact))* i(1|32) ([^,]+), (.+)',expression)
        require(match is not None,'unsupported LLVM expression '+expression)
        op,bits,a,b=match[1],int(match[2]),self.value(match[3]),self.value(match[4])
        compute={'add':lambda:a+b,'mul':lambda:a*b,'shl':lambda:a<<b,'lshr':lambda:a>>b,'ashr':lambda:signed(a)>>b,'and':lambda:a&b,'or':lambda:a|b,'xor':lambda:a^b}
        return compute[op]()&((1<<bits)-1)

    def run(self):
        current,previous='0',None
        while True:
            require(self.steps<1_000_000,'probe instruction limit exceeded')
            lines=self.blocks[current];phis={}
            for line in lines:
                match=re.match(r'(%[\w.]+) = phi i32 (.*)',line)
                if match:
                    incoming=re.findall(r'\[ ([^,]+), %([\w.]+) \]',match[2]);value=next(value for value,label in incoming if label==previous);phis[match[1]]=self.value(value)
            self.env.update(phis)
            branch=None
            for line in lines:
                self.steps+=1
                if ' = phi ' in line:continue
                if line=='ret void':return self.result()
                if line.startswith('br '):
                    match=re.fullmatch(r'br i1 ([^,]+), label %([\w.]+), label %([\w.]+)',line)
                    branch=(match[2] if self.value(match[1]) else match[3]) if match else re.fullmatch(r'br label %([\w.]+)',line)[1]
                    break
                if line.startswith('store '):
                    match=re.match(r'store i32 ([^,]+), i32\* (%[\w.]+)',line);array,index=self.env[match[2]];array[index]=self.value(match[1]);continue
                if line.startswith('call '):
                    self.evaluate(line);continue
                match=re.fullmatch(r'(%[\w.]+) = (.*)',line)
                require(match is not None,'unknown LLVM statement '+line)
                self.env[match[1]]=self.evaluate(match[2])
                if match[1]=='%1641':self.traces['A_packed_before_B']=list(self.env['%7'][:4])
                if match[1]=='%3567':self.traces['B_before_C']=list(self.env['%10'])
            require(branch is not None,'block without terminator')
            previous,current=current,branch

    def result(self):
        base=15392*(self.y+1)+16*(self.x+1)
        outputs=[]
        for bank in (0,1):
            words=[self.outputs[base+bank*8342464+4*i] for i in range(4)]
            outputs+=list(struct.unpack('<16b',struct.pack('<4I',*words)))
        return {'pixel':[self.x,self.y],'output':outputs,'trace':self.traces,'steps':self.steps,'operations':dict(self.counts)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',type=Path,default=Path('build/fsr412_upstream'));parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--x',type=int,default=0);parser.add_argument('--y',type=int,default=0)
    args=parser.parse_args();probe=Probe((args.assets/(ACTIVE_2160['pass2']+'.ll')).read_text(),(args.assets/'initializers-normal.bin').read_bytes(),args.capture,args.x,args.y);print(json.dumps(probe.run(),indent=2))
