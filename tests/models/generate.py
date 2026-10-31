#!/usr/bin/env python3
"""Generate deterministic model stages, dual-ELF sources and independent oracles.

INT8 matmul is NumPy, never QEMU output. FP reference uses mathematical NumPy
operations; its declared tolerance is ONLY QEMU calibration, RTL is byte exact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

F=np.float32
TYPES={'int8':'int8_t','int32':'int32_t','float32':'float','float16':'_Float16'}
THETA=np.array([1.,.66360,.44037,.29223,.19392,.12869,.085397,.056670,.037606,.024955,.016560,.010990,.0072927,.0048394,.0032114,.0016846,.00077941,.00028119,.000088651,.000052790,.000031436,.000018720,.000011147,.0000066380,.0000039528,.0000023539,.0000014017,.00000083469,.00000049704,.00000029598,.00000017625,.00000010496],dtype=F)
PROFILES={'quick':{'s':64,'h':128,'f':256,'heads':2,'kv':1},
          'bert-small':{'s':128,'h':512,'f':1024,'heads':4,'kv':0},
          'bert-base':{'s':128,'h':768,'f':3072,'heads':12,'kv':0},
          'llama-1b':{'s':128,'h':2048,'f':8192,'heads':32,'kv':8}}

def sat(x):return np.clip(x,-128,127).astype('int8')
def softmax(x):
    q=(x-x.max(-1,keepdims=True)).astype('int32')
    z=((-q*np.int32(65536)).astype('int32')>>16)
    qp=(q+z+2).astype('int32')
    ex=((qp*qp+3).astype('int32')>>(z&31)).astype('int32')
    sm=ex.sum(-1,dtype='int32',keepdims=True)
    y=sat(np.rint(ex.astype(F)*(F(127)/sm.astype(F))))
    return ex,y

def layernorm(x,res):
    mean=np.trunc(x.sum(-1,dtype='int32',keepdims=True).astype('float64')/x.shape[-1]).astype('int32')
    d=(x-mean).astype('int32')
    variance=np.trunc((d*d).sum(-1,dtype='int32').astype('float64')/x.shape[-1]).astype('int32')
    sd=np.array([max(1,math.isqrt(max(0,int(v)))) for v in variance],dtype=F)[:,None]
    return sat(sat(np.rint(d.astype(F)*(F(1)/sd))).astype('int16')+res.astype('int16'))

class Program:
    def __init__(self,root,name,profile):
        self.path=root/name;self.path.mkdir(parents=True,exist_ok=True)
        self.name=name;self.profile=profile;self.arr={};self.decl=[];self.code=[];self.tasks=[];self.expressions={};self.compositional=False
        self.rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little'))
    def array(self,name,value,output=False):
        v=np.ascontiguousarray(value);self.arr[name]=v
        assert v.dtype.name in TYPES and v.nbytes%64==0,(name,v.shape,v.dtype)
        ct=TYPES[v.dtype.name]
        if output:self.decl.append(f'{ct} {name}[{v.size}] __attribute__((aligned(64))) = {{1}};')
        else:
            file=self.path/(name+'.input.bin');v.tofile(file)
            self.decl.append(f'extern const {ct} {name}[{v.size}];')
            asm=f'.pushsection .rodata\n.balign 64\n.global {name}\n.type {name}, @object\n{name}:\n.incbin "{file}"\n.size {name}, .-{name}\n.popsection\n'
            self.decl.append('__asm__('+json.dumps(asm)+');')
        return name
    def inp(self,name,shape,kind='i8'):
        if kind=='i8':v=self.rng.integers(-3,4,shape,dtype='int8')
        elif kind=='bias':v=self.rng.integers(-16,17,shape,dtype='int32')
        elif kind=='f32':v=self.rng.integers(-32,33,shape).astype(F)/F(32)
        elif kind=='weight':v=self.rng.integers(8,17,shape).astype(F)/F(16)
        return self.array(name,v)
    def reference(self,op,*names,**options):
        sources=[]
        for name in names:
            a=self.arr[name];source={'dtype':a.dtype.str,'shape':list(a.shape)}
            producer=next((t for t in reversed(self.tasks) if any(o['symbol']==name for o in t['outputs'])),None)
            if producer:source.update(hart=producer['hart'],task_seq=producer['seq'],symbol=name)
            else:source['fixture']=name+'.input.bin'
            sources.append(source)
        return dict(op=op,inputs=sources,**options)
    def task(self,name,outputs,code):
        descriptors=[]
        for symbol in outputs:
            v=self.arr[symbol];file=symbol+'.oracle.bin';v.tofile(self.path/file)
            d={'symbol':symbol,'bytes':v.nbytes,'oracle':file}
            if v.dtype.kind=='f':d['oracle_compare']={'mode':'float_tolerance','dtype':v.dtype.str,'rtol':0.002 if v.dtype.itemsize==2 else 3e-5,'atol':0.0001 if v.dtype.itemsize==2 else 3e-6}
            if symbol in self.expressions:d['oracle_expression']=self.expressions[symbol]
            descriptors.append(d)
        seq=len(self.tasks)+1
        self.tasks.append({'hart':0,'seq':seq,'key':seq,'name':name,'outputs':descriptors})
        self.code += [code,f'if (!cute_checkpoint(&ctx,{seq})) return {seq};']
    def op(self,name,value,code,expression=None):
        if expression is not None:self.expressions[name]=expression
        self.array(name,value,True);self.task(name,[name],code);return name
    def gemm(self,name,a,b,bias=None):
        av,bv=self.arr[a],self.arr[b];m,k=av.shape;n=bv.shape[0]
        ref=(av.astype('int32')@bv.astype('int32').T).astype('int32')
        if bias:ref=(ref+self.arr[bias]).astype('int32')
        return self.op(name,ref,f'cute_model_gemm({a},{b},{bias or "NULL"},{name},{m},{n},{k},{k},{k},{n});',self.reference('matmul_i8',a,b,*([bias] if bias else [])) if self.compositional else None)
    def i8(self,name,x,gelu=False):
        v=self.arr[x]
        if gelu:
            sign=np.where(v<0,-1,1).astype('int32');p=np.minimum(v*sign,2)-2;v=(v*(sign*(p*p-6)-6)).astype('int32')
        return self.op(name,sat(v),f'cute_model_i8({x},{name},{v.size},{int(gelu)});')
    def norm(self,name,x,res):
        m,n=self.arr[x].shape
        return self.op(name,layernorm(self.arr[x],self.arr[res]),f'cute_model_layernorm({x},{res},{name},{m},{n});')
    def quant(self,name,x):
        v=self.arr[x];m,n=v.shape;s=np.abs(v).max(-1)/F(127);inv=np.divide(F(1),s,out=np.zeros_like(s),where=s!=0)
        self.array(name+'_scale',s,True);self.array(name,sat(np.rint(v*inv[:,None])),True)
        if self.compositional:
            self.expressions[name+'_scale']=self.reference('quant_scale',x)
            self.expressions[name]=self.reference('quant_i8',x)
        self.task(name,[name+'_scale',name],f'cute_model_quant({x},{name}_scale,{name},{m},{n});');return name,name+'_scale'
    def dequant(self,name,x,scale):
        v=self.arr[x];m,n=v.shape;v=v.astype(F)*(self.arr[scale][:,None]*F(1/128))
        return self.op(name,v,f'cute_model_dequant({x},{scale},0x1p-7f,{name},{m},{n});',self.reference('dequant',x,scale,weight_scale=1/128) if self.compositional else None)
    def resadd(self,name,x,res):
        m,n=self.arr[x].shape
        return self.op(name,(self.arr[x]+self.arr[res]).astype(F),f'cute_resadd_tile({x},{n*4},{res},{n*4},{name},{n*4},{m},{n});',self.reference('add',x,res) if self.compositional else None)
    def finish(self):
        text='#include "cute_checkpoint.h"\n#include "cute_model_stages.h"\n'+'\n'.join(self.decl)+'\nint main(void){\n cute_checkpoint_context ctx={0};\n __asm__ volatile("csrwi frm,0");\n'+'\n'.join(self.code)+'\n cute_checkpoint_finish(); return 0;\n}\n'
        (self.path/'model_program.h').write_text(text)
        (self.path/'case.json').write_text(json.dumps({'schema_version':1,'name':self.name,'scope':'single-hart DDR model stage; all outputs, raw accumulators and scales','checkpoint_scope':'quiescent hart; byte-exact RTL/QEMU','profile':self.profile,'tasks':self.tasks},indent=2)+'\n')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.path.iterdir()) if p.name != 'fixture.json'}
        (self.path/'fixture.json').write_text(json.dumps({'schema_version':1,'case':self.name,'profile':self.profile,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':hashes},indent=2)+'\n')

def bert(p,stage):
    s,h,f,heads=[p.profile[k] for k in ['s','h','f','heads']];d=h//heads
    if stage==1:
        a=p.inp('input',(s,h))
        for q in ['q','k','v']:
            b=p.inp(q+'_weight',(h,h));raw=p.gemm(q+'_acc',a,b);out=p.i8(q+'_out',raw)
            if q=='v':p.op('v_transpose',p.arr[out].T,f'cute_model_transpose_i8({out},v_transpose,{s},{h});')
    elif stage==2:
        q=p.inp('q',(s,h));k=p.inp('k',(s,h));ref=np.stack([p.arr[q][:,i*d:(i+1)*d].astype('int32')@p.arr[k][:,i*d:(i+1)*d].astype('int32').T for i in range(heads)])
        p.op('qk',ref,'\n'.join(f'cute_model_gemm(q+{i*d},k+{i*d},NULL,qk+{i*s*s},{s},{s},{d},{h},{h},{s});' for i in range(heads)))
        ex,y=softmax(ref);p.array('qexp',ex,True);p.array('probability',y,True)
        p.task('integer_softmax',['qexp','probability'],f'cute_model_softmax(qk,qexp,probability,{heads*s},{s});')
    elif stage==3:
        a=p.inp('probability',(heads,s,s));v=p.inp('v_transpose',(h,s))
        ref=np.concatenate([p.arr[a][i].astype('int32')@p.arr[v][i*d:(i+1)*d].astype('int32').T for i in range(heads)],axis=1)
        p.op('context_acc',ref,'\n'.join(f'cute_model_gemm(probability+{i*s*s},v_transpose+{i*d*s},NULL,context_acc+{i*d},{s},{d},{s},{s},{s},{h});' for i in range(heads)))
        p.i8('context','context_acc')
    else:
        n,k=(f,h) if stage==5 else (h,f if stage==6 else h)
        a=p.inp('input',(s,k));b=p.inp('weight',(n,k));bias=p.inp('bias',(n,),'bias') if stage!=4 else None
        raw=p.gemm('projection_acc',a,b,bias)
        if stage==5:p.i8('ff1',raw,True)
        else:p.norm('residual_out',raw,p.inp('residual',(s,h)))

def llama(p,stage):
    p.compositional=(stage==4)
    s,h,f,heads,kv=[p.profile[k] for k in ['s','h','f','heads','kv']];hd=h//heads;kvh=kv*hd
    if stage==1:
        for prefix in ['attn','ffn']:
            x=p.inp(prefix+'_input',(s,h),'f32');w=p.inp(prefix+'_weight',(h,),'weight')
            v=p.arr[x];partial=np.zeros((s,64),dtype=F)
            for c in range(0,h,64):partial=(partial+v[:,c:c+64]*v[:,c:c+64]).astype(F)
            sums=np.zeros(s,dtype=F)
            for c in range(64):sums=(sums+partial[:,c]).astype(F)
            norm=((v*(F(1)/np.sqrt(sums/F(h)+F(1e-5)))[:,None])*p.arr[w]).astype(F)
            y=p.op(prefix+'_norm',norm,f'cute_model_rms({x},{w},{prefix}_norm,{s},{h});');p.quant(prefix+'_q8',y)
    elif stage==2:
        a=p.inp('input_q8',(s,h));scale=p.array('input_scale',p.rng.integers(1,8,s).astype(F)/F(128));theta=p.array('theta',THETA)
        for q,n in [('q',h),('k',kvh),('v',kvh)]:
            b=p.inp(q+'_weight',(n,h));raw=p.gemm(q+'_acc',a,b);dq=p.dequant(q+'_float',raw,scale);v=p.arr[dq]
            if q!='v':
                vr=v.reshape(s,n//hd,hd);angle=np.arange(s,dtype=F)[:,None,None]*THETA[None,None,:];c=np.cos(angle);si=np.sin(angle)
                real=vr[:,:,0::2]*c-vr[:,:,1::2]*si;imag=vr[:,:,0::2]*si+vr[:,:,1::2]*c
                ref=np.concatenate([real,imag],axis=-1).reshape(s,n).astype('float16')
                p.op(q+'_fp16',ref,f'cute_model_rope({dq},{q}_fp16,theta,{s},{n},{hd});')
            else:p.op('v_fp16_transpose',v.T.astype('float16'),f'cute_model_transpose_f16({dq},v_fp16_transpose,{s},{n});')
    elif stage==3:
        x=p.inp('context',(s,h),'f32');res=p.inp('residual',(s,h),'f32');a,scale=p.quant('context_q8',x)
        b=p.inp('o_weight',(h,h));raw=p.gemm('o_acc',a,b);dq=p.dequant('o_float',raw,scale);p.resadd('o_residual',dq,res)
    else:
        a=p.inp('input_q8',(s,h));scale=p.array('input_scale',p.rng.integers(1,8,s).astype(F)/F(128));res=p.inp('residual',(s,h),'f32')
        b=p.inp('gate_weight',(f,h));raw=p.gemm('gate_acc',a,b);dq=p.dequant('gate_float',raw,scale);v=p.arr[dq]
        gate=p.op('gate_silu',(v/(F(1)+np.exp(-v))).astype(F),f'cute_model_silu({dq},gate_silu,{s*f});',p.reference('silu',dq))
        b=p.inp('up_weight',(f,h));raw=p.gemm('up_acc',a,b);up=p.dequant('up_float',raw,scale)
        code=f'for(int i=0;i<{s*f};){{size_t vl=__riscv_vsetvl_e32m4({s*f}-i);__riscv_vse32_v_f32m4(product+i,__riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4({gate}+i,vl),__riscv_vle32_v_f32m4({up}+i,vl),vl),vl);i+=vl;}}'
        prod=p.op('product',(p.arr[gate]*p.arr[up]).astype(F),code,p.reference('multiply',gate,up));qa,sc=p.quant('product_q8',prod)
        b=p.inp('down_weight',(h,f));raw=p.gemm('down_acc',qa,b);dq=p.dequant('down_float',raw,sc);p.resadd('down_residual',dq,res)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile',choices=PROFILES,default='quick');args=ap.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for family,stages,fn in [('bert',6,bert),('llama',4,llama)]:
        if args.profile.startswith('bert') and family=='llama':continue
        if args.profile=='llama-1b' and family=='bert':continue
        for stage in range(1,stages+1):
            name=f'{family}_{stage}';p=Program(args.out,name,dict(PROFILES[args.profile],name=args.profile));fn(p,stage);p.finish();cases.append(name);print(name,flush=True)
    (args.out/'suite.json').write_text(json.dumps({'profile':args.profile,'cases':cases},indent=2)+'\n')
if __name__=='__main__':main()
