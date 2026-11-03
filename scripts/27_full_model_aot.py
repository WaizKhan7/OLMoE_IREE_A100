#!/usr/bin/env python3
"""Full OLMoE AoT prefill experiment. No earlier experiment files or notebook state.

RUN (fresh A100 high-RAM Colab, Drive already mounted):
  !python /content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot.py

The launcher creates an isolated venv, installs the pinned stack, downloads the
pinned checkpoint if missing, then runs isolated subprocesses. Existing notebook
packages are not changed. Requires internet for first setup, ~100 GB free disk,
and >=48 GiB available host RAM. Compilation can take hours. NOT GPU-validated
by the author: probe and correctness gates must succeed on the target machine.
Local checks include the actual E=64 routing shapes and a tiny two-layer model: sm_80 compilation and CPU runtime
execution, shared VMFB entry points, capacity fallback, diagnostic and next-token
outputs. These checks do not establish full-checkpoint GPU correctness or speed.

Scope: batch=1, FP32, causal unpadded prefill, next-token logits by default (--logits all for all positions), no KV cache.
Not decode/continuous batching/quantization. Static lengths, dynamic token IDs,
live router/top-k in every layer. Two capacities per length; a rejected smaller-capacity attempt is rerun through an exact C=S entry point.
Both entry points share one VMFB and parameter bank; rejected work is timed.
No mid-model Python/Torch calls in an IREE inference. All compiler defaults remain
on; O3 is requested. This is not a claim of globally optimal compiler schedules.

Recovery: --resume /content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/OLD_RUN
Reuses completed references and hash-verified weights; writes a new run directory.

CLI: --root PATH --lengths 64,128,256 --repeats 20 --warmups 5
     --no-install (use already provisioned stack; exact versions checked)
     --extra-iree-flag=FLAG (repeatable, all recorded)
     --phase-timeout 14400 --model /path/to/pinned/snapshot

Artifacts: full_model.vmfb with all sequence/capacity entry points + weights.irpa, imported MLIR bytecode,
source/config/checkpoint hashes, pip freeze, compiler flags/logs, raw samples,
full-position diagnostic and pre/post serving-output correctness statistics and logits references, capacity/fallback
counts, completed-host-logit latency/throughput, initial/final baseline drift, and a small review ZIP (large weights,
VMFBs and .npy references retained separately under output).
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.metadata as md
import json, math, os, random, shutil, statistics, subprocess, sys, time, traceback, zipfile

MODEL='allenai/OLMoE-1B-7B-0125'
REV='9b0c1aa87e34a20052389dce1f0cf01da783f654'
PINS=['torch==2.11.0','transformers==4.51.3','iree-turbine==3.9.0',
      'iree-base-compiler==3.12.0','iree-base-runtime==3.12.0',
      'accelerate==1.10.1','safetensors==0.6.2','numpy==2.2.6',
      'huggingface-hub==0.34.4']
TEXTS={
 'compiler':'A sparse model routes tokens to experts. A compiler preserves numerical behavior while controlling memory movement and execution costs. ',
 'narrative':'The village library opened early. Mina sorted books while a train arrived and the market filled with people. ',
 'code':'def route(tokens, experts):\n    counts = [0] * experts\n    for token in tokens:\n        counts[token % experts] += 1\n    return counts\n',
 'repeated':'The same word repeats. '}


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def capacities(s,cfg):
    return cfg['capacity_buckets'][str(s)]


def run_logged(cmd,log,timeout):
    print('Running:', ' '.join(map(str,cmd)),flush=True)
    started=time.monotonic();rc=0
    with open(log,'w') as f:
        process=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0:
                import signal
                
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait();f.write('\nTIMEOUT\n');rc=124;break
            try:rc=process.wait(timeout=min(30,remaining));break
            except subprocess.TimeoutExpired:
                lines=Path(log).read_text(errors='replace').splitlines()
                latest=next((x.strip() for x in reversed(lines) if x.strip()),'waiting for output')
                print(f'{Path(log).name}: running {int(time.monotonic()-started)}s | {latest[-240:]}',flush=True)
    if rc:
        lines=Path(log).read_text(errors='replace').splitlines()
        errors=[line for line in lines if 'error:' in line or 'Error:' in line or 'Error code:' in line]
        print(f'FAILED ({rc}): {log}',flush=True)
        print('\n'.join(errors[:12] if errors else lines[-12:]),flush=True)
    return rc


def environment_python(env,out,timeout):
    """Keep dependency setup out of the notebook's Python environment."""
    py=env/'bin/python'
    ready=py.exists() and run_logged([py,'-m','pip','--version'],out/'venv_check.log',60)==0
    if not ready:
        rc=run_logged([sys.executable,'-m','venv',env],out/'venv_create.log',timeout)
        ready=rc==0 and run_logged([py,'-m','pip','--version'],out/'venv_check.log',60)==0
    if not ready:
        # Some Colab images omit ensurepip. Install the creator into a separate
        # directory, then use it to seed pip; notebook packages stay untouched.
        bootstrap=env.parent/'27_virtualenv_bootstrap'
        if run_logged([sys.executable,'-m','pip','install','--target',bootstrap,'virtualenv>=20.26,<21'],out/'venv_bootstrap.log',timeout):
            raise RuntimeError('Cannot bootstrap isolated environment; inspect venv logs')
        code='import sys;sys.path.insert(0,sys.argv.pop(1));import virtualenv;virtualenv.cli_run(sys.argv[1:])'
        if run_logged([sys.executable,'-c',code,bootstrap,env],out/'venv_seed.log',timeout):
            raise RuntimeError('Cannot create isolated environment; inspect venv_seed.log')
    return str(py)


def stack():
    for spec in PINS:
        name,want=spec.split('==');got=md.version(name).split('+')[0]
        if got!=want:raise RuntimeError(f'{name}: expected {want}, found {got}')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    if torch.cuda.get_device_capability(0)!=(8,0):raise RuntimeError('This experiment targets A100 sm_80')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20261028);torch.set_num_threads(8)
    return torch


def compare(a,b,rtol=3e-4,atol=3e-4):
    import numpy as np
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Shape mismatch or nonfinite result')
    d=np.abs(a.astype('float64')-b.astype('float64'))
    r={'max_abs':float(d.max()),'mean_abs':float(d.mean()),
       'allclose':bool(np.allclose(a,b,rtol=rtol,atol=atol)),
       'argmax_agreement':float((a.argmax(-1)==b.argmax(-1)).mean())}
    if not r['allclose']:raise AssertionError(json.dumps(r))
    return r


def packing_layout(member,capacity):
    """Stable, destination-owned packing without sorting or scatter writes.
    All reductions/scans stay i32 (S<=256). Only gather indices widen to i64.
    """
    import torch
    experts,sequence=member.shape
    membership=member.to(torch.int32)
    counts=membership.sum(-1,dtype=torch.int32)
    ranks=membership.cumsum(-1,dtype=torch.int32)-1
    token=torch.arange(sequence,dtype=torch.int32,device=member.device)
    slot=torch.arange(capacity,dtype=torch.int32,device=member.device)
    equal=ranks[:,None,:]==slot[None,:,None]
    owns=torch.where(member[:,None,:],equal,torch.zeros_like(equal))
    # Each valid expert/slot has exactly one token owner. Empty slots use row 0,
    # then are masked before expert math; no floating-point reduction occurs.
    rows=torch.where(owns,token[None,None,:],torch.zeros((),dtype=torch.int32,device=member.device)).amax(-1)
    valid=slot[None,:]<counts[:,None]
    return counts,ranks,rows,valid


def classes(torch):
    nn=torch.nn;F=torch.nn.functional
    class Grouped(nn.Module):
        def __init__(self,moe,capacity):
            super().__init__();self.gate=moe.gate
            self.E=moe.num_experts;self.K=moe.top_k;self.C=capacity
            self.norm=moe.norm_topk_prob
            self.gu=nn.Parameter(torch.stack([torch.cat([e.gate_proj.weight,e.up_proj.weight],0).T.contiguous() for e in moe.experts]),requires_grad=False)
            self.down=nn.Parameter(torch.stack([e.down_proj.weight.T.contiguous() for e in moe.experts]),requires_grad=False)
        def forward(self,x):
            shape=x.shape;x=x.reshape(-1,shape[-1]);s=x.shape[0]
            probs=F.softmax(self.gate(x),dim=-1,dtype=torch.float32)
            scores,ids=torch.topk(probs,self.K,dim=-1)
            if self.norm:scores=scores/scores.sum(-1,keepdim=True)
            # Unique top-k IDs imply at most S rows per expert. Static capacity
            # makes contractions AoT-friendly; assignment values remain dynamic.
            expert=torch.arange(self.E,device=x.device)
            member=(ids[:,:,None]==expert[None,None,:]).any(dim=1).T
            counts=member.to(torch.int32).sum(-1,dtype=torch.int32)
            return self.assemble(x,member,ids,scores,s),counts

        def assemble(self,x,member,ids,scores,s):
            # Candidate graphs return counts. Only the model wrapper may accept
            # their output: any overflow requires a complete exact-capacity rerun.
            return self.compute(x,member,ids,scores,self.C)

        def compute(self,x,member,ids,scores,capacity):
            s=x.shape[0]
            counts,ranks,rows,valid=packing_layout(member,capacity)
            packed=x[rows.reshape(-1).long()].reshape(self.E,capacity,-1)
            packed=torch.where(valid[:,:,None],packed,torch.zeros_like(packed))
            z=torch.bmm(packed,self.gu);g,u=z.chunk(2,-1)
            out=torch.bmm(F.silu(g)*u,self.down)
            out=torch.where(valid[:,:,None],out,torch.zeros_like(out))
            # Rank is integer-only. Each token reads its own expert-owned rows.
            rank=torch.gather(ranks.T,1,ids).clamp(min=0,max=capacity-1)
            flat=(ids*capacity+rank).reshape(-1).long()
            selected=out.reshape(self.E*capacity,-1)[flat].reshape(s,self.K,-1)
            y=(selected*scores[:,:,None]).sum(1).reshape(1,s,-1)
            return y
    class Full(nn.Module):
        def __init__(self,hf,logits_mode='all'):
            super().__init__();self.body=hf.model;self.head=hf.lm_head;self.logits_mode=logits_mode
        def forward(self,ids,diagnostic=False):
            x=self.body.embed_tokens(ids)
            pos=torch.arange(ids.shape[1],device=ids.device)[None,:]
            rope=self.body.rotary_emb(x,pos)
            counts=[];first=x;middle=x
            for index,layer in enumerate(self.body.layers):
                a=layer.self_attn(hidden_states=layer.input_layernorm(x),
                    attention_mask=None,position_ids=pos,past_key_value=None,
                    output_attentions=False,use_cache=False,
                    cache_position=pos[0],position_embeddings=rope)[0]
                x=x+a
                y,c=layer.mlp(layer.post_attention_layernorm(x))
                x=x+y;counts.append(c)
                if index==0:first=x
                if index==min(7,len(self.body.layers)-1):middle=x
            final=self.body.norm(x)
            if diagnostic:return self.head(final),torch.stack(counts),first,middle,final
            if self.logits_mode=='last':final=final[:,-1:,:]
            return self.head(final),torch.stack(counts)
    return Grouped,Full


def load_hf(cfg):
    from transformers import AutoModelForCausalLM
    import torch
    m=AutoModelForCausalLM.from_pretrained(cfg['snapshot'],local_files_only=True,
        torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='sdpa',trust_remote_code=False).eval()
    c=m.config
    actual=(c.hidden_size,c.intermediate_size,c.num_experts,c.num_experts_per_tok,c.num_hidden_layers)
    if actual!=(2048,1024,64,8,16) or c.hidden_act!='silu' or c.norm_topk_prob:
        raise RuntimeError('Unsupported checkpoint contract '+str(actual))
    if c.rope_scaling is not None:raise RuntimeError('Non-default RoPE needs separate validation')
    m.requires_grad_(False)
    return m


def grouped_model(cfg,capacity):
    import torch,gc
    Grouped,Full=classes(torch);hf=load_hf(cfg)
    for layer in hf.model.layers:
        layer.mlp=Grouped(layer.mlp,capacity);gc.collect()
    return Full(hf,cfg['logits']).eval()


def set_capacity(m,c):
    for layer in m.body.layers:layer.mlp.C=c


def make_refs(cfg,out):
    import numpy as np,torch
    m=load_hf(cfg).cuda();out.mkdir(exist_ok=True)
    checks={}
    for case in cfg['cases']:
        ids=torch.from_numpy(np.load(Path(cfg['out'])/case['input'])).cuda()
        with torch.inference_mode():
            r=m(ids,use_cache=False,output_hidden_states=True,output_router_logits=True,return_dict=True)
        logits=r.logits.float().cpu().numpy()
        np.save(out/(case['name']+'_all_logits.npy'),logits)
        np.save(out/(case['name']+'_logits.npy'),logits[:,-1:,:] if cfg['logits']=='last' else logits)
        # Layer outputs remain reviewable independently of final logits.
        for i in [1,8,16]:np.save(out/(case['name']+f'_hidden{i}.npy'),r.hidden_states[i].cpu().numpy())
        counts=[]
        for router in r.router_logits:
            chosen=torch.topk(torch.softmax(router.float(),-1),8,-1).indices
            counts.append(torch.bincount(chosen.flatten(),minlength=64).cpu().numpy().tolist())
        checks[case['name']]={'counts':counts,'logits_sha256':sha(out/(case['name']+'_logits.npy'))}
        del r
    save(out/'manifest.json',checks)


def routing_probes(cfg,out):
    """Test the actual E=64/S=64,128,256 routing shapes before model export."""
    import numpy as np,torch,iree.turbine.aot as aot,iree.runtime as rt
    class Routing(torch.nn.Module):
        def __init__(self,capacity):super().__init__();self.capacity=capacity
        def forward(self,logits):
            scores,ids=torch.topk(torch.softmax(logits,-1),8,-1)
            expert=torch.arange(64,device=logits.device)
            member=(ids[:,:,None]==expert[None,None,:]).any(1).T
            return (scores,ids,*packing_layout(member,self.capacity))
    rng=np.random.default_rng(20261028);results={}
    for sequence in cfg['lengths']:
        caps=cfg.get('capacity_buckets',{}).get(str(sequence),[sequence])
        caps=sorted(set([*caps,sequence,max(1,sequence-16)]))
        for capacity in caps:
            name=f'routing_s{sequence}_c{capacity}'
            model=Routing(capacity).eval()
            logits=np.stack([rng.permutation(64) for _ in range(sequence)]).astype(np.float32)/16
            e=aot.export(model,torch.from_numpy(logits),module_name='routing',function_name='forward')
            source=out/(name+'.mlirbc');e.save_mlir(source);del e
            compile_ir(cfg,source,out/(name+'.vmfb'),out/(name+'_compile.log'))
            # No parameter module is required for this standalone graph.
            config=rt.Config('cuda');context=rt.SystemContext(config=config)
            vm=rt.VmModule.mmap(config.vm_instance,str(out/(name+'.vmfb')));context.add_vm_module(vm)
            checks=[]
            for kind in ['random','concentrated']:
                data=logits if kind=='random' else np.repeat(logits[0:1],sequence,axis=0)
                ids=np.argsort(-data,axis=-1)[:,:8]
                member=(ids[:,:,None]==np.arange(64)[None,None,:]).any(1).T
                counts=member.sum(-1,dtype=np.int32)
                ranks=np.cumsum(member.astype(np.int32),axis=-1,dtype=np.int32)-1
                rows=np.zeros((64,capacity),np.int32)
                valid=np.arange(capacity)[None,:]<counts[:,None]
                for expert in range(64):
                    tokens=np.flatnonzero(member[expert])[:capacity]
                    rows[expert,:len(tokens)]=tokens
                probabilities=np.exp(data-data.max(-1,keepdims=True));probabilities/=probabilities.sum(-1,keepdims=True)
                scores=np.take_along_axis(probabilities,ids,-1)
                expected=(scores,ids,counts,ranks,rows,valid)
                actual=context.modules.routing.forward(rt.asdevicearray(config.device,data))
                for value,reference in zip(actual,expected):
                    actual_host=value.to_host()
                    equal=np.allclose(actual_host,reference,rtol=1e-5,atol=1e-7) if reference.dtype.kind=='f' else np.array_equal(actual_host,reference)
                    if not equal:raise AssertionError(f'{name}/{kind}: routing mismatch')
                checks.append({'fixture':kind,'status':'PASS','max_count':int(counts.max())})
            results[name]=checks;save(out/'routing_checks.json',results)
            print(name,'actual-shape CUDA routing PASS',flush=True)


def probe(cfg,out):
    """Fail early on real CUDA export/lowering/runtime support, with tiny tensors."""
    import torch,numpy as np,iree.turbine.aot as aot
    from types import SimpleNamespace
    Grouped,_=classes(torch)
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__();self.gate_proj=torch.nn.Linear(8,4,bias=False)
            self.up_proj=torch.nn.Linear(8,4,bias=False);self.down_proj=torch.nn.Linear(4,8,bias=False)
    moe=torch.nn.Module();moe.gate=torch.nn.Linear(8,4,bias=False)
    moe.experts=torch.nn.ModuleList([Expert() for _ in range(4)])
    moe.num_experts=4;moe.top_k=2;moe.norm_topk_prob=False
    g=Grouped(moe,8).eval();x=torch.randn(1,8,8)
    # Reference per-expert math with output ownership independent of grouped ranks.
    with torch.no_grad():
        p=torch.softmax(moe.gate(x.reshape(-1,8)),-1);w,ids=torch.topk(p,2,-1)
        dense=torch.stack([e.down_proj(torch.nn.functional.silu(e.gate_proj(x))*e.up_proj(x)) for e in moe.experts],dim=2)
        reference=(torch.gather(dense,2,ids.reshape(1,8,2,1).expand(1,8,2,8))*w.reshape(1,8,2,1)).sum(2)
        compare(g(x)[0].numpy(),reference.numpy())
        # Concentrated routing must report overflow, never silently pass capacity.
        g.C=1
        concentrated=g(x[:,0:1,:].expand(1,8,8).contiguous())
        if not (concentrated[1]>1).any():raise AssertionError('Overflow fixture did not overflow')
        g.C=8
    aot.save_module_parameters(out/'probe.irpa',g)
    aot.externalize_module_parameters(g,external_scope='model')
    e=aot.export(g,x,module_name='probe',function_name='forward')
    e.save_mlir(out/'probe.mlirbc')
    compile_ir(cfg,out/'probe.mlirbc',out/'probe.vmfb',out/'probe_compile.log')
    fn,keep=iree_fn(out/'probe.vmfb',out/'probe.irpa','probe')
    import iree.runtime as rt
    y,c=fn(rt.asdevicearray(keep['cfg'].device,x.numpy()))
    with torch.no_grad():expected=g(x)
    compare(y.to_host(),expected[0].numpy());assert np.array_equal(c.to_host(),expected[1].numpy())
    # Exercise embeddings, RoPE, Q/K normalization, causal SDPA, residuals,
    # grouped MoE, final normalization and the LM head before the large export.
    from transformers import OlmoeConfig,OlmoeForCausalLM
    config=OlmoeConfig(vocab_size=64,hidden_size=16,intermediate_size=8,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
        num_experts=4,num_experts_per_tok=2,norm_topk_prob=False)
    config._attn_implementation='sdpa'
    hf=OlmoeForCausalLM(config).eval();ids=torch.randint(0,64,(1,8))
    with torch.no_grad():reference=hf(ids,use_cache=False).logits
    _,Full=classes(torch)
    for layer in hf.model.layers:layer.mlp=Grouped(layer.mlp,8)
    full=Full(hf).eval()
    with torch.no_grad():
        compare(full(ids)[0].numpy(),reference.numpy())
        full.logits_mode='last';compare(full(ids)[0].numpy(),reference[:,-1:,:].numpy())
        full.logits_mode='all'
    aot.save_module_parameters(out/'probe_full.irpa',full)
    aot.externalize_module_parameters(full,external_scope='model')
    fxb=aot.FxProgramsBuilder(full)
    def entry(model,input_ids):return model(input_ids)
    fxb.export_program(entry,args=(ids,),name='forward')
    for layer in full.body.layers:layer.mlp.C=1
    fxb.export_program(entry,args=(ids,),name='candidate')
    for layer in full.body.layers:layer.mlp.C=8
    full.logits_mode='last'
    fxb.export_program(entry,args=(ids,),name='last')
    full.logits_mode='all'
    def diagnostic(model,input_ids):return model(input_ids,diagnostic=True)
    fxb.export_program(diagnostic,args=(ids,),name='diagnostic')
    e=aot.export(fxb,module_name='probe_full')
    e.save_mlir(out/'probe_full.mlirbc')
    compile_ir(cfg,out/'probe_full.mlirbc',out/'probe_full.vmfb',out/'probe_full_compile.log')
    full_fn,full_keep=iree_fn(out/'probe_full.vmfb',out/'probe_full.irpa','probe_full')
    y,counts=full_fn(rt.asdevicearray(full_keep['cfg'].device,ids.numpy()))
    compare(y.to_host(),reference.numpy())
    _,candidate_counts=full_keep['ctx'].modules.probe_full.candidate(rt.asdevicearray(full_keep['cfg'].device,ids.numpy()))
    if not (np.array(candidate_counts.to_host())>1).any():raise AssertionError('Candidate must reject concentrated capacity')
    debug=full_keep['ctx'].modules.probe_full.diagnostic(rt.asdevicearray(full_keep['cfg'].device,ids.numpy()))
    if len(debug)!=5:raise AssertionError('Diagnostic output contract')
    compare(debug[0].to_host(),reference.numpy())
    last=full_keep['ctx'].modules.probe_full.last(rt.asdevicearray(full_keep['cfg'].device,ids.numpy()))
    compare(last[0].to_host(),reference[:,-1:,:].numpy())
    # Exact entry point is still valid after candidate execution, sharing weights.
    y,_=full_fn(rt.asdevicearray(full_keep['cfg'].device,ids.numpy()))
    compare(y.to_host(),reference.numpy())
    model=IreePrefillModel(out/'probe_full.vmfb',out/'probe_full.irpa',8,1,
        module='probe_full',fast_entry='candidate',exact_entry='forward')
    final,_,fallback=model(model.upload(ids.numpy()))
    if not fallback:raise AssertionError('Expected complete-model capacity fallback')
    compare(final,reference.numpy())
    save(out/'probe.json',{'status':'PASS','scope':'tiny grouped routing, capacity rejection, external parameters, complete tiny-model CUDA lowering and execution'})


def compile_ir(cfg,source,target,log):
    import iree.compiler as ic
    t=time.perf_counter()
    dumps=Path(target).parent/'executables';dumps.mkdir(exist_ok=True)
    flags=[*cfg['iree_flags'],f'--iree-hal-dump-executable-configurations-to={dumps}',
           f'--iree-hal-dump-executable-binaries-to={dumps}']
    try:
        binary=ic.compile_file(str(source),target_backends=['cuda'],extra_args=flags)
        Path(target).write_bytes(binary)
        save(str(target)+'.json',{'seconds':time.perf_counter()-t,'sha256':sha(target),'flags':flags,'source_sha256':sha(source)})
    except Exception:
        Path(log).write_text(traceback.format_exc());raise


def iree_fn(vmfb,weights,name='olmoe',entry='forward'):
    import iree.runtime as rt
    cfg=rt.Config('cuda');idx=rt.ParameterIndex();idx.load(str(weights))
    pm=rt.create_io_parameters_module(cfg.vm_instance,idx.create_provider(scope='model'))
    vm=rt.VmModule.mmap(cfg.vm_instance,str(vmfb))
    ctx=rt.SystemContext(config=cfg);ctx.add_vm_module(pm);ctx.add_vm_module(vm)
    return getattr(getattr(ctx.modules,name),entry),{'cfg':cfg,'idx':idx,'ctx':ctx,'vm':vm,'pm':pm}


class IreePrefillModel:
    """Runtime-only model object. Loads the whole VMFB and its parameter bank.
    Inputs are uploaded once; __call__ returns completed host logits, routing
    counts and a fallback flag. No PyTorch is used by this object.
    """
    def __init__(self,vmfb,weights,sequence,capacity,module='olmoe',fast_entry=None,exact_entry=None):
        started=time.perf_counter();self.sequence=sequence;self.capacity=capacity
        fast_entry=fast_entry or f'prefill_s{sequence}_c{capacity}'
        exact_entry=exact_entry or f'prefill_s{sequence}_c{sequence}'
        self.fast,self.keep=iree_fn(vmfb,weights,module,entry=fast_entry)
        self.exact=getattr(getattr(self.keep['ctx'].modules,module),exact_entry)
        self.load_seconds=time.perf_counter()-started
    def upload(self,ids):
        import iree.runtime as rt
        if tuple(ids.shape)!=(1,self.sequence):raise ValueError('Wrong sequence bucket')
        return rt.asdevicearray(self.keep['cfg'].device,ids)
    def __call__(self,ids):
        import numpy as np
        y,counts=self.fast(ids)
        y=np.asarray(y.to_host());h=np.asarray(counts.to_host())
        fallback=bool((h>self.capacity).any())
        if fallback:
            y,counts=self.exact(ids)
            y=np.asarray(y.to_host());h=np.asarray(counts.to_host())
            if (h>self.sequence).any():raise AssertionError('Exact capacity overflow')
        return y,h,fallback


def export_models(cfg,out):
    import torch,iree.turbine.aot as aot
    m=grouped_model(cfg,max(cfg['lengths']))
    if cfg.get('weights_archive'):
        import iree.runtime as rt
        archive=Path(cfg['weights_archive'])
        # Parameter names and byte lengths must match the current model. The
        # archive checksum was verified by resume preparation, before any build.
        index=rt.ParameterIndex();index.load(str(archive))
        lengths={key:entry.length for key,entry in index.items()}
        for name,tensor in m.state_dict().items():
            if name not in lengths or lengths[name]!=tensor.numel()*tensor.element_size():
                raise RuntimeError('Reused parameter schema mismatch: '+name)
        save(out/'weights.json',{'reused_from':str(archive),'sha256':cfg['weights_sha256'],
            'bytes':archive.stat().st_size,'schema_check':'PASS'})
        print('Reusing verified parameter archive:',archive,flush=True)
    else:
        t=time.perf_counter();aot.save_module_parameters(out/'weights.irpa',m)
        save(out/'weights.json',{'sha256':sha(out/'weights.irpa'),'bytes':(out/'weights.irpa').stat().st_size,'seconds':time.perf_counter()-t})
    aot.externalize_module_parameters(m,external_scope='model')
    # FxProgramsBuilder preserves shared parameter identity across entry points.
    fxb=aot.FxProgramsBuilder(m)
    for sequence in cfg['lengths']:
        for capacity in capacities(sequence,cfg):
            name=f'prefill_s{sequence}_c{capacity}';set_capacity(m,capacity)
            print('Exporting complete model entry point:',name,flush=True)
            def entry(model,ids):return model(ids)
            fxb.export_program(entry,args=(torch.zeros((1,sequence),dtype=torch.int64),),name=name)
        set_capacity(m,sequence)
        def diagnostic(model,ids):return model(ids,diagnostic=True)
        fxb.export_program(diagnostic,args=(torch.zeros((1,sequence),dtype=torch.int64),),name=f'diagnostic_s{sequence}')
    exported=aot.export(fxb,module_name='olmoe')
    source=out/'full_model.mlirbc';exported.save_mlir(source);del exported
    compile_ir(cfg,source,out/'full_model.vmfb',out/'full_model_compile.log')
    print('Compiled complete multi-entry-point model.',flush=True)


def measure(fn,finish,warmups,repeats):
    # fn enqueues; finish establishes completion. Output validation happens AFTER
    # the timed interval; the caller must keep the returned output alive.
    for _ in range(warmups):v=fn();finish(v);del v
    for _ in range(repeats):
        t=time.perf_counter_ns();v=fn();finish(v)
        # Caller consumes one output at a time; avoid retaining repeated logits.
        yield (time.perf_counter_ns()-t)/1e6,v


def torch_phase(cfg,out,arm):
    import torch,numpy as np,gc
    if arm.startswith('hf'):
        m=load_hf(cfg).cuda()
        def call(ids):return m(ids,use_cache=False,return_dict=True,logits_to_keep=1 if cfg['logits']=='last' else 0).logits
    else:
        m=grouped_model(cfg,max(cfg['lengths'])).cuda()
        def call(ids):return m(ids)
    compiled='compile' in arm
    if compiled:
        mode='reduce-overhead' if arm.endswith('graphs') else 'max-autotune-no-cudagraphs'
        save(out/'compile_settings.json',{'backend':'inductor','mode':mode,
            'fullgraph':not arm.startswith('hf'),'dynamic':False,
            'cuda_graph_capture':'requested; actual capture must be reviewed in logs/counters' if arm.endswith('graphs') else 'disabled'})
        # Native HF dynamic expert extraction can graph-break; recorded as such.
        call=torch.compile(call,backend='inductor',fullgraph=not arm.startswith('hf'),dynamic=False,mode=mode)
    rows=[];checks={};refdir=Path(cfg.get('reference_root',str(Path(cfg['out'])/'references')))
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for case in cfg['cases']:
            ids=torch.from_numpy(np.load(Path(cfg['out'])/case['input'])).cuda()
            ref=np.load(refdir/(case['name']+'_logits.npy'))
            # Exact-capacity grouped baseline and runtime are both reported.
            if not arm.startswith('hf'):
                set_capacity(m,min(capacities(case['length'],cfg)) if 'buckets' in arm else case['length'])
            def policy_call():
                v=call(ids);fallback=False
                if not arm.startswith('hf') and bool((v[1].cpu().numpy()>m.body.layers[0].mlp.C).any()):
                    fallback=True;candidate=m.body.layers[0].mlp.C
                    # Drain rejected logits too, matching the IREE host endpoint.
                    v[0].cpu().numpy();set_capacity(m,case['length']);v=call(ids)
                    set_capacity(m,candidate)
                return v,fallback
            v,fallback=policy_call();torch.cuda.synchronize()
            logits=v if arm.startswith('hf') else v[0]
            checks[case['name']]=compare(logits.cpu().numpy(),ref)
            if not arm.startswith('hf'):
                if int(v[1].max())>case['length']:raise AssertionError('Exact capacity overflow')
                # Layerwise checks against original HF references before timing.
                candidate=m.body.layers[0].mlp.C;set_capacity(m,case['length'])
                x=m.body.embed_tokens(ids);pos=torch.arange(ids.shape[1],device='cuda')[None,:];rope=m.body.rotary_emb(x,pos)
                for i,layer in enumerate(m.body.layers):
                    x=x+layer.self_attn(hidden_states=layer.input_layernorm(x),attention_mask=None,position_ids=pos,past_key_value=None,output_attentions=False,use_cache=False,cache_position=pos[0],position_embeddings=rope)[0]
                    x=x+layer.mlp(layer.post_attention_layernorm(x))[0]
                    if i+1 in [1,8]:compare(x.cpu().numpy(),np.load(refdir/(case['name']+f'_hidden{i+1}.npy')))
                compare(m.body.norm(x).cpu().numpy(),np.load(refdir/(case['name']+'_hidden16.npy')))
                del x;set_capacity(m,candidate)
            del v,logits
            def completed_call():
                v,fallback=policy_call()
                if arm.startswith('hf'):return v.cpu().numpy(),fallback
                logits=v[0].cpu().numpy();v[1].cpu().numpy()
                return logits,fallback
            for i,(ms,v) in enumerate(measure(completed_call,lambda _:None,cfg['warmups'],cfg['repeats'])):
                rows.append({'arm':arm,'case':case['name'],'sample':i,'wall_ms':ms,'fallback':v[1]})
                del v
            post,_=policy_call()
            post_logits=post if arm.startswith('hf') else post[0]
            checks[case['name']]['postflight']=compare(post_logits.cpu().numpy(),ref)
            del post,post_logits
            save(out/'progress.json',rows)
    save(out/'checks.json',checks)
    save(out/'memory.json',{'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved(),'includes_iree_allocations':False})
    if compiled:
        try:
            from torch._dynamo.utils import counters
            save(out/'dynamo_counters.json',{k:dict(v) for k,v in counters.items()})
        except Exception:save(out/'dynamo_counters.json',{'error':traceback.format_exc()})
    save(out/'samples.json',rows)


def iree_phase(cfg,out,s,c):
    import numpy as np,iree.runtime as rt
    artifacts=Path(cfg['out'])/'build'
    model=IreePrefillModel(artifacts/'full_model.vmfb',Path(cfg.get('weights_archive',str(artifacts/'weights.irpa'))),s,c)
    invoke=model
    save(out/'load.json',{'seconds':model.load_seconds,'excludes_first_invocation':True})
    memory_samples=[]
    rows=[];checks={}
    for case in [x for x in cfg['cases'] if x['length']==s]:
        ids=model.upload(np.load(Path(cfg['out'])/case['input']))
        ref=np.load(Path(cfg.get('reference_root',str(Path(cfg['out'])/'references')))/(case['name']+'_logits.npy'))
        debug=getattr(model.keep['ctx'].modules.olmoe,f"diagnostic_s{s}")(ids)
        debug_checks={}
        for index,suffix in [(0,'_all_logits'),(2,'_hidden1'),(3,'_hidden8'),(4,'_hidden16')]:
            debug_checks[suffix]=compare(debug[index].to_host(),np.load(Path(cfg.get('reference_root',str(Path(cfg['out'])/'references')))/(case['name']+suffix+'.npy')))
        del debug
        v,h,fallback=invoke(ids)
        checks[case['name']]={'diagnostic':debug_checks,'status':'PASS','logits':compare(v,ref),'max_count':int(h.max()),
            'counts':h.tolist(),'fallback':fallback,'overflowing_layers_in_exact_route':int((h.max(axis=1)>c).sum())}
        for _ in range(cfg['warmups']):v,h,fallback=invoke(ids)
        for i in range(cfg['repeats']):
            t=time.perf_counter_ns();v,h,fallback=invoke(ids);ms=(time.perf_counter_ns()-t)/1e6
            rows.append({'arm':f'iree_c{c}','case':case['name'],'sample':i,'wall_ms':ms,'fallback':fallback})
        v,h,fallback=invoke(ids)
        checks[case['name']]['postflight']=compare(v,ref)
        import torch
        free,total=torch.cuda.mem_get_info()
        memory_samples.append({'case':case['name'],'global_device_used_bytes':total-free})
        save(out/'progress.json',rows)
    save(out/'memory.json',{'samples':memory_samples,'kind':'global device usage after each case; not peak allocation'})
    save(out/'checks.json',checks);save(out/'samples.json',rows)


def worker(a):
    cfg=json.loads(Path(a.config).read_text());out=Path(a.worker_out);out.mkdir(parents=True,exist_ok=True)
    report={'phase':a.phase,'status':'RUNNING'};save(out/'status.json',report)
    try:
        torch=stack()
        save(out/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
             'packages':{x.split('==')[0]:md.version(x.split('==')[0]) for x in PINS}})
        with torch.inference_mode():
            if a.phase=='routing_probe':routing_probes(cfg,out)
            elif a.phase=='probe':probe(cfg,out)
            elif a.phase=='references':make_refs(cfg,out)
            elif a.phase=='export':export_models(cfg,out)
            elif a.phase=='iree':iree_phase(cfg,out,a.sequence,a.capacity)
            else:torch_phase(cfg,out,a.phase)
        report['status']='PASS'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());traceback.print_exc();raise
    finally:save(out/'status.json',report)


def summary(out,cfg,phases):
    allrows=[]
    for p in phases:
        f=out/p/'samples.json'
        if f.exists():allrows.extend(json.loads(f.read_text()))
    with open(out/'samples.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','case','sample','wall_ms','fallback']);w.writeheader();w.writerows(allrows)
    result={}
    for case in cfg['cases']:
        groups={}
        for r in allrows:
            if r['case']==case['name']:groups.setdefault(r['arm'],[]).append(r['wall_ms'])
        med={k:statistics.median(v) for k,v in groups.items()}
        initial=med.get('hf_eager_initial');final=med.get('hf_eager_final')
        # Best PyTorch arm is descriptive; publish every arm to avoid concealing
        # selection. Only successfully completed, validated arms enter comparison.
        pt={k:v for k,v in med.items() if not k.startswith('iree')}
        best=min(pt.values()) if pt else None
        ratios={k:v/best for k,v in med.items() if k.startswith('iree')} if best else {}
        distributions={k:{'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v),
             'p95_ms':sorted(v)[min(len(v)-1,math.ceil(.95*len(v))-1)],
             'tokens_per_second':1000*case['length']/statistics.median(v),'n':len(v)} for k,v in groups.items()}
        result[case['name']]={'evaluation_role':'calibration' if case['name'].startswith('compiler_') else 'held_out',
          'median_ms':med,'distributions':distributions,'iree_over_best_validated_pytorch':ratios,
          'baseline_drift_percent':100*(final/initial-1) if initial and final else None,
          'baseline_stable_10pct':abs(final/initial-1)<=.1 if initial and final else False}
    save(out/'summary.json',result);return result


def prepare_resume(previous,out):
    previous=Path(previous).resolve()
    cfg=json.loads((previous/'config.json').read_text())
    if sha(previous/'experiment.py')!=cfg['source_sha256']:
        raise RuntimeError('Previous run source hash mismatch')
    if sha(previous/'checkpoint.json')!=cfg['checkpoint_manifest_sha256']:
        raise RuntimeError('Previous checkpoint manifest changed')
    reference_root=Path(cfg.get('reference_root',str(previous/'references')))
    if json.loads((reference_root/'status.json').read_text())['status']!='PASS':
        raise RuntimeError('Reference phase was not complete')
    manifest=json.loads((reference_root/'manifest.json').read_text())
    for case in cfg['cases']:
        input_path=Path(case['input'])
        if not input_path.is_absolute():input_path=previous/input_path
        if sha(input_path)!=case['sha256']:raise RuntimeError('Input changed: '+case['name'])
        case['input']=str(input_path)
        if sha(reference_root/(case['name']+'_logits.npy'))!=manifest[case['name']]['logits_sha256']:
            raise RuntimeError('Reference logits changed: '+case['name'])
        for suffix in ['_all_logits','_hidden1','_hidden8','_hidden16']:
            if not (reference_root/(case['name']+suffix+'.npy')).is_file():
                raise RuntimeError('Missing reference artifact: '+case['name']+suffix)
    weights_meta=json.loads((previous/'build/weights.json').read_text())
    archive=Path(cfg.get('weights_archive',str(previous/'build/weights.irpa')))
    print('Verifying existing weight archive checksum; no archive rewrite:',archive,flush=True)
    if sha(archive)!=weights_meta['sha256']:raise RuntimeError('Weight archive checksum mismatch')
    cfg.update(out=str(out),reference_root=str(reference_root),weights_archive=str(archive),
        weights_sha256=weights_meta['sha256'],resume_from=str(previous),source_sha256=sha(out/'experiment.py'))
    shutil.copy2(previous/'checkpoint.json',out/'checkpoint.json')
    for name in ['pip_freeze.txt','nvidia_smi.txt','transformers_modeling_olmoe.py']:
        if (previous/name).exists():shutil.copy2(previous/name,out/('previous_'+name))
    save(out/'reuse.json',{'previous_run':str(previous),'archive':str(archive),
        'archive_sha256':weights_meta['sha256'],'reference_root':str(reference_root),
        'input_hashes':'verified','serving_logit_hashes':'verified'})
    save(out/'config.json',cfg)
    return cfg


def launch(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=root/'27_full_model_aot'/stamp;out.mkdir(parents=True)
    shutil.copy2(__file__,out/'experiment.py')
    report={'status':'SETUP','out':str(out),'phases':{},'implementation':'27.1_i32_sortfree_packing',
        'sampling_design':'Sequential isolated phases, descriptive ratios, initial/final PyTorch drift bracket; no paired confidence intervals',
        'endpoint':'Resident device token IDs to completed host logits (last/all selected in config); tokenization, compilation and initial weight loading excluded',
        'validation':'IREE diagnostic entry checks all-position logits and hidden layers 1/8/16; serving logits checked before/after timing'};save(out/'report.json',report)
    py=sys.executable
    try:
        if a.resume:
            env=Path(os.environ.get('MOE_AOT_VENV','/content/27_aot_venv'))
            py=str(env/'bin/python')
            if not Path(py).is_file():raise RuntimeError('Existing isolated environment missing: '+py)
            cfg=prepare_resume(a.resume,out)
            report['resume_from']=str(Path(a.resume).resolve())
        else:
            if not a.no_install:
                # Venvs contain symlinks and many small files; keep them off Drive.
                env=Path(os.environ.get('MOE_AOT_VENV',str(Path('/content/27_aot_venv') if Path('/content').exists() else Path.home()/'.cache/olmoe_aot_venv')))
                py=environment_python(env,out,a.phase_timeout)
                rc=run_logged([py,'-m','pip','install',*PINS],out/'install.log',a.phase_timeout)
                if rc:raise RuntimeError('Dependency resolution failed; inspect install.log')
            # Parent remains stdlib-only. Setup takes place in the isolated interpreter.
            request={'out':str(out),'model':a.model,'root':str(root),'lengths':[int(x) for x in a.lengths.split(',')],
                     'warmups':a.warmups,'repeats':a.repeats,'logits':a.logits,'iree_flags':['--iree-cuda-target=sm_80','--iree-opt-level=O3','--iree-opt-numeric-precision-reduction=false','--iree-codegen-enable-default-tuning-specs=true',*a.extra_iree_flag]}
            save(out/'setup_request.json',request)
            cmd=[py,str(out/'experiment.py'),'--setup',str(out/'setup_request.json')]
            if run_logged(cmd,out/'setup.log',a.phase_timeout):raise RuntimeError('Setup failed; inspect setup.log')
            cfg=json.loads((out/'config.json').read_text())
        phases=[]
        def phase(name,kind,required=True,s=0,c=0):
            phases.append(name);print('Phase',name,flush=True);started=time.perf_counter()
            rc=run_logged([py,str(out/'experiment.py'),'--phase',kind,'--config',str(out/'config.json'),
               '--worker-out',str(out/name),'--sequence',str(s),'--capacity',str(c)],out/(name+'.log'),a.phase_timeout)
            report['phases'][name]={'returncode':rc,'required':required,'phase_wall_seconds':time.perf_counter()-started};save(out/'report.json',report)
            if rc and required:raise RuntimeError('Required phase failed: '+name+'; preserve logs')
        phase('routing_probe','routing_probe');phase('probe','probe')
        if not a.resume:phase('references','references')
        manifest=json.loads((Path(cfg.get('reference_root',str(out/'references')))/'manifest.json').read_text())
        # Compiler prompt is calibration only. Held-out routing never selects
        # a bucket, and all runtime attempts are checked and fully timed.
        cfg['capacity_buckets']={}
        cfg['capacity_calibration']={}
        for sequence in cfg['lengths']:
            counts=manifest[f'compiler_s{sequence}']['counts']
            maximum=max(max(layer) for layer in counts)
            candidate=min(sequence,math.ceil(maximum/16)*16)
            cfg['capacity_buckets'][str(sequence)]=sorted(set([candidate,sequence]))
            cfg['capacity_calibration'][str(sequence)]={'case':f'compiler_s{sequence}',
                'max_count':maximum,'candidate':candidate,'alignment':16}
        save(out/'config.json',cfg)
        report['capacity_calibration']=cfg['capacity_calibration']
        phase('build','export')
        phase('hf_eager_initial','hf_eager_initial')
        phase('hf_compile','hf_compile',required=False)
        phase('grouped_eager','grouped_eager')
        phase('grouped_compile','grouped_compile',required=False)
        phase('grouped_compile_graphs','grouped_compile_graphs',required=False)
        phase('grouped_buckets_eager','grouped_buckets_eager')
        phase('grouped_buckets_compile','grouped_buckets_compile',required=False)
        phase('grouped_buckets_compile_graphs','grouped_buckets_compile_graphs',required=False)
        for s in cfg['lengths']:
            for c in capacities(s,cfg):phase(f'iree_s{s}_c{c}','iree',s=s,c=c)
        phase('hf_eager_final','hf_eager_final')
        report['summary']=summary(out,cfg,phases)
        stable=all(v['baseline_stable_10pct'] for v in report['summary'].values())
        optional_failed=any(v['returncode'] for v in report['phases'].values())
        report['status']='COMPLETE_REVIEW_REQUIRED' if stable and not optional_failed else 'PARTIAL_OR_DRIFT_REVIEW_REQUIRED'
    except BaseException:
        report.update(status='FAILED_OR_INCOMPLETE',error=traceback.format_exc());traceback.print_exc()
    finally:
        save(out/'report.json',report)
        with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.txt','.mlir']:z.write(p,p.relative_to(out))
        print('Status:',report['status']);print('Output:',out);print('Bundle:',out/'review_bundle.zip',flush=True)
    if report['status']=='FAILED_OR_INCOMPLETE':raise SystemExit(1)


def setup(path):
    cfg=json.loads(Path(path).read_text());out=Path(cfg['out'])
    torch=stack()
    if torch.cuda.get_device_capability(0)!=(8,0):raise RuntimeError('This profile targets A100 sm_80 only')
    if cfg['repeats']<5 or cfg['warmups']<2:raise ValueError('Need >=5 repeats and >=2 warmups')
    # Device state is checked, not inherited. Another notebook retaining the
    # checkpoint can otherwise invalidate both memory and latency measurements.
    if torch.cuda.mem_get_info()[1]<38*1024**3:raise RuntimeError('Need an A100 with >=40 GB device memory')
    used=torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0]
    if used>2*1024**3:raise RuntimeError('GPU already has >2 GiB allocated; start a fresh runtime')
    available=None
    if Path('/proc/meminfo').exists():
        lines=Path('/proc/meminfo').read_text().splitlines()
        available=int(next(l for l in lines if l.startswith('MemAvailable:')).split()[1])*1024
        limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
        if limit.exists() and current.exists() and limit.read_text().strip()!='max':
            available=min(available,int(limit.read_text())-int(current.read_text()))
        if available<48*1024**3:raise RuntimeError('Need >=48 GiB available host RAM for full FP32 export')
    if shutil.disk_usage(out).free<85*1024**3:raise RuntimeError('Need >=85 GiB free output disk')
    os.environ['HF_HUB_OFFLINE']='0';os.environ['TRANSFORMERS_OFFLINE']='0'
    from huggingface_hub import snapshot_download
    # Online lookup/download permitted. Pinned revision prevents accidental model
    # upgrades; existing cache is reused automatically. No offline-only assumption.
    snapshot=cfg['model'] or snapshot_download(MODEL,revision=REV,cache_dir=str(Path(cfg['root'])/'hf_cache'),
        local_files_only=False,allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
    cfg['snapshot']=str(Path(snapshot).resolve())
    import numpy as np
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    cases=[]
    for s in cfg['lengths']:
        if s<2 or s>256:raise ValueError('Validated experiment range is 2..256 tokens')
        for name,text in TEXTS.items():
            repeat=1
            while True:
                ids=tok(text*repeat,return_tensors='np',add_special_tokens=True)['input_ids']
                if ids.shape[1]>=s:break
                repeat*=2
            ids=ids[:,:s].astype(np.int64)
            if ids.shape!=(1,s):raise ValueError('Prompt too short')
            label=f'{name}_s{s}';np.save(out/(label+'_input.npy'),ids)
            cases.append({'name':label,'length':s,'input':label+'_input.npy','sha256':sha(out/(label+'_input.npy'))})
    cfg['cases']=cases
    checkpoint={}
    for p in sorted(Path(snapshot).glob('*')):
        if p.is_file() and p.suffix in ['.json','.safetensors','.model']:
            checkpoint[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    save(out/'checkpoint.json',{'repo':MODEL,'revision':REV,'override_path_supplied':bool(cfg['model']),
         'note':'Override hashes recorded; user must supply same checkpoint revision when overriding', 'files':checkpoint})
    cfg['checkpoint_manifest_sha256']=sha(out/'checkpoint.json')
    cfg['source_sha256']=sha(out/'experiment.py')
    save(out/'config.json',cfg)
    (out/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (out/'nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    # Save exact implementation files used in the run, avoiding implicit dependence
    # on whichever Transformers release happened to be in a previous notebook.
    import inspect,transformers.models.olmoe.modeling_olmoe as modeling
    shutil.copy2(inspect.getfile(modeling),out/'transformers_modeling_olmoe.py')
    print('Setup complete. Source/config/checkpoint hashes recorded.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',default='/content/drive/MyDrive/moe_iree_olmoe')
    p.add_argument('--lengths',default='64,128,256');p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--phase-timeout',type=int,default=14400)
    p.add_argument('--model');p.add_argument('--no-install',action='store_true')
    p.add_argument('--logits',choices=['last','all'],default='last')
    p.add_argument('--extra-iree-flag',action='append',default=[])
    p.add_argument('--phase');p.add_argument('--config');p.add_argument('--worker-out')
    p.add_argument('--sequence',type=int,default=0);p.add_argument('--capacity',type=int,default=0)
    p.add_argument('--setup')
    p.add_argument('--resume',help='Completed setup/reference run directory; reuse environment, checkpoint, reference tensors and verified IRPA')
    a=p.parse_args()
    if a.setup:setup(a.setup)
    elif a.phase:worker(a)
    else:launch(a)

if __name__=='__main__':main()
