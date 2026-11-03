#!/usr/bin/env python3
"""S256 full-model comparison: host-orchestrated exact-row IREE vs static IREE.
Requires experiment 42 companion and existing Drive artifacts; no notebook state.
New split-model shell, reused pinned dynamic expert; all compute stays in IREE.
This compares implementations, NOT the isolated cost of dynamic shape support.
"""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,shutil,statistics,subprocess,sys,time,traceback,zipfile
SCHEDULE=['original','dynamic','tuned','tuned','dynamic','original']
DYNAMIC_SHA='d75d4e0abcef639a0bb49b69652cdf2ba52e4d37297a882991a38880be710a77'

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m

def shell_class(torch,m):
 class Shell(torch.nn.Module):
  def __init__(self,model):
   super().__init__();self.body=model.body;self.head=model.head
   for layer in self.body.layers:
    del layer.mlp.gu;del layer.mlp.down
  def start(self,ids):return self.body.embed_tokens(ids)
  def step(self,x,index):
   layer=self.body.layers[index];moe=layer.mlp
   s=x.shape[1];pos=torch.arange(s,device=x.device)[None,:]
   rope=self.body.rotary_emb(x,pos)
   attn=layer.self_attn(hidden_states=layer.input_layernorm(x),attention_mask=None,
    position_ids=pos,past_key_value=None,output_attentions=False,use_cache=False,
    cache_position=pos[0],position_embeddings=rope)[0]
   residual=x+attn;z=layer.post_attention_layernorm(residual).reshape(s,-1)
   probs=torch.nn.functional.softmax(moe.gate(z),dim=-1,dtype=torch.float32)
   scores,ids=torch.topk(probs,moe.K,dim=-1)
   if moe.norm:scores=scores/scores.sum(-1,keepdim=True)
   member=(ids[:,:,None]==torch.arange(moe.E,device=x.device)[None,None,:]).any(1).T
   counts,ranks,rows,valid=m.packing_layout(member,s)
   packed=z[rows.reshape(-1).long()].reshape(moe.E,s,-1)
   packed=torch.where(valid[:,:,None],packed,torch.zeros_like(packed))
   return residual,packed,counts,ids,scores,ranks
  def combine(self,residual,assembled,ids,scores,ranks):
   s=residual.shape[1];k=ids.shape[1];h=residual.shape[2]
   rank=torch.gather(ranks.T,1,ids).clamp(min=0,max=s-1)
   flat=(ids*s+rank).reshape(-1).long()
   selected=assembled.reshape(-1,h)[flat].reshape(s,k,h)
   return residual+(selected*scores[:,:,None]).sum(1).reshape(1,s,h)
  def finish(self,x):return self.head(self.body.norm(x)[:,-1:,:])
  def diagnostic(self,x):
   z=self.body.norm(x);return self.head(z),z
 return Shell

def segment_source(e,s,h):
 bank=f'tensor<{e}x{s}x{h}xf32>';rows=f'tensor<?x{h}xf32>'
 return f'''module @segments {{
 func.func @zeros() -> {bank} {{
  %z = arith.constant 0.0 : f32
  %t = tensor.empty() : {bank}
  %r = linalg.fill ins(%z : f32) outs(%t : {bank}) -> {bank}
  return %r : {bank}
 }}
 func.func @extract(%x: {bank}, %eid: i64, %count: i64) -> {rows} {{
  %e = arith.index_cast %eid : i64 to index
  %n = arith.index_cast %count : i64 to index
  %r = tensor.extract_slice %x[%e, 0, 0] [1, %n, {h}] [1, 1, 1] : {bank} to {rows}
  return %r : {rows}
 }}
 func.func @insert(%x: {bank}, %v: {rows}, %eid: i64, %count: i64) -> {bank} {{
  %e = arith.index_cast %eid : i64 to index
  %n = arith.index_cast %count : i64 to index
  %r = tensor.insert_slice %v into %x[%e, 0, 0] [1, %n, {h}] [1, 1, 1] : {rows} into {bank}
  return %r : {bank}
 }}
}}'''

def export_shell(shell,m,out,s=256):
 import torch,iree.turbine.aot as aot
 e=shell.body.layers[0].mlp.E;k=shell.body.layers[0].mlp.K;h=shell.head.in_features
 aot.externalize_module_parameters(shell,external_scope='model');fx=aot.FxProgramsBuilder(shell)
 fx.export_program(lambda model,ids:model.start(ids),args=(torch.zeros(1,s,dtype=torch.int64),),name='start')
 x=torch.zeros(1,s,h)
 for i in range(len(shell.body.layers)):
  def step(model,x,index=i):return model.step(x,index)
  fx.export_program(step,args=(x,),name=f'layer{i}')
 fx.export_program(lambda model,r,a,ids,scores,ranks:model.combine(r,a,ids,scores,ranks),
  args=(x,torch.zeros(e,s,h),torch.zeros(s,k,dtype=torch.int64),torch.zeros(s,k),torch.zeros(e,s,dtype=torch.int32)),name='combine')
 fx.export_program(lambda model,x:model.finish(x),args=(x,),name='finish')
 fx.export_program(lambda model,x:model.diagnostic(x),args=(x,),name='diagnostic')
 program=aot.export(fx,module_name='shell');program.save_mlir(out/'shell.mlirbc')
 (out/'segments.mlir').write_text(segment_source(e,s,h))

def build(run,out,q):
 import torch,iree.turbine.aot as aot
 from transformers import OlmoeConfig,OlmoeForCausalLM
 from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding
 m,cfg,_,_=q.frozen(run);m.stack()
 old=run.parent.parent/'25C2_dynamic_expert_preparation/dynamic_expert_a100.vmfb'
 if not old.is_file() or q.sha(old)!=DYNAMIC_SHA:raise RuntimeError('Missing/changed validated A100 dynamic expert: '+str(old))
 config_path=Path(cfg['snapshot'])/'config.json'
 checkpoint=Path(cfg['out'])/'checkpoint.json'
 if q.sha(checkpoint)!=cfg['checkpoint_manifest_sha256']:raise RuntimeError('Checkpoint manifest changed')
 expected=json.loads(checkpoint.read_text())['files']['config.json']['sha256']
 if not config_path.is_file():
  from huggingface_hub import hf_hub_download
  config_path=Path(hf_hub_download(repo_id=m.MODEL,filename='config.json',revision=m.REV))
 if q.sha(config_path)!=expected:raise RuntimeError('Pinned model config changed')
 config=OlmoeConfig.from_dict(json.loads(config_path.read_text()));config._attn_implementation='sdpa'
 if (config.hidden_size,config.intermediate_size,config.num_experts,config.num_experts_per_tok,config.num_hidden_layers)!=(2048,1024,64,8,16):raise RuntimeError('Unexpected model')
 Grouped,Full=m.classes(torch)
 with torch.device('meta'):
  hf=OlmoeForCausalLM(config).eval()
  for layer in hf.model.layers:layer.mlp=Grouped(layer.mlp,256)
  full=Full(hf,'last').eval()
 full.body.rotary_emb=OlmoeRotaryEmbedding(config=config,device='cpu')
 shell=shell_class(torch,m)(full).eval()
 # Materialize only non-expert shell parameters on CPU. Expert banks are not
 # loaded into the export model, avoiding a second full FP32 model in RAM.
 archive=Path(cfg.get('weights_archive',str(run/'build/weights.irpa')))
 bank=aot.ParameterArchive(archive);entries=dict(bank.items())
 for name,parameter in list(shell.named_parameters()):
  value=entries[name].as_tensor()
  if tuple(value.shape)!=tuple(parameter.shape) or value.dtype!=torch.float32:raise RuntimeError('Shell archive mismatch: '+name)
  owner,_,attribute=name.rpartition('.')
  setattr(shell.get_submodule(owner),attribute,torch.nn.Parameter(value,requires_grad=False))
 export_shell(shell,m,out)
 for name in ('shell','segments'):
  source=out/(name+('.mlirbc' if name=='shell' else '.mlir'))
  m.compile_ir(cfg,source,out/(name+'.vmfb'),out/(name+'_compile.log'))
 shutil.copyfile(old,out/'dynamic_expert.vmfb')
 # Save actual helper lowering; full-model slices/copies need review.
 import iree.compiler as ic
 for stage in ('flow','stream'):
  data=ic.compile_file(str(out/'segments.mlir'),target_backends=['cuda'],output_format='mlir-text',extra_args=[*cfg['iree_flags'],'--compile-to='+stage])
  (out/('segments_'+stage+'.mlir')).write_bytes(data)
 hashes={name:q.sha(out/name) for name in ['shell.mlirbc','shell.vmfb','segments.mlir','segments.vmfb','dynamic_expert.vmfb']}
 q.save(out/'build.json',{'hashes':hashes,'dynamic_expert_sha256':DYNAMIC_SHA,'config_sha256':expected,
  'note':'New split shell; reused old dynamic expert. Padded routing/assembly, exact-row expert GEMMs.'})

class Dynamic:
 def __init__(self,run,out,q,driver='cuda',upload_weights=True):
  import iree.runtime as rt
  import numpy as np
  m,cfg,_,_=q.frozen(run)
  meta=json.loads((out/'build.json').read_text())
  for name,digest in meta['hashes'].items():
   if q.sha(out/name)!=digest:raise RuntimeError('Build artifact changed: '+name)
  self.cfg=rt.Config(driver);self.ctx=rt.SystemContext(config=self.cfg)
  archive=Path(cfg.get('weights_archive',str(run/'build/weights.irpa')))
  if cfg.get('weights_sha256') and q.sha(archive)!=cfg['weights_sha256']:raise RuntimeError('Archive changed')
  self.index=rt.ParameterIndex();self.index.load(str(archive))
  self.ctx.add_vm_module(rt.create_io_parameters_module(self.cfg.vm_instance,self.index.create_provider(scope='model')))
  for name in ['shell','segments','dynamic_expert']:
   self.ctx.add_vm_module(rt.VmModule.mmap(self.cfg.vm_instance,str(out/(name+'.vmfb'))))
  self.shell=self.ctx.modules.shell;self.seg=self.ctx.modules.segments
  self.expert=self.ctx.modules.olmoe_expert.forward;self.weights=[]
  if upload_weights:
   from iree.turbine.aot import ParameterArchive
   self.archive=ParameterArchive(archive);entries=dict(self.archive.items())
   for layer in range(16):
    bank=[]
    for suffix,shape in [('gu',(64,2048,2048)),('down',(64,1024,2048))]:
     tensor=entries[f'body.layers.{layer}.mlp.{suffix}'].as_tensor()
     if tuple(tensor.shape)!=shape:raise RuntimeError('Expert archive shape mismatch')
     # Grouped archive already stores [H,2I] and [I,H], the dynamic ABI.
     values=[]
     for e in range(64):
      values.append(rt.asdevicearray(self.cfg.device,np.ascontiguousarray(tensor[e].numpy())))
     bank.append(values)
    self.weights.append(list(zip(*bank)))
    print('Uploaded dynamic expert weights, layer',layer,flush=True)
 def run(self,ids,diagnostic=False):
  import numpy as np
  x=self.shell.start(ids);counts=[];snapshots={};active_total=0;rows_total=0
  for layer in range(len(self.weights)):
   residual,packed,c,ids_d,scores,ranks=getattr(self.shell,f'layer{layer}')(x)
   # This synchronization and readback is deliberately INSIDE every request.
   host=np.asarray(c.to_host());counts.append(host)
   if host.shape!=(64,) or (host<0).any() or (host>256).any() or host.sum()!=2048:raise RuntimeError('Invalid live routing counts')
   assembled=self.seg.zeros()
   for e in np.flatnonzero(host):
    n=int(host[e]);v=self.seg.extract(packed,int(e),n)
    y=self.expert(v,*self.weights[layer][int(e)])
    assembled=self.seg.insert(assembled,y,int(e),n)
   active_total+=int((host>0).sum());rows_total+=int(host.sum())
   x=self.shell.combine(residual,assembled,ids_d,scores,ranks)
   if diagnostic and layer in (0,7):snapshots['_hidden'+str(layer+1)]=np.asarray(x.to_host())
  if diagnostic:
   all_logits,hidden=self.shell.diagnostic(x)
   snapshots.update(_all_logits=np.asarray(all_logits.to_host()),_hidden16=np.asarray(hidden.to_host()))
  logits=np.asarray(self.shell.finish(x).to_host())
  return logits,np.stack(counts),snapshots,{'active_expert_calls':active_total,'actual_expert_rows':rows_total,
   'metadata_readbacks':16,'iree_calls':50+3*active_total}

def worker(run,out,session,arm,a,q):
 import numpy as np,iree.runtime as rt
 m,cfg,original,_=q.frozen(run);m.stack()
 root=Path(cfg.get('reference_root',str(run/'references')));manifest=json.loads((root/'manifest.json').read_text())
 if arm=='dynamic':runner=Dynamic(run,out/'build',q);device=runner.cfg.device
 else:
  binary=original if arm=='original' else q.tuned_model(run)
  expected=q.ORIGINAL_SHA if arm=='original' else q.TUNED_SHA
  if q.sha(binary)!=expected:raise RuntimeError('Static binary changed')
  _,keep=m.iree_fn(binary,Path(cfg.get('weights_archive',str(run/'build/weights.irpa'))),entry='prefill_s256_c256')
  module=keep['ctx'].modules.olmoe;device=keep['cfg'].device
 checks={};samples=[];failures=[];cases=[c for c in cfg['cases'] if c['length']==256]
 if len(cases)!=4:raise RuntimeError('Expected four S256 fixtures')
 for case in cases:
  name=case['name'];path=run/case['input']
  if q.sha(path)!=case['sha256']:raise RuntimeError('Frozen input changed')
  ids=rt.asdevicearray(device,np.load(path));ref=np.load(root/(name+'_logits.npy'))
  def request():
   if arm=='dynamic':
    y,c,_,_=runner.run(ids);return y,c
   y,c=module.prefill_s256_c256(ids);return np.asarray(y.to_host()),np.asarray(c.to_host())
  if arm=='dynamic':y,c,diags,structure=runner.run(ids,True)
  else:
   d=module.diagnostic_s256(ids);diags={s:np.asarray(d[i].to_host()) for i,s in [(0,'_all_logits'),(2,'_hidden1'),(3,'_hidden8'),(4,'_hidden16')]}
   y,c=request();structure={'host_orchestration':'one full-model invocation'}
   np.testing.assert_array_equal(d[1].to_host(),manifest[name]['counts'])
  metrics={s:q.metric(v,np.load(root/(name+s+'.npy'))) for s,v in diags.items()}
  if not metrics['_all_logits']['allclose']:raise RuntimeError(name+' all-position logits failed')
  failures.extend({'case':name,'stage':s,**v} for s,v in metrics.items() if not v['allclose'])
  if not q.metric(y,ref)['allclose']:raise RuntimeError(name+' serving logits failed')
  np.testing.assert_array_equal(c,manifest[name]['counts'])
  direct={**diags,'serving_logits':y,'serving_counts':c}
  cross=q.direct_outputs(out,session,name,direct,session.name=='session0_original')
  checks[name]={'reference':metrics,'direct_original':cross,'structure':structure,'serving_pre':q.metric(y,ref)}
  q.save(session/'checks.json',checks)
  warm=q.stable_warmup(request,a.warmups);q.save(session/(name+'_warmup.json'),warm)
  if warm['status']!='STABLE_WARMUP':raise RuntimeError(name+' warmup did not stabilize')
  for index in range(a.repeats):
   begin=time.perf_counter_ns();y,c=request();duration=(time.perf_counter_ns()-begin)/1e6
   samples.append({'case':name,'arm':arm,'sample':index,'wall_ms':duration})
   q.save(session/'progress.json',samples)
  if not q.metric(y,ref)['allclose']:raise RuntimeError(name+' timed serving output failed')
  np.testing.assert_array_equal(c,manifest[name]['counts'])
  with np.load(out/'baseline_outputs'/(name+'.npz')) as anchor:
   post=q.metric(y,anchor['serving_logits'])
  if not post['allclose']:raise RuntimeError(name+' direct postflight failed')
  checks[name]['serving_post']=q.metric(y,ref);checks[name]['direct_post']=post
  q.save(session/'checks.json',checks)
  print(arm,name,'median_ms',statistics.median(r['wall_ms'] for r in samples if r['case']==name),flush=True)
 q.save(session/'status.json',{'arm':arm,'hidden_original_gate_failures':failures,'status':'SERVING_PASS_HIDDEN_REVIEW' if failures else 'PASS'})
 q.save(session/'samples.json',samples)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',default='/content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/20261028T140743_199932Z')
 p.add_argument('--python',default='/content/27_aot_venv/bin/python');p.add_argument('--setup',action='store_true')
 p.add_argument('--out');p.add_argument('--resume');p.add_argument('--worker',choices=['build',*set(SCHEDULE)])
 p.add_argument('--session');p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmups',type=int,default=10)
 a=p.parse_args()
 if a.repeats<10 or not 10<=a.warmups<=60:p.error('Require >=10 repeats and warmups in [10,60]')
 companion=Path(__file__).with_name('42_final_full_model_gap.py')
 import hashlib
 if hashlib.sha256(companion.read_bytes()).hexdigest()!='40387ad2605f99a8f4c8deb1db5c012b240e5312cb909b201da4659b189a5cd0':raise RuntimeError('Experiment-42 companion changed')
 q=load(companion,'gap42')
 run=Path(a.run).resolve()
 if a.worker:
  out=Path(a.out)
  if a.worker=='build':build(run,out/'build',q)
  else:worker(run,out,Path(a.session),a.worker,a,q)
  return 0
 m,cfg,original,_=q.frozen(run);tuned=q.tuned_model(run)
 out=Path(a.resume).resolve() if a.resume else run/'dynamic_full_model_comparison'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
 out.mkdir(parents=True,exist_ok=bool(a.resume));(out/'build').mkdir(exist_ok=True)
 req={'script_sha256':q.sha(__file__),'companion_sha256':q.sha(q.__file__),'schedule':SCHEDULE,'sequence':256,
  'repeats':a.repeats,'warmups':a.warmups,'original_sha256':q.sha(original),'tuned_sha256':q.sha(tuned),
  'source_sha256':cfg['source_sha256'],'cases':[c for c in cfg['cases'] if c['length']==256],'precision':'FP32; TF32 disabled',
  'boundary':'Resident input to completed host next-token logits and all-layer expert counts; no KV cache.',
  'interpretation':'Implementation comparison: orchestration, partitioning and kernels differ. Not a causal isolation of dynamic shapes.',
  'dynamic':'16 host count readbacks; host per-expert extract/expert/insert; padded route and output assembly; exact rows only in expert GEMMs.'}
 if a.resume:
  if json.loads((out/'request.json').read_text())!=req:raise RuntimeError('Resume settings changed')
 else:
  q.save(out/'request.json',req);shutil.copyfile(__file__,out/'experiment.py');shutil.copyfile(q.__file__,out/'42_final_full_model_gap.py')
 python=Path(a.python)
 if a.setup:
  python=Path(m.environment_python(python.parent.parent,out,1200))
  subprocess.run([str(python),'-m','pip','install',*m.PINS],check=True)
 if not python.is_file():raise FileNotFoundError('Pinned venv missing; use --setup')
 common=[str(python),str(out/'experiment.py'),'--run',str(run),'--out',str(out),'--repeats',str(a.repeats),'--warmups',str(a.warmups)]
 report={'status':'STARTED','request':req};rc=0
 try:
  if not (out/'build/build.json').exists():
   if q.logged([*common,'--worker','build'],out/'build.log',2400):raise RuntimeError('Build failed; preserve log. No timing conclusion.')
  rows=[];statuses=[]
  for i,arm in enumerate(SCHEDULE):
   session=out/f'session{i}_{arm}';session.mkdir(exist_ok=True)
   if not (session/'samples.json').exists():
    if q.logged([*common,'--worker',arm,'--session',str(session)],out/(session.name+'.log'),3600):raise RuntimeError('Session failed: '+session.name)
   data=json.loads((session/'samples.json').read_text());status=json.loads((session/'status.json').read_text())
   if len(data)!=4*a.repeats or status['arm']!=arm:raise RuntimeError('Incomplete/mismatched session')
   rows.extend({**r,'session':i} for r in data);statuses.append(status);q.save(out/'validated_samples.json',rows)
  summary={};reasons=[]
  for case in sorted({r['case'] for r in rows}):
   med={arm:statistics.median(r['wall_ms'] for r in rows if r['case']==case and r['arm']==arm) for arm in set(SCHEDULE)}
   sm={i:statistics.median(r['wall_ms'] for r in rows if r['case']==case and r['session']==i) for i in range(6)}
   drift={arm:100*(sm[max(i for i,v in enumerate(SCHEDULE) if v==arm)]/sm[min(i for i,v in enumerate(SCHEDULE) if v==arm)]-1) for arm in set(SCHEDULE)}
   within={}
   for i in range(6):
    v=[r['wall_ms'] for r in rows if r['case']==case and r['session']==i];n=len(v)//2
    within[i]=100*(statistics.median(v[n:])/statistics.median(v[:n])-1)
   if any(abs(v)>2 for v in [*drift.values(),*within.values()]):reasons.append(case+': drift >2%')
   summary[case]={'median_ms':med,'session_medians_ms':sm,'repeat_drift_percent':drift,'within_session_drift_percent':within,
    'dynamic_over_original':med['dynamic']/med['original'],'original_over_tuned':med['original']/med['tuned'],'dynamic_over_tuned':med['dynamic']/med['tuned']}
  hidden=[{'session':i,**f} for i,s in enumerate(statuses) for f in s['hidden_original_gate_failures']]
  if hidden:reasons.append('reference_hidden_gate_failures; inspect whether new dynamic deviations differ')
  means={arm:statistics.mean(v['median_ms'][arm] for v in summary.values()) for arm in set(SCHEDULE)}
  report.update(status='COMPLETE_REVIEW_REQUIRED',cases=summary,mean_of_prompt_medians_ms=means,review_reasons=reasons,hidden_failures=hidden,
   dynamic_over_tuned=means['dynamic']/means['tuned'],original_over_tuned=means['original']/means['tuned'])
 except BaseException:
  report.update(status='FAILED_OR_INCOMPLETE',error=traceback.format_exc());print(report['error'],flush=True);rc=1
 q.save(out/'report.json',report)
 bundle=out/'dynamic_full_model_comparison_bundle.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
  for f in out.rglob('*'):
   if f.is_file() and f.suffix in ('.py','.json','.log','.mlir'):z.write(f,f.relative_to(out))
 print('Status:',report['status']);print('Output:',out);print('Bundle:',bundle)
 if 'mean_of_prompt_medians_ms' in report:print(report['mean_of_prompt_medians_ms'])
 return rc
if __name__=='__main__':sys.exit(main())
