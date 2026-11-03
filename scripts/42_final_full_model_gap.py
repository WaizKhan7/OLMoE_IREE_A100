#!/usr/bin/env python3
"""Final frozen full-model gap comparison: original IREE, tuned IREE, Torch.
Original/Torch/tuned/tuned/Torch/original isolated processes; eight frozen cases.
Tuned model is frozen experiment 41: C256 gate/up rows4 + down rows4.
No new kernel search, IREE compilation, model export, or checkpoint download.
Bounded stable warm-up; 20 measured calls per case per process, no sample trimming.
Torch uses the SAME archive, fullgraph reduce-overhead compilation and exact
capacity C=S. IREE artifacts are reused without compilation. All arms use FP32,
resident input -> completed HOST serving logits and expert counts. No KV cache.
Diagnostic hidden failures against HF remain explicit; tolerances stay 3e-4.
Torch all-position/hidden diagnostics validate the eager grouped formulation;
actual compiled serving logits/counts are separately blocking before/after timing.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,importlib.util,json,math,os
import re,shutil,statistics,subprocess,sys,time,traceback,zipfile

RUN='/content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/20261028T140743_199932Z'
TOL=3e-4
ORIGINAL_SHA='58cd4c2ce0c9b9a7ebd8cc15a848897e6dc56a0ac6287c43e2bff78fcafb83ce'
TUNED_SHA='66134907277398d4b445d13289e8fa726cf9ecde5021dd2d8fda483b4c5d84d3'
SCHEDULE=['original','torch','tuned','tuned','torch','original']


def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def frozen(run):
 cfg=json.loads((run/'config.json').read_text());source=run/'experiment.py'
 if sha(source)!=cfg['source_sha256']:raise RuntimeError('Frozen experiment source changed')
 spec=importlib.util.spec_from_file_location('frozen42',source)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 binary=run/'build/full_model.vmfb'
 if sha(binary)!=ORIGINAL_SHA:raise RuntimeError('Original model VMFB changed')
 return m,cfg,binary,{'sha256':ORIGINAL_SHA}


def tuned_model(run):
 matches=[]
 for path in sorted((run/'full_model_down_rows4_ab').glob('*/build.json')):
  meta=json.loads(path.read_text());binary=path.parent/'tuned.vmfb'
  if meta.get('tuned_vmfb_sha256')==TUNED_SHA and binary.exists():
   if sha(binary)!=TUNED_SHA:raise RuntimeError('Tuned VMFB hash mismatch')
   matches.append(binary)
 if not matches:raise FileNotFoundError('Missing experiment-41 gate/up rows4 + down rows4 VMFB on Drive')
 return matches[-1]


def archive_model(m,config,archive,device='cuda'):
 """Construct metadata on meta, then materialize only grouped archived tensors."""
 import torch
 from transformers import OlmoeForCausalLM
 from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding
 from iree.turbine.aot import ParameterArchive
 Grouped,Full=m.classes(torch)
 with torch.device('meta'):
  hf=OlmoeForCausalLM(config).eval()
  for layer in hf.model.layers:layer.mlp=Grouped(layer.mlp,256)
  model=Full(hf,'last').eval()
 bank=ParameterArchive(archive);entries=dict(bank.items());state=model.state_dict()
 if set(entries)!=set(state):raise RuntimeError('Grouped archive schema differs from model: '+str(set(entries)^set(state)))
 for name,parameter in state.items():
  value=entries[name].as_tensor()
  if tuple(value.shape)!=tuple(parameter.shape) or value.dtype!=torch.float32:raise RuntimeError('Weight layout/type mismatch: '+name)
  value=value.to(device=device)
  owner,_,attribute=name.rpartition('.')
  setattr(model.get_submodule(owner),attribute,torch.nn.Parameter(value,requires_grad=False))
 # Nonpersistent rotary buffers are computed, rather than read from the archive.
 model.body.rotary_emb=OlmoeRotaryEmbedding(config=config,device=device)
 if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):raise RuntimeError('Unmaterialized model tensor')
 model.requires_grad_(False);return model,bank


def torch_module(m,cfg,out):
 import torch
 from transformers import OlmoeConfig
 checkpoint=Path(cfg['out'])/'checkpoint.json'
 if sha(checkpoint)!=cfg['checkpoint_manifest_sha256']:raise RuntimeError('Checkpoint manifest changed')
 expected=json.loads(checkpoint.read_text())['files']['config.json']['sha256']
 config_path=Path(cfg['snapshot'])/'config.json'
 if not config_path.exists():
  from huggingface_hub import hf_hub_download
  print('Fetching pinned model configuration only; archived weights reused',flush=True)
  config_path=Path(hf_hub_download(repo_id=m.MODEL,filename='config.json',revision=m.REV))
 if sha(config_path)!=expected:raise RuntimeError('Pinned model configuration hash mismatch')
 config=OlmoeConfig.from_dict(json.loads(config_path.read_text()));config._attn_implementation='sdpa'
 if (config.hidden_size,config.intermediate_size,config.num_experts,config.num_experts_per_tok,config.num_hidden_layers)!=(2048,1024,64,8,16):raise RuntimeError('Unexpected model contract')
 archive=Path(cfg.get('weights_archive',str(Path(cfg['out'])/'build/weights.irpa')))
 model,bank=archive_model(m,config,archive)
 compiled=torch.compile(model,backend='inductor',fullgraph=True,dynamic=False,mode='reduce-overhead')
 save(out/'compile_settings.json',{'backend':'inductor','mode':'reduce-overhead','fullgraph':True,'dynamic':False,
   'tf32':False,'capacity':'C=S','config_sha256':expected,'archive':str(archive),
   'diagnostic':'Eager grouped formulation; actual compiled serving outputs checked separately.',
   'cuda_graphs':'Requested; inspect Dynamo counters rather than assuming capture.'})
 class Host:
  def __init__(self,tensor):self.tensor=tensor
  def to_host(self):return self.tensor.detach().cpu().numpy()
 class Module:
  def __getattr__(self,name):
   sequence=int(name.split('_s')[1].split('_')[0]);m.set_capacity(model,sequence)
   def invoke(ids):
    with torch.inference_mode():
     values=model(ids,diagnostic=True) if name.startswith('diagnostic') else compiled(ids)
    return tuple(Host(v) for v in values)
   return invoke
 return Module(),{'model':model,'compiled':compiled,'bank':bank}


def metric(actual,reference):
 import numpy as np
 if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
  raise AssertionError('Shape or nonfinite output failure')
 delta=np.abs(actual.astype(np.float64)-reference.astype(np.float64))
 return {'allclose':bool(np.allclose(actual,reference,rtol=TOL,atol=TOL)),
   'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
   'violations':int((delta>TOL+TOL*np.abs(reference.astype(np.float64))).sum()),'rtol':TOL,'atol':TOL}


def direct_outputs(out,session,name,arrays,is_anchor):
 import numpy as np
 root=out/'baseline_outputs';root.mkdir(exist_ok=True)
 path=root/(name+'.npz');index=root/(name+'.json')
 if is_anchor:
  temporary=path.with_suffix('.tmp')
  with open(temporary,'wb') as f:np.savez(f,**arrays)
  temporary.replace(path);save(index,{'sha256':sha(path),'keys':sorted(arrays)})
  return {'anchor':True,'sha256':sha(path)}
 info=json.loads(index.read_text())
 if sha(path)!=info['sha256']:raise RuntimeError('A0 tensor snapshot changed: '+name)
 checks={}
 with np.load(path) as baseline:
  if sorted(arrays)!=info['keys']:raise RuntimeError('Direct-output key mismatch')
  for key,value in arrays.items():
   if key.endswith('counts'):
    checks[key]={'equal':bool(np.array_equal(value,baseline[key]))}
    passed=checks[key]['equal']
   else:
    checks[key]=metric(value,baseline[key]);checks[key]['elementwise_equal']=bool(np.array_equal(value,baseline[key]));passed=checks[key]['allclose']
   if not passed:
    save(session/'direct_failure.json',{'case':name,'stage':key,'checks':checks})
    raise AssertionError(name+' direct A0 comparison failed: '+key)
 return checks


def stable_warmup(request,minimum_calls):
 # No timed sample is discarded: stabilization happens before measurement.
 durations=[];started=time.monotonic()
 for number in range(max(60,minimum_calls)):
  call_start=time.perf_counter_ns();output=request()
  durations.append((time.perf_counter_ns()-call_start)/1e6);del output
  elapsed=time.monotonic()-started
  if len(durations)>=minimum_calls and elapsed>=5:
   previous=statistics.median(durations[-10:-5]);recent=statistics.median(durations[-5:])
   drift=100*(recent/previous-1)
   if abs(drift)<=2:
    return {'status':'STABLE_WARMUP','wall_ms':durations,'seconds':elapsed,'last_window_drift_percent':drift,
      'criterion':'At least 10 calls/5 seconds, last two 5-call medians within 2%; bounded at 60 calls/45 seconds.'}
  if elapsed>=45:break
 return {'status':'UNSTABLE_WARMUP','wall_ms':durations,'seconds':time.monotonic()-started}


def capture(fn,path):
 from torch.profiler import profile,ProfilerActivity,record_function
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
  with record_function('full_model.completed'):output=fn()
 prof.export_chrome_trace(str(path));del prof
 events=json.loads(path.read_text())['traceEvents']
 kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
 if not kernels:raise RuntimeError('No CUDA kernels captured')
 grouped={}
 for e in kernels:
  g=grouped.setdefault(e['name'],{'calls':0,'total_ms':0.,'grid':e.get('args',{}).get('grid'),
                  'block':e.get('args',{}).get('block')})
  g['calls']+=1;g['total_ms']+=e['dur']/1000
 return output,{'kernel_sum_ms':sum(e['dur'] for e in kernels)/1000,
     'ranked_kernels':sorted([{'name':name,**v} for name,v in grouped.items()],key=lambda x:-x['total_ms']),
     'cuda_graph_launch_events':sum('cudaGraphLaunch' in e.get('name','') for e in events)}


def measure(run,out,session,a):
 import numpy as np,iree.runtime as rt
 m,cfg,original,meta=frozen(run);m.stack()
 if cfg['logits']!='last':raise RuntimeError('This comparison fixes next-token logits')
 for key in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']:
  if os.environ.get(key) not in (None,'','0'):raise RuntimeError('Unexpected TF32 override: '+key)
 import torch
 expected=json.loads((out/'artifacts.json').read_text())
 binary=original if a.arm=='original' else Path(expected['tuned_path'])
 if a.arm!='torch' and sha(binary)!=(ORIGINAL_SHA if a.arm=='original' else TUNED_SHA):raise RuntimeError('VMFB changed')
 weights=Path(cfg.get('weights_archive',str(run/'build/weights.irpa')))
 print('Loading',a.arm,'from the saved archive; startup excluded',flush=True)
 start=time.perf_counter()
 if a.arm=='torch':module,keep=torch_module(m,cfg,session);device=None
 else:
  fn,keep=m.iree_fn(binary,weights,entry='prefill_s64_c64');module=keep['ctx'].modules.olmoe;device=keep['cfg'].device
 load_seconds=time.perf_counter()-start
 print('Loaded',a.arm,'seconds',round(load_seconds,2),flush=True)
 reference_root=Path(cfg.get('reference_root',str(run/'references')))
 manifest=json.loads((reference_root/'manifest.json').read_text())
 cases=[case for case in cfg['cases'] if case['length'] in (64,256)]
 if len(cases)!=8:raise RuntimeError('Expected four frozen prompts at each of two lengths')
 checks={};samples=[];profiles={}
 is_anchor=session.name=='session0_original'
 for case in cases:
  name=case['name'];s=case['length'];entry=getattr(module,f'prefill_s{s}_c{s}')
  input_path=run/case['input']
  if sha(input_path)!=case['sha256']:raise RuntimeError('Frozen input changed')
  ids=torch.from_numpy(np.load(input_path)).cuda() if a.arm=='torch' else rt.asdevicearray(device,np.load(input_path))
  ref=np.load(reference_root/(name+'_logits.npy'))
  def request():
   y,h=entry(ids);return np.asarray(y.to_host()),np.asarray(h.to_host())
  debug=getattr(module,f'diagnostic_s{s}')(ids);diag={};direct={} 
  for index,suffix in [(0,'_all_logits'),(2,'_hidden1'),(3,'_hidden8'),(4,'_hidden16')]:
   value=np.asarray(debug[index].to_host());direct[suffix]=value;reference=np.load(reference_root/(name+suffix+'.npy'))
   diag[suffix]=metric(value,reference)
   if suffix=='_all_logits' and not diag[suffix]['allclose']:
    save(session/'validation_failure.json',{'case':name,'stage':suffix,'metrics':diag[suffix]})
    raise AssertionError(name+' all-position logits failed')
  direct['diagnostic_counts']=np.asarray(debug[1].to_host())
  if not np.array_equal(direct['diagnostic_counts'],np.asarray(manifest[name]['counts'])):raise AssertionError('Diagnostic counts differ')
  del debug
  value,counts=request();pre=metric(value,ref)
  direct['serving_logits']=value;direct['serving_counts']=counts
  cross=direct_outputs(out,session,name,direct,is_anchor);del direct
  if not pre['allclose']:
   save(session/'validation_failure.json',{'case':name,'stage':'serving_pre','metrics':pre})
   raise AssertionError(name+' serving logits failed')
  if not np.array_equal(counts,np.asarray(manifest[name]['counts'])):raise AssertionError(name+' expert counts differ')
  if (counts>s).any():raise AssertionError('Unexpected exact-capacity overflow')
  checks[name]={'diagnostic':diag,'serving_pre':pre,'counts_equal':True,'direct_A0':cross}
  save(session/'checks.json',checks)
  warm=stable_warmup(request,a.warmups)
  save(session/(name+'_warmup.json'),warm)
  if warm['status']!='STABLE_WARMUP':raise RuntimeError(name+' failed bounded warm-up stability; preserve output')
  checks[name]['warmup_seconds']=warm['seconds']
  if a.arm=='torch':
   from torch._dynamo.utils import counters
   graphs_before=counters['stats']['unique_graphs']
  for sample in range(a.repeats):
   start=time.perf_counter_ns();value,counts=request()
   samples.append({'arm':a.arm,'case':name,'sample':sample,'wall_ms':(time.perf_counter_ns()-start)/1e6})
  if a.arm=='torch':
   graphs_after=counters['stats']['unique_graphs']
   checks[name]['compiled_graphs_during_timing']=graphs_after-graphs_before
   if graphs_after!=graphs_before:raise RuntimeError('Torch compiled a new graph during timing: '+name)
  value,counts=request();post=metric(value,ref)
  checks[name]['serving_post']=post
  with np.load(out/'baseline_outputs'/(name+'.npz')) as baseline:
   direct_post=metric(value,baseline['serving_logits'])
  checks[name]['direct_serving_post']=direct_post
  if not direct_post['allclose']:raise AssertionError(name+' direct postflight logits failed')
  if not post['allclose']:
   save(session/'validation_failure.json',{'case':name,'stage':'serving_post','metrics':post})
   raise AssertionError(name+' postflight logits failed')
  if not np.array_equal(counts,np.asarray(manifest[name]['counts'])):raise AssertionError('Postflight counts differ')
  if a.profile and name in ('compiler_s64','compiler_s256'):
   (value,counts),p=capture(request,session/(name+'.trace.json'))
   if not metric(value,ref)['allclose'] or not np.array_equal(counts,np.asarray(manifest[name]['counts'])):raise AssertionError('Profile output/counts failed')
   profiles[name]=p;save(session/'profiles.json',profiles)
   if a.arm=='torch' and p['cuda_graph_launch_events']==0:raise RuntimeError(name+' Torch CUDA graph launch not observed; preserve trace')
  save(session/'checks.json',checks);save(session/'progress.json',samples)
  vals=[v['wall_ms'] for v in samples if v['case']==name]
  print(a.arm,name,'median_ms',round(statistics.median(vals),3),'logits/counts PASS',flush=True)
 hidden_fail=[{'case':name,'stage':stage,**v} for name,c in checks.items()
             for stage,v in c['diagnostic'].items() if stage!='_all_logits' and not v['allclose']]
 save(session/'status.json',{'status':'FUNCTIONAL_OUTPUTS_PASS_INTERNAL_REVIEW' if hidden_fail else 'PASS',
       'arm':a.arm,'load_seconds':load_seconds,'vmfb_sha256':None if a.arm=='torch' else sha(binary),
       'hidden_original_gate_failures':hidden_fail})
 if a.arm=='torch':
  from torch._dynamo.utils import counters
  save(session/'dynamo_counters.json',{k:dict(v) for k,v in counters.items()})
 # Final samples exist only after all gates in the session complete.
 save(session/'samples.json',samples)


def logged(cmd,path,timeout):
 start=time.monotonic()
 with open(path,'w') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
  while proc.poll() is None:
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:
    lines=Path(path).read_text(errors='replace').splitlines()
    print(Path(path).stem,int(time.monotonic()-start),'seconds |',lines[-1][-260:] if lines else 'starting',flush=True)
    if time.monotonic()-start>timeout:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     return 124
 if proc.returncode:print(Path(path).read_text(errors='replace')[-4000:],flush=True)
 return proc.returncode


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',default=RUN);p.add_argument('--python',default='/content/27_aot_venv/bin/python')
 p.add_argument('--setup',action='store_true');p.add_argument('--resume')
 p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmups',type=int,default=10)
 p.add_argument('--worker',action='store_true');p.add_argument('--out');p.add_argument('--session')
 p.add_argument('--arm',choices=['original','torch','tuned']);p.add_argument('--profile',action='store_true')
 a=p.parse_args()
 if a.repeats<10 or not 10<=a.warmups<=60:p.error('Require repeats >=10 and warmups in [10,60]')
 run=Path(a.run).resolve()
 if a.worker:measure(run,Path(a.out),Path(a.session),a);return 0
 m,cfg,original,meta=frozen(run);tuned=tuned_model(run)
 out=Path(a.resume).resolve() if a.resume else run/'final_full_model_gap'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
 out.mkdir(parents=True,exist_ok=bool(a.resume))
 request={'script_sha256':sha(__file__),'source_run':str(run),'source_sha256':cfg['source_sha256'],
   'repeats':a.repeats,'warmups':a.warmups,'schedule':SCHEDULE,'original_sha256':ORIGINAL_SHA,'tuned_sha256':TUNED_SHA,
   'torch_compile':{'mode':'reduce-overhead','fullgraph':True,'dynamic':False},'lengths':[64,256],
   'boundary':'Resident input -> completed host serving logits and all layer expert counts; no KV cache.',
   'warmup':'Minimum 10 calls and five seconds, adjacent five-call medians within 2%; bounded 60 calls/45 seconds.',
   'tuned_configuration':'C256 gate/up rows4 + down rows4; S64 same as previous tuned baseline.',
   'precision':'FP32, TF32 disabled', 'drift_threshold_percent':2}
 if a.resume:
  if json.loads((out/'request.json').read_text())!=request or sha(out/'experiment.py')!=sha(__file__):raise RuntimeError('Resume settings/script changed')
 else:
  save(out/'request.json',request);shutil.copyfile(__file__,out/'experiment.py')
  save(out/'artifacts.json',{'original_path':str(original),'original_sha256':ORIGINAL_SHA,'tuned_path':str(tuned),'tuned_sha256':TUNED_SHA})
 python=Path(a.python)
 if a.setup:
  python=Path(m.environment_python(python.parent.parent,out,1200))
  probe=subprocess.run([str(python),'-c',"import importlib.metadata as m,sys;sys.exit(any(m.version(s.split('==')[0]).split('+')[0]!=s.split('==')[1] for s in sys.argv[1:]))",*m.PINS],capture_output=True)
  if probe.returncode:subprocess.run([str(python),'-m','pip','install',*m.PINS],check=True)
 if not python.is_file():raise FileNotFoundError('Missing pinned venv; use --setup')
 common=[str(python),str(out/'experiment.py'),'--run',str(run),'--out',str(out)]
 report={'status':'STARTED','request':request,'sampling':'Six isolated processes; two per arm; descriptive medians, not paired confidence intervals.',
   'limitations':['Torch actual compiled serving outputs/counts are blocking; Torch all-position/hidden diagnostic tensors come from eager grouped formulation.',
    'Functional output checks do not close the pre-existing HF-reference hidden-state numerical review.',
    'CUDA graphs requested for Torch; actual capture/recompilation evidence retained in counters/logs.',
    'Configuration may be fetched at pinned revision if absent locally; no checkpoint weight download.']}
 save(out/'report.json',report);rc=0
 try:
  all_rows=[];statuses=[]
  for number,arm in enumerate(SCHEDULE):
   session=out/f'session{number}_{arm}';session.mkdir(exist_ok=True)
   if (session/'samples.json').exists():
    status=json.loads((session/'status.json').read_text())
    expected=None if arm=='torch' else ORIGINAL_SHA if arm=='original' else TUNED_SHA
    if status['arm']!=arm or status['vmfb_sha256']!=expected:raise RuntimeError('Resume session identity mismatch')
    for case in cfg['cases']:
     if case['length'] not in (64,256):continue
     name=case['name'];snapshot=out/'baseline_outputs'/(name+'.npz')
     if sha(snapshot)!=json.loads(snapshot.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Anchor output changed')
   else:
    cmd=[*common,'--worker','--session',str(session),'--arm',arm,'--repeats',str(a.repeats),'--warmups',str(a.warmups)]
    if number<3:cmd.append('--profile')
    # Torch first-call compilation and Drive cold loads are outside timing.
    code=logged(cmd,out/f'session{number}_{arm}.log',2400)
    if code:raise RuntimeError(f'{arm} session {number} failed; inspect preserved log and use --resume after resolving environment failures')
   rows=json.loads((session/'samples.json').read_text())
   if len(rows)!=8*a.repeats:raise RuntimeError('Incomplete sample count')
   all_rows.extend({**row,'session':number} for row in rows)
   statuses.append(json.loads((session/'status.json').read_text()))
   save(out/'validated_samples.json',all_rows)
  summary={}
  for name in sorted({x['case'] for x in all_rows}):
   med={arm:statistics.median(x['wall_ms'] for x in all_rows if x['case']==name and x['arm']==arm) for arm in set(SCHEDULE)}
   session_med={str(i):statistics.median(x['wall_ms'] for x in all_rows if x['case']==name and x['session']==i) for i in range(6)}
   within={}
   for i in range(6):
    values=[x['wall_ms'] for x in all_rows if x['case']==name and x['session']==i]
    half=len(values)//2
    within[str(i)]=100*(statistics.median(values[half:])/statistics.median(values[:half])-1)
   drift={arm:100*(session_med[str(max(i for i,x in enumerate(SCHEDULE) if x==arm))]/session_med[str(min(i for i,x in enumerate(SCHEDULE) if x==arm))]-1) for arm in set(SCHEDULE)}
   original_excess=med['original']-med['torch']
   summary[name]={'median_ms':med,'session_medians_ms':session_med,'repeat_drift_percent':drift,'within_session_half_drift_percent':within,
    'original_iree_over_torch':med['original']/med['torch'],'tuned_iree_over_torch':med['tuned']/med['torch'],
    'iree_speedup':med['original']/med['tuned'],'iree_latency_reduction_percent':100*(1-med['tuned']/med['original']),
    'excess_latency_gap_closed_percent':100*(med['original']-med['tuned'])/original_excess if original_excess>0 else None}
  hidden=[{'session':i,**item} for i,status in enumerate(statuses) for item in status['hidden_original_gate_failures']]
  drift=any(abs(v)>2 for c in summary.values() for v in c['repeat_drift_percent'].values())
  within_drift=any(abs(v)>2 for c in summary.values() for v in c['within_session_half_drift_percent'].values())
  reasons=(['reference_hidden_gate_failures'] if hidden else [])+(['repeat_drift_over_2_percent'] if drift else [])+(['within_session_drift_over_2_percent'] if within_drift else [])
  by_length={}
  for length in (64,256):
   selected=[v for k,v in summary.items() if k.endswith('_s'+str(length))]
   med={arm:statistics.mean(v['median_ms'][arm] for v in selected) for arm in set(SCHEDULE)}
   excess=med['original']-med['torch']
   by_length[str(length)]={'mean_of_prompt_medians_ms':med,'iree_speedup':med['original']/med['tuned'],
    'tuned_iree_over_torch':med['tuned']/med['torch'],
    'excess_latency_gap_closed_percent':100*(med['original']-med['tuned'])/excess if excess>0 else None}
  save(out/'summary.json',{'cases':summary,'by_length':by_length})
  report.update(status='PROVISIONAL_NUMERICAL_OR_DRIFT_REVIEW_REQUIRED' if reasons else 'COMPLETE_REVIEW_REQUIRED',
    review_reasons=reasons,summary=summary,by_length=by_length,hidden_original_gate_failures=hidden)
 except BaseException:
  report.update(status='FAILED_OR_INCOMPLETE',error=traceback.format_exc());print(report['error'],flush=True);rc=1
 save(out/'report.json',report)
 bundle=out/'final_full_model_gap_bundle.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.json','.log','.py'):z.write(path,str(path.relative_to(out)))
 for name,c in report.get('summary',{}).items():print(name,c['median_ms'],'remaining gap',round(c['tuned_iree_over_torch'],3),'closed %',c['excess_latency_gap_closed_percent'])
 print('Status:',report['status']);print('Review reasons:',report.get('review_reasons',[]));print('Output:',out);print('Bundle:',bundle)
 return rc


if __name__=='__main__':sys.exit(main())
