#!/usr/bin/env python3
"""Fresh S256 four-way A100 comparison using frozen experiments 42 and 44.

Copy this single file to /content/drive/MyDrive/moe_iree_olmoe/ and run:
  !/content/27_aot_venv/bin/python /content/drive/MyDrive/moe_iree_olmoe/46_four_way_full_model_comparison.py

Reuses the existing experiment-44 build, experiment-42 helper, and frozen
experiment-27 weights. No new IREE compilation or weight download. Eight
isolated workers, four prompts, 20 measured calls per prompt and worker.
The numerical policy is written to request.json before any workers execute.
"""
from pathlib import Path
import argparse, datetime, importlib.util, json, os, shutil, statistics, subprocess, sys, time, traceback, zipfile

RUN = Path('/content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/20261028T140743_199932Z')
BUILD = RUN/'dynamic_full_model_comparison/20261030T111420_950487Z'
Q_SHA = '40387ad2605f99a8f4c8deb1db5c012b240e5312cb909b201da4659b189a5cd0'
D_SHA = 'e246b4a311e2bbc169d05febe76b2bbe7bbb800244f1143321f41f7baf5c5bf2'
SCHEDULE = ['original', 'dynamic', 'tuned', 'torch', 'torch', 'tuned', 'dynamic', 'original']
STAGES = ('_all_logits', '_hidden1', '_hidden8', '_hidden16')
KNOWN = (0, 0, 505)  # Audited code_s256 hidden16 divergence only.
POLICY = {
 'reference': 'Pinned HF tensors are the primary numerical reference. All-position and serving logits use rtol=atol=3e-4; counts are exact.',
 'hidden': 'Unchanged rtol=atol=3e-4. Dynamic must pass HF at all stages. Other arms may retain exactly the previously audited code_s256 _hidden16 coordinate (0,0,505) as a disclosed HF failure; any other violation blocks timing.',
 'cross_arm': 'All direct logits and counts must pass versus original. A direct hidden failure is advisory only if every violating element is exactly the audited code_s256 _hidden16 coordinate and the dynamic hidden tensor passes HF. No tolerance is enlarged.',
 'scope': 'The static HF hidden exception and direct dynamic/static exception remain numerical qualifications, not a claim of full hidden equivalence.',
}


def load(path, name):
 spec = importlib.util.spec_from_file_location(name, path)
 mod = importlib.util.module_from_spec(spec)
 sys.modules[name] = mod
 spec.loader.exec_module(mod)
 return mod


def save(path, value):
 import numpy as np
 path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
 def native(x):
  if isinstance(x, np.generic): return x.item()
  if isinstance(x, np.ndarray): return x.tolist()
  raise TypeError(type(x).__name__)
 temp = path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=native)+'\n')
 temp.replace(path)


def modules(build):
 q_path = build/'42_final_full_model_gap.py'
 d_path = build/'experiment.py'
 import hashlib
 for path, digest in [(q_path,Q_SHA),(d_path,D_SHA)]:
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
   raise RuntimeError('Frozen companion missing/changed: '+str(path))
 return load(q_path,'gap42_46'),load(d_path,'dynamic44_46')


def violations(actual, reference, tol):
 import numpy as np
 delta=np.abs(actual.astype(np.float64)-reference.astype(np.float64))
 return np.argwhere(delta>tol+tol*np.abs(reference.astype(np.float64)))


def allowed_exception(arm,name,stage,actual,reference,kind,primary_dynamic_ok=False):
 import numpy as np
 if name!='code_s256' or stage!='_hidden16':return False
 if kind=='hf' and arm=='dynamic':return False
 if kind=='direct' and not primary_dynamic_ok:return False
 indices=violations(actual,reference,3e-4)
 return indices.shape==(1,3) and tuple(int(v) for v in indices[0])==KNOWN


def validate(q,arm,name,stage,actual,reference,kind,primary_dynamic_ok=False):
 metric=q.metric(actual,reference)
 metric['exception']=False
 if not metric['allclose']:
  metric['exception']=allowed_exception(arm,name,stage,actual,reference,kind,primary_dynamic_ok)
  metric['violating_indices']=violations(actual,reference,3e-4)[:8].tolist()
  if metric['exception']:
   metric['known_coordinate_actual']=float(actual[KNOWN]);metric['known_coordinate_reference']=float(reference[KNOWN])
 return metric


def checked(check,where,session):
 if not check['allclose'] and not check.get('exception'):
  save(session/'validation_failure.json',{'where':where,'check':check})
  raise AssertionError('Numerical gate failed: '+where)


def profile_once(q,request,session,name,arm,ref,expected_counts):
 import numpy as np
 trace=session/(name+'.trace.json')
 (y,c),p=q.capture(request,trace)
 if not q.metric(y,ref)['allclose'] or not np.array_equal(c,expected_counts):
  raise AssertionError('Profile output/counts failed: '+arm)
 events=json.loads(trace.read_text())['traceEvents']
 kernels=sorted([(float(e['ts']),float(e['ts'])+float(e['dur'])) for e in events if e.get('cat')=='kernel' and e.get('ph')=='X'])
 end=-1.; busy=0.
 for left,right in kernels:
  busy+=max(0.,right-max(left,end));end=max(end,right)
 p['gpu_kernel_union_ms']=busy/1000
 p['kernel_events']=len(kernels)
 p['cuda_runtime_launch_events']=sum('cudaLaunchKernel' in e.get('name','') for e in events)
 p['cuda_runtime_graph_events']=sum('cudaGraphLaunch' in e.get('name','') for e in events)
 p['profiled_wall_ms']=None  # Profiler wall includes its own overhead; benchmark samples are separate.
 p.pop('ranked_kernels',None)
 save(session/'profile.json',p)
 if arm=='torch' and p['cuda_graph_launch_events']==0:
  raise RuntimeError('Torch CUDA graph launch missing in trace')
 return p


def worker(run,build,out,session,arm,args,q,d):
 import numpy as np,torch,iree.runtime as rt
 m,cfg,original,_=q.frozen(run);m.stack()
 for key in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
  if os.environ.get(key) not in (None,'','0'):raise RuntimeError('TF32 override: '+key)
 root=Path(cfg.get('reference_root',str(run/'references')))
 manifest=json.loads((root/'manifest.json').read_text())
 start=time.perf_counter()
 if arm=='dynamic':
  runner=d.Dynamic(run,build/'build',q);module=None;device=runner.cfg.device
 elif arm=='torch':
  module,keep=q.torch_module(m,cfg,session);device=None;runner=None
 else:
  binary=original if arm=='original' else q.tuned_model(run)
  if q.sha(binary)!=(q.ORIGINAL_SHA if arm=='original' else q.TUNED_SHA):raise RuntimeError('Static VMFB changed')
  _,keep=m.iree_fn(binary,Path(cfg.get('weights_archive',str(run/'build/weights.irpa'))),entry='prefill_s256_c256')
  module=keep['ctx'].modules.olmoe;device=keep['cfg'].device;runner=None
 load_seconds=time.perf_counter()-start
 cases=[c for c in cfg['cases'] if c['length']==256]
 if len(cases)!=4:raise RuntimeError('Expected exactly four S256 prompts')
 checks={};samples=[];exceptions=[]
 for case in cases:
  name=case['name'];path=run/case['input']
  if q.sha(path)!=case['sha256']:raise RuntimeError('Frozen prompt changed: '+name)
  data=np.load(path)
  ids=torch.from_numpy(data).cuda() if arm=='torch' else rt.asdevicearray(device,data)
  ref=np.load(root/(name+'_logits.npy'));counts_ref=np.asarray(manifest[name]['counts'])
  def request():
   if arm=='dynamic':y,c,_,_=runner.run(ids)
   else:
    y,c=module.prefill_s256_c256(ids);y,c=y.to_host(),c.to_host()
   return np.asarray(y),np.asarray(c)
  if arm=='dynamic':
   y,c,diag,structure=runner.run(ids,True)
   # The diagnostic head uses the same normalized x as the serving head.
   diag_counts=c
  else:
   values=module.diagnostic_s256(ids)
   diag={stage:np.asarray(values[i].to_host()) for stage,i in zip(STAGES,(0,2,3,4))}
   diag_counts=np.asarray(values[1].to_host());del values
   y,c=request();structure={'host_orchestration':'single model invocation', 'diagnostic':'eager grouped Torch' if arm=='torch' else 'compiled IREE'}
  if not np.array_equal(diag_counts,counts_ref) or not np.array_equal(c,counts_ref):
   raise AssertionError('Diagnostic or serving expert counts changed: '+name)
  hf={};cross={};anchor=out/'baseline_outputs'/(name+'.npz')
  for stage,value in diag.items():
   reference=np.load(root/(name+stage+'.npy'))
   hf[stage]=validate(q,arm,name,stage,value,reference,'hf')
   checked(hf[stage],name+stage+' HF',session)
   if hf[stage]['exception']:exceptions.append({'case':name,'stage':stage,'kind':'HF','detail':hf[stage]})
  serving=validate(q,arm,name,'serving_logits',y,ref,'hf')
  checked(serving,name+' serving HF',session)
  direct={**diag,'serving_logits':y,'serving_counts':c,'diagnostic_counts':diag_counts}
  if arm=='original' and session.name=='session0_original':
   anchor.parent.mkdir(exist_ok=True)
   with open(anchor.with_suffix('.tmp'),'wb') as f:np.savez(f,**direct)
   anchor.with_suffix('.tmp').replace(anchor)
   save(anchor.with_suffix('.json'),{'sha256':q.sha(anchor),'keys':sorted(direct)})
   cross={'anchor_sha256':q.sha(anchor)}
  else:
   index=json.loads(anchor.with_suffix('.json').read_text())
   if q.sha(anchor)!=index['sha256'] or sorted(direct)!=index['keys']:
    raise RuntimeError('Original tensor anchor mismatch: '+name)
   with np.load(anchor) as baseline:
    for stage,value in direct.items():
     if stage.endswith('counts'):
      cross[stage]={'equal':bool(np.array_equal(value,baseline[stage]))}
      if not cross[stage]['equal']:raise AssertionError('Direct counts mismatch: '+name+' '+stage)
     else:
      cross[stage]=validate(q,arm,name,stage,value,baseline[stage],'direct',hf.get(stage,{}).get('allclose',False) if arm=='dynamic' else True)
      checked(cross[stage],name+stage+' direct original',session)
      if cross[stage]['exception']:exceptions.append({'case':name,'stage':stage,'kind':'direct_original','detail':cross[stage]})
  checks[name]={'hf':hf,'serving_pre':serving,'direct_original':cross,'structure':structure}
  save(session/'checks.json',checks);save(session/'exceptions.json',exceptions)
  warm=q.stable_warmup(request,args.warmups);save(session/(name+'_warmup.json'),warm)
  if warm['status']!='STABLE_WARMUP':raise RuntimeError('Warmup unstable: '+name)
  if arm=='torch':
   from torch._dynamo.utils import counters
   before=counters['stats']['unique_graphs']
  for index in range(args.repeats):
   start_ns=time.perf_counter_ns();y,c=request();elapsed=(time.perf_counter_ns()-start_ns)/1e6
   samples.append({'case':name,'arm':arm,'sample':index,'wall_ms':elapsed})
   save(session/'progress.json',samples)
  if arm=='torch':
   after=counters['stats']['unique_graphs']
   checks[name]['compiled_graphs_during_timing']=after-before
   if after!=before:raise RuntimeError('Torch compiled during timing: '+name)
  y,c=request()
  checks[name]['serving_post']=validate(q,arm,name,'serving_logits',y,ref,'hf')
  checked(checks[name]['serving_post'],name+' postflight HF',session)
  if not np.array_equal(c,counts_ref):raise AssertionError('Postflight expert counts mismatch')
  with np.load(anchor) as baseline:
   checks[name]['serving_post_direct']=validate(q,arm,name,'serving_logits',y,baseline['serving_logits'],'direct')
  checked(checks[name]['serving_post_direct'],name+' postflight direct',session)
  save(session/'checks.json',checks)
  print(arm,name,'median_ms',statistics.median(s['wall_ms'] for s in samples if s['case']==name),flush=True)
  if args.profile and name=='compiler_s256':
   profile_once(q,request,session,name,arm,ref,counts_ref)
 if arm=='torch':
  from torch._dynamo.utils import counters
  save(session/'dynamo_counters.json',{k:dict(v) for k,v in counters.items()})
 save(session/'status.json',{'arm':arm,'load_seconds':load_seconds,'exceptions':exceptions,'status':'PASS_WITH_KNOWN_NUMERICAL_QUALIFICATION' if exceptions else 'PASS'})
 save(session/'samples.json',samples)


def summarize(rows, statuses, out, repeats):
 summary={};review=[]
 for name in sorted({r['case'] for r in rows}):
  by_session={str(i):statistics.median(r['wall_ms'] for r in rows if r['case']==name and r['session']==i) for i in range(8)}
  med={arm:statistics.mean(by_session[str(i)] for i,x in enumerate(SCHEDULE) if x==arm) for arm in set(SCHEDULE)}
  drift={arm:100*(by_session[str(max(i for i,x in enumerate(SCHEDULE) if x==arm))]/by_session[str(min(i for i,x in enumerate(SCHEDULE) if x==arm))]-1) for arm in set(SCHEDULE)}
  within={}
  for i in range(8):
   v=[r['wall_ms'] for r in rows if r['case']==name and r['session']==i];half=len(v)//2
   within[str(i)]=100*(statistics.median(v[half:])/statistics.median(v[:half])-1)
  if any(abs(v)>3 for v in drift.values()):review.append(name+': process median drift exceeds 3%')
  if any(abs(v)>2 for v in within.values()):review.append(name+': within-process half drift exceeds 2%')
  summary[name]={'mean_of_two_process_medians_ms':med,'individual_process_medians_ms':by_session,
                 'between_process_drift_percent':drift,'within_process_half_drift_percent':within,
                 'original_over_tuned':med['original']/med['tuned'],'dynamic_over_tuned':med['dynamic']/med['tuned'],
                 'tuned_over_torch':med['tuned']/med['torch']}
 means={arm:statistics.mean(s['mean_of_two_process_medians_ms'][arm] for s in summary.values()) for arm in set(SCHEDULE)}
 report={'status':'COMPLETE_REVIEW_REQUIRED' if review else 'COMPLETE_WITH_KNOWN_NUMERICAL_QUALIFICATION',
         'sample_count':len(rows),'unit':'each process median over 20 completed-output requests; two process medians averaged per prompt; four prompt means averaged for overall latency',
         'cases':summary,'mean_of_prompt_medians_ms':means,'ratios':{'original_over_tuned':means['original']/means['tuned'],
         'dynamic_over_tuned':means['dynamic']/means['tuned'],'tuned_over_torch':means['tuned']/means['torch']},
         'review_reasons':review,'numerical_qualifications':[{'session':i,'arm':SCHEDULE[i],**x} for i,s in enumerate(statuses) for x in s['exceptions']]}
 save(out/'summary.json',report)
 return report


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--build',type=Path,default=BUILD)
 parser.add_argument('--python',type=Path,default=Path('/content/27_aot_venv/bin/python'))
 parser.add_argument('--out',type=Path);parser.add_argument('--resume',type=Path)
 parser.add_argument('--repeats',type=int,default=20);parser.add_argument('--warmups',type=int,default=10)
 parser.add_argument('--worker',choices=list(set(SCHEDULE)));parser.add_argument('--session',type=Path)
 parser.add_argument('--profile',action='store_true')
 args=parser.parse_args()
 if args.repeats<10 or not 10<=args.warmups<=60:parser.error('Require >=10 repeats and warmups in [10,60]')
 run=args.run.resolve();build=args.build.resolve();q,d=modules(build)
 if args.worker:
  worker(run,build,args.out.resolve(),args.session.resolve(),args.worker,args,q,d)
  return 0
 m,cfg,original,_=q.frozen(run);tuned=q.tuned_model(run)
 for name in ('shell.vmfb','segments.vmfb','dynamic_expert.vmfb','build.json'):
  if not (build/'build'/name).is_file():raise FileNotFoundError(build/'build'/name)
 out=(args.resume or args.out or run/'four_way_full_model_comparison'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
 out.mkdir(parents=True,exist_ok=bool(args.resume))
 request={'script_sha256':q.sha(__file__),'companion_42_sha256':Q_SHA,'companion_44_sha256':D_SHA,
          'schedule':SCHEDULE,'repeats':args.repeats,'warmups':args.warmups,'run':str(run),'build':str(build),
          'source_sha256':cfg['source_sha256'],'original_sha256':q.sha(original),'tuned_sha256':q.sha(tuned),
          'dynamic_build_hashes':json.loads((build/'build/build.json').read_text())['hashes'],
          'cases':[c for c in cfg['cases'] if c['length']==256],'policy':POLICY,
          'boundary':'Resident input to host-ready next-token logits and all-layer expert counts, batch 1 S256, no KV cache.',
          'precision':'A100 sm80, FP32 with TF32 disabled, same frozen weights/prompts.',
          'lowering':{'dynamic':'Untuned original dynamic expert VMFB; per-expert actual row count; padded routing/assembly; Python host loops and 16 count readbacks included.',
                     'original':'Original full static padded VMFB and default lowering.',
                     'tuned':'Full static padded VMFB, C256 gate/up rows4 and down rows4.',
                     'torch':'Grouped padded PyTorch, torch.compile Inductor fullgraph=True dynamic=False mode=reduce-overhead, C=S.'},
          'interpretation':'Four implementations, not an isolated causal estimate for dynamic shape support. Profiles are separate from timed requests.'}
 if args.resume:
  if json.loads((out/'request.json').read_text())!=request or q.sha(out/'experiment.py')!=q.sha(__file__):
   raise RuntimeError('Resume configuration/script mismatch')
 else:
  save(out/'request.json',request);shutil.copyfile(__file__,out/'experiment.py')
 if not args.python.is_file():raise FileNotFoundError('Pinned Python missing: '+str(args.python))
 report={'status':'STARTED','request':request};save(out/'report.json',report)
 rc=0
 try:
  rows=[];statuses=[]
  for i,arm in enumerate(SCHEDULE):
   session=out/f'session{i}_{arm}';session.mkdir(exist_ok=True)
   if not (session/'samples.json').exists():
    cmd=[str(args.python),str(out/'experiment.py'),'--run',str(run),'--build',str(build),'--out',str(out),'--worker',arm,
         '--session',str(session),'--repeats',str(args.repeats),'--warmups',str(args.warmups)]
    if i<4:cmd.append('--profile')
    if q.logged(cmd,out/(session.name+'.log'),3600):raise RuntimeError('Worker failed: '+session.name)
   data=json.loads((session/'samples.json').read_text());status=json.loads((session/'status.json').read_text())
   if status['arm']!=arm or len(data)!=4*args.repeats or any(v['arm']!=arm for v in data):
    raise RuntimeError('Incomplete or wrong worker: '+session.name)
   for case in request['cases']:
    name=case['name'];anchor=out/'baseline_outputs'/(name+'.npz')
    if q.sha(anchor)!=json.loads(anchor.with_suffix('.json').read_text())['sha256']:
     raise RuntimeError('Changed original output anchor: '+name)
   rows.extend({**v,'session':i} for v in data);statuses.append(status)
   save(out/'validated_samples.json',rows)
  report.update(summarize(rows,statuses,out,args.repeats))
 except BaseException:
  report.update(status='FAILED_OR_INCOMPLETE',error=traceback.format_exc());print(report['error'],flush=True);rc=1
 save(out/'report.json',report)
 bundle=out/'four_way_full_model_comparison_bundle.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.json','.log','.py') and path!=bundle:
    z.write(path,path.relative_to(out))
 print('Status:',report['status']);print('Output:',out);print('Bundle:',bundle)
 if 'mean_of_prompt_medians_ms' in report:print('Four-way latency (ms):',report['mean_of_prompt_medians_ms'])
 return rc


if __name__=='__main__':sys.exit(main())
