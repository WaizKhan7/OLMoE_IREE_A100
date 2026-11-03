#!/usr/bin/env python3
"""Full-model confirmation: current tuned VMFB vs C256 down rows2.

Only the C256 down root changes; validated gate/up rows2 remains in both arms. A/B/B/A isolated processes,
four S256 cases plus four unchanged S64 controls. Ten samples/case/session.
Reuses saved configured IR, weights and reference arrays. No model export,
checkpoint download, new tuning sweep or PyTorch benchmark. --setup restores
pinned packages; no prior notebook variables. Existing Drive artifacts required.
Direct cross-arm diagnostic/serving comparisons are blocking. Existing finite
reference hidden-state failures remain explicitly unresolved at original tolerance.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,importlib.util,json,math,os
import re,shutil,statistics,subprocess,sys,time,traceback,zipfile

RUN='/content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/20261028T140743_199932Z'
TOL=3e-4
BASELINE='full_model_rows2_ab/20261029T143306_641782Z'
BASELINE_SHA='95b2c3be7e47f1182c11a7e89ccc0e2490bffc54da6799106b7b824410eee4e9'


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
 if sha(source)!=cfg['source_sha256']:raise RuntimeError('Frozen source changed')
 spec=importlib.util.spec_from_file_location('frozen36',source)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 folder=run/BASELINE;previous=json.loads((folder/'build.json').read_text())
 if previous['source_sha256']!=cfg['source_sha256']:raise RuntimeError('Prior model source mismatch')
 binary=folder/'tuned.vmfb';configured=folder/'tuned_configured.mlir'
 if previous['tuned_vmfb_sha256']!=BASELINE_SHA or sha(binary)!=BASELINE_SHA:
  raise RuntimeError('Current tuned model differs from verified experiment-34 baseline')
 if sha(configured)!=previous['tuned_configured_sha256']:raise RuntimeError('Current configured IR changed')
 return m,cfg,binary,{'sha256':BASELINE_SHA,'source_sha256':previous['tuned_configured_sha256']}


def patch_experts(text):
 lines=text.splitlines(keepends=True);edits=[]
 starts=[i for i,s in enumerate(lines) if re.match(r'\s*func\.func\b',s)]+[len(lines)]
 for lo,hi in zip(starts,starts[1:]):
  matches=[]
  for i in range(lo,hi):
   line=lines[i]
   if 'iterator_types = ["parallel", "parallel", "parallel", "reduction"]' not in line:continue
   if re.search(r'ins\([^\n]*: tensor<64x256x1024xf32>, tensor<64x1024x2048xf32>\)',line):matches.append(i)
  if not matches:continue
  if len(matches)!=1:raise RuntimeError('Ambiguous C256 down function')
  i=matches[0];line=lines[i]
  for expected in ['reduction = [0, 0, 0, 32]','thread = [1, 1, 8, 0]',
                   'workgroup = [1, 32, 64, 1]','(d0, d1, d3)','(d0, d3, d2)','(d0, d1, d2)']:
   if expected not in line:raise RuntimeError('Unexpected current C256 down configuration: '+expected)
  if '#iree_gpu.pipeline<TileAndFuse>' not in lines[lo] or 'workgroup_size = [32, 8, 1]' not in lines[lo]:
   raise RuntimeError('Unexpected C256 down pipeline/launch')
  replacement='#iree_gpu.lowering_config<{reduction = [0, 0, 0, 32], thread = [1, 2, 8, 0], workgroup = [1, 64, 64, 1]}>'
  modified,n=re.subn(r'#iree_gpu.lowering_config<\{[^}]+\}>',replacement,line)
  if n!=1:raise RuntimeError('Expected one root attribute')
  edits.append({'line':i+1,'before':line.rstrip(),'after':modified.rstrip(),'capacity':256,'operation':'down'})
  lines[i]=modified
 if len(edits)!=1:raise RuntimeError('Expected exactly one deduplicated C256 down root; found '+str(len(edits)))
 restored=lines.copy();original=text.splitlines(keepends=True)
 for e in edits:restored[e['line']-1]=original[e['line']-1]
 if ''.join(restored)!=text:raise RuntimeError('Unexpected nonconfiguration change')
 return ''.join(lines),{'edits':edits,'coverage':{'c256_down':1},
     'scope':'Exactly one C256 down lowering_config; every other byte preserved.'}


def build(run,out):
 import iree.compiler as ic
 m,cfg,current,meta=frozen(run)
 if importlib.metadata.version('iree-base-compiler')!='3.12.0':raise RuntimeError('Pinned compiler required')
 source=run/BASELINE/'tuned_configured.mlir'
 flags=[x for x in cfg['iree_flags'] if not x.startswith('--iree-hal-dump-')]
 if '--iree-opt-numeric-precision-reduction=false' not in flags:raise RuntimeError('Strict precision flag missing')
 started=time.perf_counter();text=source.read_text()
 tuned,audit=patch_experts(text);(out/'tuned_configured.mlir').write_text(tuned)
 save(out/'configuration_diff.json',audit)
 print('Compiling saved configured graph with only C256 down rows2 changed',flush=True)
 binary=ic.compile_str(tuned,target_backends=['cuda'],extra_args=[*flags,'--compile-from=executable-configurations'])
 candidate=out/'tuned.vmfb';candidate.write_bytes(binary)
 save(out/'build.json',{'status':'PASS','seconds':time.perf_counter()-started,
   'source_run':str(run),'source_sha256':cfg['source_sha256'],'original_vmfb_sha256':meta['sha256'],
   'saved_ir_sha256':sha(source),'saved_ir_path':str(source),'tuned_vmfb_sha256':sha(candidate),'flags':flags,
   'tuned_configured_sha256':sha(out/'tuned_configured.mlir'),'coverage':audit['coverage']})
 print('Candidate VMFB compiled; current baseline preserved',flush=True)


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
     'ranked_kernels':sorted([{'name':name,**v} for name,v in grouped.items()],key=lambda x:-x['total_ms'])}


def measure(run,out,session,a):
 import numpy as np,iree.runtime as rt
 m,cfg,original,meta=frozen(run);m.stack()
 for key in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']:
  if os.environ.get(key) not in (None,'','0'):raise RuntimeError('Unexpected TF32 override: '+key)
 build_meta=json.loads((out/'build.json').read_text())
 if build_meta['original_vmfb_sha256']!=meta['sha256']:raise RuntimeError('Build/source mismatch')
 candidate=out/'tuned.vmfb'
 if sha(candidate)!=build_meta['tuned_vmfb_sha256']:raise RuntimeError('Candidate changed')
 binary=original if a.arm=='A' else candidate
 weights=Path(cfg.get('weights_archive',str(run/'build/weights.irpa')))
 print('Loading full-model VMFB and existing parameter archive; startup excluded',flush=True)
 start=time.perf_counter();fn,keep=m.iree_fn(binary,weights,entry='prefill_s64_c64')
 module=keep['ctx'].modules.olmoe;device=keep['cfg'].device
 load_seconds=time.perf_counter()-start
 print('Loaded',a.arm,'seconds',round(load_seconds,2),flush=True)
 reference_root=Path(cfg.get('reference_root',str(run/'references')))
 manifest=json.loads((reference_root/'manifest.json').read_text())
 cases=[case for case in cfg['cases'] if case['length'] in (64,256)]
 if len(cases)!=8:raise RuntimeError('Expected four frozen prompts at each of two lengths')
 checks={};samples=[];profiles={}
 is_anchor=session.name=='session0_A'
 for case in cases:
  name=case['name'];s=case['length'];entry=getattr(module,f'prefill_s{s}_c{s}')
  ids=rt.asdevicearray(device,np.load(run/case['input']))
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
  for _ in range(a.warmups):request()
  for sample in range(a.repeats):
   start=time.perf_counter_ns();value,counts=request()
   samples.append({'arm':a.arm,'case':name,'sample':sample,'wall_ms':(time.perf_counter_ns()-start)/1e6})
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
  save(session/'checks.json',checks);save(session/'progress.json',samples)
  vals=[v['wall_ms'] for v in samples if v['case']==name]
  print(a.arm,name,'median_ms',round(statistics.median(vals),3),'logits/counts PASS',flush=True)
 hidden_fail=[{'case':name,'stage':stage,**v} for name,c in checks.items()
             for stage,v in c['diagnostic'].items() if stage!='_all_logits' and not v['allclose']]
 save(session/'status.json',{'status':'FUNCTIONAL_OUTPUTS_PASS_INTERNAL_REVIEW' if hidden_fail else 'PASS',
       'arm':a.arm,'load_seconds':load_seconds,'vmfb_sha256':sha(binary),
       'hidden_original_gate_failures':hidden_fail})
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
 p.add_argument('--setup',action='store_true');p.add_argument('--resume',help='Existing A/B output directory')
 p.add_argument('--repeats',type=int,default=10);p.add_argument('--warmups',type=int,default=2)
 p.add_argument('--worker',choices=['build','measure']);p.add_argument('--out');p.add_argument('--session')
 p.add_argument('--arm',choices=['A','B']);p.add_argument('--profile',action='store_true')
 a=p.parse_args()
 if min(a.repeats,a.warmups)<1:p.error('Positive sample counts required')
 run=Path(a.run).resolve()
 if a.worker=='build':build(run,Path(a.out));return 0
 if a.worker=='measure':measure(run,Path(a.out),Path(a.session),a);return 0
 m,cfg,original,meta=frozen(run)
 if a.resume:out=Path(a.resume).resolve()
 else:
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
  out=run/'full_model_down_rows2_ab'/stamp;out.mkdir(parents=True,exist_ok=False)
 python=Path(a.python)
 if a.setup:
  python=Path(m.environment_python(python.parent.parent,out,1200))
  probe=subprocess.run([str(python),'-c',"import importlib.metadata as m,sys;sys.exit(any(m.version(s.split('==')[0]).split('+')[0]!=s.split('==')[1] for s in sys.argv[1:]))",*m.PINS],capture_output=True)
  if probe.returncode:subprocess.run([str(python),'-m','pip','install',*m.PINS],check=True)
 if not python.is_file():raise FileNotFoundError('Missing venv; use --setup')
 script=out/'experiment.py'
 if a.resume:
  if sha(script)!=sha(__file__):raise RuntimeError('Resume script differs; preserve output and start a new run')
 else:shutil.copyfile(__file__,script)
 request={'source_run':str(run),'repeats':a.repeats,'warmups':a.warmups,'schedule':['A','B','B','A'],
     'lengths':[64,256],'capacities':[64,256],'current_vmfb_sha256':BASELINE_SHA,
     'change':'C256 down thread [1,2,8,0], workgroup tile [1,64,64,1]' }
 if a.resume:
  if json.loads((out/'request.json').read_text())!=request:raise RuntimeError('Resume measurement settings differ')
 else:save(out/'request.json',request)
 common=[str(python),str(script),'--run',str(run),'--out',str(out)]
 report={'status':'STARTED','source_run':str(run),'current_vmfb_sha256':meta['sha256'],
   'script_sha256':sha(script),'sampling':'Four isolated sessions A/B/B/A; descriptive medians, no paired CI',
   'scope':'Only C256 down rows2 versus current tuned model; S64 unchanged control.',
   'validation':'Strict reference logits/counts plus direct cross-arm diagnostic/serving gates; reference hidden failures recorded unchanged.',
   'prediction':{'s256_candidate_ms':673,'speedup_over_current':1.16,
    'basis':'Experiment-34 latency ~780 ms minus ~245 ms down plus 245/1.773; hypothesis only.'}}
 save(out/'report.json',report)
 code=0
 try:
  if not (out/'build.json').exists():
   rc=logged([*common,'--worker','build'],out/'build.log',1200)
   if rc:raise RuntimeError('Candidate build failed; inspect build.log')
  build_meta=json.loads((out/'build.json').read_text())
  if sha(out/'tuned_configured.mlir')!=build_meta['tuned_configured_sha256']:raise RuntimeError('Candidate IR changed')
  if build_meta['source_sha256']!=cfg['source_sha256'] or build_meta['original_vmfb_sha256']!=meta['sha256']:
   raise RuntimeError('Build metadata does not match source run')
  if sha(out/'tuned.vmfb')!=build_meta['tuned_vmfb_sha256']:raise RuntimeError('Candidate binary changed')
  all_rows=[];statuses=[]
  for number,arm in enumerate(['A','B','B','A']):
   session=out/f'session{number}_{arm}';session.mkdir(exist_ok=True)
   if (session/'samples.json').exists():
    status=json.loads((session/'status.json').read_text())
    expected=meta['sha256'] if arm=='A' else build_meta['tuned_vmfb_sha256']
    if status['vmfb_sha256']!=expected:raise RuntimeError('Session artifact mismatch')
    for case in cfg['cases']:
     if case['length'] not in (64,256):continue
     name=case['name'];snapshot=out/'baseline_outputs'/(name+'.npz')
     if sha(snapshot)!=json.loads(snapshot.with_suffix('.json').read_text())['sha256']:raise RuntimeError('A0 snapshot changed')
   else:
    cmd=[*common,'--worker','measure','--session',str(session),'--arm',arm,
         '--repeats',str(a.repeats),'--warmups',str(a.warmups)]
    if number<2:cmd.append('--profile')
    rc=logged(cmd,out/f'session{number}_{arm}.log',1200)
    if rc:raise RuntimeError(f'Session {number}/{arm} failed; partial progress preserved')
   rows=json.loads((session/'samples.json').read_text())
   for row in rows:row['session']=number
   all_rows.extend(rows);statuses.append(json.loads((session/'status.json').read_text()))
   save(out/'validated_samples.json',all_rows)
  summary={}
  for name in sorted(set(row['case'] for row in all_rows)):
   med={arm:statistics.median(row['wall_ms'] for row in all_rows if row['case']==name and row['arm']==arm) for arm in ['A','B']}
   session_medians={str(i):statistics.median(row['wall_ms'] for row in all_rows if row['case']==name and row['session']==i) for i in range(4)}
   summary[name]={'median_ms':med,'speedup_A_over_B':med['A']/med['B'],
      'latency_reduction_percent':100*(1-med['B']/med['A']),'session_medians_ms':session_medians,
      'A_bracket_drift_percent':100*(session_medians['3']/session_medians['0']-1),
      'B_repeat_drift_percent':100*(session_medians['2']/session_medians['1']-1)}
  hidden=[{'session':i,**item} for i,status in enumerate(statuses) for item in status['hidden_original_gate_failures']]
  drift=any(abs(v[key])>10 for v in summary.values() for key in ['A_bracket_drift_percent','B_repeat_drift_percent'])
  review_reasons=(['reference_hidden_gate_failures'] if hidden else [])+(['session_drift_over_10_percent'] if drift else [])
  report.update(review_reasons=review_reasons,status='PROVISIONAL_NUMERICAL_OR_DRIFT_REVIEW_REQUIRED' if hidden or drift else 'COMPLETE_REVIEW_REQUIRED',
      summary=summary,hidden_original_gate_failures=hidden,drift_over_10_percent=drift,
      candidate_vmfb_sha256=build_meta['tuned_vmfb_sha256'])
  save(out/'summary.json',summary)
 except BaseException:
  report.update(status='FAILED_OR_INCOMPLETE',error=traceback.format_exc());print(report['error'],flush=True);code=1
 save(out/'report.json',report)
 with zipfile.ZipFile(out/'full_model_down_rows2_ab_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.json','.log','.py'):z.write(path,str(path.relative_to(out)))
 for name,v in report.get('summary',{}).items():print(name,v['median_ms'],'speedup',round(v['speedup_A_over_B'],3))
 print('Review reasons:',report.get('review_reasons',[]))
 print('Reference hidden failures:',len(report.get('hidden_original_gate_failures',[])))
 print('Status:',report['status']);print('Output:',out);print('Bundle:',out/'full_model_down_rows2_ab_bundle.zip')
 return code


if __name__=='__main__':sys.exit(main())
