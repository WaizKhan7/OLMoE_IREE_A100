"""Diagnostic bridge; run with restored Colab globals via runpy, never as 25C2.
Reuses historical VMFBs; new prepared-only graphs are saved in a fresh directory.
No assembly rewrite or full-path optimization is performed here.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, importlib.metadata, json, os, random, runpy, subprocess, time, traceback, zipfile
import numpy as np
import torch
from iree import compiler as ireec, runtime as ireert
from torch.profiler import profile, ProfilerActivity, record_function
ROOT=Path('/content/drive/MyDrive/moe_iree_olmoe')
HIDDEN=2048
INTERMEDIATE=1024
BLOCKS=10
CAP=900
EXPECTED_CANDIDATE='31bf04c7d655288d5088ac6f58ebdce57fdcf496d66b8cc29523034821882927'
EXPECTED_NATIVE='0db1f8fe8b971ab4a06353a995d900dc241738f1662e73938e8a3a4053f56afa'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def assert_close(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError('Nonfinite valid output')
    np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-4)
    return float(np.max(np.abs(a-b)))


def balanced(arms):
    if BLOCKS%len(arms):raise ValueError('Need exact position balance')
    r=random.Random(20261028);result=[]
    for _ in range(BLOCKS//len(arms)):
        base=list(arms);r.shuffle(base)
        cycle=[base[k:]+base[:k] for k in range(len(base))];r.shuffle(cycle);result.extend(cycle)
    return result

def static_source(module_name: str, expert_count: int, rows: int) -> str:
    """Exact no-bias OLMoE expert math with static [E,M,H] tensors."""
    shape_x = f"tensor<{expert_count}x{rows}x{HIDDEN}xf32>"
    shape_gu = f"tensor<{expert_count}x{HIDDEN}x{2 * INTERMEDIATE}xf32>"
    shape_proj = f"tensor<{expert_count}x{rows}x{2 * INTERMEDIATE}xf32>"
    shape_half = f"tensor<{expert_count}x{rows}x{INTERMEDIATE}xf32>"
    shape_down = f"tensor<{expert_count}x{INTERMEDIATE}x{HIDDEN}xf32>"
    return f'''module @{module_name} {{
  func.func @forward(%x: {shape_x}, %gate_up: {shape_gu}, %down: {shape_down}) -> {shape_x} {{
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %projected_empty = tensor.empty() : {shape_proj}
    %projected_init = linalg.fill ins(%zero : f32) outs(%projected_empty : {shape_proj}) -> {shape_proj}
    %projected = linalg.batch_matmul ins(%x, %gate_up : {shape_x}, {shape_gu}) outs(%projected_init : {shape_proj}) -> {shape_proj}
    %gate = tensor.extract_slice %projected[0, 0, 0] [{expert_count}, {rows}, {INTERMEDIATE}] [1, 1, 1] : {shape_proj} to {shape_half}
    %up = tensor.extract_slice %projected[0, 0, {INTERMEDIATE}] [{expert_count}, {rows}, {INTERMEDIATE}] [1, 1, 1] : {shape_proj} to {shape_half}
    %activated_empty = tensor.empty() : {shape_half}
    %activated = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%gate, %up : {shape_half}, {shape_half}) outs(%activated_empty : {shape_half}) {{
      ^bb0(%g: f32, %u: f32, %out: f32):
        %neg = arith.negf %g : f32
        %exp = math.exp %neg : f32
        %denom = arith.addf %one, %exp : f32
        %silu = arith.divf %g, %denom : f32
        %value = arith.mulf %silu, %u : f32
        linalg.yield %value : f32
    }} -> {shape_half}
    %output_empty = tensor.empty() : {shape_x}
    %output_init = linalg.fill ins(%zero : f32) outs(%output_empty : {shape_x}) -> {shape_x}
    %output = linalg.batch_matmul ins(%activated, %down : {shape_half}, {shape_down}) outs(%output_init : {shape_x}) -> {shape_x}
    return %output : {shape_x}
  }}
}}\n'''


def main(env,out,report):
    import iree._runtime_libs._runtime as native
    candidate=ROOT/'25C2_runner_validation.py'
    if sha(candidate)!=EXPECTED_CANDIDATE or sha(native.__file__)!=EXPECTED_NATIVE:raise RuntimeError('Candidate/runtime mismatch')
    if torch.__version__!='2.11.0+cu130' or torch.cuda.get_device_name(0)!='NVIDIA A100-SXM4-40GB':raise RuntimeError('Pinned A100/PyTorch required')
    if os.environ.get('NVIDIA_TF32_OVERRIDE') not in (None,''):raise RuntimeError('Unexpected TF32 override')
    runner=runpy.run_path(str(candidate),run_name='bridge_definitions')['Runner'](env)
    ref=json.loads((ROOT/'25C2_runner_validation_runs/20261027T145528_778878Z/report.json').read_text())
    if runner.hashes!=ref['artifact_sha256'] or runner.weight_hashes!=ref['weights_sha256']:raise RuntimeError('Changed modules/weights')
    if {k:v['fingerprint'] for k,v in runner.fixtures.items()}!=ref['trace_fingerprints']:raise RuntimeError('Changed trace')
    historical=ROOT/'25B_a100_or_blackwell_static_group_ladder_final'
    old_report=historical/'25B_a100_or_blackwell_static_group_ladder_final_report.json'
    old=json.loads(old_report.read_text())
    audit=historical/'25B_a100_or_blackwell_static_group_ladder_final_final_audit.json'
    if json.loads(audit.read_text())['status']!='PASS_25B_COMPLETED_RESULTS_VALID':raise RuntimeError('Historical audit not passing')
    for key,p in [('trace',ROOT/'olmoe_layer7_prefill64_decode1.pt'),('weights',ROOT/'olmoe_layer7_weights.safetensors')]:
        if sha(p)!=old['hashes'][key]:raise RuntimeError('Historical '+key+' changed')
    # Verify the actual historical primary fixture before substituting notebook arrays.
    old_fixture=torch.load(ROOT/'olmoe_layer7_prefill64_decode1.pt',map_location='cpu',weights_only=False)['prefill64']
    np.testing.assert_array_equal(env['primary']['x'],old_fixture['moe_input'][0].numpy())
    np.testing.assert_array_equal(env['primary']['selected'],old_fixture['expert_ids'].numpy())
    del old_fixture
    report.update(candidate_sha256=sha(candidate),native_sha256=sha(native.__file__),artifact_sha256=runner.hashes,
        weights_sha256=runner.weight_hashes,trace_fingerprints=ref['trace_fingerprints'],
        historical_report_sha256=sha(old_report),historical_audit_sha256=sha(audit),
        compiler_version=importlib.metadata.version('iree-base-compiler'),modules={},checks={},
        precision={'matmul_precision':torch.get_float32_matmul_precision(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32},
        boundary={'historical':'50 core experts; same M47 sentinel invocation and sentinel host readback included in both arms',
                  'matched':'prepared expert-only call, ends at successful IREE return; readback/check outside timer',
                  'full':'unchanged routed path, ends at successful IREE return; readback/check outside timer'},
        exclusions=['All prepared input/selected-weight gathering is outside timing.',
                    'Matched cores return padded grouped tensors; no scatter, combine, or route.',
                    'Historical repeat loads the original three VMFBs, but not all modules/allocations of the original ladder.',
                    'This is diagnostic, not a frozen benchmark or isolated additive cost ladder.',
                    'No excess-over-core dominance classification: new core lowering may differ.'])
    ctx=ireert.SystemContext(config=runner.cfg)
    def load(label,path,expected=None):
        if expected is not None and sha(path)!=expected:raise RuntimeError('VMFB hash mismatch: '+label)
        module=ireert.VmModule.copy_buffer(ctx.instance,path.read_bytes());ctx.add_vm_module(module)
        # Preserve full disassembly for later structural completion/codegen review.
        proc=subprocess.run(['iree-dump-module','--output=disassembly',str(path)],capture_output=True,text=True,timeout=60,check=True)
        (out/(label+'.vm.txt')).write_text(proc.stdout)
        report['modules'][label]={'path':str(path),'sha256':sha(path),'module_name':module.name}
        return getattr(ctx.modules,module.name)['forward']
    hist={}
    for label in ('dynamic','padded_e50','sentinel'):
        m=old['compile']['modules'][label];hist[label]=load('historical_'+label,Path(m['path']),m['sha256'])
    matched={}
    for e in (51,55,64):
        name=f'bridge_prepared_e{e}';src=static_source(name,e,47);source=out/(name+'.mlir');source.write_text(src)
        vmfb=out/(name+'.vmfb')
        print('Compiling prepared E='+str(e),flush=True)
        vmfb.write_bytes(ireec.compile_str(src,target_backends=['cuda'],extra_args=['--iree-cuda-target=sm_80']))
        matched[e]=load(name,vmfb);report['modules'][name]['source_sha256']=sha(source)
    report['gpu_before']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=15).stdout
    save(out/'report.json',report)
    deadline=time.monotonic()+CAP
    def cap():
        if time.monotonic()>=deadline:raise TimeoutError('900-second diagnostic cap reached between calls; partial data retained')
    def device(a):return ireert.asdevicearray(runner.cfg.device,np.ascontiguousarray(a,dtype=np.float32))
    rows=[];report['rows']=rows
    with torch.inference_mode(),(out/'raw_rows.jsonl').open('w') as log:
        for trace in ('primary','second'):
            fixture=env[trace];ids=np.asarray(fixture['selected']);x=np.asarray(fixture['x'])
            active=[int(e) for e in runner.fixtures[trace]['active']]
            token_rows={e:np.ascontiguousarray(x[np.where(ids==e)[0]],dtype=np.float32) for e in active}
            references={}
            for e in active:
                tx=torch.as_tensor(token_rows[e],device='cuda');gu=tx@runner.tgw[e];g,u=gu.chunk(2,dim=-1)
                references[e]=(torch.nn.functional.silu(g)*u)@runner.tdw[e]
            references={e:v.cpu().numpy() for e,v in references.items()}
            gw=runner.tgw.cpu().numpy();dw=runner.tdw.cpu().numpy()
            def prepared(experts,padding=0.0):
                px=np.full((len(experts),47,HIDDEN),padding,dtype=np.float32)
                for j,e in enumerate(experts):
                    n=len(token_rows.get(e,[]))
                    if n:px[j,:n]=token_rows[e]
                return (device(px),device(gw[experts]),device(dw[experts]))
            args_active=prepared(active);args_stationary=prepared(list(range(64)))
            def check_group(value,experts):
                host=np.asarray(value.to_host());err=0.
                for j,e in enumerate(experts):
                    if e in references:err=max(err,assert_close(host[j,:len(token_rows[e])],references[e]))
                return err
            # Poison only unused padded input rows: verify every valid expert result.
            for label,fn,experts,args in [('active',matched[len(active)],active,args_active),('stationary',matched[64],list(range(64)),args_stationary)]:
                cap();zero=fn(*args);check_group(zero,experts)
                host_zero=np.asarray(zero.to_host())
                if not np.isfinite(host_zero).all():raise AssertionError('Nonfinite zero-padded core')
                for j,e in enumerate(experts):
                    n=len(token_rows.get(e,[]))
                    np.testing.assert_allclose(host_zero[j,n:],0.0,rtol=0,atol=3e-4)
                del host_zero,zero
                for poison in (np.nan,np.inf):
                    cap();pa=prepared(experts,poison);v=fn(*pa);check_group(v,experts);del v,pa
                report['checks'][trace+'_'+label]='zero/NaN/Inf valid-row checks PASS'
            calls={'prepared_active':lambda:matched[len(active)](*args_active),
                   'prepared_stationary':lambda:matched[64](*args_stationary),
                   'full_dynamic':lambda:runner.invoke('dynamic',trace),
                   'full_active':lambda:runner.invoke('active',trace),
                   'full_stationary':lambda:runner.invoke('iree_stationary',trace)}
            def check(name,value):
                if name=='prepared_active':return check_group(value,active)
                if name=='prepared_stationary':return check_group(value,list(range(64)))
                return assert_close(runner.host_ready(value),runner.fixtures[trace]['reference'])
            # Historical repeat has its own boundary and separate schedule.
            if trace=='primary':
                sentinel=int(np.argmax(runner.fixtures[trace]['counts']));core=[e for e in active if e!=sentinel]
                if len(core)!=50 or sum(len(token_rows[e]) for e in core)!=465:raise RuntimeError('Historical core mismatch')
                if core!=old['artifacts']['trace']['core_expert_ids_in_execution_order']:raise RuntimeError('Historical expert order mismatch')
                core_args=prepared(core);sent_args=prepared([sentinel])
                dyn_args=[(device(token_rows[e]),*runner.pairs[e]) for e in core]
                for e,a in zip(core,dyn_args):assert_close(hist['dynamic'](*a).to_host(),references[e])
                check_group(hist['padded_e50'](*core_args),core);check_group(hist['sentinel'](*sent_args),[sentinel])
                def historical_call(grouped):
                    if grouped:hist['padded_e50'](*core_args)
                    else:
                        for a in dyn_args:hist['dynamic'](*a)
                    return np.asarray(hist['sentinel'](*sent_args).to_host())
                history_calls={'historical_dynamic':lambda:historical_call(False),'historical_e50':lambda:historical_call(True)}
            else:history_calls={}
            for family,these in [('matched',calls),('historical',history_calls)]:
                if not these:continue
                # Completion diagnostic: profiled kernels must finish before the chosen endpoint.
                for name,call in these.items():
                    cap()
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                        with record_function('bridge.endpoint'):v=call()
                        if family=='matched':check(name,v)
                        else:assert_close(v[0],references[sentinel])
                    path=out/(trace+'_'+name+'_completion.json');prof.export_chrome_trace(str(path))
                    ev=json.loads(path.read_text())['traceEvents'];scope=[e for e in ev if e.get('cat')=='user_annotation' and e.get('name')=='bridge.endpoint' and e.get('ph')=='X']
                    if len(scope)!=1:raise RuntimeError('Missing completion annotation')
                    end=scope[0]['ts']+scope[0]['dur'];kernels=[e for e in ev if e.get('cat')=='kernel' and e.get('ph')=='X']
                    if not kernels or any(e['ts']+e['dur']>end for e in kernels):raise RuntimeError('Unverified completion endpoint: '+name)
                    report['checks'][trace+'_'+name+'_completion']='captured kernels finish before endpoint; empirical diagnostic only'
                    for _ in range(2):
                        cap();v=call()
                        if family=='matched':check(name,v)
                        else:assert_close(v[0],references[sentinel])
                schedule=balanced(tuple(these));report.setdefault('schedules',{})[trace+'_'+family]=schedule;save(out/'report.json',report)
                for block,order in enumerate(schedule):
                    for position,name in enumerate(order):
                        cap();was=gc.isenabled();gc.disable()
                        try:start=time.perf_counter_ns();v=these[name]();stop=time.perf_counter_ns()
                        finally:
                            if was:gc.enable()
                        err=check(name,v) if family=='matched' else assert_close(v[0],references[sentinel])
                        row={'trace':trace,'family':family,'block':block,'position':position,'arm':name,'ms':(stop-start)/1e6,'max_abs_error':err}
                        rows.append(row);log.write(json.dumps(row)+'\n');log.flush();del v
                    if (block+1)%5==0:print(trace,family,'block',block+1,'PASS',flush=True)
            del args_active,args_stationary,references,gw,dw,calls,history_calls
    report['medians_ms']={}
    for row in rows:
        key=row['trace']+'/'+row['arm'];report['medians_ms'][key]=float(np.median([r['ms'] for r in rows if r['trace']+'/'+r['arm']==key]))
    if len(rows)!=120:raise RuntimeError('Incomplete schedule')
    report['gpu_after']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=15).stdout
    report['status']='BRIDGE_COMPLETE_REVIEW_REQUIRED'
    return report


if __name__=='__main__':
    out=ROOT/'25C3_prepared_core_bridge'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
    report={'status':'STARTED','scope':'25C3 diagnostic; separate from frozen 25C2','script_sha256':sha(__file__),
            'plan':{'blocks':BLOCKS,'warmups':2,'cap_seconds':CAP,'cap_excludes_compilation_and_initial_setup':True}}
    try:report=main(globals(),out,report)
    except BaseException:report.update(status='FAILED_OR_INCOMPLETE_REVIEW_REQUIRED',error=traceback.format_exc());print(report['error'],flush=True)
    save(out/'report.json',report)
    with zipfile.ZipFile(out/'prepared_core_bridge_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.mlir'):z.write(p,p.name)
    print('Status:',report['status']);print('Report:',out/'report.json');print('Bundle:',out/'prepared_core_bridge_bundle.zip')
    if report['status']!='BRIDGE_COMPLETE_REVIEW_REQUIRED':raise RuntimeError('Bridge incomplete; preserve output '+str(out))
