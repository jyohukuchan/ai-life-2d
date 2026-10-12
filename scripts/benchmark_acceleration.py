"""Matched pre-birth throughput and fixed-input numerical checks; no saved trial is resumed."""
import argparse,dataclasses,gc,importlib.util,json,statistics,time
from pathlib import Path
import torch
from ailife.config import Config
from ailife.learner import Learners,linear,private_adam,prediction_loss
from ailife.observer import atomic_json
from ailife.runner import iteration,sync
from ailife.world import World


def reference_class(root):
 spec=importlib.util.spec_from_file_location('ailife.frozen_reference',root/'reference_learner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.Learners


def numerical(cfg,cls,initial,device):
 refcfg=dataclasses.replace(cfg,precision='fp32',execution='eager')
 w=World(refcfg,device);w.load_state_dict(initial['world'])
 ref=Learners(refcfg,device);ref.load_state_dict(initial['learners'])
 cand=cls(cfg,device);cand.load_state_dict(initial['learners'])
 raw=w.observe();action=ref.act(raw,w.ids,w.clock,w.alive)
 cand.act(raw,w.ids,w.clock,w.alive)
 t=w.step(action);target=w.observe(t.positions,t.directions,t.energy).reshape(ref.count,-1)[:,:cfg.sensory_size]
 values=[];grads=[];deltas=[];losses=[]
 for l in [ref,cand]:
  _,h,pre,z,_,_=l.cache;onehot=l.onehot(action.reshape(-1,4))
  dtype=getattr(l,'compute_dtype',torch.float32)
  loss=prediction_loss(z,onehot,l.params,target,(2*cfg.radius+1)**2*3,cfg.symbols,cfg.interoception,dtype)
  params={n:l.params[n] for n in l.pred_names};scale=cfg.fp16_loss_scale if dtype==torch.float16 else 1.
  gs=torch.autograd.grad((loss*t.was_alive.flatten()).sum()*scale,tuple(params.values()))
  gs=tuple(g.float()/scale for g in gs)
  mask=t.was_alive.flatten();flat=torch.cat([g[mask].flatten() for g in gs])
  before={n:p.detach().clone() for n,p in params.items()}
  with torch.no_grad():private_adam(params,gs,l.moments,l.adam_count,l.log_rates[:,0].exp(),mask)
  delta=torch.cat([(params[n]-before[n])[mask].flatten() for n in params])
  grads.append(flat);deltas.append(delta);values.append(z.detach()[mask].flatten());losses.append(float(loss[mask].mean()))
 def relative(x,y):return float((y-x).norm()/x.norm().clamp_min(1e-12))
 cosine=float(torch.nn.functional.cosine_similarity(grads[0][None],grads[1][None]))
 result={'loss_reference':losses[0],'loss_candidate':losses[1],
  'loss_relative_difference':abs(losses[1]-losses[0])/max(losses[0],1e-12),
  'features_relative_l2':relative(*values),'gradient_relative_l2':relative(*grads),'gradient_cosine':cosine,
  'update_relative_l2':relative(*deltas),'candidate_update_rms':float(deltas[1].square().mean().sqrt()),
  'candidate_finite':cand.finite(),'master_dtype':str(cand.params['enc_w'].dtype),'body_dtype':str(w.energy.dtype)}
 result['pass']=bool(result['candidate_finite'] and cosine>=.98 and result['gradient_relative_l2']<=.2 and result['loss_relative_difference']<=.05 and result['candidate_update_rms']>0)
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',default='runs/acceleration-20261011');p.add_argument('--output',required=True)
 p.add_argument('--variants',default='reference,fp32-eager,fp32-kernels,fp32-compiled,bf16-eager,bf16-compiled,fp16-eager,fp16-compiled')
 p.add_argument('--steps',type=int,default=160);p.add_argument('--warmup',type=int,default=32);p.add_argument('--repeats',type=int,default=5);p.add_argument('--profile',action='store_true')
 args=p.parse_args();root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);device='cuda:0';torch.backends.cuda.matmul.allow_tf32=False
 initial=torch.load('runs/compass-ablation-initial/absolute.pt',map_location='cpu',weights_only=True)
 base=dataclasses.replace(Config.load('configs/experiments/2026-10-11-compass/compass-absolute.yaml'),log_reward_components=False,elite_count=0,replay_count=0,fast_forward_extinct=False)
 if args.steps+args.warmup>=base.maturity:raise ValueError('throughput window must precede births')
 refcls=reference_class(root)
 report={'device':torch.cuda.get_device_name(),'arch':torch.cuda.get_device_properties(0).gcnArchName,'torch':torch.__version__,'hip':torch.version.hip,'steps':args.steps,'warmup':args.warmup,'repeats':args.repeats,'config':base.as_dict(),'variants':{}}
 for name in args.variants.split(','):
  precision,execution=('fp32','eager') if name=='reference' else name.split('-')
  cfg=dataclasses.replace(base,precision=precision,execution=execution)
  cls=refcls if name=='reference' else Learners
  print(json.dumps({'phase':'start','variant':name}),flush=True)
  try:
   torch.cuda.reset_peak_memory_stats();w,l=World(cfg,device),cls(cfg,device)
   w.load_state_dict(initial['world']);l.load_state_dict(initial['learners']);sync(device)
   start=time.perf_counter();iteration(w,l);sync(device);first=time.perf_counter()-start
   timings=[]
   for repeat in range(args.repeats):
    w.load_state_dict(initial['world']);l.load_state_dict(initial['learners'])
    for _ in range(args.warmup):iteration(w,l)
    sync(device);start=time.perf_counter()
    for _ in range(args.steps):iteration(w,l)
    sync(device);elapsed=time.perf_counter()-start
    assert w.alive.sum().item()==cfg.worlds*cfg.initial
    timings.append(elapsed/args.steps*1000)
   result={'status':'ok','first_transition_seconds':first,'ms_per_tick':timings,'median_ms_per_tick':statistics.median(timings),
    'world_ticks_per_second':cfg.worlds*1000/statistics.median(timings),'agent_decisions_per_second':cfg.worlds*cfg.initial*1000/statistics.median(timings),
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'finite':l.finite()}
   if args.profile:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for _ in range(12):iteration(w,l)
     sync(device)
    trace=out/(name+'-trace.json');prof.export_chrome_trace(str(trace))
    es=json.loads(trace.read_text())['traceEvents'];ks=[e for e in es if e.get('cat')=='kernel']
    result['profile']={'kernels_per_tick':len(ks)/12,'kernel_sum_ms_per_tick':sum(e.get('dur',0) for e in ks)/12000}
   del w,l;gc.collect()
   result['numerical']=numerical(cfg,cls,initial,device)
   if not result['numerical']['pass']:result['status']='rejected_numerical'
   report['variants'][name]=result
  except Exception as exc:
   import traceback
   traceback.print_exc();report['variants'][name]={'status':'failed','error':str(exc)}
  atomic_json(out/'results.json',report);print(json.dumps({'variant':name,**report['variants'][name]}),flush=True)
  gc.collect();torch.cuda.empty_cache()
 atomic_json(out/'results.json',report)


if __name__=='__main__':main()
