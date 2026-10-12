"""Separate new trials: preserve visible learning trends, not exact stochastic trajectories."""
import argparse,dataclasses,gc,json,time
from pathlib import Path
import torch
from ailife.config import Config
from ailife.observer import atomic_json
import ailife.runner as runner


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variants',default='fp32-eager,fp32-compiled,bf16-compiled,fp16-compiled')
p.add_argument('--ticks',type=int,default=3000);p.add_argument('--tag',default='r9700')
args=p.parse_args();torch.set_num_threads(4)
out=Path('runs/acceleration-20261011')/(args.tag+'-learning');out.mkdir(parents=True,exist_ok=True)
factory=runner.make_policy;integrity={};holder=[]
def monitored_policy(cfg,device):
 learner=factory(cfg,device);original=learner.learn
 counters={k:torch.zeros(cfg.worlds,dtype=torch.int64,device=device) for k in ['decisions','prediction_updates','skipped_updates','invalid_counter_changes']}
 def learn(raw,t):
  before=learner.adam_count.clone();result=original(raw,t);change=learner.adam_count-before
  counters['decisions'].add_(t.was_alive.sum(1))
  counters['prediction_updates'].add_(change.reshape(cfg.worlds,cfg.capacity).sum(1))
  counters['skipped_updates'].add_((t.was_alive.flatten()&(change==0)).reshape(cfg.worlds,cfg.capacity).sum(1))
  counters['invalid_counter_changes'].add_(((change<0)|(change>1)|(~t.was_alive.flatten()&(change!=0))).reshape(cfg.worlds,cfg.capacity).sum(1))
  return result
 learner.learn=learn;holder[:]=[learner,counters];return learner
runner.make_policy=monitored_policy
report={'ticks':args.ticks,'paired_map_session':9,'initial_states':{'cold':'session-selection-initial/fresh-s08.pt','warm':'session-selection-initial/acquired-s08.pt'},'variants':{}}
for variant in args.variants.split(','):
 precision,execution=variant.split('-');report['variants'][variant]={}
 for condition in ['cold','warm']:
  source=Path('runs')/report['initial_states'][condition]
  saved=torch.load(source,map_location='cpu',weights_only=True)
  cfg=dataclasses.replace(Config.from_saved(saved['config']),precision=precision,execution=execution,ticks=args.ticks,elite_count=0,fast_forward_extinct=False,log_reward_components=False,log_ticks=250,checkpoint_ticks=0,replay_count=3,replay_length=128)
  directory=Path('runs')/f'accel-{args.tag}-{variant}-{condition}'
  if directory.exists():raise FileExistsError(directory)
  del saved;gc.collect()
  print(json.dumps({'phase':'learning_start','variant':variant,'condition':condition}),flush=True)
  summary=runner.run(cfg,'cuda:0',directory,resume=source)
  learner,counters=holder
  check={k:v.cpu().tolist() for k,v in counters.items()}
  check.update(finite=learner.finite(),master_fp32=all(p.dtype==torch.float32 for p in learner.params.values()),statistics_fp32=all(getattr(learner,k).dtype==torch.float32 for k in ['obs_n','obs_mean','obs_m2','reward_n','reward_mean','reward_m2','hidden']))
  assert check['finite'] and check['master_fp32'] and check['statistics_fp32']
  assert sum(check['invalid_counter_changes'])==0
  atomic_json(directory/'learning-integrity.json',check)
  report['variants'][variant][condition]={'run':directory.name,'integrity':check,'summary':{k:summary[k] for k in ['status','eaten','alive','extinction_tick','elapsed_seconds','world_ticks_per_second']}}
  atomic_json(out/'results.json',report)
  print(json.dumps({'phase':'learning_complete','variant':variant,'condition':condition,**report['variants'][variant][condition]}),flush=True)
  holder.clear();del learner,counters,check;gc.collect();torch.cuda.empty_cache()
atomic_json(out/'results.json',report)
