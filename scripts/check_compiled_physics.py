"""Same forced actions, FP32 physical state: validate compilation across births/deaths."""
import dataclasses,json,torch
from pathlib import Path
from ailife.config import Config
from ailife.world import World
from ailife.random import hash32

torch.set_num_threads(4)
base=dataclasses.replace(Config.load('configs/experiments/2026-10-11-compass/compass-absolute.yaml'),log_reward_components=False)
a=World(base,'cuda');b=World(dataclasses.replace(base,execution='compiled'),'cuda')
state=torch.load('runs/compass-ablation-initial/absolute.pt',map_location='cpu',weights_only=True)
a.load_state_dict(state['world']);b.load_state_dict(state['world'])
for tick in range(600):
 key=a.ids*65537+tick*104729+a.seeds[:,None]*1000003
 actions=torch.stack([hash32(key+j*1237)%s for j,s in enumerate(base.action_sizes)],-1)
 a.step(actions);b.step(actions)
checks={};x,y=a.state_dict(),b.state_dict()
for key in x:
 if x[key].is_floating_point():
  error=float((x[key]-y[key]).abs().max());checks[key]=error
  torch.testing.assert_close(x[key],y[key],rtol=2e-6,atol=5e-4)
 else:assert torch.equal(x[key],y[key]),key
assert a.valid() and b.valid() and a.births_total.sum()>0 and a.deaths_total.sum()>0
report={'ticks':600,'same_forced_actions':True,'integer_and_boolean_state_exact':True,'max_float_errors':checks,'births':a.births_total.tolist(),'deaths':a.deaths_total.tolist(),'pass':True}
Path('runs/acceleration-20261011/compiled-physics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
