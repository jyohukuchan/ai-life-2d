"""Create matched, entirely untrained CPU initial states for compass ablations."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
import yaml

from ailife.config import Config
from ailife.learner import Learners
from ailife.observer import atomic_json, replay_schedule
from ailife.runner import checkpoint
from ailife.world import World


def state_digest(state):
    digest=hashlib.sha256()
    def visit(value,path):
        digest.update(path.encode())
        if isinstance(value,torch.Tensor):
            digest.update(str((value.dtype,tuple(value.shape))).encode())
            digest.update(value.contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for k in sorted(value):visit(value[k],path+'/'+k)
        elif isinstance(value,tuple):
            for i,v in enumerate(value):visit(v,path+'/'+str(i))
    visit(state,'learner')
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='configs/experiments/2026-10-11-feeding-calibrated-l3/feeding-calibrated-L3.yaml')
    parser.add_argument('--output',default='runs/compass-ablation-initial')
    parser.add_argument('--config-prefix',default='configs/experiments/2026-10-11-compass/compass')
    args=parser.parse_args();base=Config.load(args.base);torch.set_num_threads(base.threads)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configs={mode:dataclasses.replace(base,mode='L3',compass=mode,log_reward_components=True) for mode in ['absolute','masked']}
    configs['random']=dataclasses.replace(base,mode='RANDOM',compass='none',log_reward_components=False)
    for name,cfg in configs.items():
        path=Path(args.config_prefix+'-'+name+'.yaml');path.parent.mkdir(parents=True,exist_ok=True)
        text=yaml.safe_dump(cfg.as_dict(),sort_keys=False)
        if path.exists() and path.read_text()!=text:raise FileExistsError(path)
        path.write_text(text);Config.load(path)
    cfg=configs['masked'];world,learners=World(cfg),Learners(cfg)
    assert world.clock.item()==0 and world.alive.sum().item()==cfg.worlds*cfg.initial
    for field in ['obs_n','reward_n','adam_count','hidden','last_action']:
        assert not getattr(learners,field).any(),field
    for name in learners.params:assert torch.equal(learners.params[name],learners.genome[name])
    for mode in ['absolute','masked']:
        checkpoint(out/(mode+'.pt'),configs[mode],world,learners,replay_schedule(configs[mode]))
    summary={'initial_world_digest':world.digest(),'initial_learner_digest':state_digest(learners.state_dict()),
             'clock':0,'pretraining':False,'all_history_and_optimizer_counts_zero':True,
             'parameters_per_agent':learners.parameter_count(),'observation_size':cfg.observation_size,
             'prediction_target_size':cfg.sensory_size,'compass_order':['north','east','south','west'],
             'extra_encoder_parameters':4*32,
             'note':'Both L3 trials load identical world/genome/weights/state tensors. Per-individual states remain private; matching copies are across experimental conditions only. Compass is policy input, not reward or prediction target.'}
    atomic_json(out/'initialization.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
