"""Session-boundary selection; never supplies a reward or input to an agent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .learner import Learners
from .world import World


NORMALIZATION=('obs_n','obs_mean','obs_m2')


def model_row(learners,index):
    def row(tensor):return tensor[index].detach().cpu().clone()
    return {'params':{k:row(v) for k,v in learners.params.items()},
            'genome':{k:row(v) for k,v in learners.genome.items()},
            'normalization':{k:row(getattr(learners,k)) for k in NORMALIZATION},
            'normalization_genome':{k:row(v) for k,v in learners.normalization_genome.items()},
            **{k:row(getattr(learners,k)) for k in ('log_rates','genome_rates','eval_w','genome_eval')}}


class EliteArchive:
    """Capture terminal weights before a newborn can overwrite the same slot."""
    def __init__(self,cfg,device):
        self.cfg=cfg
        self.deaths=[[] for _ in range(cfg.worlds)]
        self.eaten=torch.zeros((cfg.worlds,cfg.capacity),dtype=torch.int64,device=device)

    def observe(self,world,learners,t,old):
        self.eaten.add_(t.ate)
        tick=int(world.clock.item())
        for wi,slot in t.terminal.nonzero().cpu().tolist():
            record={**model_row(learners,wi*self.cfg.capacity+slot),
                    'world':wi,'id':int(t.ids[wi,slot]),'death_tick':tick,'selected_alive':False,
                    'age':int(old['age'][wi,slot])+1,'birth_tick':int(old['birth_tick'][wi,slot]),
                    'generation':int(old['generation'][wi,slot]),'eaten':int(self.eaten[wi,slot])}
            rows=self.deaths[wi];rows.append(record)
            rows.sort(key=lambda x:(x['death_tick'],x['id']))
            del rows[:-self.cfg.elite_count]
        self.eaten.masked_fill_(t.births,0)

    def selected(self,world,learners):
        result=[]
        for wi in range(self.cfg.worlds):
            alive=world.alive[wi].nonzero().flatten().cpu().numpy()
            if len(alive):
                rng=np.random.default_rng(self.cfg.observer_seed+wi*1009+int(world.clock.item()))
                chosen=sorted(rng.choice(alive,min(len(alive),self.cfg.elite_count),replace=False).tolist())
                rows=[{**model_row(learners,wi*self.cfg.capacity+slot),'world':wi,
                       'id':int(world.ids[wi,slot]),'death_tick':None,'selected_alive':True,
                       'age':int(world.age[wi,slot]),'birth_tick':int(world.birth_tick[wi,slot]),
                       'generation':int(world.generation[wi,slot]),'eaten':int(self.eaten[wi,slot])} for slot in chosen]
            else:rows=self.deaths[wi]
            if not rows:raise ValueError(f'no candidates in world {wi}')
            result.append(rows)
        return {'version':1,'config':self.cfg.as_dict(),'tick':int(world.clock.item()),'worlds':result,
                'selection':'alive uniform up to K, otherwise K latest deaths; no feeding score'}

    def state_dict(self):return {'deaths':self.deaths,'eaten':self.eaten.detach().cpu().clone()}

    def load_state_dict(self,state):
        self.deaths=state['deaths'];self.eaten.copy_(state['eaten'].to(self.eaten.device))


def import_completed_elites(directory,k=4):
    """Import an older completed run only when every selected ID still occupies its slot."""
    from .config import Config
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='complete':raise ValueError('source run is not complete')
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    cfg=Config.from_saved(state['config']);torch.set_num_threads(cfg.threads)
    world=World(cfg);world.load_state_dict(state['world'])
    learners=Learners(cfg);learners.load_state_dict(state['learners'])
    deaths=[[] for _ in range(cfg.worlds)];food={}
    for line in (directory/'events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['event']=='death':deaths[event['world']].append(event)
    for path in sorted((directory/'reward-components').glob('*.npz')):
        with np.load(path) as data:
            ate=data['ate'];wis=data['world'][ate];ids=data['id'][ate]
            for wi,uid in zip(wis,ids):food[int(wi),int(uid)]=food.get((int(wi),int(uid)),0)+1
    result=[]
    for wi in range(cfg.worlds):
        if world.alive[wi].any():raise ValueError('legacy import currently requires extinction')
        chosen=sorted(deaths[wi],key=lambda e:(e['tick'],e['id']))[-k:]
        rows=[]
        for event in chosen:
            slots=(world.ids[wi]==event['id']).nonzero().flatten()
            if len(slots)!=1:raise ValueError('selected weights were overwritten; replay the source run')
            slot=int(slots[0]);rows.append({**model_row(learners,wi*cfg.capacity+slot),
                'world':wi,'id':event['id'],'death_tick':event['tick'],'selected_alive':False,
                'age':int(world.age[wi,slot]),'birth_tick':int(world.birth_tick[wi,slot]),
                'generation':int(world.generation[wi,slot]),'eaten':food.get((wi,event['id']),0)})
        if not rows:raise ValueError('empty selection pool')
        result.append(rows)
    return {'version':1,'config':cfg.as_dict(),'tick':int(world.clock.item()),'worlds':result,
            'source':str(directory.resolve()),'import_verified_ids':True,
            'selection':'K latest deaths; no feeding score'}


@torch.no_grad()
def seed_population(cfg,pack,arm,original=None):
    """Each clone has separate storage/state. No optimizer or recurrent history is copied."""
    if arm not in {'acquired','genome','fresh'}:raise ValueError(arm)
    world,learners=World(cfg),Learners(cfg)
    assignments=[]
    if arm=='fresh':
        if original is None:raise ValueError('fresh arm needs the original founder bank')
        learners.load_state_dict(original['learners'])
    else:
        if len(pack['worlds'])!=cfg.worlds:raise ValueError('world count changed')
        for wi,rows in enumerate(pack['worlds']):
            for slot in range(cfg.initial):
                donor=rows[slot%len(rows)];idx=wi*cfg.capacity+slot
                weights=donor['params' if arm=='acquired' else 'genome']
                norm=donor['normalization' if arm=='acquired' else 'normalization_genome']
                for name,value in weights.items():
                    learners.params[name][idx].copy_(value);learners.genome[name][idx].copy_(value)
                for name,value in norm.items():
                    getattr(learners,name)[idx].copy_(value);learners.normalization_genome[name][idx].copy_(value)
                for dest,source in [('log_rates','log_rates' if arm=='acquired' else 'genome_rates'),
                                    ('eval_w','eval_w' if arm=='acquired' else 'genome_eval')]:
                    getattr(learners,dest)[idx].copy_(donor[source])
                learners.genome_rates[idx].copy_(learners.log_rates[idx]);learners.genome_eval[idx].copy_(learners.eval_w[idx])
                assignments.append({'world':wi,'founder_id':slot+1,'donor_id':donor['id'],
                                    'donor_death_tick':donor['death_tick'],'donor_eaten':donor['eaten']})
    for name in ['hidden','last_action','reward_u','reward_n','reward_mean','reward_m2','adam_count']:
        assert not getattr(learners,name).any(),name
    assert world.valid() and learners.finite()
    return world,learners,assignments
