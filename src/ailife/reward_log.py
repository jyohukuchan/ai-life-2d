"""Record actual private evaluator inputs without affecting policy or learning."""
from pathlib import Path

import numpy as np
import torch

from .observer import atomic_json


def feature_groups(cfg):
    groups={'energy':[0],'visual':list(range(1,4)),'received_sound':list(range(4,4+cfg.symbols))}
    offset=4+cfg.symbols
    for name,size in zip(['movement','mouth','sent_sound','reproduction'],cfg.action_sizes):
        groups[name]=list(range(offset,offset+size));offset+=size
    return groups


class RewardLog:
    """Bounded device buffer; only old live IDs, including terminal transitions."""
    def __init__(self,cfg,directory):
        self.cfg=cfg;self.directory=Path(directory)/'reward-components'
        self.directory.mkdir(exist_ok=True);self.frames=[]
        atomic_json(self.directory/'schema.json',{
            'version':1,'groups':feature_groups(cfg),
            'features':['own_energy_delta_or_zero','mean_rgb_delta_R','mean_rgb_delta_G','mean_rgb_delta_B']+
                       [f'received_symbol_{i}' for i in range(1,cfg.symbols+1)]+
                       [f'action_{j}_choice_{k}' for j,size in enumerate(cfg.action_sizes) for k in range(size)],
            'terms':'eval_w * phi; additive BEFORE tanh',
            'raw_reward':'tanh(sum(terms))','scaled_reward':'raw_reward / scale',
            'group_removal_effect':'tanh(sum(terms)) - tanh(sum(terms) - sum(group terms)); non-additive',
            'sampling':'all transitions of old live individuals, including their final transition',
            'time':'tick is the completed transition, founder first transition is tick 1',
            'not_agent_input':True,'birth_parent':'successful parent; both partners for sexual reproduction'})

    def append(self,t,diagnostics,tick):
        parent_count=torch.zeros_like(t.ids)
        parent_count.scatter_add_(1,t.parent_a,t.births.long())
        parent_count.scatter_add_(1,t.parent_b.clamp_min(0),(t.births&(t.parent_b>=0)).long())
        integer=torch.stack((t.was_alive.long(),t.ids,t.ate.long(),t.moved.long(),t.terminal.long(),(parent_count>0).long()),-1)
        floats=torch.cat((diagnostics['phi'],diagnostics['terms'],
                          torch.stack([diagnostics[k] for k in ['raw_reward','scaled_reward','scale','energy_coefficient']],-1)), -1)
        self.frames.append((integer,floats.detach()));self.last_tick=tick
        if len(self.frames)>=self.cfg.reward_log_block:self.flush()

    def flush(self):
        if not self.frames:return
        first=self.last_tick-len(self.frames)+1
        ints=torch.stack([x[0] for x in self.frames]).cpu().numpy()
        floats=torch.stack([x[1] for x in self.frames]).reshape(len(self.frames),self.cfg.worlds,self.cfg.capacity,-1).cpu().numpy()
        self.frames.clear()
        dt,world,slot=np.nonzero(ints[...,0]);rows=ints[dt,world,slot];values=floats[dt,world,slot]
        count=4+self.cfg.symbols+self.cfg.action_size
        path=self.directory/f'{first:09d}-{self.last_tick:09d}.npz'
        np.savez_compressed(path,tick=(dt+first).astype(np.int32),world=world.astype(np.int16),slot=slot.astype(np.int16),
                            id=rows[:,1],ate=rows[:,2].astype(bool),moved=rows[:,3].astype(bool),terminal=rows[:,4].astype(bool),
                            birth_parent=rows[:,5].astype(bool),phi=values[:,:count],terms=values[:,count:2*count],
                            raw_reward=values[:,-4],scaled_reward=values[:,-3],scale=values[:,-2],energy_coefficient=values[:,-1])
