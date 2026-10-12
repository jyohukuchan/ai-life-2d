import dataclasses
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ailife.config import Config
from ailife.learner import Learners
from ailife.observer import Observer, atomic_json
from ailife.reward_log import RewardLog, feature_groups
from ailife.runner import iteration, run
from ailife.server import create_app
from ailife.terrain import make_obstacles, preserve_food_total
from ailife.world import World
from test_obstacles import reachable_count


def rectangle(**changes):
    values=dict(worlds=2,initial=4,capacity=16,size=57,height=28,width=112,
                food_x_min=.75,spawn_x_max=.25,remote_births=True,
                food_budget='preserve_total',food_reference_size=57,hidden=8,radius=1)
    return Config(**(values|changes))


def test_rectangle_regions_connectivity_and_food_budget():
    cfg=rectangle();w=World(cfg)
    reference=World(dataclasses.replace(cfg,height=57,width=57,food_x_min=0,spawn_x_max=1,remote_births=False,food_budget='local'))
    assert w.food.shape==(2,28,112)
    assert (w.xy[...,1][w.alive]<28).all()
    assert not w.food[:,:,:84].any() and w.food[:,:,84:].any()
    for i in range(2):assert reachable_count(w.walkable[i].numpy())==int(w.walkable[i].sum())
    torch.testing.assert_close(w.fertility.sum((1,2)),reference.fertility.sum((1,2)))
    assert (cfg.initial_food*w.fertility<=1).all()
    assert (cfg.regrowth*w.fertility<=1).all()
    # Force regrowth high while retaining the source region constraint.
    w.cfg=dataclasses.replace(cfg,regrowth=1,metabolism=0)
    w.food.zero_();w.step(torch.zeros(2,16,4,dtype=torch.long))
    assert w.food[:,:,84:].any() and not w.food[:,:,:84].any()
    assert w.valid()


@pytest.mark.parametrize('sexual',[False,True])
def test_children_spawn_left_when_parents_are_right_and_conserve_energy(sexual):
    cfg=rectangle(worlds=1,initial=2 if sexual else 1,capacity=4,metabolism=0,move_cost=0,regrowth=0,initial_food=0,obstacle_density=0)
    w=World(cfg);w.xy[0,0]=torch.tensor([10,90]);w.xy[0,1]=torch.tensor([10,91])
    w.energy[w.alive]=100;w.age.fill_(cfg.maturity)
    actions=torch.zeros(1,4,4,dtype=torch.long);actions[0,:cfg.initial,3]=2 if sexual else 1
    before=w.energy.sum();t=w.step(actions)
    assert t.births.sum()==1
    assert (w.xy[...,1][t.births]<28).all()
    assert w.energy.sum()==before-cfg.birth_overhead
    assert w.xy[0,0,1]==90 and w.valid()
    assert w.parents[t.births].tolist()==[[1,2 if sexual else 0]]


def test_no_free_spawn_cell_does_not_charge_parent():
    cfg=rectangle(worlds=1,initial=1,capacity=4,metabolism=0,regrowth=0,initial_food=0,obstacle_density=0)
    w=World(cfg);w.spawn_zone.zero_();w.energy[0,0]=100;w.age.fill_(cfg.maturity)
    actions=torch.zeros(1,4,4,dtype=torch.long);actions[0,0,3]=1
    t=w.step(actions)
    assert not t.births.any() and w.energy[0,0]==100


def test_rectangle_observation_stride_and_no_zone_label_in_vision():
    cfg=rectangle(worlds=1,initial=1,capacity=4,obstacle_density=0,initial_food=0,regrowth=0)
    w=World(cfg);w.xy[0,0]=torch.tensor([10,90]);w.direction[0,0]=1
    w.food[0,10,91]=True
    pixels=w.observe()[0,0,:27].reshape(3,3,3)
    torch.testing.assert_close(pixels[0,1],torch.tensor([.3,.8,.2]))
    torch.testing.assert_close(pixels[1,0],torch.tensor([.12,.16,.14]))
    w.xy[0,0]=torch.tensor([10,10]);w.food.zero_()
    torch.testing.assert_close(w.observe()[0,0,:27].reshape(3,3,3)[1,0],pixels[1,0])


def test_reward_log_is_exact_observation_and_keeps_old_terminal_id(tmp_path):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,log_reward_components=True,reward_log_block=3,metabolism=20,initial_food=0,regrowth=0)
    off=dataclasses.replace(cfg,log_reward_components=False)
    a,b=World(cfg),World(off);la,lb=Learners(cfg),Learners(off)
    logger=RewardLog(cfg,tmp_path);expected=[]
    for tick in range(1,6):
        t,result=iteration(a,la);iteration(b,lb)
        d=result['reward_diagnostics'];torch.testing.assert_close(d['terms'].sum(-1).tanh(),d['raw_reward'],rtol=0,atol=0)
        expected.extend([(tick,int(uid)) for uid in t.ids[t.was_alive]])
        logger.append(t,d,tick)
    logger.flush()
    assert a.digest()==b.digest()
    for name in la.params:assert torch.equal(la.params[name],lb.params[name])
    records=[];terminal=[]
    for path in sorted((tmp_path/'reward-components').glob('*.npz')):
        with np.load(path) as data:
            records.extend(zip(data['tick'].tolist(),data['id'].tolist()))
            terminal.extend(data['id'][data['terminal']].tolist())
            np.testing.assert_allclose(np.tanh(data['terms'].sum(-1)),data['raw_reward'],rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(data['raw_reward']/data['scale'],data['scaled_reward'])
    assert records==expected and sorted(terminal)==[1,2]
    assert sorted(sum(feature_groups(cfg).values(),[]))==list(range(22))


def test_rectangle_checkpoint_and_replay_round_trip(tmp_path):
    cfg=rectangle(worlds=1,initial=2,capacity=4,ticks=8,log_ticks=4,checkpoint_ticks=0,replay_count=1,replay_length=3,threads=1,log_reward_components=True)
    root=tmp_path/'runs'
    run(cfg,directory=root/'first',observe=False)
    extended=dataclasses.replace(cfg,ticks=12,log_reward_components=False)
    run(extended,directory=root/'resumed',resume=root/'first'/'checkpoint.pt',observe=False)
    run(extended,directory=root/'whole',observe=False)
    a=torch.load(root/'resumed'/'checkpoint.pt',weights_only=True)
    b=torch.load(root/'whole'/'checkpoint.pt',weights_only=True)
    for name,value in a['world'].items():assert torch.equal(value,b['world'][name])
    for name,value in a['learners']['params'].items():assert torch.equal(value,b['learners']['params'][name])
    client=TestClient(create_app(root))
    clips=client.get('/api/replays?run=first').json()['clips']
    clip=client.get('/api/replay/'+clips[0]['id']+'?run=first').json()
    assert clip['frames_data'][0]['height']==28 and clip['frames_data'][0]['width']==112
    assert len(clip['frames_data'][0]['food'])==28*112
    # A pre-rectangle replay falls back to its saved square size.
    old_cfg=Config(worlds=1,initial=1,capacity=1,size=12,hidden=8,radius=1)
    arrays={k:v[0].numpy() for k,v in World(old_cfg).state_dict().items() if k in ('xy','direction','energy','age','alive','ids','parents','generation','signal','food')}
    from ailife.observer import arrays_snapshot
    saved=old_cfg.as_dict();saved.pop('height');saved.pop('width')
    assert arrays_snapshot(arrays,saved,0,0,np.zeros((12,12)))['width']==12


def test_impossible_food_budget_rejected():
    with pytest.raises(ValueError):preserve_food_total(np.zeros((2,2)),1,1)
    assert not preserve_food_total(np.ones((2,2)),0,1).any()
