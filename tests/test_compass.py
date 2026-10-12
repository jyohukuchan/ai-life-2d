import dataclasses
import json

import pytest
import torch
from fastapi.testclient import TestClient

from ailife.config import Config
from ailife.learner import Learners
from ailife.observer import atomic_json
from ailife.policy import RandomPolicy
from ailife.runner import iteration, run
from ailife.server import create_app
from ailife.world import World


def config(**changes):
    return Config(**(dict(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,compass='absolute',
                          obstacle_density=0,initial_food=0,regrowth=0,metabolism=0,move_cost=0)|changes))


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        for k in a:equal(a[k],b[k])
    elif isinstance(a,tuple):
        for x,y in zip(a,b):equal(x,y)


def test_compass_tracks_cardinal_heading_and_relative_controls():
    cfg=config(initial=1);w=World(cfg);w.xy[0,0]=torch.tensor([5,5])
    action=torch.zeros(1,4,4,dtype=torch.long)
    for heading in range(4):
        w.direction[0,0]=heading
        obs=w.observe();assert obs.shape==(1,4,cfg.observation_size)
        torch.testing.assert_close(obs[0,0,-4:],torch.eye(4)[heading],rtol=0,atol=0)
        before=w.xy[0,0].clone()
        action.zero_();action[0,0,0]=2;w.step(action)
        assert w.direction[0,0]==(heading-1)%4 and torch.equal(w.xy[0,0],before)
        torch.testing.assert_close(w.observe()[0,0,-4:],torch.eye(4)[(heading-1)%4])
        action[0,0,0]=3;w.step(action)
        assert w.direction[0,0]==heading and torch.equal(w.xy[0,0],before)
        action[0,0,0]=1;w.step(action)
        assert torch.equal(w.xy[0,0],before+w.delta[heading])
        w.xy[0,0]=before


def test_masked_pair_has_identical_initial_weights_and_physics():
    cfg=config();masked=dataclasses.replace(cfg,compass='masked');legacy=dataclasses.replace(cfg,compass='none')
    a,b=World(cfg),World(masked);la,lb=Learners(cfg),Learners(masked)
    assert a.digest()==b.digest();equal(la.state_dict(),lb.state_dict())
    assert la.parameter_count()==lb.parameter_count()==Learners(legacy).parameter_count()+128
    assert not b.observe()[...,-4:].any()
    torch.testing.assert_close(a.observe()[...,:cfg.sensory_size],b.observe()[...,:cfg.sensory_size],rtol=0,atol=0)
    with torch.no_grad():
        za=la._core(a.observe().reshape(4,-1),la.hidden,la.last_action)[2]
        zb=lb._core(b.observe().reshape(4,-1),lb.hidden,lb.last_action)[2]
    assert not torch.equal(za,zb),'compass must reach the encoder'
    assert la.params['pred_w'].shape[-1]==cfg.sensory_size
    assert la.obs_mean.shape[-1]==cfg.sensory_size


def test_compass_is_excluded_from_intrinsic_reward_and_energy_feature():
    cfg=config(log_reward_components=True);a,b=Learners(cfg),Learners(cfg);w=World(cfg)
    raw=w.observe().reshape(4,-1);next_raw=raw.clone();next_raw[:,cfg.sensory_size-1]+=.1
    different=next_raw.clone();different[:,-4:]=torch.roll(next_raw[:,-4:],1,-1)
    actions=a.onehot(torch.zeros(4,4,dtype=torch.long));active=w.alive.flatten()
    ra,sa=a._reward(raw,next_raw,actions,active);rb,sb=b._reward(raw,different,actions,active)
    assert torch.equal(ra,rb) and torch.equal(sa,sb)
    torch.testing.assert_close(a.reward_diagnostics['phi'][:,0],torch.full((4,),.1))
    assert a.reward_diagnostics['phi'].shape[-1]==4+cfg.symbols+cfg.action_size


def test_compass_is_not_an_additional_prediction_target():
    cfg=config(mode='L2');w=World(cfg);a,b=Learners(cfg),Learners(cfg)
    raw=w.observe();actions=a.act(raw,w.ids,w.clock,w.alive);assert torch.equal(actions,b.act(raw,w.ids,w.clock,w.alive))
    t=w.step(actions);next_raw=w.observe(t.positions,t.directions,t.energy);different=next_raw.clone();different[...,-4:]=torch.roll(next_raw[...,-4:],1,-1)
    ra,rb=a.learn(next_raw,t),b.learn(different,t)
    assert torch.equal(ra['prediction_loss'],rb['prediction_loss'])
    equal(a.state_dict(),b.state_dict())


def test_terminal_observation_uses_old_heading_after_slot_reuse():
    cfg=config(initial=2,capacity=2,metabolism=.02);w=World(cfg)
    w.xy[0]=torch.tensor([[5,5],[5,6]]);w.age.fill_(cfg.maturity);w.energy[0]=torch.tensor([100.,.01]);w.direction[0,1]=3
    action=torch.zeros(1,2,4,dtype=torch.long);action[0,0,3]=1;action[0,1,0]=2
    t=w.step(action);assert t.terminal[0,1] and t.births[0,1] and t.ids[0,1]!=w.ids[0,1]
    # Distinguish the newly occupied slot's direction from the old terminal body.
    w.direction[0,1]=(t.directions[0,1]+1)%4
    final=w.observe(t.positions,t.directions,t.energy)
    torch.testing.assert_close(final[0,1,-4:],torch.eye(4)[t.directions[0,1]])
    assert not torch.equal(final[0,1,-4:],w.observe()[0,1,-4:])


def test_random_physics_ignores_compass_flag():
    cfg=config(mode='RANDOM');other=dataclasses.replace(cfg,compass='masked')
    a,b=World(cfg),World(other);pa,pb=RandomPolicy(cfg),RandomPolicy(other)
    for _ in range(12):iteration(a,pa);iteration(b,pb)
    assert a.digest()==b.digest()


def test_legacy_configuration_has_no_new_channels_and_resume_rejects_switch(tmp_path):
    saved=config(compass='none').as_dict();saved.pop('compass')
    old=Config.from_saved(saved);assert old.compass=='none' and old.observation_size==old.sensory_size
    c=config(ticks=3,log_ticks=3,checkpoint_ticks=0,replay_count=0,threads=1)
    run(c,directory=tmp_path/'first',observe=False)
    with pytest.raises(ValueError,match='compass'):
        run(dataclasses.replace(c,compass='masked',ticks=4),directory=tmp_path/'wrong',resume=tmp_path/'first/checkpoint.pt',observe=False)


def test_comparison_can_label_two_learning_conditions(tmp_path):
    root=tmp_path/'runs';root.mkdir()
    for name,compass in [('with','absolute'),('without','masked')]:
        cfg=config(compass=compass);p=root/name;p.mkdir()
        atomic_json(p/'latest.json',{'tick':2,'worlds':[World(cfg).snapshot()]})
        (p/'metrics.jsonl').write_text(json.dumps({'tick':2,'alive':[2]})+'\n')
    atomic_json(root/'comparison.json',{'learned':'with','control':'without','labels':{'learned':'L3 compass','control':'L3 masked'},'display_labels':{'learned':'L3 方角あり','control':'L3 方角なし'}})
    result=TestClient(create_app(root)).get('/api/comparison').json()
    assert [g['label'] for g in result['groups']]==['L3 方角あり','L3 方角なし']
    assert result['paired_population']=={'tick':2,'learned':[2],'control':[2]}
