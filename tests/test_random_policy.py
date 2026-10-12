import dataclasses

import torch

from ailife.config import Config
from ailife.policy import RandomPolicy, make_policy
from ailife.runner import iteration, run
from ailife.world import World


def test_actions_are_uniform_and_ignore_observations():
    cfg=Config(worlds=2,initial=8,capacity=16,size=20,mode='RANDOM')
    world=World(cfg);policy=make_policy(cfg)
    assert isinstance(policy,RandomPolicy) and policy.parameter_count()==0
    samples=[]
    for tick in range(400):
        clock=torch.tensor(tick)
        a=policy.act(None,world.ids,clock,world.alive)
        b=policy.act(torch.randn(2,16,cfg.sensory_size),world.ids,clock,world.alive)
        assert torch.equal(a,b)
        assert not a[~world.alive].any()
        samples.append(a[world.alive])
    samples=torch.cat(samples)
    for j,size in enumerate(cfg.action_sizes):
        frequency=torch.bincount(samples[:,j],minlength=size)/len(samples)
        assert ((frequency-1/size).abs()<.025).all()


def test_control_uses_identical_initial_world_and_resumes_exactly():
    cfg=Config(worlds=2,initial=4,capacity=16,size=20,mode='RANDOM')
    world,other=World(cfg),World(dataclasses.replace(cfg,mode='L3'))
    assert world.digest()==other.digest()
    policy=make_policy(cfg)
    for _ in range(20):iteration(world,policy)
    restored=World(cfg);restored.load_state_dict(world.state_dict())
    other_policy=make_policy(cfg);other_policy.load_state_dict(policy.state_dict())
    for _ in range(20):iteration(world,policy);iteration(restored,other_policy)
    assert world.digest()==restored.digest()
    assert world.valid() and policy.state_dict()=={}


def test_random_mode_runs_and_saves_with_no_learning(tmp_path):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,mode='RANDOM',ticks=12,log_ticks=6,checkpoint_ticks=0,replay_count=0,threads=1)
    summary=run(cfg,directory=tmp_path/'random')
    assert summary['status']=='complete' and summary['parameters_per_agent']==0
    assert summary['action_policy']=='independent_uniform_per_component_each_tick'
    assert summary['update_targets']=={'prediction':[],'local':[],'rl':[]}
    state=torch.load(tmp_path/'random/checkpoint.pt',weights_only=True)
    assert state['learners']=={}
