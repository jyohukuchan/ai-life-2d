import dataclasses
import json

import numpy as np
import torch

from ailife.config import Config
from ailife.learner import Learners
from ailife.runner import iteration, run
from ailife.selection import EliteArchive, model_row, seed_population
from ailife.world import World


def small(**kw):
    return Config(**dict(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,threads=1,
                         ticks=12,founder_energy=10,initial_food=0,regrowth=0,replay_count=2,replay_length=4,log_ticks=4,checkpoint_ticks=0,
                         elite_count=2,**kw))


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_selection_capture_precedes_reuse_and_is_not_feeding_score():
    torch.set_num_threads(1);cfg=small(lifespan=3)
    world,learner=World(cfg),Learners(cfg);archive=EliteArchive(cfg,'cpu')
    learner.params['act_b'].data[0].fill_(7)
    old={k:getattr(world,k).clone() for k in ('age','birth_tick','generation')}
    action=torch.zeros(1,4,4,dtype=torch.long)
    learner.act(world.observe(),world.ids,world.clock,world.alive)
    world.energy[0,0]=.001
    t=world.step(action);learner.learn(world.observe(t.positions,t.directions,t.energy),t)
    archive.observe(world,learner,t,old)
    saved=archive.deaths[0][0]['params']['act_b'].clone()
    learner.params['act_b'].data[0].fill_(-123)
    assert torch.equal(archive.deaths[0][0]['params']['act_b'],saved)
    for _ in range(2):iteration(world,learner,archive)
    pack=archive.selected(world,learner)
    assert [x['id'] for x in pack['worlds'][0]]==[1,2]
    assert pack['worlds'][0][1]['death_tick']==3


def test_acquired_vs_genome_exact_transfer_and_private_clone_reset():
    torch.set_num_threads(1);cfg=small();learner=Learners(cfg)
    learner.params['enc_b'].data[0].fill_(5)
    learner.obs_n[0]=10;learner.obs_mean[0]=3;learner.obs_m2[0]=2
    donor={**model_row(learner,0),'id':17,'death_tick':10,'eaten':0}
    pack={'worlds':[[donor]]}
    world,acquired,assign=seed_population(cfg,pack,'acquired')
    _,genome,_=seed_population(cfg,pack,'genome')
    assert torch.equal(acquired.params['enc_b'][0],learner.params['enc_b'][0])
    assert torch.equal(genome.params['enc_b'][0],learner.genome['enc_b'][0])
    assert acquired.obs_n[0]==10 and genome.obs_n[0]==0
    assert torch.equal(acquired.normalization_genome['obs_mean'][0],learner.obs_mean[0])
    assert not acquired.adam_count.any() and not acquired.hidden.any()
    assert world.age.sum()==0 and world.energy[world.alive].unique().tolist()==[cfg.founder_energy]
    acquired.params['enc_b'].data[0,0]=90
    assert acquired.params['enc_b'][1,0]==5
    assert acquired.genome['enc_b'][0,0]==5
    assert len(assign)==cfg.initial
    # Birth inherits the calibration attached to the birth-time model, never the current statistics.
    t=world.step(torch.zeros((1,4,4),dtype=torch.long));t.births.zero_();t.births[0,2]=True;t.donor[0,2]=0;t.parent_a[0,2]=0;t.parent_b[0,2]=-1
    acquired.obs_mean[0].fill_(999);world.ids[0,2]=20
    acquired.inherit(t,world.ids)
    assert acquired.obs_mean[2].unique().tolist()==[3.0]


def test_legacy_normalization_defaults_and_archive_resume(tmp_path):
    cfg=small(lifespan=6)
    learner=Learners(cfg);state=learner.state_dict();state.pop('normalization_genome')
    restored=Learners(cfg);restored.load_state_dict(state)
    assert not any(t.any() for t in restored.normalization_genome.values())
    run(dataclasses.replace(cfg,ticks=4),directory=tmp_path/'first',observe=False)
    run(cfg,directory=tmp_path/'resumed',resume=tmp_path/'first/checkpoint.pt',observe=False)
    run(cfg,directory=tmp_path/'full',observe=False)
    a=torch.load(tmp_path/'resumed/elites.pt',weights_only=True)
    b=torch.load(tmp_path/'full/elites.pt',weights_only=True)
    equal(a,b)


def test_extinct_fast_forward_is_exact_and_preserves_replays(tmp_path):
    cfg=small(lifespan=2)
    run(cfg,directory=tmp_path/'full',observe=False)
    run(dataclasses.replace(cfg,fast_forward_extinct=True),directory=tmp_path/'fast',observe=False)
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'fast/checkpoint.pt',weights_only=True)
    equal(a['world'],b['world']);equal(a['learners'],b['learners']);equal(a['elite_archive'],b['elite_archive'])
    for path in (tmp_path/'full/replays').glob('*.npz'):
        with np.load(path) as left,np.load(tmp_path/'fast/replays'/path.name) as right:
            for k in left.files:assert np.array_equal(left[k],right[k]),k
    summary=json.loads((tmp_path/'fast/summary.json').read_text())
    assert summary['tick']==12 and summary['measured_ticks']==2 and summary['fast_forwarded_ticks']==10


def test_real_death_and_birth_in_same_tick_keeps_old_individual(monkeypatch):
    cfg=dataclasses.replace(small(),capacity=2,initial=2,founder_energy=50,maturity_ratio=0,
                            lifespan=100,remote_births=True)
    world,learner=World(cfg),Learners(cfg);archive=EliteArchive(cfg,'cpu')
    world.energy[0,0]=.001
    learner.params['enc_b'].data[0].fill_(17)
    original=learner.act
    def forced(raw,ids,clock,alive):
        original(raw,ids,clock,alive)
        actions=torch.zeros(1,2,4,dtype=torch.long);actions[0,1,3]=1
        return actions
    monkeypatch.setattr(learner,'act',forced)
    t,_=iteration(world,learner,archive)
    assert t.terminal[0,0] and t.births[0,0]
    assert t.ids[0,0]==1 and world.ids[0,0]==3
    saved=archive.deaths[0][0]
    assert saved['id']==1 and saved['birth_tick']==0 and saved['age']==1
    assert saved['params']['enc_b'].mean()>16
    assert learner.params['enc_b'][0].abs().max()<1


def test_horizon_survivors_take_precedence_over_dead_food_winners():
    cfg=small();world,learner=World(cfg),Learners(cfg);archive=EliteArchive(cfg,'cpu')
    archive.deaths[0]=[{'id':99,'death_tick':1,'eaten':999}]
    world.alive[0,1]=False
    pack=archive.selected(world,learner)
    assert [r['id'] for r in pack['worlds'][0]]==[1]
    assert pack['worlds'][0][0]['selected_alive']
