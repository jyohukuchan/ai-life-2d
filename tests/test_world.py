import torch

from ailife.config import Config
from ailife.world import World


def setup(initial=1,capacity=4,**kwargs):
    c=Config(worlds=1,initial=initial,capacity=capacity,size=12,metabolism=0,move_cost=0,regrowth=0,initial_food=0,obstacle_density=0,**kwargs)
    w=World(c)
    w.food.zero_(); w.xy.zero_()
    w.xy[0,0]=torch.tensor([5,5]); w.age.fill_(c.maturity)
    if initial>1:w.xy[0,1]=torch.tensor([5,6])
    return w


def test_clone_energy_and_identity():
    w=setup(); w.energy[0,0]=100
    actions=torch.zeros(1,4,4,dtype=torch.long);actions[0,0,3]=1
    before=w.energy.sum().item();t=w.step(actions)
    assert t.births.sum()==1
    assert w.energy.sum().item()==before-10
    assert sorted(w.energy[w.alive].tolist())==[30,60]
    assert len(torch.unique(w.ids[w.alive]))==2
    assert w.parents[0,t.births[0],0].item()==1


def test_sexual_energy_and_two_parents():
    w=setup(initial=2);w.energy[0,:2]=100
    actions=torch.zeros(1,4,4,dtype=torch.long);actions[0,:2,3]=2
    before=w.energy.sum().item();t=w.step(actions)
    assert t.births.sum()==1
    assert w.energy.sum().item()==before-10
    assert sorted(w.parents[t.births].flatten().tolist())==[1,2]


def test_capacity_rejection_costs_nothing():
    w=setup(initial=1,capacity=1); w.energy.fill_(100)
    actions=torch.zeros(1,1,4,dtype=torch.long);actions[0,0,3]=1
    t=w.step(actions)
    assert not t.births.any()
    assert w.energy[0,0]==100
    assert w.capacity_rejected.sum()==1


def test_parent_spending_all_energy_dies_immediately():
    w=setup();w.energy[0,0]=40
    actions=torch.zeros(1,4,4,dtype=torch.long);actions[0,0,3]=1
    t=w.step(actions)
    assert t.births.sum()==1
    assert t.terminal[0,0]
    assert not w.alive[0,0]
    assert w.alive.sum()==1
    assert t.energy[0,0]==0


def test_snapshot_resume_and_observation_contract():
    c=Config(worlds=2,initial=4,capacity=16,size=20,radius=2,seed=17)
    a,b=World(c),World(c)
    for i in range(12):
        action=torch.randint(0,2,(2,16,4))
        a.step(action);b.step(action)
    assert a.digest()==b.digest()
    b.load_state_dict(a.state_dict())
    act=torch.zeros(2,16,4,dtype=torch.long)
    a.step(act);b.step(act)
    assert a.digest()==b.digest()
    obs=a.observe()
    assert obs.shape==(2,16,c.sensory_size)
    assert torch.isfinite(obs).all()


def test_collision_does_not_overlap():
    w=setup(initial=2,capacity=2)
    w.xy[0,0]=torch.tensor([5,4]);w.xy[0,1]=torch.tensor([5,6])
    w.direction[0,:2]=torch.tensor([1,3])
    action=torch.zeros(1,2,4,dtype=torch.long);action[:,:,0]=1
    t=w.step(action)
    assert t.moved.sum()==1
    assert not torch.equal(w.xy[0,0],w.xy[0,1])


def test_sound_is_delayed_and_excludes_self():
    w=setup(initial=2,capacity=2)
    before=w.observe()
    action=torch.zeros(1,2,4,dtype=torch.long);action[0,0,2]=1
    w.step(action)
    after=w.observe()
    visual=3*(2*w.cfg.radius+1)**2
    assert before[...,visual:visual+4].sum()==0
    assert after[0,0,visual:visual+4].sum()==0
    assert after[0,1,visual]>0
