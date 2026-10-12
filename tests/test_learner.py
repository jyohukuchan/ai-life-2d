import math

import pytest
import torch

from ailife.config import Config
from ailife.learner import Learners, private_adam
from ailife.world import World


def cycle(world, learners):
    action=learners.act(world.observe(),world.ids,world.clock,world.alive)
    transition=world.step(action)
    result=learners.learn(world.observe(transition.positions,transition.directions,transition.energy),transition)
    learners.inherit(transition,world.ids)
    return result


@pytest.mark.parametrize('mode',['L0','L1','L2','L3'])
@pytest.mark.parametrize('encoder_width',[32,64])
def test_modes_freeze_inactive_and_update_only_designated_weights(mode,encoder_width):
    torch.set_num_threads(1)
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,mode=mode,encoder_width=encoder_width)
    w,l=World(cfg),Learners(cfg)
    before=l.state_dict()
    for _ in range(4):cycle(w,l)
    for name,p in l.params.items():
        assert torch.equal(p[2:],before['params'][name][2:])
        if mode=='L0' or (mode=='L1' and name!='feat_w') or (mode=='L2' and name not in l.pred_names):
            assert torch.equal(p,before['params'][name])
    assert torch.equal(l.hidden[2:],before['hidden'][2:])
    assert l.obs_n[:2].min()==4
    if mode in {'L2','L3'}:
        assert l.adam_count.tolist()==[4,4,0,0]
        assert not torch.equal(l.params['pred_w'],before['params']['pred_w'])
    if mode=='L3':
        assert not torch.equal(l.params['act_w'],before['params']['act_w'])
        assert torch.equal(l.eval_w,before['eval_w'])
    assert l.finite()


def test_private_adam_matches_independent_optimizers_with_different_ages():
    torch.manual_seed(4)
    params={'w':torch.randn(3,2,3),'b':torch.randn(3,3)}
    reference=[{k:torch.nn.Parameter(v[i].clone()) for k,v in params.items()} for i in range(3)]
    rates=torch.tensor([1e-4,2e-4,5e-4])
    optimizers=[torch.optim.Adam(list(p.values()),lr=float(rates[i])) for i,p in enumerate(reference)]
    moments={k:(torch.zeros_like(v),torch.zeros_like(v)) for k,v in params.items()}
    counts=torch.zeros(3,dtype=torch.long)
    for step in range(8):
        active=torch.tensor([step<5,step>=3,step%2==0])
        grads=[torch.randn_like(p)*(.03 if step%2 else 3) for p in params.values()]
        with torch.no_grad():private_adam(params,grads,moments,counts,rates,active)
        for i,opt in enumerate(optimizers):
            if not active[i]:continue
            for p,g in zip(reference[i].values(),grads):p.grad=g[i].clone()
            torch.nn.utils.clip_grad_norm_(list(reference[i].values()),1.)
            opt.step();opt.zero_grad()
        for k,p in params.items():
            for i in range(3):torch.testing.assert_close(p[i],reference[i][k],rtol=2e-5,atol=2e-7)
            for i,opt in enumerate(optimizers):
                state=opt.state.get(reference[i][k])
                if state:
                    torch.testing.assert_close(moments[k][0][i],state['exp_avg'],rtol=2e-5,atol=2e-7)
                    torch.testing.assert_close(moments[k][1][i],state['exp_avg_sq'],rtol=2e-5,atol=2e-7)
    assert counts.tolist()==[5,5,4]


@pytest.mark.parametrize('encoder_width',[32,64])
def test_checkpoint_restores_exact_next_transitions_and_updates(encoder_width):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,mode='L3',encoder_width=encoder_width)
    w,l=World(cfg),Learners(cfg)
    for _ in range(5):cycle(w,l)
    other_w,other_l=World(cfg),Learners(cfg)
    other_w.load_state_dict(w.state_dict());other_l.load_state_dict(l.state_dict())
    for _ in range(5):cycle(w,l);cycle(other_w,other_l)
    assert w.digest()==other_w.digest()
    for key,p in l.params.items():assert torch.equal(p,other_l.params[key])
    assert torch.equal(l.adam_count,other_l.adam_count)
    assert torch.equal(l.reward_m2,other_l.reward_m2)


@pytest.mark.parametrize('encoder_width',[32,64])
def test_child_inherits_genome_not_acquired_weights_and_resets_private_state(encoder_width):
    cfg=Config(worlds=1,initial=1,capacity=3,size=12,hidden=8,radius=1,mode='L3',mutation=0,metabolism=0,move_cost=0,regrowth=0,initial_food=0,encoder_width=encoder_width)
    w,l=World(cfg),Learners(cfg)
    for _ in range(3):cycle(w,l)
    w.age.fill_(cfg.maturity);w.energy[0,0]=100
    # Deliberately dirty the free slots to exercise reuse reset.
    l.hidden[1:]=2;l.obs_n[1:]=7;l.adam_count[1:]=12
    for m,v in l.moments.values():m[1:]=3;v[1:]=4
    action=torch.zeros(1,3,4,dtype=torch.long);action[0,0,3]=1
    t=w.step(action);l.inherit(t,w.ids)
    idx=int(t.births.flatten().nonzero()[0])
    for name,p in l.params.items():assert torch.equal(p[idx],l.genome[name][0])
    assert l.hidden[idx].count_nonzero()==0
    assert l.obs_n[idx]==l.adam_count[idx]==0
    for m,v in l.moments.values():assert m[idx].count_nonzero()==v[idx].count_nonzero()==0


def test_stream_head_updates_obey_per_element_bounds_and_terminal_bootstrap():
    cfg=Config(worlds=1,initial=1,capacity=2,size=12,hidden=8,radius=1,mode='L3')
    l=Learners(cfg)
    z=torch.ones(2,8);next_z=z*100
    action=l.onehot(torch.zeros(2,4,dtype=torch.long))
    probs=[torch.full((2,s),1/s) for s in cfg.action_sizes]
    active=torch.tensor([True,False]);terminal=torch.tensor([True,False])
    before={k:v.clone() for k,v in l.params.items()}
    delta=l._rl_update(z,next_z,action,probs,torch.ones(2),terminal,active)
    expected=1-(z[0]@before['value_w'][0]).item()-before['value_b'][0].item()
    assert delta[0].item()==pytest.approx(expected,abs=1e-6)
    for name in l.traces:
        rate=l.log_rates[0,1 if name.startswith('act') else 2].exp()
        assert (l.params[name][0]-before[name][0]).abs().max()<=rate+1e-7
        assert torch.equal(l.params[name][1],before[name][1])


def test_prediction_learns_repeated_synthetic_observations():
    cfg=Config(worlds=1,initial=1,capacity=1,size=12,hidden=8,radius=1,mode='L2',metabolism=0,move_cost=0,regrowth=0,initial_food=0)
    w,l=World(cfg),Learners(cfg)
    raw=torch.zeros(1,1,cfg.sensory_size)
    l.log_rates[:,0]=math.log(1e-3)
    losses=[]
    for _ in range(60):
        l.act(raw,w.ids,w.clock,w.alive)
        constant_action=l.onehot(torch.zeros(1,4,dtype=torch.long))
        l.cache=(*l.cache[:4],constant_action,l.cache[5])
        t=w.step(torch.zeros(1,1,4,dtype=torch.long))
        losses.append(float(l.learn(raw,t)['prediction_loss'].mean()))
    assert sum(losses[-10:])<sum(losses[:10])*.8


def test_death_slot_reuse_preserves_old_final_observation_and_clears_history():
    cfg=Config(worlds=1,initial=2,capacity=2,size=12,hidden=8,radius=1,mode='L3',regrowth=0,initial_food=0,obstacle_density=0)
    w,l=World(cfg),Learners(cfg)
    w.xy[0]=torch.tensor([[5,5],[5,6]])
    w.energy[0]=torch.tensor([100.,.01]);w.age.fill_(cfg.maturity)
    l.act(w.observe(),w.ids,w.clock,w.alive)
    action=torch.zeros(1,2,4,dtype=torch.long);action[0,0,3]=1
    t=w.step(action)
    assert t.terminal[0,1] and t.births[0,1]
    assert t.ids[0,1]==2 and w.ids[0,1]==3
    final=w.observe(t.positions,t.directions,t.energy)
    assert final[0,1,-1]==0 and w.observe()[0,1,-1]==.3
    l.learn(final,t);assert l.adam_count[1]==1
    l.inherit(t,w.ids)
    assert l.adam_count[1]==0 and l.obs_n[1]==0
    assert w.valid() and l.finite()
