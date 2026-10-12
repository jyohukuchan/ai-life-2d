import dataclasses
import pytest
import torch
from ailife.config import Config
from ailife.learner import Learners,private_adam
from ailife.runner import iteration
from ailife.world import World


@pytest.mark.parametrize('precision',['fp32','bf16','fp16'])
def test_mixed_compute_keeps_private_master_and_body_state_fp32(precision):
 torch.set_num_threads(1)
 cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,precision=precision)
 w,l=World(cfg),Learners(cfg);before=l.state_dict()
 for _ in range(8):iteration(w,l)
 assert w.energy.dtype==torch.float32 and l.hidden.dtype==torch.float32
 for n,p in l.params.items():
  assert p.dtype==torch.float32
  assert torch.equal(p[2:],before['params'][n][2:])
 assert l.adam_count.tolist()==[8,8,0,0]
 assert not l.skipped_prediction_updates.any() and l.finite() and w.valid()
 assert (l.params['enc_g'][:2]-before['params']['enc_g'][:2]).abs().sum()>0
 for tensor in [l.obs_n,l.obs_mean,l.obs_m2,l.reward_mean,l.reward_m2,*l.traces.values(),*l.bounds.values()]:assert tensor.dtype==torch.float32
 restored=Learners(cfg);restored.load_state_dict(l.state_dict())
 assert torch.equal(restored.params['enc_w'],l.params['enc_w'])


@pytest.mark.parametrize('loss_scale',[1.,128.])
def test_fp16_overflow_is_handled_per_individual_not_across_population(loss_scale):
 params={'w':torch.ones(3,2)};moments={'w':(torch.zeros(3,2),torch.zeros(3,2))}
 counts=torch.zeros(3,dtype=torch.long);skipped=torch.zeros_like(counts)
 grads=(torch.tensor([[128.,256.],[float('inf'),128.],[0.,0.]]),)
 private_adam(params,grads,moments,counts,torch.ones(3)*1e-4,torch.tensor([True,True,False]),loss_scale,skipped)
 assert counts.tolist()==[1,0,0] and skipped.tolist()==[0,1,0]
 assert params['w'][0].max()<1 and params['w'][1:].eq(1).all()
 assert torch.isfinite(params['w']).all()


@pytest.mark.parametrize('precision',['fp16','bf16'])
def test_known_r9700_eager_training_fault_is_rejected_before_allocation(monkeypatch,precision):
 from types import SimpleNamespace
 monkeypatch.setattr(torch,'__version__','2.12.0+rocm7.14.0')
 monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(gcnArchName='gfx1201'))
 with pytest.raises(ValueError,match='use --execution compiled'):
  Learners(Config(precision=precision,execution='eager'),'cuda')
