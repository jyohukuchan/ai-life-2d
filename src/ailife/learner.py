from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import Config
from .random import normal, uniform


def layer_norm(x,g,b):
    centered=x-x.mean(-1,keepdim=True)
    return centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)*g+b


def linear(x,w,b):
    return torch.bmm(x.unsqueeze(1),w).squeeze(1).float()+b


def private_adam(params,grads,moments,counts,rates,active,loss_scale=1.0,skipped=None):
    """Bias correction is per individual, never the Parameter tensor's age."""
    if skipped is not None:
        grads=tuple(g.float()/loss_scale for g in grads)
        valid=torch.stack([g.isfinite().flatten(1).all(1) for g in grads]).all(0)
        skipped.add_((active & ~valid).long())
        active=active & valid
        grads=tuple(torch.where(valid.reshape((-1,)+(1,)*(g.ndim-1)),g,0) for g in grads)
    counts.add_(active.long())
    norm=sum(g.square().flatten(1).sum(1) for g in grads).sqrt()
    clip=(1/(norm+1e-12)).clamp_max(1)
    t=counts.clamp_min(1)
    for (name,p),g in zip(params.items(),grads):
        shape=(-1,)+(1,)*(p.ndim-1)
        mask=active.reshape(shape)
        g=g*clip.reshape(shape)
        m,v=moments[name]
        m.copy_(torch.where(mask,.9*m+.1*g,m))
        v.copy_(torch.where(mask,.999*v+.001*g.square(),v))
        mh=m/(1-.9**t.float()).reshape(shape)
        vh=v/(1-.999**t.float()).reshape(shape)
        update=rates.reshape(shape)*mh/(vh.sqrt()+1e-8)
        p.sub_(torch.where(mask,update,0))


def sensory_core(raw,previous_hidden,previous_action,p,obs_n,obs_mean,obs_m2,sensory_size,dtype):
    base=raw[:,:sensory_size]
    mean=torch.where(obs_n>=2,obs_mean,0)
    var=obs_m2/(obs_n-1).clamp_min(1)
    scale=torch.where(obs_n>=2,var.clamp_min(0).sqrt().clamp_min(.01),1)
    normalized=((base-mean)/scale).clamp(-5,5)
    x=torch.cat((normalized,previous_action,raw[:,sensory_size:]),-1)
    with torch.autocast(device_type=raw.device.type,dtype=dtype,enabled=dtype!=torch.float32):
        enc=F.leaky_relu(layer_norm(linear(x,p['enc_w'],p['enc_b']),p['enc_g'],p['enc_n']),.01)
        ir,iz,inn=linear(enc,p['gru_iw'],p['gru_ib']).chunk(3,-1)
        hr,hz,hn=linear(previous_hidden,p['gru_hw'],p['gru_hb']).chunk(3,-1)
        reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz)
        candidate=torch.tanh(inn+reset*hn)
        hidden=(1-update)*candidate+update*previous_hidden
        pre=layer_norm(hidden,p['core_g'],p['core_n'])
        z=torch.tanh(linear(pre,p['feat_w'],p['feat_b']))
    return hidden,pre,z


def choose_actions(z,weight,bias,seeds,ids,clock,alive,sizes):
    logits=linear(z,weight,bias)
    choices=[];probs=[]
    key=seeds*1000003+ids.flatten()*65537+clock*104729
    for j,part in enumerate(logits.split(sizes,-1)):
        prob=part.softmax(-1)
        u=uniform(key,601+j*17)
        choice=(u[:,None]>prob.cumsum(-1)).sum(-1).clamp_max(part.shape[-1]-1)
        choices.append(choice);probs.append(prob)
    actions=torch.stack(choices,-1)
    actions=torch.where(alive.flatten()[:,None],actions,0)
    onehot=torch.cat([F.one_hot(actions[:,j],size) for j,size in enumerate(sizes)],-1).float()
    return actions,onehot,probs


def prediction_loss(z,action,params,target,visual,symbols,interoception,dtype):
    with torch.autocast(device_type=z.device.type,dtype=dtype,enabled=dtype!=torch.float32):
        prediction=linear(torch.cat((z,action),-1),params['pred_w'],params['pred_b'])
    square=(prediction-target).square()
    parts=[square[:,:visual].mean(-1)]
    if symbols:parts.append(square[:,visual:visual+symbols].mean(-1))
    if interoception:parts.append(square[:,-1])
    return torch.stack(parts,-1).mean(-1)


class Learners:
    """A separate genome, recurrent state and optimizer for every population slot."""

    def __init__(self,cfg:Config,device="cpu"):
        self.cfg,self.device=cfg,torch.device(device)
        if self.device.type=='cuda' and cfg.execution=='eager' and cfg.precision!='fp32' and cfg.mode in {'L2','L3'}:
            arch=getattr(torch.cuda.get_device_properties(self.device),'gcnArchName','').split(':')[0]
            if arch=='gfx1201' and str(torch.__version__).startswith('2.12.0+rocm7.14'):
                raise ValueError('Eager FP16/BF16 training reproducibly faults in BLAS on gfx1201 with this Torch/ROCm build; use --execution compiled or FP32.')

        self.count=cfg.worlds*cfg.capacity
        a,h,d,e=self.count,cfg.hidden,cfg.sensory_size+cfg.action_size+cfg.compass_size,cfg.encoder_width
        self.world_seed=(torch.arange(cfg.worlds,device=self.device)*1009+cfg.seed).repeat_interleave(cfg.capacity)
        ids=torch.arange(1,cfg.capacity+1,device=self.device).repeat(cfg.worlds)
        self.params={};self.genome={};self.scales={};self.tags={}
        shapes={
            'enc_w':(d,e),'enc_b':(e,),'enc_g':(e,),'enc_n':(e,),
            'gru_iw':(e,3*h),'gru_hw':(h,3*h),'gru_ib':(3*h,),'gru_hb':(3*h,),
            'core_g':(h,),'core_n':(h,),'feat_w':(h,h),'feat_b':(h,),
            'act_w':(h,cfg.action_size),'act_b':(cfg.action_size,),
            'value_w':(h,1),'value_b':(1,),
            'pred_w':(h+cfg.action_size,cfg.sensory_size),'pred_b':(cfg.sensory_size,),
        }
        pred_names={'enc_w','enc_b','enc_g','enc_n','gru_iw','gru_hw','gru_ib','gru_hb','core_g','core_n','pred_w','pred_b'}
        self.pred_names=tuple(name for name in shapes if name in pred_names)
        for tag,(name,shape) in enumerate(shapes.items()):
            self.tags[name]=100+tag*7
            fan=shape[0] if len(shape)==2 else (d if name.startswith('enc') else e if name=='gru_ib' else h)
            scale=1/math.sqrt(fan)
            self.scales[name]=1.0 if name.endswith(('_g','_n')) else scale
            key=self._parameter_keys(ids,shape,tag*7+100)
            data=normal(key,tag*7+100)*scale
            if len(shape)==1:data=torch.ones_like(data) if name.endswith('_g') else torch.zeros_like(data)
            self.genome[name]=data.clone()
            self.params[name]=torch.nn.Parameter(data,requires_grad=cfg.mode in {'L2','L3'} and name in pred_names)
        self.log_rates=torch.tensor([math.log(1e-4),math.log(1e-4),math.log(3e-4),math.log(1e-3)],device=self.device).expand(a,4).clone()
        self.genome_rates=self.log_rates.clone()
        self.eval_w=normal(self._parameter_keys(ids,(4+cfg.symbols+cfg.action_size,),501),501)*.01
        self.genome_eval=self.eval_w.clone()
        self.hidden=torch.zeros(a,h,device=self.device)
        self.last_action=torch.zeros(a,cfg.action_size,device=self.device)
        self.obs_n=torch.zeros(a,1,device=self.device)
        self.obs_mean=torch.zeros(a,cfg.sensory_size,device=self.device)
        self.obs_m2=torch.zeros_like(self.obs_mean)
        # Birth-time input calibration belongs to the inherited model. Legacy genomes use zeros.
        self.normalization_genome={name:torch.zeros_like(getattr(self,name)) for name in ("obs_n","obs_mean","obs_m2")}
        self.reward_u=torch.zeros(a,device=self.device)
        self.reward_n=torch.zeros(a,device=self.device)
        self.reward_mean=torch.zeros(a,device=self.device)
        self.reward_m2=torch.zeros(a,device=self.device)
        self.adam_count=torch.zeros(a,device=self.device,dtype=torch.int64)
        self.moments={n:(torch.zeros_like(self.params[n]),torch.zeros_like(self.params[n])) for n in self.pred_names} if cfg.mode in {'L2','L3'} else {}
        self.traces={n:torch.zeros_like(self.params[n]) for n in ['act_w','act_b','value_w','value_b']} if cfg.mode=='L3' else {}
        self.bounds={n:torch.zeros_like(x) for n,x in self.traces.items()}
        self.cache=None
        self.skipped_prediction_updates=torch.zeros_like(self.adam_count)
        self.compute_dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[cfg.precision]
        compile_kernel=lambda f:torch.compile(f,fullgraph=True) if cfg.execution!='eager' else f
        self._core_kernel=compile_kernel(sensory_core)
        self._action_kernel=compile_kernel(choose_actions)
        self._loss_kernel=compile_kernel(prediction_loss)
        self._adam_kernel=compile_kernel(private_adam)
        if cfg.execution!='eager' and cfg.mode=='L3':
            self._reward=torch.compile(self._reward,fullgraph=True)
            self._rl_update=torch.compile(self._rl_update,fullgraph=True)

    def _parameter_keys(self,ids,shape,tag,world_seed=None):
        seed=self.world_seed if world_seed is None else world_seed
        base=seed*1000003+ids*65537+tag*7919
        coords=torch.arange(math.prod(shape),device=self.device).reshape((1,)+shape)
        return base.reshape((-1,)+(1,)*len(shape))+coords*97

    def _normalized(self,raw):
        n=self.obs_n
        mean=torch.where(n>=2,self.obs_mean,0)
        var=self.obs_m2/(n-1).clamp_min(1)
        scale=torch.where(n>=2,var.clamp_min(0).sqrt().clamp_min(.01),1)
        return ((raw-mean)/scale).clamp(-5,5)

    def _core(self,raw,previous_hidden,previous_action):
        return self._core_kernel(raw,previous_hidden,previous_action,self.params,self.obs_n,self.obs_mean,self.obs_m2,self.cfg.sensory_size,self.compute_dtype)

    def act(self,raw,ids,clock,alive):
        c=self.cfg
        raw=raw.reshape(self.count,-1)
        with torch.set_grad_enabled(c.mode in {'L2','L3'}):
            h,pre,z=self._core(raw,self.hidden.detach(),self.last_action)
        with torch.no_grad():
            actions,onehot,probs=self._action_kernel(z.detach(),self.params['act_w'],self.params['act_b'],self.world_seed,ids,clock,alive,c.action_sizes)
        self.cache=(raw,h,pre,z,onehot,probs)
        return actions.reshape(c.worlds,c.capacity,4)

    def onehot(self,actions):
        return torch.cat([F.one_hot(actions[:,j],size) for j,size in enumerate(self.cfg.action_sizes)],-1).float()

    @torch.no_grad()
    def _reward(self,raw,next_raw,action,active):
        c=self.cfg; visual=(2*c.radius+1)**2*3
        raw=raw[:,:c.sensory_size];next_raw=next_raw[:,:c.sensory_size]
        old_rgb=raw[:,:visual].reshape(self.count,-1,3).mean(1)
        new_rgb=next_raw[:,:visual].reshape(self.count,-1,3).mean(1)
        de=(next_raw[:,-1]-raw[:,-1]) if c.interoception else torch.zeros(self.count,device=self.device)
        phi=torch.cat((de[:,None],new_rgb-old_rgb,next_raw[:,visual:visual+c.symbols],action),-1)
        terms=self.eval_w*phi
        raw_reward=torch.tanh(terms.sum(-1))
        u=.99*self.reward_u+raw_reward
        n=self.reward_n+active.float()
        delta=u-self.reward_mean
        mean=self.reward_mean+delta/n.clamp_min(1)
        m2=self.reward_m2+delta*(u-mean)
        self.reward_u.copy_(torch.where(active,u,self.reward_u))
        self.reward_mean.copy_(torch.where(active,mean,self.reward_mean))
        self.reward_m2.copy_(torch.where(active,m2,self.reward_m2))
        self.reward_n.copy_(n)
        scale=torch.where(n>=2,(self.reward_m2/(n-1).clamp_min(1)).clamp_min(0).sqrt().clamp_min(.01),1)
        scaled=raw_reward/scale
        self.reward_diagnostics={'phi':phi,'terms':terms,'raw_reward':raw_reward,'scaled_reward':scaled,'scale':scale,'energy_coefficient':self.eval_w[:,0].clone()} if c.log_reward_components else None
        return raw_reward,scaled

    @torch.no_grad()
    def _rl_update(self,z,next_z,action,probs,reward,terminal,active):
        p=self.params
        value=linear(z,p['value_w'],p['value_b']).squeeze(-1)
        future=linear(next_z,p['value_w'],p['value_b']).squeeze(-1)
        delta=reward+.99*torch.where(terminal,0,future)-value
        grads=[];offset=0
        for prob,size in zip(probs,self.cfg.action_sizes):
            logp=prob.clamp_min(1e-12).log()
            entropy=-(prob*logp).sum(-1,keepdim=True)
            grad_entropy=-prob*(logp+entropy)
            grads.append(action[:,offset:offset+size]-prob+.01*delta.sign()[:,None]*grad_entropy)
            offset+=size
        logits_grad=torch.cat(grads,-1)
        gradients={'act_w':z[:,:,None]*logits_grad[:,None,:],'act_b':logits_grad,'value_w':z[:,:,None],'value_b':torch.ones_like(p['value_b'])}
        for name,gradient in gradients.items():
            shape=(-1,)+(1,)*(gradient.ndim-1)
            mask=active.reshape(shape)
            e=self.traces[name];bound=self.bounds[name]
            e.copy_(torch.where(mask,.99*.8*e+gradient,e))
            increment=delta.reshape(shape)*e
            bound.copy_(torch.where(mask,torch.maximum(.99995*bound,increment.abs()),bound))
            rate=self.log_rates[:,1 if name.startswith('act') else 2].exp().reshape(shape)
            p[name].add_(torch.where(mask,rate*increment/(bound+1e-8),0))
        return delta

    def learn(self,next_raw,transition):
        raw,h,pre,z,action,probs=self.cache
        active=transition.was_alive.flatten()
        terminal=transition.terminal.flatten()
        next_raw=next_raw.reshape(self.count,-1)
        c=self.cfg
        loss_by_agent=torch.zeros(self.count,device=self.device)
        reward=torch.zeros_like(loss_by_agent)
        if c.mode=='L3':
            with torch.no_grad():
                _,_,next_z=self._core(next_raw,h.detach(),action)
                reward,scaled=self._reward(raw,next_raw,action,active)
                self._rl_update(z.detach(),next_z,action,probs,scaled,terminal,active)
        if c.mode in {'L2','L3'}:
            loss_by_agent=self._loss_kernel(z,action,self.params,next_raw[:,:c.sensory_size],(2*c.radius+1)**2*3,c.symbols,c.interoception,self.compute_dtype)
            params={n:self.params[n] for n in self.pred_names}
            loss_scale=c.fp16_loss_scale if c.precision=='fp16' else 1.0
            grads=torch.autograd.grad((loss_by_agent*active).sum()*loss_scale,tuple(params.values()))
            with torch.no_grad():self._adam_kernel(params,grads,self.moments,self.adam_count,self.log_rates[:,0].exp(),active,loss_scale,self.skipped_prediction_updates if c.precision=='fp16' else None)
        elif c.mode=='L1':
            with torch.no_grad():
                a=self.params['feat_w']
                # Stored matrices have [input, output] orientation.
                update=pre[:,:,None]*z[:,None,:]-a*z[:,None,:].square()
                candidate=a+self.log_rates[:,3].exp()[:,None,None]/c.hidden*update
                norm=candidate.square().sum(1,keepdim=True).sqrt().clamp_min(1e-12)
                candidate=candidate*(2/norm).clamp_max(1)
                a.copy_(torch.where(active[:,None,None],candidate,a))
        with torch.no_grad():
            n=self.obs_n+active[:,None]
            difference=raw[:,:c.sensory_size]-self.obs_mean
            mean=self.obs_mean+difference/n.clamp_min(1)
            self.obs_m2.copy_(torch.where(active[:,None],self.obs_m2+difference*(raw[:,:c.sensory_size]-mean),self.obs_m2))
            self.obs_mean.copy_(torch.where(active[:,None],mean,self.obs_mean))
            self.obs_n.copy_(n)
            keep=active & ~terminal
            self.hidden.copy_(torch.where(keep[:,None],h.detach(),0))
            self.last_action.copy_(torch.where(keep[:,None],action,0))
        self.cache=None
        result={'prediction_loss':loss_by_agent.detach().reshape(c.worlds,c.capacity),'intrinsic_reward':reward.reshape(c.worlds,c.capacity)}
        if c.mode=='L3' and c.log_reward_components:
            result['reward_diagnostics']=self.reward_diagnostics
            self.reward_diagnostics=None
        return result

    @torch.no_grad()
    def inherit(self,transition,new_ids):
        mask=transition.births.flatten()
        idx=mask.nonzero().flatten()
        if not idx.numel():return 0
        n=self.cfg.capacity
        world_offset=(idx//n)*n
        donor=transition.donor.flatten()[idx]+world_offset
        pa=transition.parent_a.flatten()[idx]+world_offset
        pb=transition.parent_b.flatten()[idx]
        sexual=pb>=0
        pb=pb.clamp_min(0)+world_offset
        ids=new_ids.flatten()[idx]
        seeds=self.world_seed[idx]
        for name,g in self.genome.items():
            shape=g.shape[1:]
            key=self._parameter_keys(ids,shape,self.tags[name],seeds)
            child=g[donor]+self.cfg.mutation*self.scales[name]*normal(key,self.tags[name]+9)
            if name.endswith(('_g','_n')):child=child.clamp(-5,5)
            g[idx]=child;self.params[name][idx]=child
        rate_key=self._parameter_keys(ids,(4,),701,seeds)
        choose=(uniform(rate_key,701)<.5)&sexual[:,None]
        rates=torch.where(choose,self.genome_rates[pb],self.genome_rates[pa])
        rates+=self.cfg.mutation*normal(rate_key,703)
        lower=torch.full_like(rates,math.log(1e-6))
        upper=torch.tensor([math.log(1e-2)]*3+[math.log(1e-1)],device=self.device)
        rates=torch.maximum(lower,torch.minimum(upper,rates))
        self.genome_rates[idx]=rates;self.log_rates[idx]=rates
        shape=(self.eval_w.shape[-1],)
        key=self._parameter_keys(ids,shape,711,seeds)
        choose=(uniform(key,711)<.5)&sexual[:,None]
        coeff=torch.where(choose,self.genome_eval[pb],self.genome_eval[pa])
        coeff=(coeff+self.cfg.mutation*.1*normal(key,713)).clamp(-2,2)
        self.genome_eval[idx]=coeff;self.eval_w[idx]=coeff
        for name,baseline in self.normalization_genome.items():
            inherited=baseline[donor].clone()
            baseline[idx]=inherited;getattr(self,name)[idx]=inherited
        for name in ['hidden','last_action','reward_u','reward_n','reward_mean','reward_m2','adam_count','skipped_prediction_updates']:
            getattr(self,name)[idx]=0
        for m,v in self.moments.values():m[idx]=0;v[idx]=0
        for e in self.traces.values():e[idx]=0
        for b in self.bounds.values():b[idx]=0
        return int(idx.numel())

    def finite(self):
        tensors=list(self.params.values())+list(self.genome.values())+[self.hidden,self.log_rates,self.eval_w,self.obs_n,self.obs_mean,self.obs_m2,self.reward_u,self.reward_n,self.reward_mean,self.reward_m2]
        tensors += list(self.normalization_genome.values())
        tensors += [v for pair in self.moments.values() for v in pair]+list(self.traces.values())+list(self.bounds.values())
        return bool(torch.stack([torch.isfinite(p).all() for p in tensors]).all().item())

    def parameter_count(self):
        return sum(p[0].numel() for p in self.params.values())+4+self.eval_w.shape[1]

    def state_dict(self):
        def clone(x):
            if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
            if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
            if isinstance(x,tuple):return tuple(clone(v) for v in x)
            return x
        keys=['skipped_prediction_updates','normalization_genome','params','genome','log_rates','genome_rates','eval_w','genome_eval','hidden','last_action','obs_n','obs_mean','obs_m2','reward_u','reward_n','reward_mean','reward_m2','adam_count','moments','traces','bounds']
        return {k:clone(getattr(self,k)) for k in keys}

    @torch.no_grad()
    def load_state_dict(self,state):
        def copy(dest,source):
            if isinstance(dest,torch.Tensor):dest.copy_(source.to(self.device))
            elif isinstance(dest,dict):
                for k in dest:copy(dest[k],source[k])
            elif isinstance(dest,tuple):
                for d,s in zip(dest,source):copy(d,s)
        for key,value in state.items():copy(getattr(self,key),value)
        self.cache=None
