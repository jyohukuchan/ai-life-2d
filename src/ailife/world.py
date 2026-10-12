from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import torch

from .config import Config
from .random import hash32, uniform
from .terrain import make_obstacles, fertility_field, preserve_food_total


@dataclasses.dataclass
class Transition:
    was_alive: torch.Tensor
    terminal: torch.Tensor
    positions: torch.Tensor
    directions: torch.Tensor
    energy: torch.Tensor
    ids: torch.Tensor
    births: torch.Tensor
    donor: torch.Tensor
    parent_a: torch.Tensor
    parent_b: torch.Tensor
    ate: torch.Tensor
    moved: torch.Tensor
    death_cause: torch.Tensor


class World:
    """Batched synchronous tile ecology. The world never returns a task reward."""

    DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, cfg: Config, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        w, n, h, v = cfg.worlds, cfg.capacity, cfg.height, cfg.width
        self.seeds = torch.arange(w, device=self.device, dtype=torch.int64) * 1009 + cfg.seed
        self.clock = torch.zeros((), dtype=torch.int64, device=self.device)
        self.slots = torch.arange(n, device=self.device).expand(w, n)
        self.delta = torch.tensor(self.DIRECTIONS, device=self.device)
        self.alive = self.slots < cfg.initial
        self.ids = torch.where(self.alive, self.slots + 1, 0)
        self.next_id = torch.full((w,), cfg.initial + 1, device=self.device, dtype=torch.int64)
        terrain=np.stack([make_obstacles(h,cfg.obstacle_density,cfg.seed+1009*i+29173,width=v) for i in range(w)])
        self.obstacles=torch.as_tensor(terrain,device=self.device)
        floor=np.ones((h,v),dtype=bool)
        floor[[0,-1],:]=False;floor[:,[0,-1]]=False
        floors=floor[None,:,:]&~terrain
        self.walkable=torch.as_tensor(floors,device=self.device)
        xx=np.arange(v)[None,:]
        food_region=np.broadcast_to((xx>=cfg.food_x_min*v)&(xx<cfg.food_x_max*v),(h,v))
        spawn_region=np.broadcast_to((xx>=cfg.spawn_x_min*v)&(xx<cfg.spawn_x_max*v),(h,v))
        self.food_zone=torch.as_tensor(floors&food_region,device=self.device)
        self.spawn_zone=torch.as_tensor(floors&spawn_region,device=self.device)
        self.xy = torch.zeros((w, n, 2), device=self.device, dtype=torch.int64)
        for i in range(w):
            rng = np.random.default_rng(cfg.seed + 1009 * i + 9973)
            spawn_cells=np.flatnonzero(floors[i]&spawn_region)
            if len(spawn_cells)<cfg.initial:raise ValueError('not enough free founder spawn cells')
            cells = rng.choice(spawn_cells, cfg.initial, replace=False)
            positions = np.stack((cells // v, cells % v), axis=-1)
            self.xy[i, :cfg.initial] = torch.as_tensor(positions, device=self.device)
        self.direction = (hash32(self._keys(1)) % 4).long()
        self.age = torch.zeros_like(self.ids)
        self.energy = self.alive.float() * cfg.founder_energy
        self.parents = torch.zeros((w, n, 2), device=self.device, dtype=torch.int64)
        self.generation = torch.zeros_like(self.ids)
        self.birth_tick = torch.zeros_like(self.ids)
        self.signal = torch.zeros_like(self.ids)
        self.extinction_tick = torch.full((w,), -1, device=self.device, dtype=torch.int64)
        self.eaten_total = torch.zeros((w,), device=self.device, dtype=torch.int64)
        self.births_total = torch.zeros((w, 2), device=self.device, dtype=torch.int64)
        self.deaths_total = torch.zeros((w, 2), device=self.device, dtype=torch.int64)
        self.capacity_rejected = torch.zeros((w,), device=self.device, dtype=torch.int64)
        self.capacity_ticks = torch.zeros((w,), device=self.device, dtype=torch.int64)
        fields = []
        for i in range(w):
            seed=cfg.seed+i*1009
            field=fertility_field(h,v,seed+1741,cfg.fertility,cfg.food_x_min,cfg.food_x_max)
            field=field*floors[i]*food_region
            if cfg.food_budget=='preserve_total':
                ref=cfg.food_reference_size or cfg.size
                reference=fertility_field(ref,ref,seed+1741,cfg.fertility)
                ref_floor=np.ones((ref,ref),dtype=bool)
                ref_floor[[0,-1],:]=False;ref_floor[:,[0,-1]]=False
                ref_floor &= ~make_obstacles(ref,cfg.obstacle_density,seed+29173)
                target=float((reference*ref_floor).sum())
                field=preserve_food_total(field,target,1/max(cfg.initial_food,cfg.regrowth,1e-12))
            fields.append(field)
        self.fertility = torch.as_tensor(np.stack(fields), device=self.device)
        self.grid_keys = self.seeds[:, None, None] * 1000003 + torch.arange(h*v, device=self.device).reshape(1, h, v)
        self.food = (uniform(self.grid_keys, 11) < cfg.initial_food * self.fertility) & self.food_zone
        self.offsets = torch.stack(torch.meshgrid(torch.arange(-cfg.radius, cfg.radius+1, device=self.device), torch.arange(-cfg.radius, cfg.radius+1, device=self.device), indexing="ij"), -1).reshape(-1, 2)
        if cfg.execution=='compiled':
            self.step=torch.compile(self.step,fullgraph=True)
            self.observe=torch.compile(self.observe,fullgraph=True)

    def _keys(self, stream):
        return self.seeds[:, None] * 1000003 + self.ids * 65537 + self.clock * 104729 + stream * 7919

    def _owner(self):
        c = self.cfg
        index = self.xy[..., 0] * c.width + self.xy[..., 1]
        index = torch.where(self.alive, index, 0)
        owner = torch.zeros((c.worlds, c.height*c.width), dtype=torch.int64, device=self.device)
        owner.scatter_(1, index, torch.where(self.alive, self.slots+1, 0))
        return owner

    def _neighbors(self, xy):
        c = self.cfg
        pos = xy[:, :, None, :] + self.delta[None, None, :, :]
        valid = (pos[..., 0] > 0) & (pos[..., 0] < c.height-1) & (pos[..., 1] > 0) & (pos[..., 1] < c.width-1)
        index = (pos[..., 0].clamp(0, c.height-1) * c.width + pos[..., 1].clamp(0, c.width-1)).long()
        valid &= self.walkable.flatten(1).gather(1,index.flatten(1)).reshape_as(index)
        return pos, index, valid

    def _winners(self, indices, candidates, priorities):
        c = self.cfg
        big = 1 << 62
        result = torch.full((c.worlds, c.height*c.width), big, device=self.device, dtype=torch.int64)
        result.scatter_reduce_(1, torch.where(candidates, indices, 0), torch.where(candidates, priorities, big), reduce="amin", include_self=True)
        return candidates & (priorities == result.gather(1, indices))

    @torch.no_grad()
    def step(self, actions):
        c = self.cfg
        was_alive, old_ids = self.alive.clone(), self.ids.clone()
        old_active = was_alive.any(1)
        key = self._keys(2)
        owner = self._owner()
        movement, mouth, signal, reproduction = actions.unbind(-1)
        self.direction = torch.where(self.alive, (self.direction + (movement == 3).long() - (movement == 2).long()) % 4, self.direction)
        proposed = self.xy + self.delta[self.direction]
        valid = (proposed[..., 0] > 0) & (proposed[..., 0] < c.height-1) & (proposed[..., 1] > 0) & (proposed[..., 1] < c.width-1)
        index = proposed[..., 0].clamp(0, c.height-1)*c.width + proposed[..., 1].clamp(0, c.width-1)
        valid &= self.walkable.flatten(1).gather(1,index)
        candidate = self.alive & (movement == 1) & valid & (owner.gather(1, index) == 0)
        priorities = hash32(key) * (c.capacity+1) + self.slots
        moved = self._winners(index, candidate, priorities)
        self.xy = torch.where(moved[..., None], proposed, self.xy)
        self.energy -= moved.float() * c.move_cost

        front = self.xy + self.delta[self.direction]
        valid = (front[..., 0] > 0) & (front[..., 0] < c.height-1) & (front[..., 1] > 0) & (front[..., 1] < c.width-1)
        index = front[..., 0].clamp(0, c.height-1)*c.width + front[..., 1].clamp(0, c.width-1)
        valid &= self.walkable.flatten(1).gather(1,index)
        candidate = self.alive & (mouth == 1) & valid & self.food.flatten(1).gather(1, index)
        ate = self._winners(index, candidate, hash32(key+10007)*(c.capacity+1)+self.slots)
        removal = torch.zeros_like(self.food.flatten(1), dtype=torch.int64)
        removal.scatter_add_(1, index, ate.long())
        self.food &= removal.reshape_as(self.food) == 0
        self.energy = (self.energy + ate.float()*c.food_energy).clamp_max(c.energy_max)
        self.eaten_total += ate.sum(1)
        self.age += self.alive.long()
        self.energy = (self.energy - self.alive.float()*c.metabolism).clamp_min(0)
        terminal = self.alive & ((self.energy <= 0) | (self.age >= c.lifespan))
        cause = torch.where(terminal & (self.energy <= 0), 1, torch.where(terminal, 2, 0))
        self.deaths_total[:, 0] += (cause == 1).sum(1)
        self.deaths_total[:, 1] += (cause == 2).sum(1)
        self.alive &= ~terminal
        # The old individual's final sensory location survives slot reuse.
        final_xy, final_direction, final_energy = self.xy.clone(), self.direction.clone(), self.energy.clone()
        owner = self._owner()
        neighbor_xy, neighbor_index, neighbor_valid = self._neighbors(self.xy)
        neighbor = (owner.gather(1, neighbor_index.flatten(1)).reshape_as(neighbor_index)-1)
        neighbor_slot = neighbor.clamp_min(0)
        cost = c.child_energy+c.birth_overhead
        mature = self.alive & (self.age >= c.maturity)
        sexual = mature & (reproduction == 2) & (self.energy >= cost/2)
        if c.reproduction == "clone":
            sexual = torch.zeros_like(sexual)
        neighbor_sexual = sexual.gather(1, neighbor_slot.flatten(1)).reshape_as(neighbor_slot)
        possible = sexual[..., None] & neighbor_sexual & (neighbor >= 0) & neighbor_valid
        a = torch.minimum(old_ids[..., None], old_ids.gather(1, neighbor_slot.flatten(1)).reshape_as(neighbor_slot))
        b = torch.maximum(old_ids[..., None], old_ids.gather(1, neighbor_slot.flatten(1)).reshape_as(neighbor_slot))
        edge_priority = hash32(self.seeds[:, None, None]*1000003 + self.clock*104729 + a*65537+b*97+31)
        chosen = torch.where(possible, edge_priority, 1 << 62).argmin(-1)
        mate = neighbor_slot.gather(-1, chosen[..., None]).squeeze(-1)
        mate = torch.where(possible.any(-1), mate, -1)
        mutual = (mate >= 0) & (mate.gather(1, mate.clamp_min(0)) == self.slots)
        sex_leader = sexual & mutual & (self.slots < mate)
        clone = mature & (reproduction == 1) & (self.energy >= cost)
        if c.reproduction == "sexual":
            clone = torch.zeros_like(clone)
        candidate = sex_leader | clone
        # The base-network donor is also the spatial anchor, equally likely per parent.
        provider = torch.where(sex_leader & (uniform(key, 35) < .5), mate.clamp_min(0), self.slots)
        birth_xy = neighbor_xy.gather(1, provider[..., None, None].expand(-1,-1,4,2))
        birth_index = neighbor_index.gather(1, provider[..., None].expand(-1,-1,4))
        valid_birth = neighbor_valid.gather(1, provider[..., None].expand(-1,-1,4))
        empty = valid_birth & self.spawn_zone.flatten(1).gather(1,birth_index.flatten(1)).reshape_as(birth_index) & (owner.gather(1, birth_index.flatten(1)).reshape_as(birth_index) == 0) & ~self.food.flatten(1).gather(1, birth_index.flatten(1)).reshape_as(birth_index)
        choices = hash32(key[..., None]+torch.arange(4,device=self.device)*1237+71)
        chosen = torch.where(empty, choices, 1 << 62).argmin(-1)
        target = birth_index.gather(-1, chosen[..., None]).squeeze(-1)
        position = birth_xy.gather(-2, chosen[..., None, None].expand(-1,-1,1,2)).squeeze(-2)
        if c.remote_births:
            free_cells=self.spawn_zone.flatten(1)&(owner==0)&~self.food.flatten(1)
            pool_size=free_cells.sum(1)
            cell_ids=torch.arange(c.height*c.width,device=self.device).expand(c.worlds,-1)
            pool=torch.where(free_cells,cell_ids,c.height*c.width).sort(1).values
            pick=torch.minimum((uniform(key,73)*pool_size[:,None]).long(),(pool_size[:,None]-1).clamp_min(0))
            target=pool.gather(1,pick).clamp_max(c.height*c.width-1)
            position=torch.stack((target//c.width,target%c.width),-1)
            candidate &= pool_size[:,None]>0
        else:
            candidate &= empty.any(-1)
        priority = hash32(key+11113)*(c.capacity+1)+self.slots
        won = self._winners(target, candidate, priority)
        free = ~self.alive
        free_count = free.sum(1)
        n_births = torch.minimum(won.sum(1), free_count)
        self.capacity_rejected += (won.sum(1)-n_births)
        order = torch.where(won, priority, 1 << 62).argsort(1)
        rank = free.long().cumsum(1)-1
        source = order.gather(1, rank.clamp(0,c.capacity-1))
        births = free & (rank < n_births[:, None])
        parent_a = source
        is_sex = sex_leader.gather(1, source)
        parent_b = torch.where(is_sex, mate.gather(1, source), -1)
        donor = provider.gather(1, source)
        child_positions = position.gather(1, source[..., None].expand(-1,-1,2))
        costs_a = torch.where(is_sex, cost/2, cost)*births.float()
        self.energy.scatter_add_(1, parent_a, -costs_a)
        self.energy.scatter_add_(1, parent_b.clamp_min(0), -(is_sex & births).float()*cost/2)
        self.births_total[:, 0] += (births & ~is_sex).sum(1)
        self.births_total[:, 1] += (births & is_sex).sum(1)
        child_ids = self.next_id[:, None] + rank
        child_gen = 1+torch.maximum(self.generation.gather(1,parent_a), torch.where(is_sex,self.generation.gather(1,parent_b.clamp_min(0)),0))
        pa_ids = old_ids.gather(1,parent_a)
        pb_ids = torch.where(is_sex,old_ids.gather(1,parent_b.clamp_min(0)),0)
        self.parents = torch.where(births[...,None],torch.stack((pa_ids,pb_ids),-1),self.parents)
        self.generation = torch.where(births,child_gen,self.generation)
        self.birth_tick = torch.where(births,self.clock+1,self.birth_tick)
        self.xy = torch.where(births[..., None],child_positions,self.xy)
        self.ids = torch.where(births,child_ids,self.ids)
        self.next_id += n_births
        self.age = torch.where(births,0,self.age)
        self.energy = torch.where(births,c.child_energy,self.energy)
        self.direction = torch.where(births,(hash32(self._keys(41)) % 4).long(),self.direction)
        self.alive |= births
        post_birth_death = self.alive & ~births & (self.energy <= 0)
        self.deaths_total[:, 0] += post_birth_death.sum(1)
        self.alive &= ~post_birth_death
        cause = torch.where(post_birth_death, 1, cause)
        final_energy = torch.where(was_alive & ~terminal, self.energy, final_energy)
        terminal |= post_birth_death
        self.signal = torch.where(self.alive & ~births,signal,0)
        self.capacity_ticks += (self.alive.sum(1) >= c.capacity).long()
        self.extinction_tick = torch.where((self.extinction_tick < 0) & old_active & ~self.alive.any(1),self.clock+1,self.extinction_tick)
        occupied = self._owner().reshape(c.worlds,c.height,c.width)>0
        growth = uniform(self.grid_keys+(self.clock+1)*104729,51) < c.regrowth*self.fertility
        self.food |= growth & ~occupied & self.food_zone & old_active[:,None,None]
        self.clock += 1
        return Transition(was_alive,terminal,final_xy,final_direction,final_energy,old_ids,births,donor,parent_a,parent_b,ate,moved,cause)

    def observe(self, xy=None, direction=None, energy=None):
        """Local RGB/sound/energy, then optional policy-only compass channels."""
        c = self.cfg
        xy = self.xy if xy is None else xy
        direction = self.direction if direction is None else direction
        energy = self.energy if energy is None else energy
        owner = self._owner().reshape(c.worlds,c.height,c.width)
        rgb = torch.zeros((c.worlds,c.height,c.width,3),device=self.device)
        rgb[...,0] = .12; rgb[...,1] = .16; rgb[...,2] = .14
        rgb = torch.where(self.food[...,None],torch.tensor([.3,.8,.2],device=self.device),rgb)
        rgb = torch.where((owner>0)[...,None],torch.tensor([.25,.55,1.],device=self.device),rgb)
        rgb = torch.where(self.walkable[...,None],rgb,torch.tensor([.45,.48,.52],device=self.device))
        oy, ox = self.offsets.unbind(-1)
        d = direction[...,None]
        dy = torch.where(d==0,oy,torch.where(d==1,ox,torch.where(d==2,-oy,-ox)))
        dx = torch.where(d==0,ox,torch.where(d==1,-oy,torch.where(d==2,-ox,oy)))
        yy, xx = xy[...,0,None]+dy, xy[...,1,None]+dx
        inside = (yy>=0)&(xx>=0)&(yy<c.height)&(xx<c.width)
        index = yy.clamp(0,c.height-1)*c.width+xx.clamp(0,c.width-1)
        flat = rgb.flatten(1,2)
        pixels = flat.gather(1,index.flatten(1)[...,None].expand(-1,-1,3)).reshape(c.worlds,c.capacity,-1,3)
        pixels = torch.where(inside[...,None],pixels,0)
        parts = [pixels.flatten(2)]
        if c.symbols:
            dist = (xy[:,:,None,:]-self.xy[:,None,:,:]).abs().sum(-1)
            valid = (dist<=c.sound_radius)&self.alive[:,None,:]&(self.signal[:,None,:]>0)&(self.ids[:,None,:]!=self.ids[:,:,None])
            gain = valid.float()/(dist.float()+1)
            encoded = torch.nn.functional.one_hot(self.signal.clamp(0,c.symbols),c.symbols+1)[...,1:].float()
            sound = torch.bmm(gain,encoded).clamp_max(1)
            parts.append(torch.log1p(sound))
        if c.interoception:
            parts.append((energy/c.energy_max)[...,None])
        if c.compass_size:
            compass=torch.nn.functional.one_hot(direction.long(),4).float() if c.compass=="absolute" else torch.zeros((*direction.shape,4),device=self.device)
            parts.append(compass)
        return torch.cat(parts,-1)

    def state_dict(self):
        return {k:v.detach().cpu().clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor)}

    def valid(self):
        c=self.cfg
        inside=(self.xy[...,0]>0)&(self.xy[...,0]<c.height-1)&(self.xy[...,1]>0)&(self.xy[...,1]<c.width-1)
        index=self.xy[...,0].clamp(0,c.height-1)*c.width+self.xy[...,1].clamp(0,c.width-1)
        inside &= self.walkable.flatten(1).gather(1,index)
        body=(self.energy>0)&(self.energy<=c.energy_max)&(self.age<c.lifespan)&inside
        occupied=(self._owner()>0).sum(1)
        checks=torch.stack((torch.isfinite(self.energy).all(),((~self.alive)|body).all(),(occupied==self.alive.sum(1)).all(),((~self.food)|self.food_zone).all()))
        return bool(checks.all().item())

    def load_state_dict(self,state):
        for k,v in state.items(): setattr(self,k,v.to(self.device).clone())
        if self.walkable.ndim==2:
            self.walkable=self.walkable[None].expand(self.cfg.worlds,-1,-1).clone()
        if 'obstacles' not in state:self.obstacles.zero_()

    def digest(self):
        h=hashlib.sha256()
        for k,v in sorted(self.state_dict().items()):
            h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
        return h.hexdigest()

    def snapshot(self,world=0):
        c=self.cfg
        fields={k:getattr(self,k)[world].detach().cpu().numpy() for k in ['xy','direction','energy','age','alive','ids','parents','generation','signal','food','fertility','obstacles']}
        ids=np.flatnonzero(fields['alive'])
        agents=[{'slot':int(i),'id':int(fields['ids'][i]),'y':int(fields['xy'][i,0]),'x':int(fields['xy'][i,1]),'direction':int(fields['direction'][i]),'energy':round(float(fields['energy'][i]),3),'age':int(fields['age'][i]),'generation':int(fields['generation'][i]),'parents':fields['parents'][i].tolist(),'signal':int(fields['signal'][i])} for i in ids]
        return {'world':world,'tick':int(self.clock.item()),'size':c.size,'height':c.height,'width':c.width,'food':fields['food'].astype(np.uint8).flatten().tolist(),'fertility':fields['fertility'].round(3).flatten().tolist(),'obstacles':fields['obstacles'].astype(np.uint8).flatten().tolist(),'agents':agents,'extinction_tick':int(self.extinction_tick[world].item()),'config':c.as_dict()}
