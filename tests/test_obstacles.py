from collections import deque

import numpy as np
import pytest
import torch

from ailife.config import Config
from ailife.terrain import make_obstacles
from ailife.world import World
from test_world import setup


def reachable_count(floor):
    start=tuple(np.argwhere(floor)[0])
    seen={start};pending=deque([start])
    while pending:
        y,x=pending.popleft()
        for dy,dx in World.DIRECTIONS:
            neighbor=(y+dy,x+dx)
            if floor[neighbor] and neighbor not in seen:
                seen.add(neighbor);pending.append(neighbor)
    return len(seen)


@pytest.mark.parametrize('size,density',[(7,.04),(9,.2),(12,.04),(20,.2),(57,.04)])
def test_sparse_terrain_is_repeatable_and_all_floor_is_connected(size,density):
    for seed in range(3):
        obstacles=make_obstacles(size,density,seed)
        assert int(obstacles.sum())==round((size-2)**2*density)
        assert np.array_equal(obstacles,make_obstacles(size,density,seed))
        assert not obstacles[[0,1,-2,-1],:].any()
        assert not obstacles[:,[0,1,-2,-1]].any()
        floor=~obstacles
        floor[[0,-1],:]=False;floor[:,[0,-1]]=False
        assert reachable_count(floor)==int(floor.sum())


def test_obstacles_block_movement_births_and_appear_in_rotated_vision():
    w=setup();w.direction[0,0]=1;w.energy[0,0]=100
    w.obstacles[0,5,6]=True;w.walkable[0,5,6]=False
    action=torch.zeros(1,4,4,dtype=torch.long);action[0,0,0]=1
    t=w.step(action)
    assert not t.moved.any()
    assert w.xy[0,0].tolist()==[5,5] and w.energy[0,0]==100
    c=w.cfg;r=c.radius
    pixels=w.observe()[0,0,:3*(2*r+1)**2].reshape(2*r+1,2*r+1,3)
    torch.testing.assert_close(pixels[r-1,r],torch.tensor([.45,.48,.52]))
    for dy,dx in World.DIRECTIONS:
        w.obstacles[0,5+dy,5+dx]=True;w.walkable[0,5+dy,5+dx]=False
    action.zero_();action[0,0,3]=1
    t=w.step(action)
    assert not t.births.any() and w.energy[0,0]==100


def test_founders_food_and_children_stay_off_obstacles():
    cfg=Config(worlds=2,initial=8,capacity=32,size=20,obstacle_density=.15,initial_food=1,regrowth=1,metabolism=0,move_cost=0,founder_energy=100)
    w=World(cfg);w.age.fill_(cfg.maturity)
    assert not (w.food&w.obstacles).any()
    w.food.zero_()
    action=torch.zeros(2,32,4,dtype=torch.long);action[:,:8,3]=1
    t=w.step(action)
    assert t.births.any() and w.valid()
    for _ in range(5):w.step(torch.zeros_like(action))
    assert not (w.food&w.obstacles).any()
    assert not w.fertility[w.obstacles].any()
    assert w.valid()
    assert w.obstacles[0].any() and not torch.equal(w.obstacles[0],w.obstacles[1])


def test_old_saved_world_is_restored_without_adding_obstacles():
    cfg=Config(worlds=2,initial=2,capacity=4,size=12,obstacle_density=0)
    w=World(cfg);state=w.state_dict()
    state.pop('obstacles');state['walkable']=state['walkable'][0]
    saved=cfg.as_dict();saved.pop('obstacle_density')
    restored_cfg=Config.from_saved(saved)
    assert restored_cfg.obstacle_density==0
    restored=World(restored_cfg);restored.load_state_dict(state)
    assert restored.digest()==w.digest()
    action=torch.zeros(2,4,4,dtype=torch.long)
    w.step(action);restored.step(action)
    assert restored.digest()==w.digest()


def test_obstacle_density_validation_and_disabled_condition():
    with pytest.raises(ValueError):Config(obstacle_density=.3)
    with pytest.raises(ValueError):Config(size=12,initial=1,capacity=100,obstacle_density=.04)
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,obstacle_density=0)
    assert not World(cfg).obstacles.any()
