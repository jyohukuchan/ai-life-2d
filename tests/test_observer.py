import dataclasses
import json

import numpy as np
from fastapi.testclient import TestClient

from ailife.config import Config
from ailife.learner import Learners
from ailife.observer import Observer, atomic_json, replay_schedule
from ailife.runner import iteration
from ailife.server import create_app
from ailife.world import World


def test_replay_sampling_is_preselected_repeatable_and_not_biased_by_survival():
    cfg=Config(worlds=2,initial=2,capacity=4,size=12,hidden=8,radius=1,ticks=60,replay_count=3,replay_length=8)
    a=replay_schedule(cfg)
    assert a==replay_schedule(cfg)
    assert a!=replay_schedule(dataclasses.replace(cfg,observer_seed=22))
    assert all(0<=c['world']<2 and 0<=c['start']<=53 and c['length']==8 for c in a)


def test_observing_never_changes_simulation_or_learning_and_server_is_read_only(tmp_path):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12,hidden=8,radius=1,ticks=12,replay_count=1,replay_length=5,publish_seconds=0)
    w1,l1=World(cfg),Learners(cfg);w2,l2=World(cfg),Learners(cfg)
    root=tmp_path/'runs';directory=root/'example'
    observer=Observer(cfg,directory)
    atomic_json(directory/'config.json',cfg.as_dict());atomic_json(root/'active.json',{'run':'example'})
    observer.capture(w1,0)
    for tick in range(1,13):
        iteration(w1,l1);iteration(w2,l2)
        observer.capture(w1,tick);observer.publish(w1,tick)
    observer.close(w1,12,'complete')
    assert w1.digest()==w2.digest()
    for name in l1.params:assert (l1.params[name]==l2.params[name]).all()
    client=TestClient(create_app(root))
    assert client.get('/').status_code==200
    assert client.get('/api/latest').json()['tick']==12
    assert client.post('/api/latest',json={'pause':True}).status_code==405
    assert client.get('/api/latest?run=..').status_code==404
    clips=client.get('/api/replays').json()['clips']
    assert len(clips)==1
    data=client.get('/api/replay/'+clips[0]['id']).json()
    ticks=[f['tick'] for f in data['frames_data']]
    assert ticks==list(range(clips[0]['start'],clips[0]['start']+5))
    assert data['selection']=='uniform_preselected'
    assert data['obstacles']==w1.obstacles[0].long().flatten().tolist()
    assert client.get('/api/latest').json()['worlds'][0]['obstacles']==data['obstacles']
    # Old saved replays have no static obstacle field; they remain viewable.
    path=directory/'replays'/(clips[0]['id']+'.npz')
    with np.load(path) as old:arrays={k:old[k] for k in old.files if k!='obstacles'}
    np.savez_compressed(path,**arrays)
    old_client=TestClient(create_app(root))
    assert old_client.get('/api/replay/'+clips[0]['id']).json()['obstacles']==[]


def test_extinct_world_remains_an_eligible_replay(tmp_path):
    cfg=Config(worlds=1,initial=1,capacity=1,size=12,hidden=8,radius=1,ticks=10,replay_count=1,replay_length=3,metabolism=100)
    world,learners=World(cfg),Learners(cfg)
    selected=[{'id':'random-000','world':0,'start':5,'length':3,'selection':'uniform_preselected'}]
    observer=Observer(cfg,tmp_path,schedule=selected)
    for tick in range(1,11):iteration(world,learners);observer.capture(world,tick)
    observer.close(world,10,'complete')
    with np.load(tmp_path/'replays'/'random-000.npz') as data:
        assert data['ticks'].tolist()==[5,6,7]
        assert not data['alive'].any()


def test_comparison_uses_matching_ticks_not_latest_wall_time(tmp_path):
    cfg=Config(worlds=1,initial=2,capacity=4,size=12)
    root=tmp_path/'runs'
    for name,mode,rows in [('learned','L3',[(2,[3]),(4,[4])]),('control','RANDOM',[(2,[2]),(3,[1])])]:
        path=root/name;path.mkdir(parents=True)
        world=World(dataclasses.replace(cfg,mode=mode))
        atomic_json(path/'latest.json',{'tick':rows[-1][0],'worlds':[world.snapshot(0)]})
        (path/'metrics.jsonl').write_text(''.join(json.dumps({'tick':t,'alive':alive})+'\n' for t,alive in rows))
    atomic_json(root/'comparison.json',{'learned':'learned','control':'control'})
    client=TestClient(create_app(root))
    result=client.get('/api/comparison').json()
    assert len(result['groups'])==2
    assert result['paired_population']=={'tick':2,'learned':[3],'control':[2]}
    atomic_json(root/'archived-study.json',{'learned':'control','control':'learned'})
    archived=client.get('/api/comparison?study=archived-study').json()
    assert archived['paired_population']=={'tick':2,'learned':[2],'control':[3]}
    assert client.get('/api/comparison?study=../outside').status_code==404
