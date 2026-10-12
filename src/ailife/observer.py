"""Read-only observation: selection RNG and queues never affect agent decisions."""
from __future__ import annotations

import json
import queue
import threading
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


FRAME_FIELDS=('xy','direction','energy','age','alive','ids','parents','generation','signal','food')


def atomic_json(path, value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        temp=Path(f.name)
        try:
            f.write(json.dumps(value,ensure_ascii=False,separators=(',',':')))
            f.close()
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)


def replay_schedule(cfg,start_tick=0):
    rng=np.random.default_rng(cfg.observer_seed)
    length=min(cfg.replay_length,cfg.ticks-start_tick+1)
    starts=max(1,cfg.ticks-start_tick-length+2)
    choices=rng.choice(cfg.worlds*starts,min(cfg.replay_count,cfg.worlds*starts),replace=False)
    return sorted([{'id':f'random-{i:03d}','world':int(k//starts),'start':start_tick+int(k%starts),'length':length,'selection':'uniform_preselected'} for i,k in enumerate(choices)],key=lambda x:x['start'])


def arrays_snapshot(arrays, cfg, world, tick, fertility, extinction=-1, obstacles=None):
    alive=np.flatnonzero(arrays['alive'])
    agents=[{'slot':int(i),'id':int(arrays['ids'][i]),'y':int(arrays['xy'][i,0]),'x':int(arrays['xy'][i,1]),'direction':int(arrays['direction'][i]),'energy':round(float(arrays['energy'][i]),3),'age':int(arrays['age'][i]),'generation':int(arrays['generation'][i]),'parents':arrays['parents'][i].tolist(),'signal':int(arrays['signal'][i])} for i in alive]
    return {'world':world,'tick':tick,'size':cfg['size'],'height':cfg.get('height',cfg['size']),'width':cfg.get('width',cfg['size']),'food':arrays['food'].astype(np.uint8).flatten().tolist(),'fertility':fertility.flatten().round(3).tolist(),'obstacles':[] if obstacles is None else obstacles.astype(np.uint8).flatten().tolist(),'agents':agents,'extinction_tick':extinction,'config':cfg}


class Observer:
    def __init__(self,cfg,directory,enabled=True,schedule=None):
        self.cfg,self.directory,self.enabled=cfg,Path(directory),enabled
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/'replays').mkdir(exist_ok=True)
        self.schedule=replay_schedule(cfg) if schedule is None else schedule
        atomic_json(self.directory/'replay-selection.json',{'observer_seed':cfg.observer_seed,'selected_before_run':True,'clips':self.schedule})
        self.frozen_after=None
        self.buffers={x['id']:[] for x in self.schedule}
        self.pending=queue.Queue(maxsize=1)
        self.error=None;self.dropped=0;self.last_publish=0.;self.work_seconds=0.
        self.worker=threading.Thread(target=self._writer,daemon=True)
        self.worker.start()

    def _writer(self):
        while True:
            item=self.pending.get()
            try:
                if item is None:return
                begin=time.perf_counter()
                atomic_json(self.directory/'latest.json',item)
                self.work_seconds+=time.perf_counter()-begin
            except Exception as exc:self.error=repr(exc)
            finally:self.pending.task_done()

    def publish(self,world,tick,status='running',force=False):
        if not self.enabled and not force:return
        now=time.monotonic()
        if not force and now-self.last_publish<self.cfg.publish_seconds:return
        self.last_publish=now
        if self.error:raise RuntimeError(f'observer writer failed: {self.error}')
        if self.pending.full() and not force:
            self.dropped+=1;return
        payload={'run':self.directory.name,'status':status,'published_unix':time.time(),'tick':tick,'worlds':[world.snapshot(i) for i in range(self.cfg.worlds)],'snapshot_drops':self.dropped}
        # Never wait for a viewer or for the previous live snapshot to be written.
        try:self.pending.put_nowait(payload)
        except queue.Full:
            self.dropped+=1
            if force:
                self.pending.join();self.pending.put_nowait(payload)

    def capture(self,world,tick):
        for clip in self.schedule:
            if clip['start']<=tick<clip['start']+clip['length']:
                wi=clip['world']
                frame={k:getattr(world,k)[wi].detach().clone() for k in FRAME_FIELDS}
                self.buffers[clip['id']].append(frame)
                if tick==clip['start']+clip['length']-1:self.flush_clip(clip,world)

    def fill_extinct(self,world,tick):
        """Complete preselected clips exactly when all future frames are identical."""
        if world.alive.any():raise ValueError('cannot skip live simulation')
        self.frozen_after=tick
        for clip in self.schedule:
            if clip['id'] not in self.buffers:continue
            start=max(tick+1,clip['start'])
            end=clip['start']+clip['length']
            if start>=end:continue
            frame={k:getattr(world,k)[clip['world']].detach().clone() for k in FRAME_FIELDS}
            self.buffers[clip['id']].extend([frame]*(end-start))
            self.flush_clip(clip,world)

    def flush_clip(self,clip,world):
        frames=self.buffers.pop(clip['id'],[])
        if not frames:return
        arrays={k:torch.stack([f[k] for f in frames]).cpu().numpy() for k in FRAME_FIELDS}
        arrays['fertility']=world.fertility[clip['world']].cpu().numpy()
        arrays['obstacles']=world.obstacles[clip['world']].cpu().numpy()
        arrays['ticks']=np.arange(clip['start'],clip['start']+len(frames))
        path=self.directory/'replays'/clip['id']
        np.savez_compressed(path.with_suffix('.npz'),**arrays)
        atomic_json(path.with_suffix('.json'),{**clip,'frames':len(frames),'config':self.cfg.as_dict(),'extinction_tick':int(world.extinction_tick[clip['world']].item()),'frozen_after_extinction':self.frozen_after})

    def close(self,world,tick,status):
        for clip in self.schedule:
            if self.buffers.get(clip['id']):self.flush_clip(clip,world)
        self.publish(world,tick,status,force=True)
        self.pending.join();self.pending.put(None);self.worker.join(timeout=10)
        if self.error:raise RuntimeError(self.error)


class EventLog:
    """Observational lineage buffer, never read by the learners."""
    def __init__(self,path):
        self.path=Path(path);self.frames=[]

    def append(self,world,transition,tick):
        t=transition
        # Keep fixed-shape device tensors until the next low-frequency log flush.
        self.frames.append(torch.stack((t.ids,t.death_cause,t.births.long(),world.ids,world.parents[...,0],world.parents[...,1],world.generation,t.energy.long()),-1).detach())
        self.last_tick=tick

    def flush(self):
        if not self.frames:return
        data=torch.stack(self.frames).cpu().numpy()
        first=self.last_tick-len(self.frames)+1
        self.frames.clear()
        death=np.argwhere(data[...,1]>0);birth=np.argwhere(data[...,2]>0)
        events=[]
        for dt,wi,slot in death:
            row=data[dt,wi,slot]
            events.append({'tick':int(first+dt),'world':int(wi),'event':'death','id':int(row[0]),'cause':'energy' if row[1]==1 else 'age'})
        for dt,wi,slot in birth:
            row=data[dt,wi,slot]
            events.append({'tick':int(first+dt),'world':int(wi),'event':'birth','id':int(row[3]),'parents':[int(row[4]),int(row[5])],'generation':int(row[6])})
        with self.path.open('a') as f:
            for event in sorted(events,key=lambda e:e['tick']):f.write(json.dumps(event)+'\n')
