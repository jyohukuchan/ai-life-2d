from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import torch

from .config import Config
from .policy import make_policy
from .observer import Observer
from .runner import iteration, sync
from .world import World


def benchmark(args):
    torch.set_num_threads(args.threads)
    cfg=Config(precision=args.precision,execution=args.execution,worlds=args.worlds,initial=args.initial,capacity=args.capacity,hidden=args.hidden,encoder_width=getattr(args,'encoder_width',32),mode=args.mode,ticks=args.warmup+args.steps,replay_count=1 if args.observe else 0,replay_length=min(args.steps,64),publish_seconds=.5,obstacle_density=args.obstacle_density,compass=args.compass)
    device=torch.device(args.device)
    h,d,n=cfg.hidden,cfg.sensory_size+cfg.action_size+cfg.compass_size,cfg.worlds*cfg.capacity
    # Parameter + genome + Adam + gradients + activations; conservative allocation gate.
    predicted=4*n*(d*cfg.encoder_width+3*h*cfg.encoder_width+4*h*h+(h+cfg.action_size)*cfg.sensory_size)* (7 if cfg.mode in {'L2','L3'} else 3)
    if cfg.mode=='RANDOM':predicted=cfg.worlds*(cfg.height*cfg.width*64+cfg.capacity*256)
    row={'device':str(device),'torch':torch.__version__,'hip':torch.version.hip,'mode':cfg.mode,'worlds':cfg.worlds,'initial_per_world':cfg.initial,'capacity_per_world':cfg.capacity,'hidden':cfg.hidden,'steps':args.steps,'warmup':args.warmup,'observation_enabled':args.observe,'predicted_memory_bytes':predicted}
    row['obstacle_density']=cfg.obstacle_density
    row['compass']=cfg.compass
    row['encoder_width']=cfg.encoder_width
    row['precision']=cfg.precision;row['execution']=cfg.execution
    if device.type=='cuda':
        row['device_name']=torch.cuda.get_device_name(device)
        free,total=torch.cuda.mem_get_info(device)
        if predicted>free*.7:
            row['status']='skipped_memory_estimate'
            return write_result(args.output,row)
        torch.cuda.reset_peak_memory_stats(device)
    world,learners=World(cfg,device),make_policy(cfg,device)
    for _ in range(args.warmup):iteration(world,learners)
    temp=tempfile.TemporaryDirectory()
    observer=Observer(cfg,Path(temp.name)/'observe',enabled=args.observe) if args.observe else None
    sync(device);begin=time.perf_counter()
    decisions=torch.zeros((),device=device,dtype=torch.int64)
    for tick in range(args.steps):
        t,result=iteration(world,learners);decisions+=t.was_alive.sum()
        if observer:
            observer.capture(world,tick);observer.publish(world,tick)
    sync(device);elapsed=time.perf_counter()-begin
    if observer:observer.close(world,args.steps,'complete')
    row.update(status='ok',elapsed_seconds=elapsed,world_ticks_per_second=cfg.worlds*args.steps/elapsed,agent_decisions_per_second=int(decisions.item())/elapsed,actual_agent_decisions=int(decisions.item()),parameters_per_agent=learners.parameter_count(),alive_at_end=world.alive.sum(1).cpu().tolist(),finite=learners.finite(),peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    temp.cleanup()
    return write_result(args.output,row)


def write_result(output,row):
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row)+'\n')
    return row
