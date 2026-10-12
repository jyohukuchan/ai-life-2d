from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import signal
import threading
import time
from pathlib import Path

import torch

from .config import Config
from .policy import make_policy
from .observer import EventLog, Observer, atomic_json, replay_schedule
from .world import World
from .reward_log import RewardLog
from .selection import EliteArchive


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def checkpoint(path,cfg,world,learners,schedule,archive=None,initialization=None):
    path=Path(path);temporary=path.with_suffix('.tmp')
    payload={'version':1,'config':cfg.as_dict(),'world':world.state_dict(),'learners':learners.state_dict(),'replay_schedule':schedule}
    if archive is not None:payload['elite_archive']=archive.state_dict()
    if initialization is not None:payload['initialization']=initialization
    torch.save(payload,temporary)
    temporary.replace(path)


def iteration(world,learners,archive=None):
    raw=None if world.cfg.mode=='RANDOM' else world.observe()
    actions=learners.act(raw,world.ids,world.clock,world.alive)
    old={k:getattr(world,k).clone() for k in ("age","birth_tick","generation")} if archive is not None else None
    transition=world.step(actions)
    next_raw=None if world.cfg.mode=='RANDOM' else world.observe(transition.positions,transition.directions,transition.energy)
    result=learners.learn(next_raw,transition)
    if archive is not None:archive.observe(world,learners,transition,old)
    learners.inherit(transition,world.ids)
    return transition,result


def statistics(world,learners,tick):
    return {'tick':tick,'alive':world.alive.sum(1).cpu().tolist(),'food':world.food.flatten(1).sum(1).cpu().tolist(),'energy':world.energy.where(world.alive,0).sum(1).cpu().tolist(),'births_clone_sexual':world.births_total.cpu().tolist(),'deaths_energy_age':world.deaths_total.cpu().tolist(),'eaten':world.eaten_total.cpu().tolist(),'max_generation':world.generation.max(1).values.cpu().tolist(),'capacity_rejected':world.capacity_rejected.cpu().tolist(),'capacity_ticks':world.capacity_ticks.cpu().tolist(),'extinction_tick':world.extinction_tick.cpu().tolist()}


def run(cfg:Config,device='cpu',directory=None,resume=None,observe=True,save=True):
    torch.set_num_threads(cfg.threads)
    device=torch.device(device)
    directory=Path(directory or f'runs/{time.strftime("%Y%m%d-%H%M%S")}-{cfg.mode}-{cfg.digest()}').resolve()
    directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'config.json',cfg.as_dict())
    atomic_json(directory.parent/'active.json',{'run':directory.name})
    world,learners=World(cfg,device),make_policy(cfg,device)
    schedule=None;initialization=None
    archive=EliteArchive(cfg,device) if cfg.elite_count else None
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=True)
        # The tick horizon and observation frequency may change; physics/model may not.
        previous=Config.from_saved(state['config']).as_dict();current=cfg.as_dict()
        for k in current:
            if k not in {'ticks','publish_seconds','log_ticks','checkpoint_ticks','replay_count','replay_length','observer_seed','threads','log_reward_components','reward_log_block','elite_count','fast_forward_extinct','precision','execution','fp16_loss_scale'} and previous[k]!=current[k]:raise ValueError(f'resume configuration mismatch: {k}')
        world.load_state_dict(state['world']);learners.load_state_dict(state['learners'])
        initialization=state.get('initialization')
        if archive is not None:
            if 'elite_archive' in state:
                if previous['elite_count']!=cfg.elite_count:raise ValueError('elite count cannot change mid-session')
                archive.load_state_dict(state['elite_archive'])
            elif world.clock.item()>0:raise ValueError('cannot recover discarded elite weights from an unfinished legacy checkpoint')
    start_tick=int(world.clock.item())
    if start_tick>cfg.ticks:raise ValueError('tick horizon precedes checkpoint')
    # A resumed run gets its own observational clips, chosen before its execution.
    if resume:schedule=replay_schedule(cfg,start_tick)
    observer=Observer(cfg,directory,enabled=observe,schedule=schedule)
    events=EventLog(directory/'events.jsonl')
    rewards=RewardLog(cfg,directory) if cfg.log_reward_components and cfg.mode=='L3' else None
    metadata={'python':platform.python_version(),'torch':torch.__version__,'hip':torch.version.hip,'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),'rocr_visible_devices':os.getenv('ROCR_VISIBLE_DEVICES'),'parameters_per_agent':learners.parameter_count(),'population_slots':cfg.worlds*cfg.capacity,'update_targets':{'prediction':list(learners.pred_names) if cfg.mode in {'L2','L3'} else [],'local':['feat_w'] if cfg.mode=='L1' else [],'rl':['act_w','act_b','value_w','value_b'] if cfg.mode=='L3' else []},'pretraining':bool(initialization and initialization.get('contains_acquired_weights')),'external_pretraining':False,'initialization':initialization,'external_task_rewards':False}
    metadata.update(config_hash=cfg.digest(),start_tick=start_tick,resumed_from=str(Path(resume).resolve()) if resume else None)
    metadata['compass']={'mode':cfg.compass,'channels':cfg.compass_size,'order':['north','east','south','west'],'encoding':'one_hot' if cfg.compass=='absolute' else 'zero' if cfg.compass=='masked' else 'absent','policy_input_only':True,'prediction_target':False,'intrinsic_reward_feature':False}
    metadata['compute']={'precision':cfg.precision,'execution':cfg.execution,'master_parameters':'float32','statistics':'float32','world_physics':'float32','fp16_loss_scale':cfg.fp16_loss_scale if cfg.precision=='fp16' else None}
    metadata['observation_size']=cfg.observation_size
    metadata['architecture']={'encoder_width':cfg.encoder_width,'gru_hidden_size':cfg.hidden,'feature_size':cfg.hidden,'prediction_bptt':1,'rl_updates_core':False}
    metadata['prediction_target_size']=cfg.sensory_size
    metadata['movement_frame']='relative_wait_forward_left90_right90'
    metadata['action_policy']='independent_uniform_per_component_each_tick' if cfg.mode=='RANDOM' else 'individual_neural_policy'
    code=hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*')):
        if path.is_file() and path.suffix in {'.py','.html'}:
            code.update(str(path.relative_to(Path(__file__).parent)).encode());code.update(path.read_bytes())
    metadata['source_hash']=code.hexdigest()
    metadata['rng']='counter_hash32_by_world_seed_individual_id_tick_stream; observer_numpy_generator_separate'
    metadata['evaluation_features']=['own_energy_delta_or_zero','mean_rgb_delta_R','mean_rgb_delta_G','mean_rgb_delta_B']+[f'received_symbol_{i}' for i in range(1,cfg.symbols+1)]+[f'action_{j}_choice_{k}' for j,size in enumerate(cfg.action_sizes) for k in range(size)]
    atomic_json(directory/'metadata.json',metadata)
    observer.capture(world,start_tick);observer.publish(world,start_tick,force=True)
    sync(device);begin=time.perf_counter();last=begin
    decisions=torch.zeros((),device=device,dtype=torch.int64)
    loss_total=torch.zeros((),device=device);reward_total=torch.zeros((),device=device)
    status='running';tick=start_tick;executed_ticks=0;skipped_ticks=0
    def log_row(log_tick):
        sync(device);now=time.perf_counter()
        if not learners.finite():raise FloatingPointError('non-finite learner state')
        if not world.valid():raise AssertionError('invalid world state: energy, positions or collision invariant')
        row={**statistics(world,learners,log_tick),'elapsed_seconds':now-begin,
             'world_ticks_per_second':cfg.worlds*executed_ticks/max(now-begin,1e-9),
             'fast_forwarded_ticks':skipped_ticks,
             'agent_decisions_per_second':float(decisions.item())/max(now-begin,1e-9),
             'mean_prediction_loss':float(loss_total.item())/max(int(decisions.item()),1),
             'mean_raw_intrinsic_reward':float(reward_total.item())/max(int(decisions.item()),1)}
        with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        events.flush();print(json.dumps({'run':directory.name,**row}),flush=True)
    stop_requested=[];previous_handlers={}
    if threading.current_thread() is threading.main_thread():
        def request_stop(signum,frame):stop_requested.append(signum)
        for signum in [signal.SIGINT,signal.SIGTERM]:
            previous_handlers[signum]=signal.signal(signum,request_stop)
    try:
        for tick in range(start_tick+1,cfg.ticks+1):
            t,result=iteration(world,learners,archive)
            executed_ticks+=1
            decisions+=t.was_alive.sum()
            loss_total+=(result['prediction_loss']*t.was_alive).sum()
            reward_total+=(result['intrinsic_reward']*t.was_alive).sum()
            events.append(world,t,tick)
            if rewards:rewards.append(t,result['reward_diagnostics'],tick)
            observer.capture(world,tick);observer.publish(world,tick)
            extinct=cfg.fast_forward_extinct and not bool(world.alive.any())
            if tick%cfg.log_ticks==0 or tick==cfg.ticks or extinct:log_row(tick)
            if save and cfg.checkpoint_ticks and tick%cfg.checkpoint_ticks==0:
                checkpoint(directory/'checkpoint.pt',cfg,world,learners,observer.schedule,archive,initialization)
            if stop_requested:raise KeyboardInterrupt('Stopped after a complete transition; checkpoint is resumable')
            if extinct and tick<cfg.ticks:
                observer.fill_extinct(world,tick)
                actual_tick=tick
                for logical_tick in sorted(set(range((tick//cfg.log_ticks+1)*cfg.log_ticks,cfg.ticks+1,cfg.log_ticks))|{cfg.ticks}):
                    skipped_ticks=logical_tick-actual_tick;log_row(logical_tick)
                skipped_ticks=cfg.ticks-actual_tick
                world.clock.fill_(cfg.ticks);tick=cfg.ticks
                break
        status='complete'
    except BaseException as exc:
        status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        atomic_json(directory/'error.json',{'type':type(exc).__name__,'message':str(exc),'tick':tick})
        raise
    finally:
        if rewards:rewards.flush()
        events.flush();observer.close(world,int(world.clock.item()),status)
        if save:checkpoint(directory/'checkpoint.pt',cfg,world,learners,observer.schedule,archive,initialization)
        if archive is not None and status=='complete':
            temp=directory/'elites.tmp';torch.save(archive.selected(world,learners),temp);temp.replace(directory/'elites.pt')
        sync(device);elapsed=time.perf_counter()-begin
        summary={**metadata,**statistics(world,learners,int(world.clock.item())),'status':status,'config_hash':cfg.digest(),'elapsed_seconds':elapsed,'measured_ticks':executed_ticks,'fast_forwarded_ticks':skipped_ticks,'world_ticks_per_second':cfg.worlds*executed_ticks/max(elapsed,1e-9),'agent_decisions_per_second':float(decisions.item())/max(elapsed,1e-9),'snapshot_drops':observer.dropped,'observer_write_seconds':observer.work_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
        atomic_json(directory/'summary.json',summary)
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
    return summary
