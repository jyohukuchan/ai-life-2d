"""Repeat independent, last-survivor selection chains on a matched sequence of new maps."""
import argparse
import dataclasses
import fcntl
import gc
import json
from pathlib import Path

import torch

from ailife.config import Config
from ailife.observer import atomic_json, replay_schedule
from ailife.runner import checkpoint, run
from ailife.selection import import_completed_elites, seed_population


LABELS={'acquired':'獲得重みを継承','genome':'出生時の重みを継承','fresh':'毎回未学習から開始'}


def measurements(directory):
    summary=json.loads((directory/'summary.json').read_text())
    rows={0:[0]*len(summary['eaten'])}
    for line in (directory/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);rows[row['tick']]=row['eaten']
    horizon=summary['tick'];bounds=[horizon-3000,horizon-2000,horizon-1000,horizon]
    late=[[rows[b][wi]-rows[a][wi] for a,b in zip(bounds,bounds[1:])]
          for wi in range(len(summary['eaten']))]
    return {'food':summary['eaten'],'extinction_tick':summary['extinction_tick'],'alive':summary['alive'],
            'late_food_windows':late,'stable_this_session':[bool(n and all(x>0 for x in windows)) for n,windows in zip(summary['alive'],late)],
            'elapsed_seconds':summary['elapsed_seconds'],'status':summary['status']}


def publish_comparison(root):
    with (root/'session-selection.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=[root/f'session-selection-{arm}.json' for arm in ['acquired','genome']]
        if not all(p.exists() for p in paths):return
        a,b=[json.loads(p.read_text()) for p in paths]
        common=set(map(int,a['sessions']))&set(map(int,b['sessions']))
        if not common:return
        session=max(common)
        spec={'learned':a['sessions'][str(session)]['run'],'control':b['sessions'][str(session)]['run'],
              'display_labels':{'learned':LABELS['acquired'],'control':LABELS['genome']},
              'labels':{'learned':'Acquired weights','control':'Birth-time genome'},
              'session':session+1,'planned_sessions':a['planned_sessions'],
              'selection':'within-world last survivors; round-robin clones; fresh map each session'}
        atomic_json(root/'session-selection-study.json',spec);atomic_json(root/'comparison.json',spec)
        atomic_json(root/'active.json',{'run':spec['learned']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='configs/experiments/2026-10-11-compass/compass-absolute.yaml')
    parser.add_argument('--source',default='runs/compass-absolute-L3-10k')
    parser.add_argument('--original',default='runs/compass-ablation-initial/absolute.pt')
    parser.add_argument('--arm',choices=list(LABELS),default='acquired')
    parser.add_argument('--sessions',type=int,default=10)
    parser.add_argument('--elite-count',type=int,default=4)
    parser.add_argument('--seed-step',type=int,default=1000003)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();source=Path(args.source).resolve();root=source.parent
    cfg=dataclasses.replace(Config.load(args.base),elite_count=args.elite_count,fast_forward_extinct=True,
                            log_reward_components=False)
    torch.set_num_threads(cfg.threads)
    if cfg.ticks!=10000 or cfg.log_ticks!=1000:raise ValueError('this study uses the 10000-tick, 1000-tick-window protocol')
    initial=root/'session-selection-initial';initial.mkdir(exist_ok=True)
    packfile=initial/'source-elites.pt'
    if args.prepare_only:
        pack=import_completed_elites(source,args.elite_count)
        torch.save(pack,packfile)
        protocol={'sessions':args.sessions,'worlds_per_arm':cfg.worlds,'founders_per_world':cfg.initial,
                  'elite_count':args.elite_count,'source_session_1':source.name,'config':cfg.as_dict(),
                  'arms':LABELS,'seed_sequence':[cfg.seed+i*args.seed_step for i in range(args.sessions)],
                  'selection':'if alive at horizon, uniform up to K alive; otherwise K latest deaths ordered by tick then ID; no food fitness',
                  'transfer':'acquired params and observation normalization rebased as genome; birth-time params+calibration in genome arm; private founder copies',
                  'within_session_birth':'genome+birth-time calibration; mutation unchanged; never current acquired weights',
                  'reset':['body','age','energy','hidden','previous_action','optimizer','eligibility_traces','reward_statistics'],
                  'stability':'same chain alive at 10000 and food>0 in EACH of ticks 7001-8000, 8001-9000, 9001-10000, for 3 consecutive sessions',
                  'external_pretraining':False,'task_reward_added':False,'extra_boundary_mutation':False,
                  'source_selected':[[{k:r[k] for k in ['id','death_tick','age','eaten']} for r in rows] for rows in pack['worlds']]}
        atomic_json(root/'session-selection-protocol.json',protocol);print(json.dumps(protocol,ensure_ascii=False));return
    if not packfile.exists():raise ValueError('run --prepare-only first')
    original=torch.load(args.original,map_location='cpu',weights_only=True) if args.arm=='fresh' else None
    manifestfile=root/f'session-selection-{args.arm}.json'
    if manifestfile.exists():
        manifest=json.loads(manifestfile.read_text())
        if manifest['planned_sessions']!=args.sessions:raise ValueError('study horizon differs')
    else:manifest={'arm':args.arm,'planned_sessions':args.sessions,'status':'running','sessions':{'0':{'run':source.name,'seed':cfg.seed,'reused_shared_baseline':True,**measurements(source)}}}
    atomic_json(manifestfile,manifest);publish_comparison(root)
    try:
        for index in range(1,args.sessions):
            prior=manifest['sessions'][str(index-1)]
            priorpack=packfile if index==1 else root/prior['run']/'elites.pt'
            pack=torch.load(priorpack,map_location='cpu',weights_only=True)
            sessioncfg=dataclasses.replace(cfg,seed=cfg.seed+index*args.seed_step,observer_seed=cfg.observer_seed+index*17)
            destination=root/f'session-{args.arm}-s{index:02d}'
            if destination.exists():
                if (destination/'summary.json').exists() and json.loads((destination/'summary.json').read_text())['status']=='complete':
                    manifest['sessions'][str(index)]={**manifest['sessions'].get(str(index),{}),'run':destination.name,'seed':sessioncfg.seed,**measurements(destination)}
                    atomic_json(manifestfile,manifest);publish_comparison(root);continue
                raise FileExistsError(f'incomplete trial preserved: {destination}; resume it explicitly before continuing')
            world,learners,assignments=seed_population(sessioncfg,pack,args.arm,original)
            info={'type':'session_selection','arm':args.arm,'session':index+1,'source_run':prior['run'] if args.arm!='fresh' else None,
                  'contains_acquired_weights':args.arm=='acquired','external_pretraining':False,
                  'initial_world_digest':world.digest(),'assignments':assignments}
            initfile=initial/f'{args.arm}-s{index:02d}.pt'
            checkpoint(initfile,sessioncfg,world,learners,replay_schedule(sessioncfg),initialization=info)
            manifest['sessions'][str(index)]={'run':destination.name,'seed':sessioncfg.seed,'status':'starting','initialization':info}
            atomic_json(manifestfile,manifest);publish_comparison(root)
            del world,learners,pack;gc.collect()
            summary=run(sessioncfg,args.device,destination,resume=initfile)
            if summary['status']!='complete':raise RuntimeError('session did not complete')
            manifest['sessions'][str(index)].update(measurements(destination))
            chosen=torch.load(destination/'elites.pt',map_location='cpu',weights_only=True)
            manifest['sessions'][str(index)]['selected']=[[{k:r[k] for k in ['id','death_tick','selected_alive','age','eaten']} for r in rows] for rows in chosen['worlds']]
            del chosen;gc.collect()
            if torch.device(args.device).type=='cuda':torch.cuda.empty_cache()
            atomic_json(manifestfile,manifest);publish_comparison(root)
            print(json.dumps({'completed_session':index+1,'arm':args.arm,**measurements(destination)}),flush=True)
        manifest['status']='complete'
        chains=[]
        for wi in range(cfg.worlds):
            streak=best=0
            for index in range(args.sessions):
                streak=streak+1 if manifest['sessions'][str(index)]['stable_this_session'][wi] else 0;best=max(best,streak)
            chains.append({'world':wi,'best_consecutive_stable_sessions':best,'criterion_met':best>=3})
        manifest['stability']=chains
    except BaseException as exc:
        manifest['status']='failed';manifest['error']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:atomic_json(manifestfile,manifest);publish_comparison(root)


if __name__=='__main__':main()
