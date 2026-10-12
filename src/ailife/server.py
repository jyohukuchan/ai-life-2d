from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .observer import arrays_snapshot


def create_app(root='runs'):
    root=Path(root).resolve()
    app=FastAPI(title='AI Life 2D · read-only observatory',docs_url=None,redoc_url=None)

    def directory(run):
        if not run:
            try:run=json.loads((root/'active.json').read_text())['run']
            except (FileNotFoundError,KeyError,json.JSONDecodeError):raise HTTPException(404,'No simulation yet')
        path=(root/run).resolve()
        if path.parent!=root or not path.is_dir():raise HTTPException(404,'Unknown run')
        return path

    @app.get('/')
    def index():return FileResponse(Path(__file__).parent/'static'/'index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/api/runs')
    def runs():
        items=[]
        for path in sorted(root.glob('*/config.json'),reverse=True):
            try:
                cfg=json.loads(path.read_text())
                summary=path.parent/'summary.json'
                items.append({'id':path.parent.name,'mode':cfg['mode'],'seed':cfg['seed'],'status':json.loads(summary.read_text())['status'] if summary.exists() else 'running'})
            except (OSError,KeyError,json.JSONDecodeError):continue
        return {'runs':items}

    @app.get('/api/latest')
    def latest(run:str|None=None):
        path=directory(run)/'latest.json'
        if not path.exists():raise HTTPException(404,'Snapshot not published yet')
        return FileResponse(path,media_type='application/json',headers={'Cache-Control':'no-store'})

    @app.get('/api/replays')
    def replays(run:str|None=None):
        path=directory(run)
        completed=[json.loads(p.read_text()) for p in sorted((path/'replays').glob('*.json'))]
        selection=json.loads((path/'replay-selection.json').read_text())
        return {'run':path.name,'selection':selection,'clips':completed}

    @functools.lru_cache(maxsize=2)
    def read_clip(path):
        path=Path(path)
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
            frames=[]
            for i,tick in enumerate(data['ticks']):
                frame=arrays_snapshot({k:data[k][i] for k in data.files if k not in {'fertility','ticks','obstacles'}},meta['config'],meta['world'],int(tick),data['fertility'],meta['extinction_tick'])
                frame.pop('config');frame.pop('fertility');frame.pop('obstacles');frames.append(frame)
            fertility=data['fertility'].flatten().round(3).tolist()
            obstacles=data['obstacles'].astype(np.uint8).flatten().tolist() if 'obstacles' in data.files else []
        return {**meta,'fertility':fertility,'obstacles':obstacles,'frames_data':frames}

    @app.get('/api/comparison')
    def comparison(study:str|None=None):
        spec_path=(root/(f'{study}.json' if study else 'comparison.json')).resolve()
        if spec_path.parent!=root or not spec_path.is_file():raise HTTPException(404,'No comparison configured')
        spec=json.loads(spec_path.read_text())
        if not {'learned','control'}<=spec.keys():raise HTTPException(404,'Not a comparison study')
        groups=[];metrics=[]
        for role in ['learned','control']:
            path=directory(spec[role]);snapshot=path/'latest.json'
            if not snapshot.exists():raise HTTPException(404,'Comparison world is starting')
            payload=json.loads(snapshot.read_text())
            label=spec.get('display_labels',{}).get(role,spec.get('labels',{}).get(role,payload['worlds'][0]['config']['mode']))
            groups.append({'role':role,'run':path.name,'label':label,'snapshot':payload})
            rows={}
            for run in spec.get(role+'_chain',[spec[role]]):
                metric_path=directory(run)/'metrics.jsonl'
                if not metric_path.exists():continue
                for line in metric_path.read_text().splitlines(keepends=True):
                    if line.endswith('\n'):
                        row=json.loads(line);rows[row['tick']]=row['alive']
            metrics.append(rows)
        common=metrics[0].keys()&metrics[1].keys()
        paired=None
        if common:
            tick=max(common)
            paired={'tick':tick,'learned':metrics[0][tick],'control':metrics[1][tick]}
        return {'groups':groups,'paired_population':paired,'time_basis':'same_simulation_tick','session':spec.get('session'),'planned_sessions':spec.get('planned_sessions')}

    @app.get('/api/replay/{clip_id}')
    def replay(clip_id:str,run:str|None=None):
        parent=directory(run)/'replays'
        path=(parent/clip_id).resolve()
        if path.parent!=parent or not path.with_suffix('.npz').exists():raise HTTPException(404,'Replay not available')
        return JSONResponse(read_clip(str(path)),headers={'Cache-Control':'public, max-age=3600'})

    @app.get('/api/health')
    def health():return {'status':'ok','read_only':True}

    return app
