"""Predeclared independent-trial blocks for encoder/hidden and session comparisons."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

from .config import Config
from .observer import atomic_json
from .scaling import prepare_study, study_path


def build_plan(cfg, study='rep20-20261012'):
    study_path('.', study, 'plan.json')
    blocks = [dict(id='b00', offset=0, worlds=7, rocr='0', cores='16-19'),
        dict(id='b01', offset=7, worlds=7, rocr='1', cores='24-27'),
        dict(id='b02', offset=14, worlds=6, rocr='2', cores='32-35')]
    seeds = {'A': 1000000007, 'B': 2000000011, 'evaluation': 3000000019}
    cells = [dict(id=f'A-h{h:03d}-e{e:03d}-s005', cohort='A', hidden=h,
        encoder_width=e, session=5, trials=20) for e in (32, 64) for h in (64, 192)]
    cells += [dict(id=f'B-h064-e032-s{n:03d}', cohort='B', hidden=64,
        encoder_width=32, session=n, trials=20) for n in (5, 20)]
    protocols, jobs = {}, []
    for block in blocks:
        for cohort, encoder, sizes, sessions, milestones in [
                ('A', 32, (64, 192), 5, (5,)),
                ('A', 64, (64, 192), 5, (5,)),
                ('B', 32, (64,), 20, (5, 20))]:
            name = f'{study}-{cohort.lower()}-e{encoder:03d}-{block["id"]}'
            c = dataclasses.replace(cfg, worlds=block['worlds'], hidden=64,
                encoder_width=encoder, seed=seeds[cohort] + block['offset'] * 1009,
                observer_seed=cfg.observer_seed + block['offset'] * 1009 + (500000 if cohort == 'B' else 0))
            protocols[name] = {'config': c.as_dict(), 'sizes': list(sizes),
                'sessions': sessions, 'milestones': list(milestones),
                'evaluation_seed': seeds['evaluation'] + block['offset'] * 1009}
            for hidden in sizes:
                jobs.append({'id': f'{cohort}-h{hidden:03d}-e{encoder:03d}-{block["id"]}',
                    'study': name, 'cohort': cohort, 'hidden': hidden,
                    'encoder_width': encoder, 'sessions': sessions, 'milestones': list(milestones),
                    'block': block['id'], 'trial_offset': block['offset'],
                    'worlds': block['worlds'], 'rocr': block['rocr'], 'cores': block['cores']})
    return {'version': 1, 'study': study, 'trials_per_condition': 20,
        'independent_learning_chains': 100, 'primary_evaluation_outcomes': 120,
        'training_world_sessions': 800, 'evaluation_world_sessions': 120,
        'blocks': blocks, 'seeds': seeds, 'cells': cells, 'protocols': protocols, 'jobs': jobs,
        'pairing': 'same trial index uses same maps and GPU across architecture conditions; B is a separate 20-chain cohort with paired session5/session20 evaluations',
        'terminology': 'encoder_width is the user-clarified 32/64 factor; GRU state and feature widths both equal hidden',
        'primary_outcome': 'world persistence capped at 10000, on the shared held-out map for each trial',
        'predeclared_contrasts': ['hidden192-64 at encoder32', 'hidden192-64 at encoder64',
            'encoder64-32 at hidden64', 'encoder64-32 at hidden192',
            'hidden-by-encoder interaction', 'session20-5 in cohortB'],
        'statistics': 'trial-level paired mean differences, medians, win fractions and pointwise bootstrap intervals; repeated sessions are not independent trials',
        'external_pretraining': False, 'task_rewards_added': False}


def prepare_replication(root, cfg, study='rep20-20261012'):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(cfg, study)
    path = study_path(root, study, 'plan.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved[k] != v for k, v in plan.items()):
            raise ValueError('existing replication plan differs')
        return saved
    total = 0
    expected = {}
    for name, spec in plan['protocols'].items():
        protocol = prepare_study(root, name, Config.from_saved(spec['config']),
            sizes=spec['sizes'], sessions=spec['sessions'], milestones=spec['milestones'],
            evaluation_seed=spec['evaluation_seed'])
        for model in protocol['models'].values():
            bank_bytes = Path(model['initial_bank']).stat().st_size
            total += 2 * bank_bytes * (spec['sessions'] + len(spec['milestones']))
            expected[f'h{model["hidden"]:03d}-e{model["encoder_width"]:03d}'] = model['parameters_per_agent']
        print(json.dumps({'prepared': name, 'worlds': protocol['independent_chains_per_size']}), flush=True)
    free = shutil.disk_usage(root).free
    if total * 1.2 > free:
        raise OSError(f'checkpoint storage estimate {total} exceeds available budget {free}')
    plan['parameters_per_individual'] = expected
    plan['estimated_initial_and_final_checkpoint_bytes'] = total
    plan['free_disk_bytes_at_preparation'] = free
    plan['plan_hash'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic_json(path, plan)
    return plan
