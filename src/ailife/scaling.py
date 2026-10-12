"""Matched model-size and session-count study; selection stays outside learning."""
from __future__ import annotations

import dataclasses
import fcntl
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import torch

from .config import Config
from .learner import Learners
from .observer import atomic_json, replay_schedule
from .runner import checkpoint, run
from .selection import seed_population
from .world import World


def read_json(path):
    return json.loads(Path(path).read_text())


def arm_name(hidden):
    return f'h{hidden:03d}'


def study_path(root, study, suffix):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', study):
        raise ValueError('study must be a simple filename prefix')
    return Path(root) / f'{study}-{suffix}'


def session_config(protocol, hidden, number):
    cfg = Config.from_saved(protocol['config'])
    return dataclasses.replace(cfg, hidden=hidden,
        seed=protocol['seed_sequence'][number - 1],
        observer_seed=cfg.observer_seed + (number - 1) * 17)


def _check_untrained(learner):
    for name in ('hidden', 'last_action', 'obs_n', 'reward_n', 'adam_count'):
        if getattr(learner, name).any():
            raise AssertionError(f'nonzero initial history: {name}')
    for name, value in learner.params.items():
        if not torch.equal(value, learner.genome[name]):
            raise AssertionError(f'nonrandom-training initialization: {name}')


def prepare_study(root, study, cfg, sizes=(64, 128, 192), sessions=40,
                  milestones=(5, 10, 20, 40), seed_step=1000003, evaluation_seed=9000001):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = study_path(root, study, 'protocol.json')
    sizes = tuple(sorted(set(sizes)))
    milestones = tuple(sorted(set(milestones)))
    if not sizes or min(sizes) < 1 or sessions < 1:
        raise ValueError('invalid study size')
    if not milestones or min(milestones) < 1 or max(milestones) > sessions:
        raise ValueError('milestones must lie within the study')
    if cfg.mode != 'L3' or cfg.elite_count < 1 or cfg.ticks < 3 * cfg.log_ticks:
        raise ValueError('study requires L3, elites and three final logging windows')
    if cfg.ticks % cfg.log_ticks:
        raise ValueError('session horizon must be a multiple of the logging window')
    torch.set_num_threads(cfg.threads)
    request = {'version': 1, 'study': study, 'config': cfg.as_dict(),
        'sizes': list(sizes), 'planned_sessions': sessions,
        'milestones': list(milestones), 'seed_step': seed_step,
        'evaluation_seed': evaluation_seed}
    with study_path(root, study, 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            prior = read_json(path)
            if any(prior[k] != v for k, v in request.items()):
                raise ValueError('existing study protocol differs; use a new study name')
            return prior
        initial = study_path(root, study, 'initial')
        initial.mkdir(exist_ok=True)
        protocol = {**request,
            'seed_sequence': [cfg.seed + i * seed_step for i in range(sessions)],
            'independent_chains_per_size': cfg.worlds,
            'arm': 'acquired', 'external_pretraining': False,
            'task_rewards_added': False,
            'selection': 'uniform up to K survivors at horizon; else K latest deaths, tick then ID; food is not fitness',
            'session_transfer': 'acquired weights, rates, evaluator, observation calibration rebased as genome; round-robin private clones',
            'within_session_birth': 'birth-time genome and calibration; existing mutation and reproduction rules',
            'reset_at_boundary': ['world', 'body', 'age', 'energy', 'hidden', 'previous_action',
                'optimizer', 'eligibility_traces', 'reward_statistics'],
            'width_change': f'GRU hidden, feature and downstream projections; encoder width {cfg.encoder_width}',
            'stability': 'alive at horizon and food in each of final 3 log windows, for 3 consecutive sessions',
            'statistics_note': 'sessions are dependent observations within each chain; not independent replicates',
            'models': {}, 'initial_world_digests': {}}
        if evaluation_seed in protocol['seed_sequence']:
            raise ValueError('evaluation seed overlaps the training sequence')
        evalcfg = dataclasses.replace(cfg, seed=evaluation_seed, elite_count=0)
        protocol['evaluation'] = {'seed': evaluation_seed,
            'world_seeds': [evaluation_seed + wi * 1009 for wi in range(cfg.worlds)],
            'initial_world_digest': World(evalcfg).digest(),
            'learning': 'same L3 learning within the evaluation episode',
            'feedback_to_training': False,
            'selection_from_evaluation': False,
            'note': 'same held-out map set at every milestone; descriptive reuse, not multiple independent test sets'}
        for number in range(1, sessions + 1):
            c = session_config(protocol, sizes[0], number)
            protocol['initial_world_digests'][str(number)] = World(c).digest()
        for hidden in sizes:
            c = session_config(protocol, hidden, 1)
            world, learner = World(c), Learners(c)
            _check_untrained(learner)
            digest = world.digest()
            if digest != protocol['initial_world_digests']['1']:
                raise AssertionError('initial physical worlds differ across sizes')
            info = {'type': 'model_session_scaling', 'study': study, 'hidden': hidden,
                'session': 1, 'contains_acquired_weights': False,
                'external_pretraining': False, 'initial_world_digest': digest,
                'initialization': 'counter-keyed random weights; no training history'}
            bank = initial / f'{arm_name(hidden)}-s001.pt'
            if bank.exists():
                raise FileExistsError(f'preserved incomplete preparation: {bank}')
            checkpoint(bank, c, world, learner, replay_schedule(c), initialization=info)
            protocol['models'][str(hidden)] = {'hidden': hidden,
                'encoder_width': c.encoder_width,
                'parameters_per_agent': learner.parameter_count(),
                'initial_bank': str(bank), 'initial_world_digest': digest}
            del world, learner
            gc.collect()
        reference = protocol['models'][str(sizes[0])]['parameters_per_agent']
        for model in protocol['models'].values():
            model['parameter_ratio'] = model['parameters_per_agent'] / reference
        protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        atomic_json(path, protocol)
        return protocol


def measurements(root, record, cfg):
    """Merge physical counters across resumed segments, without double-counting losses."""
    rows = {0: {'eaten': [0] * cfg.worlds}}
    loss_sum = decisions = elapsed = 0.
    diagnostic_complete = True
    parts = record['parts']
    for index, name in enumerate(parts):
        directory = Path(root) / name
        meta = read_json(directory / 'metadata.json')
        if index + 1 < len(parts):
            cutoff = read_json(Path(root) / parts[index + 1] / 'metadata.json')['start_tick']
        else:
            cutoff = read_json(directory / 'summary.json')['tick']
        valid = []
        for line in (directory / 'metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['tick'] <= cutoff:
                rows[row['tick']] = row
                valid.append(row)
        if valid:
            last = valid[-1]
            n = last['elapsed_seconds'] * last['agent_decisions_per_second']
            decisions += n
            loss_sum += n * last['mean_prediction_loss']
            elapsed += last['elapsed_seconds']
            diagnostic_complete &= last['tick'] == cutoff
        else:
            diagnostic_complete &= cutoff == meta['start_tick']
    summary = read_json(Path(root) / record['run'] / 'summary.json')
    if summary['status'] != 'complete' or summary['tick'] != cfg.ticks:
        raise ValueError('cannot measure an unfinished session as complete')
    bounds = [cfg.ticks - i * cfg.log_ticks for i in (3, 2, 1, 0)]
    late = [[rows[b]['eaten'][wi] - rows[a]['eaten'][wi]
        for a, b in zip(bounds, bounds[1:])] for wi in range(cfg.worlds)]
    early = {str(t): rows[t]['eaten'] for t in (1000, 2000, 3000) if t in rows}
    return {'status': 'complete', 'food': summary['eaten'], 'alive': summary['alive'],
        'extinction_tick': summary['extinction_tick'],
        'persistence_capped': [t if t >= 0 else cfg.ticks for t in summary['extinction_tick']],
        'early_food': early, 'late_food_windows': late,
        'stable_this_session': [bool(n and all(x > 0 for x in window))
            for n, window in zip(summary['alive'], late)],
        'births_clone_sexual': summary['births_clone_sexual'],
        'deaths_energy_age': summary['deaths_energy_age'],
        'prediction_loss': loss_sum / max(decisions, 1),
        'prediction_diagnostic_complete': diagnostic_complete,
        'elapsed_seconds_from_metrics': elapsed,
        'parameters_per_agent': summary['parameters_per_agent'],
        'device_name': summary['device_name'],
        'source_hash': summary['source_hash']}


def _stability(manifest, worlds):
    results = []
    for wi in range(worlds):
        streak = best = 0
        for _, row in sorted(manifest['sessions'].items(), key=lambda pair: int(pair[0])):
            if row.get('status') != 'complete':
                break
            streak = streak + 1 if row['stable_this_session'][wi] else 0
            best = max(best, streak)
        results.append({'world': wi, 'best_consecutive_stable_sessions': best, 'criterion_met': best >= 3})
    return results


def publish_study(root, study, protocol):
    root = Path(root)
    with study_path(root, study, 'publish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifests = {}
        for hidden in protocol['sizes']:
            path = study_path(root, study, f'{arm_name(hidden)}.json')
            if path.exists():
                manifests[hidden] = read_json(path)
        status = {'study': study, 'planned_sessions': protocol['planned_sessions'],
            'milestones': protocol['milestones'], 'models': {}}
        for hidden, manifest in manifests.items():
            complete = [int(n) for n, row in manifest['sessions'].items() if row.get('status') == 'complete']
            newest = max(map(int, manifest['sessions']), default=0)
            status['models'][str(hidden)] = {
                **protocol['models'][str(hidden)], 'status': manifest['status'],
                'completed_sessions': len(complete), 'latest_session': newest,
                'latest_run': manifest['sessions'][str(newest)]['run'] if newest else None,
                'completed_milestones': [n for n in protocol['milestones'] if n in complete],
                'completed_evaluations': [int(n) for n, r in manifest.get('evaluations', {}).items() if r.get('status') == 'complete'],
                'phase': manifest.get('phase'), 'active_run': manifest.get('active_run'),
                'stability': manifest.get('stability', [])}
        for i, left in enumerate(protocol['sizes']):
            for right in protocol['sizes'][i + 1:]:
                if left not in manifests or right not in manifests:
                    continue
                a, b = manifests[left], manifests[right]
                common = sorted(set(map(int, a['sessions'])) & set(map(int, b['sessions'])))
                available = [n for n in common if all((root / m['sessions'][str(n)]['run'] / 'latest.json').exists() for m in (a, b))]
                if not available:
                    continue
                targets = [(max(available), '')] + [(n, f's{n:03d}-') for n in protocol['milestones']
                    if n in available and all(m['sessions'][str(n)].get('status') == 'complete' for m in (a, b))]
                for number, tag in targets:
                    ra, rb = a['sessions'][str(number)], b['sessions'][str(number)]
                    label = lambda h: f"{protocol['models'][str(h)]['parameter_ratio']:.2f}倍モデル（hidden {h}, encoder {protocol['config'].get('encoder_width',32)}）"
                    spec = {'learned': ra['run'], 'control': rb['run'],
                        'learned_chain': ra['parts'], 'control_chain': rb['parts'],
                        'display_labels': {'learned': label(left), 'control': label(right)},
                        'session': number, 'planned_sessions': protocol['planned_sessions'],
                        'selection': protocol['selection'], 'study': study}
                    atomic_json(study_path(root, study, f'{tag}{arm_name(left)}-{arm_name(right)}.json'), spec)
                    if not tag and left == protocol['sizes'][0] and right == protocol['sizes'][-1]:
                        atomic_json(root / 'comparison.json', spec)
                for number in protocol['milestones']:
                    ra = a.get('evaluations', {}).get(str(number))
                    rb = b.get('evaluations', {}).get(str(number))
                    if not ra or not rb or not all((root / r['run'] / 'latest.json').exists() for r in (ra, rb)):
                        continue
                    spec = {'learned': ra['run'], 'control': rb['run'],
                        'learned_chain': ra['parts'], 'control_chain': rb['parts'],
                        'display_labels': {'learned': label(left) + '・共通マップ評価',
                            'control': label(right) + '・共通マップ評価'},
                        'session': number, 'planned_sessions': protocol['planned_sessions'],
                        'study': study, 'evaluation': protocol['evaluation']}
                    atomic_json(study_path(root, study, f'e{number:03d}-{arm_name(left)}-{arm_name(right)}.json'), spec)
        status['status'] = 'complete' if len(manifests) == len(protocol['sizes']) and all(m['status'] == 'complete' for m in manifests.values()) else 'running'
        if any(m['status'] == 'failed' for m in manifests.values()):
            status['status'] = 'failed'
        atomic_json(study_path(root, study, 'status.json'), status)


def run_evaluation(root, study, protocol, manifest, hidden, number, device):
    """Branch from training elites onto common maps; never feed evaluation back."""
    cfg = dataclasses.replace(session_config(protocol, hidden, 1),
        seed=protocol['evaluation_seed'], elite_count=0,
        observer_seed=protocol['config']['observer_seed'] + 10000019)
    records = manifest.setdefault('evaluations', {})
    key = str(number)
    record = records.get(key)
    manifest_path = study_path(root, study, f'{arm_name(hidden)}.json')
    if record and record.get('status') == 'complete':
        return
    if record:
        previous = root / record['run']
        summary = read_json(previous / 'summary.json') if (previous / 'summary.json').exists() else None
        if summary and summary['status'] == 'failed':
            raise RuntimeError(f'failed evaluation requires review: {previous}')
        if summary and summary['status'] == 'complete':
            record.update(measurements(root, record, cfg))
            return
        if not previous.exists():
            destination, resume = previous, Path(record['initial_bank'])
        else:
            resume = previous / 'checkpoint.pt'
            if not resume.exists():
                raise FileNotFoundError(f'evaluation has no checkpoint: {previous}')
            destination = root / f'{study}-{arm_name(hidden)}-e{number:03d}-r{len(record["parts"]):02d}'
            record['parts'].append(destination.name)
            record['run'] = destination.name
    else:
        source = manifest['sessions'][key]['run']
        pack = torch.load(root / source / 'elites.pt', map_location='cpu', weights_only=True)
        world, learner, assignments = seed_population(cfg, pack, 'acquired')
        if world.digest() != protocol['evaluation']['initial_world_digest']:
            raise AssertionError('evaluation physical state is not matched')
        initial = study_path(root, study, 'initial') / f'{arm_name(hidden)}-e{number:03d}.pt'
        if initial.exists():
            raise FileExistsError(f'preserved unregistered evaluation bank: {initial}')
        info = {'type': 'model_session_scaling_evaluation', 'study': study,
            'hidden': hidden, 'session': number, 'source_run': source,
            'contains_acquired_weights': True, 'external_pretraining': False,
            'feedback_to_training': False, 'initial_world_digest': world.digest(),
            'assignments': assignments}
        checkpoint(initial, cfg, world, learner, replay_schedule(cfg), initialization=info)
        del world, learner, pack
        gc.collect()
        destination = root / f'{study}-{arm_name(hidden)}-e{number:03d}'
        resume = initial
        record = {'run': destination.name, 'parts': [destination.name],
            'initial_bank': str(initial), 'source_run': source, 'seed': cfg.seed,
            'initial_world_digest': protocol['evaluation']['initial_world_digest'],
            'feedback_to_training': False, 'assignments': assignments}
        records[key] = record
    record['status'] = 'running'
    manifest['phase'], manifest['active_run'] = 'evaluation', destination.name
    atomic_json(manifest_path, manifest)
    publish_study(root, study, protocol)
    run(cfg, device, destination, resume=resume)
    record.update(measurements(root, record, cfg))
    atomic_json(manifest_path, manifest)
    publish_study(root, study, protocol)
    print(json.dumps({'study': study, 'hidden': hidden, 'evaluated_after_session': number,
        'food': record['food'], 'alive': record['alive'],
        'extinction_tick': record['extinction_tick']}, ensure_ascii=False), flush=True)
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def run_arm(root, study, hidden, device='cuda:0'):
    root = Path(root).resolve()
    protocol = read_json(study_path(root, study, 'protocol.json'))
    if hidden not in protocol['sizes']:
        raise ValueError('model is not in this study')
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest_path = study_path(root, study, f'{arm_name(hidden)}.json')
    with study_path(root, study, f'{arm_name(hidden)}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(manifest_path) if manifest_path.exists() else {
            'study': study, 'hidden': hidden, 'protocol_hash': protocol['protocol_hash'],
            'planned_sessions': protocol['planned_sessions'], 'sessions': {}}
        if manifest['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('manifest protocol differs')
        manifest['status'] = 'running'
        manifest['compute'] = {'precision': protocol['config']['precision'],
            'execution': protocol['config']['execution'], 'matmul_allow_tf32': False}
        manifest.pop('error', None)
        try:
            for number in range(1, protocol['planned_sessions'] + 1):
                key = str(number)
                cfg = session_config(protocol, hidden, number)
                record = manifest['sessions'].get(key)
                if record and record.get('status') == 'complete':
                    if number in protocol['milestones']:
                        run_evaluation(root, study, protocol, manifest, hidden, number, device)
                    continue
                if record:
                    previous = root / record['run']
                    summary_path = previous / 'summary.json'
                    summary = read_json(summary_path) if summary_path.exists() else None
                    if summary and summary['status'] == 'failed':
                        raise RuntimeError(f'failed session requires review: {previous}')
                    if summary and summary['status'] == 'complete':
                        record.update(measurements(root, record, cfg))
                        pack = torch.load(previous / 'elites.pt', map_location='cpu', weights_only=True)
                        record['selected'] = [[{k: r[k] for k in ('id', 'death_tick', 'selected_alive', 'age', 'eaten')}
                            for r in rows] for rows in pack['worlds']]
                        del pack
                        atomic_json(manifest_path, manifest)
                        if number in protocol['milestones']:
                            run_evaluation(root, study, protocol, manifest, hidden, number, device)
                        continue
                    resume = previous / 'checkpoint.pt'
                    if not resume.exists():
                        # A manifest may have been saved just before run() created its directory.
                        if not previous.exists():
                            resume = Path(record['initial_bank'])
                            destination = previous
                        else:
                            raise FileNotFoundError(f'preserved trial has no checkpoint: {previous}')
                    else:
                        destination = root / f'{study}-{arm_name(hidden)}-s{number:03d}-r{len(record["parts"]):02d}'
                        record['parts'].append(destination.name)
                        record['run'] = destination.name
                else:
                    initial = study_path(root, study, 'initial') / f'{arm_name(hidden)}-s{number:03d}.pt'
                    source = None
                    assignments = []
                    if number > 1:
                        prior = manifest['sessions'][str(number - 1)]
                        source = prior['run']
                        pack = torch.load(root / source / 'elites.pt', map_location='cpu', weights_only=True)
                        world, learner, assignments = seed_population(cfg, pack, 'acquired')
                        digest = world.digest()
                        if digest != protocol['initial_world_digests'][key]:
                            raise AssertionError('physical initialization changed across model sizes')
                        info = {'type': 'model_session_scaling', 'study': study, 'hidden': hidden,
                            'session': number, 'source_run': source, 'contains_acquired_weights': True,
                            'external_pretraining': False, 'initial_world_digest': digest,
                            'assignments': assignments}
                        if initial.exists():
                            raise FileExistsError(f'preserved unregistered initial bank: {initial}')
                        checkpoint(initial, cfg, world, learner, replay_schedule(cfg), initialization=info)
                        del world, learner, pack
                        gc.collect()
                    destination = root / f'{study}-{arm_name(hidden)}-s{number:03d}'
                    resume = initial
                    record = {'run': destination.name, 'parts': [destination.name], 'seed': cfg.seed,
                        'status': 'starting', 'initial_bank': str(initial), 'source_run': source,
                        'initial_world_digest': protocol['initial_world_digests'][key],
                        'assignments': assignments}
                    manifest['sessions'][key] = record
                record['status'] = 'running'
                manifest['phase'], manifest['active_run'] = 'training', destination.name
                atomic_json(manifest_path, manifest)
                publish_study(root, study, protocol)
                summary = run(cfg, device, destination, resume=resume)
                if summary['status'] != 'complete':
                    raise RuntimeError('session did not complete')
                record.update(measurements(root, record, cfg))
                pack = torch.load(destination / 'elites.pt', map_location='cpu', weights_only=True)
                record['selected'] = [[{k: r[k] for k in ('id', 'death_tick', 'selected_alive', 'age', 'eaten')}
                    for r in rows] for rows in pack['worlds']]
                del pack
                gc.collect()
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
                manifest['stability'] = _stability(manifest, cfg.worlds)
                atomic_json(manifest_path, manifest)
                publish_study(root, study, protocol)
                print(json.dumps({'study': study, 'hidden': hidden, 'completed_session': number,
                    'food': record['food'], 'alive': record['alive'],
                    'extinction_tick': record['extinction_tick']}, ensure_ascii=False), flush=True)
                if number in protocol['milestones']:
                    run_evaluation(root, study, protocol, manifest, hidden, number, device)
            manifest['status'] = 'complete'
            manifest['phase'] = 'complete'
            manifest['stability'] = _stability(manifest, cfg.worlds)
        except BaseException as exc:
            manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            manifest['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            atomic_json(manifest_path, manifest)
            publish_study(root, study, protocol)
    return manifest
