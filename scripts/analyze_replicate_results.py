"""Audit final twenty-trial results and reconstruct individual survival from events."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def interval(values, groups, seed=121046):
    values, groups = np.asarray(values, float), np.asarray(groups)
    rng = np.random.default_rng(seed)
    totals = np.zeros(20000)
    for group in sorted(set(groups)):
        block = values[groups == group]
        totals += block[rng.integers(0, len(block), size=(len(totals), len(block)))].sum(1)
    return np.quantile(totals / len(values), [.025, .975]).tolist()


def exact_signflip(differences):
    d = np.asarray(differences, dtype=np.int64)
    if d.ndim == 1:
        d = d[:, None]
    n = d.shape[0]
    observed = np.abs(d.sum(0))
    count = np.zeros(d.shape[1], dtype=np.int64)
    for start in range(0, 2 ** n, 2 ** 15):
        numbers = np.arange(start, min(start + 2 ** 15, 2 ** n), dtype=np.uint32)
        signs = ((numbers[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int64) * 2 - 1
        count += (np.abs(signs @ d) >= observed).sum(0)
    return count / (2 ** n)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return result


def reconstruct(root, run, cfg):
    births = [{i: 0 for i in range(1, cfg['initial'] + 1)} for _ in range(cfg['worlds'])]
    birth_ticks = [[] for _ in births]
    deaths = [[] for _ in births]
    for line in (root / run / 'events.jsonl').read_text().splitlines():
        e = json.loads(line)
        wi = e['world']
        if e['event'] == 'birth':
            assert e['id'] not in births[wi]
            births[wi][e['id']] = e['tick']
            birth_ticks[wi].append(e['tick'])
        else:
            age = e['tick'] - births[wi][e['id']]
            deaths[wi].append({'tick': e['tick'], 'age': age,
                'birth_tick': births[wi][e['id']], 'cause': e['cause']})
    result = []
    for wi, history in enumerate(deaths):
        assert len(history) == len(births[wi])
        last_tick = max(e['tick'] for e in history)
        last = [e for e in history if e['tick'] == last_tick]
        result.append({'agent_ticks': sum(e['age'] for e in history),
            'mean_individual_lifetime': float(np.mean([e['age'] for e in history])),
            'last_survivor_age': float(np.mean([e['age'] for e in last])),
            'last_survivor_birth_tick': float(np.mean([e['birth_tick'] for e in last])),
            'last_death_age_cause': all(e['cause'] == 'age' for e in last),
            'age_deaths': sum(e['cause'] == 'age' for e in history),
            'mean_birth_tick': float(np.mean(birth_ticks[wi])) if birth_ticks[wi] else 0.,
            'max_birth_tick': max(birth_ticks[wi], default=0), 'death_count': len(history)})
    last_metric = json.loads((root / run / 'metrics.jsonl').read_text().splitlines()[-1])
    decisions = round(last_metric['agent_decisions_per_second'] * last_metric['elapsed_seconds'])
    assert sum(e['agent_ticks'] for e in result) == decisions
    return result, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='rep20-20261012')
    args = parser.parse_args()
    root = Path(args.root)
    study = args.study
    out = root / f'{study}-final-analysis'
    out.mkdir(exist_ok=True)
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    plan = json.loads((root / f'{study}-plan.json').read_text())
    auto = json.loads((root / f'{study}-analysis/results.json').read_text())
    validation = json.loads((root / f'{study}-validation.json').read_text())
    execution = json.loads((root / f'{study}-execution.json').read_text())
    assert auto['status'] == 'complete' and auto['completed_primary_outcomes'] == 120
    assert execution['returncodes'] == [0, 0, 0]
    df = pd.read_csv(root / f'{study}-analysis/trials.csv')
    assert len(df) == 120 and not df.censored.any() and not df.alive.any()
    for _, rows in df.groupby('cell'):
        assert sorted(rows.trial.tolist()) == list(range(1, 21))
    hashes = {}
    paths = [root / f'{study}-{s}.json' for s in ('plan', 'execution', 'validation')]
    paths += [root / f'{study}-analysis/results.json', root / f'{study}-analysis/trials.csv']
    audit = {'training_runs': 0, 'evaluation_runs': 0, 'training_world_sessions': 0,
        'evaluation_world_sessions': 0, 'training_censored': [],
        'training_stable_world_sessions': 0, 'maximum_consecutive_stable_sessions': 0,
        'source_hash': validation['source_hash']}
    prediction = []
    for job in plan['jobs']:
        manifest_path = root / f"{job['study']}-h{job['hidden']:03d}.json"
        protocol_path = root / f"{job['study']}-protocol.json"
        paths += [manifest_path, protocol_path]
        m = json.loads(manifest_path.read_text())
        protocol = json.loads(protocol_path.read_text())
        assert m['status'] == 'complete'
        audit['maximum_consecutive_stable_sessions'] = max(audit['maximum_consecutive_stable_sessions'],
            max(s['best_consecutive_stable_sessions'] for s in m['stability']))
        for role, collection in [('training', 'sessions'), ('evaluation', 'evaluations')]:
            for number, record in m[collection].items():
                assert record['status'] == 'complete' and record['prediction_diagnostic_complete']
                assert len(record['parts']) == 1
                run = record['run']
                meta = json.loads((root / run / 'metadata.json').read_text())
                summary = json.loads((root / run / 'summary.json').read_text())
                cfg = json.loads((root / run / 'config.json').read_text())
                assert summary['status'] == 'complete' and summary['tick'] == 10000
                assert summary['eaten'] == record['food'] and summary['alive'] == record['alive']
                assert meta['source_hash'] == validation['source_hash']
                assert meta['architecture']['encoder_width'] == job['encoder_width']
                assert meta['architecture']['gru_hidden_size'] == job['hidden']
                assert meta['rocr_visible_devices'] == job['rocr']
                assert meta['compute']['precision'] == 'fp32' and meta['compute']['execution'] == 'compiled'
                assert cfg['hidden'] == job['hidden'] and cfg['encoder_width'] == job['encoder_width']
                assert cfg['worlds'] == job['worlds']
                assert cfg['seed'] == (protocol['seed_sequence'][int(number) - 1] if role == 'training' else protocol['evaluation_seed'])
                assert meta['parameters_per_agent'] == plan['parameters_per_individual'][f'h{job["hidden"]:03d}-e{job["encoder_width"]:03d}']
                expected_digest = protocol['initial_world_digests'][number] if role == 'training' else protocol['evaluation']['initial_world_digest']
                assert record['initial_world_digest'] == expected_digest
                audit[f'{role}_runs'] += 1
                audit[f'{role}_world_sessions'] += job['worlds']
                if role == 'training':
                    audit['training_stable_world_sessions'] += sum(record['stable_this_session'])
                    audit['training_censored'] += [{'job': job['id'], 'session': int(number),
                        'trial': job['trial_offset'] + wi + 1} for wi, alive in enumerate(record['alive']) if alive]
                    continue
                life, decisions = reconstruct(root, run, cfg)
                prediction.append({'cohort': job['cohort'], 'hidden': job['hidden'],
                    'encoder_width': job['encoder_width'], 'session': int(number),
                    'gpu': job['rocr'], 'loss': record['prediction_loss'], 'decisions': decisions})
                rows = df[df.run == run].sort_values('trial')
                assert len(rows) == job['worlds']
                for wi, (index, row) in enumerate(rows.iterrows()):
                    detail = life[wi]
                    assert detail['death_count'] == cfg['initial'] + row.births
                    assert detail['last_survivor_age'] + detail['last_survivor_birth_tick'] == row.persistence_capped
                    for name, value in detail.items():
                        df.loc[index, name] = value
                    df.loc[index, 'metabolic_energy'] = detail['agent_ticks'] * cfg['metabolism']
                    df.loc[index, 'birth_overhead_energy'] = row.births * cfg['birth_overhead']
                    df.loc[index, 'other_energy_residual'] = cfg['initial'] * cfg['founder_energy'] + row.food * cfg['food_energy'] - detail['agent_ticks'] * cfg['metabolism'] - row.births * cfg['birth_overhead']
    assert audit['training_world_sessions'] == 800 and audit['evaluation_world_sessions'] == 120
    for p in paths:
        data = p.read_bytes()
        name = p.name if p.parent == root else f'{p.parent.name}-{p.name}'
        (inputs / name).write_bytes(data)
        hashes[str(p)] = hashlib.sha256(data).hexdigest()
    result = {'study': study, 'audit': audit, 'input_hashes': hashes,
        'primary': auto, 'secondary': {}, 'supplementary_tests': {},
        'methods': {'primary_intervals': 'predeclared GPU-stratified paired-trial percentile bootstrap, 10000 draws, pointwise 95%',
            'secondary_intervals': 'same trial/GPU strata, 20000 draws; exploratory',
            'signflip': 'post-hoc supplementary exact sign reversal of 20 paired differences, two-sided absolute sum, all 2^20 patterns; requires sign symmetry / within-pair exchangeability',
            'multiple_testing': 'Holm across the six predeclared contrast statistics for supplementary p-values',
            'energy_residual': 'movement plus discarded age-death energy plus food overflow, minus terminal clamp overshoot, with floating-point error; cannot identify movement alone',
            'prediction_loss': 'transition-weighted population averages on each model own trajectories, not matched observation difficulty'}}
    means = df.groupby('cell', sort=False).mean(numeric_only=True)
    for cell, rows in df.groupby('cell', sort=False):
        result['secondary'][cell] = {name: float(rows[name].mean()) for name in
            ('agent_ticks', 'mean_individual_lifetime', 'last_survivor_age', 'last_survivor_birth_tick',
             'food', 'food_at_1000', 'births', 'metabolic_energy', 'birth_overhead_energy', 'other_energy_residual')}
        result['secondary'][cell]['last_death_age_cause_count'] = int(rows.last_death_age_cause.sum())
    contrasts = auto['paired_contrasts']
    names = list(contrasts)
    differences = np.array([contrasts[name]['differences'] for name in names], dtype=np.int64).T
    pvalues = exact_signflip(differences)
    corrected = holm(pvalues)
    for name, p, adjusted in zip(names, pvalues, corrected):
        result['supplementary_tests'][name] = {'exact_signflip_p': float(p), 'holm_p': float(adjusted)}
    secondary_pairs = [('A-h064-e064-s005', 'A-h192-e064-s005'), ('B-h064-e032-s005', 'B-h064-e032-s020')]
    result['secondary_pairs'] = {}
    for a, b in secondary_pairs:
        left = df[df.cell == a].set_index('trial').sort_index()
        right = df[df.cell == b].set_index('trial').sort_index()
        differences = {}
        for name in ('agent_ticks', 'mean_individual_lifetime', 'last_survivor_age', 'last_survivor_birth_tick', 'food', 'food_at_1000', 'births'):
            d = right[name].to_numpy() - left[name].to_numpy()
            differences[name] = {'mean': float(d.mean()), 'mean_95_interval': interval(d, left.rocr.to_numpy()),
                'values': d.tolist()}
        result['secondary_pairs'][f'{b} minus {a}'] = differences
    result['prediction'] = prediction
    result['prediction_weighted'] = {}
    for n in (5, 20):
        selected = [p for p in prediction if p['cohort'] == 'B' and p['session'] == n]
        result['prediction_weighted'][str(n)] = sum(p['loss'] * p['decisions'] for p in selected) / sum(p['decisions'] for p in selected)
    df.to_csv(out / 'trials-with-lifetimes.csv', index=False)
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    colors = {64: '#30789e', 192: '#c17a20'}
    for h in (64, 192):
        for trial in range(1, 21):
            values = [float(df[(df.cohort == 'A') & (df.hidden == h) & (df.encoder_width == e) & (df.trial == trial)].persistence_capped.iloc[0]) for e in (32, 64)]
            axes[0, 0].plot([32, 64], values, color=colors[h], alpha=.12, linewidth=.8)
        values = [auto['cells'][f'A-h{h:03d}-e{e:03d}-s005']['mean_persistence_capped'] for e in (32, 64)]
        axes[0, 0].plot([32, 64], values, '-o', color=colors[h], linewidth=2.5, label=f'hidden {h}')
    axes[0, 0].set(title='5セッション：入力幅を増やしても改善は見えない', xlabel='入力エンコーダ幅', ylabel='最後の個体までの存続tick', xticks=[32, 64])
    axes[0, 0].legend()
    labels = ['hidden増・入力32', 'hidden増・入力64', '入力増・hidden64', '入力増・hidden192', '交互作用', '20−5セッション']
    for i, name in enumerate(names):
        c = contrasts[name]
        color = '#b56235' if name == 'hidden at input64' else '#30789e'
        axes[0, 1].plot(c['bootstrap_mean_95_interval'], [i, i], color=color, linewidth=2)
        axes[0, 1].scatter(c['mean'], i, color=color)
    axes[0, 1].axvline(0, color='#777777', linewidth=1)
    axes[0, 1].set(title='対応する20系列の平均差・個別95%区間', yticks=range(6), yticklabels=labels, xlabel='存続tickの差（右が改善）')
    axes[0, 1].invert_yaxis()
    left = df[df.cell == 'B-h064-e032-s005'].set_index('trial').sort_index()
    right = df[df.cell == 'B-h064-e032-s020'].set_index('trial').sort_index()
    for trial in left.index:
        axes[1, 0].plot([5, 20], [left.loc[trial, 'persistence_capped'], right.loc[trial, 'persistence_capped']], '-o', color='#777777', alpha=.3, markersize=3)
    axes[1, 0].plot([5, 20], [left.persistence_capped.mean(), right.persistence_capped.mean()], '-o', color=colors[64], linewidth=3, label='20系列平均')
    axes[1, 0].set(title='hidden64・入力32：5→20セッションで平均は横ばい', xticks=[5, 20], xlabel='学習セッション数', ylabel='最後の個体までの存続tick')
    axes[1, 0].legend()
    a, b = 'A-h064-e064-s005', 'A-h192-e064-s005'
    metrics = ['persistence_capped', 'agent_ticks', 'mean_individual_lifetime', 'food', 'food_at_1000', 'births']
    percent = [100 * (means.loc[b, name] / means.loc[a, name] - 1) for name in metrics]
    axes[1, 1].barh(range(len(metrics)), percent, color=['#b56235', '#30789e', '#30789e', '#48815c', '#48815c', '#777777'])
    axes[1, 1].set(yticks=range(6), yticklabels=['最後の個体まで', '全個体の総生存tick', '個体平均寿命', '総採食', '最初1000tick採食', '出生数'],
        xlabel='hidden64→192による平均値の変化（%）', title='入力64：総生存量はほぼ同じでも最後の存続は短い')
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(0, color='#777777', linewidth=1)
    for i, value in enumerate(percent):
        axes[1, 1].annotate(f'{value:+.1f}%', (value, i), xytext=(4 if value >= 0 else -4, 0), textcoords='offset points', va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    axes[1, 1].set_xlim(-18, 24)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle('各条件20系列：120件の共通マップ評価が完了', fontsize=13)
    fig.savefig(out / 'results.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'audit': audit, 'supplementary_tests': result['supplementary_tests'],
        'secondary_pairs': {k: {n: v[n]['mean'] for n in ('agent_ticks', 'mean_individual_lifetime', 'last_survivor_age', 'last_survivor_birth_tick')} for k, v in result['secondary_pairs'].items()},
        'prediction_weighted': result['prediction_weighted']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
