"""Aggregate twenty independent paired trials per requested condition."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def paired_difference(lookup, terms, seed):
    available = set.intersection(*(set(lookup[cell]) for cell, _ in terms))
    ids = sorted(available)
    values = np.array([sum(weight * lookup[cell][i]['persistence_capped'] for cell, weight in terms) for i in ids])
    result = {'n': len(ids), 'trial_ids': ids, 'differences': values.tolist(),
        'mean': float(values.mean()) if len(ids) else None,
        'median': float(np.median(values)) if len(ids) else None,
        'positive_fraction': float((values > 0).mean()) if len(ids) else None,
        'bootstrap_mean_95_interval': None}
    if len(ids) >= 2:
        rng = np.random.default_rng(seed)
        groups = {}
        for k, trial in enumerate(ids):
            gpu = lookup[terms[0][0]][trial]['rocr']
            assert all(lookup[cell][trial]['rocr'] == gpu for cell, _ in terms)
            groups.setdefault(gpu, []).append(k)
        totals = np.zeros(10000)
        for indices in groups.values():
            block = values[indices]
            totals += block[rng.integers(0, len(block), size=(10000, len(block)))].sum(1)
        means = totals / len(ids)
        result['bootstrap_mean_95_interval'] = np.quantile(means, [.025, .975]).tolist()
    return result


def publish_comparisons(root, plan, manifests):
    def publish(name, left_job, right_job, left_session=None, right_session=None):
        a, b = manifests.get(left_job['id']), manifests.get(right_job['id'])
        if not a or not b:
            return
        phase = 'evaluations'
        if left_session is None:
            target = left_job['milestones'][0]
            if str(target) in a.get(phase, {}) and str(target) in b.get(phase, {}):
                left_session = right_session = target
            else:
                phase = 'sessions'
                common = set(a[phase]) & set(b[phase])
                common = [int(n) for n in common if all((root / m[phase][str(n)]['run'] / 'latest.json').exists() for m in (a, b))]
                if not common:
                    return
                left_session = right_session = max(common)
        ra, rb = a.get(phase, {}).get(str(left_session)), b.get(phase, {}).get(str(right_session))
        if not ra or not rb or not all((root / r['run'] / 'latest.json').exists() for r in (ra, rb)):
            return
        label = lambda j, n: f"hidden {j['hidden']}・入力 {j['encoder_width']}・{n}セッション（試行{j['trial_offset']+1}–{j['trial_offset']+j['worlds']}）"
        spec = {'learned': ra['run'], 'control': rb['run'],
            'learned_chain': ra['parts'], 'control_chain': rb['parts'],
            'display_labels': {'learned': label(left_job, left_session), 'control': label(right_job, right_session)},
            'planned_sessions': max(left_job['sessions'], right_job['sessions']),
            'session': left_session if left_session == right_session else None,
            'phase': phase, 'study': plan['study']}
        save(root / f'{plan["study"]}-{name}.json', spec)
        if name == 'hidden-e032-b00':
            save(root / 'comparison.json', spec)
    for block in plan['blocks']:
        jobs = {(j['cohort'], j['hidden'], j['encoder_width']): j for j in plan['jobs'] if j['block'] == block['id']}
        for e in (32, 64):
            publish(f'hidden-e{e:03d}-{block["id"]}', jobs['A', 64, e], jobs['A', 192, e])
        for h in (64, 192):
            publish(f'encoder-h{h:03d}-{block["id"]}', jobs['A', h, 32], jobs['A', h, 64])
        publish(f'sessions-{block["id"]}', jobs['B', 64, 32], jobs['B', 64, 32], 5, 20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs')
    p.add_argument('--study', default='rep20-20261012')
    args = p.parse_args()
    root = Path(args.root)
    plan = read(root / f'{args.study}-plan.json')
    out = root / f'{args.study}-analysis'
    out.mkdir(exist_ok=True)
    manifests, rows, progress = {}, [], {}
    lookup = {c['id']: {} for c in plan['cells']}
    for job in plan['jobs']:
        path = root / f'{job["study"]}-h{job["hidden"]:03d}.json'
        if not path.exists():
            progress[job['id']] = {'status': 'pending', 'completed_sessions': 0, 'planned_sessions': job['sessions']}
            continue
        m = read(path)
        manifests[job['id']] = m
        progress[job['id']] = {'status': m['status'], 'completed_sessions': sum(r['status'] == 'complete' for r in m['sessions'].values()),
            'planned_sessions': job['sessions'], 'phase': m.get('phase'), 'active_run': m.get('active_run')}
        for n in job['milestones']:
            record = m.get('evaluations', {}).get(str(n))
            if not record or record.get('status') != 'complete':
                continue
            meta = read(root / record['run'] / 'metadata.json')
            assert meta['architecture']['encoder_width'] == job['encoder_width']
            assert meta['architecture']['gru_hidden_size'] == job['hidden']
            assert meta['compute']['precision'] == 'fp32'
            cell = f"{job['cohort']}-h{job['hidden']:03d}-e{job['encoder_width']:03d}-s{n:03d}"
            for wi in range(job['worlds']):
                trial = job['trial_offset'] + wi + 1
                row = {'cell': cell, 'cohort': job['cohort'], 'hidden': job['hidden'],
                    'encoder_width': job['encoder_width'], 'sessions': n, 'trial': trial,
                    'rocr': job['rocr'], 'run': record['run'],
                    'evaluation_map_seed': plan['seeds']['evaluation'] + (trial - 1) * 1009,
                    'persistence_capped': record['persistence_capped'][wi],
                    'censored': record['alive'][wi] > 0, 'alive': record['alive'][wi],
                    'food': record['food'][wi], 'food_at_1000': record['early_food']['1000'][wi],
                    'late_food': sum(record['late_food_windows'][wi]),
                    'stable_this_evaluation': record['stable_this_session'][wi],
                    'births': sum(record['births_clone_sexual'][wi]),
                    'parameters_per_agent': record['parameters_per_agent']}
                assert trial not in lookup[cell]
                lookup[cell][trial] = row
                rows.append(row)
    summaries = {}
    for cell in plan['cells']:
        data = list(lookup[cell['id']].values())
        y = [r['persistence_capped'] for r in data]
        summaries[cell['id']] = {**cell, 'n': len(data),
            'mean_persistence_capped': float(np.mean(y)) if y else None,
            'median_persistence_capped': float(np.median(y)) if y else None,
            'censored_worlds': sum(r['censored'] for r in data),
            'mean_food': float(np.mean([r['food'] for r in data])) if data else None,
            'stable_worlds': sum(r['stable_this_evaluation'] for r in data)}
    A = lambda h, e: f'A-h{h:03d}-e{e:03d}-s005'
    definitions = {
        'hidden at input32': [(A(192, 32), 1), (A(64, 32), -1)],
        'hidden at input64': [(A(192, 64), 1), (A(64, 64), -1)],
        'input at hidden64': [(A(64, 64), 1), (A(64, 32), -1)],
        'input at hidden192': [(A(192, 64), 1), (A(192, 32), -1)],
        'interaction': [(A(192, 64), 1), (A(192, 32), -1), (A(64, 64), -1), (A(64, 32), 1)],
        'sessions20 minus5': [('B-h064-e032-s020', 1), ('B-h064-e032-s005', -1)]}
    contrasts = {name: paired_difference(lookup, terms, 121020 + i)
        for i, (name, terms) in enumerate(definitions.items())}
    complete = all(s['n'] == 20 for s in summaries.values()) and all(p['status'] == 'complete' for p in progress.values())
    failed = any(p['status'] == 'failed' for p in progress.values())
    result = {'study': args.study, 'status': 'failed' if failed else 'complete' if complete else 'running',
        'completed_primary_outcomes': len(rows), 'planned_primary_outcomes': 120,
        'cells': summaries, 'paired_contrasts': contrasts, 'jobs': progress,
        'note': '80 architecture outcomes plus 40 paired time outcomes; 100 learning chains. Intervals resample paired trials within fixed GPU blocks, are pointwise exploratory intervals, and are not adjusted for multiple contrasts.'}
    save(out / 'results.json', result)
    save(root / f'{args.study}-status.json', result)
    publish_comparisons(root, plan, manifests)
    if rows:
        temp = out / 'trials.csv.tmp'
        with temp.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temp.replace(out / 'trials.csv')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for x, cell in enumerate(plan['cells']):
        values = [r['persistence_capped'] for r in lookup[cell['id']].values()]
        if values:
            jitter = np.linspace(-.15, .15, len(values))
            axes[0].scatter(x + jitter, values, s=18, alpha=.55)
            axes[0].scatter(x, np.mean(values), marker='D', color='black')
    axes[0].set_xticks(range(6), labels=['A\nh64/e32', 'A\nh192/e32', 'A\nh64/e64', 'A\nh192/e64', 'B\n5回', 'B\n20回'])
    axes[0].set(title='条件別：独立系列の共通マップ評価', ylabel='存続tick（上限10,000）', ylim=(0, 10500))
    for i, (name, c) in enumerate(contrasts.items()):
        if c['n']:
            axes[1].scatter(c['mean'], i, color='#30789e')
            if c['bootstrap_mean_95_interval']:
                lo, hi = c['bootstrap_mean_95_interval']; axes[1].plot([lo, hi], [i, i], color='#30789e')
    axes[1].axvline(0, color='#777777', linewidth=1)
    axes[1].set_yticks(range(6), labels=['hidden増・入力32', 'hidden増・入力64', '入力増・hidden64', '入力増・hidden192', '交互作用', '20−5セッション'])
    axes[1].set(title='対応する系列の平均差と95% bootstrap区間', xlabel='存続tickの差（右が改善）')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"20系列比較：{len(rows)}/120件完了", fontsize=13)
    temp = out / 'replicates.png.tmp'
    fig.savefig(temp, format='png', dpi=160); temp.replace(out / 'replicates.png'); plt.close(fig)
    lines = ['# 20系列によるモデル幅・セッション数の比較', '',
        f"状態：{result['status']}。条件別評価 {len(rows)}/120件完了。", '',
        '入力幅32/64は入力エンコーダ。GRU状態幅はhiddenと同じ。Aは4条件×20系列、Bは別の20系列の5/20時点。独立した学習系列は100、評価結果は120。', '',
        '| GPU（ROCR） | 完了ジョブ/5 | 実行中の構成・セッション |', '| --- | ---: | --- |']
    for gpu in ('0', '1', '2'):
        jobs = [j for j in plan['jobs'] if j['rocr'] == gpu]
        active = [(j, progress[j['id']]) for j in jobs if progress[j['id']]['status'] not in ('pending', 'complete')]
        text = ' / '.join(f"{j['cohort']} h{j['hidden']}/e{j['encoder_width']}: {s['completed_sessions']}/{j['sessions']} ({s.get('phase', s['status'])})" for j, s in active) or '次の条件を準備中、または完了'
        lines.append(f"| {gpu} | {sum(progress[j['id']]['status']=='complete' for j in jobs)} | {text} |")
    lines += ['',
        '| 条件 | hidden | 入力幅 | セッション | 完了/20 | 平均存続tick | 中央値 | 10,000tick時点存続 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt = lambda x: '—' if x is None else f'{x:,.1f}'
    for s in summaries.values():
        lines.append(f"| {s['cohort']} | {s['hidden']} | {s['encoder_width']} | {s['session']} | {s['n']} | {fmt(s['mean_persistence_capped'])} | {fmt(s['median_persistence_capped'])} | {s['censored_worlds']} |")
    lines += ['', '| 対応比較 | 対応数/20 | 平均差tick | 95% bootstrap区間 |', '| --- | ---: | ---: | --- |']
    for name, c in contrasts.items():
        interval = c['bootstrap_mean_95_interval']
        lines.append(f"| {name} | {c['n']} | {fmt(c['mean'])} | {'—' if interval is None else ' 〜 '.join(fmt(v) for v in interval)} |")
    lines += ['', '区間はGPUブロック内で独立した試行番号ごとの対応差を再標本化する。途中結果は一部のGPUブロックのみの場合がある。区間は個別の探索的区間で、多重比較の調整はしていない。10,000は全滅時刻とは限らず、上限で打ち切った存続。', '',
        '![条件と対応差](replicates.png)', '', '[JSON](results.json) · [試行別CSV](trials.csv)', '']
    temp = out / 'report.md.tmp'; temp.write_text('\n'.join(lines)); temp.replace(out / 'report.md')
    print(json.dumps({'status': result['status'], 'outcomes': len(rows),
        'completed_jobs': sum(p['status'] == 'complete' for p in progress.values())}), flush=True)


if __name__ == '__main__':
    main()
