"""Publish descriptive scaling results, preserving chain dependence and censoring."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='scale40-20261011')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.output or root / f'{args.study}-analysis')
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / f'{args.study}-protocol.json').read_text())
    report = {'protocol': protocol, 'models': {},
        'note': 'Three independent chains per size; sessions and reused evaluation maps are dependent observations. Persistence is censored at the session horizon.'}
    rows = []
    manifests = {}
    for hidden in protocol['sizes']:
        path = root / f'{args.study}-h{hidden:03d}.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        manifests[hidden] = manifest
        model = {**protocol['models'][str(hidden)], 'status': manifest['status'],
            'completed_sessions': 0, 'milestones': {}, 'stability': manifest.get('stability', [])}
        for role, collection in [('training', 'sessions'), ('evaluation', 'evaluations')]:
            for number, record in sorted(manifest.get(collection, {}).items(), key=lambda p: int(p[0])):
                if record.get('status') != 'complete':
                    continue
                if role == 'training':
                    model['completed_sessions'] += 1
                for wi in range(protocol['independent_chains_per_size']):
                    row = {'hidden': hidden, 'parameters': model['parameters_per_agent'],
                        'role': role, 'session': int(number), 'chain': wi, 'seed': record['seed'],
                        'run': record['run'], 'food': record['food'][wi],
                        'alive_at_horizon': record['alive'][wi],
                        'extinction_tick': record['extinction_tick'][wi],
                        'persistence_capped': record['persistence_capped'][wi],
                        'late_food': sum(record['late_food_windows'][wi]),
                        'stable_this_session': record['stable_this_session'][wi],
                        'food_at_1000': record['early_food'].get('1000', [None] * protocol['independent_chains_per_size'])[wi],
                        'prediction_loss_population_mean': record['prediction_loss'],
                        'prediction_diagnostic_complete': record['prediction_diagnostic_complete']}
                    rows.append(row)
                if role == 'evaluation':
                    model['milestones'][number] = {k: record[k] for k in
                        ('run', 'food', 'alive', 'extinction_tick', 'persistence_capped',
                         'early_food', 'late_food_windows', 'stable_this_session', 'prediction_loss')}
        report['models'][str(hidden)] = model
    report['status'] = 'complete' if len(manifests) == len(protocol['sizes']) and all(m['status'] == 'complete' for m in manifests.values()) else 'running'
    if any(m['status'] == 'failed' for m in manifests.values()):
        report['status'] = 'failed'
    atomic(out / 'results.json', report)
    if rows:
        temp = out / 'world-sessions.csv.tmp'
        with temp.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(out / 'world-sessions.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#30789e', '#d18538', '#658553']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for index, hidden in enumerate(protocol['sizes']):
        label = f"{hidden}: {protocol['models'][str(hidden)]['parameter_ratio']:.2f}x parameters"
        color = colors[index % len(colors)]
        for row_index, role in enumerate(('training', 'evaluation')):
            selected = [r for r in rows if r['hidden'] == hidden and r['role'] == role]
            numbers = sorted({r['session'] for r in selected})
            for col, metric in enumerate(('food', 'persistence_capped', 'late_food')):
                means = [np.mean([r[metric] for r in selected if r['session'] == n]) for n in numbers]
                axes[row_index, col].plot(numbers, means, '-o', color=color, label=label, markersize=3)
                for r in selected:
                    axes[row_index, col].scatter(r['session'], r[metric], s=12, color=color, alpha=.3)
    for row_index, role in enumerate(('Training maps', 'Common evaluation maps')):
        for col, title in enumerate(('Total food / world', 'Persistence (censored)', 'Food in final 3k ticks')):
            axes[row_index, col].set(title=f'{role}: {title}', xlabel='Training sessions completed')
            axes[row_index, col].grid(alpha=.2)
            if row_index:
                axes[row_index, col].set_xticks(protocol['milestones'])
    axes[0, 0].legend()
    fig.suptitle('Model size and selection sessions | dots: independent chains; lines: descriptive means')
    figure = out / 'scaling.png.tmp'
    fig.savefig(figure, format='png', dpi=160)
    figure.replace(out / 'scaling.png')
    plt.close(fig)
    lines = ['# モデル規模・セッション数の比較結果', '',
        f"状態：{report['status']}。各規模{protocol['independent_chains_per_size']}独立系列。数値は記述値で、セッションを独立標本として扱わない。", '',
        '| hidden | パラメータ数 | 完了セッション | 評価済み節目 |',
        '| --- | ---: | ---: | --- |']
    for hidden, model in report['models'].items():
        lines.append(f"| {hidden} | {model['parameters_per_agent']:,} | {model['completed_sessions']}/{protocol['planned_sessions']} | {', '.join(model['milestones']) or '未到達'} |")
    lines.extend(['', '共通マップ評価はL3による生涯学習を許す独立コピーで行い、更新された重み・成績を訓練へ戻さない。同じ評価マップを節目ごとに再使用する。', '',
        '| hidden | 学習セッション数 | 総採食 W0 / W1 / W2 | 存続tick（上限打切り） | 最後3区間の採食合計 | 終了時生存数 |',
        '| --- | ---: | --- | --- | --- | --- |'])
    for hidden, model in report['models'].items():
        for number, row in model['milestones'].items():
            text = lambda values: ' / '.join(map(str, values))
            lines.append(f"| {hidden} | {number} | {text(row['food'])} | {text(row['persistence_capped'])} | {text([sum(w) for w in row['late_food_windows']])} | {text(row['alive'])} |")
    lines.extend(['', '![訓練と共通マップ評価](scaling.png)', '',
        '[機械可読の結果](results.json) · [世界・セッション別CSV](world-sessions.csv)', ''])
    temp = out / 'report.md.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(out / 'report.md')
    print(json.dumps({'status': report['status'], 'completed': {h: m['completed_sessions'] for h, m in report['models'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
