"""Descriptive size/time relationships, respecting repeated selection chains."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlations(x, y):
    return {'spearman_rho': float(spearmanr(x, y).statistic),
        'pearson_r': float(pearsonr(x, y).statistic)}


def trend(numbers, values):
    numbers, values = np.asarray(numbers), np.asarray(values, dtype=float)
    per_chain = [correlations(numbers, values[:, wi]) for wi in range(values.shape[1])]
    return {'sessions': numbers.tolist(), 'chain_correlations': per_chain,
        'mean_chain_spearman_rho': float(np.mean([r['spearman_rho'] for r in per_chain])),
        'pooled_descriptive': correlations(np.repeat(numbers, values.shape[1]), values.ravel()),
        'chain_linear_slopes_per_doubling': np.polyfit(np.log2(numbers), values, 1)[0].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='scale40-20261011')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.output or root / f'{args.study}-relations')
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    p = root / f'{args.study}-protocol.json'
    protocol = json.loads(p.read_text())
    paths = [p] + [root / f'{args.study}-h{h:03d}.json' for h in protocol['sizes']]
    input_hashes = {}
    for path in paths:
        data = path.read_bytes()
        (inputs / path.name).write_bytes(data)
        input_hashes[path.name] = hashlib.sha256(data).hexdigest()
    manifests = {h: json.loads(paths[i + 1].read_text()) for i, h in enumerate(protocol['sizes'])}
    complete = {h: {int(n): r for n, r in m['sessions'].items() if r['status'] == 'complete'}
        for h, m in manifests.items()}
    evaluations = {h: {int(n): r for n, r in m['evaluations'].items() if r['status'] == 'complete'}
        for h, m in manifests.items()}
    sessions = sorted(set.intersection(*(set(m) for m in complete.values())))
    milestones = sorted(set.intersection(*(set(m) for m in evaluations.values())))
    if len(sessions) != protocol['planned_sessions'] or milestones != protocol['milestones']:
        raise ValueError('this final analysis requires all planned sessions and evaluations')
    rows = []
    hashes = set()
    censoring = {'training': [], 'evaluation': []}
    for h in protocol['sizes']:
        for role, collection in [('training', complete[h]), ('evaluation', evaluations[h])]:
            for number, record in sorted(collection.items()):
                directory = root / record['run']
                meta = json.loads((directory / 'metadata.json').read_text())
                summary = json.loads((directory / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['tick'] == protocol['config']['ticks']
                assert summary['eaten'] == record['food'] and summary['alive'] == record['alive']
                assert summary['extinction_tick'] == record['extinction_tick']
                assert meta['compute']['precision'] == 'fp32' and meta['compute']['execution'] == 'compiled'
                assert meta['parameters_per_agent'] == protocol['models'][str(h)]['parameters_per_agent']
                assert record['prediction_diagnostic_complete']
                hashes.add(meta['source_hash'])
                expected = protocol['initial_world_digests'][str(number)] if role == 'training' else protocol['evaluation']['initial_world_digest']
                assert record['initial_world_digest'] == expected
                for wi in range(protocol['independent_chains_per_size']):
                    censored = record['extinction_tick'][wi] < 0
                    assert censored == (record['alive'][wi] > 0)
                    if censored:
                        censoring[role].append({'hidden': h, 'session': number, 'chain': wi,
                            'alive': record['alive'][wi], 'late_food_windows': record['late_food_windows'][wi]})
                    rows.append({'role': role, 'hidden': h, 'session': number, 'chain': wi,
                        'persistence_capped': record['persistence_capped'][wi], 'censored': censored,
                        'food': record['food'][wi], 'food_at_1000': record['early_food']['1000'][wi],
                        'late_food': sum(record['late_food_windows'][wi]),
                        'births': sum(record['births_clone_sexual'][wi]),
                        'prediction_loss_population_mean': record['prediction_loss']})
    assert len(hashes) == 1
    train = {h: np.array([complete[h][n]['persistence_capped'] for n in sessions]) for h in protocol['sizes']}
    ev = {h: np.array([evaluations[h][n]['persistence_capped'] for n in milestones]) for h in protocol['sizes']}
    result = {'study': args.study, 'input_hashes': input_hashes,
        'validation': {'training_world_sessions': sum(r['role'] == 'training' for r in rows),
            'evaluation_world_sessions': sum(r['role'] == 'evaluation' for r in rows),
            'source_hashes': sorted(hashes), 'censoring': censoring,
            'all_summaries_consistent': True, 'all_prediction_diagnostics_complete': True},
        'methods': {'outcome': 'last individual alive in each world; min(extinction tick, 10000)',
            'session_relation': 'Spearman per chain, then descriptive average; also pooled coefficients for transparency',
            'size_relation': 'paired differences on identical session/map/chain index; pooled Spearman descriptive only',
            'p_values': 'not reported: only 3 chains per size, repeated sessions and evaluation maps',
            'limitations': ['session count includes selection, mutation and inheritance, not learning alone',
                'encoder stays width 32; hidden changes only part of the architecture',
                'hidden192 used R9700; hidden64/128 used separate V620s',
                'no held-out evaluation at session zero; changes compare session5 to session40',
                'evaluation allows L3 learning but is never fed back into training']},
        'models': {}, 'paired_size_differences': {}}
    for h in protocol['sizes']:
        t, e = train[h], ev[h]
        delta = e[-1] - e[0]
        bins = [(1, 5), (6, 10), (11, 20), (21, 40)]
        loss = [evaluations[h][n]['prediction_loss'] for n in milestones]
        result['models'][str(h)] = {
            'parameters_per_agent': protocol['models'][str(h)]['parameters_per_agent'],
            'training_mean_capped': float(t.mean()), 'training_median_capped': float(np.median(t)),
            'training_chain_means': t.mean(0).tolist(),
            'training_quantiles_10_25_50_75_90': np.quantile(t, [.1, .25, .5, .75, .9]).tolist(),
            'training_sessions_at_least_5000': int((t >= 5000).sum()),
            'training_bins': {f'{a}-{b}': float(t[a-1:b].mean()) for a, b in bins},
            'training_trend_all': trend(sessions, t), 'training_trend_after5': trend(sessions[5:], t[5:]),
            'evaluation_trend': trend(milestones, e), 'evaluation_values': e.tolist(),
            'evaluation_means': e.mean(1).tolist(),
            'evaluation_food_means': [float(np.mean(evaluations[h][n]['food'])) for n in milestones],
            'evaluation_prediction_losses': loss,
            'prediction_loss_change_percent_5_to_40': 100 * (loss[-1] / loss[0] - 1),
            'evaluation_delta_5_to_40_by_chain': delta.tolist(),
            'evaluation_delta_5_to_40_mean': float(delta.mean()),
            'evaluation_delta_5_to_40_percent_of_mean': 100 * (e[-1].mean() / e[0].mean() - 1),
            'stability': manifests[h]['stability']}
    for role, values in [('training', train), ('evaluation', ev)]:
        x = np.concatenate([np.full(values[h].size, h) for h in protocol['sizes']])
        y = np.concatenate([values[h].ravel() for h in protocol['sizes']])
        result[f'pooled_hidden_{role}_descriptive'] = correlations(x, y)
    for a, b in [(128, 64), (192, 64), (192, 128)]:
        d, e = train[a] - train[b], ev[a] - ev[b]
        result['paired_size_differences'][f'{a}-{b}'] = {
            'training_mean': float(d.mean()), 'training_mean_by_chain': d.mean(0).tolist(),
            'training_wins_losses_ties': [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
            'evaluation_mean': float(e.mean()), 'evaluation_mean_by_chain': e.mean(0).tolist(),
            'evaluation_means_by_milestone': e.mean(1).tolist(), 'evaluation_difference_at40': e[-1].tolist()}
    (out / 'relations.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (out / 'world-sessions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    colors = {64: '#30789e', 128: '#c17a20', 192: '#48815c'}
    for h in protocol['sizes']:
        y = train[h].mean(1)
        axes[0, 0].plot(sessions, y, color=colors[h], alpha=.22, linewidth=1)
        axes[0, 0].plot(np.array(sessions)[4:], np.convolve(y, np.ones(5) / 5, mode='valid'),
            color=colors[h], linewidth=2, label=f'hidden {h}')
        axes[0, 1].plot(milestones, ev[h].mean(1), '-o', color=colors[h], linewidth=2, label=f'hidden {h}')
        for wi in range(3):
            axes[0, 1].plot(milestones, ev[h][:, wi], '--', color=colors[h], alpha=.25, linewidth=1)
    axes[0, 0].set(title='訓練：序盤の変化後、存続はおおむね横ばい',
        xlabel='セッション数', ylabel='平均存続tick（3世界）')
    axes[0, 0].text(.02, .96, '薄線：各セッション平均　太線：直近5セッション平均',
        transform=axes[0, 0].transAxes, va='top', fontsize=9)
    axes[0, 0].legend()
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].set_xticks(milestones, labels=milestones)
    axes[0, 1].set(title='共通マップ評価：128の改善が目立つが単調ではない',
        xlabel='学習済みセッション数', ylabel='存続tick')
    axes[0, 1].text(.02, .96, '太線：3系列平均　破線：各系列',
        transform=axes[0, 1].transAxes, va='top', fontsize=9)
    positions = np.arange(3)
    boxes = axes[1, 0].boxplot([train[h].ravel() for h in protocol['sizes']],
        positions=positions, widths=.5, whis=(10, 90), patch_artist=True, showfliers=True)
    for patch, h in zip(boxes['boxes'], protocol['sizes']):
        patch.set_facecolor(colors[h])
        patch.set_alpha(.4)
    for position, h in enumerate(protocol['sizes']):
        axes[1, 0].scatter(position, train[h].mean(), color=colors[h], marker='D', zorder=4)
        axes[1, 1].scatter(position + np.array([-.12, 0, .12]), ev[h][-1] - ev[h][0],
            color=colors[h], s=42)
        axes[1, 1].scatter(position, (ev[h][-1] - ev[h][0]).mean(), color=colors[h], marker='D', s=65)
    for ax in axes[1]:
        ax.set_xticks(positions, labels=[str(h) for h in protocol['sizes']])
        ax.set_xlabel('hidden')
    axes[1, 0].set(title='訓練全40セッション：128は長い存続が増える', ylabel='存続tick（上限10,000）')
    axes[1, 0].text(.02, .85, '箱：四分位範囲　ひげ：10–90%　菱形：平均\n各規模120件は3系列内の反復観測',
        transform=axes[1, 0].transAxes, va='top', fontsize=9)
    axes[1, 0].scatter(1, 10000, marker='^', facecolors='white', edgecolors=colors[128], s=70, zorder=5)
    axes[1, 0].annotate('未全滅：≥10,000（1件）', xy=(1, 10000), xytext=(1.2, 9750),
        fontsize=8, color=colors[128], arrowprops={'arrowstyle': '-', 'color': colors[128]})
    axes[1, 1].axhline(0, color='#777777', linewidth=1)
    axes[1, 1].set(title='共通マップ：5 → 40セッションの変化', ylabel='存続tickの増減')
    axes[1, 1].text(.02, .96, '丸：同じ系列・マップの変化　菱形：3系列平均',
        transform=axes[1, 1].transAxes, va='top', fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle('モデル規模とセッション数の関係：各規模3独立系列、評価は同じ3マップ', fontsize=13)
    fig.savefig(out / 'relations.png', dpi=160)
    plt.close(fig)
    print(json.dumps({h: {k: v[k] for k in ('training_mean_capped', 'evaluation_means',
        'evaluation_delta_5_to_40_mean')} for h, v in result['models'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
