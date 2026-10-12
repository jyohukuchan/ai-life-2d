"""Reconstruct exact population counts from observational birth/death events.

Run with the host Python (NumPy and matplotlib), independently of training.
Pass consecutive output directories from the same checkpoint chain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def records(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith('\n')]


def analyze(paths, output):
    paths = [Path(p).resolve() for p in paths]
    configs = [json.loads((p / 'config.json').read_text()) for p in paths]
    metadata = [json.loads((p / 'metadata.json').read_text()) for p in paths]
    cfg = configs[0]
    if metadata[0]['start_tick'] != 0:
        raise ValueError('Include the original trial so founder counts are known')
    for i in range(1, len(paths)):
        if Path(metadata[i]['resumed_from']).resolve().parent != paths[i - 1]:
            raise ValueError('Trials must be consecutive continuations, not independent seeds')
        defaults={'compass':'none','obstacle_density':0.,'food_x_min':0.,'food_x_max':1.,'spawn_x_min':0.,'spawn_x_max':1.,'remote_births':False,'food_budget':'local','food_reference_size':0}
        def value(config,key):
            if key in {'height','width'}:return config.get(key) or config['size']
            return config.get(key,defaults.get(key))
        for key in ['worlds', 'initial', 'capacity', 'size', 'height', 'width', 'seed', 'mode', 'hidden', 'obstacle_density', 'compass',
                    'food_x_min','food_x_max','spawn_x_min','spawn_x_max','remote_births','food_budget','food_reference_size',
                    'regrowth','initial_food','metabolism','move_cost','lifespan','founder_energy','child_energy','birth_overhead','energy_max','food_energy','fertility','reproduction','mutation','radius','symbols','interoception','sound_radius','maturity_ratio']:
            if value(configs[i],key) != value(cfg,key):
                raise ValueError(f'Changed experimental condition: {key}')

    metrics = sorted([r for p in paths for r in records(p / 'metrics.jsonl')], key=lambda r: r['tick'])
    if not metrics:
        raise ValueError('No completed metric interval yet')
    end = metrics[-1]['tick']
    delta = np.zeros((end + 1, cfg['worlds']), dtype=np.int64)
    births = np.zeros((cfg['worlds'], 2), dtype=np.int64)
    deaths = np.zeros((cfg['worlds'], 2), dtype=np.int64)
    generations = np.zeros(cfg['worlds'], dtype=np.int64)
    for path in paths:
        for event in records(path / 'events.jsonl'):
            tick, world = event['tick'], event['world']
            if tick > end:
                continue
            if event['event'] == 'birth':
                delta[tick, world] += 1
                births[world, int(bool(event['parents'][1]))] += 1
                generations[world] = max(generations[world], event['generation'])
            elif event['event'] == 'death':
                delta[tick, world] -= 1
                deaths[world, 0 if event['cause'] == 'energy' else 1] += 1
            else:
                raise ValueError(f'Unknown event: {event}')
    population = cfg['initial'] + delta.cumsum(axis=0)
    if (population < 0).any() or (population > cfg['capacity']).any():
        raise ValueError('Population accounting exceeds physical bounds')
    for row in metrics:
        if population[row['tick']].tolist() != row['alive']:
            raise ValueError(f"Events and population metrics disagree at tick {row['tick']}; a running log may still be flushing")
    last = metrics[-1]
    if births.tolist() != last['births_clone_sexual'] or deaths.tolist() != last['deaths_energy_age']:
        raise ValueError('Cumulative birth/death records disagree with world counters')

    window = min(10000, end + 1)
    worlds = []
    for i in range(cfg['worlds']):
        trace, tail = population[:, i], population[-window:, i]
        zero = np.flatnonzero(trace == 0)
        worlds.append({
            'world': i, 'seed': cfg['seed'] + 1009 * i,
            'initial': cfg['initial'], 'final': int(trace[-1]),
            'maximum': int(trace.max()), 'maximum_tick': int(trace.argmax()),
            'minimum': int(trace.min()), 'extinction_tick': int(zero[0]) if len(zero) else None,
            'last_window_mean': float(tail.mean()), 'last_window_minimum': int(tail.min()),
            'last_window_maximum': int(tail.max()),
            'births_clone_sexual': births[i].tolist(), 'deaths_energy_age': deaths[i].tolist(),
            'maximum_generation_seen': int(generations[i]),
        })
    summary_path = paths[-1] / 'summary.json'
    status = json.loads(summary_path.read_text())['status'] if summary_path.exists() else 'running'
    result = {'status': status, 'through_tick': end, 'trials': [str(p) for p in paths],
              'last_window_start': end + 1 - window, 'last_window_ticks': window, 'worlds': worlds,
              'every_metric_matches_event_accounting': True}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'population-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    np.savetxt(output / 'population.csv', np.column_stack((np.arange(end + 1), population)), fmt='%d', delimiter=',',
               header='tick,' + ','.join(f'world_{i}' for i in range(cfg['worlds'])), comments='')
    return result, population


def plot(result, population, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    end = result['through_tick']
    ticks = np.arange(end + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    colors = ['#197f69', '#416bb0', '#c06d2e']
    for i, world in enumerate(result['worlds']):
        for ax in axes:
            ax.plot(ticks, population[:, i], color=colors[i % len(colors)], lw=1.15,
                    label=f"World {i} (seed {world['seed']})")
    axes[0].set_title(f"Population through tick {end:,} — same worlds, continued from checkpoint")
    axes[1].set_title('First 10,000 ticks')
    axes[1].set_xlim(0, min(end, 10000))
    for ax in axes:
        ax.set_xlabel('Simulation tick')
        ax.set_ylabel('Living individuals')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(Path(output) / 'population.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials', nargs='+')
    parser.add_argument('--output', default='runs/population-analysis')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    result, population = analyze(args.trials, args.output)
    if not args.no_plot:
        plot(result, population, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
