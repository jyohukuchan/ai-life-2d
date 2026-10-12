"""Prepare or continue matched model-size / session-count selection chains."""
import argparse
import dataclasses
import json

from ailife.config import Config
from ailife.scaling import prepare_study, run_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='scale40-20261011')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--base', default='configs/experiments/2026-10-11-compass/compass-absolute.yaml')
    parser.add_argument('--sizes', default='64,128,192')
    parser.add_argument('--sessions', type=int, default=40)
    parser.add_argument('--milestones', default='5,10,20,40')
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.prepare:
        cfg = dataclasses.replace(Config.load(args.base), precision='fp32', execution='compiled',
            elite_count=4, fast_forward_extinct=True, log_reward_components=False,
            checkpoint_ticks=2000)
        result = prepare_study(args.root, args.study, cfg,
            tuple(map(int, args.sizes.split(','))), args.sessions,
            tuple(map(int, args.milestones.split(','))))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        if args.hidden is None:
            parser.error('--hidden is required when running a model')
        run_arm(args.root, args.study, args.hidden, args.device)


if __name__ == '__main__':
    main()
