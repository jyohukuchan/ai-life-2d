"""Prepare the requested 80 architecture and 40 session-count outcomes."""
import argparse
import dataclasses
import json

from ailife.config import Config
from ailife.replication import prepare_replication


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs')
    p.add_argument('--study', default='rep20-20261012')
    p.add_argument('--base', default='configs/experiments/2026-10-11-compass/compass-absolute.yaml')
    args = p.parse_args()
    cfg = dataclasses.replace(Config.load(args.base), precision='fp32', execution='compiled',
        elite_count=4, fast_forward_extinct=True, log_reward_components=False,
        checkpoint_ticks=2000)
    plan = prepare_replication(args.root, cfg, args.study)
    print(json.dumps({'study': plan['study'], 'learning_chains': plan['independent_learning_chains'],
        'evaluation_outcomes': plan['primary_evaluation_outcomes'],
        'parameters': plan['parameters_per_individual'],
        'checkpoint_gib': plan['estimated_initial_and_final_checkpoint_bytes'] / 2**30}), flush=True)


if __name__ == '__main__':
    main()
