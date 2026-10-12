"""Check production-shape FP32 updates, births, state validity and device memory."""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from ailife.learner import Learners
from ailife.observer import atomic_json
from ailife.runner import iteration, sync
from ailife.scaling import read_json, session_config
from ailife.selection import EliteArchive
from ailife.world import World


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs')
    parser.add_argument('--study', default='scale40-20261011')
    parser.add_argument('--hidden', type=int, required=True)
    parser.add_argument('--steps', type=int, default=360)
    args = parser.parse_args()
    root = Path(args.root)
    protocol = read_json(root / f'{args.study}-protocol.json')
    cfg = session_config(protocol, args.hidden, 1)
    torch.set_num_threads(cfg.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    initial = torch.load(protocol['models'][str(args.hidden)]['initial_bank'],
        weights_only=True, map_location='cpu')
    world, learner = World(cfg, 'cuda:0'), Learners(cfg, 'cuda:0')
    world.load_state_dict(initial['world'])
    learner.load_state_dict(initial['learners'])
    archive = EliteArchive(cfg, 'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    before = learner.params['enc_w'].detach().clone()
    begin = time.monotonic()
    for tick in range(1, args.steps + 1):
        counts = learner.adam_count.clone()
        t, _ = iteration(world, learner, archive)
        expected = torch.where(t.births.flatten(), 0, counts + t.was_alive.flatten())
        if not torch.equal(learner.adam_count, expected):
            raise AssertionError('per-agent prediction update counter mismatch')
        if tick % 60 == 0:
            assert world.valid() and learner.finite()
            print(json.dumps({'hidden': args.hidden, 'smoke_tick': tick}), flush=True)
    sync('cuda:0')
    result = {'hidden': args.hidden, 'device': torch.cuda.get_device_name(),
        'arch': torch.cuda.get_device_properties(0).gcnArchName,
        'rocr_visible_devices': os.environ.get('ROCR_VISIBLE_DEVICES'),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'torch': torch.__version__, 'hip': torch.version.hip,
        'steps': args.steps, 'precision': cfg.precision, 'execution': cfg.execution,
        'encoder_width': cfg.encoder_width, 'worlds': cfg.worlds,
        'parameters_per_agent': learner.parameter_count(),
        'per_agent_update_counter_checks': args.steps,
        'world_valid': world.valid(), 'finite': learner.finite(),
        'skipped_updates': int(learner.skipped_prediction_updates.sum()),
        'encoder_weights_changed': not torch.equal(before, learner.params['enc_w']),
        'births': world.births_total.cpu().tolist(), 'alive': world.alive.sum(1).cpu().tolist(),
        'food': world.eaten_total.cpu().tolist(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds_including_compile': time.monotonic() - begin,
        'used_as_training': False}
    assert result['world_valid'] and result['finite'] and result['encoder_weights_changed']
    assert result['skipped_updates'] == 0
    atomic_json(root / f'{args.study}-h{args.hidden:03d}-smoke.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
