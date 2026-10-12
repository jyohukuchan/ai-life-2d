"""Uniform random control and the independent neural learners share a runner."""
from __future__ import annotations

import torch

from .learner import Learners
from .random import uniform


class RandomPolicy:
    """Every action component is uniform each tick, regardless of observation.

    There are no network weights, learning updates or inherited policy biases.
    Invalid physical actions are still attempted and resolved by the world.
    """

    pred_names = ()

    def __init__(self, cfg, device='cpu'):
        self.cfg, self.device = cfg, torch.device(device)
        self.world_seed = (torch.arange(cfg.worlds, device=self.device) * 1009 + cfg.seed)[:, None]

    @torch.no_grad()
    def act(self, raw, ids, clock, alive):
        key = self.world_seed * 1000003 + ids * 65537 + clock * 104729
        choices = [(uniform(key, 1001 + j * 17) * size).long().clamp_max(size - 1)
                   for j, size in enumerate(self.cfg.action_sizes)]
        return torch.where(alive[..., None], torch.stack(choices, -1), 0)

    def learn(self, next_raw, transition):
        zero = torch.zeros_like(transition.energy)
        return {'prediction_loss': zero, 'intrinsic_reward': zero}

    def inherit(self, transition, new_ids):
        return 0

    def finite(self):
        return True

    def parameter_count(self):
        return 0

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError('The random control has no learned policy state')


def make_policy(cfg, device='cpu'):
    return RandomPolicy(cfg, device) if cfg.mode == 'RANDOM' else Learners(cfg, device)
