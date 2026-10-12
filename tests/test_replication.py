import dataclasses

import pytest

from ailife.config import Config
from ailife.replication import build_plan


def test_replication_counts_pairing_and_distinct_cohorts():
    p = build_plan(Config())
    assert len(p['cells']) == 6 and len(p['jobs']) == 15
    assert sum(c['trials'] for c in p['cells']) == 120
    assert sum(j['worlds'] for j in p['jobs']) == 100
    assert sum(j['worlds'] * j['sessions'] for j in p['jobs']) == 800
    assert sum(j['worlds'] * len(j['milestones']) for j in p['jobs']) == 120
    for block in p['blocks']:
        jobs = [j for j in p['jobs'] if j['block'] == block['id']]
        assert len(jobs) == 5
        assert len({j['rocr'] for j in jobs}) == 1
        specs = [p['protocols'][j['study']] for j in jobs]
        assert len({s['evaluation_seed'] for s in specs}) == 1
        assert len({s['config']['seed'] for s in specs if s['sessions'] == 5}) == 1
        assert specs[0]['config']['seed'] != specs[-1]['config']['seed']
    for e in (32, 64):
        for h in (64, 192):
            assert sum(j['worlds'] for j in p['jobs'] if j['cohort'] == 'A' and j['hidden'] == h and j['encoder_width'] == e) == 20
    trials = [block['offset'] + wi for block in p['blocks'] for wi in range(block['worlds'])]
    assert sorted(trials) == list(range(20))


@pytest.mark.parametrize('width', [0, -1, 32.5, True])
def test_encoder_width_rejects_invalid_values(width):
    with pytest.raises(ValueError, match='encoder_width'):
        Config(encoder_width=width)


def test_legacy_configuration_defaults_to_original_encoder():
    saved = Config().as_dict()
    saved.pop('encoder_width')
    assert Config.from_saved(saved).encoder_width == 32
    assert dataclasses.replace(Config(), encoder_width=64).encoder_width == 64
