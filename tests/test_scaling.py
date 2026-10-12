import dataclasses
import json
import os
import signal

import pytest
import torch

from ailife.config import Config
from ailife.scaling import prepare_study, read_json, run_arm


def config():
    return Config(worlds=2, initial=2, capacity=4, size=12, hidden=8, radius=1,
        threads=1, ticks=12, lifespan=8, initial_food=0, regrowth=0,
        log_ticks=4, checkpoint_ticks=4, elite_count=2,
        replay_count=1, replay_length=4, fast_forward_extinct=True)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def prepare(root, study):
    return prepare_study(root, study, config(), sizes=(8, 16), sessions=2, milestones=(1, 2))


def test_scaling_initial_worlds_match_and_history_is_fresh(tmp_path):
    protocol = prepare(tmp_path, 'matched')
    states = [torch.load(protocol['models'][str(h)]['initial_bank'], weights_only=True)
        for h in (8, 16)]
    equal(states[0]['world'], states[1]['world'])
    assert protocol['models']['16']['parameters_per_agent'] > protocol['models']['8']['parameters_per_agent']
    for state in states:
        assert state['world']['clock'] == 0
        for key in ('hidden', 'obs_n', 'reward_n', 'adam_count'):
            assert not state['learners'][key].any()
    assert prepare(tmp_path, 'matched') == protocol
    with pytest.raises(ValueError, match='protocol differs'):
        prepare_study(tmp_path, 'matched', dataclasses.replace(config(), metabolism=.03),
            sizes=(8, 16), sessions=2, milestones=(1, 2))


def test_scaling_chains_and_comparisons_use_same_session(tmp_path):
    protocol = prepare(tmp_path, 'chains')
    a = run_arm(tmp_path, 'chains', 8, 'cpu')
    b = run_arm(tmp_path, 'chains', 16, 'cpu')
    assert a['status'] == b['status'] == 'complete'
    for number in ('1', '2'):
        assert a['sessions'][number]['seed'] == b['sessions'][number]['seed']
        assert a['sessions'][number]['initial_world_digest'] == b['sessions'][number]['initial_world_digest']
        assert a['sessions'][number]['prediction_diagnostic_complete']
    prior = torch.load(tmp_path / a['sessions']['1']['run'] / 'elites.pt', weights_only=True)
    child = torch.load(a['sessions']['2']['initial_bank'], weights_only=True)
    for wi, donors in enumerate(prior['worlds']):
        for slot in range(config().initial):
            idx = wi * config().capacity + slot
            for name, p in donors[slot % len(donors)]['params'].items():
                assert torch.equal(p, child['learners']['params'][name][idx])
                assert torch.equal(p, child['learners']['genome'][name][idx])
    assert not child['learners']['adam_count'].any()
    comparison = read_json(tmp_path / 'chains-h008-h016.json')
    assert comparison['session'] == 2
    assert comparison['learned'] == a['sessions']['2']['run']
    assert comparison['control'] == b['sessions']['2']['run']
    assert (tmp_path / 'chains-s001-h008-h016.json').exists()
    assert (tmp_path / 'chains-e001-h008-h016.json').exists()
    for manifest in (a, b):
        for number in ('1', '2'):
            evaluation = manifest['evaluations'][number]
            assert evaluation['source_run'] == manifest['sessions'][number]['run']
            assert evaluation['initial_world_digest'] == protocol['evaluation']['initial_world_digest']
            assert not evaluation['feedback_to_training']
            assert not (tmp_path / evaluation['run'] / 'elites.pt').exists()
    assert read_json(tmp_path / 'chains-status.json')['status'] == 'complete'
    before = (tmp_path / a['sessions']['2']['run'] / 'elites.pt').read_bytes()
    run_arm(tmp_path, 'chains', 8, 'cpu')
    assert before == (tmp_path / a['sessions']['2']['run'] / 'elites.pt').read_bytes()


def test_scaling_interrupted_chain_resumes_exactly_without_reusing_directory(tmp_path, monkeypatch):
    import ailife.runner as runner
    prepare(tmp_path, 'full')
    prepare(tmp_path, 'resume')
    full = run_arm(tmp_path, 'full', 8, 'cpu')
    original = runner.iteration
    interrupted = False

    def stop_once(world, learners, archive=None):
        nonlocal interrupted
        result = original(world, learners, archive)
        if int(world.clock) == 4 and not interrupted:
            interrupted = True
            os.kill(os.getpid(), signal.SIGINT)
        return result

    monkeypatch.setattr(runner, 'iteration', stop_once)
    with pytest.raises(KeyboardInterrupt):
        run_arm(tmp_path, 'resume', 8, 'cpu')
    paused = read_json(tmp_path / 'resume-h008.json')
    original_directory = tmp_path / paused['sessions']['1']['run']
    assert read_json(original_directory / 'summary.json')['status'] == 'interrupted'
    resumed = run_arm(tmp_path, 'resume', 8, 'cpu')
    assert resumed['status'] == 'complete'
    assert len(resumed['sessions']['1']['parts']) == 2
    assert original_directory.exists()
    for number in ('1', '2'):
        a = torch.load(tmp_path / full['sessions'][number]['run'] / 'checkpoint.pt', weights_only=True)
        b = torch.load(tmp_path / resumed['sessions'][number]['run'] / 'checkpoint.pt', weights_only=True)
        for key in ('world', 'learners', 'elite_archive'):
            equal(a[key], b[key])
        assert full['sessions'][number]['food'] == resumed['sessions'][number]['food']
        assert full['sessions'][number]['late_food_windows'] == resumed['sessions'][number]['late_food_windows']
