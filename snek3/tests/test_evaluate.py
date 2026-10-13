"""`evaluate.py one` runs in its own process, several at once during a hof-promote."""

import torch

import evaluate


def test_measure_one_pins_torch_to_a_single_thread(monkeypatch, tmp_path):
    # Five unpinned copies on the desktop each took a thread per core and thrashed for two hours
    # with no row (2026-10-12); pinned, each finished in ~44 min.
    torch.set_num_threads(max(2, torch.get_num_threads()))
    seen = {}
    arch = {'fc_layer_params': [320], 'obs_era': 'obs26-20260907'}
    monkeypatch.setattr(evaluate.restore, 'policy_dir', lambda policy: str(tmp_path))
    monkeypatch.setattr(evaluate.restore, 'restore',
                        lambda directory, step, variant=None: (object(), arch, 1000))

    def measure(policy_fn, episodes, lanes=None, seed=0):
        seen['threads'] = torch.get_num_threads()
        return None

    monkeypatch.setattr(evaluate.engine, 'measure', measure)
    monkeypatch.setattr(evaluate.eval_plan, 'build_row', lambda step, held: {'step': step, 'perfect_percent': 0.0})
    monkeypatch.setattr(evaluate.eval_plan, 'one_line', lambda row: str(row))

    evaluate.measure_one('whatever', step=1000, episodes=3)
    assert seen['threads'] == 1
