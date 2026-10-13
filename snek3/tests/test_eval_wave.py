"""`tools/eval_wave.ArmWave`: an arm already merged is not measured again."""

import json
import os

from tools import eval_wave
from tools import results


def _arm(monkeypatch, steps):
    monkeypatch.setattr(eval_wave.restore, 'policy_dir', lambda policy: '/nowhere/' + policy)
    monkeypatch.setattr(eval_wave.selectors, 'resolve',
                        lambda directory, selector, policy=None: (list(steps), 'fixture'))


def _merged(policy, label, steps):
    path = results.stage_b_path(policy, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump({'policy': policy, 'rows': [{'step': step, 'perfect_percent': 99.0} for step in steps]}, handle)
    return path


def test_an_arm_whose_merged_file_covers_its_steps_runs_no_shards_and_keeps_the_file(monkeypatch, capsys):
    """2026-09-05: a close-out relaunched after a kill measured two already-merged arms again from
    scratch -- the merge had deleted their shard files, so shard-level resume had nothing to resume."""
    _arm(monkeypatch, [100, 200, 300])
    path = _merged('b1a-x', None, [100, 200, 300])
    before = open(path).read()
    wave = eval_wave.ArmWave('b1a-x', shards=4)
    assert wave.already_merged and wave.shards == 0 and wave.pending == []
    wave.announce(requested=4)
    assert 'kept, not measured again' in capsys.readouterr().out
    assert wave.finish() == 0
    assert open(path).read() == before, 'not rewritten, not emptied'
    assert wave.finished() and wave.counts() == []


def test_a_merged_file_missing_a_selected_step_does_not_count(monkeypatch):
    _arm(monkeypatch, [100, 200, 300])
    _merged('b1a-x', None, [100, 200])                  # a new checkpoint qualified since the merge
    wave = eval_wave.ArmWave('b1a-x', shards=4)
    # Until 2026-10-12 every step was measured again here (3 shards); now the two merged rows are kept
    # and one shard measures the new step -- the merge folds the file back in.
    assert not wave.already_merged and wave.shards == 1 and wave.todo == [300]


def test_no_resume_measures_afresh_and_a_torn_file_is_not_a_reason_to_skip(monkeypatch):
    _arm(monkeypatch, [100, 200])
    path = _merged('b1a-x', 'hof5000', [100, 200])
    assert eval_wave.ArmWave('b1a-x', label='hof5000', shards=2, resume=False).shards == 2
    with open(path, 'w') as handle:
        handle.write('{"rows": [')
    assert eval_wave.merged_steps('b1a-x', 'hof5000') == set()
    assert eval_wave.ArmWave('b1a-x', label='hof5000', shards=2).shards == 2


def test_an_arm_with_no_candidates_is_unchanged_by_the_rule(monkeypatch):
    _arm(monkeypatch, [])
    wave = eval_wave.ArmWave('b1a-x', shards=4)
    assert not wave.already_merged and wave.shards == 0


def test_the_shard_command_carries_the_stop_target_only_when_there_is_one():
    from tools import eval_wave
    plain = eval_wave.shard_command('b1a', 'screen', 500, None, 0, 4, None, 0, True)
    assert '--stop' not in plain
    stopped = eval_wave.shard_command('b1a', 'above:99.2', 5000, 'hof5000', 0, 4, None, 0, True, 99.6)
    assert stopped[stopped.index('--stop') + 1] == '99.6'


def test_a_merged_file_covering_part_of_the_selection_runs_shards_over_the_rest_only(monkeypatch, capsys):
    """2026-10-12: b52 i-l extended 1.8M -> 2.4M after their passes had merged. The merged rows are kept,
    the shards are sized to the new checkpoints, and the row-count guard at the merge counts those."""
    _arm(monkeypatch, [100, 200, 300, 400, 500])
    _merged('b1a-x', None, [100, 200, 300])
    wave = eval_wave.ArmWave('b1a-x', shards=4)
    assert not wave.already_merged and wave.todo == [400, 500] and wave.shards == 2
    wave.announce(requested=4)
    out = capsys.readouterr().out
    assert '3 already merged, kept' in out and '1000 episodes total' in out
    fresh = eval_wave.ArmWave('b1a-x', shards=4, resume=False)
    assert fresh.todo == [100, 200, 300, 400, 500] and fresh.shards == 4, '--no-resume measures afresh'
