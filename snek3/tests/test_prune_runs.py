"""The disk pruner: what it deletes, and the three cases where it must refuse.

Every fixture here is about *not* losing a measurement. The tool exists because an arm keeps a
checkpoint per rollout and a stage-B pass leaves its shard files behind, so the interesting
assertions are the guards rather than the deletions.
"""

import json
import os

import pytest

from env import constants
from tools import checkpoints, eval_plan, prune_runs, results


def row(step, perfect, episodes=500, arrays=True):
    """A stored row, with or without the two arrays that were dropped on 2026-09-01."""
    scores = [95] * perfect + [40] * (episodes - perfect)
    built = {'step': int(step), 'episodes': episodes, 'perfect_games': perfect,
             'perfect_percent': round(100.0 * perfect / episodes, 1),
             'perfect_ci95': [90.0, 100.0], 'episode_scores': scores}
    if arrays:
        built['episode_perfect'] = [int(s == 95) for s in scores]
        built['episode_rewards'] = [float(s) for s in scores]
    return built


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    policies = tmp_path / 'savedPolicies'
    runs.mkdir()
    policies.mkdir()
    monkeypatch.setattr(constants, 'RUNS_DIR', str(runs))
    monkeypatch.setattr(constants, 'POLICY_DIR', str(policies))
    monkeypatch.setattr(prune_runs.live_runs, 'live', lambda *a, **k: [])
    return str(runs)


def write_pass(policy, label, rows, shard=None, shards=None):
    path = results.stage_b_path(policy, label, shard=shard, shards=shards)
    results.write(path, {'policy': policy, 'rows': rows})
    return path


def make_arm(policy, steps, ckpt_bytes=1000):
    directory = os.path.join(constants.POLICY_DIR, policy)
    os.makedirs(directory, exist_ok=True)
    for step in steps:
        with open(checkpoints.path(directory, step), 'wb') as handle:
            handle.write(b'x' * ckpt_bytes)
    return directory


# --- shards ---------------------------------------------------------------------------------------

def test_shards_go_once_the_merge_covers_every_row(runs_dir):
    write_pass('arm', None, [row(1000, 490)], shard=0, shards=2)
    write_pass('arm', None, [row(2000, 491)], shard=1, shards=2)
    write_pass('arm', None, [row(1000, 490), row(2000, 491)])
    freed = prune_runs.prune_shards(apply=True)
    assert freed > 0
    assert results.shard_paths('arm', None) == [], 'the shard files are gone'
    assert len(results.rows_of(results.read(results.stage_b_path('arm', None)))) == 2


def test_a_dry_run_deletes_nothing_and_still_reports_the_bytes(runs_dir):
    write_pass('arm', None, [row(1000, 490)], shard=0, shards=2)
    write_pass('arm', None, [row(2000, 491)], shard=1, shards=2)
    write_pass('arm', None, [row(1000, 490), row(2000, 491)])
    freed = prune_runs.prune_shards(apply=False)
    assert freed > 0 and len(results.shard_paths('arm', None)) == 2


def test_shards_are_kept_when_there_is_no_merged_file(runs_dir):
    # The in-flight case: a wave still running has shards and no merge, and `stage_b_chart` reads
    # them. Deleting these would throw away live measurements.
    write_pass('arm', None, [row(1000, 490)], shard=0, shards=2)
    prune_runs.prune_shards(apply=True)
    assert len(results.shard_paths('arm', None)) == 1


def test_shards_are_kept_when_a_row_never_reached_the_merge(runs_dir):
    # A killed wave that was merged anyway, or a merge from a partial set.
    write_pass('arm', None, [row(1000, 490)], shard=0, shards=2)
    write_pass('arm', None, [row(2000, 491)], shard=1, shards=2)
    write_pass('arm', None, [row(1000, 490)])
    prune_runs.prune_shards(apply=True)
    assert len(results.shard_paths('arm', None)) == 2, 'step 2000 is only in the shard'


def test_shards_are_kept_when_the_merge_holds_a_shorter_sample(runs_dir):
    write_pass('arm', None, [row(1000, 490, episodes=500)], shard=0, shards=1)
    write_pass('arm', None, [row(1000, 98, episodes=100)])
    ok, why = prune_runs.covered(results.stage_b_path('arm', None),
                                 results.shard_paths('arm', None))
    assert not ok and 'shorter' in why


# --- arrays ---------------------------------------------------------------------------------------

def test_the_two_dead_arrays_go_and_the_scores_stay(runs_dir):
    path = write_pass('arm', None, [row(1000, 490), row(2000, 491)])
    before = os.path.getsize(path)
    prune_runs.prune_arrays(apply=True)
    rows = results.rows_of(results.read(path))
    assert os.path.getsize(path) < before
    for stored in rows:
        assert 'episode_perfect' not in stored and 'episode_rewards' not in stored
        assert len(stored['episode_scores']) == 500


def test_the_perfect_count_survives_the_rewrite_exactly(runs_dir):
    # The point of the whole change: the flags are recoverable from the scores. If this ever fails,
    # the arrays are not redundant and must not be dropped.
    path = write_pass('arm', None, [row(1000, 490), row(2000, 3)])
    before = {r['step']: [bool(f) for f in r['episode_perfect']]
              for r in results.rows_of(results.read(path))}
    prune_runs.prune_arrays(apply=True)
    for stored in results.rows_of(results.read(path)):
        assert eval_plan.perfect_flags(stored) == before[stored['step']]
        assert sum(eval_plan.perfect_flags(stored)) == stored['perfect_games']


def test_rewriting_twice_is_a_no_op(runs_dir):
    path = write_pass('arm', None, [row(1000, 490)])
    prune_runs.prune_arrays(apply=True)
    payload = results.read(path)
    assert prune_runs.prune_arrays(apply=True) == 0
    assert results.read(path) == payload


# --- checkpoints ----------------------------------------------------------------------------------

def test_checkpoints_below_the_threshold_go_and_measured_ones_above_it_stay(runs_dir):
    make_arm('arm', [1000, 2000, 3000, 4000])
    write_pass('arm', None, [row(1000, 480), row(2000, 493), row(3000, 495)])   # 96.0, 98.6, 99.0
    keep, drop, _ = prune_runs.checkpoint_plan('arm', 97.5)
    assert keep == {2000, 3000}
    assert drop == {1000, 4000}, 'the unmeasured 4000 goes too: stage A screened it out'
    prune_runs.prune_checkpoints(['arm'], 97.5, apply=True)
    directory = os.path.join(constants.POLICY_DIR, 'arm')
    assert sorted(checkpoints.step_of(n) for n in os.listdir(directory)) == [2000, 3000]


def test_the_best_row_is_kept_however_high_the_threshold(runs_dir):
    # Otherwise a threshold nobody clears deletes the arm's own champion.
    make_arm('arm', [1000, 2000])
    write_pass('arm', None, [row(1000, 400), row(2000, 450)])
    keep, drop, _ = prune_runs.checkpoint_plan('arm', 99.9)
    assert keep == {2000} and drop == {1000}


def test_it_refuses_an_arm_with_no_stage_b_pass(runs_dir):
    make_arm('arm', [1000, 2000])
    keep, drop, why = prune_runs.checkpoint_plan('arm', 97.5)
    assert (keep, drop) == (set(), set()) and 'no stage-B pass' in why


def test_it_refuses_a_running_arm(runs_dir, monkeypatch):
    make_arm('arm', [1000, 2000])
    write_pass('arm', None, [row(1000, 480), row(2000, 495)])
    monkeypatch.setattr(prune_runs.live_runs, 'live', lambda *a, **k: [('arm', 4242)])
    keep, drop, why = prune_runs.checkpoint_plan('arm', 97.5)
    assert (keep, drop) == (set(), set()) and 'running' in why


def test_a_dry_run_leaves_every_checkpoint_on_disk(runs_dir):
    make_arm('arm', [1000, 2000])
    write_pass('arm', None, [row(1000, 400), row(2000, 495)])
    freed = prune_runs.prune_checkpoints(['arm'], 97.5, apply=False)
    directory = os.path.join(constants.POLICY_DIR, 'arm')
    assert freed > 0 and len(os.listdir(directory)) == 2


def test_a_labelled_pass_can_drive_the_pruning(runs_dir):
    # `hof5000` rows are the 5,000-episode re-measure; pruning against them is the tighter cut.
    make_arm('arm', [1000, 2000, 3000])
    write_pass('arm', 'hof5000', [row(2000, 4930, episodes=5000), row(3000, 4800, episodes=5000)])
    keep, drop, _ = prune_runs.checkpoint_plan('arm', 97.5, label='hof5000')
    assert keep == {2000} and drop == {1000, 3000}


def test_git_tracked_files_are_skipped_by_default(runs_dir, monkeypatch):
    """Rewriting a tracked file trades working-tree bytes for permanent `.git` growth."""
    path = write_pass('arm', None, [row(1000, 490)])
    monkeypatch.setattr(prune_runs, 'tracked_paths', lambda: {os.path.realpath(path)})
    assert prune_runs.prune_arrays(apply=True) == 0
    assert 'episode_rewards' in results.rows_of(results.read(path))[0]

    assert prune_runs.prune_arrays(apply=True, include_tracked=True) > 0
    assert 'episode_rewards' not in results.rows_of(results.read(path))[0]


def test_tracked_paths_returns_paths_that_exist(runs_dir):
    # `git ls-files` without `--full-name` returns cwd-relative names, and joining those to the repo
    # root produces paths that match nothing — so the guard silently stops guarding.
    for path in prune_runs.tracked_paths():
        assert os.path.isabs(path)


def test_apply_is_accepted_after_the_subcommand(runs_dir, capsys):
    """`--apply` on the top-level parser made every documented invocation an argparse error.

    argparse accepts an option only before the subcommand it was declared on, so this is the one
    fixture that catches the difference between the CLI and the docstring that describes it.
    """
    write_pass('arm', None, [row(1000, 490)], shard=0, shards=1)
    write_pass('arm', None, [row(1000, 490)])
    assert prune_runs.main(['shards', '--apply']) == 0
    assert results.shard_paths('arm', None) == []
    assert 'DRY RUN' not in capsys.readouterr().out


def test_without_apply_every_subcommand_is_a_dry_run(runs_dir, capsys):
    make_arm('arm', [1000, 2000])
    write_pass('arm', None, [row(1000, 400), row(2000, 495)])
    for argv in (['shards'], ['arrays'], ['checkpoints', 'arm']):
        assert prune_runs.main(argv) == 0
        assert 'DRY RUN' in capsys.readouterr().out
    assert len(os.listdir(os.path.join(constants.POLICY_DIR, 'arm'))) == 2


def test_arrays_leaves_the_shards_of_an_unmerged_pass_alone(runs_dir):
    """The race: a shard process is still writing this file, and `results.write` is atomic, so one
    of the two writes is silently discarded. The desktop runs evals unattended."""
    live = write_pass('arm', None, [row(1000, 490)], shard=0, shards=2)
    done = write_pass('other', None, [row(1000, 490)], shard=0, shards=2)
    write_pass('other', None, [row(1000, 490)])          # `other` merged; `arm` did not
    prune_runs.prune_arrays(apply=True)
    assert 'episode_rewards' in results.rows_of(results.read(live))[0], 'in flight, untouched'
    assert 'episode_rewards' not in results.rows_of(results.read(done))[0], 'merged, pruned'
