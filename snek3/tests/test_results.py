"""Result-file paths, shard files, and the merge at the end of a wave.

**The naming is tested because it drifted in snek2.** A shard writing `_s1of4` beside a controller
looking for `-s1of4` is a wave that reports zero progress and finishes with an empty file, and the
two spellings lived in different modules. One module owns them here, so one set of tests covers
every reader.

`RUNS_DIR` is monkeypatched throughout, so nothing here writes into the real `runs/`.
"""

import json
import os

import pytest

from env import constants
from tools import results


@pytest.fixture
def runs(tmp_path, monkeypatch):
    directory = tmp_path / 'runs'
    directory.mkdir()
    monkeypatch.setattr(constants, 'RUNS_DIR', str(directory))
    return str(directory)


# ------------------------------------------------------------------------ naming

def test_a_policy_path_and_a_bare_name_key_the_same_files(runs):
    # Both spellings are used constantly: a bare arm name from a batch spec, and a path to a
    # directory outside savedPolicies/ such as a hallOfFame entry. They must not produce two files.
    assert results.run_name('b1a-thing') == 'b1a-thing'
    assert results.run_name('savedPolicies/b1a-thing/') == 'b1a-thing'
    assert results.stage_a_path('b1a-thing') == results.stage_a_path('savedPolicies/b1a-thing')


def test_shard_filenames_are_one_based(runs):
    # `-s1of8` reads better in a log than `-s0of8`, and every reader has to agree which it is.
    assert results.stage_b_path('a', shard=0, shards=8).endswith('-s1of8.json')
    assert results.stage_b_path('a', shard=7, shards=8).endswith('-s8of8.json')


def test_a_label_separates_two_passes(runs):
    assert results.stage_b_path('a') != results.stage_b_path('a', 'ab3222')
    assert 'ab3222' in results.stage_b_path('a', 'ab3222')


def test_a_labelled_shard_is_not_found_by_the_unlabelled_pass(runs):
    """The reason `shard_paths` is a regex over the basename and not just a glob.

    `a_checkpoint_evals_ab3222-s1of8.json` matches the glob for the *unlabelled* pass, because
    `_ab3222` looks like part of a name — so an A/B's shards would be merged into the main file.
    """
    for shard in range(2):
        results.write(results.stage_b_path('a', 'ab3222', shard=shard, shards=2), {'rows': []})
    assert results.shard_paths('a') == []
    assert len(results.shard_paths('a', 'ab3222')) == 2


def test_shard_paths_come_back_in_shard_order(runs):
    # Lexical order puts s10 before s2, and the merge relies on nothing but this ordering to be
    # deterministic.
    for shard in range(12):
        results.write(results.stage_b_path('a', shard=shard, shards=12), {'rows': []})
    ordered = [os.path.basename(path) for path in results.shard_paths('a')]
    assert ordered[1] == 'a_checkpoint_evals-s2of12.json'
    assert ordered[-1] == 'a_checkpoint_evals-s12of12.json'


# ------------------------------------------------------------------------ reading and writing

def test_an_absent_file_reads_as_none_and_a_corrupt_one_raises(runs):
    """Absent and corrupt are different things and must not be conflated.

    A wave that has not started is normal; a truncated file is a bug, and treating it as "no
    results" would let a wave resume from zero and silently re-measure everything.
    """
    path = os.path.join(runs, 'nope.json')
    assert results.read(path) is None
    with open(path, 'w') as handle:
        handle.write('{"rows": [')
    with pytest.raises(ValueError):
        results.read(path)


def test_a_write_is_atomic(runs):
    # Progress is read off these files while shards write them, so a reader must never see a
    # half-serialised one. Checked by the absence of the staging file afterwards.
    path = os.path.join(runs, 'x.json')
    results.write(path, {'rows': [{'step': 1}]})
    assert sorted(os.listdir(runs)) == ['x.json']
    assert results.read(path)['rows'] == [{'step': 1}]


def test_rows_of_tolerates_an_empty_or_missing_payload():
    assert results.rows_of(None) == []
    assert results.rows_of({}) == []
    assert results.rows_of({'rows': [{'step': 1}]}) == [{'step': 1}]


# ------------------------------------------------------------------------ merging

def a_row(step, episodes=500, perfect=490):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1)}


def test_a_merge_orders_by_step_across_shards(runs):
    results.write(results.stage_b_path('a', shard=0, shards=2),
                  {'episodes': 500, 'rows': [a_row(3000), a_row(1000)]})
    results.write(results.stage_b_path('a', shard=1, shards=2),
                  {'episodes': 500, 'rows': [a_row(2000)]})
    path, rows = results.merge('a')
    assert [row['step'] for row in rows] == [1000, 2000, 3000]
    assert results.read(path)['shards'] == 2
    assert results.read(path)['episodes'] == 500


def test_a_duplicated_step_keeps_the_longer_sample(runs):
    """Two shards should never measure the same step, and in snek2 a re-dispatched one did.

    Keeping the longer sample is the only choice that cannot lose episodes; keeping either
    arbitrarily could replace a full row with a partial one.
    """
    results.write(results.stage_b_path('a', shard=0, shards=2),
                  {'rows': [a_row(1000, episodes=500, perfect=495)]})
    results.write(results.stage_b_path('a', shard=1, shards=2),
                  {'rows': [a_row(1000, episodes=100, perfect=97)]})
    _, rows = results.merge('a')
    assert len(rows) == 1 and rows[0]['episodes'] == 500


def test_merging_nothing_writes_an_empty_pass_rather_than_failing(runs):
    # A wave whose shards all had nothing to do is not an error, and the file it leaves has to be
    # readable so a rerun can resume from it.
    path, rows = results.merge('a')
    assert rows == []
    assert results.read(path)['rows'] == []


def test_the_shard_files_survive_a_merge_unless_asked(runs):
    # They are the resumable state. snek2 lost 192 rows once; deleting them by default would be the
    # same mistake with an extra step.
    results.write(results.stage_b_path('a', shard=0, shards=1), {'rows': [a_row(1000)]})
    results.merge('a')
    assert results.shard_paths('a')
    results.merge('a', delete_shards=True)
    assert results.shard_paths('a') == []


def test_a_merge_carries_the_shards_header_fields(runs):
    results.write(results.stage_b_path('a', shard=0, shards=1),
                  {'episodes': 500, 'seed': 7, 'config': 'grid 10x10', 'rows': [a_row(1000)]})
    path, _ = results.merge('a')
    merged = results.read(path)
    assert merged['seed'] == 7 and merged['config'] == 'grid 10x10'
    assert merged['policy'] == 'a' and merged['label'] is None


# ------------------------------------------------------------------- the early stop

def test_a_merge_keeps_the_stop_target_and_refuses_two_targets_once_a_row_was_stopped(runs):
    """A stopped row means something only against the target it was stopped under, and a merged file
    has one header -- so shards stopped under different targets do not merge. Full rows are full under
    any target, so files with none stopped merge whatever their headers say (`plans/archive/early-stop.md`)."""
    stopped = dict(a_row(2000), episodes=120, abandoned=True, episodes_planned=500)
    results.write(results.stage_b_path('a', shard=0, shards=2),
                  {'episodes': 500, 'stop_target': 99.6, 'rows': [a_row(1000)]})
    results.write(results.stage_b_path('a', shard=1, shards=2),
                  {'episodes': 500, 'stop_target': 99.6, 'rows': [stopped]})
    path, rows = results.merge('a')
    assert results.read(path)['stop_target'] == 99.6 and [r['step'] for r in rows] == [1000, 2000]
    results.write(results.stage_b_path('a', shard=0, shards=2),
                  {'episodes': 500, 'stop_target': None, 'rows': [a_row(1000)]})
    with pytest.raises(ValueError, match='different stop targets'):
        results.merge('a')
    results.write(results.stage_b_path('a', shard=1, shards=2),
                  {'episodes': 500, 'stop_target': 99.6, 'rows': [a_row(2000)]})      # nothing stopped
    _, rows = results.merge('a')
    assert [r['step'] for r in rows] == [1000, 2000]


# ------------------------------------------------------------------ columns (2026-09-11)

def test_columns_round_trip_rows_with_a_nested_block_and_uneven_keys():
    rows = [{'step': 1, 'avg_score': 1.5, 'ppo': {'approx_kl': 0.01, 'entropy': 1.0}},
            {'step': 2, 'avg_score': 2.5, 'epsilon': 0.3},
            {'step': 3, 'avg_score': 3.5, 'ppo': {'approx_kl': 0.02}}]
    columns = results.to_columns(rows)
    assert list(columns) == ['step', 'avg_score', 'ppo.approx_kl', 'ppo.entropy', 'epsilon']
    assert columns['epsilon'] == [None, 0.3, None]
    assert results.from_columns(columns) == rows


def test_read_hands_back_rows_whichever_shape_the_file_stored(runs):
    rows = [{'step': 1, 'perfect_percent': 50.0}, {'step': 2, 'perfect_percent': 60.0}]
    old = os.path.join(runs, 'old_evals.json')
    new = os.path.join(runs, 'new_evals.json')
    with open(old, 'w') as handle:
        json.dump({'summary': {'step': 2}, 'evals': rows, 'resumes': []}, handle)
    results.write(new, results.stage_a_payload({'step': 2}, rows, []))
    assert results.read(old)['evals'] == rows
    assert results.read(new)['evals'] == rows
    assert results.read(new)['format'] == results.COLUMNS_FORMAT
    assert os.path.getsize(new) < os.path.getsize(old)


def test_a_merge_stamps_the_pass_from_the_first_shard_start_to_the_last_shard_write(runs):
    for index, (started, finished) in enumerate([('2026-09-11T10:00:00', '2026-09-11T11:00:00'),
                                                 ('2026-09-11T10:05:00', '2026-09-11T12:30:00')]):
        results.write(results.stage_b_path('p', shard=index, shards=2),
                      {'policy': 'p', 'started': started, 'finished': finished,
                       'rows': [{'step': index, 'episodes': 5, 'perfect_games': 5}]})
    _, _ = results.merge('p')
    merged = results.read(results.stage_b_path('p'))
    assert (merged['started'], merged['finished']) == ('2026-09-11T10:00:00', '2026-09-11T12:30:00')
    assert merged['wall_seconds'] == 9000


def test_seconds_between_tolerates_a_missing_stamp():
    assert results.seconds_between(None, '2026-09-11T10:00:00') is None
    assert results.seconds_between('2026-09-11T10:00:00', '2026-09-11T10:00:30') == 30
