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
