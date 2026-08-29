"""Sharding a vectorised eval across processes: the parser and the partition property.

One `vec_eval.py` process saturates about one core (measured ~1.1 of 14 on this laptop, because the
observation build is single-threaded numpy and is 95% of a step), so filling the box means running
several and merging. Both things that can go quietly wrong are pinned here: a shard spec that is
misread, and a stride that drops or duplicates a checkpoint. Either produces a result file that
*looks* complete.
"""

from vectorized import vec_eval


def test_a_shard_spec_is_parsed_one_based():
    assert vec_eval.parse_shard('') == (1, 1)
    assert vec_eval.parse_shard('1/1') == (1, 1)
    assert vec_eval.parse_shard('3/12') == (3, 12)
    assert vec_eval.parse_shard('12/12') == (12, 12)


def test_a_bad_shard_spec_is_refused_rather_than_guessed():
    for raw in ('0/4', '5/4', '-1/4', '4/0', 'x/4', '3', '3/4/5', '3/'):
        try:
            vec_eval.parse_shard(raw)
        except SystemExit:
            continue
        raise AssertionError('{0!r} should have been refused'.format(raw))


def test_the_shards_partition_the_selection_exactly_once():
    """Concatenating every shard must reproduce the selection, with nothing lost or repeated."""
    for total in (1, 2, 7, 40, 3222):
        steps = [1000 * (i + 1) for i in range(total)]
        for shards in (1, 2, 3, 4, 12, 16):
            pieces = [vec_eval.shard_steps(steps, i, shards) for i in range(1, shards + 1)]
            flat = [step for piece in pieces for step in piece]
            assert sorted(flat) == steps, (total, shards)
            assert len(flat) == len(set(flat)) == total, (total, shards)
            # Sizes differ by at most one, so no shard is handed a disproportionate share.
            sizes = [len(piece) for piece in pieces]
            assert max(sizes) - min(sizes) <= 1, (total, shards, sizes)


def test_shards_interleave_rather_than_taking_contiguous_blocks():
    """Per-checkpoint cost tracks policy quality, which drifts along a run, so contiguous blocks
    would give one shard every slow checkpoint. The stride is what keeps the shards finishing
    together — and a contiguous implementation passes the partition test above."""
    steps = list(range(24))
    first = vec_eval.shard_steps(steps, 1, 4)
    assert first == [0, 4, 8, 12, 16, 20], first
    assert vec_eval.shard_steps(steps, 4, 4) == [3, 7, 11, 15, 19, 23]
    # A contiguous split would have handed shard 1 a solid run of the earliest checkpoints.
    assert first != steps[:6]


# ------------------------------------------------------------------------------------- resume

def _payload(tmp, episodes, rows):
    import json, os
    path = os.path.join(tmp, 'r_checkpoint_evals_x.json')
    with open(path, 'w') as handle:
        json.dump({'episodes_per_checkpoint': episodes, 'results': rows}, handle)
    return path


def _row(step, episodes, perfect_at=None, **extra):
    scores = [95] * episodes
    if perfect_at is not None:
        scores[perfect_at] = 40
    row = {'step': step, 'episodes': episodes, 'episode_scores': scores,
           'episode_perfect': [int(s == 95) for s in scores],
           'episode_rewards': [float(s) for s in scores], 'seconds': 1.0}
    row.update(extra)
    return row


def test_resume_reuses_only_rows_measured_at_this_runs_depth():
    """A 100-episode row is not a 500-episode measurement of that step.

    Reusing one would put rows of two different weights in a file that claims to be flat, which is
    the single property that makes a vec result safe to pool without an equal-effort correction.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = _payload(tmp, 100, [_row(1000, 100), _row(2000, 100)])
        assert sorted(vec_eval.load_resumable(path, 100)) == [1000, 2000]
        # whole-file depth mismatch: the caller changed VEC_EVAL_EPISODES, so reuse nothing
        assert vec_eval.load_resumable(path, 500) == {}


def test_resume_skips_rows_it_cannot_faithfully_rebuild():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            _row(1000, 100),                                  # good
            _row(2000, 100, abandoned=True),                  # gated short row
            {'step': 3000, 'episodes': 100},                  # predates per-episode storage
            _row(4000, 60),                                   # wrong depth for this run
        ]
        rows[0]['step'] = 1000
        path = _payload(tmp, 100, rows)
        got = vec_eval.load_resumable(path, 100)
        assert sorted(got) == [1000], sorted(got)
        assert len(got[1000]['scores']) == 100


def test_resume_of_a_missing_or_corrupt_file_is_not_an_error():
    """A first run has no file, and a run killed mid-write can leave a truncated one. Neither may
    take down the relaunch that is supposed to recover from it."""
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert vec_eval.load_resumable(os.path.join(tmp, 'nope.json'), 100) == {}
        bad = os.path.join(tmp, 'bad.json')
        with open(bad, 'w') as handle:
            handle.write('{"results": [')
        assert vec_eval.load_resumable(bad, 100) == {}


def test_a_resumable_row_round_trips_through_build_row():
    """What resume feeds back must be what the engine would have produced, or the rebuilt file
    differs from an uninterrupted one in a field nobody checks."""
    import tempfile, eval_plan
    with tempfile.TemporaryDirectory() as tmp:
        original = _row(7000, 100, perfect_at=3)
        path = _payload(tmp, 100, [original])
        held = vec_eval.load_resumable(path, 100)[7000]
        rebuilt = eval_plan.build_row(7000, held, None)
        assert rebuilt['episodes'] == 100
        assert rebuilt['perfect_games'] == 99
        assert rebuilt['episode_scores'] == original['episode_scores']


def test_a_whole_file_depth_mismatch_says_so_instead_of_blaming_the_rows():
    """The two depth checks overlap on the return value, and only this pins the outer one.

    Dropping the whole-file check leaves the per-row check rejecting every row, so
    `load_resumable` still returns `{}` and every other fixture here passes — a surviving mutant.
    What changes is the message, and the message is the whole point: a human relaunching a job
    reads it to decide whether the resume did what they meant. "it holds 100-episode rows, this
    run wants 500" is actionable; "1200 stored rows could not be reused" reads like corruption.
    """
    import contextlib, io, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = _payload(tmp, 100, [_row(1000, 100), _row(2000, 100), _row(3000, 100)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert vec_eval.load_resumable(path, 500) == {}
        said = buf.getvalue()
        assert 'holds 100-episode rows' in said, said
        assert 'could not be reused' not in said, (
            'a depth mismatch was reported as unusable rows: {0!r}'.format(said))
