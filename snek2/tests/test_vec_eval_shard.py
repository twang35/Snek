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
