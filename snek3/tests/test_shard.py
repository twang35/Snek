"""A shard measuring a slice, and the two properties that make a wave safe to interrupt.

**A shard's rows are on disk after every completed measurement**, so a `kill -9` costs nothing and
the wave can read exact progress off the files. **A restarted shard skips what it already did**, so a
resumed file is identical to an uninterrupted one rather than merely similar — snek2 lost 192 rows
and 7,534 episodes in one incident for want of this.

The nets are tiny (`30 -> 4 -> 3`) and untrained, so the policies die quickly and a whole shard runs
in a second. The observation length and action count have to be real, because the real env is driven.
"""

import os

import pytest

from dqn import net as network
from tools import arch as arch_tools
from tools import checkpoints
from tools import shard


@pytest.fixture
def policy(tmp_path):
    """A policy directory with six checkpoints of a tiny untrained net."""
    directory = str(tmp_path / 'policy')
    arch = arch_tools.build_arch([4], 3, 30, __import__('env.constants', fromlist=['x']).OBS_ERA)
    arch_tools.write_arch(directory, arch)
    for step in range(1000, 7000, 1000):
        checkpoints.save(directory, step, network.QNet(30, [4], 3))
    return directory


def test_a_slice_is_measured_and_written(tmp_path, policy):
    out = str(tmp_path / 'out.json')
    rows = shard.measure_slice(policy, [1000, 2000, 3000], 4, out)
    assert [row['step'] for row in rows] == [1000, 2000, 3000]
    assert all(row['episodes'] == 4 for row in rows)
    assert os.path.exists(out)


def test_the_rows_are_full_length(tmp_path, policy):
    # The single-stage protocol's whole promise. A short row would reach a result file as a shorter
    # sample whose rate is not comparable with its neighbours'.
    rows = shard.measure_slice(policy, [1000, 2000], 6, str(tmp_path / 'out.json'))
    for row in rows:
        assert len(row['episode_scores']) == 6
        assert len(row['episode_perfect']) == 6
        assert len(row['episode_rewards']) == 6
        assert None not in row['episode_scores']


def test_a_second_run_skips_what_the_first_measured(tmp_path, policy, capsys):
    out = str(tmp_path / 'out.json')
    first = shard.measure_slice(policy, [1000, 2000], 4, out)
    capsys.readouterr()
    again = shard.measure_slice(policy, [1000, 2000], 4, out)
    assert 'already measured' in capsys.readouterr().out
    # Byte-identical rows, not merely the same count: a resume must not re-measure and overwrite.
    assert again == first


def test_a_resume_tops_up_the_steps_it_has_not_done(tmp_path, policy):
    out = str(tmp_path / 'out.json')
    shard.measure_slice(policy, [1000, 2000], 4, out)
    rows = shard.measure_slice(policy, [1000, 2000, 3000, 4000], 4, out)
    assert [row['step'] for row in rows] == [1000, 2000, 3000, 4000]


def test_no_resume_remeasures_everything(tmp_path, policy):
    """`resume=False` measures a step already on disk; the default skips it.

    Asserted through `on_row`, which fires once per *measurement*, rather than by comparing the two
    rows under different seeds. That comparison could not work: the fixture's net is untrained and
    dies on move one, so it scores 0 on every episode of every seed and the rows come out identical
    whether the step was re-measured or silently kept. What made it pass at all was the `seconds`
    field differing by timing luck — so it was flaky as well as degenerate, and it failed for real
    once the two runs landed in the same tick.
    """
    out = str(tmp_path / 'out.json')
    shard.measure_slice(policy, [1000], 4, out)

    measured = []
    shard.measure_slice(policy, [1000], 4, out, resume=False,
                        on_row=lambda row: measured.append(row['step']))
    assert measured == [1000], 'resume=False must measure the step again'

    del measured[:]
    shard.measure_slice(policy, [1000], 4, out,
                        on_row=lambda row: measured.append(row['step']))
    assert measured == [], 'the default must skip a step already on disk'


def test_the_file_carries_the_header_a_reader_needs(tmp_path, policy):
    out = str(tmp_path / 'out.json')
    shard.measure_slice(policy, [1000], 4, out, policy='b1a-thing')
    from tools import results
    payload = results.read(out)
    assert payload['policy'] == 'b1a-thing'
    assert payload['episodes'] == 4
    # The reward config, because two runs of the same checkpoint can legitimately differ in
    # `avg_reward` and nothing else records which config produced a file.
    assert 'chase_safe' in payload['config']
    assert payload['arch']['fc_layer_params'] == [4]


def test_the_stage_a_screen_is_carried_into_the_rows(tmp_path, policy):
    rows = shard.measure_slice(policy, [1000, 2000], 4, str(tmp_path / 'out.json'),
                               stage_a={1000: 96.0})
    by_step = {row['step']: row for row in rows}
    assert by_step[1000]['stage_a_percent'] == 96.0
    assert by_step[2000]['stage_a_percent'] is None


def test_the_net_pool_is_reused_rather_than_grown(tmp_path, policy):
    """One net per *resident* checkpoint, not one per checkpoint measured.

    Not about memory — a net is 45 KB — but about the pool actually working: if it never handed one
    back, a 3,000-checkpoint arm would allocate 3,000 identical modules.
    """
    pool = shard._NetPool(arch_tools.build_arch([4], 3, 30, 'x'))
    first = pool.take()
    pool.give_back(first)
    assert pool.take() is first
    assert pool.built == 1
    second = pool.take()
    assert second is not first and pool.built == 2


def test_a_shard_writes_after_every_row(tmp_path, policy):
    """Which is what lets the wave read exact progress and makes a kill free.

    Checked by counting rows on disk from inside the completion callback, one measurement before the
    shard finishes.
    """
    from tools import results
    out = str(tmp_path / 'out.json')
    seen = []

    def on_row(row):
        seen.append(len(results.rows_of(results.read(out))))

    shard.measure_slice(policy, [1000, 2000, 3000], 4, out, on_row=on_row)
    # `on_row` fires after the flush, so by the first callback there is already a row on disk.
    assert seen and seen[0] >= 1
    assert seen == sorted(seen)
