"""`tools/eta.py` and the estimate arithmetic in `desktop/daemon/daemon.py`'s `build_at_a_glance`:
the `| ~40m left` and `| ~4.5h` on status.json's lines and the box's `remaining` total (2026-09-05).
"""

import json
import os
import time

import pytest

from desktop.daemon import daemon as daemon_module
from tools import eta
from tools import live_runs


# ---------------------------------------------------------------- the arithmetic on the lines

def test_format_eta_reads_as_minutes_then_tenths_of_hours_then_whole_hours():
    assert daemon_module.format_eta(None) is None
    assert daemon_module.format_eta(0) == '<1m'
    assert daemon_module.format_eta(59) == '<1m'
    assert daemon_module.format_eta(40 * 60) == '40m'
    assert daemon_module.format_eta(1.4 * 3600) == '1.4h'
    assert daemon_module.format_eta(12.4 * 3600) == '12h'


def test_a_wave_takes_its_slowest_arm_and_waves_and_passes_add():
    """Eight arms of one wave run at once, so their group is the longest of them; wave 2 runs after
    wave 1, and every pass runs alone, so those add. A job without an estimate contributes nothing."""
    arms = [{'type': 'train', 'wave': 1, 'eta_seconds': 3600}, {'type': 'train', 'wave': 1, 'eta_seconds': 5400},
            {'type': 'train', 'wave': 2, 'eta_seconds': 1800}, {'type': 'train', 'wave': 2}]
    assert daemon_module.group_eta(arms) == 5400 + 1800
    passes = [{'type': 'eval', 'eta_seconds': 600}, {'type': 'eval', 'eta_seconds': 300}]
    assert daemon_module.group_eta(passes) == 900
    # running arms carry no wave: they are one wave
    assert daemon_module.group_eta([{'type': 'train', 'eta_seconds': 100}, {'type': 'train', 'eta_seconds': 250}]) == 250
    assert daemon_module.group_eta([{'type': 'train'}, {'type': 'eval'}]) is None


def test_the_lines_carry_the_estimates_and_remaining_adds_them_up():
    now = time.mktime((2026, 9, 5, 12, 0, 0, 0, 0, -1))         # a Saturday, noon
    running = [{'id': 'b17-stageb-w2', 'type': 'eval', 'policies': ['b17ai-clip015-seed1', 'b17aj-clip015-seed2'],
                'eta_seconds': 40 * 60}]
    queued = [{'id': 'b18{0}-gc0-seed1'.format(letter), 'type': 'train', 'policy': 'b18{0}-gc0-seed1'.format(letter),
               'policies': ['b18{0}-gc0-seed1'.format(letter)], 'wave': wave, 'eta_seconds': seconds}
              for letter, wave, seconds in (('a', 1, 5000), ('b', 1, 5400), ('c', 2, 5400))]
    queued.append({'id': 'b18-stageb', 'type': 'eval', 'policies': ['b18a-gc0-seed1', 'b18b-gc0-seed1'],
                   'eta_seconds': 3600})
    glance = daemon_module.build_at_a_glance(running, queued, {}, now=now)
    assert glance['running'] == ['b17 | clip015 | stage B (2 arms) | ~40m left']
    assert glance['queued'] == ['b18 training | gc0 (3 arms) | ~3.0h',           # 5400 + 5400, not 3 x 5400
                                'b18 evals | gc0 (2 arms) | ~1.0h']
    # 40 m + 3 h + 1 h = 4.7 h from noon
    assert glance['remaining'] == '~4.7h of work: 40m running + 4.0h queued; clear ~Sat 16:40'


def test_without_estimates_the_lines_are_as_before_and_remaining_is_null():
    glance = daemon_module.build_at_a_glance(
        [{'id': 'b1a-x', 'type': 'train', 'policy': 'b1a-x', 'step': 1, 'max_steps': 2}],
        [{'id': 'b1-stageb', 'type': 'eval', 'policies': ['b1a-x']}], {}, now=0)
    assert glance['running'] == ['b1 | x | training 50% (1 arm)']
    assert glance['queued'] == ['b1 evals | x (1 arm)']
    assert glance['remaining'] is None


def test_a_partly_estimated_queue_says_so_and_an_idle_box_names_only_the_queued_part():
    now = time.mktime((2026, 9, 5, 12, 0, 0, 0, 0, -1))
    queued = [{'id': 'b19a-x-seed1', 'type': 'train', 'policy': 'b19a-x-seed1', 'wave': 1},
              {'id': 'b19-stageb', 'type': 'eval', 'policies': ['b19a-x-seed1'], 'eta_seconds': 1800}]
    glance = daemon_module.build_at_a_glance([], queued, {}, now=now)
    assert glance['queued'] == ['b19 training | x (1 arm)', 'b19 evals | x (1 arm) | ~30m']
    assert glance['remaining'] == '~30m of work: 30m queued; clear ~Sat 12:30'
    # a group with an estimate on some jobs and none on others is flagged, since its total is a floor
    mixed = [{'id': 'b19a-x-seed1', 'type': 'train', 'policy': 'b19a-x-seed1', 'wave': 1, 'eta_seconds': 600},
             {'id': 'b19b-x-seed2', 'type': 'train', 'policy': 'b19b-x-seed2', 'wave': 2}]
    assert daemon_module.build_at_a_glance([], mixed, {}, now=now)['remaining'].endswith('(some jobs have no estimate yet)')


def test_the_laptops_remaining_is_folded_in_beside_its_lines():
    text = json.dumps({'iso': '2026-09-05T12:00:00', 'at_a_glance': {
        'running': [], 'queued': ['b19 evals | x (1 arm) | ~30m'], 'attention': [],
        'remaining': '~30m of work: 30m queued; clear ~Sat 12:30'}})
    glance = daemon_module.with_laptop({'running': [], 'queued': [], 'attention': [], 'remaining': None}, text)
    assert glance['laptop_remaining'] == '~30m of work: 30m queued; clear ~Sat 12:30'
    assert daemon_module.with_laptop({}, '')['laptop_remaining'] is None


# ---------------------------------------------------------------- an arm's rates

def _arm(tmp_path, policy, rows, started, ended):
    """An arm on disk: `arch.json` touched at `started`, `_evals.json` with `rows` touched at `ended`."""
    runs, policies = tmp_path / 'runs', tmp_path / 'policies'
    (policies / policy).mkdir(parents=True, exist_ok=True)
    runs.mkdir(exist_ok=True)
    arch = policies / policy / 'arch.json'
    arch.write_text('{}')
    os.utime(str(arch), (started, started))
    evals = runs / '{0}_evals.json'.format(policy)
    evals.write_text(json.dumps({'summary': {'step': rows[-1][0] if rows else 0},
                                 'evals': [{'step': step, 'steps_per_second': rate} for step, rate in rows]}))
    os.utime(str(evals), (ended, ended))
    return str(runs), str(policies)


def test_an_arms_wall_rate_is_its_steps_over_the_gap_between_its_two_files(tmp_path):
    """b16bg: rows at 20k steps/s accounted for 42 minutes of an 84-minute life. The wall rate is the
    one a queued arm is estimated at, and it is half the row rate."""
    rows = [(i * 1000, 20000.0) for i in range(1, 11)]                 # 10,000 steps, 0.5 s of rows
    runs, policies = _arm(tmp_path, 'b1a-x', rows, started=1000.0, ended=1000.0 + 100.0)
    assert eta.wall_seconds('b1a-x', runs, policies) == 100.0
    assert eta.wall_rate('b1a-x', runs, policies) == 100.0              # 10,000 steps / 100 s
    seconds, step = eta.accounted_seconds('b1a-x', runs)
    assert seconds == pytest.approx(0.5) and step == 10000
    # too young to read, and missing either file: None, never a guess
    young, _ = _arm(tmp_path / 'y', 'b1b-x', rows, started=1000.0, ended=1030.0)
    assert eta.wall_rate('b1b-x', young, str(tmp_path / 'y' / 'policies')) is None
    assert eta.wall_rate('b1c-x', runs, policies) is None


def test_a_running_arm_is_its_recent_rate_plus_the_overhead_it_has_shown(tmp_path):
    """Half-way at 100 steps/s in the loop, with rows accounting for half its wall clock: the second
    half takes the loop's 100 s plus the same overhead again, 200 s -- not the 100 s the rows say."""
    rows = [(i * 1000, 100.0) for i in range(1, 11)]                   # 10,000 steps, 100 s of rows
    runs, policies = _arm(tmp_path, 'b1a-x', rows, started=5000.0, ended=5200.0)
    assert eta.running_arm_seconds('b1a-x', 10000, 20000, runs, policies, now=5200.0) == 200.0
    # before its first row: the whole cap at the fallback rate, or nothing
    fresh, fresh_policies = _arm(tmp_path / 'f', 'b1b-x', [], started=1.0, ended=1.0)
    assert eta.running_arm_seconds('b1b-x', None, 20000, fresh, fresh_policies, fallback_rate=50.0) == 400.0
    assert eta.running_arm_seconds('b1b-x', None, 20000, fresh, fresh_policies) is None


def test_the_reference_rate_is_the_batchs_finished_arms_else_the_boxs_recent_arms(tmp_path):
    rows = [(i * 1000, 20000.0) for i in range(1, 11)]
    runs, policies = _arm(tmp_path, 'b1a-x', rows, started=0.0, ended=100.0)       # 100 steps/s
    _arm(tmp_path, 'b1b-x', rows, started=0.0, ended=200.0)                        # 50 steps/s
    _arm(tmp_path, 'b1c-x', rows, started=0.0, ended=400.0)                        # 25 steps/s
    assert eta.reference_rate(['b1a-x', 'b1b-x'], runs, policies) == 75.0
    assert eta.reference_rate([], runs, policies) == 50.0                          # the box's median
    # a pass's merged file is not an arm, whatever its name ends in
    (tmp_path / 'runs' / 'b1a-x_checkpoint_evals.json').write_text('{}')
    (tmp_path / 'runs' / 'b1a-x_checkpoint_evals_hof5000.json').write_text('{}')
    assert eta.recent_arms_rate(runs, policies) == 50.0
    # an arm training now is not the box's rate: with b1c live, the median is over b1a and b1b
    live_runs.register('b1c-x', os.getpid(), runs)
    assert eta.recent_arms_rate(runs, policies) == 75.0
    # ... unless nothing finished has ever run here, when the live arms are all there is
    live_runs.register('b1a-x', os.getpid(), runs)
    live_runs.register('b1b-x', os.getpid(), runs)
    assert eta.recent_arms_rate(runs, policies) == 50.0
    assert eta.reference_rate([], str(tmp_path / 'empty'), policies) is None
    assert eta.arm_seconds(30000, 50000, 100.0) == 200.0 and eta.arm_seconds(None, 50000, None) is None


# ---------------------------------------------------------------- passes and the ledger

def test_a_pass_is_the_ledgers_median_per_arm_times_the_arms_and_the_default_before_that(tmp_path):
    runs = str(tmp_path)
    assert eta.pass_seconds('stageb', 8, runs) == 55 * 60
    assert eta.pass_seconds('hof30k', 4, runs) == 4.5 * 60 / 2
    assert eta.pass_seconds('smoke', 4, runs) is None
    live_runs.record_duration('stageb', 1600.0, runs, arms=8, label='b1-stageb')      # 200 s per arm
    live_runs.record_duration('stageb', 800.0, runs, arms=8, label='b1-stageb-w2')    # 100 s per arm
    live_runs.record_duration('stageb', 1200.0, runs, arms=4, label='b1-stageb-w3')   # 300 s per arm
    assert eta.pass_seconds('stageb', 8, runs) == 1600.0                              # median 200 x 8
    assert eta.running_pass_seconds('stageb', 8, 1000.0, runs) == 600.0
    assert eta.running_pass_seconds('stageb', 8, 5000.0, runs) == eta.MIN_RUNNING_PASS_SECONDS


def _stage_b(runs, policy, label, percents, shard=None):
    """A stage-B file (or one shard) with one row per percent."""
    name = '{0}_checkpoint_evals{1}{2}.json'.format(policy, '_' + label if label else '',
                                                    '-s{0}of12'.format(shard) if shard else '')
    with open(os.path.join(runs, name), 'w') as handle:
        json.dump({'rows': [{'step': i, 'perfect_percent': p} for i, p in enumerate(percents)]}, handle)


def test_a_pass_measures_what_its_selector_picks_from_the_file_before_it(tmp_path):
    """stage B counts the stage-A rows at the screen, hof5000 the stage-B rows at 99, hof30k the hof5000
    rows at 99; an arm whose merged file exists counts exactly; a missing input file is None, not 0.
    b25a put 811 checkpoints into hof5000 where b19a put 20 (2026-09-07)."""
    runs = str(tmp_path)
    with open(os.path.join(runs, 'a_evals.json'), 'w') as handle:
        json.dump({'evals': [{'step': 1, 'perfect_percent': 97}, {'step': 2, 'perfect_percent': 96.9},
                             {'step': 3, 'perfect_percent': 100}, {'step': 4}]}, handle)
    assert eta.selected_checkpoints('stageb', 'a', runs) == 2
    assert eta.selected_checkpoints('hof5000', 'a', runs) is None            # no stage-B file yet
    _stage_b(runs, 'a', None, [99.2, 99.0, 99.4])                          # the cut is 99.2: 99.0 is out
    assert eta.selected_checkpoints('hof5000', 'a', runs) == 2
    assert eta.pass_checkpoints('stageb', ['a'], runs) == 3                  # merged: its rows, not the screen
    assert eta.pass_checkpoints('hof5000', ['a', 'b'], runs) is None         # b's input is not there
    assert eta.known_checkpoints('hof5000', ['a', 'b'], runs) == (2, 1)      # a's 2, b unknown
    _stage_b(runs, 'b', None, [99.9])
    assert eta.pass_checkpoints('hof5000', ['a', 'b'], runs) == 3
    assert eta.pass_checkpoints('hof30k', ['a'], runs) is None
    assert eta.pass_checkpoints('stageb', ['a'], runs, pending={'a'}) is None   # still training: unknown
    _stage_b(runs, 'a', 'hof5000', [99.2, 99.6, 50.0])                     # the hof30k cut is 99.6: 99.2 is out
    assert eta.pass_checkpoints('hof30k', ['a'], runs) == 1
    assert eta.selected_checkpoints('smoke', 'a', runs) is None


def test_a_pass_is_estimated_per_checkpoint_when_its_count_is_known_and_per_arm_when_not(tmp_path):
    runs = str(tmp_path)
    assert eta.pass_seconds('hof5000', 8, runs, checkpoints=100) == 370.0            # the default per checkpoint
    assert eta.pass_seconds('hof5000', 8, runs) == 9.5 * 60                          # unknown: per arm
    live_runs.record_duration('hof5000', 800.0, runs, arms=8, checkpoints=200)      # 4 s per checkpoint
    live_runs.record_duration('hof5000', 1200.0, runs, arms=8, checkpoints=200)     # 6
    live_runs.record_duration('hof5000', 400.0, runs, arms=8)                       # no count: per arm only
    assert eta.pass_seconds('hof5000', 8, runs, checkpoints=100) == 500.0           # median 5 x 100
    assert eta.pass_seconds('hof5000', 8, runs) == 800.0                            # median 100 per arm x 8
    assert eta.pass_seconds('hof5000', 8, runs, checkpoints=0) == 0.0               # nothing selected
    # 5 arms known at 100 checkpoints, 3 arms not yet knowable: 5 x 100 + 3 x 100 per arm
    assert eta.pass_seconds('hof5000', 8, runs, checkpoints=100, unknown_arms=3) == 800.0


def test_a_running_pass_is_estimated_from_its_own_progress(tmp_path):
    """Two arms done (10 and 20 rows) and the third's shards at 10 of its 30, 4,000 s in: 40 of the
    pass's 60 checkpoints at 100 s each leaves 2,000 s -- whatever the ledger says. Before its first
    row the ledger estimate less elapsed stands in, and never below the floor."""
    runs = str(tmp_path)
    for policy, rows in (('a', 10), ('b', 20), ('c', 30)):
        _stage_b(runs, policy, None, [99.5] * rows)
    _stage_b(runs, 'a', 'hof5000', [99.0] * 10)
    _stage_b(runs, 'b', 'hof5000', [99.0] * 20)
    _stage_b(runs, 'c', 'hof5000', [99.0] * 4, shard=1)
    _stage_b(runs, 'c', 'hof5000', [99.0] * 6, shard=2)
    assert eta.pass_progress('hof5000', ['a', 'b', 'c'], runs) == (40, 60)
    live_runs.record_duration('hof5000', 60.0, runs, arms=8, checkpoints=60)        # 1 s per checkpoint
    assert eta.running_pass_seconds('hof5000', 3, 4000.0, runs, policies=['a', 'b', 'c']) == 2000.0
    assert eta.running_pass_seconds('hof5000', 3, 4000.0, runs) == eta.MIN_RUNNING_PASS_SECONDS
    # nothing measured yet: the ledger's per-checkpoint estimate (200 x 1 s) less the 10 s elapsed
    _stage_b(runs, 'd', None, [99.5] * 200)
    assert eta.running_pass_seconds('hof5000', 1, 10.0, runs, policies=['d']) == 190.0
    assert eta.running_pass_seconds('hof5000', 1, 199.0, runs, policies=['d']) == eta.MIN_RUNNING_PASS_SECONDS
    # its total unknowable (no input file): the per-arm ledger stands in
    assert eta.running_pass_seconds('stageb', 1, 0.0, runs, policies=['e']) == 55 * 60 / 8.0


def test_the_ledger_keeps_the_last_entries_per_kind_and_shrugs_at_a_bad_file(tmp_path):
    runs = str(tmp_path)
    for i in range(live_runs.DURATIONS_KEEP + 3):
        live_runs.record_duration('hof5000', float(i), runs, arms=8)
    entries = live_runs.durations(runs)['hof5000']
    assert len(entries) == live_runs.DURATIONS_KEEP and entries[-1]['seconds'] == live_runs.DURATIONS_KEEP + 2
    with open(live_runs.durations_path(runs), 'w') as handle:
        handle.write('not json')
    assert live_runs.durations(runs) == {}
    live_runs.record_duration('hof30k', 10.0, runs, arms=2)
    assert list(live_runs.durations(runs)) == ['hof30k']
