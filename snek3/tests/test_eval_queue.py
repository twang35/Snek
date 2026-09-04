"""The stage-A queue's protocol: ownership, claiming, ordering and the forward-progress guarantee.

Every fixture here is about a property the design *rests* on rather than about plumbing, because the
plumbing is a rename and a JSON file. The ones that matter most are `take_back` (no arm can wait on a
worker that is not coming) and the episode-count grouping (a row's denominator must be the one the arm
asked for).
"""

import json
import os
import time

import pytest

from tools import eval_queue
from tools import eval_worker


@pytest.fixture
def queue(tmp_path):
    """A runs directory of its own, so nothing here can see or touch a real arm's queue."""
    return str(tmp_path / 'runs')


FIELDS = {'epsilon': 0.4, 'guided_fraction': 0.0, 'steps_per_second': 500.0, 'fork': None}


# ---------------------------------------------------------------- claiming

def test_only_one_claimer_wins_the_same_step(queue):
    """The rename *is* the claim, so a second attempt must come back empty rather than duplicate work."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    first = eval_queue.claim('arm', 1000, queue)
    second = eval_queue.claim('arm', 1000, queue)
    assert first is not None and first['fields']['epsilon'] == 0.4
    assert second is None


def test_a_claimed_step_is_no_longer_pending(queue):
    """Otherwise a second worker would keep picking it up and failing the claim every round."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    assert eval_queue.pending(queue) == {'arm': [1000]}
    eval_queue.claim('arm', 1000, queue)
    assert eval_queue.pending(queue) == {}


# ---------------------------------------------------------------- the two halves of a row

def test_the_done_file_carries_the_trainers_half_too(queue):
    """A resume has no memory of what it offered, so the result file has to be self-contained.

    Without this the fields would live only in the trainer's process, and a restarted arm could not
    build a row for a measurement that landed while it was down — `steps_per_second` and the fork
    counters describe a window of training that is over and cannot be recomputed.
    """
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    payload = eval_queue.claim('arm', 1000, queue)
    eval_queue.complete('arm', 1000, {'scores': [1.0]}, payload['fields'], queue)
    landed = eval_queue.landed('arm', 1000, queue)
    assert landed['held'] == {'scores': [1.0]}
    assert landed['fields'] == FIELDS


def test_fields_are_readable_at_every_stage_a_step_passes_through(queue):
    """`fields_of` is what a reclaim after a grace period reads, and by then the step may be anywhere."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    assert eval_queue.fields_of('arm', 1000, queue) == FIELDS      # unclaimed request
    payload = eval_queue.claim('arm', 1000, queue)
    assert eval_queue.fields_of('arm', 1000, queue) == FIELDS      # claim, pid not ours to guess
    eval_queue.complete('arm', 1000, {'scores': [1.0]}, payload['fields'], queue)
    assert eval_queue.fields_of('arm', 1000, queue) == FIELDS      # done


# ---------------------------------------------------------------- forward progress

def test_take_back_succeeds_on_an_unclaimed_request_and_returns_its_fields(queue):
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    assert eval_queue.take_back('arm', 1000, queue) == FIELDS


def test_take_back_reports_failure_when_a_worker_holds_the_step(queue):
    """The trainer measures it anyway after a grace period — but it has to *know* it is duplicating."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    eval_queue.claim('arm', 1000, queue)
    assert eval_queue.take_back('arm', 1000, queue) is None


def test_a_worker_cannot_publish_a_step_the_trainer_reclaimed(queue):
    """The stale-result guard. Its absence leaves a `.done` nobody reads for the length of the batch.

    Modelled exactly as it happens: the worker claims, the trainer takes the work back and retires the
    step, and only then does the worker finish.
    """
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    payload = eval_queue.claim('arm', 1000, queue)
    eval_queue.retire('arm', 1000, queue)
    assert eval_queue.complete('arm', 1000, {'scores': [1.0]}, payload['fields'], queue) is False
    assert eval_queue.landed('arm', 1000, queue) is None


def test_retire_removes_every_file_for_a_step_whatever_stage_it_reached(queue):
    """A step can hold a request, a claim and a done at once after a reclaim races a worker."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    eval_queue.claim('arm', 1000, queue)
    eval_queue.complete('arm', 1000, {'scores': [1.0]}, FIELDS, queue)
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)          # a second offer, as a resume makes
    eval_queue.retire('arm', 1000, queue)
    assert eval_queue.outstanding('arm', queue) == []


# ---------------------------------------------------------------- resume

def test_outstanding_reports_a_step_at_any_stage_so_a_resume_adopts_it(queue):
    """A killed trainer leaves requests and claims behind; both are steps whose row is still owed."""
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    eval_queue.enqueue('arm', 2000, FIELDS, 100, queue)
    eval_queue.enqueue('arm', 3000, FIELDS, 100, queue)
    eval_queue.claim('arm', 2000, queue)
    eval_queue.complete('arm', 2000, {'scores': [1.0]}, FIELDS, queue)
    assert eval_queue.outstanding('arm', queue) == [1000, 2000, 3000]


def test_outstanding_ignores_a_half_written_staging_file(queue):
    """A trainer killed mid-write leaves `<step>.req.<pid>.partial`, which is not a step at all."""
    folder = eval_queue.policy_directory('arm', queue)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, '1000.req.999.partial'), 'w') as handle:
        handle.write('{')
    assert eval_queue.outstanding('arm', queue) == []


def test_sweep_keeps_what_is_still_pending_and_removes_the_rest(queue):
    eval_queue.enqueue('arm', 1000, FIELDS, 100, queue)
    eval_queue.enqueue('arm', 2000, FIELDS, 100, queue)
    assert eval_queue.sweep('arm', keep=[2000], runs_dir=queue) == 1
    assert eval_queue.outstanding('arm', queue) == [2000]


# ---------------------------------------------------------------- episode depth

def test_a_round_never_mixes_episode_counts(queue):
    """`measure_stream` takes one count for the whole call, so a mixed round would silently
    re-denominate somebody's rows — and rows at two depths are not comparable (invariant 8)."""
    for step in (1000, 2000, 3000):
        eval_queue.enqueue('real', step, FIELDS, 100, queue)
    eval_queue.enqueue('smoke', 1000, FIELDS, 10, queue)
    candidates = eval_worker._round_steps(eval_queue.pending(queue))
    episodes, chosen = eval_worker._same_depth(
        candidates, lambda policy, step: eval_queue.episodes_of(policy, step, queue))
    assert episodes == 100
    assert chosen == [('real', 1000), ('real', 2000), ('real', 3000)]


def test_the_majority_depth_wins_rather_than_the_first_seen(queue):
    """Round robin puts the alphabetically first arm in slot 0, so "first" would let one smoke test
    halve three real arms' throughput every round."""
    eval_queue.enqueue('aaa-smoke', 1000, FIELDS, 10, queue)
    for step in (1000, 2000):
        eval_queue.enqueue('zzz-real', step, FIELDS, 100, queue)
    candidates = eval_worker._round_steps(eval_queue.pending(queue))
    assert candidates[0][0] == 'aaa-smoke'
    episodes, _ = eval_worker._same_depth(
        candidates, lambda policy, step: eval_queue.episodes_of(policy, step, queue))
    assert episodes == 100


def test_a_round_interleaves_arms_rather_than_draining_one(queue):
    """Each arm's schedule lags by its own depth, so one arm's backlog must not hold another's."""
    taken = eval_worker._round_steps({'a': [1, 2, 3], 'b': [10, 11, 12]})
    assert taken == [('a', 1), ('b', 10), ('a', 2), ('b', 11), ('a', 3), ('b', 12)]


# ---------------------------------------------------------------- worker slots

def test_a_slot_is_claimed_once_however_many_arms_race_for_it(queue):
    """Four arms launched in the same second must produce `workers` workers, not four times as many.

    The second attempt returns False *because this process is alive and holds it*, which is the whole
    mechanism: a live holder blocks the slot whoever it is, so a caller cannot spawn a duplicate by
    asking twice. `take_slot` is not re-entrant and must not be.
    """
    assert eval_queue.take_slot(0, queue) is True
    assert eval_queue.take_slot(0, queue) is False
    assert eval_queue.live_runs.read(eval_queue.worker_slot(0, queue)) == os.getpid()
    assert eval_queue.live_workers(queue) == [0]


def test_a_slot_left_by_a_dead_process_is_reclaimable(queue):
    """Otherwise a box that ran one batch would never start a worker again."""
    os.makedirs(eval_queue.workers_directory(queue), exist_ok=True)
    with open(eval_queue.worker_slot(0, queue), 'w') as handle:
        handle.write('999999999\n')                 # a pid that cannot exist
    assert eval_queue.live_workers(queue) == []
    assert eval_queue.take_slot(0, queue) is True


# ---------------------------------------------------------------- the switch

def test_the_queue_is_on_unless_turned_off(monkeypatch):
    """On by default since 2026-08-29; `SNEK_EVAL_QUEUE=0` is how b1 and b2 stay diffable.

    Both directions are asserted because the off switch is the load-bearing one: a queued arm's
    stage-A rows are not bit-reproducible from its seed, so an arm being diffed byte-for-byte against
    b1 or b2 must be able to get the unqueued path back.

    Read from `train.build_config` rather than from a helper here, because `train.py` is the single
    reader of these three variables — a second reader in this module is exactly how a config passed in
    programmatically got silently overridden by the environment.
    """
    import train
    monkeypatch.delenv('SNEK_EVAL_QUEUE', raising=False)
    assert train.build_config()['eval_queue'] is True
    monkeypatch.setenv('SNEK_EVAL_QUEUE', '0')
    assert train.build_config()['eval_queue'] is False


def test_depth_zero_survives_the_config_because_it_is_the_verification_mode(monkeypatch):
    """At 0 a queued arm reclaims in the same drain and reproduces an unqueued one bit for bit.

    A `max(1, ...)` clamp anywhere on this path would silently turn the equivalence fixture into a
    depth-1 run, which lags by one interval and is *not* bit-identical — a test that passes for the
    wrong reason.
    """
    import train
    monkeypatch.setenv('SNEK_EVAL_QUEUE_DEPTH', '0')
    assert train.build_config()['eval_queue_depth'] == 0
    monkeypatch.delenv('SNEK_EVAL_QUEUE_DEPTH')
    assert train.build_config()['eval_queue_depth'] == eval_queue.DEFAULT_DEPTH


# ---------------------------------------------------------------- a real round

def _plant_arm(tmp_path, policy, width=4):
    """A policy directory with a sidecar and one checkpoint at step 1000, ready to be measured.

    Real files rather than mocks, because the thing under test is that a worker restores a checkpoint
    by step and measures it at the *requested* depth — a fake `policy_fn` would test the plumbing and
    skip both halves of that.
    """
    import torch
    from dqn import net as network
    from env import constants
    from tools import arch as arch_tools, checkpoints

    folder = str(tmp_path / 'savedPolicies' / policy)
    arch = arch_tools.build_arch([width], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='dqn')
    arch_tools.write_arch(folder, arch)
    checkpoints.save(folder, 1000, network.build(arch, 'cpu', seed=1))
    return folder


def test_a_round_measures_each_arm_at_the_depth_it_asked_for(tmp_path, monkeypatch):
    """**The row's denominator must be the arm's, never the worker's.** A worker that imposed its own
    count gave a `SNEK_GRAPH_EVAL_EPISODES=10` smoke arm 100-episode rows, and rows at two depths are
    not comparable (`docs/invariants.md` invariant 8).

    Two arms at different depths, so the round has to pick one and leave the other for the next round.
    A direct `_same_depth` fixture does not cover this: the bug was in `run_round` ignoring what
    `_same_depth` returned, and that mutation survived until this test existed.
    """
    from env import constants as env_constants
    queue = str(tmp_path / 'runs')
    monkeypatch.setattr(env_constants, 'POLICY_DIR', str(tmp_path / 'savedPolicies'))
    _plant_arm(tmp_path, 'deep')
    _plant_arm(tmp_path, 'shallow')
    eval_queue.enqueue('deep', 1000, FIELDS, 6, queue)
    eval_queue.enqueue('shallow', 1000, FIELDS, 3, queue)

    worker = eval_worker.Worker(0, runs_dir=queue, episodes=999)   # 999 must never be used
    assert worker.run_round() == 1, 'one round, one depth'
    assert worker.run_round() == 1, 'the other arm follows in the next round'

    for policy, asked in (('deep', 6), ('shallow', 3)):
        landed = eval_queue.landed(policy, 1000, queue)
        assert len(landed['held']['scores']) == asked, policy
        assert len(landed['held']['perfect']) == asked
        assert landed['fields'] == FIELDS


def test_a_round_skips_a_checkpoint_whose_file_has_gone(tmp_path, monkeypatch):
    """The trainer prunes rejected checkpoints and retires ones it measured itself, so a round can
    reach a step whose file is no longer there. One missing file must not abandon the rest of the
    round.
    """
    from env import constants as env_constants
    queue = str(tmp_path / 'runs')
    monkeypatch.setattr(env_constants, 'POLICY_DIR', str(tmp_path / 'savedPolicies'))
    folder = _plant_arm(tmp_path, 'arm')
    eval_queue.enqueue('arm', 1000, FIELDS, 3, queue)
    eval_queue.enqueue('arm', 2000, FIELDS, 3, queue)      # never had a checkpoint written

    worker = eval_worker.Worker(0, runs_dir=queue)
    assert worker.run_round() == 1
    assert eval_queue.landed('arm', 1000, queue) is not None
    assert eval_queue.landed('arm', 2000, queue) is None


def test_a_slot_is_never_empty_between_creation_and_pid(queue):
    """Regression for 2026-09-03: eight arms launched in the same instant gave seven slot-0 workers.

    The file used to be created with `O_EXCL` and written a moment later; a rival reading it in that
    moment saw no pid, took the slot for stale, and claimed it. Now the file is born holding the pid,
    so the first read of a freshly claimed slot is the claimer.
    """
    assert eval_queue.take_slot(0, queue) is True
    assert eval_queue.live_runs.read(eval_queue.worker_slot(0, queue)) == os.getpid()
    assert not [n for n in os.listdir(eval_queue.workers_directory(queue)) if 'claim' in n]


def test_an_empty_fresh_slot_is_a_claim_in_progress_not_a_stale_one(queue):
    """An older process, or a crash mid-write, can still leave an empty slot; young means in progress."""
    os.makedirs(eval_queue.workers_directory(queue), exist_ok=True)
    path = eval_queue.worker_slot(0, queue)
    open(path, 'w').close()
    assert eval_queue.take_slot(0, queue) is False
    assert eval_queue.live_runs.read(path) is None
    stale = time.time() - eval_queue.SLOT_CLAIM_GRACE_SECONDS - 1
    os.utime(path, (stale, stale))
    assert eval_queue.take_slot(0, queue) is True
    assert eval_queue.live_runs.read(path) == os.getpid()
