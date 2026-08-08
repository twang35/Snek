"""Tests for `eval_workers` — independent evaluation workers.

The process machinery is not what can silently go wrong here; the **scheduling and counting logic**
is, which is why `split_quota` and `collect_results` are separate functions taking a queue and a flag
rather than methods reaching into `multiprocessing`. These tests drive them with fakes: no
TensorFlow, no subprocesses, no checkpoints.

The property most worth pinning is the one the batched path's barrier existed to protect.
`run_round`'s docstring: truncating in-flight episodes **reads high**, because perfect games average
~1780 steps against ~2200 for failures, so dropping slow episodes preferentially drops failures. The
quota design replaces the barrier, and the dangerous moment is abandonment — if `collect_results`
stopped draining the moment the stop flag went up, it would discard exactly the long episodes still
in flight and inflate the perfect rate. Several tests below exist only to hold that line.
"""
import collections

import eval_workers
from eval_workers import Episode, collect_results, split_quota


class FakeFlag:
    """Stands in for multiprocessing.Event, and records how often it was set."""

    def __init__(self):
        self.sets = 0

    def set(self):
        self.sets += 1

    def is_set(self):
        return self.sets > 0

    def clear(self):
        self.sets = 0


class FakeQueue:
    """A queue preloaded with the exact message sequence a test wants to replay."""

    def __init__(self, messages):
        self._messages = collections.deque(messages)
        self.gets = 0

    def get(self):
        self.gets += 1
        if not self._messages:
            raise AssertionError('collect_results asked for more messages than the test provided; '
                                 'it is probably not counting a DONE')
        return self._messages.popleft()


def episode(rank=0, score=95.0, perfect=True, reward=100.0, steps=1500):
    return (eval_workers.TAG_EPISODE, Episode(rank, score, perfect, reward, steps))


def done(rank=0, completed=1):
    return (eval_workers.TAG_DONE, rank, completed)


# ------------------------------------------------------------------ split_quota

def test_quota_splits_evenly_when_it_divides():
    assert split_quota(100, 10) == [10] * 10
    assert split_quota(80, 4) == [20, 20, 20, 20]


def test_quota_sums_to_exactly_the_request_when_it_does_not_divide():
    # The batched path rounded *up* to whole rounds, so 80 episodes on 14 workers became 84 - or in
    # the prototype's case silently dropped 10. Exactness is the point of this function.
    for episodes, workers in ((80, 14), (100, 3), (7, 5), (100, 7), (13, 4)):
        quotas = split_quota(episodes, workers)
        assert sum(quotas) == episodes, (episodes, workers, quotas)
        assert len(quotas) == workers
        assert max(quotas) - min(quotas) <= 1, 'work should be spread evenly: {0}'.format(quotas)


def test_quota_gives_idle_workers_zero_rather_than_failing():
    # Keeps the collect loop's "wait for every worker" invariant true with no special case.
    assert split_quota(3, 5) == [1, 1, 1, 0, 0]


def test_quota_rejects_nonsense():
    for episodes, workers in ((10, 0), (10, -1), (-1, 4)):
        try:
            split_quota(episodes, workers)
        except ValueError:
            continue
        raise AssertionError('split_quota({0}, {1}) should raise'.format(episodes, workers))


# ------------------------------------------------------------------ collect_results, ordinary path

def test_collects_every_episode_and_waits_for_every_worker():
    flag = FakeFlag()
    queue = FakeQueue([episode(0), episode(1), done(0, 1), episode(1), done(1, 2)])
    scores, perfect, rewards, steps, abandoned = collect_results(queue, 2, flag)
    assert len(scores) == 3
    assert len(perfect) == 3 and len(rewards) == 3
    assert steps == 4500
    assert abandoned is False
    assert flag.sets == 0


def test_perfect_flags_and_scores_are_carried_through_unchanged():
    flag = FakeFlag()
    queue = FakeQueue([
        episode(0, score=95.0, perfect=True, reward=180.0, steps=1780),
        episode(1, score=61.0, perfect=False, reward=40.0, steps=2200),
        done(0), done(1)])
    scores, perfect, rewards, steps, _ = collect_results(queue, 2, flag)
    assert scores == [95.0, 61.0]
    assert perfect == [True, False]
    assert rewards == [180.0, 40.0]
    assert steps == 3980


def test_a_worker_error_is_raised_not_swallowed():
    flag = FakeFlag()
    queue = FakeQueue([(eval_workers.TAG_ERROR, 2, 'boom')])
    try:
        collect_results(queue, 2, flag)
    except RuntimeError as error:
        assert 'boom' in str(error) and '2' in str(error)
        return
    raise AssertionError('a worker traceback must surface, not hang the measurement')


def test_stray_ready_and_loaded_acks_do_not_wedge_collection():
    """A late ack from the load phase must not be counted as an episode or as a done."""
    flag = FakeFlag()
    queue = FakeQueue([(eval_workers.TAG_READY, 0), (eval_workers.TAG_LOADED, 0, 1190000),
                       episode(0), done(0)])
    scores, _, _, _, _ = collect_results(queue, 1, flag)
    assert len(scores) == 1


# ------------------------------------------------------------------ abandonment and the bias line

def test_abandonment_sets_the_flag_once_the_rule_fires():
    flag = FakeFlag()
    queue = FakeQueue([episode(0, perfect=False), episode(1, perfect=False), done(0), done(1)])
    _, _, _, _, abandoned = collect_results(
        queue, 2, flag, should_abandon=lambda perfect, episodes: episodes >= 2)
    assert abandoned is True
    assert flag.sets == 1, 'the flag should be set exactly once, not per episode after'


def test_episodes_arriving_after_the_stop_flag_are_still_counted():
    """The bias line. Discarding in-flight episodes would inflate the perfect rate.

    Failures run ~2200 steps against ~1780 for wins, so the episodes still running when the flag
    goes up are disproportionately failures. Dropping them is exactly the truncation the barrier
    used to prevent, so `collect_results` must keep draining to the last DONE.
    """
    flag = FakeFlag()
    queue = FakeQueue([
        episode(0, perfect=False, steps=2200),
        # abandon fires here; these two are already in flight and must survive
        episode(1, perfect=False, steps=2200),
        episode(0, perfect=True, steps=1780),
        done(0), done(1)])
    scores, perfect, _, steps, abandoned = collect_results(
        queue, 2, flag, should_abandon=lambda p, e: e >= 1)
    assert abandoned is True
    assert len(scores) == 3, 'episodes completed after the flag were dropped: {0}'.format(scores)
    assert perfect == [False, False, True]
    assert steps == 6180


def test_the_abandon_rule_sees_the_running_tally_not_just_the_last_episode():
    seen = []
    flag = FakeFlag()
    queue = FakeQueue([episode(0, perfect=True), episode(0, perfect=False),
                       episode(0, perfect=True), done(0)])

    def rule(perfect_so_far, episodes_so_far):
        seen.append((perfect_so_far, episodes_so_far))
        return False

    collect_results(queue, 1, flag, should_abandon=rule)
    assert seen == [(1, 1), (1, 2), (2, 3)], seen


def test_no_abandon_rule_means_no_flag_and_no_abandonment():
    flag = FakeFlag()
    queue = FakeQueue([episode(0), episode(0), done(0)])
    _, _, _, _, abandoned = collect_results(queue, 1, flag)
    assert abandoned is False
    assert flag.sets == 0


# ------------------------------------------------------------------ progress reporting

def test_progress_is_reported_once_per_completed_wave():
    calls = []
    flag = FakeFlag()
    queue = FakeQueue([episode(0, perfect=True), episode(1, perfect=False),
                       episode(0, perfect=True), episode(1, perfect=True), done(0), done(1)])
    collect_results(queue, 2, flag, episodes_total=4,
                    on_progress=lambda *args: calls.append(args))
    # 4 episodes on 2 workers = 2 waves, so 2 reports rather than 4. Per-episode reporting made
    # write_results re-serialise the whole arm up to 100 times a checkpoint.
    assert len(calls) == 2, calls
    # (wave, waves_total, perfect_so_far, episodes_so_far, per_wave_perfect)
    assert calls[0][0] == 1 and calls[0][3] == 2
    assert calls[1][0] == 2 and calls[1][3] == 4


def test_the_wave_total_is_constant_from_the_first_report():
    """The x-axis bug: waves_total used to be "episodes seen so far" and grew every episode."""
    calls = []
    flag = FakeFlag()
    messages = [episode(i % 4) for i in range(12)] + [done(r) for r in range(4)]
    collect_results(FakeQueue(messages), 4, flag, episodes_total=100,
                   on_progress=lambda *args: calls.append(args))
    totals = {call[1] for call in calls}
    assert totals == {25}, 'waves_total moved during one checkpoint: {0}'.format(sorted(totals))


def test_the_wave_total_matches_a_screen_as_well_as_a_full_pass():
    for episodes, workers, expected in ((100, 4, 25), (20, 4, 5), (80, 4, 20), (100, 10, 10),
                                        (20, 3, 7)):
        calls = []
        messages = [episode(0)] * workers + [done(r) for r in range(workers)]
        collect_results(FakeQueue(messages), workers, FakeFlag(), episodes_total=episodes,
                       on_progress=lambda *args: calls.append(args))
        assert calls and calls[0][1] == expected, (episodes, workers, calls[0][1] if calls else None)


def test_per_wave_perfect_counts_group_by_worker_count():
    """`on_round`'s contract in eval_checkpoints is a list of per-round perfect counts.

    There are no real rounds here, so a "wave" is num_workers completed episodes — enough to keep
    the existing live chart meaningful without changing eval_progress.py.
    """
    calls = []
    flag = FakeFlag()
    queue = FakeQueue([episode(0, perfect=True), episode(1, perfect=True),
                       episode(0, perfect=False), episode(1, perfect=True),
                       done(0), done(1)])
    collect_results(queue, 2, flag, episodes_total=4,
                    on_progress=lambda *args: calls.append(args))
    assert calls[0][4] == [2], 'first wave of 2 episodes had 2 perfect'
    assert calls[1][4] == [2, 1], 'second wave had 1 perfect'


def test_progress_is_optional():
    flag = FakeFlag()
    queue = FakeQueue([episode(0), done(0)])
    scores, _, _, _, _ = collect_results(queue, 1, flag)   # no on_progress, no episodes_total
    assert len(scores) == 1


# ------------------------------------------------------------------ the unbiasedness invariant

def test_collection_does_not_depend_on_episode_duration():
    """The estimate must be identical however long the episodes took.

    This is the invariant the whole design rests on: `collect_results` may not treat a long episode
    differently from a short one. Same outcomes, wildly different step counts, same rate.
    """
    flag = FakeFlag()
    slow = FakeQueue([episode(0, perfect=True, steps=9000), episode(1, perfect=False, steps=50),
                      done(0), done(1)])
    fast = FakeQueue([episode(0, perfect=True, steps=50), episode(1, perfect=False, steps=9000),
                      done(0), done(1)])
    slow_scores, slow_perfect, _, _, _ = collect_results(slow, 2, flag)
    fast_scores, fast_perfect, _, _, _ = collect_results(fast, 2, FakeFlag())
    assert slow_perfect == fast_perfect == [True, False]
    assert len(slow_scores) == len(fast_scores) == 2
