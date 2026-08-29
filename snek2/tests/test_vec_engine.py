"""Tests for `vectorized/vec_engine.py` — the batched measurement loop.

The env is covered by `test_vectorized_parity.py` and `test_vectorized_env.py`; what is left here is
the *accounting*, and it is the part where a bug produces a plausible number rather than a crash.
Three things can go wrong and all three are silent:

- **selection bias** — stopping at the first N completions favours short episodes, so a policy looks
  worse than it is, uniformly, and nothing in the output looks odd;
- **misattribution** — lanes migrate between checkpoints, so an episode can be banked against the
  wrong one, which reads as noise across a close-out rather than as an error;
- **capacity collapse** — a `max_live` too low for the width idles the batch, which is only a
  throughput bug but cost a 10x slowdown while looking like the design was simply slow.

The policies here are hand-written, so none of this needs TensorFlow.
"""

import numpy as np

from vectorized import config as C
from vectorized import vec_engine
from vectorized import vec_env as V


def survival_policy(obs):
    """Prefers a legal move, then a reachable tail, then closing on the food."""
    return np.argmax(obs[:, 6:9] * 100.0 + obs[:, 9:14:2] * 10.0 + obs[:, 0:6:2], axis=1)


def suicidal_policy(obs):
    """Always turns the same way, so it coils into itself within a handful of moves."""
    return np.zeros(obs.shape[0], dtype=np.int64)


def _run(jobs, episodes, width, max_live=None, seed=0):
    queue = list(jobs)
    out = {}
    stats = vec_engine.measure_stream(
        lambda: queue.pop(0) if queue else None,
        lambda key, held: out.__setitem__(key, held),
        episodes, width=width, max_live=max_live, seed=seed)
    return out, stats


# ------------------------------------------------------------------ episode counts

def test_every_checkpoint_gets_exactly_the_requested_episodes():
    """Exactly, not at least. An extra episode means a lane was restarted past its quota; a missing
    one means a finished episode was dropped, and both change a rate that nothing else flags."""
    jobs = [('c%d' % i, survival_policy) for i in range(5)]
    out, stats = _run(jobs, episodes=25, width=60, seed=1)
    assert sorted(out) == sorted(k for k, _ in jobs)
    for key, held in out.items():
        assert len(held['scores']) == 25, '{0} got {1}'.format(key, len(held['scores']))
        assert len(held['perfect']) == 25 and len(held['rewards']) == 25
        assert not held['abandoned'], 'the vec engine has no abandon gate'
    assert stats['checkpoints'] == 5
    assert stats['episodes'] == 125


def test_the_single_checkpoint_helper_agrees_with_the_stream():
    """`measure` is expressed through `measure_stream`, so this pins that it stays that way.

    Two builders of the same thing that drift apart is a failure this project has paid for more than
    once, and an engine used only by tests is exactly the copy that drifts.
    """
    held = vec_engine.measure(survival_policy, 40, lanes=20, seed=2)
    assert len(held['scores']) == 40
    assert len(held['rewards']) == 40
    assert held['seconds'] > 0


# ------------------------------------------------------------------- attribution

def test_each_checkpoint_gets_its_own_policys_episodes_not_its_neighbours():
    """**The lane-migration test.** A freed lane is handed to whichever checkpoint still needs
    episodes, so an off-by-one in the owner map would bank an episode against the wrong checkpoint.

    Two policies with grossly different outcomes make that visible: the suicidal one cannot score
    more than a couple of food, the survival one averages tens. Interleaving them means almost every
    lane changes owner mid-run, so a mix-up shows up as a suicidal checkpoint with a good score.
    """
    jobs = []
    for i in range(6):
        jobs.append(('good%d' % i, survival_policy) if i % 2 == 0
                    else ('bad%d' % i, suicidal_policy))
    out, _ = _run(jobs, episodes=30, width=70, seed=3)
    for key, held in out.items():
        mean = float(np.mean(held['scores']))
        if key.startswith('bad'):
            assert mean < 5, '{0} (suicidal) scored {1:.1f} — episodes attributed wrongly'.format(
                key, mean)
        else:
            assert mean > 10, '{0} (survival) scored {1:.1f} — episodes attributed wrongly'.format(
                key, mean)


def test_a_perfect_flag_is_the_score_not_the_reward():
    """Every recorded flag must equal `score == MAX_POSSIBLE_SCORE`, with no reward anywhere in it.

    Three counters in this project compared a final reward with `PERFECT_GAME_REWARD`, and the moment
    a shaping term shipped a perfect game paid 99.9 instead of 100, every counter read 0%, and eight
    arms trained handicapped because the exploration schedule reads that number.
    """
    out, _ = _run([('c', survival_policy)], episodes=60, width=30, seed=4)
    held = out['c']
    for score, flag in zip(held['scores'], held['perfect']):
        assert bool(flag) == (score == C.MAX_POSSIBLE_SCORE), (score, flag)


# ---------------------------------------------------------------- selection bias

def test_the_sample_is_not_biased_toward_short_episodes():
    """**The statistical correctness test.**

    Running N lanes and taking the first N completions is biased: a lane that dies early finishes
    early, so the survivors are the good episodes and the sample is short-episode-heavy. The engine
    instead *starts* exactly `episodes` episodes and runs every one to completion.

    A bias would depend on how much migration happens, so this compares a run with almost none
    (width equal to the quota, so every episode starts at once) against one with a great deal
    (a narrow batch cycling many times). Under correct accounting these are the same distribution;
    under the biased rule the narrow run would read systematically better, because it is the one
    that keeps replacing finished episodes.
    """
    wide, _ = _run([('w', survival_policy)], episodes=300, width=300, seed=5)
    narrow, _ = _run([('n', survival_policy)], episodes=300, width=25, seed=6)
    mw = float(np.mean(wide['w']['scores']))
    mn = float(np.mean(narrow['n']['scores']))
    spread = float(np.std(wide['w']['scores'] + narrow['n']['scores']))
    # Two independent 300-episode means of the same distribution differ by ~sqrt(2/300) sd; 4x that
    # is a generous bound that still rejects the biased rule, which shifts the mean by a large
    # fraction of an sd rather than a few percent of one.
    tolerance = 4.0 * spread * (2.0 / 300) ** 0.5
    assert abs(mw - mn) < tolerance, (
        'mean score {0:.2f} at width 300 vs {1:.2f} at width 25, tolerance {2:.2f} — the sample '
        'depends on how often lanes recycle, which is what a length-selected sample looks like'
        .format(mw, mn, tolerance))


def test_rewards_are_the_undiscounted_episode_return():
    """`avg_reward` must mean what `under_the_hood.run_parallel_eval_episodes` means by it.

    That accumulates each step's reward over the episode with no discounting, so a single-lane run
    driven by hand must reproduce the engine's number exactly. Checked against the env directly
    rather than against a second copy of the loop.
    """
    out, _ = _run([('c', suicidal_policy)], episodes=8, width=1, seed=7)
    engine_rewards = out['c']['rewards']

    vec = V.VecSnake(1, seed=7)
    replay = []
    total = 0.0
    while len(replay) < 8:
        obs = vec.observe()
        _, reward, done, _ = vec.step(suicidal_policy(obs), autoreset=False, observe=False)
        total += float(reward[0])
        if done[0]:
            replay.append(total)
            total = 0.0
            vec.reset_rows([0])
    assert np.allclose(engine_rewards, replay), (engine_rewards, replay)


# -------------------------------------------------------------------- capacity

def test_max_live_is_derived_with_enough_slack_to_keep_the_batch_full():
    for width, episodes in ((1024, 100), (1024, 500), (4000, 500), (512, 20)):
        live = vec_engine.default_max_live(width, episodes)
        assert live * episodes > width, (width, episodes, live)
        assert live >= vec_engine.MIN_MAX_LIVE


def test_a_capacity_that_would_collapse_utilisation_is_refused():
    """The exact configuration that measured 4% utilisation must raise rather than run slowly.

    A throughput bug that looks like "this design is just slow" is worse than a crash: it was read as
    the approach failing, not as one number being wrong.
    """
    try:
        _run([('c', survival_policy)], episodes=100, width=1200, max_live=12)
    except ValueError as error:
        assert 'utilisation' in str(error)
        return
    raise AssertionError('max_live * episodes == width was accepted')


def test_a_generous_max_live_keeps_utilisation_high():
    """The positive half — the guard could be satisfied and the batch still drain."""
    jobs = [('c%d' % i, survival_policy) for i in range(24)]
    _, stats = _run(jobs, episodes=20, width=100, seed=8)
    assert stats['utilisation'] > 0.5, 'utilisation was {0:.0%}'.format(stats['utilisation'])


def test_an_empty_stream_is_not_an_error():
    """Most HOF selections are empty, and that has to exit cleanly rather than raise."""
    out, stats = _run([], episodes=10, width=10)
    assert out == {}
    assert stats['checkpoints'] == 0


# --------------------------------------------------------------------------- start-order recording

def test_a_job_banks_each_episode_at_its_start_slot_not_its_completion_order():
    """The ordering guarantee `eval_plan.equal_effort_pooled` depends on, asserted directly.

    Banking out of order is the normal case, not an edge case: lanes finish whenever their episode
    ends, so slot 3 routinely lands before slot 0. An append-based implementation passes every other
    test in this file — the row's totals are identical — and fails only here.
    """
    job = vec_engine._Job('ckpt', lambda obs: obs, 4)
    job.started = 4                                  # as if all four lanes were assigned
    job.record(2, 95, 95.0)                          # completions arrive scrambled
    job.record(0, 40, 39.5)
    job.record(3, 95, 95.0)
    job.record(1, 95, 95.0)
    assert job.scores == [40, 95, 95, 95], job.scores
    assert job.perfect == [0, 1, 1, 1], job.perfect
    assert job.rewards == [39.5, 95.0, 95.0, 95.0], job.rewards
    assert job.banked == 4 and job.done == 4


def test_banking_one_slot_twice_is_refused():
    job = vec_engine._Job('ckpt', lambda obs: obs, 2)
    job.record(0, 95, 95.0)
    try:
        job.record(0, 40, 39.0)
    except RuntimeError as error:
        assert 'twice' in str(error)
    else:
        raise AssertionError('a double-banked slot must raise')


def test_a_partial_job_refuses_to_hand_out_a_sample():
    """A gap would reach the result file as a JSON null, which no reader checks for."""
    job = vec_engine._Job('ckpt', lambda obs: obs, 3)
    job.record(0, 95, 95.0)
    try:
        job.held()
    except RuntimeError as error:
        assert 'banked' in str(error)
    else:
        raise AssertionError('held() must refuse a sample with an unfilled slot')


def test_no_slot_is_left_unfilled_by_a_real_run():
    results, _ = _run([('c%d' % i, survival_policy) for i in range(3)],
                      episodes=24, width=64, max_live=8)
    for key, held in results.items():
        assert len(held['scores']) == 24, (key, len(held['scores']))
        assert all(s is not None for s in held['scores']), key
        assert all(p is not None for p in held['perfect']), key
        assert all(r is not None for r in held['rewards']), key


def test_a_prefix_of_a_row_is_a_fair_sample_of_it():
    """The statistical property, on the real env: the first half must look like the second half.

    This is the test that would have caught the completion-ordered bug in production. Episode length
    correlates with outcome -- a high-scoring episode ate more food and ran longer, a quick death
    ends in a few dozen steps -- so under completion ordering the outcomes arrive sorted by length
    and a prefix is a biased sample. Measured on a real arm, failures landed at mean position 0.92
    of a completion-ordered array and a 20-of-100 prefix read 0.25% failures against a true 2.23%.

    **Measured on mean score, not on the perfect-game count, because the first version of this test
    was vacuous.** `survival_policy` is a heuristic that never wins: over 360 episodes it scored a
    mean of 42 with **zero** perfect games, so comparing perfect counts compared 0 against 0 and
    passed under the very mutant it existed to catch. Score is the statistic that actually varies
    here, and the guard below fails if that ever stops being true.
    """
    episodes = 60
    results, _ = _run([('c%d' % i, survival_policy) for i in range(6)],
                      episodes=episodes, width=128, max_live=16)
    assert results, 'no checkpoints measured'
    half = episodes // 2
    first = [s for held in results.values() for s in held['scores'][:half]]
    second = [s for held in results.values() for s in held['scores'][half:]]

    # Vacuity guard: the comparison below is only meaningful while scores have real spread. If a
    # future policy makes every episode identical this fails loudly instead of passing for free.
    allsc = first + second
    spread = max(allsc) - min(allsc)
    assert spread >= 20, 'scores span only {0} — this test can no longer detect sorting'.format(spread)

    mean_first = sum(first) / float(len(first))
    mean_second = sum(second) / float(len(second))
    # Loose on purpose: the point is to catch *sorting*, a gross effect, not to assay a mean. Under
    # completion ordering this gap ran to ~30 points; ordinary sampling noise here is a few points.
    assert abs(mean_first - mean_second) <= 0.20 * spread, (
        'first half of each row means {0:.1f} and the second half {1:.1f} over a spread of {2} — the '
        'episodes are sorted by length, so a prefix of this row is not a fair sample of '
        'it'.format(mean_first, mean_second, spread))
