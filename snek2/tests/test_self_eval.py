"""The training self-eval: engine selection, seeding, and the shared metric fold.

Written alongside the 2026-08-27 change that moved the self-eval off the forked
`ParallelPyEnvironment` and onto the in-process vectorised engine, because the forked path was
measured at **88% of a training arm's wall clock**.

Three things here are worth more than the rest.

`test_the_vec_path_builds_no_eval_environments` is the fixture for the actual saving. The evaluator
could perfectly well have decided the engine internally, and the code would look the same and cost
the same 20 forked pygame processes per arm — the saving exists only because the decision is made
*before* construction. Nothing else in the repo would notice if that inverted.

`test_the_fork_seed_no_longer_depends_on_the_eval_episode_count` pins a coupling this change walked
straight into: the forking collector's RNG stream was `num_eval_episodes + 1`, so bumping the eval
count from 20 to 100 silently moved the *training* env's food sequence from stream 21 to 101. A diff
that changes training while claiming to change measurement is the failure this project has paid for
most often.

`test_fold_counts_perfect_games_off_the_score_not_the_reward` is the standing tripwire in its new
home. Both engines now feed one fold, which is the point — three separate counters comparing a final
reward with `PERFECT_GAME_REWARD` is how eight arms trained blind for 300k+ steps.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

import self_eval
import snake_constants
import training
import under_the_hood


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeMetrics:
    """Only the fields `fold_episode_sample` touches."""

    def __init__(self):
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        self.min_score = float('inf')
        self.max_score = float('-inf')
        self.last_eval_perfect_percent = None
        self.appended = []

    def append_perfect_percent(self, value):
        self.appended.append(value)


class _EngineEnv:
    """Sets SNEK_TRAIN_EVAL_ENGINE for a block and restores whatever was there."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get(self_eval.ENGINE_ENV)
        if self.value is None:
            os.environ.pop(self_eval.ENGINE_ENV, None)
        else:
            os.environ[self_eval.ENGINE_ENV] = self.value
        return self

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop(self_eval.ENGINE_ENV, None)
        else:
            os.environ[self_eval.ENGINE_ENV] = self.old
        return False


# ------------------------------------------------------------------ engine selection


def test_the_default_engine_is_vec():
    for value in (None, '', 'vec', 'VEC', '  vec  '):
        with _EngineEnv(value):
            assert self_eval.engine_name() == 'vec', 'value {0!r}'.format(value)


def test_scalar_is_selectable_and_garbage_raises():
    with _EngineEnv('scalar'):
        assert self_eval.engine_name() == 'scalar'
    for bad in ('vectorized', 'fast', '1', 'none'):
        with _EngineEnv(bad):
            try:
                self_eval.engine_name()
            except ValueError:
                continue
            raise AssertionError('{0!r} should not be a valid engine'.format(bad))


def test_only_the_scalar_path_wants_eval_environments():
    with _EngineEnv('vec'):
        assert self_eval.needs_eval_envs() is False
    with _EngineEnv('scalar'):
        assert self_eval.needs_eval_envs() is True


def test_scalar_without_its_environments_fails_loudly():
    """Rather than silently falling back to vec, which would hide a misconfiguration."""
    with _EngineEnv('scalar'):
        try:
            self_eval.build(policy=None, obs_len=30, seed=1, shaping_discount=0.99)
        except ValueError as exc:
            assert 'needs_eval_envs' in str(exc), str(exc)
            return
    raise AssertionError('scalar with no parallel environment must raise')


def test_describe_names_the_engine_and_the_episode_count():
    with _EngineEnv('vec'):
        text = self_eval.describe(100)
        assert 'vec' in text and '100' in text and 'no worker processes' in text, text
    with _EngineEnv('scalar'):
        text = self_eval.describe(100)
        assert 'scalar' in text and '100' in text, text


# ------------------------------------------------------------------ seeding


def _evaluator(seed=1):
    with _EngineEnv('vec'):
        return self_eval.build(policy=_ConstantPolicy(), obs_len=30, seed=seed,
                               shaping_discount=0.9975)


def test_each_eval_gets_a_different_board_sequence():
    """A fixed seed would replay one board set forever and change what the graph measures."""
    ev = _evaluator()
    seeds = [ev.eval_seed(step) for step in (1000, 2000, 3000, 4000)]
    assert len(set(seeds)) == 4, seeds


def test_the_eval_seed_is_reproducible_from_seed_and_step():
    assert _evaluator(seed=7).eval_seed(5000) == _evaluator(seed=7).eval_seed(5000)


def test_arms_of_one_wave_do_not_share_boards():
    """Sharing them would correlate the graph noise of seeds meant to be independent."""
    step = 12000
    seeds = {_evaluator(seed=s).eval_seed(step) for s in (1, 2, 3, 4)}
    assert len(seeds) == 4, seeds


def test_seeding_off_stays_off():
    assert _evaluator(seed=None).eval_seed(1000) is None


def test_the_fork_seed_no_longer_depends_on_the_eval_episode_count():
    """`derive_seed(seed, stream=num_eval_episodes + 1)` coupled training to the eval count.

    Bumping 20 -> 100 moved the forking collector from stream 21 to stream 101, changing the
    *training* env's food sequence as a side effect of a measurement change. Asserted against the
    source because the value is only reachable by launching a trainer.
    """
    source = open(os.path.join(REPO, 'snek2.py')).read()
    assert 'stream=num_eval_episodes + 1' not in source, (
        'the forking collector is seeded off the eval episode count again')
    assert 'stream=self_eval_mod.FORK_SEED_STREAM' in source, (
        'the forking collector should use the named FORK_SEED_STREAM')
    # And the named stream must not collide with the scalar path's per-worker eval streams,
    # which are 1..num_eval_episodes.
    assert self_eval.FORK_SEED_STREAM > training.num_eval_episodes, (
        'FORK_SEED_STREAM {0} collides with an eval worker stream at num_eval_episodes={1}'.format(
            self_eval.FORK_SEED_STREAM, training.num_eval_episodes))


# ------------------------------------------------------------------ the saving


def test_the_vec_path_builds_no_eval_environments():
    """The saving lives in `needs_eval_envs()` being consulted *before* construction.

    A check inside the evaluator would read the same and cost the same 20 forked pygame processes
    per arm, which were the 88%.
    """
    tree = ast.parse(open(os.path.join(REPO, 'snek2.py')).read())
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if 'needs_eval_envs' not in test_src:
            continue
        body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if 'ParallelPyEnvironment' in body_src:
            guarded.append(node)
    assert len(guarded) == 1, (
        'expected exactly one `if needs_eval_envs()` guarding the ParallelPyEnvironment '
        'construction, found {0}'.format(len(guarded)))


def test_nothing_constructs_the_parallel_eval_env_outside_that_guard():
    """Otherwise a second, ungated construction would quietly restore the cost."""
    source = open(os.path.join(REPO, 'snek2.py')).read()
    assert source.count('ParallelPyEnvironment(') == 1, (
        'more than one ParallelPyEnvironment construction in snek2.py')


# ------------------------------------------------------------------ the shared fold


def test_fold_counts_perfect_games_off_the_score_not_the_reward():
    """The standing tripwire, in its new shared home.

    A shaped arm pays `-c*Phi(s)` at the winning step, so a perfect game's reward is 99.9 rather
    than PERFECT_GAME_REWARD. Rewards here are deliberately nowhere near it.
    """
    metrics = FakeMetrics()
    perfect = float(snake_constants.MAX_POSSIBLE_SCORE)
    scores = [perfect, perfect, 40.0, 12.0]
    rewards = [99.9, 99.9, 30.0, -5.0]
    avg_reward, avg_score = under_the_hood.fold_episode_sample(
        rewards, scores, metrics, eval_only=False, num_episodes=4)
    assert metrics.last_eval_perfect_percent == 0.5, metrics.last_eval_perfect_percent
    assert abs(avg_score - (perfect * 2 + 52.0) / 4) < 1e-9
    assert abs(avg_reward - (99.9 * 2 + 25.0) / 4) < 1e-9
    assert metrics.max_score == perfect and metrics.min_score == 12.0
    assert metrics.max_reward == 99.9 and metrics.min_reward == -5.0


def test_fold_rejects_a_sample_of_the_wrong_length():
    """A short sample would otherwise divide by `num_episodes` and read low with nothing raised."""
    metrics = FakeMetrics()
    try:
        under_the_hood.fold_episode_sample([1.0, 2.0], [10.0, 20.0], metrics,
                                           eval_only=False, num_episodes=100)
    except ValueError as exc:
        assert '2 rewards' in str(exc), str(exc)
        return
    raise AssertionError('a 2-episode sample must not pass as 100 episodes')


def test_eval_only_records_the_perfect_percent_history():
    metrics = FakeMetrics()
    under_the_hood.fold_episode_sample([1.0], [float(snake_constants.MAX_POSSIBLE_SCORE)],
                                       metrics, eval_only=True, num_episodes=1)
    assert metrics.appended == [1.0], metrics.appended


# ------------------------------------------------------------------ end to end


class _ConstantPolicy:
    """A stand-in for `agent.policy`: always turns left. Enough to drive real vec episodes."""

    def action(self, time_step):
        rows = tf.shape(time_step.observation)[0]

        class _Step:
            pass

        out = _Step()
        out.action = tf.zeros([rows], dtype=tf.int32)
        return out


def test_the_vec_evaluator_runs_real_episodes_and_folds_them():
    """End to end through the real vectorised engine, with a trivial policy.

    Small (6 episodes) because the point is that the wiring works — the observation width matches,
    the `tf.function` accepts a variable batch, and `held` arrives in the shape the fold consumes.
    """
    from vectorized import config as vec_config

    ev = _evaluator(seed=3)
    metrics = FakeMetrics()
    avg_reward, avg_score = ev.run(metrics, eval_only=False, num_episodes=6, step=1000)

    assert metrics.last_eval_perfect_percent == 0.0, 'a left-turning policy should win nothing'
    assert np.isfinite(avg_reward) and np.isfinite(avg_score)
    assert 0 <= avg_score <= vec_config.MAX_POSSIBLE_SCORE, avg_score
    assert metrics.max_score >= metrics.min_score


def test_the_evaluator_compiles_its_policy_once():
    """Not per eval. 3000 evals on a 3M-step arm would mean 3000 traces of the policy."""
    ev = _evaluator(seed=4)
    metrics = FakeMetrics()
    ev.run(metrics, eval_only=False, num_episodes=4, step=1000)
    first = ev._act.experimental_get_tracing_count()
    ev.run(FakeMetrics(), eval_only=False, num_episodes=4, step=2000)
    assert ev._act.experimental_get_tracing_count() == first, (
        'the policy retraced between evals: {0} -> {1}'.format(
            first, ev._act.experimental_get_tracing_count()))


def test_the_lane_count_falling_does_not_retrace():
    """The reason the signature's batch dimension is None: lanes finish at different times."""
    ev = _evaluator(seed=5)
    ev._policy_fn(np.zeros((6, 30), dtype=np.float32))
    count = ev._act.experimental_get_tracing_count()
    for width in (5, 3, 1):
        ev._policy_fn(np.zeros((width, 30), dtype=np.float32))
    assert ev._act.experimental_get_tracing_count() == count, 'a narrower batch retraced'
