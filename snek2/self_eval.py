"""The training self-eval, on the vectorised engine or the original forked one.

**Why this module exists: the self-eval was 88% of a training arm's wall clock.** Measured
2026-08-26 on two live `b46b` arms by sampling each trainer's forked children in `/proc` at 2 Hz —
eval-active wall clock **88.8%** and **88.0%**, with bursts of median 20-23 s against a 24 s
eval cycle. So a 3M-step arm spent roughly **18.5 h measuring and 2.3 h learning**. The cause was
contention rather than episode cost: four arms x 20 forked pygame envs is 80 processes on 8 physical
cores, about 9.9 cores of demand on 8, leaving each eval worker 0.09 of a core.

The vectorised engine that replaced `eval_checkpoints.py` for *checkpoint* measurement on 2026-08-24
never touched this path. It is the same engine, called in-process here through
`vec_engine.measure`, and it needs no worker processes at all: one batched numpy env `num_episodes`
lanes wide, stepped in the trainer's own process.

**Two things make this a training change and not only a speedup, both accepted deliberately
(2026-08-27, at the user's direction).**

- `metrics.last_eval_perfect_percent` feeds `training.epsilon_for`'s refinement phase, so anything
  that changes the perfect rate changes the *exploration schedule*. This is the same coupling that
  made the chase-safe shaping bug pin epsilon at 0.0125 for 300k+ steps across eight arms.
- `num_eval_episodes` went 20 -> 100 in the same change, which is a measurement boundary like the
  10 -> 20 change of 2026-08-19. `strong_eval_fraction` is a threshold-crossing statistic and is
  biased by episode count, so cross-boundary comparisons must go through
  `hyperparamTuning/perDiagnostics/sef_common_footing.py` rather than being read raw.

**The greedy-policy question is settled and does not apply here.** An earlier note worried that the
engines' -0.058 pp agreement was validated on greedy policies at fixed checkpoints while a self-eval
runs mid-training. Only the second half is true: `training` calls this with `agent.policy`, which is
the *greedy* policy — `agent.collect_policy` is the epsilon-greedy one and is never evaluated. So the
validated regime is the regime, and the only untested difference is that the weights are live rather
than restored, which no part of the engine can see.

`SNEK_TRAIN_EVAL_ENGINE=scalar` restores the forked path, for the same reason `SNEK_EVAL_ENGINE`
exists: a regression here has to be answerable without a deploy. It is deliberately *not* the same
variable as `SNEK_EVAL_ENGINE`, so the desktop's `runtime.json` cannot reach training by accident.
"""

import os

import tensorflow as tf
from tf_agents.trajectories import time_step as ts

import under_the_hood
from vectorized import vec_engine


ENGINE_ENV = 'SNEK_TRAIN_EVAL_ENGINE'
ENGINES = ('vec', 'scalar')

# The forking collector's RNG stream, and the reason it is a named constant is a trap this change
# walked into. It used to be `derive_seed(seed, stream=num_eval_episodes + 1)`, which silently
# couples the *training* env's food sequence to the *eval* episode count — bumping 20 -> 100 moved
# the collector from stream 21 to stream 101 and would have changed training with no diff saying so.
# Well clear of the scalar path's per-worker eval streams, which are 1..num_eval_episodes.
FORK_SEED_STREAM = 1001

# Per-eval seed stream. The vec env's RNG is an isolated `np.random.default_rng`, so this cannot
# collide with the global `random` module that `Snake.Food` draws from; it only has to be distinct
# from itself across evals, which the step mixed in below provides.
EVAL_SEED_STREAM = 2001


def engine_name():
    """The configured engine, validated. Unset means `vec`."""
    name = (os.environ.get(ENGINE_ENV) or 'vec').strip().lower()
    if name not in ENGINES:
        raise ValueError('{0}={1!r}: expected one of {2}'.format(
            ENGINE_ENV, os.environ.get(ENGINE_ENV), ', '.join(ENGINES)))
    return name


class VecSelfEval:
    """Self-eval through the in-process vectorised engine. No worker processes at all.

    The `tf.function` is built **once**, in `__init__`, with a batch dimension of `None`. That is
    load-bearing for exactly the reason `vec_eval.AgentPool` gives: the lane count falls as episodes
    finish, so a signature fixed to one width would retrace the policy on almost every step and
    hand back the per-call overhead this engine exists to escape. Building it per eval would be
    worse still — 3000 traces over a 3M-step arm.

    It closes over the live `agent.policy`, so it reads current weights with no restore. There is no
    checkpoint in this path and nothing to keep in sync.
    """

    def __init__(self, policy, obs_len, seed, shaping_discount):
        self._seed = seed
        self._shaping_discount = float(shaping_discount)

        @tf.function(input_signature=[tf.TensorSpec([None, int(obs_len)], tf.float32)])
        def act(observation):
            rows = tf.shape(observation)[0]
            step = ts.TimeStep(
                step_type=tf.fill([rows], tf.constant(ts.StepType.MID, dtype=tf.int32)),
                reward=tf.zeros([rows], dtype=tf.float32),
                discount=tf.ones([rows], dtype=tf.float32),
                observation=observation)
            return policy.action(step).action

        self._act = act

    def eval_seed(self, step):
        """A fresh, reproducible board sequence per eval.

        **Fresh matters.** A fixed seed would replay the same `num_episodes` games at every eval,
        turning the graph from a sample of the policy into a score on one board set — lower variance,
        but a different statistic, and one that rewards memorising those boards. The forked path it
        replaces advanced each worker's RNG across evals, so fresh per eval is what preserves the
        meaning of the graph.

        Mixing the step in keeps it reproducible from `(SNEK_SEED, step)`, and deriving through
        `derive_seed` keeps arms of one wave on separate streams — sharing boards across the wave's
        four seeds would correlate their graph noise.
        """
        base = under_the_hood.derive_seed(self._seed, stream=EVAL_SEED_STREAM)
        if base is None:
            return None
        return (base * 1000003 + int(step)) % (2 ** 31 - 1)

    def run(self, metrics, eval_only, num_episodes, step):
        held = vec_engine.measure(self._policy_fn, num_episodes,
                                  seed=self.eval_seed(step) or 0,
                                  shaping_discount=self._shaping_discount)
        # `held['perfect']` is deliberately ignored: `fold_episode_sample` recounts off the score
        # through `state_helpers.is_perfect_score`, which is the project's single definition of a
        # perfect game. Two counters is how the last one went silently wrong.
        return under_the_hood.fold_episode_sample(
            list(held['rewards']), list(held['scores']), metrics, eval_only, num_episodes)

    def _policy_fn(self, observation):
        return self._act(tf.convert_to_tensor(observation)).numpy()


class ScalarSelfEval:
    """Self-eval through the original `ParallelPyEnvironment` of forked pygame envs.

    Kept only as the `SNEK_TRAIN_EVAL_ENGINE=scalar` opt-out. Its 20-to-100 worker processes are
    built by the caller, so selecting `vec` means they are never created — which is most of the
    point, since those processes were the 88%.
    """

    def __init__(self, parallel_environment, policy):
        self._env = parallel_environment
        self._policy = policy

    def run(self, metrics, eval_only, num_episodes, step):
        del step                      # the forked workers carry their own advancing RNG state
        return under_the_hood.compute_avg_return(
            self._env, self._policy, metrics, eval_only, num_episodes)


def needs_eval_envs():
    """Whether the caller has to build the forked eval environments at all.

    Separate from `build` because the environments are expensive enough that the decision has to be
    made *before* they are constructed — 20 forked pygame processes per arm, and on the vec path
    they would sit idle for the whole run.
    """
    return engine_name() == 'scalar'


def build(policy, obs_len, seed, shaping_discount, parallel_environment=None):
    """The configured evaluator. `parallel_environment` is required only on the scalar path."""
    name = engine_name()
    if name == 'scalar':
        if parallel_environment is None:
            raise ValueError(
                '{0}=scalar needs the forked eval environments; check needs_eval_envs() before '
                'building the evaluator'.format(ENGINE_ENV))
        return ScalarSelfEval(parallel_environment, policy)
    return VecSelfEval(policy, obs_len, seed, shaping_discount)


def describe(num_episodes):
    """One line for the startup banner, so a log says which engine measured the graph."""
    if engine_name() == 'scalar':
        return 'scalar, {0} forked pygame envs'.format(num_episodes)
    return 'vec, {0} lanes in-process, no worker processes'.format(num_episodes)
