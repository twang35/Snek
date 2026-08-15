"""C51 — the categorical (distributional) DQN agent, and the support arithmetic that sizes it.

Design, measurements and pre-registered criteria: [`plans/distributional-c51.md`](plans/distributional-c51.md).

A scalar agent predicts `Q(s, a)`. This one predicts a *distribution* over the return on a fixed grid
of `num_atoms` atoms spanning `[v_min, v_max]`, and trains it by cross-entropy against the projected
Bellman target. `tf_agents` ships the machinery — `CategoricalDqnAgent`, `CategoricalQNetwork`,
`CategoricalQPolicy`, `project_distribution` — so this module is a subclass plus the two things the
upstream agent does not give us:

| upstream defect | why it matters here | fix |
|---|---|---|
| `_loss` accepts `weights` and **silently drops them** | `SNEK_IS_WEIGHTS` is a live knob and PER is on in every arm, so a whole feature would quietly do nothing | `sample_weight=weights` into `aggregate_losses` |
| `_loss` returns `DqnLossInfo(td_loss=(), td_error=())` — a `TODO(b/127318640)` in the source | `training.py` does `signal.numpy()` on that field, so PER has no priority signal and would raise on an empty tuple | return a per-example signal in **both** fields |

Both were verified against tf_agents 0.18.0 before this was written: skewing the weights 32x moved this
loss and left upstream's bit-identical, and upstream's `extra` really is a pair of empty tuples.

**The loss is reimplemented rather than vendored, and `test_categorical_agent.py` pins it to upstream's
number.** With double selection off and no weights, `_loss` must equal
`CategoricalDqnAgent._loss` on the same batch — measured 4.706757 both ways. That test is the tripwire
for a tf_agents upgrade changing the projection under us, and it is a stronger guarantee than a copied
block, which drifts without saying so.

**Three deliberate narrowings, each a loud failure rather than an untested code path.** The upstream
loss carries branches for `n_step_update > 1`, for RNN state (`batch_squash`), and for an
observation/action-constraint splitter. No arm in this project uses any of them — n=3 measured null in
batch 15 and n=5 is closed — so rather than reimplement three paths nothing exercises, the constructor
refuses them. The n-step target support is ~10 more lines if a batch ever wants it.

**Priority is the KL, not the cross-entropy** (`priority_signal='kl'`, the default). The two differ by
`H(target)`, which is never 0 because the projection spreads mass over two atoms, so cross-entropy
carries an irreducible per-transition floor. Left in, that floor compresses the relative spread PER's
`alpha` exponent acts on — the same defect this repo already documented for Huber `td_loss`, whose
log-log slope against `|delta|` measures 1.92-1.99. **One thing is worse here than in the Huber case:**
`td_loss` and `td_error` are monotone in each other (top-1000 Jaccard 1.0000 on 8 of 8 arms), so that
choice only ever changed how much mass the top got. CE and KL differ by a quantity that is *not*
monotone in KL, so they genuinely reorder transitions. `priority_signal='ce'` is the ablation.

The signal goes into **both** `td_loss` and `td_error`, which is what lets `training.py` stay untouched:
it reads one or the other according to `SNEK_PRIORITY_SIGNAL` and gets the same tensor either way.
"""
import math

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import snake_constants

# Name of this algorithm in `arch.json` and in `SNEK_ALGO`. The other value is DDQN_ALGO below, and
# `policy_arch` reads a *missing* field as that one, so every pre-C51 policy directory keeps loading.
C51_ALGO = 'c51'
DDQN_ALGO = 'ddqn'
ALGOS = (DDQN_ALGO, C51_ALGO)

# Priority signals a c51 arm can use. `SNEK_PRIORITY_SIGNAL`'s scalar values (`td_error`, `td_loss`)
# both map to `kl` — there is no TD error in a distributional agent, so the knob's two settings would
# otherwise name a quantity that does not exist. See `priority_signal_for` below.
PRIORITY_KL = 'kl'
PRIORITY_CE = 'ce'
PRIORITY_SIGNALS = (PRIORITY_KL, PRIORITY_CE)

# The largest return the *measurement* found: 104.38, over 3 checkpoints and 60 greedy episodes
# (perDiagnostics/return_distribution.py, 2026-08-15). Rounded up. A `v_max` below this clips returns
# that real policies demonstrably reach, which is a mistake rather than a trade-off, so it is a hard
# failure. Above it and below the derived bound is a judgement — see `check_support`.
MEASURED_MAX_RETURN = 105.0


def theoretical_max_return():
    """The largest return the reward function can pay, for any `gamma <= 1`.

    A perfect game pays `FOOD_REWARD` for the first `MAX_POSSIBLE_SCORE - 1` foods and then
    `PERFECT_GAME_REWARD` for the last one — **not both**, because `Snake.step` *overwrites*
    `reward = FOOD_REWARD` with the win (`Snake.py`, the `check_perfect_game()` branch). So the bound is
    `94 * 1 + 100 = 194` on the shipped constants, attained from the opening state in the limiting case
    where every food spawns one step from the head.

    Derived rather than written down, so it moves when a reward does. The first draft of the plan
    hard-coded `PERFECT_GAME_REWARD + FOOD_REWARD` (101) and that was **too low**: a near-win state still
    collects food on the way, which is why the measured max is 104.38 rather than ~100.
    """
    foods = snake_constants.MAX_POSSIBLE_SCORE - 1
    return foods * snake_constants.FOOD_REWARD + snake_constants.PERFECT_GAME_REWARD


def min_possible_return():
    """The most negative return reachable, which is one step's worth and nothing more.

    `Snake.step` **assigns** each outcome reward rather than accumulating, so death, starve, food and
    the win are mutually exclusive — a death pays exactly `DEATH_REWARD`, never
    `DEATH_REWARD + STARVE_REWARD`. Any earlier death is discounted *toward* zero, so a single-step
    death is the floor.

    The one additive term that can land on a terminal step is the chase-safe shaping, which pays
    `-c * Phi(s)` there with `Phi` in [0, 1]; `FOOD_DISTANCE_REWARD` is skipped whenever the episode
    ends. So the floor is `DEATH_REWARD - CHASE_SAFE_SHAPING`, which is exactly `DEATH_REWARD` for any
    arm with shaping off.
    """
    return snake_constants.DEATH_REWARD - abs(snake_constants.CHASE_SAFE_SHAPING)


def check_support(v_min, v_max, num_atoms, allow_clipping=False):
    """Validates a support grid. Returns a list of warning strings; raises on the hard failures.

    Three levels, because a single "must cover the theoretical max" rule would refuse to start the
    configuration this feature was built for (`v_max = 120`, deliberately below the derived 194):

    | condition | level |
    |---|---|
    | `v_min` above the reachable minimum | hard failure — costs nothing to satisfy, and a clipped death value is a wrong terminal target |
    | `v_max` below `MEASURED_MAX_RETURN` | hard failure — clipping returns real policies reach is a mistake, not a trade |
    | `v_max` below `theoretical_max_return()` | **warning**, carried into `run_config` so `runs/<policy>.md` records the judgement |

    `allow_clipping` overrides **only the two hard failures**. It deliberately does not suppress the
    warning: the warning is the record of a choice, and an escape hatch that erased it would leave no
    trace of the decision in the run report.
    """
    if num_atoms < 2:
        raise ValueError('SNEK_NUM_ATOMS={0} must be at least 2'.format(num_atoms))
    if v_max <= v_min:
        raise ValueError('SNEK_V_MAX={0} must be above SNEK_V_MIN={1}'.format(v_max, v_min))

    floor, ceiling = min_possible_return(), theoretical_max_return()
    problems = []
    if v_min > floor:
        problems.append(
            'SNEK_V_MIN={0} is above the reachable minimum return {1} (DEATH_REWARD {2} minus '
            'CHASE_SAFE_SHAPING {3}), so a death would be clipped'.format(
                v_min, floor, snake_constants.DEATH_REWARD, snake_constants.CHASE_SAFE_SHAPING))
    if v_max < MEASURED_MAX_RETURN:
        problems.append(
            'SNEK_V_MAX={0} is below the measured maximum return {1}, which real policies reach '
            '(3 checkpoints, 60 episodes)'.format(v_max, MEASURED_MAX_RETURN))
    if problems and not allow_clipping:
        raise SystemExit(
            'refusing to start: {0}. Set SNEK_C51_ALLOW_CLIPPING=1 to override.'
            .format('; and '.join(problems)))

    warnings = []
    if v_max < ceiling:
        spacing = (v_max - v_min) / (num_atoms - 1)
        warnings.append(
            'support [{0}, {1}] is below the derived maximum return {2}, so a return above {1} would '
            'be clipped. Measured max is {3} ({4:.0f}% headroom); spacing {5:.3f}. This is a '
            'judgement, not an error.'.format(v_min, v_max, ceiling, MEASURED_MAX_RETURN,
                                              100.0 * (v_max - MEASURED_MAX_RETURN)
                                              / MEASURED_MAX_RETURN, spacing))
    return warnings


def priority_signal_for(configured):
    """Maps `SNEK_PRIORITY_SIGNAL` onto a signal this agent can actually produce.

    A distributional agent has no TD error, so `td_error` and `td_loss` — the two values the scalar
    arms use — both resolve to the KL. Mapping rather than rejecting keeps a c51 arm launchable with a
    control's environment block copied verbatim, which is how every batch in this project is launched;
    rejecting would make the *control's own config* a startup failure.
    """
    if configured in (PRIORITY_CE,):
        return PRIORITY_CE
    if configured in (PRIORITY_KL, 'td_error', 'td_loss'):
        return PRIORITY_KL
    raise ValueError(
        'SNEK_PRIORITY_SIGNAL={0!r} is not usable with SNEK_ALGO=c51. Use kl (or td_error/td_loss, '
        'which map to it) or ce.'.format(configured))


def zero_init_lambda(v_min, v_max, num_atoms, tolerance=1e-9):
    """The decay rate for a bias ramp that makes the initial expected Q zero.

    **A categorical head's initial expected Q is the grid's midpoint**, because the atom logits start
    near-uniform — 57.5 for `[-5, 120]`, against a scalar head that starts at Q ~ 0 (its final layer is
    `RandomUniform(+/-0.03)`). That is a *second* difference between a c51 arm and its ddqn control, on
    top of the algorithm, and its size scales with `v_max`. This solves for the `lambda` in
    `bias_i = -lambda * (z_i - v_min)` that puts the initial expectation back at 0.

    Bisected rather than written down, because a pasted constant would go stale the moment the grid
    changes — and it barely depends on `v_max` (the mass concentrates in the bottom few atoms either
    way), which is exactly the kind of near-invariance that makes a stale constant look correct.

    Raises if the grid does not straddle zero, since no amount of downweighting reaches 0 from a
    strictly positive support.
    """
    if not v_min < 0.0 < v_max:
        raise ValueError(
            'zero-init needs a support straddling 0, got [{0}, {1}]'.format(v_min, v_max))
    support = [v_min + i * (v_max - v_min) / (num_atoms - 1) for i in range(num_atoms)]

    def expected_q(rate):
        weights = [math.exp(-rate * (z - v_min)) for z in support]
        total = sum(weights)
        return sum(z * w for z, w in zip(support, weights)) / total

    low, high = 0.0, 1.0
    while expected_q(high) > 0.0:
        high *= 2.0
        if high > 1e6:  # unreachable for any sane grid; a runaway loop here would be silent
            raise ValueError('no bias ramp reaches zero expected Q on this support')
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if expected_q(middle) > 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


class SnekCategoricalDqnAgent(categorical_dqn_agent.CategoricalDqnAgent):
    """`CategoricalDqnAgent` with importance-sampling weights honoured, a PER priority signal, and
    optional double (online-argmax) atom selection.

    `double=True` is the default because the control is `DdqnAgent`, which selects its target action
    with the online network. Upstream's C51 selects with the *target* network, so taking it as-is
    would change two things at once; the knob keeps that measurable instead of baked in.
    """

    def __init__(self, *args, **kwargs):
        double = kwargs.pop('double', True)
        priority_signal = kwargs.pop('priority_signal', PRIORITY_KL)
        if priority_signal not in PRIORITY_SIGNALS:
            raise ValueError('priority_signal must be one of {0}, got {1!r}'
                             .format(PRIORITY_SIGNALS, priority_signal))
        if kwargs.get('n_step_update', 1) != 1:
            raise ValueError(
                'SNEK_ALGO=c51 supports n_step_update=1 only. The n-step target support is ~10 lines '
                'and no arm since batch 15 uses n>1 (n=3 measured null, n=5 closed), so it is left '
                'unwritten rather than shipped untested.')
        if kwargs.get('observation_and_action_constraint_splitter') is not None:
            raise ValueError('SnekCategoricalDqnAgent does not implement the observation/action '
                             'constraint splitter; nothing in this project uses one.')
        super(SnekCategoricalDqnAgent, self).__init__(*args, **kwargs)
        if self._q_network.state_spec not in ((), None):
            raise ValueError('SnekCategoricalDqnAgent does not implement the RNN batch-squash path; '
                             'this project uses feed-forward networks only.')
        # Assigned after super().__init__ on purpose: these are read only inside `_loss`, never during
        # construction, and setting attributes on a tf.Module before its __init__ has run is its own
        # class of obscure failure.
        self._double = bool(double)
        self._priority_signal = priority_signal

    @property
    def double(self):
        """Exposed for tests and the run report; nothing in the training loop reads it."""
        return self._double

    @property
    def priority_signal(self):
        return self._priority_signal

    def _next_q_distribution(self, next_time_steps):
        """The target atom probabilities for the next state, under the selected action.

        Upstream picks that action with the **target** network. With `double=True` the atoms still come
        from the target network but the action comes from the **online** one, which is what
        `DdqnAgent` — the control — does, and what Rainbow does.

        **A test for this must desync the target network first.** `initialize()` copies the online
        weights into the target, so immediately after construction both selections agree exactly and a
        fixture that skips the desync passes whichever branch is live.
        """
        if not self._double:
            return super(SnekCategoricalDqnAgent, self)._next_q_distribution(next_time_steps)
        observation = next_time_steps.observation
        target_logits, _ = self._target_q_network(
            observation, step_type=next_time_steps.step_type, training=False)
        target_probabilities = tf.nn.softmax(target_logits)
        online_logits, _ = self._q_network(
            observation, step_type=next_time_steps.step_type, training=False)
        online_q = tf.reduce_sum(self._support * tf.nn.softmax(online_logits), axis=-1)
        greedy = tf.argmax(online_q, axis=-1, output_type=tf.int32)[:, None]
        rows = tf.range(tf.shape(target_probabilities)[0])[:, None]
        return tf.gather_nd(target_probabilities, tf.concat([rows, greedy], axis=-1))

    def target_distribution(self, next_time_steps, gamma=1.0, reward_scale_factor=1.0):
        """The projected Bellman target, `[batch, num_atoms]`.

        Split out of `_loss` so the terminal contract is testable on its own. The whole of it is in the
        one line `target_support = reward + gamma * discount * tiled_support`: the environment's
        per-step `discount` is **0 on a terminal step** (`snake_environment.to_tensor_time_step`, the
        only thing that stops the bootstrap), so at d=0 the support collapses to the reward and the
        projection returns a point mass there. A `1 - done` spelling, or dropping `discount`, would
        keep training terminal states against a bootstrapped tail — silently, since the loss stays
        finite and the numbers stay plausible.
        """
        next_q_distribution = self._next_q_distribution(next_time_steps)
        batch_size = next_q_distribution.shape[0] or tf.shape(next_q_distribution)[0]
        tiled_support = tf.reshape(tf.tile(self._support, [batch_size]),
                                  [batch_size, self._num_atoms])
        discount = tf.expand_dims(next_time_steps.discount, -1)
        reward = tf.expand_dims(next_time_steps.reward, -1) * reward_scale_factor
        target_support = reward + gamma * discount * tiled_support
        return tf.stop_gradient(categorical_dqn_agent.project_distribution(
            target_support, next_q_distribution, self._support))

    def _loss(self, experience, td_errors_loss_fn=None, gamma=1.0, reward_scale_factor=1.0,
              weights=None, training=False):
        """Cross-entropy against the projected Bellman target, with IS weights and a per-example signal.

        `td_errors_loss_fn` is accepted and unused — there is no TD error to apply it to. It stays in
        the signature because `DqnAgent._train` passes it positionally by keyword.
        """
        time_steps, policy_steps, next_time_steps = trajectory.experience_to_transitions(
            experience, squeeze_time_dim=True)
        actions = policy_steps.action
        if actions.shape.rank > 1:
            actions = tf.squeeze(actions, list(range(1, actions.shape.rank)))

        with tf.name_scope('critic_loss'):
            q_logits, _ = self._q_network(
                time_steps.observation, step_type=time_steps.step_type, training=training)
            target_distribution = self.target_distribution(
                next_time_steps, gamma=gamma, reward_scale_factor=reward_scale_factor)

            batch_size = q_logits.shape[0] or tf.shape(q_logits)[0]
            row = tf.range(batch_size, dtype=actions.dtype)
            chosen_action_logits = tf.gather_nd(q_logits, tf.stack([row, actions], axis=-1))

            # The gradient term, and deliberately the same call upstream makes, so the
            # equal-to-upstream test compares like with like rather than two spellings of a formula.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=target_distribution, logits=chosen_action_logits)
            if self._priority_signal == PRIORITY_KL:
                # KL = CE - H(target). xlogy is 0 where the target is, which it is for all but the two
                # atoms the projection lands on. Floored at 0 because the subtraction of two ~equal
                # floats can go a few ulps negative, and a negative priority is meaningless.
                entropy = -tf.reduce_sum(
                    tf.math.xlogy(target_distribution, target_distribution), axis=-1)
                per_example_signal = tf.maximum(cross_entropy - entropy, 0.0)
            else:
                per_example_signal = cross_entropy

            # `sample_weight=weights` is the whole reason this method is overridden. Upstream omits it,
            # so every importance-sampling weight this project computes was being discarded.
            aggregated = common.aggregate_losses(
                per_example_loss=cross_entropy,
                sample_weight=weights,
                regularization_loss=self._q_network.losses)
            total_loss = aggregated.total_loss

            common.summarize_scalar_dict(
                {'critic_loss': aggregated.weighted,
                 'reg_loss': aggregated.regularization,
                 'total_loss': total_loss},
                step=self.train_step_counter, name_scope='Losses/')

            # The same tensor in both fields, which is what lets training.py stay untouched: it reads
            # `extra.td_error` or `extra.td_loss` according to SNEK_PRIORITY_SIGNAL and must get a
            # usable signal either way. Upstream returns `()` for both, which would raise on .numpy().
            return tf_agent.LossInfo(
                total_loss,
                dqn_agent.DqnLossInfo(td_loss=per_example_signal, td_error=per_example_signal))
