"""Exploration-only safety shield for the collect policy.

Batch 12 established that a mastery-gated epsilon schedule deadlocks: epsilon pinned at the
refinement ceiling 0.05 for up to 942k steps, and all four arms sat at 0% perfect games with
greedy trailing scores of 55-62 against batch 11's 84-88. The mechanism is that 3.3% of
*collected* actions are random and a random move with a long snake is usually fatal, so the
replay buffer fills with trajectories that die before the endgame, the greedy policy never
learns to finish, the perfect rate stays 0, and the schedule never descends. See
hyperparamTuning/runs.md.

This shield attacks the cost of exploration rather than its quantity. During a *guided*
episode, an action drawn by the epsilon coin is sampled from the moves that do not kill the
snake this step, instead of uniformly from all three.

**Only the exploration draw is shielded. The greedy action is never overridden.** That
asymmetry is the whole design:

| branch | shielded? | why |
|---|---|---|
| epsilon picked a random move | yes | a blunder the agent did not choose teaches it nothing |
| the network's own argmax is fatal | **no** | it must eat the -5 and learn, exactly as before |

Overriding a fatal *greedy* action would mean `Q(s, a_fatal)` never gets updated toward
DEATH_REWARD for the states where the network is wrong, so those values would drift on
generalisation alone — and evals run unshielded (`agent.policy`), so the arm would walk into
walls it was never allowed to learn about. Shielding exploration only removes the tax without
removing any learning signal: every death the policy earns itself still happens.

The shield is one step deep, which is deliberate. Snake's hard problem is sealing itself into a
region it cannot escape, and that is untouched here — an arm still has to learn it. All this
removes is "the coin flipped and the snake drove into its own body".
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step

# Observation indices 6-8 are "is the move safe (not body or wall)", 1 for safe and 0 for a
# collision, per action in ACTIONS order. Read from the observation rather than recomputed:
# `state_helpers.body_and_wall_collisions` already does this exactly, including the one case a
# naive check gets wrong — the cell the tail is vacating this step is safe to move into. Keep
# in step with the layout table in `state_helpers.get_observations`.
SAFETY_OBS_START = 6
SAFETY_OBS_END = 9

# Logit for an action the shield masks out, giving it exactly zero probability. -inf rather than
# a large finite negative deliberately: with a finite value, an all-masked row still samples
# uniformly, because `tf.random.categorical` shifts each row by its maximum before drawing. That
# would make the `~any_safe` fallback below *look* redundant — a mutation removing it survives
# the suite — while leaving the boxed-in case resting on an undocumented internal of TF. With
# -inf the fallback is the only thing standing between a boxed-in snake and a NaN, so it is
# load-bearing and tested as such.
MASKED_LOGIT = float('-inf')


class ShieldedEpsilonGreedyPolicy(tf_policy.TFPolicy):
    """Epsilon-greedy where the random branch avoids immediately fatal moves.

    Drop-in replacement for `agent.collect_policy`. With `guided_fraction` at 0 it is
    behaviourally identical to `tf_agents`' `EpsilonGreedyPolicy` — same epsilon semantics
    (the draw is uniform over *all* actions, so it can re-pick the greedy one, which is why
    the effective non-greedy rate is `epsilon * 2/3`), and the same greedy action otherwise.

    `guided_fraction` is read as a probability and resolved **once per episode**, not per step,
    so an episode is guided end to end or not at all. A per-step coin would mix the two
    regimes inside one trajectory and make the resulting buffer impossible to reason about.
    """

    def __init__(self, greedy_policy, epsilon, guided_fraction, name=None):
        if greedy_policy.info_spec != ():
            # The exploration branch reports the greedy step's info, which is only honest when
            # there is none. Fail loudly rather than silently mislabelling explored actions.
            raise ValueError(
                'ShieldedEpsilonGreedyPolicy needs a policy with an empty info_spec, got {0}'
                .format(greedy_policy.info_spec))
        super(ShieldedEpsilonGreedyPolicy, self).__init__(
            greedy_policy.time_step_spec,
            greedy_policy.action_spec,
            policy_state_spec=greedy_policy.policy_state_spec,
            info_spec=greedy_policy.info_spec,
            emit_log_probability=False,
            name=name)
        self._greedy_policy = greedy_policy
        self._epsilon = epsilon
        self._guided_fraction = guided_fraction
        self._action_offset = int(greedy_policy.action_spec.minimum)
        self._num_actions = int(greedy_policy.action_spec.maximum
                                - greedy_policy.action_spec.minimum + 1)
        # Whether the episode in progress is guided. A Variable rather than policy state
        # because the collect driver runs a single environment and re-enters `_action` inside a
        # `tf.function`, where a Python attribute would be baked in at trace time — the same
        # reason epsilon is a Variable. See the comment above `epsilon` in snek2.py.
        self._guided_episode = tf.Variable(False, dtype=tf.bool, trainable=False,
                                           name='guided_episode')

    @property
    def guided_episode(self):
        """Exposed for tests and diagnostics; the driver never reads it."""
        return self._guided_episode

    def _variable_value(self, value):
        return tf.convert_to_tensor(value() if callable(value) else value, dtype=tf.float32)

    def _resolve_guided(self, time_step, seed):
        """The per-episode guided flag, redrawn on the first step of each episode."""
        redraw = tf.reduce_any(time_step.is_first())
        fraction = self._variable_value(self._guided_fraction)

        def draw():
            drawn = tf.random.uniform([], seed=seed) < fraction
            self._guided_episode.assign(drawn)
            return drawn

        return tf.cond(redraw, draw, self._guided_episode.value)

    def _action(self, time_step, policy_state, seed):
        seed_stream = tfp.util.SeedStream(seed=seed, salt='shielded_epsilon_greedy')
        greedy_step = self._greedy_policy.action(time_step, policy_state)
        guided = self._resolve_guided(time_step, seed_stream())

        safe = time_step.observation[:, SAFETY_OBS_START:SAFETY_OBS_END] > 0.5
        # A state with no safe move lets everything through, so the snake takes the move and
        # dies — there is nothing to steer it to. An unguided episode lets everything through
        # too, which is what makes guided_fraction=0 reproduce the unshielded behaviour.
        any_safe = tf.reduce_any(safe, axis=-1, keepdims=True)
        allowed = safe | ~any_safe | ~guided
        logits = tf.where(allowed, tf.zeros_like(safe, dtype=tf.float32),
                          tf.fill(tf.shape(safe), MASKED_LOGIT))
        drawn = tf.random.categorical(logits, 1, dtype=greedy_step.action.dtype,
                                      seed=seed_stream())
        explore_action = tf.squeeze(drawn, axis=-1) + self._action_offset

        rng = tf.random.uniform(tf.shape(greedy_step.action), dtype=tf.float32,
                                seed=seed_stream())
        explore = rng < self._variable_value(self._epsilon)
        action = tf.where(explore, explore_action, greedy_step.action)
        return policy_step.PolicyStep(action, greedy_step.state, greedy_step.info)
