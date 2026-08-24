"""Why the vectorised engine can measure a c51 arm without doing any atom arithmetic of its own.

`vectorized/vec_eval.py` was written refusing categorical policies, on the stated reasoning that
supporting them meant reading the support out of `arch.json` and reducing over atoms in that file. The
reasoning was wrong about *where* the reduction lives, not about the reduction: `AgentPool` never
touches a Q head. It builds through `eval_agent.build_eval_agent` and asks `policy.action(...)`, and a
categorical agent's greedy policy already reduces over its own support.

So the property the refusal was protecting is real and is enforced one layer down, by two things this
module pins:

1. `build_eval_agent` picks the agent class off the sidecar, so a c51 directory yields a categorical
   agent whose policy is greedy over `CategoricalQPolicy`.
2. That policy's action *is* `argmax_a sum_i z_i p_i(s, a)` over the support the sidecar named --
   checked against the number computed by hand from the network's own logits.

Together those are what make a vec measurement of a c51 checkpoint the same policy a scalar
measurement would have run, which the head-to-head then confirmed empirically (six `b38a` checkpoints,
200 episodes per engine, -0.17 pp, z = -0.10; recorded in `vectorized/README.md`).

A third test records something found while mutation-testing the two above: **the greedy action does not
depend on the support's range at all**, only on its orientation. `sum_i p_i = 1`, so swapping `z` for
`a*z + b` swaps every action's `Q` for `a*Q + b`, and for `a > 0` that is monotone. A mutant that
hardcoded `min_q_value=-10, max_q_value=10` in place of the sidecar's `[-5, 120]` therefore changed
nothing, on any of 256 states -- which is why the support is pinned here at *construction* (where a
hardcoded range is a real defect, because a diagnostic reading `Q` values would be wrong) rather than
through the actions (where it is invisible by arithmetic). `eval_agent.py`'s docstring carries the
correction; the field an evaluation can actually be silently wrong about is `num_atoms`.

Nothing here restores a checkpoint. `build_eval_agent` builds from the sidecar alone, and building is
the step where the algorithm is chosen -- a restore into the wrong class is what `assert_restorable`
exists to prevent, and `tests/test_policy_arch.py` covers that.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import categorical_q_policy, greedy_policy
from tf_agents.trajectories import time_step as ts

import eval_agent
import policy_arch
from snake_environment import SnakeEnvironment, OBS_ERA

# The b31-b39 support. Written out rather than imported so a change to the *default* support does not
# quietly change what this test measures -- the point is that the numbers come from the sidecar.
NUM_ATOMS, V_MIN, V_MAX = 51, -5.0, 120.0
ROWS = 256


# Fixed so an untrained network's argmax is reproducibly *spread* across all three actions. Without
# it the weight draw depends on test order, and a draw whose argmax is constant makes every comparison
# below pass vacuously -- which is exactly what happened on the first run of this module (224/32/0).
# Seed 2 gives 68/118/70 on the 256 probe states.
BUILD_SEED = 2


def _build(**arch_kwargs):
    """`(agent, arch, tempdir)` for a policy directory holding nothing but a sidecar."""
    tf.random.set_seed(BUILD_SEED)
    root = tempfile.mkdtemp()
    policy_dir = os.path.join(root, 'probe')
    os.makedirs(policy_dir)
    arch = policy_arch.build_arch((320,), 3, 30, OBS_ERA, **arch_kwargs)
    with open(os.path.join(policy_dir, policy_arch.ARCH_FILENAME), 'w') as handle:
        json.dump(arch, handle)
    py_env = SnakeEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    agent = eval_agent.build_eval_agent(tf_env, py_env, policy_dir)[0]
    return agent, arch, root


def _time_step(observations):
    rows = observations.shape[0]
    return ts.TimeStep(
        step_type=tf.fill([rows], tf.constant(ts.StepType.MID, dtype=tf.int32)),
        reward=tf.zeros([rows], dtype=tf.float32),
        discount=tf.ones([rows], dtype=tf.float32),
        observation=tf.constant(observations))


def test_a_c51_sidecar_yields_a_greedy_policy_over_a_categorical_q_policy():
    """The shape of the thing `AgentPool` calls. If this ever became a scalar `QPolicy`, the vec engine
    would be evaluating a different policy from the scalar engine while both loaded without complaint
    -- the exact silent failure the refusal was named for."""
    agent, _, root = _build(algo=policy_arch.CATEGORICAL_ALGO, num_atoms=NUM_ATOMS,
                            v_min=V_MIN, v_max=V_MAX)
    try:
        assert isinstance(agent.policy, greedy_policy.GreedyPolicy), type(agent.policy)
        assert isinstance(agent.policy.wrapped_policy, categorical_q_policy.CategoricalQPolicy), (
            type(agent.policy.wrapped_policy))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_greedy_action_is_the_atom_weighted_argmax_over_the_sidecars_support():
    """The reduction the refusal thought `vec_eval` would have to write, shown to happen already.

    Computed from the network's own logits so the assertion does not depend on trained weights -- a
    freshly built net is enough, and its untrained outputs are noisy enough to exercise all three
    actions, which the last assertion checks so this cannot pass vacuously on a constant argmax.
    """
    agent, arch, root = _build(algo=policy_arch.CATEGORICAL_ALGO, num_atoms=NUM_ATOMS,
                               v_min=V_MIN, v_max=V_MAX)
    try:
        observations = np.random.RandomState(0).rand(ROWS, 30).astype('float32')
        chosen = agent.policy.action(_time_step(observations)).action.numpy()

        logits = agent._q_network(tf.constant(observations))[0]
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        support = policy_arch.support_from_arch(arch)
        assert probabilities.shape == (ROWS, 3, NUM_ATOMS), probabilities.shape
        # `support_from_arch` is the sidecar's own reading of the support, and it has to be the one the
        # policy used -- a policy built on `linspace(v_min, v_max)` and compared against a hand-typed
        # range would agree by luck for as long as the two happened to match.
        assert np.allclose(support, np.linspace(V_MIN, V_MAX, NUM_ATOMS)), support
        by_hand = (probabilities * support).sum(axis=-1).argmax(axis=-1)

        assert np.array_equal(chosen, by_hand), (
            'greedy action disagrees with the atom-weighted argmax on {0} of {1} rows'.format(
                int((chosen != by_hand).sum()), ROWS))
        assert set(chosen.tolist()) == {0, 1, 2}, (
            'the probe states do not exercise all three actions: {0}'.format(
                np.bincount(chosen, minlength=3).tolist()))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_ddqn_sidecar_still_yields_a_scalar_policy_from_the_same_call():
    """The other branch of the same `build_eval_agent`, so 'reads the algorithm off the sidecar' is
    pinned as a choice rather than as c51 being the only thing it can build."""
    agent, _, root = _build()
    try:
        assert not isinstance(agent.policy.wrapped_policy,
                              categorical_q_policy.CategoricalQPolicy), 'ddqn built a c51 policy'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_policys_support_is_the_sidecars_support():
    """Construction, not behaviour -- and that split is the point.

    A hardcoded range cannot be caught through the chosen actions (see the module docstring: the argmax
    is invariant to any increasing support), so the only place a wrong support is observable is the
    tensor the policy holds. Anything that reads a `Q` *value* rather than an argmax -- a saliency
    probe, a training resume -- reads it through this.
    """
    agent, arch, root = _build(algo=policy_arch.CATEGORICAL_ALGO, num_atoms=NUM_ATOMS,
                               v_min=V_MIN, v_max=V_MAX)
    try:
        support = agent.policy.wrapped_policy._support.numpy()
        assert np.allclose(support, policy_arch.support_from_arch(arch)), support
        assert support[0] == V_MIN and support[-1] == V_MAX, (support[0], support[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_greedy_action_depends_on_the_supports_orientation_and_not_its_range():
    """The measurement behind the docstring correction in `eval_agent.py`.

    Every policy here wraps the *same* network, so the only thing varying is the support -- which a
    second `build_eval_agent` call could not give us, since it would draw new weights.
    """
    agent, _, root = _build(algo=policy_arch.CATEGORICAL_ALGO, num_atoms=NUM_ATOMS,
                            v_min=V_MIN, v_max=V_MAX)
    try:
        observations = np.random.RandomState(0).rand(ROWS, 30).astype('float32')
        py_env = SnakeEnvironment()
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        def actions_under(low, high):
            policy = greedy_policy.GreedyPolicy(categorical_q_policy.CategoricalQPolicy(
                tf_env.time_step_spec(), tf_env.action_spec(), agent._q_network,
                min_q_value=low, max_q_value=high))
            return policy.action(_time_step(observations)).action.numpy()

        baseline = agent.policy.action(_time_step(observations)).action.numpy()
        assert set(baseline.tolist()) == {0, 1, 2}, (
            'the probe states do not exercise all three actions: {0}'.format(
                np.bincount(baseline, minlength=3).tolist()))

        # `a > 0`: a monotone transform of every action's Q, so not one state may change.
        for low, high in ((-10.0, 10.0), (0.0, 1.0), (-1000.0, 3.0)):
            other = actions_under(low, high)
            assert np.array_equal(other, baseline), (
                'support [{0}, {1}] changed {2} of {3} actions'.format(
                    low, high, int((other != baseline).sum()), ROWS))

        # `a < 0`: an argmin. Included so the invariance above reads as arithmetic rather than as the
        # support being ignored -- a policy that never looked at `z` would pass the loop too.
        reversed_support = actions_under(V_MAX, V_MIN)
        assert not np.array_equal(reversed_support, baseline), (
            'a reversed support chose the same actions, so the support is not being used at all')
    finally:
        shutil.rmtree(root, ignore_errors=True)
