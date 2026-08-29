"""The evaluation agent, built in exactly one place.

`eval_checkpoints.py` builds a greedy `DdqnAgent` to restore checkpoints into, and every
independent worker in `eval_workers.py` now needs an identical one. A second copy of this
construction is precisely the failure this repo has been bitten by twice: nothing checks that a
restored checkpoint *means* what the network thinks it means, and `expect_partial()` hides a
mismatch, so two builders that drift apart produce a policy that loads silently and plays like a
beginner.

Two settings here are load-bearing:

- **`epsilon_greedy=0.0`.** Eval is greedy; epsilon only ever shapes the collect policy. Every
  number this project reports is a greedy measurement.
- **`build_q_net`** is shared with training and `watch.py`. Its shape used to be read from
  `SNEK_FC_LAYERS` (hardcoded `(50, 100, 50)` here before that), so a checkpoint trained with the
  override was measured against the wrong network — silently, because `expect_partial()` suppresses
  the shape complaint. Now the shape comes from the checkpoint's own `arch.json` via
  `policy_arch.assert_restorable`, which also fails loudly if the observation length or meaning-era
  disagrees, so there is nothing left to forget to set.

The optimizer and `target_update_period` are required by the agent's constructor and are never used:
nothing here trains. They are left at training's values so the checkpoint's variable set matches.

**Which agent to build is read off `arch.json`, never off the environment.** A c51 checkpoint restored
into a scalar network fails on shape, which is fine. So the algorithm and the support both come from
the sidecar the checkpoint was written with, and `SNEK_ALGO`/`SNEK_V_MAX` are not read here at all —
the same rule `SNEK_FC_LAYERS` already follows.

**‡ This docstring used to say a wrong *support* would "load perfectly and evaluate a different
policy". For a greedy evaluation that is false, and it is worth knowing why** (measured 2026-08-24,
pinned in `tests/test_c51_eval_path.py`). The greedy action is `argmax_a sum_i z_i p_i(s, a)`, and
`sum_i p_i = 1`, so replacing the support `z` with `a·z + b` replaces every action's `Q` with
`a·Q + b` — a monotone transform when `a > 0`, which leaves the argmax untouched. Measured on 256
states: `[-5, 120]`, `[-10, 10]`, `[0, 1]` and `[-1000, 3]` all chose the *same* action every time,
and only a **reversed** support (`v_min > v_max`, so `a < 0`) differed — on all 256, because it is
then an argmin.

Two things follow, and they cut in opposite directions. `v_min`/`v_max` are not the field an
evaluation can be silently wrong about, so **do not cite the range as the reason a c51 eval needs the
sidecar** — the reasons are `num_atoms`, which sets the logits width and therefore fails the restore
on shape, and `algo` itself. And the invariance is *not* a licence to stop recording the range: it
holds for the greedy action only, so anything that reads a `Q` *value* — a diagnostic, a saliency
probe, a training resume — still needs the support the checkpoint was trained with.
"""
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import categorical_agent
import policy_arch
from snake_environment import OBS_ERA
from snek2 import build_categorical_q_net, build_q_net


def build_eval_agent(tf_env, py_env, policy_dir):
    """Returns `(agent, checkpoint, global_step)` for greedy evaluation of the policy in `policy_dir`.

    `tf_env` supplies the time-step and action specs; `py_env` supplies the raw action spec used to
    count actions and the observation length. Both come from the same `SnakeEnvironment`, so passing
    a spec env and its TF wrapper is the normal call.

    `policy_dir` is the checkpoint directory. Its `arch.json` is required and checked against this
    environment (`policy_arch.assert_restorable`) before the network is built from the *recorded*
    layer widths, so a missing sidecar or a shape/observation mismatch stops here rather than
    restoring silently.

    The returned `checkpoint` mirrors the keys `common.Checkpointer` uses in `snek2.py`, which is
    what lets a specific `ckpt-<step>` be restored rather than only the latest.
    """
    action_tensor_spec = tensor_spec.from_spec(py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    obs_len = int(py_env.observation_spec().shape[0])
    arch = policy_arch.assert_restorable(policy_dir, num_actions, obs_len, OBS_ERA)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if policy_arch.is_categorical(arch):
        q_net = build_categorical_q_net(
            tf_env.observation_spec(), action_tensor_spec, arch['fc_layer_params'],
            arch['num_atoms'])
        agent = categorical_agent.SnekCategoricalDqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            categorical_q_network=q_net,
            min_q_value=arch['v_min'],
            max_q_value=arch['v_max'],
            epsilon_greedy=0.0,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            target_update_period=8,
            train_step_counter=global_step)
    else:
        q_net = build_q_net(num_actions, arch['fc_layer_params'])
        agent = dqn_agent.DdqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=0.0,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            td_errors_loss_fn=common.element_wise_huber_loss,
            target_update_period=8,
            train_step_counter=global_step)
    agent.initialize()

    checkpoint = tf.train.Checkpoint(agent=agent, policy=agent.policy, global_step=global_step)
    return agent, checkpoint, global_step
