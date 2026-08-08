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
- **`build_q_net`** is shared with training and `watch.py` and reads `SNEK_FC_LAYERS` the way
  training does. This used to be hardcoded to `(50, 100, 50)` here while training took the
  override, so a run with `SNEK_FC_LAYERS` set was measured against the wrong network — silently,
  because `expect_partial()` suppresses the shape complaint.

The optimizer and `target_update_period` are required by the agent's constructor and are never used:
nothing here trains. They are left at training's values so the checkpoint's variable set matches.
"""
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from snek2 import build_q_net


def build_eval_agent(tf_env, py_env):
    """Returns `(agent, checkpoint, global_step)` for greedy evaluation.

    `tf_env` supplies the time-step and action specs; `py_env` supplies the raw action spec used to
    count actions. Both come from the same `SnakeEnvironment`, so passing a spec env and its TF
    wrapper is the normal call.

    The returned `checkpoint` mirrors the keys `common.Checkpointer` uses in `snek2.py`, which is
    what lets a specific `ckpt-<step>` be restored rather than only the latest.
    """
    action_tensor_spec = tensor_spec.from_spec(py_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    q_net = build_q_net(num_actions)

    global_step = tf.compat.v1.train.get_or_create_global_step()
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
