"""One rollout: every lane steps `T` times, and every step is stored with the value and log-prob the
policy actually produced.

Much simpler than `dqn/collect.py`, and the deleted parts are the interesting half:

| `dqn/collect.py` has | why PPO does not |
|---|---|
| forking — a cloned state with a forced action | **off-policy by construction.** The clipped ratio `pi(a|s)/pi_old(a|s)` assumes the batch came from `pi_old`, and a forced action came from nothing. PPO gets the fork's benefit for free: it samples, so both actions at a decision point are tried with a correctly attributed log-prob |
| the exploration shield's `guided` flags | there is no epsilon coin to shield. Exploration is the entropy bonus, a cost term rather than a forced random-action rate |
| per-lane n-step windows | GAE's lambda is the generalisation of an n-step window, and it lives in `rollout.py` |

What is kept is the one thing that made that file short: **every lane steps on every call.** `VecSnake`
advances all of them at once, so this is a `T`-long loop over one `vec.step()` and one policy forward.

## Two things stored here that DQN never needed

**`log_prob` at collect time.** It is `pi_old` — the denominator of PPO's ratio — and recomputing it
later from the same weights is not the same guarantee: the first epoch's first minibatch must see a
ratio of exactly 1.0, and that is only true if the stored value came from the same forward pass as the
action. `tests/test_ppo_agent.py` asserts it as an equality, because it is the single best detector of
a wrong log-prob anywhere in the stack.

**`value` at collect time**, for the same reason in the other direction: GAE needs `V(s_t)` under the
policy that *collected* `s_t`, and by the time the epochs have run the critic has moved.
"""

import numpy as np


class Collector(object):
    """Fills a `Rollout` and reports what happened while it did.

    Holds no episode state of its own beyond the current observation: `VecSnake` auto-resets, and a
    rollout boundary is not an episode boundary, so there is nothing to flush.
    """

    def __init__(self, vec, agent, rollout):
        if vec.n != rollout.lanes:
            raise ValueError('the env has {0} lanes and the rollout has {1}'.format(
                vec.n, rollout.lanes))
        self.vec = vec
        self.agent = agent
        self.rollout = rollout
        self.counters = {name: 0 for name in
                         ('episodes', 'transitions', 'perfect_games', 'rollouts')}
        self.obs = vec.reset_all()
        if self.obs is None:
            self.obs = vec.observe()

    def collect(self):
        """One full rollout. Returns the transitions banked, which is `steps * lanes` exactly.

        Exactly, always — unlike a DQN step, where an episode boundary emits a whole n-step window.
        Nothing here is conditional, which is why a PPO arm's step count and its game-move count are
        the same number.
        """
        for t in range(self.rollout.steps):
            actions, log_probs, values = self.agent.act(self.obs)
            previous = self.obs
            self.obs, rewards, done, info = self.vec.step(actions)
            self.rollout.add(t, previous, actions, log_probs, values, rewards, done)
            finished = int(np.count_nonzero(done))
            if finished:
                self.counters['episodes'] += finished
                self.counters['perfect_games'] += int(np.count_nonzero(info['perfect']))
        # `V` of the state every lane is in *now*, for the truncated rollout's bootstrap. Taken after
        # the loop rather than inside it, because the value of the state after the last stored step is
        # the one GAE needs and it is not any state the loop already scored.
        last_values = self.agent.values(self.obs)
        self.rollout.finish(last_values, self.agent.discount, self.agent.gae_lambda)
        banked = self.rollout.size
        self.counters['transitions'] += banked
        self.counters['rollouts'] += 1
        return banked

    def snapshot(self):
        """The counters, for the eval row. Same role as `dqn/collect.py`'s fork block."""
        return dict(self.counters)
