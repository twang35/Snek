"""PPO: the clipped surrogate, the value loss, the entropy bonus, and the epoch loop over a rollout.

The same seam shape `dqn/agent.py` sits on, with three methods instead of two because a rollout needs
the value and the log-prob at collect time:

    agent.act(obs)      -> (actions, log_probs, values)   # (n, 30) -> three (n,)
    agent.values(obs)   -> (n,)                           # V only, for GAE's bootstrap
    agent.update(rollout) -> metrics                      # every epoch, every minibatch
    agent.policy_fn     -> callable                       # greedy, for engine.measure
    agent.state_dict() / load_state_dict()

## The objective

    ratio = exp(logp - logp_old)
    L_pi  = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).mean()
    L_v   = huber(V, returns)
    loss  = L_pi + vf_coef * L_v - entropy_coef * entropy.mean()

**`min`, and the sign of the advantage is what makes it a trust region.** With `A > 0` the objective
is capped at `1+eps`, so pushing the action further costs nothing and gains nothing — no gradient. With
`A < 0` it is capped at `1-eps` in the other direction. Flip the `min` to a `max` and it becomes an
*anti*-trust region that rewards leaving the region, which trains until it diverges and looks like a
learning-rate problem. `tests/test_ppo_agent.py` asserts the **gradient** is zero in both clipped
branches rather than asserting the loss value, because a loss-value fixture passes with the wrong
branch selected.

## Three defaults that are deliberate deviations, each with its reason

**Learning rate 3e-4, not DQN's 1e-5.** PPO takes ~64x fewer gradient steps per transition — one
minibatch per 256 samples per epoch, against DQN's one batch of 128 per *transition* — so the same
learning rate is ~64x less total parameter movement over an arm. Reusing DQN's number is the single
easiest way to conclude "PPO does not work here". Hence a distinct knob name, `SNEK_PPO_LEARNING_RATE`,
and `SNEK_LEARNING_RATE` is refused rather than ignored.

**Huber on the value loss, not squared error.** The same argument `dqn/agent.py` gives for its TD
loss: a perfect game pays +100 against a typical step's ~0.001, so one terminal return in a minibatch
of 256 would dominate a squared error. This is a departure from textbook PPO and it is therefore a
knob, `SNEK_PPO_VALUE_LOSS=mse`.

**No value clipping.** The other half of the "clipped value loss" folklore has no consistent
empirical support, and it interacts with the huber choice above in a way nothing here has measured.
Left out rather than added untested.

## What the row carries, and why it is not the log

`entropy`, `approx_kl`, `clip_fraction`, `explained_variance`, `policy_loss`, `value_loss` go into
every eval row's `ppo` block — the same place a DQN row keeps its fork counters. These are what
separate "PPO does not work on this task" from "the policy collapsed at 200k transitions", and a
diagnostic that only reaches the log is a diagnostic nobody reads at close-out.

`explained_variance` is `1 - Var(returns - V) / Var(returns)`: 1 is a perfect critic, 0 is a critic no
better than predicting the mean, and **negative means worse than the mean**, which is the reading that
tells you the critic is the problem.
"""

import numpy as np
import torch
import torch.nn.functional as F

from dqn.agent import build_adam
from ppo import net as network
from ppo.rollout import normalise

# `approx_kl` estimator. `mean(logp_old - logp)` is the naive one and is *signed* — it can read
# negative, which is impossible for a KL and makes an early-stop threshold meaningless. This is
# Schulman's low-variance unbiased estimator, which is non-negative by construction.
#
#     kl ~= mean(exp(d) - 1 - d)  where d = logp - logp_old
#
# Worth spelling out because the naive version is what most implementations print, and a negative KL
# in a report is otherwise read as a bug in the log rather than in the estimator.
def approx_kl(log_probs, old_log_probs):
    delta = log_probs - old_log_probs
    return float((delta.exp() - 1.0 - delta).mean())


def explained_variance(returns, values):
    """`1 - Var(returns - values) / Var(returns)`. Negative means worse than predicting the mean."""
    returns = np.asarray(returns, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    spread = returns.var()
    if spread < 1e-12:
        return 0.0
    return float(1.0 - (returns - values).var() / spread)


class PpoAgent(object):
    """The actor, the critic, one optimiser over both, and the update.

    One optimiser rather than two: the towers share no parameters, and Adam's state is per-parameter,
    so a single optimiser over the concatenation is exactly two optimisers with the same
    hyperparameters — one fewer thing to checkpoint and to keep in step.
    """

    def __init__(self, arch, config, device='cpu'):
        self.arch = arch
        self.config = config
        self.device = device
        self.num_actions = int(arch['num_actions'])
        self.discount = float(config['discount'])
        self.gae_lambda = float(config['ppo_gae_lambda'])
        self.clip = float(config['ppo_clip'])
        self.vf_coef = float(config['ppo_vf_coef'])
        self.epochs = int(config['ppo_epochs'])
        self.minibatch = int(config['ppo_minibatch'])
        self.normalise_advantages = bool(config['ppo_normalize_adv'])
        self.target_kl = float(config['ppo_target_kl'])
        self.gradient_clipping = float(config['ppo_gradient_clipping'])
        self.value_loss = str(config['ppo_value_loss'])
        if self.value_loss not in ('huber', 'mse'):
            raise ValueError("SNEK_PPO_VALUE_LOSS={0!r} is not 'huber' or 'mse'".format(
                self.value_loss))

        seed = config['seed']
        self.actor = network.build(arch, device=device, seed=seed)
        self.critic = network.build_critic(arch, device=device, seed=seed)
        self.optimizer = build_adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                   float(config['ppo_learning_rate']),
                                   float(config['ppo_adam_epsilon']))
        # The action draw's own stream. Not torch's global generator and not the env's numpy one:
        # `dqn/collect.py`'s note applies unchanged — an arm's decisions must not depend on how many
        # food cells the env rejected.
        self.torch_rng = torch.Generator(device=device)
        if seed is not None:
            self.torch_rng.manual_seed(int(seed))
        # And the minibatch shuffle's, which is a third independent consumer.
        self.shuffle_rng = np.random.default_rng(seed)
        self.entropy_coef = float(config['ppo_entropy_coef'])
        self.train_step = 0

    # ------------------------------------------------------------ acting

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.actor, self.device)

    def set_learning_rate(self, value):
        """Adam's step size, on every parameter group — there is one optimiser over both towers."""
        for group in self.optimizer.param_groups:
            group['lr'] = float(value)

    def learning_rate(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _tensor(self, observations):
        return torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)

    def act(self, observations):
        """Sample. Returns `(actions, log_probs, values)` as numpy, all `(n,)`.

        `no_grad` is correctness rather than speed: a rollout of 16,384 steps built under autograd
        would hold every intermediate activation of every step alive until the buffer was overwritten.
        """
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            batch = self._tensor(observations)
            actions, log_probs = network.sample(self.actor(batch), generator=self.torch_rng)
            values = self.critic(batch).squeeze(-1)
        return (actions.to(torch.int64).cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy())

    def values(self, observations):
        """`V` only, for GAE's bootstrap off the state after the last stored step."""
        self.critic.eval()
        with torch.no_grad():
            return self.critic(self._tensor(observations)).squeeze(-1).cpu().numpy()

    # ------------------------------------------------------------ learning

    def _value_loss(self, predicted, returns):
        if self.value_loss == 'mse':
            return F.mse_loss(predicted, returns)
        return F.huber_loss(predicted, returns, delta=1.0)

    def update(self, rollout):
        """Every epoch, every minibatch. Returns the diagnostics an eval row carries.

        Stops early between *epochs* when `approx_kl` exceeds `SNEK_PPO_TARGET_KL`, never mid-epoch:
        a half-finished epoch leaves some samples used more often than others, which is a silent bias
        toward whatever the shuffle happened to put first. Zero disables the check, and the KL is
        reported either way — which is the point, since a KL that would have tripped a threshold
        nobody set is exactly the thing worth seeing at close-out.
        """
        self.actor.train()
        self.critic.train()
        totals = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0,
                  'clip_fraction': 0.0}
        batches = 0
        epochs_run = 0
        stopped_early = False

        for _ in range(self.epochs):
            epoch_kl = 0.0
            epoch_batches = 0
            for batch in rollout.minibatches(self.minibatch, self.shuffle_rng):
                advantages = batch['advantages']
                if self.normalise_advantages:
                    advantages = normalise(advantages)
                obs = self._tensor(batch['obs'])
                actions = torch.as_tensor(batch['actions'], device=self.device).long()
                old_log_probs = self._tensor(batch['log_probs'])
                advantage = self._tensor(advantages)
                returns = self._tensor(batch['returns'])

                log_probs, entropy = network.evaluate(self.actor(obs), actions)
                ratio = (log_probs - old_log_probs).exp()
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = self._value_loss(self.critic(obs).squeeze(-1), returns)
                entropy_mean = entropy.mean()
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.entropy_coef * entropy_mean)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clipping > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.gradient_clipping)
                self.optimizer.step()
                self.train_step += 1

                with torch.no_grad():
                    kl = approx_kl(log_probs, old_log_probs)
                    # The share of samples the clip was actually binding on. Near 0 means the update
                    # is not being constrained at all; near 1 means the step is far too large for the
                    # clip and almost nothing is learning.
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > self.clip).to(torch.float32).mean())
                totals['policy_loss'] += float(policy_loss.detach())
                totals['value_loss'] += float(value_loss.detach())
                totals['entropy'] += float(entropy_mean.detach())
                totals['approx_kl'] += kl
                totals['clip_fraction'] += clip_fraction
                epoch_kl += kl
                epoch_batches += 1
                batches += 1

            epochs_run += 1
            if self.target_kl > 0.0 and epoch_batches and epoch_kl / epoch_batches > self.target_kl:
                stopped_early = True
                break

        metrics = {key: (value / batches if batches else 0.0) for key, value in totals.items()}
        flat = rollout.flat()
        metrics['explained_variance'] = explained_variance(flat['returns'], flat['values'])
        metrics['epochs_run'] = epochs_run
        metrics['stopped_early'] = stopped_early
        metrics['train_step'] = self.train_step
        return metrics

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_step': self.train_step,
                'torch_rng': self.torch_rng.get_state(),
                'shuffle_rng': self.shuffle_rng.bit_generator.state}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.train_step = int(state.get('train_step', 0))
        if state.get('torch_rng') is not None:
            self.torch_rng.set_state(state['torch_rng'])
        if state.get('shuffle_rng') is not None:
            self.shuffle_rng.bit_generator.state = state['shuffle_rng']
