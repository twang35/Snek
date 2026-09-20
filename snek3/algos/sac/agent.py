"""Discrete SAC: the soft target, the twin critics, the categorical actor and the temperature.

The categorical policy makes every expectation over actions exact rather than sampled, so (Christodoulou
2019):

    V_soft(s')    = sum_a pi(a|s') [ Q'(s', a) - alpha log pi(a|s') ]        Q' = min (or avg) of the target critics
    y             = r + discount * V_soft(s')                                 discount already gamma**n, 0 at a terminal
    L_critic      = Huber(Q_i(s, a), y) for i = 1, 2                         (B1), or the Q-clipped squared error (B2)
    L_actor       = sum_a pi(a|s) [ alpha log pi(a|s) - Q(s, a) ]           = alpha * KL(pi || softmax(Q / alpha)) + const
    L_alpha       = log alpha * (H(pi) - H_target)                           H_target = ratio * log |A|; auto only

and Zhou et al. 2022's three fixes, each a knob: `combine='avg'` (double average Q), `q_clip > 0` (the
critic loss is the larger of the plain squared error and the one where Q moves at most `c` from the
target critic's value), and `entropy_penalty > 0` (beta * 1/2 (H_prev - H)^2 on the actor loss, H_prev the
mean entropy at the previous update, so the penalty has a gradient; at the same batch it would not).

The six pure functions are the loss arithmetic, kept out of the class so a fixture can hand them a
two-action example and so a mutant in one of them is a mutant of one line.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from algos.dqn.agent import build_adam
from algos.sac import net as network

COMBINES = ('min', 'avg')
CRITIC_LOSSES = ('mse', 'huber')


# ---------------------------------------------------------------- the arithmetic

def combine(q_a, q_b, how):
    """The two critics' read of Q(s, .): elementwise min (SAC) or mean (double average Q)."""
    if how == 'min':
        return torch.minimum(q_a, q_b)
    if how == 'avg':
        return 0.5 * (q_a + q_b)
    raise ValueError('critic combine {0!r} is not one of {1}'.format(how, COMBINES))


def soft_state_value(log_pi, q, alpha):
    """`sum_a pi(a|s) (q(s, a) - alpha log pi(a|s))`, per row. `log_pi` and `q` are `(B, A)`."""
    return (log_pi.exp() * (q - alpha * log_pi)).sum(dim=1)


def actor_loss(log_pi, q, alpha):
    """`sum_a pi (alpha log pi - q)`, per row: minimised by `pi = softmax(q / alpha)`."""
    return (log_pi.exp() * (alpha * log_pi - q)).sum(dim=1)


def alpha_loss(log_alpha, entropy, target_entropy):
    """Its gradient on `log_alpha` is `H - H_target`: alpha falls while the policy is more random than
    the target asks and rises while it is less."""
    return log_alpha * (entropy.detach() - float(target_entropy))


def clipped_critic_loss(q, q_old, y, clip):
    """Zhou et al.'s Q-clip: `max((q - y)^2, (q_old + clamp(q - q_old, -c, c) - y)^2)`, `q_old` the target
    critic's value of the same (s, a). Returns `(loss per row, whether the clipped branch won)`."""
    plain = (q - y) ** 2
    clipped = (q_old + torch.clamp(q - q_old, -float(clip), float(clip)) - y) ** 2
    return torch.maximum(plain, clipped), (clipped > plain)


def entropy_penalty(old_entropy, entropy, beta):
    """Zhou et al.'s entropy-penalty, per replayed state: `beta * 1/2 (H_old(s) - H(s))^2`, `H_old(s)`
    the entropy the *collecting* policy had at `s`, stored with the transition (`replay.aux`), `H(s)` the
    current policy's. Returns one value per row; the caller averages. The per-state form is the point:
    a mean-to-mean difference lets two states whose entropies moved in opposite directions cancel, and
    penalises a change in which states were sampled rather than a change in the policy (an earlier
    version here compared consecutive minibatches' means and was wrong, 2026-09-20)."""
    if beta <= 0.0:
        return entropy * 0.0
    return float(beta) * 0.5 * (old_entropy.detach() - entropy) ** 2


def target_entropy_for(num_actions, ratio):
    return float(ratio) * math.log(int(num_actions))


# ---------------------------------------------------------------- the agent

class SacAgent(object):
    """The actor (the checkpoint), two critics and their targets, three optimisers at most."""

    def __init__(self, arch, config, device='cpu'):
        self.arch = arch
        self.config = config
        self.device = device
        seed = config['seed']
        self.num_actions = int(arch['num_actions'])
        self.actor = network.build(arch, device, seed=seed)
        self.q1 = network.build_critic(arch, device, seed=seed, stream=network.CRITIC_SEED_STREAMS[0])
        self.q2 = network.build_critic(arch, device, seed=seed, stream=network.CRITIC_SEED_STREAMS[1])
        self.q1_target = network.build_critic(arch, device)
        self.q2_target = network.build_critic(arch, device)
        for target, online in ((self.q1_target, self.q1), (self.q2_target, self.q2)):
            target.load_state_dict(online.state_dict())
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        eps = float(config['sac_adam_epsilon'])
        self.actor_optimizer = build_adam(self.actor.parameters(), float(config['sac_learning_rate']), eps)
        self.critic_optimizer = build_adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                           float(config['sac_critic_learning_rate']), eps)

        # The temperature: `auto` tunes log alpha by gradient toward a target entropy; a number fixes it
        # and builds no optimiser (B2's fixed 0.05).
        self.target_entropy = target_entropy_for(self.num_actions, config['sac_target_entropy_ratio'])
        self.auto_alpha = str(config['sac_alpha']).strip().lower() == 'auto'
        initial = float(config['sac_init_alpha']) if self.auto_alpha else float(config['sac_alpha'])
        if initial <= 0.0:
            raise ValueError('alpha must be positive, got {0}'.format(initial))
        self.log_alpha = torch.tensor([math.log(initial)], dtype=torch.float32, device=device,
                                      requires_grad=self.auto_alpha)
        self.alpha_optimizer = None
        if self.auto_alpha:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(config['sac_alpha_learning_rate']))

        self.combine = str(config['sac_critic_combine'])
        if self.combine not in COMBINES:
            raise ValueError('SNEK_SAC_CRITIC_COMBINE={0!r} is not one of {1}'.format(self.combine, COMBINES))
        self.q_clip = float(config['sac_q_clip'])
        self.critic_loss_kind = str(config['sac_critic_loss'])
        if self.critic_loss_kind not in CRITIC_LOSSES:
            raise ValueError('SNEK_SAC_CRITIC_LOSS={0!r} is not one of {1}'.format(self.critic_loss_kind, CRITIC_LOSSES))
        self.entropy_penalty = float(config['sac_entropy_penalty'])
        self.target_update_period = int(config['sac_target_update_period'])
        self.tau = float(config['sac_tau'])
        self.use_is_weights = float(config['sac_priority_exponent']) > 0.0

        # The policy's own sampling stream, as PPO's: never torch's global one, so the env's food draws
        # and the actor's action draws stay independent.
        self.torch_rng = torch.Generator(device=device)
        if seed is not None:
            self.torch_rng.manual_seed(int(seed))
        self.train_step = 0
        # The policy's entropy at each lane's state on the last `act()`, one float per lane: the collector
        # stores it with the transition, and the entropy-penalty reads it back as `H_old(s)`.
        self.act_aux = None
        self.last_metrics = {}

    # ---------------------------------------------------------------- acting

    @property
    def alpha(self):
        return float(self.log_alpha.detach().exp())

    @property
    def policy_fn(self):
        return network.greedy_policy_fn(self.actor, self.device)

    def act(self, observations, epsilon=0.0, guided=False):
        """A sample from pi. `epsilon` and `guided` are the collector's DQN arguments and mean nothing here:
        SAC's exploration is its own entropy, and shielding a stochastic policy's draws would put a
        distribution under the critics that the actor does not induce."""
        self.actor.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            logits = self.actor(tensor)
            actions, _ = network.sample(logits, generator=self.torch_rng)
            log_pi = network.log_softmax(logits)
            self.act_aux = (-(log_pi.exp() * log_pi).sum(dim=1)).cpu().numpy().astype(np.float32)
        return actions.cpu().numpy().astype(np.int64)

    # ---------------------------------------------------------------- learning

    def update(self, batch, weights=None):
        """One update of the critics, the actor and (auto) the temperature. Returns `(td_errors, metrics)`;
        the errors feed the replay's priorities when it is prioritised."""
        self.actor.train(); self.q1.train(); self.q2.train()
        obs = torch.as_tensor(batch['obs'], device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        action = torch.as_tensor(batch['action'], device=self.device).long().unsqueeze(1)
        reward = torch.as_tensor(batch['reward'], device=self.device).float()
        discount = torch.as_tensor(batch['discount'], device=self.device).float()
        old_entropy = torch.as_tensor(batch['aux'], device=self.device).float()
        alpha = self.alpha
        is_weights = None
        if weights is not None and self.use_is_weights:
            is_weights = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=self.device)

        with torch.no_grad():
            log_pi_next = network.log_softmax(self.actor(next_obs))
            q_next = combine(self.q1_target(next_obs), self.q2_target(next_obs), self.combine)
            y = reward + discount * soft_state_value(log_pi_next, q_next, alpha)

        # -- the critics
        q1_all, q2_all = self.q1(obs), self.q2(obs)
        q1_a, q2_a = q1_all.gather(1, action).squeeze(1), q2_all.gather(1, action).squeeze(1)
        clip_wins = None
        if self.q_clip > 0.0:
            with torch.no_grad():
                q1_old = self.q1_target(obs).gather(1, action).squeeze(1)
                q2_old = self.q2_target(obs).gather(1, action).squeeze(1)
            loss_1, won_1 = clipped_critic_loss(q1_a, q1_old, y, self.q_clip)
            loss_2, won_2 = clipped_critic_loss(q2_a, q2_old, y, self.q_clip)
            clip_wins = float(torch.cat([won_1, won_2]).float().mean())
        elif self.critic_loss_kind == 'huber':
            # The local departure, as `algos/dqn/agent.py` argues for a +100 terminal against ~0.01 steps.
            loss_1 = F.huber_loss(q1_a, y, reduction='none', delta=1.0)
            loss_2 = F.huber_loss(q2_a, y, reduction='none', delta=1.0)
        else:
            # MSE, both papers' critic loss; the default.
            loss_1 = (q1_a - y) ** 2
            loss_2 = (q2_a - y) ** 2
        critic_losses = loss_1 + loss_2
        if is_weights is not None:
            critic_losses = critic_losses * is_weights
        critic_loss = critic_losses.mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # -- the actor, against the critics *as just updated*: both papers' code steps the critics and
        # then evaluates the actor objective on a fresh critic pass (Christodoulou's SAC_Discrete,
        # Zhou et al.'s Tianshou policy). Haarnoja's original computed both gradients from the same
        # parameters, which is what reusing `q1_all`/`q2_all` here did until 2026-09-20 (review).
        log_pi = network.log_softmax(self.actor(obs))
        with torch.no_grad():
            q_pi = combine(self.q1(obs), self.q2(obs), self.combine)
        entropy = -(log_pi.exp() * log_pi).sum(dim=1)
        mean_entropy = entropy.mean()
        penalty = entropy_penalty(old_entropy, entropy, self.entropy_penalty).mean()
        policy_loss = actor_loss(log_pi, q_pi, alpha).mean() + penalty
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()

        # -- the temperature
        temperature_loss = None
        if self.auto_alpha:
            temperature_loss = alpha_loss(self.log_alpha, mean_entropy, self.target_entropy)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.alpha_optimizer.step()

        self.train_step += 1
        self.maybe_update_target()
        td_error = (y - 0.5 * (q1_a + q2_a)).detach()
        self.last_metrics = {
            'alpha': self.alpha, 'entropy': float(mean_entropy.detach()),
            'critic_loss': float(critic_loss.detach()), 'actor_loss': float(policy_loss.detach()),
            'alpha_loss': None if temperature_loss is None else float(temperature_loss.detach()),
            'entropy_penalty': float(penalty.detach()),
            'clip_fraction': clip_wins, 'target_entropy': self.target_entropy,
            'train_step': self.train_step, 'mean_abs_td': float(td_error.abs().mean())}
        return td_error.cpu().numpy(), self.last_metrics

    def maybe_update_target(self):
        """Every `target_update_period` updates: a hard copy at tau 1.0, a Polyak step below it."""
        if self.target_update_period <= 0 or self.train_step % self.target_update_period:
            return False
        with torch.no_grad():
            for target, online in ((self.q1_target, self.q1), (self.q2_target, self.q2)):
                if self.tau >= 1.0:
                    target.load_state_dict(online.state_dict())
                else:
                    for lagged, live in zip(target.parameters(), online.parameters()):
                        lagged.mul_(1.0 - self.tau).add_(live, alpha=self.tau)
        return True

    # ---------------------------------------------------------------- persistence

    def state_dict(self):
        state = {'actor': self.actor.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
                 'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(),
                 'critic_optimizer': self.critic_optimizer.state_dict(),
                 'log_alpha': self.log_alpha.detach().clone(), 'train_step': self.train_step,
                 'torch_rng': self.torch_rng.get_state()}
        if self.alpha_optimizer is not None:
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.q1.load_state_dict(state['q1']); self.q2.load_state_dict(state['q2'])
        self.q1_target.load_state_dict(state['q1_target']); self.q2_target.load_state_dict(state['q2_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'].to(self.device))
        if self.alpha_optimizer is not None and 'alpha_optimizer' in state:
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        self.train_step = int(state.get('train_step', 0))
        # `previous_entropy` in a checkpoint from before 2026-09-20 is ignored: the penalty reads the replay.
        if state.get('torch_rng') is not None:
            self.torch_rng.set_state(state['torch_rng'].cpu())
