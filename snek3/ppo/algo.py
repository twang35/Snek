"""PPO behind the seam `train.py` drives. The mirror of [`../dqn/algo.py`](../dqn/algo.py).

Read that file for the seam's fourteen members and why one `train.py` serves both algorithms. What is
worth stating here is the three places PPO's answers differ, because each is a decision rather than a
detail.

## 1. A step is a transition, so a PPO step number is a game-move count

`step_granularity` is `collect_envs * ppo_rollout` — 16,384 at the defaults — and `advance()` returns
that number as *both* the step increment and the transition count. `dqn/algo.py` returns `(1, width)`.

The consequence is the one [`../docs/findings.md`](../docs/findings.md) exists to prevent being
misread: **a PPO step number is directly comparable to a snek2 step number**, and to a snek3 DQN step
number after the documented 4x. `SNEK_MAX_STEPS` on a PPO arm is a budget in game moves.

`train.py` rounds the eval and checkpoint intervals up to whole algorithm steps, so the default
`SNEK_EVAL_INTERVAL=1000` becomes one eval per rollout and the "every checkpoint is screened"
invariant holds without a PPO-specific knob. Set `SNEK_EVAL_INTERVAL` to a multiple of the rollout
size to evaluate less often; it is rounded up either way, so it cannot land off the grid.

## 2. The DQN-only knobs are refused, not ignored

`SNEK_FORK_*`, `SNEK_INITIAL_EPSILON`, `SNEK_MIN_EPSILON`, `SNEK_GUIDED_FRACTION`, `SNEK_REPLAY_*`,
`SNEK_PRIORITY_EXPONENT`, `SNEK_IS_*`, `SNEK_BETA_ANNEAL_STEPS`, `SNEK_TARGET_UPDATE_*`,
`SNEK_N_STEP_UPDATE`, `SNEK_LEARNING_RATE`, `SNEK_BATCH_SIZE`, `SNEK_ADAM_EPSILON`,
`SNEK_GRADIENT_CLIPPING`, `SNEK_INITIAL_COLLECT_STEPS` all raise.

This project's rule is that an override which quietly does something else is worse than one that
refuses — and here the cost of ignoring is concrete: a PPO arm launched from a copied DQN command line
would silently take DQN's `1e-5` learning rate, which is ~64x too little total parameter movement, and
report "PPO does not learn". That is why the learning rate and the batch size have `PPO_` names at all.

## 3. There is no prefill and no replay buffer

`prefill()` returns 0 and `save_side_state` / `load_side_state` do nothing: the first rollout *is* the
first batch, and there is no buffer beside the weights to keep in step with them. `load_side_state`
returns False, which is what makes the trainer's resume line say "buffer empty" rather than claim one
came back.
"""

import os

from ppo import collect
from ppo import rollout as rollout_module
from ppo import schedules
from ppo.agent import PpoAgent
from vectorized.vec_env import VecSnake

NAME = 'ppo'

# Knobs that belong to DQN and cannot mean anything here. Refused rather than ignored — see §2.
# `SNEK_COLLECT_ENVS` and `SNEK_DISCOUNT` are deliberately *not* in this list: both mean the same
# thing to both algorithms, and sharing the name keeps "lanes" and "gamma" one concept each.
REJECTED = (
    'FORK_BRANCHES', 'FORK_PROB', 'FORK_MIN_LENGTH', 'FORK_MAX_STEPS',
    'INITIAL_EPSILON', 'MIN_EPSILON', 'GUIDED_FRACTION',
    'REPLAY_RATIO', 'REPLAY_BUFFER_MAX_LENGTH', 'INITIAL_COLLECT_STEPS',
    'PRIORITY_EXPONENT', 'IS_BETA', 'IS_BETA_FINAL', 'IS_WEIGHTS', 'BETA_ANNEAL_STEPS',
    'TARGET_UPDATE_PERIOD', 'TARGET_UPDATE_TAU', 'N_STEP_UPDATE',
    'LEARNING_RATE', 'BATCH_SIZE', 'ADAM_EPSILON', 'GRADIENT_CLIPPING',
)


def _refuse_dqn_knobs():
    """Raises if the environment sets a knob this algorithm cannot honour.

    Reads `os.environ` directly rather than going through `tuned()`, because the point is to catch a
    knob that is *set* — `tuned()` would read it and return a value, which is the ignoring this
    prevents.
    """
    present = sorted(name for name in REJECTED if os.environ.get('SNEK_' + name) is not None)
    if present:
        raise ValueError(
            'SNEK_ALGO=ppo cannot honour {0}: {1}. Those are DQN knobs — the replay, the target '
            'network, the epsilon schedule, the shield and the forking collector have no PPO '
            'analogue, and the learning rate and batch size have PPO_ names because DQN\'s values '
            'are ~64x too small for an algorithm that takes 64x fewer gradient steps. Drop them or '
            'drop SNEK_ALGO=ppo.'.format(
                'these knobs' if len(present) > 1 else 'this knob',
                ', '.join('SNEK_' + name for name in present)))


def build_config(tuned):
    """PPO's knobs. Every key is its `SNEK_` variable lowercased, as `train.py`'s contract requires."""
    _refuse_dqn_knobs()
    config = {
        # Shared with DQN, same meaning, **different default.** 128 lanes rather than 1: `VecSnake.step`
        # costs 536 us at one lane and 950 us at 64, so almost all of it is per-call numpy overhead and
        # width is very nearly free. This is the knob that makes PPO worth running on this env.
        'collect_envs': int(tuned('COLLECT_ENVS', 128, int)),
        'discount': tuned('DISCOUNT', 0.99),
        'ppo_rollout': int(tuned('PPO_ROLLOUT', 128, int)),
        'ppo_epochs': int(tuned('PPO_EPOCHS', 4, int)),
        'ppo_minibatch': int(tuned('PPO_MINIBATCH', 256, int)),
        'ppo_clip': tuned('PPO_CLIP', 0.2),
        # 0.98, not the conventional 0.95. `1/(1 - gamma*lambda)` is the steps an advantage sees: 19 at
        # 0.95 and gamma 0.9975, 44.5 at 0.98, 400 at 1.0. See `rollout.py`: none of them reaches
        # the +100, which is why the critic carries it and the shaping terms matter more here.
        'ppo_gae_lambda': tuned('PPO_GAE_LAMBDA', 0.98),
        'ppo_entropy_coef': tuned('PPO_ENTROPY_COEF', 0.01),
        # Absent means "no anneal", which is p1's setting. `tuned` needs a sentinel rather than None
        # because it casts, so an empty string is the spelling for absent.
        'ppo_entropy_coef_final': _optional_float(tuned('PPO_ENTROPY_COEF_FINAL', '', str)),
        'ppo_vf_coef': tuned('PPO_VF_COEF', 0.5),
        'ppo_learning_rate': tuned('PPO_LEARNING_RATE', 3e-4),
        'ppo_adam_epsilon': tuned('PPO_ADAM_EPSILON', 1e-7),
        'ppo_target_kl': tuned('PPO_TARGET_KL', 0.0),
        'ppo_gradient_clipping': tuned('PPO_GRADIENT_CLIPPING', 0.5),
        'ppo_normalize_adv': bool(int(tuned('PPO_NORMALIZE_ADV', 1, int))),
        'ppo_value_loss': str(tuned('PPO_VALUE_LOSS', 'huber', str)),
    }
    if config['ppo_minibatch'] > config['collect_envs'] * config['ppo_rollout']:
        raise ValueError(
            'SNEK_PPO_MINIBATCH={0} is larger than a whole rollout ({1} lanes x {2} steps = {3}). '
            'Every epoch would be one partial batch, so "{4} epochs" would mean {4} gradient steps '
            'rather than {4} passes.'.format(
                config['ppo_minibatch'], config['collect_envs'], config['ppo_rollout'],
                config['collect_envs'] * config['ppo_rollout'], config['ppo_epochs']))
    if config['ppo_epochs'] < 1:
        raise ValueError('SNEK_PPO_EPOCHS={0} must be at least 1; 0 would collect and never '
                         'learn'.format(config['ppo_epochs']))
    if not 0.0 < config['ppo_clip'] < 1.0:
        raise ValueError('SNEK_PPO_CLIP={0} is outside (0, 1). At 0 no update is ever allowed '
                         'through; at 1 or above the lower bound reaches a ratio of 0 and the clip '
                         'stops being a trust region.'.format(config['ppo_clip']))
    if not 0.0 <= config['ppo_gae_lambda'] <= 1.0:
        raise ValueError('SNEK_PPO_GAE_LAMBDA={0} is not in [0, 1]'.format(
            config['ppo_gae_lambda']))
    return config


def _optional_float(raw):
    """`''` -> None, anything else -> float. The spelling for an absent numeric knob."""
    raw = str(raw).strip()
    return None if not raw else float(raw)


def reportable(config):
    """Every key prints as a table row already; the horizon is added because it is derived.

    `ppo_horizon` has no knob and is the one number in the block that explains the other two — a
    reader checking whether lambda was set sensibly should not have to compute `1/(1 - gamma*lambda)`
    by hand.
    """
    out = dict(config)
    out['ppo_horizon'] = round(rollout_module.horizon(config['discount'],
                                                     config['ppo_gae_lambda']), 1)
    out['ppo_transitions_per_rollout'] = config['collect_envs'] * config['ppo_rollout']
    return out


def build(config, arch, device='cpu'):
    return PpoAlgo(config, arch, device=device)


class PpoAlgo(object):
    """The actor, the critic, one rollout buffer, and the collector that fills it."""

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        self.agent = PpoAgent(arch, config, device=device)
        self.rollout = rollout_module.Rollout(config['ppo_rollout'], config['collect_envs'],
                                              arch['obs_len'])
        # **`shaping_discount` is the agent's gamma, and passing it is not optional** — see
        # `dqn/algo.py` for the full note and the 2.5e-4-a-step bias that leaving it at 1.0 caused.
        self.collector = collect.Collector(
            VecSnake(config['collect_envs'], seed=config['seed'],
                     shaping_discount=config['discount']),
            self.agent, self.rollout)
        self.step = 0
        self.last_metrics = {}

    # ------------------------------------------------------------ the seam

    @property
    def step_granularity(self):
        """Transitions in one rollout, which is one `advance()` and one step increment."""
        return self.rollout.size

    @property
    def net(self):
        """**The actor, and only the actor.** That is what `ckpt-<step>.pt` holds and what stage B
        measures: the critic is a training aid with no place in a policy checkpoint, and keeping it
        out is what lets `arch.json` describe a PPO checkpoint with no new field."""
        return self.agent.actor

    @property
    def policy_fn(self):
        return self.agent.policy_fn

    def describe(self):
        return ('{0} lane(s) x {1} rollout = {2:,} transitions, {3} epoch(s), minibatch {4}, '
                'GAE horizon {5:.0f} steps'.format(
                    self.config['collect_envs'], self.config['ppo_rollout'], self.rollout.size,
                    self.config['ppo_epochs'], self.config['ppo_minibatch'],
                    rollout_module.horizon(self.config['discount'], self.config['ppo_gae_lambda'])))

    def prefill(self):
        """Nothing to pre-fill: the first rollout is the first batch."""
        return 0

    def advance(self):
        """One rollout, then its epochs. Returns `(steps, transitions)` — **the same number.**

        Equal because a PPO step *is* a game move: nothing here is conditional on an episode boundary,
        unlike a DQN step where a terminal transition emits a whole n-step window. That equality is
        the whole reason a PPO step count can be read against a snek2 one.
        """
        transitions = self.collector.collect()
        self.step += transitions
        # Read before the update rather than after, so the coefficient the epochs use is the one this
        # rollout's step number implies. Constant at p1's settings, where `final` is absent.
        self.agent.entropy_coef = schedules.entropy_coef_for(
            self.step, self.config['max_steps'], self.config['ppo_entropy_coef'],
            self.config['ppo_entropy_coef_final'])
        self.last_metrics = self.agent.update(self.rollout)
        return transitions, transitions

    # ------------------------------------------------------------ the row

    def fields(self):
        """What the arm ran under, captured at the step being evaluated.

        The `ppo` block is the analogue of a DQN row's `fork` block: a window of training that is over
        and cannot be reconstructed once the queue lands the measurement several intervals later.
        """
        metrics = self.last_metrics
        block = {'entropy': _round(metrics.get('entropy'), 4),
                 'approx_kl': _round(metrics.get('approx_kl'), 6),
                 'clip_fraction': _round(metrics.get('clip_fraction'), 4),
                 'explained_variance': _round(metrics.get('explained_variance'), 4),
                 'policy_loss': _round(metrics.get('policy_loss'), 5),
                 'value_loss': _round(metrics.get('value_loss'), 4),
                 'epochs_run': metrics.get('epochs_run'),
                 'stopped_early': metrics.get('stopped_early')}
        block.update(self.collector.snapshot())
        return {'entropy_coef': round(float(self.agent.entropy_coef), 6), 'ppo': block}

    def on_eval(self, eval_rows, measured):
        """PPO's schedule is a function of the step, so this has nothing to advance.

        It still exists and still returns the coefficient, because `train.py` calls it in both queue
        modes and the row needs a value either way. **Returning the live coefficient rather than
        recomputing it** is what keeps the queued and unqueued spellings the same number here: unlike
        DQN's epsilon, PPO's coefficient does not move at an eval, so there is no one-row shift to
        document.
        """
        return {'entropy_coef': round(float(self.agent.entropy_coef), 6)}

    def log_note(self, row):
        block = row.get('ppo') or {}
        return 'ent {0:<6} kl {1:<8}'.format(row.get('entropy_coef', 0.0),
                                             block.get('approx_kl', 0.0))

    def log_extra(self, row):
        block = row.get('ppo') or {}
        if not block:
            return []
        return ['           entropy {0}  clip {1}  ev {2}  vloss {3}  episodes {4:,}'.format(
            block.get('entropy'), block.get('clip_fraction'),
            block.get('explained_variance'), block.get('value_loss'),
            block.get('episodes', 0))]

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'agent': self.agent.state_dict(), 'step': int(self.step)}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        # The algorithm's own step, which the entropy ramp reads. `train.py` restores its own
        # `self.step` separately and they agree; this one is kept so the ramp survives a resume even
        # if the two ever diverge.
        self.step = int(state.get('step', 0))

    def save_side_state(self, policy_dir):
        """Nothing beside the weights: there is no replay buffer."""

    def load_side_state(self, policy_dir):
        """False, so the resume line says "buffer empty" rather than claiming one came back."""
        return False


def _round(value, places):
    return None if value is None else round(float(value), places)
