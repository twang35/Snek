"""Discrete SAC behind the seam `train.py` drives (`algos/dqn/algo.py` for the seam's members).

Where the answers differ from DQN's: the collector runs with the fork and the shield **off** and the
agent samples from pi, so `collector.step(epsilon)` is called with 0 and the agent ignores it; the replay
is `algos/dqn/replay.py` with the priority exponent at 0 by default (uniform, the papers'), and at 0.6 it is
the local plumbing's PER; the checkpoint is the actor alone, so `arch.json` gains no field and
`tools/restore.py` reads it as a PPO checkpoint. Every DQN, PPO and distributional knob is **refused by
name** (`algos/ppo/algo.py` §2 on why refusing beats ignoring), and SAC's own carry `SAC_`.

`sac` is Christodoulou 2019's agent at its paper values; `sac2` (`sac2.py`) is the same class with Zhou et
al. 2022's values as its defaults: alpha fixed at 0.05, entropy-penalty 0.5, double average Q with a
Q-clip of 0.5, Polyak 0.005, lr 1e-5, 3-step, 1e5 replay, 0.1 updates a move (`b-entropy.md` §2b).
"""

import os

from algos.dqn import collect
from algos.dqn.replay import PrioritizedReplay
from algos.sac.agent import SacAgent, COMBINES, CRITIC_LOSSES
from tools import checkpoints
from vectorized.vec_env import VecSnake

NAME = 'sac'

# One table of the two rows' defaults, so a `sac` spec can set any of B2's values and a `sac2` spec can
# put alpha back on `auto` for the ablation that asks whether the fixed temperature is itself a fix.
DEFAULTS = {
    'sac': {'lr': 3e-4, 'period': 2000, 'tau': 1.0, 'alpha': 'auto', 'ratio': 0.25, 'n_step': 1,
            'penalty': 0.0, 'combine': 'min', 'q_clip': 0.0, 'replay': 1000000, 'critic_loss': 'mse'},
    'sac2': {'lr': 1e-5, 'period': 1, 'tau': 0.005, 'alpha': '0.05', 'ratio': 0.1, 'n_step': 3,
             'penalty': 0.5, 'combine': 'avg', 'q_clip': 0.5, 'replay': 100000, 'critic_loss': 'mse'},
}

REJECTED = (
    'FORK_BRANCHES', 'FORK_PROB', 'FORK_MIN_LENGTH', 'FORK_MAX_STEPS',
    'INITIAL_EPSILON', 'MIN_EPSILON', 'GUIDED_FRACTION', 'EPSILON_SCHEDULE', 'EPSILON_ANNEAL_STEPS',
    'REPLAY_RATIO', 'REPLAY_BUFFER_MAX_LENGTH', 'INITIAL_COLLECT_STEPS',
    'PRIORITY_EXPONENT', 'IS_BETA', 'IS_BETA_FINAL', 'IS_WEIGHTS', 'BETA_ANNEAL_STEPS',
    'TARGET_UPDATE_PERIOD', 'TARGET_UPDATE_TAU', 'N_STEP_UPDATE',
    'LEARNING_RATE', 'BATCH_SIZE', 'ADAM_EPSILON', 'GRADIENT_CLIPPING',
    'MUNCHAUSEN_ALPHA', 'MUNCHAUSEN_TAU', 'MUNCHAUSEN_L0',
    'RESET_INTERVAL', 'RESET_ALPHA', 'RESET_STOP_AFTER',
    'RESET_ANNEAL_N_STEP', 'RESET_ANNEAL_GAMMA', 'RESET_ANNEAL_STEPS',
    'DIST_ATOMS', 'DIST_V_MIN', 'DIST_V_MAX', 'DIST_QUANTILES', 'DIST_TAU_SAMPLES',
    'DIST_TAU_PRIME_SAMPLES', 'DIST_POLICY_SAMPLES', 'DIST_EMBEDDING', 'DIST_KAPPA',
    'DIST_FRACTION_LR', 'DIST_FRACTION_ENTROPY', 'DIST_RISK_ALPHA', 'DIST_RISK_TRAIN',
    'PPO_ROLLOUT', 'PPO_EPOCHS', 'PPO_MINIBATCH', 'PPO_CLIP', 'PPO_CLIP_FINAL', 'PPO_GAE_LAMBDA',
    'PPO_GAE_LAMBDA_FINAL', 'PPO_DISCOUNT_FINAL', 'PPO_ENTROPY_COEF', 'PPO_ENTROPY_COEF_FINAL',
    'PPO_VF_COEF', 'PPO_LEARNING_RATE', 'PPO_LEARNING_RATE_FINAL', 'PPO_ANNEAL_FRACTION',
    'PPO_ADAM_EPSILON', 'PPO_TARGET_KL', 'PPO_GRADIENT_CLIPPING', 'PPO_NORMALIZE_ADV', 'PPO_VALUE_LOSS',
    'BBF_WEIGHT_DECAY', 'BBF_SPR_WEIGHT', 'BBF_SPR_STEPS', 'BBF_PROJECTION', 'BBF_TRANSITION_WIDTH', 'BBF_DUELING', 'BBF_DOUBLE',
)

# The knobs this module reads, for `algos/ppo/algo.py`'s refusal list and the docs.
SAC_KNOBS = (
    'SAC_LEARNING_RATE', 'SAC_CRITIC_LEARNING_RATE', 'SAC_ADAM_EPSILON', 'SAC_BATCH_SIZE',
    'SAC_TARGET_UPDATE_PERIOD', 'SAC_TAU', 'SAC_ALPHA', 'SAC_TARGET_ENTROPY_RATIO', 'SAC_INIT_ALPHA',
    'SAC_ALPHA_LEARNING_RATE', 'SAC_REPLAY_RATIO', 'SAC_N_STEP', 'SAC_ENTROPY_PENALTY',
    'SAC_CRITIC_COMBINE', 'SAC_Q_CLIP', 'SAC_REPLAY_BUFFER_MAX_LENGTH', 'SAC_PRIORITY_EXPONENT',
    'SAC_PREFILL', 'SAC_CRITIC_LOSS',
)


def _refuse_foreign_knobs(name):
    present = sorted(knob for knob in REJECTED if os.environ.get('SNEK_' + knob) is not None)
    if present:
        raise ValueError(
            'SNEK_ALGO={0} cannot honour {1}: {2}. SAC has no epsilon, shield, fork, Munchausen term, '
            'distributional head or PPO rollout, and its replay, target and optimiser knobs carry SAC_ '
            'names so a copied DQN command line cannot set them by accident.'.format(
                name, 'these knobs' if len(present) > 1 else 'this knob',
                ', '.join('SNEK_' + knob for knob in present)))


def build_config(tuned, name='sac'):
    """SAC's knobs. Every key is its `SNEK_` variable lowercased, as `train.py`'s contract requires."""
    _refuse_foreign_knobs(name)
    d = DEFAULTS[name]
    config = {
        # Shared with DQN and PPO, same meaning: lanes, and gamma.
        'collect_envs': int(tuned('COLLECT_ENVS', 16, int)),
        'discount': tuned('DISCOUNT', 0.99),
        'sac_learning_rate': tuned('SAC_LEARNING_RATE', d['lr']),
        'sac_critic_learning_rate': tuned('SAC_CRITIC_LEARNING_RATE', d['lr']),
        'sac_adam_epsilon': tuned('SAC_ADAM_EPSILON', 1e-8),
        'sac_batch_size': int(tuned('SAC_BATCH_SIZE', 64, int)),
        # Counted in gradient updates, as DQN's is. Hard copy at tau 1.0; `sac2` is Polyak 0.005 every update.
        'sac_target_update_period': int(tuned('SAC_TARGET_UPDATE_PERIOD', d['period'], int)),
        'sac_tau': tuned('SAC_TAU', d['tau']),
        # `auto` tunes the temperature toward `ratio * log |A|` from `init_alpha`; a number fixes it.
        'sac_alpha': str(tuned('SAC_ALPHA', d['alpha'], str)).strip(),
        'sac_target_entropy_ratio': tuned('SAC_TARGET_ENTROPY_RATIO', 0.98),
        'sac_init_alpha': tuned('SAC_INIT_ALPHA', 1.0),
        'sac_alpha_learning_rate': tuned('SAC_ALPHA_LEARNING_RATE', 3e-4),
        # Gradient updates per game move, carried as a fraction across steps as DQN's is.
        'sac_replay_ratio': tuned('SAC_REPLAY_RATIO', d['ratio']),
        'sac_n_step': int(tuned('SAC_N_STEP', d['n_step'], int)),
        'sac_entropy_penalty': tuned('SAC_ENTROPY_PENALTY', d['penalty']),
        'sac_critic_combine': str(tuned('SAC_CRITIC_COMBINE', d['combine'], str)).strip(),
        'sac_q_clip': tuned('SAC_Q_CLIP', d['q_clip']),
        # Both papers' critic loss is MSE (the Q-clip, when on, is on the squared error regardless);
        # `huber` is the local departure for the unclipped +100 terminal, the local cell's choice.
        'sac_critic_loss': str(tuned('SAC_CRITIC_LOSS', d['critic_loss'], str)).strip().lower(),
        'sac_replay_buffer_max_length': int(tuned('SAC_REPLAY_BUFFER_MAX_LENGTH', d['replay'], int)),
        # 0 is uniform replay, the papers'; 0.6 is the local plumbing's PER, with DQN's beta anneal.
        'sac_priority_exponent': tuned('SAC_PRIORITY_EXPONENT', 0.0),
        # Transitions banked from the untrained actor before the first update.
        'sac_prefill': int(tuned('SAC_PREFILL', 20000, int)),
    }
    if config['sac_critic_combine'] not in COMBINES:
        raise ValueError('SNEK_SAC_CRITIC_COMBINE={0!r} is not one of {1}'.format(
            config['sac_critic_combine'], COMBINES))
    if config['sac_critic_loss'] not in CRITIC_LOSSES:
        raise ValueError('SNEK_SAC_CRITIC_LOSS={0!r} is not one of {1}'.format(
            config['sac_critic_loss'], CRITIC_LOSSES))
    if config['sac_alpha'].lower() != 'auto':
        try:
            fixed = float(config['sac_alpha'])
        except ValueError:
            raise ValueError('SNEK_SAC_ALPHA={0!r} is neither "auto" nor a number'.format(config['sac_alpha']))
        if fixed <= 0.0:
            raise ValueError('SNEK_SAC_ALPHA={0} must be positive'.format(fixed))
    for key in ('sac_replay_ratio', 'sac_learning_rate', 'sac_critic_learning_rate', 'sac_init_alpha'):
        if config[key] <= 0.0:
            raise ValueError('SNEK_{0}={1} must be positive'.format(key.upper(), config[key]))
    for key in ('sac_q_clip', 'sac_entropy_penalty', 'sac_priority_exponent'):
        if config[key] < 0.0:
            raise ValueError('SNEK_{0}={1} must not be negative'.format(key.upper(), config[key]))
    if not 0.0 < config['sac_tau'] <= 1.0:
        raise ValueError('SNEK_SAC_TAU={0} is outside (0, 1]'.format(config['sac_tau']))
    if config['sac_target_update_period'] < 1 or config['sac_n_step'] < 1 or config['sac_batch_size'] < 1:
        raise ValueError('SNEK_SAC_TARGET_UPDATE_PERIOD, SNEK_SAC_N_STEP and SNEK_SAC_BATCH_SIZE must be at least 1')
    return config


def reportable(config):
    """Every key prints as a table row already; the entropy target is added because it is derived."""
    out = dict(config)
    import math
    out['sac_target_entropy'] = round(float(config['sac_target_entropy_ratio']) * math.log(3), 6)
    return out


def build(config, arch, device='cpu'):
    return SacAlgo(config, arch, device=device)


class SacAlgo(object):
    """The actor, two critics, DQN's replay and collector with the fork and the shield off."""

    step_granularity = 1

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        self.agent = SacAgent(arch, config, device=device)
        self.buffer = PrioritizedReplay(config['sac_replay_buffer_max_length'], arch['obs_len'],
                                        alpha=config['sac_priority_exponent'],
                                        initial_beta=0.4, final_beta=1.0, beta_anneal_steps=300000,
                                        seed=config['seed'])
        # **`shaping_discount` is the agent's gamma** -- `algos/dqn/algo.py` on the bias leaving it at 1.0 caused.
        self.collector = collect.Collector(
            VecSnake(config['collect_envs'], seed=config['seed'], shaping_discount=config['discount']),
            self.agent, self.buffer,
            discount=config['discount'], n_step=config['sac_n_step'],
            collect_envs=config['collect_envs'], fork=collect.ForkConfig(branches=1),
            guided_fraction=0.0, seed=config['seed'])
        self.gradient_debt = 0.0

    # ------------------------------------------------------------ what a checkpoint and an eval see

    @property
    def net(self):
        """**The actor, and only the actor** -- what `ckpt-<step>.pt` holds and stage B measures."""
        return self.agent.actor

    @property
    def policy_fn(self):
        return self.agent.policy_fn

    def describe(self):
        alpha = 'alpha auto -> H* {0:.3f}'.format(self.agent.target_entropy) if self.agent.auto_alpha \
            else 'alpha fixed {0}'.format(self.agent.alpha)
        fixes = []
        if self.agent.combine == 'avg':
            fixes.append('avg-Q')
        if self.agent.q_clip > 0.0:
            fixes.append('Q-clip {0}'.format(self.agent.q_clip))
        if self.agent.entropy_penalty > 0.0:
            fixes.append('entropy-penalty {0}'.format(self.agent.entropy_penalty))
        return '{0} lane(s), {1} updates a move, batch {2}, {3}{4}, critic {7}, replay {5:,}{6}'.format(
            self.collector.vec.n, self.config['sac_replay_ratio'], self.config['sac_batch_size'], alpha,
            (', ' + ', '.join(fixes)) if fixes else '', self.config['sac_replay_buffer_max_length'],
            ' PER {0}'.format(self.config['sac_priority_exponent']) if self.config['sac_priority_exponent'] > 0 else '',
            self.agent.critic_loss_kind)

    # ------------------------------------------------------------ the loop

    def prefill(self):
        banked = 0
        while self.buffer.size < self.config['sac_prefill']:
            banked += self.collector.step(0.0)
        return banked

    def advance(self):
        transitions = self.collector.step(0.0)
        self._learn(transitions)
        return 1, transitions

    def _learn(self, transitions):
        self.gradient_debt += transitions * self.config['sac_replay_ratio']
        while self.gradient_debt >= 1.0:
            self.gradient_debt -= 1.0
            drawn = self.buffer.sample(self.config['sac_batch_size'], self.agent.train_step)
            if drawn is None:
                continue
            batch, indexes, weights = drawn
            td_errors, _ = self.agent.update(batch, weights)
            if self.agent.use_is_weights:
                self.buffer.update_priorities(indexes, td_errors)

    # ------------------------------------------------------------ the row

    def fields(self):
        m = self.agent.last_metrics
        block = {key: _round(m.get(key), 5) for key in
                 ('entropy', 'critic_loss', 'actor_loss', 'alpha_loss', 'entropy_penalty', 'clip_fraction')}
        block['target_entropy'] = round(self.agent.target_entropy, 5)
        block['episodes'] = self.collector.counters['episodes']
        return {'alpha': round(self.agent.alpha, 6), 'sac': block}

    def on_eval(self, eval_rows, measured):
        """SAC has no eval-driven schedule; the temperature moves by gradient. Returns it for the row."""
        return {'alpha': round(self.agent.alpha, 6)}

    def log_note(self, row):
        block = row.get('sac') or {}
        return 'alpha {0:<8} H {1:<6}'.format(row.get('alpha', '?'), block.get('entropy', '?'))

    def log_extra(self, row):
        block = row.get('sac') or {}
        if not block:
            return []
        return ['           H* {0}  closs {1}  aloss {2}  pen {3}  clip {4}  episodes {5:,}'.format(
            block.get('target_entropy'), block.get('critic_loss'), block.get('actor_loss'),
            block.get('entropy_penalty'), block.get('clip_fraction'), block.get('episodes', 0))]

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'agent': self.agent.state_dict(), 'gradient_debt': float(self.gradient_debt)}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        self.gradient_debt = float(state.get('gradient_debt', 0.0))

    def init_from(self, source_dir, step):
        """The actor from another arm's checkpoint; critics, targets, temperature and buffer fresh."""
        checkpoint = checkpoints.path(source_dir, step)
        checkpoints.load(checkpoint, self.agent.actor, device=self.device)
        return 'actor from {0}; critics, temperature and buffer fresh'.format(checkpoint)

    def save_side_state(self, policy_dir):
        self.buffer.save(policy_dir)

    def load_side_state(self, policy_dir):
        return self.buffer.load(policy_dir)


def _round(value, places):
    return None if value is None else round(float(value), places)
