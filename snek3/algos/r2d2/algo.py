"""R2D2 -- Recurrent Replay Distributed DQN (Kapturowski et al. 2019) -- behind the seam `train.py` drives
(`algos/dqn/algo.py` for the seam's members). `SNEK_ALGO=r2d2`. `plans/algoExploration/e-memory.md` §2.

Built 2026-09-30 for Group E's E2: the paper's learner and its actor recipe on this game's 32 lanes in one
process. Its own package because nothing in `algos/dqn/` stores sequences: the replay holds windows over rows
with the recurrent state at each window's start (`replay.py`), the collector cuts an episode into 40-step
loss blocks (`collect.py`), the net carries an LSTM between a trunk and dueling streams (`net.py`), and the
agent burns in from the stored state before the loss (`agent.py`). What it shares: `QNet`'s initialiser,
Group A's C51 projection for the local recipe, the Ape-X ladder and the linear anneal in `algos/dqn/schedules.py`,
and every eval, chart and pass through `train.py`.

| knob | paper | notes |
|---|---|---|
| `SNEK_LEARNING_RATE`, `SNEK_ADAM_EPSILON` | 1e-4, 1e-3 | Adam |
| `SNEK_BATCH_SIZE` | 64 | windows per update |
| `SNEK_DISCOUNT`, `SNEK_N_STEP_UPDATE` | 0.997, 5 | n-step double-Q |
| `SNEK_TARGET_UPDATE_PERIOD` | 2500 | a hard copy, in updates |
| `SNEK_GRADIENT_CLIPPING` | 40 | |
| `SNEK_COLLECT_ENVS` | 256 actors; **32** here | lanes in one process |
| `SNEK_EPSILON_SCHEDULE` | `apex` | the ladder `base^(1 + alpha i/(N-1))` over the lanes; `linear` is one shared anneal |
| `SNEK_INITIAL_EPSILON`, `SNEK_APEX_ALPHA` | 0.4, 7 | the ladder's base and spread |
| `SNEK_MIN_EPSILON`, `SNEK_EPSILON_ANNEAL_STEPS` | -- | `linear` only |
| `SNEK_R2D2_SEQ_LENGTH`, `SNEK_R2D2_BURN_IN`, `SNEK_R2D2_STRIDE` | 120, 40, 40 | the stored window, its burn-in prefix and the step between consecutive windows' loss blocks: the paper's m = 80 loss steps with l = 40 burn-in, adjacent sequences overlapping by 40 (§2.3, §3) |
| `SNEK_R2D2_WINDOWS` | 1e5 sequences | the replay, in windows; a desktop arm is sized to its memory (the manifest says) |
| `SNEK_PRIORITY_EXPONENT`, `SNEK_R2D2_IS_BETA`, `SNEK_R2D2_PRIORITY_ETA` | 0.9, 0.6, 0.9 | PER alpha, the IS exponent (held), the max/mean mix |
| `SNEK_R2D2_RESCALE`, `SNEK_R2D2_RESCALE_EPS` | on, 1e-3 | `h()` on the targets; **off** for the C51 recipe |
| `SNEK_R2D2_HIDDEN`, `SNEK_R2D2_STREAM_WIDTH` | 512, 512 | the LSTM and the dueling streams |
| `SNEK_R2D2_RECURRENT` | `lstm` | `dense` is the paper's feed-forward control (`ffr2d2`): same loss positions, no state |
| `SNEK_R2D2_PREV_INPUT` | on | the previous action one-hot and reward into the cell |
| `SNEK_R2D2_HEAD` | `scalar` | `c51` is the local recipe, with `SNEK_DIST_ATOMS/V_MIN/V_MAX` |
| `SNEK_REPLAY_RATIO` | -- | **gradient steps per banked transition**, not the paper's replayed-rows-per-row; 1/32 is one update per lockstep of 32 lanes. A deliberate adaptation (§0): the paper's ratio over 256 actors has no analogue in one process |
| `SNEK_INITIAL_COLLECT_STEPS` | -- | moves before the first update; learning also waits for `batch_size` windows |

**Units.** A counted step is one lockstep of every lane: `collect_envs` moves, `collect_envs` rows.
"""

import os

from algos.dqn import schedules
from algos.r2d2 import knobs
from algos.r2d2 import net as network
from algos.r2d2.agent import R2d2Agent
from algos.r2d2.collect import Collector
from algos.r2d2.replay import SequenceReplay
from algos.bbf import algo as bbf_algo
from algos.sac import algo as sac_algo
from tools import checkpoints
from vectorized.vec_env import VecSnake

NAME = 'r2d2'

EPSILON_SCHEDULES = ('apex', 'linear')

REJECTED = (
    'FORK_BRANCHES', 'FORK_PROB', 'FORK_MIN_LENGTH', 'FORK_MAX_STEPS', 'GUIDED_FRACTION',
    'TARGET_UPDATE_TAU', 'IS_BETA', 'IS_BETA_FINAL', 'IS_WEIGHTS', 'BETA_ANNEAL_STEPS', 'IS_NORMALIZATION',
    'MUNCHAUSEN_ALPHA', 'MUNCHAUSEN_TAU', 'MUNCHAUSEN_L0',
    'RESET_INTERVAL', 'RESET_ALPHA', 'RESET_STOP_AFTER', 'RESET_ANNEAL_N_STEP', 'RESET_ANNEAL_GAMMA', 'RESET_ANNEAL_STEPS',
    'DIST_QUANTILES', 'DIST_TAU_SAMPLES', 'DIST_TAU_PRIME_SAMPLES', 'DIST_POLICY_SAMPLES', 'DIST_EMBEDDING',
    'DIST_KAPPA', 'DIST_FRACTION_LR', 'DIST_FRACTION_ENTROPY', 'DIST_RISK_ALPHA', 'DIST_RISK_TRAIN',
    'RAINBOW_HEAD', 'RAINBOW_NOISY', 'RAINBOW_NOISY_SIGMA', 'RAINBOW_DUELING', 'RAINBOW_DOUBLE',
    'RAINBOW_EPSILON_ZERO_AT', 'BTR_RESIDUAL', 'BTR_BLOCKS', 'BTR_SPECTRAL_NORM', 'BTR_LAYER_NORM',
    'PPO_RECURRENT', 'PPO_RECURRENT_HIDDEN', 'PPO_SEQ_MINIBATCH',
    'PPO_ROLLOUT', 'PPO_EPOCHS', 'PPO_MINIBATCH', 'PPO_CLIP', 'PPO_CLIP_FINAL', 'PPO_GAE_LAMBDA',
    'PPO_GAE_LAMBDA_FINAL', 'PPO_DISCOUNT_FINAL', 'PPO_ENTROPY_COEF', 'PPO_ENTROPY_COEF_FINAL',
    'PPO_VF_COEF', 'PPO_LEARNING_RATE', 'PPO_LEARNING_RATE_FINAL', 'PPO_ANNEAL_FRACTION',
    'PPO_ADAM_EPSILON', 'PPO_TARGET_KL', 'PPO_GRADIENT_CLIPPING', 'PPO_NORMALIZE_ADV', 'PPO_VALUE_LOSS',
) + bbf_algo.BBF_KNOBS + sac_algo.SAC_KNOBS


def _refuse_foreign_knobs():
    present = sorted(knob for knob in REJECTED if os.environ.get('SNEK_' + knob) is not None)
    if present:
        raise ValueError('SNEK_ALGO=r2d2 cannot honour {0}: {1}. R2D2 has no fork, shield, EMA target, IS-beta anneal, '
                         'Munchausen term, resets, quantile head, noisy net, PPO rollout, SPR or SAC temperature.'.format(
                             'these knobs' if len(present) > 1 else 'this knob', ', '.join('SNEK_' + knob for knob in present)))


def build_config(tuned):
    """The paper's values under the shared names plus `R2D2_`. Every key is its `SNEK_` variable lowercased."""
    _refuse_foreign_knobs()
    config = {
        'learning_rate': tuned('LEARNING_RATE', 1e-4),
        'adam_epsilon': tuned('ADAM_EPSILON', 1e-3),
        'batch_size': int(tuned('BATCH_SIZE', 64, int)),
        'discount': tuned('DISCOUNT', 0.997),
        'n_step_update': int(tuned('N_STEP_UPDATE', 5, int)),
        'target_update_period': int(tuned('TARGET_UPDATE_PERIOD', 2500, int)),
        'gradient_clipping': tuned('GRADIENT_CLIPPING', 40.0),
        'collect_envs': int(tuned('COLLECT_ENVS', 32, int)),
        'epsilon_schedule': str(tuned('EPSILON_SCHEDULE', 'apex', str)),
        'initial_epsilon': tuned('INITIAL_EPSILON', 0.4),
        'min_epsilon': tuned('MIN_EPSILON', 0.0),
        'epsilon_anneal_steps': int(tuned('EPSILON_ANNEAL_STEPS', 0, int)),
        'apex_alpha': tuned('APEX_ALPHA', 7.0),
        'replay_ratio': tuned('REPLAY_RATIO', 1.0 / 32.0),
        'initial_collect_steps': int(tuned('INITIAL_COLLECT_STEPS', 20000, int)),
        'priority_exponent': tuned('PRIORITY_EXPONENT', 0.9),
        'r2d2_is_beta': tuned('R2D2_IS_BETA', 0.6),
        'r2d2_priority_eta': tuned('R2D2_PRIORITY_ETA', 0.9),
        'r2d2_windows': int(tuned('R2D2_WINDOWS', 100000, int)),
        'r2d2_seq_length': int(tuned('R2D2_SEQ_LENGTH', 120, int)),
        'r2d2_burn_in': int(tuned('R2D2_BURN_IN', 40, int)),
        'r2d2_stride': int(tuned('R2D2_STRIDE', 40, int)),
        'r2d2_rescale': bool(int(tuned('R2D2_RESCALE', 1, int))),
        'r2d2_rescale_eps': tuned('R2D2_RESCALE_EPS', 1e-3),
        'r2d2_hidden': int(tuned('R2D2_HIDDEN', 512, int)),
        'r2d2_stream_width': int(tuned('R2D2_STREAM_WIDTH', 512, int)),
        'r2d2_recurrent': str(tuned('R2D2_RECURRENT', 'lstm', str)),
        'r2d2_prev_input': bool(int(tuned('R2D2_PREV_INPUT', 1, int))),
        'r2d2_head': str(tuned('R2D2_HEAD', 'scalar', str)),
        'dist_atoms': int(tuned('DIST_ATOMS', 51, int)),
        'dist_v_min': tuned('DIST_V_MIN', -10.0),
        'dist_v_max': tuned('DIST_V_MAX', 110.0),
    }
    if config['epsilon_schedule'] not in EPSILON_SCHEDULES:
        raise ValueError('SNEK_EPSILON_SCHEDULE={0!r} is not one of {1}'.format(config['epsilon_schedule'], EPSILON_SCHEDULES))
    if not 0.0 <= config['min_epsilon'] <= config['initial_epsilon'] <= 1.0:
        raise ValueError('SNEK_MIN_EPSILON={0} and SNEK_INITIAL_EPSILON={1} must satisfy 0 <= min <= initial <= 1'.format(
            config['min_epsilon'], config['initial_epsilon']))
    if config['epsilon_schedule'] == 'apex':
        schedules.apex_epsilons(config['collect_envs'], config['initial_epsilon'], config['apex_alpha'])
    if config['replay_ratio'] <= 0.0:
        raise ValueError('SNEK_REPLAY_RATIO={0} must be positive'.format(config['replay_ratio']))
    if config['n_step_update'] < 1:
        raise ValueError('SNEK_N_STEP_UPDATE={0} must be at least 1'.format(config['n_step_update']))
    if config['collect_envs'] < 1 or config['batch_size'] < 1:
        raise ValueError('SNEK_COLLECT_ENVS and SNEK_BATCH_SIZE must be at least 1')
    if not 0 <= config['r2d2_burn_in'] < config['r2d2_seq_length']:
        raise ValueError('SNEK_R2D2_BURN_IN={0} must be in [0, SNEK_R2D2_SEQ_LENGTH={1})'.format(
            config['r2d2_burn_in'], config['r2d2_seq_length']))
    if not 1 <= config['r2d2_stride'] <= config['r2d2_seq_length'] - config['r2d2_burn_in']:
        raise ValueError('SNEK_R2D2_STRIDE={0} must be in [1, the loss block {1}]'.format(
            config['r2d2_stride'], config['r2d2_seq_length'] - config['r2d2_burn_in']))
    if config['r2d2_windows'] < config['batch_size']:
        raise ValueError('SNEK_R2D2_WINDOWS={0} cannot hold a batch of {1}'.format(config['r2d2_windows'], config['batch_size']))
    if not 0.0 <= config['r2d2_priority_eta'] <= 1.0:
        raise ValueError('SNEK_R2D2_PRIORITY_ETA={0} must be in [0, 1]'.format(config['r2d2_priority_eta']))
    if config['r2d2_recurrent'] not in network.RECURRENT_KINDS:
        raise ValueError('SNEK_R2D2_RECURRENT={0!r} is not one of {1}'.format(config['r2d2_recurrent'], network.RECURRENT_KINDS))
    if config['r2d2_head'] not in network.HEAD_TYPES:
        raise ValueError('SNEK_R2D2_HEAD={0!r} is not one of {1}'.format(config['r2d2_head'], network.HEAD_TYPES))
    if config['r2d2_head'] == 'c51' and config['r2d2_rescale']:
        raise ValueError('SNEK_R2D2_HEAD=c51 runs with SNEK_R2D2_RESCALE=0: the categorical support is the return itself')
    if config['r2d2_hidden'] < 1 or config['r2d2_stream_width'] < 0:
        raise ValueError('SNEK_R2D2_HIDDEN must be at least 1 and SNEK_R2D2_STREAM_WIDTH at least 0')
    if config['dist_v_max'] <= config['dist_v_min']:
        raise ValueError('SNEK_DIST_V_MAX must exceed SNEK_DIST_V_MIN')
    return config


def arch_fields(config):
    """The sidecar's `recurrent` block (cell, width, streams, inputs) and the `head` for the C51 recipe."""
    out = {'recurrent': {'type': config['r2d2_recurrent'], 'hidden': int(config['r2d2_hidden']),
                         'stream_width': int(config['r2d2_stream_width']), 'prev_input': bool(config['r2d2_prev_input'])}}
    if config['r2d2_head'] == 'c51':
        out['head'] = {'type': 'c51', 'atoms': int(config['dist_atoms']), 'v_min': float(config['dist_v_min']),
                       'v_max': float(config['dist_v_max'])}
    else:
        out['head'] = {'type': 'scalar'}
    return out


def reportable(config):
    return dict(config)


class R2d2Algo(object):

    step_granularity = 1

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        block = config['r2d2_seq_length'] - config['r2d2_burn_in']
        self.agent = R2d2Agent(arch, learning_rate=config['learning_rate'], adam_epsilon=config['adam_epsilon'],
                               gradient_clipping=config['gradient_clipping'], target_update_period=config['target_update_period'],
                               discount=config['discount'], n_step=config['n_step_update'], burn_in=config['r2d2_burn_in'],
                               block=block, rescale_targets=config['r2d2_rescale'], rescale_eps=config['r2d2_rescale_eps'],
                               priority_eta=config['r2d2_priority_eta'], seed=config['seed'], device=device)
        self.buffer = SequenceReplay(config['r2d2_windows'], arch['obs_len'], config['collect_envs'], self.agent.state_width,
                                     seq_length=config['r2d2_seq_length'], burn_in=config['r2d2_burn_in'],
                                     lookahead=config['n_step_update'], alpha=config['priority_exponent'],
                                     beta=config['r2d2_is_beta'], eta=config['r2d2_priority_eta'], seed=config['seed'])
        self.collector = Collector(VecSnake(config['collect_envs'], seed=config['seed'], shaping_discount=config['discount']),
                                   self.agent, self.buffer, self._epsilons(0), stride=config['r2d2_stride'], seed=config['seed'])
        self.moves = 0
        self.gradient_debt = 0.0

    # ------------------------------------------------------------ exploration

    def _epsilons(self, moves):
        c = self.config
        if c['epsilon_schedule'] == 'apex':
            return schedules.apex_epsilons(c['collect_envs'], c['initial_epsilon'], c['apex_alpha'])
        shared = schedules.linear_epsilon(moves, c['initial_epsilon'], c['min_epsilon'], c['epsilon_anneal_steps'])
        return [shared] * c['collect_envs']

    @property
    def epsilon(self):
        """One number for the row: the ladder's base (its most exploratory lane) or the shared value."""
        return float(self.collector.epsilons.max())

    # ------------------------------------------------------------ what a checkpoint and an eval see

    @property
    def net(self):
        return self.agent.net

    @property
    def policy_fn(self):
        return self.agent.policy_fn

    def describe(self):
        c = self.config
        cell = 'LSTM {0}'.format(c['r2d2_hidden']) if c['r2d2_recurrent'] == 'lstm' else 'dense {0} (feed-forward control)'.format(c['r2d2_hidden'])
        parts = ['r2d2', '{0} lane(s)'.format(c['collect_envs']), cell,
                 'dueling streams {0}'.format(c['r2d2_stream_width']) if c['r2d2_stream_width'] else 'dueling, one linear per stream',
                 'C51 {0} atoms'.format(c['dist_atoms']) if c['r2d2_head'] == 'c51' else 'scalar Q',
                 'prev action+reward in' if c['r2d2_prev_input'] else 'no prev inputs',
                 'windows {0} = {1} burn-in + {2} loss, stride {3}, n-step {4}'.format(
                     c['r2d2_seq_length'], c['r2d2_burn_in'], c['r2d2_seq_length'] - c['r2d2_burn_in'], c['r2d2_stride'], c['n_step_update']),
                 'replay {0:,} windows'.format(c['r2d2_windows']), 'batch {0}'.format(c['batch_size']),
                 'replay ratio {0:.5g}'.format(c['replay_ratio']),
                 'PER alpha {0} beta {1} eta {2}'.format(c['priority_exponent'], c['r2d2_is_beta'], c['r2d2_priority_eta']),
                 'rescale h() eps {0}'.format(c['r2d2_rescale_eps']) if c['r2d2_rescale'] else 'no rescaling',
                 'Adam lr {0} eps {1}'.format(c['learning_rate'], c['adam_epsilon']),
                 'target copy every {0:,} updates'.format(c['target_update_period']), 'clip {0}'.format(c['gradient_clipping'])]
        if c['epsilon_schedule'] == 'apex':
            ladder = self._epsilons(0)
            parts.append('epsilon ladder {0:.3g} .. {1:.3g} over the lanes (alpha {2})'.format(ladder[0], ladder[-1], c['apex_alpha']))
        else:
            parts.append('epsilon linear {0} -> {1} over {2:,} moves'.format(c['initial_epsilon'], c['min_epsilon'], c['epsilon_anneal_steps']))
        return ', '.join(parts)

    # ------------------------------------------------------------ the loop

    def prefill(self):
        """Moves until `initial_collect_steps` rows are banked **and** a batch of windows exists."""
        banked = 0
        while self.buffer.rows < self.config['initial_collect_steps'] or self.buffer.n_windows < self.config['batch_size']:
            banked += self.collector.step()
        return banked

    def advance(self):
        if self.config['epsilon_schedule'] == 'linear':
            self.collector.set_epsilons(self._epsilons(self.moves))
        transitions = self.collector.step()
        self.moves += int(transitions)
        self._learn(transitions)
        return 1, transitions

    def _learn(self, transitions):
        self.gradient_debt += transitions * self.config['replay_ratio']
        while self.gradient_debt >= 1.0:
            self.gradient_debt -= 1.0
            drawn = self.buffer.sample(self.config['batch_size'])
            if drawn is None:
                continue
            batch, slots, weights = drawn
            priorities, _ = self.agent.update(batch, weights)
            self.buffer.update_priorities(slots, priorities)

    # ------------------------------------------------------------ the row

    def fields(self):
        out = {'epsilon': round(self.epsilon, 5),
               'r2d2': {'train_step': int(self.agent.train_step), 'windows': int(self.buffer.n_windows),
                        'pending_windows': int(self.collector.snapshot()['pending_windows'])}}
        for key in ('td_loss', 'grad_norm'):
            if key in self.agent.last:
                out['r2d2'][key] = round(float(self.agent.last[key]), 5)
        return out

    def on_eval(self, eval_rows, measured):
        return {'epsilon': round(self.epsilon, 5)}

    def log_note(self, row):
        return 'eps {0:<7}'.format(row.get('epsilon', '?'))

    def log_extra(self, row):
        r = row.get('r2d2') or {}
        if not r:
            return []
        return ['           r2d2 updates {0:,}  td {1}  grad {2}  windows {3:,} (+{4} pending)'.format(
            r.get('train_step', 0), r.get('td_loss', '?'), r.get('grad_norm', '?'), r.get('windows', 0), r.get('pending_windows', 0))]

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'agent': self.agent.state_dict(), 'moves': int(self.moves), 'gradient_debt': float(self.gradient_debt)}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        self.moves = int(state.get('moves', 0))
        self.gradient_debt = float(state.get('gradient_debt', 0.0))

    def init_from(self, source_dir, step):
        checkpoint = checkpoints.path(source_dir, step)
        checkpoints.load(checkpoint, self.agent.net, device=self.device)
        self.agent.target.load_state_dict(self.agent.net.state_dict())
        return 'net and target from {0}'.format(checkpoint)

    def save_side_state(self, policy_dir):
        self.buffer.save(policy_dir)

    def load_side_state(self, policy_dir):
        return self.buffer.load(policy_dir)


def build(config, arch, device='cpu'):
    return R2d2Algo(config, arch, device=device)
