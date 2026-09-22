"""BBF -- Bigger, Better, Faster (Schwarzer et al. 2023) -- behind the seam `train.py` drives
(`algos/dqn/algo.py` for the seam's members). `SNEK_ALGO=bbf`.

**This is the paper's recipe, whole, on the shared game and the shared measurement**, built 2026-09-20 to read
how BBF does on Snake; it is expected to run a few batches and then rest (`plans/algoExploration/d-data-efficiency.md`
§7). It is its own package and not a subclass of `DqnAlgo` because two of its pieces do not fit the shared path:
the n-step return is computed at **sample time** from a sequential replay (`replay.py`), so the anneal of n and
gamma reaches every drawn transition at once, as the paper's does; and SPR needs the K observations after each
state. What it shares: the collector (`algos/dqn/collect.py`, at `n_step` 1, fork and shield off), the Rainbow
network (`algos/rainbow/net.py`), Group A's C51 losses, the reset and cycle schedules (`algos/dqn/resets.py`), and
every eval, chart and pass through `train.py`.

Knob names: the ones DQN owns keep DQN's names where the meaning is the same (`SNEK_LEARNING_RATE`, `SNEK_BATCH_SIZE`,
`SNEK_REPLAY_RATIO`, `SNEK_RESET_*`, `SNEK_DIST_ATOMS`, ...), and the defaults are the paper's; what is BBF's carries
`BBF_`. A knob this algorithm cannot honour is refused by name (`REJECTED`). Units: moves for the epsilon anneal and
the prefill, gradient steps for the reset and the cycle, transitions for the replay.

| knob | paper | notes |
|---|---|---|
| `SNEK_LEARNING_RATE`, `SNEK_ADAM_EPSILON`, `SNEK_BBF_WEIGHT_DECAY` | 1e-4, 1.5e-4, 0.1 | AdamW over the Q net and the SPR heads |
| `SNEK_BATCH_SIZE`, `SNEK_REPLAY_RATIO` | 32, 8 | gradient steps per game move, carried as a fraction |
| `SNEK_DISCOUNT`, `SNEK_N_STEP_UPDATE` | 0.997, 3 | the **end** of each cycle; the anneal starts each cycle at `SNEK_RESET_ANNEAL_GAMMA`'s / `_N_STEP`'s first value |
| `SNEK_RESET_INTERVAL`, `_ALPHA`, `_STOP_AFTER` | 40,000, 0.5, 0 | shrink-and-perturb, in gradient steps |
| `SNEK_RESET_ANNEAL_N_STEP`, `_GAMMA`, `_STEPS` | `10,3`, `0.97,0.997`, 10,000 | the within-cycle anneal; empty strings turn it off (constant n and gamma) |
| `SNEK_TARGET_UPDATE_TAU` | 0.005 | EMA every gradient step |
| `SNEK_GRADIENT_CLIPPING` | 10 | |
| `SNEK_INITIAL_EPSILON`, `SNEK_MIN_EPSILON`, `SNEK_EPSILON_ANNEAL_STEPS` | 1, 0, 2001 | linear in moves after the prefill, then held. **No hard floor**: the paper trains at epsilon 0 |
| `SNEK_INITIAL_COLLECT_STEPS`, `SNEK_REPLAY_BUFFER_MAX_LENGTH`, `SNEK_PRIORITY_EXPONENT` | 2,000, 1,000,000, 0.5 | the prefill (at epsilon 1), the replay, PER |
| `SNEK_COLLECT_ENVS` | 1 | lanes; the replay keeps each lane's sequence |
| `SNEK_DIST_ATOMS`, `SNEK_DIST_V_MIN`, `SNEK_DIST_V_MAX` | 51, -10, 110 | the support, this game's return range (`plans/algoExploration/README.md`) |
| `SNEK_BBF_SPR_WEIGHT`, `SNEK_BBF_SPR_STEPS`, `SNEK_BBF_PROJECTION` | 5, 5, 512 | the SPR loss weight, K, and the projection width (BBF's is the head's first layer; ours is a linear of this width) |
| `SNEK_BBF_TRANSITION_WIDTH` | 256 | the transition model's hidden width; 0 is the latent's own. BBF's transition model is two 64-channel convolutions beside an encoder many times their size, and at the latent's width (2048) it was 8x the whole step's cost here |
| `SNEK_BBF_DUELING`, `SNEK_BBF_DOUBLE` | 1, 1 | the value stream; the target action from the online net |
| `SNEK_FC_LAYERS` | `1280,2048` for the paper cell | the plan's MLP analogue of the x4 IMPALA encoder; `train.py`'s knob |

**The reward/discount coupling under a sample-time anneal.** The env's `shaping_discount` follows the cycle's gamma
at collection time, which is exact when the potential-based shaping terms are off (`SNEK_CHASE_SAFE_SHAPING=0`, the
paper cell) and approximate otherwise: a transition collected under 0.97 can be drawn into a 0.997 target. The
paper cell runs shaping off for that reason, and a spec that turns it on accepts the approximation.
"""

import os

from algos.bbf.agent import BbfAgent
from algos.bbf.replay import SequentialReplay
from algos.dqn import collect
from algos.dqn import resets
from algos.dqn import schedules
from algos.rainbow import algo as rainbow_algo
from algos.sac import algo as sac_algo
from tools import checkpoints
from vectorized.vec_env import VecSnake

NAME = 'bbf'

# The knobs this module reads, for the other algorithms' refusal lists and the docs.
BBF_KNOBS = ('BBF_WEIGHT_DECAY', 'BBF_SPR_WEIGHT', 'BBF_SPR_STEPS', 'BBF_PROJECTION', 'BBF_TRANSITION_WIDTH',
             'BBF_DUELING', 'BBF_DOUBLE')

REJECTED = (
    'FORK_BRANCHES', 'FORK_PROB', 'FORK_MIN_LENGTH', 'FORK_MAX_STEPS', 'GUIDED_FRACTION', 'EPSILON_SCHEDULE',
    'TARGET_UPDATE_PERIOD', 'IS_BETA', 'IS_BETA_FINAL', 'IS_WEIGHTS', 'BETA_ANNEAL_STEPS',
    'MUNCHAUSEN_ALPHA', 'MUNCHAUSEN_TAU', 'MUNCHAUSEN_L0',
    'DIST_QUANTILES', 'DIST_TAU_SAMPLES', 'DIST_TAU_PRIME_SAMPLES', 'DIST_POLICY_SAMPLES', 'DIST_EMBEDDING',
    'DIST_KAPPA', 'DIST_FRACTION_LR', 'DIST_FRACTION_ENTROPY', 'DIST_RISK_ALPHA', 'DIST_RISK_TRAIN',
    'RAINBOW_HEAD', 'RAINBOW_NOISY', 'RAINBOW_NOISY_SIGMA', 'RAINBOW_DUELING', 'RAINBOW_DOUBLE',
    'RAINBOW_EPSILON_ZERO_AT', 'BTR_RESIDUAL', 'BTR_BLOCKS', 'BTR_SPECTRAL_NORM', 'BTR_LAYER_NORM',
    'PPO_ROLLOUT', 'PPO_EPOCHS', 'PPO_MINIBATCH', 'PPO_CLIP', 'PPO_CLIP_FINAL', 'PPO_GAE_LAMBDA',
    'PPO_GAE_LAMBDA_FINAL', 'PPO_DISCOUNT_FINAL', 'PPO_ENTROPY_COEF', 'PPO_ENTROPY_COEF_FINAL',
    'PPO_VF_COEF', 'PPO_LEARNING_RATE', 'PPO_LEARNING_RATE_FINAL', 'PPO_ANNEAL_FRACTION',
    'PPO_ADAM_EPSILON', 'PPO_TARGET_KL', 'PPO_GRADIENT_CLIPPING', 'PPO_NORMALIZE_ADV', 'PPO_VALUE_LOSS',
) + sac_algo.SAC_KNOBS


def _refuse_foreign_knobs():
    present = sorted(knob for knob in REJECTED if os.environ.get('SNEK_' + knob) is not None)
    if present:
        raise ValueError('SNEK_ALGO=bbf cannot honour {0}: {1}. BBF has no fork, shield, eval-driven epsilon, hard '
                         'target period, IS-beta anneal, Munchausen term, quantile head, noisy net, PPO rollout or '
                         'SAC temperature.'.format('these knobs' if len(present) > 1 else 'this knob',
                                                    ', '.join('SNEK_' + knob for knob in present)))


def build_config(tuned):
    """The paper's values under the shared names plus `BBF_`. Every key is its `SNEK_` variable lowercased."""
    _refuse_foreign_knobs()
    config = {
        'learning_rate': tuned('LEARNING_RATE', 1e-4),
        'adam_epsilon': tuned('ADAM_EPSILON', 1.5e-4),
        'batch_size': int(tuned('BATCH_SIZE', 32, int)),
        'discount': tuned('DISCOUNT', 0.997),
        'n_step_update': int(tuned('N_STEP_UPDATE', 3, int)),
        'target_update_tau': tuned('TARGET_UPDATE_TAU', 0.005),
        'gradient_clipping': tuned('GRADIENT_CLIPPING', 10.0),
        'initial_epsilon': tuned('INITIAL_EPSILON', 1.0),
        'min_epsilon': tuned('MIN_EPSILON', 0.0),
        'epsilon_anneal_steps': int(tuned('EPSILON_ANNEAL_STEPS', 2001, int)),
        'collect_envs': int(tuned('COLLECT_ENVS', 1, int)),
        'replay_ratio': tuned('REPLAY_RATIO', 8.0),
        'replay_buffer_max_length': int(tuned('REPLAY_BUFFER_MAX_LENGTH', 1000000, int)),
        'initial_collect_steps': int(tuned('INITIAL_COLLECT_STEPS', 2000, int)),
        'priority_exponent': tuned('PRIORITY_EXPONENT', 0.5),
        'reset_interval': int(tuned('RESET_INTERVAL', 40000, int)),
        'reset_alpha': tuned('RESET_ALPHA', 0.5),
        'reset_stop_after': int(tuned('RESET_STOP_AFTER', 0, int)),
        'reset_anneal_n_step': str(tuned('RESET_ANNEAL_N_STEP', '10,3', str)).strip(),
        'reset_anneal_gamma': str(tuned('RESET_ANNEAL_GAMMA', '0.97,0.997', str)).strip(),
        'reset_anneal_steps': int(tuned('RESET_ANNEAL_STEPS', 10000, int)),
        'dist_atoms': int(tuned('DIST_ATOMS', 51, int)),
        'dist_v_min': tuned('DIST_V_MIN', -10.0),
        'dist_v_max': tuned('DIST_V_MAX', 110.0),
        'bbf_weight_decay': tuned('BBF_WEIGHT_DECAY', 0.1),
        'bbf_spr_weight': tuned('BBF_SPR_WEIGHT', 5.0),
        'bbf_spr_steps': int(tuned('BBF_SPR_STEPS', 5, int)),
        'bbf_projection': int(tuned('BBF_PROJECTION', 512, int)),
        'bbf_transition_width': int(tuned('BBF_TRANSITION_WIDTH', 256, int)),
        'bbf_dueling': bool(int(tuned('BBF_DUELING', 1, int))),
        'bbf_double': bool(int(tuned('BBF_DOUBLE', 1, int))),
    }
    resets.ResetSchedule(config['reset_interval'], config['reset_alpha'], config['reset_stop_after'])
    cycle = resets.cycle_from_config(config)
    if not 0.0 <= config['min_epsilon'] <= config['initial_epsilon'] <= 1.0:
        raise ValueError('SNEK_MIN_EPSILON={0} and SNEK_INITIAL_EPSILON={1} must satisfy 0 <= min <= initial <= 1'.format(
            config['min_epsilon'], config['initial_epsilon']))
    if config['replay_ratio'] <= 0.0:
        raise ValueError('SNEK_REPLAY_RATIO={0} must be positive'.format(config['replay_ratio']))
    if config['n_step_update'] < 1:
        raise ValueError('SNEK_N_STEP_UPDATE={0} must be at least 1'.format(config['n_step_update']))
    if config['bbf_spr_steps'] < 0 or config['bbf_projection'] < 1 or config['bbf_transition_width'] < 0:
        raise ValueError('SNEK_BBF_SPR_STEPS must be >= 0, SNEK_BBF_PROJECTION >= 1 and SNEK_BBF_TRANSITION_WIDTH >= 0')
    if config['dist_v_max'] <= config['dist_v_min']:
        raise ValueError('SNEK_DIST_V_MAX must exceed SNEK_DIST_V_MIN')
    if not 0.0 < config['target_update_tau'] <= 1.0:
        raise ValueError('SNEK_TARGET_UPDATE_TAU={0} must be in (0, 1]'.format(config['target_update_tau']))
    horizon = horizon_of(config, cycle)
    if config['replay_buffer_max_length'] < 2 * (horizon + 1) * config['collect_envs']:
        raise ValueError('SNEK_REPLAY_BUFFER_MAX_LENGTH={0} cannot hold {1} lane(s) at horizon {2}'.format(
            config['replay_buffer_max_length'], config['collect_envs'], horizon))
    return config


def horizon_of(config, cycle):
    """The longest in-lane lookahead a batch needs: the largest n the cycle can ask for, or K."""
    largest_n = max(cycle.n_steps) if cycle.enabled else int(config['n_step_update'])
    return max(largest_n, int(config['bbf_spr_steps']), 1)


def arch_fields(config):
    """`head` and `trunk` for the sidecar: dueling C51, noisy off, `QNet`'s plain stack. A `bbf` checkpoint is a
    Rainbow network, and `tools/restore.py` reads it as one."""
    head = {'type': 'c51', 'atoms': config['dist_atoms'], 'v_min': float(config['dist_v_min']),
            'v_max': float(config['dist_v_max'])}
    trunk = {'dueling': bool(config['bbf_dueling']), 'noisy': False, 'noisy_sigma': 0.5, 'residual': False,
             'blocks': 1, 'spectral_norm': False, 'layer_norm': False}
    return {'head': head, 'trunk': trunk}


def reportable(config):
    return dict(config)


class BbfAlgo(object):

    step_granularity = 1

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        cycle = resets.cycle_from_config(config)
        self.agent = BbfAgent(arch, learning_rate=config['learning_rate'], adam_epsilon=config['adam_epsilon'],
                              weight_decay=config['bbf_weight_decay'], gradient_clipping=config['gradient_clipping'],
                              target_tau=config['target_update_tau'], seed=config['seed'], device=device,
                              double=config['bbf_double'], spr_weight=config['bbf_spr_weight'],
                              spr_steps=config['bbf_spr_steps'], projection_dim=config['bbf_projection'],
                              transition_width=config['bbf_transition_width'],
                              reset_interval=config['reset_interval'], reset_alpha=config['reset_alpha'],
                              reset_stop_after=config['reset_stop_after'], cycle=cycle)
        self.buffer = SequentialReplay(config['replay_buffer_max_length'], arch['obs_len'],
                                       lanes=config['collect_envs'], horizon=horizon_of(config, cycle),
                                       alpha=config['priority_exponent'], seed=config['seed'])
        _, gamma = self.agent.cycle_values()
        # The collector at n_step 1: every lane banks one raw row per step, which is the replay's contract.
        # Its `discount` is only the terminal flag here (0 on death); the return is summed at sample time.
        self.collector = collect.Collector(VecSnake(config['collect_envs'], seed=config['seed'], shaping_discount=gamma),
                                           self.agent, self.buffer, discount=gamma, n_step=1,
                                           collect_envs=config['collect_envs'], fork=collect.ForkConfig(branches=1),
                                           guided_fraction=0.0, seed=config['seed'])
        self.epsilon = float(config['initial_epsilon'])
        self.moves = 0
        self.gradient_debt = 0.0

    # ------------------------------------------------------------ what a checkpoint and an eval see

    @property
    def net(self):
        return self.agent.net

    @property
    def policy_fn(self):
        return self.agent.policy_fn

    def describe(self):
        c = self.config
        parts = ['bbf', '{0} lane(s)'.format(c['collect_envs']), 'replay ratio {0}'.format(c['replay_ratio']),
                 'batch {0}'.format(c['batch_size']),
                 'AdamW lr {0} wd {1}'.format(c['learning_rate'], c['bbf_weight_decay']),
                 'EMA tau {0}'.format(c['target_update_tau']),
                 '{0}C51 {1} atoms'.format('dueling ' if c['bbf_dueling'] else '', c['dist_atoms']),
                 'double-Q' if c['bbf_double'] else 'target argmax',
                 'SPR K {0} weight {1}'.format(c['bbf_spr_steps'], c['bbf_spr_weight']) if c['bbf_spr_weight'] > 0 else 'no SPR',
                 'epsilon linear {0} -> {1} over {2:,} moves'.format(c['initial_epsilon'], c['min_epsilon'], c['epsilon_anneal_steps'])]
        if self.agent.reset_schedule.enabled:
            parts.append(self.agent.reset_schedule.describe())
        if self.agent.cycle.enabled:
            parts.append(self.agent.cycle.describe())
        else:
            parts.append('n-step {0} at gamma {1} throughout'.format(c['n_step_update'], c['discount']))
        return ', '.join(parts)

    # ------------------------------------------------------------ the loop

    def prefill(self):
        """`initial_collect_steps` rows at epsilon 1, as the paper's warm-up (Dopamine's `min_replay_history`)."""
        banked = 0
        while self.buffer.size < self.config['initial_collect_steps']:
            banked += self.collector.step(1.0)
        return banked

    def _linear_epsilon(self):
        return schedules.linear_epsilon(self.moves, self.config['initial_epsilon'], self.config['min_epsilon'],
                                        self.config['epsilon_anneal_steps'])

    def advance(self):
        self.epsilon = self._linear_epsilon()
        _, gamma = self.agent.cycle_values()
        self.collector.discount = gamma
        self.collector.vec.shaping_discount = gamma
        transitions = self.collector.step(self.epsilon)
        self.moves += int(transitions)
        self._learn(transitions)
        return 1, transitions

    def _learn(self, transitions):
        self.gradient_debt += transitions * self.config['replay_ratio']
        while self.gradient_debt >= 1.0:
            self.gradient_debt -= 1.0
            n_step, gamma = self.agent.cycle_values()
            drawn = self.buffer.sample(self.config['batch_size'], n_step, gamma, self.config['bbf_spr_steps'])
            if drawn is None:
                continue
            batch, indexes, weights = drawn
            td, _ = self.agent.update(batch, weights)
            self.buffer.update_priorities(indexes, td)

    # ------------------------------------------------------------ the row

    def fields(self):
        n_step, gamma = self.agent.cycle_values()
        out = {'epsilon': round(float(self.epsilon), 5),
               'cycle': {'n_step': int(n_step), 'gamma': round(float(gamma), 5), 'since_reset': int(self.agent.steps_since_reset)},
               'bbf': {'resets': int(self.agent.resets), 'train_step': int(self.agent.train_step)}}
        for key in ('td_loss', 'spr_loss', 'grad_norm'):
            if key in self.agent.last:
                out['bbf'][key] = round(float(self.agent.last[key]), 5)
        return out

    def on_eval(self, eval_rows, measured):
        self.epsilon = self._linear_epsilon()
        return {'epsilon': round(float(self.epsilon), 5)}

    def log_note(self, row):
        return 'eps {0:<7}'.format(row.get('epsilon', '?'))

    def log_extra(self, row):
        lines = []
        bbf = row.get('bbf') or {}
        if bbf:
            lines.append('           bbf resets {0}  td {1}  spr {2}  (gradient step {3:,})'.format(
                bbf.get('resets', 0), bbf.get('td_loss', '?'), bbf.get('spr_loss', '?'), bbf.get('train_step', 0)))
        if row.get('cycle'):
            cycle = row['cycle']
            lines.append('           cycle n-step {0}  gamma {1}  ({2:,} gradient steps since reset)'.format(
                cycle['n_step'], cycle['gamma'], cycle['since_reset']))
        return lines

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'agent': self.agent.state_dict(), 'epsilon': float(self.epsilon), 'moves': int(self.moves)}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        self.epsilon = float(state.get('epsilon', self.epsilon))
        self.moves = int(state.get('moves', 0))

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
    return BbfAlgo(config, arch, device=device)
