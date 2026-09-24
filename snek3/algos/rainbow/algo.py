"""Rainbow and Beyond the Rainbow behind the seam: `DqnAlgo` with a `RainbowAgent`, two names, one module.

`rainbow` (Hessel et al. 2018) and `btr` (Clark, Towers, Evers & Hare 2024) are one network assembled from
flags and one agent, so they share this module; the two names differ in their **defaults**, which are each
paper's settings translated by `plans/algoExploration/README.md`'s rules (`c-value-stack.md` §2b). A spec
states only what departs. The knobs DQN owns keep DQN's names (`SNEK_LEARNING_RATE`, `SNEK_BATCH_SIZE`,
...), the head's keep Group A's `SNEK_DIST_*`, and what is this group's carries `RAINBOW_` or `BTR_`:

| knob | rainbow | btr | what |
|---|---|---|---|
| `SNEK_RAINBOW_HEAD` | c51 | iqn | `c51`, `quantile` (QR-DQN, the BTR local cell's head) or `iqn` |
| `SNEK_RAINBOW_NOISY`, `SNEK_RAINBOW_NOISY_SIGMA` | 1, 0.5 | 1, 0.5 | noisy streams; **lifts the epsilon hard floor**, so `SNEK_MIN_EPSILON=0` is legal only with it on |
| `SNEK_RAINBOW_DUELING` | 1 | 1 | the value stream |
| `SNEK_RAINBOW_DOUBLE` | 1 | 0 | the target action from the online net (1) or the target net (0); moot with Munchausen on |
| `SNEK_RAINBOW_EPSILON_ZERO_AT` | 0 | 0.5 | with the linear schedule, epsilon drops to 0 at this fraction of the run's moves (`max_steps` x lanes); 0 never |
| `SNEK_BTR_RESIDUAL`, `SNEK_BTR_BLOCKS` | 0, 3 | 1, 3 | the residual trunk in place of `QNet`'s stack |
| `SNEK_BTR_SPECTRAL_NORM`, `SNEK_BTR_LAYER_NORM` | 1, 0 | 1, 0 | inside the residual blocks only |
| `SNEK_RAINBOW_STREAM_WIDTH` | 512 | 512 | the hidden layer in each dueling stream (`net.py`), the papers' 512; 0 is one linear per stream |
| `SNEK_IS_NORMALIZATION` | batch_max | batch_max | importance weights over the batch maximum, as PER, Dopamine and the BTR code; `mean` is this codebase's (`algos/dqn/replay.py`) |
| `SNEK_RAINBOW_PREFILL_EPSILON` | random | schedule | the warmup at epsilon 1 off the clock (Dopamine), or acting under the schedule with its moves on the clock (the BTR code) |
| `SNEK_RAINBOW_EPSILON_DECAY` | linear | geometric | the shape of the linear schedule's anneal: a straight line to `min_epsilon` at `epsilon_anneal_steps`, or BTR's code, `eps -= (eps - min) / steps` per move (about 0.37 at `steps`) |
| `SNEK_RAINBOW_MUNCHAUSEN_LOGPI` | target | online | the net that reads the taken action's `tau log pi(a|s)` (`agent.py`) |

**Which source wins where they disagree** (decided 2026-09-23): `rainbow` is the **paper**, with Dopamine only
for what the paper does not state; `btr` is the **authors' released code**, since the code produced the
published numbers. For BTR that means PER's importance exponent is 0.2, not the declared 0.45 (`PER.py`
uses `alpha` for it, "an accident but actually performed better"), epsilon decays geometrically, and
Munchausen's log-policy is the online net's, and the warmup acts under the schedule.

**The paper cell is the paper wherever the game allows** (the user's rule, 2026-09-23): losses and their
reduction, priorities, importance-weight normalisation, clipping, the optimiser, widths that have a
counterpart. The only departures are the series README's translation table (no reward clipping, supports
from the return range, the MLP trunk for the CNN) and what the game cannot express. This codebase's own
choices -- mean-normalised weights, the fork, the shield, the eval-driven epsilon -- are the local cell's,
set in its spec.

`SNEK_BETA_ANNEAL_STEPS` 0 (Rainbow's default) means **the run's cap**: `max_steps` x lanes x replay ratio
gradient updates, so beta reaches 1 at the end of whatever run the spec asks for rather than of a 50M-move one.

`algos/dqn/` and `algos/dist/` are imported and not edited (decided 2026-09-20). What this module redoes
rather than reuses is `build_config`, because the defaults are the papers' and the epsilon floor is
conditional on the noisy flag, which DQN's validation cannot know about.
"""

import math
import os

from algos.dqn import algo as dqn_algo
from algos.dqn import collect
from algos.dqn import replay
from algos.dqn import resets
from algos.dqn import schedules
from algos.rainbow import net as network
from algos.rainbow.agent import MUNCHAUSEN_LOGPI, RainbowAgent
from algos.sac import algo as sac_algo

NAMES = ('rainbow', 'btr')
EPSILON_DECAYS = ('linear', 'geometric')
PREFILL_EPSILONS = ('random', 'schedule')

# The papers' settings, per name. Rainbow: Hessel et al. 2018 Table 1 and Dopamine's `rainbow.gin`;
# BTR: Table D6 and the authors' code. Units: moves for the anneal, gradient updates for the target
# period and the beta anneal, transitions for the replay and the prefill.
PAPER = {
    'rainbow': {
        'learning_rate': 6.25e-5, 'adam_epsilon': 1.5e-4, 'batch_size': 32, 'discount': 0.99,
        'n_step_update': 3, 'target_update_period': 2000, 'gradient_clipping': 0.0,
        'initial_epsilon': 0.0, 'min_epsilon': 0.0, 'epsilon_anneal_steps': 1,
        'collect_envs': 1, 'replay_ratio': 0.25, 'replay_buffer_max_length': 1000000,
        'initial_collect_steps': 20000, 'priority_exponent': 0.5, 'is_beta': 0.4, 'is_beta_final': 1.0,
        # 0 is the run's cap (`RainbowAlgo`): beta reaches 1.0 at the last update whatever `max_steps` is.
        'beta_anneal_steps': 0,
        'munchausen_alpha': 0.0, 'head': 'c51', 'double': 1, 'epsilon_zero_at': 0.0, 'residual': 0,
        'epsilon_decay': 'linear', 'munchausen_logpi': 'target', 'prefill_epsilon': 'random',
    },
    'btr': {
        'learning_rate': 1e-4, 'adam_epsilon': 0.005 / 256, 'batch_size': 256, 'discount': 0.997,
        'n_step_update': 3, 'target_update_period': 500, 'gradient_clipping': 10.0,
        'initial_epsilon': 1.0, 'min_epsilon': 0.01, 'epsilon_anneal_steps': 2000000,
        'collect_envs': 64, 'replay_ratio': 1.0 / 64.0, 'replay_buffer_max_length': 2 ** 20,
        'initial_collect_steps': 200000, 'priority_exponent': 0.2, 'is_beta': 0.2, 'is_beta_final': 0.2,
        'beta_anneal_steps': 1,
        'munchausen_alpha': 0.9, 'head': 'iqn', 'double': 0, 'epsilon_zero_at': 0.5, 'residual': 1,
        'epsilon_decay': 'geometric', 'munchausen_logpi': 'online', 'prefill_epsilon': 'schedule',
    },
}

REJECTED = (
    'PPO_ROLLOUT', 'PPO_EPOCHS', 'PPO_MINIBATCH', 'PPO_CLIP', 'PPO_CLIP_FINAL', 'PPO_GAE_LAMBDA',
    'PPO_GAE_LAMBDA_FINAL', 'PPO_DISCOUNT_FINAL', 'PPO_ENTROPY_COEF', 'PPO_ENTROPY_COEF_FINAL',
    'PPO_VF_COEF', 'PPO_LEARNING_RATE', 'PPO_LEARNING_RATE_FINAL', 'PPO_ANNEAL_FRACTION',
    'PPO_ADAM_EPSILON', 'PPO_TARGET_KL', 'PPO_GRADIENT_CLIPPING', 'PPO_NORMALIZE_ADV', 'PPO_VALUE_LOSS',
    # FQF's proposal net and the risk-sensitive read are A5's and A4's; neither paper here has them.
    'DIST_FRACTION_LR', 'DIST_FRACTION_ENTROPY', 'DIST_RISK_ALPHA', 'DIST_RISK_TRAIN',
    'BBF_WEIGHT_DECAY', 'BBF_SPR_WEIGHT', 'BBF_SPR_STEPS', 'BBF_PROJECTION', 'BBF_TRANSITION_WIDTH', 'BBF_DUELING', 'BBF_DOUBLE',
) + sac_algo.SAC_KNOBS


def _refuse_foreign_knobs(name):
    present = sorted(knob for knob in REJECTED if os.environ.get('SNEK_' + knob) is not None)
    if present:
        raise ValueError('SNEK_ALGO={0} cannot honour {1}: {2}. Rainbow has no PPO rollout, SAC temperature, '
                         'FQF proposal net or risk read.'.format(
                             name, 'these knobs' if len(present) > 1 else 'this knob',
                             ', '.join('SNEK_' + knob for knob in present)))


def build_config(tuned, name):
    """The paper's defaults for `name` under DQN's and Group A's knob names, plus this group's own.
    Every key is its `SNEK_` variable lowercased (`tests/test_train.py`)."""
    _refuse_foreign_knobs(name)
    paper = PAPER[name]
    fork = collect.ForkConfig(branches=int(tuned('FORK_BRANCHES', 1, int)),
                              prob=tuned('FORK_PROB', 0.5),
                              min_length=int(tuned('FORK_MIN_LENGTH', 85, int)),
                              max_steps=int(tuned('FORK_MAX_STEPS', 60, int)))
    config = {
        'learning_rate': tuned('LEARNING_RATE', paper['learning_rate']),
        'adam_epsilon': tuned('ADAM_EPSILON', paper['adam_epsilon']),
        'batch_size': int(tuned('BATCH_SIZE', paper['batch_size'], int)),
        'discount': tuned('DISCOUNT', paper['discount']),
        'n_step_update': int(tuned('N_STEP_UPDATE', paper['n_step_update'], int)),
        'target_update_period': int(tuned('TARGET_UPDATE_PERIOD', paper['target_update_period'], int)),
        'target_update_tau': tuned('TARGET_UPDATE_TAU', 1.0),
        'gradient_clipping': tuned('GRADIENT_CLIPPING', paper['gradient_clipping']),
        'initial_epsilon': tuned('INITIAL_EPSILON', paper['initial_epsilon']),
        'min_epsilon': tuned('MIN_EPSILON', paper['min_epsilon']),
        'guided_fraction': tuned('GUIDED_FRACTION', 0.0),
        'collect_envs': int(tuned('COLLECT_ENVS', paper['collect_envs'], int)),
        'replay_ratio': tuned('REPLAY_RATIO', paper['replay_ratio']),
        'replay_buffer_max_length': int(tuned('REPLAY_BUFFER_MAX_LENGTH', paper['replay_buffer_max_length'], int)),
        'initial_collect_steps': int(tuned('INITIAL_COLLECT_STEPS', paper['initial_collect_steps'], int)),
        'priority_exponent': tuned('PRIORITY_EXPONENT', paper['priority_exponent']),
        'is_beta': tuned('IS_BETA', paper['is_beta']),
        'is_beta_final': tuned('IS_BETA_FINAL', paper['is_beta_final']),
        'beta_anneal_steps': int(tuned('BETA_ANNEAL_STEPS', paper['beta_anneal_steps'], int)),
        'is_weights': bool(int(tuned('IS_WEIGHTS', 1, int))),
        'epsilon_schedule': str(tuned('EPSILON_SCHEDULE', 'linear', str)),
        'epsilon_anneal_steps': int(tuned('EPSILON_ANNEAL_STEPS', paper['epsilon_anneal_steps'], int)),
        'munchausen_alpha': tuned('MUNCHAUSEN_ALPHA', paper['munchausen_alpha']),
        'munchausen_tau': tuned('MUNCHAUSEN_TAU', 0.03),
        'munchausen_l0': tuned('MUNCHAUSEN_L0', -1.0),
        'reset_interval': int(tuned('RESET_INTERVAL', 0, int)),
        'reset_alpha': tuned('RESET_ALPHA', 0.5),
        'reset_stop_after': int(tuned('RESET_STOP_AFTER', 0, int)),
        'dist_atoms': int(tuned('DIST_ATOMS', 51, int)),
        'dist_v_min': tuned('DIST_V_MIN', -10.0),
        'dist_v_max': tuned('DIST_V_MAX', 110.0),
        'dist_quantiles': int(tuned('DIST_QUANTILES', 32, int)),
        'dist_tau_samples': int(tuned('DIST_TAU_SAMPLES', 8, int)),
        'dist_tau_prime_samples': int(tuned('DIST_TAU_PRIME_SAMPLES', 8, int)),
        'dist_policy_samples': int(tuned('DIST_POLICY_SAMPLES', 8, int)),
        'dist_embedding': int(tuned('DIST_EMBEDDING', 64, int)),
        'dist_kappa': tuned('DIST_KAPPA', 1.0),
        'rainbow_head': str(tuned('RAINBOW_HEAD', paper['head'], str)),
        'rainbow_noisy': bool(int(tuned('RAINBOW_NOISY', 1, int))),
        'rainbow_noisy_sigma': tuned('RAINBOW_NOISY_SIGMA', 0.5),
        'rainbow_dueling': bool(int(tuned('RAINBOW_DUELING', 1, int))),
        'rainbow_double': bool(int(tuned('RAINBOW_DOUBLE', paper['double'], int))),
        'rainbow_epsilon_zero_at': tuned('RAINBOW_EPSILON_ZERO_AT', paper['epsilon_zero_at']),
        'btr_residual': bool(int(tuned('BTR_RESIDUAL', paper['residual'], int))),
        'btr_blocks': int(tuned('BTR_BLOCKS', 3, int)),
        'btr_spectral_norm': bool(int(tuned('BTR_SPECTRAL_NORM', 1, int))),
        'btr_layer_norm': bool(int(tuned('BTR_LAYER_NORM', 0, int))),
        'rainbow_stream_width': int(tuned('RAINBOW_STREAM_WIDTH', 512, int)),
        'is_normalization': str(tuned('IS_NORMALIZATION', 'batch_max', str)),
        'rainbow_prefill_epsilon': str(tuned('RAINBOW_PREFILL_EPSILON', paper['prefill_epsilon'], str)),
        'rainbow_epsilon_decay': str(tuned('RAINBOW_EPSILON_DECAY', paper['epsilon_decay'], str)),
        'rainbow_munchausen_logpi': str(tuned('RAINBOW_MUNCHAUSEN_LOGPI', paper['munchausen_logpi'], str)),
        'fork': fork,
    }
    config.update(resets.anneal_config(tuned))
    resets.ResetSchedule(config['reset_interval'], config['reset_alpha'], config['reset_stop_after'])
    resets.cycle_from_config(config)
    if config['epsilon_schedule'] not in dqn_algo.EPSILON_SCHEDULES:
        raise ValueError('SNEK_EPSILON_SCHEDULE={0!r} is not one of {1}'.format(
            config['epsilon_schedule'], sorted(dqn_algo.EPSILON_SCHEDULES)))
    if config['rainbow_head'] not in network.HEAD_TYPES:
        raise ValueError('SNEK_RAINBOW_HEAD={0!r} is not one of {1}'.format(config['rainbow_head'], network.HEAD_TYPES))
    if config['min_epsilon'] < 0.0:
        raise ValueError('SNEK_MIN_EPSILON={0} is negative'.format(config['min_epsilon']))
    # The hard floor protects an epsilon-greedy agent from an exploration-free run. Noisy nets supply
    # the exploration the floor protects, so the flag lifts it (decided 2026-09-20); without them it holds.
    if not config['rainbow_noisy'] and config['min_epsilon'] < schedules.EPSILON_HARD_FLOOR:
        raise ValueError('SNEK_MIN_EPSILON={0} is below the hard floor {1}; only SNEK_RAINBOW_NOISY=1 '
                         'lifts it'.format(config['min_epsilon'], schedules.EPSILON_HARD_FLOOR))
    if config['replay_ratio'] <= 0.0:
        raise ValueError('SNEK_REPLAY_RATIO={0} must be positive'.format(config['replay_ratio']))
    if not 0.0 <= config['rainbow_epsilon_zero_at'] <= 1.0:
        raise ValueError('SNEK_RAINBOW_EPSILON_ZERO_AT={0} must be in [0, 1]'.format(config['rainbow_epsilon_zero_at']))
    if config['rainbow_epsilon_decay'] not in EPSILON_DECAYS:
        raise ValueError('SNEK_RAINBOW_EPSILON_DECAY={0!r} is not one of {1}'.format(
            config['rainbow_epsilon_decay'], EPSILON_DECAYS))
    if config['rainbow_munchausen_logpi'] not in MUNCHAUSEN_LOGPI:
        raise ValueError('SNEK_RAINBOW_MUNCHAUSEN_LOGPI={0!r} is not one of {1}'.format(
            config['rainbow_munchausen_logpi'], MUNCHAUSEN_LOGPI))
    if config['is_normalization'] not in replay.IS_NORMALIZATIONS:
        raise ValueError('SNEK_IS_NORMALIZATION={0!r} is not one of {1}'.format(
            config['is_normalization'], replay.IS_NORMALIZATIONS))
    if config['rainbow_prefill_epsilon'] not in PREFILL_EPSILONS:
        raise ValueError('SNEK_RAINBOW_PREFILL_EPSILON={0!r} is not one of {1}'.format(
            config['rainbow_prefill_epsilon'], PREFILL_EPSILONS))
    if config['rainbow_stream_width'] < 0:
        raise ValueError('SNEK_RAINBOW_STREAM_WIDTH={0} is negative; 0 means one linear per stream'.format(
            config['rainbow_stream_width']))
    if config['beta_anneal_steps'] < 0:
        raise ValueError('SNEK_BETA_ANNEAL_STEPS={0} is negative; 0 means the run\'s cap'.format(
            config['beta_anneal_steps']))
    if config['btr_blocks'] < 1:
        raise ValueError('SNEK_BTR_BLOCKS={0} must be at least 1'.format(config['btr_blocks']))
    if config['rainbow_head'] == 'c51' and config['dist_v_max'] <= config['dist_v_min']:
        raise ValueError('SNEK_DIST_V_MAX must exceed SNEK_DIST_V_MIN')
    return config


def arch_fields(config):
    """The `head` and `trunk` the sidecar records, from the config. `train.py` calls this hook."""
    kind = config['rainbow_head']
    if kind == 'c51':
        head = {'type': 'c51', 'atoms': config['dist_atoms'], 'v_min': float(config['dist_v_min']),
                'v_max': float(config['dist_v_max'])}
    elif kind == 'quantile':
        head = {'type': 'quantile', 'n': config['dist_quantiles']}
    else:
        head = {'type': 'iqn', 'embedding': config['dist_embedding'],
                'n_tau': config['dist_tau_samples'], 'k': config['dist_policy_samples']}
    trunk = {'dueling': bool(config['rainbow_dueling']), 'noisy': bool(config['rainbow_noisy']),
             'noisy_sigma': float(config['rainbow_noisy_sigma']), 'residual': bool(config['btr_residual']),
             'blocks': int(config['btr_blocks']), 'spectral_norm': bool(config['btr_spectral_norm']),
             'layer_norm': bool(config['btr_layer_norm']), 'stream_width': int(config['rainbow_stream_width'])}
    return {'head': head, 'trunk': trunk}


def reportable(config):
    return dqn_algo.reportable(config)


def geometric_epsilon(moves, initial, final, steps):
    """BTR's `EpsilonGreedy.update_eps` after `moves` calls: `eps -= (eps - final) / steps` each move, so
    `final + (initial - final) (1 - 1/steps) ** moves` -- about `final + 0.37 (initial - final)` at `steps`."""
    if steps <= 1:
        return float(final)
    return float(final + (initial - final) * math.exp(moves * math.log1p(-1.0 / steps)))


class RainbowAlgo(dqn_algo.DqnAlgo):
    """`DqnAlgo` whose agent is a `RainbowAgent`. The parent builds a `DdqnAgent` first; it is replaced
    before anything runs, and the collector is pointed at the replacement."""

    def __init__(self, config, arch, device='cpu', name='rainbow'):
        self.name = name
        lanes = int(config['collect_envs']) * int(config['fork'].branches)
        if config['beta_anneal_steps'] == 0:
            # Resolved in place, so the report shows the number the buffer ran with.
            config['beta_anneal_steps'] = max(1, int(round(
                int(config.get('max_steps', 0)) * lanes * float(config['replay_ratio']))))
        super().__init__(config, arch, device=device)
        if network.head_of(arch)['type'] != config['rainbow_head']:
            raise ValueError('arch head {0!r} is not the configured {1}'.format(arch['head'], config['rainbow_head']))
        self.agent = RainbowAgent(arch, double=config['rainbow_double'],
                                  munchausen_logpi=config['rainbow_munchausen_logpi'],
                                  learning_rate=config['learning_rate'],
                                  adam_epsilon=config['adam_epsilon'],
                                  target_update_period=config['target_update_period'],
                                  target_update_tau=config['target_update_tau'],
                                  gradient_clipping=config['gradient_clipping'],
                                  use_is_weights=config['is_weights'],
                                  seed=config['seed'], device=device,
                                  munchausen_alpha=config['munchausen_alpha'],
                                  munchausen_tau=config['munchausen_tau'],
                                  munchausen_l0=config['munchausen_l0'],
                                  reset_interval=config['reset_interval'],
                                  reset_alpha=config['reset_alpha'],
                                  reset_stop_after=config['reset_stop_after'],
                                  cycle=resets.cycle_from_config(config), on_cycle=self.apply_cycle,
                                  kappa=config['dist_kappa'],
                                  n_tau=config['dist_tau_samples'],
                                  n_tau_prime=config['dist_tau_prime_samples'])
        self.collector.agent = self.agent
        self.buffer.normalization = config['is_normalization']
        if config['reset_interval'] > 0:
            # Refused here, before the first step, rather than at the first reset: `resets.partition` finds
            # the trunk by the `hidden.` name, which the residual trunk's stem and blocks do not carry.
            try:
                resets.partition(self.agent.net)
            except ValueError as error:
                raise ValueError('SNEK_RESET_INTERVAL={0} cannot run on this net ({1}): {2}'.format(
                    config['reset_interval'], 'residual trunk' if arch['trunk']['residual'] else 'plain trunk',
                    error)) from None
        # BTR's second half at epsilon 0: a fraction of the run's moves, the cap in counted steps times
        # the moves one step is (every lane advances once). 0 means never.
        fraction = float(config['rainbow_epsilon_zero_at'])
        self.epsilon_zero_after = int(round(fraction * int(config.get('max_steps', 0)) * lanes)) if fraction > 0.0 else 0

    def prefill(self):
        """The warmup. `random` is the parent's: epsilon 1, off the clock (Dopamine). `schedule` is BTR's
        code: the agent acts under the linear schedule from move 0, noisy greedy under the epsilon mask,
        and the warmup's moves count toward the anneal and the zero point."""
        if self.config['rainbow_prefill_epsilon'] != 'schedule' or not self.linear_schedule:
            return super().prefill()
        target = self.config['initial_collect_steps']
        banked = 0
        while self.buffer.size < target:
            self.epsilon = self._linear_epsilon()
            transitions = self.collector.step(self.epsilon)
            self.moves += int(transitions)
            banked += transitions
        return banked

    def _linear_epsilon(self):
        if self.epsilon_zero_after > 0 and self.moves >= self.epsilon_zero_after:
            return 0.0
        if self.config['rainbow_epsilon_decay'] == 'geometric':
            return geometric_epsilon(self.moves, self.config['initial_epsilon'], self.config['min_epsilon'],
                                     self.config['epsilon_anneal_steps'])
        return super()._linear_epsilon()

    def describe(self):
        trunk = self.arch['trunk']
        head = self.arch['head']
        parts = [self.name, '{0} head'.format(head['type']),
                 'noisy sigma {0}'.format(trunk['noisy_sigma']) if trunk['noisy'] else 'epsilon-greedy',
                 'dueling' if trunk['dueling'] else 'single stream',
                 'residual trunk x{0}{1}{2}'.format(trunk['blocks'], ' spectral' if trunk['spectral_norm'] else '',
                                                    ' layernorm' if trunk['layer_norm'] else '')
                 if trunk['residual'] else 'plain trunk',
                 'double-Q' if self.agent.double else 'target argmax',
                 'streams {0} wide'.format(trunk['stream_width']) if trunk.get('stream_width') else 'one-linear streams',
                 'importance weights {0}'.format(self.buffer.normalization)]
        if self.config['rainbow_prefill_epsilon'] == 'schedule' and self.linear_schedule:
            parts.append('warmup under the schedule')
        if self.linear_schedule and self.config['rainbow_epsilon_decay'] == 'geometric':
            parts.append('epsilon decay geometric (the linear line above is its time constant)')
        if self.config['munchausen_alpha'] > 0.0:
            parts.append('munchausen log-pi from the {0} net'.format(self.agent.munchausen_logpi))
        if self.epsilon_zero_after > 0:
            parts.append('epsilon 0 from move {0:,}'.format(self.epsilon_zero_after))
        return '{0}, {1}'.format(super().describe(), ', '.join(parts))


def build(config, arch, device='cpu', name='rainbow'):
    return RainbowAlgo(config, arch, device=device, name=name)
