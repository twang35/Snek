"""NGU -- Never Give Up (Badia et al. 2020) -- behind the seam `train.py` drives (`algos/dqn/algo.py` for the seam's
members). `SNEK_ALGO=ngu`. `plans/algoExploration/f-ngu-agent57.md` §1-2.

Built 2026-10-25 for Group F's F1: R2D2 (E2, `algos/r2d2/`) with the paper's intrinsic reward, a family of 32
policies indexed by (β_j, γ_j) in one network, and the transformed Retrace target. Its own package because every
piece of E2 gains a column -- the net two cell inputs and the intrinsic modules (`net.py`), the replay the board,
μ and both intrinsic rewards (`replay.py`), the collector the per-lane episodic memory and snapshot (`collect.py`,
`intrinsic.py`), the agent the Retrace trace and the two auxiliary losses (`agent.py`). What it shares with E2 is
imported: the window geometry and the loss-block rule, the burn-in, the rescaling, the priority mix, the ladder.

| knob | paper | notes |
|---|---|---|
| `SNEK_LEARNING_RATE`, `SNEK_ADAM_EPSILON` | 1e-4, **1e-4** | Adam (NGU Table 6; E2 ran 1e-3) |
| `SNEK_BATCH_SIZE`, `SNEK_GRADIENT_CLIPPING` | 64, 40 | |
| `SNEK_DISCOUNT`, `SNEK_NGU_GAMMA_MIN` | 0.997, 0.99 | γ_0 (the exploitative end) and γ_{N−1}; the ladder between is App. A's |
| `SNEK_TARGET_UPDATE_PERIOD` | **1,500** | a hard copy, in updates (E2 ran 2,500) |
| `SNEK_COLLECT_ENVS`, `SNEK_NGU_MIXTURES` | 256 actors / 32 mixtures; **32 / 32** here | lane j holds mixture j (`SNEK_NGU_ASSIGN=fixed`); `episode` redraws uniformly per episode (F2's control) |
| `SNEK_NGU_BETA_MAX` | 0.3 | β_{N−1}; **0 is F1's control cell**: every β_j 0, everything else identical |
| `SNEK_EPSILON_SCHEDULE`, `SNEK_INITIAL_EPSILON`, `SNEK_APEX_ALPHA` | apex, 0.4, **8** | the Ape-X ladder (Agent57 App. E's 0.4^(1 + 8 l/(N−1))) |
| `SNEK_NGU_EPSILON_PERMUTE` | -- | **1**: the ladder's rungs are shuffled over the lanes once, the same in every arm and seed, so ε and β are uncorrelated on 32 lanes (§0, review 3); 0 is rung i on lane i |
| `SNEK_NGU_RETRACE_LAMBDA`, `SNEK_NGU_TARGET_EPSILON` | 0.95, 0.01 | transformed Retrace; π ε-soft greedy on the online net |
| `SNEK_R2D2_SEQ_LENGTH`, `SNEK_R2D2_BURN_IN`, `SNEK_R2D2_STRIDE` | 120, 40, 40 | trace 80 / replay period 40 with R2D2's 40 burn-in; `SNEK_N_STEP_UPDATE` is the lookahead the Retrace trace runs into past the block (5) |
| `SNEK_R2D2_WINDOWS`, `SNEK_NGU_MIN_WINDOWS`, `SNEK_INITIAL_COLLECT_STEPS` | 5e6 rows; 6,250 sequences before learning | windows the replay holds (the box's 30,000 in the specs); windows **and** rows before the first update |
| `SNEK_PRIORITY_EXPONENT`, `SNEK_R2D2_IS_BETA`, `SNEK_R2D2_PRIORITY_ETA` | 0.9, **0**, **1.0** | PER; IS off; η (App. A) |
| `SNEK_R2D2_RESCALE`, `SNEK_R2D2_RESCALE_EPS` | on, 1e-3 | `h()` on the targets |
| `SNEK_R2D2_HIDDEN`, `SNEK_R2D2_STREAM_WIDTH`, `SNEK_R2D2_RECURRENT`, `SNEK_R2D2_PREV_INPUT` | 512, 512, lstm, on | E2's core; the mixture one-hot always enters the cell |
| `SNEK_NGU_INPUT` | the frame | **`board`**: the 10 x 10 x 4 board tensor for f and RND (§0); `obs` is the degenerate form a test pins |
| `SNEK_NGU_EMBED_DIM`, `SNEK_NGU_EMBED_HIDDEN`, `SNEK_NGU_EMBED_STEPS`, `SNEK_NGU_EMBED_LR`, `SNEK_NGU_EMBED_L2` | 32, 128, 5, 5e-4, 1e-5 | the inverse-dynamics embedding, trained on the last 5 pairs of each sampled window |
| `SNEK_NGU_RND_HIDDEN`, `SNEK_NGU_RND_OUT` | 128, 128 | the RND pair |
| `SNEK_NGU_KNN_K`, `SNEK_NGU_KERNEL_EPSILON`, `SNEK_NGU_CLUSTER_DISTANCE`, `SNEK_NGU_PSEUDO_COUNT`, `SNEK_NGU_MAX_SIMILARITY`, `SNEK_NGU_MEMORY` | 10, 1e-4, 8e-3, 1e-3, 8, 30,000 | Alg. 1 |
| `SNEK_NGU_LIFELONG_MAX` | L = 5 | the clip on the RND factor |
| `SNEK_REPLAY_RATIO` | -- | gradient steps per banked transition, E2's unit |

**Units.** A counted step is one lockstep of every lane: `collect_envs` moves, `collect_envs` rows.
"""

import os
import resource
import sys
import time

import numpy as np

from algos.dqn import schedules
from algos.ngu import knobs
from algos.ngu import net as network
from algos.ngu.agent import NguAgent
from algos.ngu.collect import NguCollector, ASSIGNMENTS
from algos.ngu.intrinsic import LaneNovelty
from algos.ngu.replay import NguReplay
from algos.r2d2 import algo as r2d2_algo
from algos.r2d2 import knobs as r2d2_knobs
from tools import checkpoints
from vectorized.vec_env import VecSnake, BOARD_CELLS

NAME = 'ngu'

# The one shuffle of the ε ladder over the lanes: fixed, independent of SNEK_SEED, so every arm's lane j holds the
# same rung. Chosen so the rung order and the mixture index are uncorrelated (Spearman −0.05 at 32 lanes).
PERMUTE_SEED = 1

REJECTED = tuple(knob for knob in r2d2_algo.REJECTED if knob not in knobs.NGU_KNOBS)


def _refuse_foreign_knobs():
    present = sorted(knob for knob in REJECTED if os.environ.get('SNEK_' + knob) is not None)
    if present:
        raise ValueError('SNEK_ALGO=ngu cannot honour {0}: {1}. NGU has no fork, shield, EMA target, IS-beta anneal, '
                         'Munchausen term, resets, quantile head, noisy net, PPO rollout, SPR or SAC temperature.'.format(
                             'these knobs' if len(present) > 1 else 'this knob', ', '.join('SNEK_' + knob for knob in present)))


def permuted_rungs(lanes, permute):
    """Lane i's rung of the ladder: `i` in the identity order, or the fixed shuffle."""
    if not permute:
        return np.arange(int(lanes))
    return np.random.default_rng(PERMUTE_SEED).permutation(int(lanes))


def build_config(tuned):
    """The paper's values under the shared names plus `R2D2_` and `NGU_`. Every key is its `SNEK_` variable lowercased."""
    _refuse_foreign_knobs()
    config = {
        'learning_rate': tuned('LEARNING_RATE', 1e-4),
        'adam_epsilon': tuned('ADAM_EPSILON', 1e-4),
        'batch_size': int(tuned('BATCH_SIZE', 64, int)),
        'discount': tuned('DISCOUNT', 0.997),
        'n_step_update': int(tuned('N_STEP_UPDATE', 5, int)),
        'target_update_period': int(tuned('TARGET_UPDATE_PERIOD', 1500, int)),
        'gradient_clipping': tuned('GRADIENT_CLIPPING', 40.0),
        'collect_envs': int(tuned('COLLECT_ENVS', 32, int)),
        'epsilon_schedule': str(tuned('EPSILON_SCHEDULE', 'apex', str)),
        'initial_epsilon': tuned('INITIAL_EPSILON', 0.4),
        'min_epsilon': tuned('MIN_EPSILON', 0.0),
        'epsilon_anneal_steps': int(tuned('EPSILON_ANNEAL_STEPS', 0, int)),
        'apex_alpha': tuned('APEX_ALPHA', 8.0),
        'replay_ratio': tuned('REPLAY_RATIO', 1.0 / 32.0),
        'initial_collect_steps': int(tuned('INITIAL_COLLECT_STEPS', 20000, int)),
        'priority_exponent': tuned('PRIORITY_EXPONENT', 0.9),
        'r2d2_is_beta': tuned('R2D2_IS_BETA', 0.0),
        'r2d2_priority_eta': tuned('R2D2_PRIORITY_ETA', 1.0),
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
        'r2d2_head': 'scalar',
        'dist_atoms': 51, 'dist_v_min': -10.0, 'dist_v_max': 110.0,
        'ngu_mixtures': int(tuned('NGU_MIXTURES', 32, int)),
        'ngu_beta_max': tuned('NGU_BETA_MAX', 0.3),
        'ngu_gamma_min': tuned('NGU_GAMMA_MIN', 0.99),
        'ngu_assign': str(tuned('NGU_ASSIGN', 'fixed', str)),
        'ngu_retrace_lambda': tuned('NGU_RETRACE_LAMBDA', 0.95),
        'ngu_target_epsilon': tuned('NGU_TARGET_EPSILON', 0.01),
        'ngu_min_windows': int(tuned('NGU_MIN_WINDOWS', 6250, int)),
        'ngu_epsilon_permute': bool(int(tuned('NGU_EPSILON_PERMUTE', 1, int))),
        'ngu_input': str(tuned('NGU_INPUT', 'board', str)),
        'ngu_embed_dim': int(tuned('NGU_EMBED_DIM', 32, int)),
        'ngu_embed_hidden': int(tuned('NGU_EMBED_HIDDEN', 128, int)),
        'ngu_embed_steps': int(tuned('NGU_EMBED_STEPS', 5, int)),
        'ngu_embed_lr': tuned('NGU_EMBED_LR', 5e-4),
        'ngu_embed_l2': tuned('NGU_EMBED_L2', 1e-5),
        'ngu_rnd_hidden': int(tuned('NGU_RND_HIDDEN', 128, int)),
        'ngu_rnd_out': int(tuned('NGU_RND_OUT', 128, int)),
        'ngu_knn_k': int(tuned('NGU_KNN_K', 10, int)),
        'ngu_kernel_epsilon': tuned('NGU_KERNEL_EPSILON', 1e-4),
        'ngu_cluster_distance': tuned('NGU_CLUSTER_DISTANCE', 8e-3),
        'ngu_pseudo_count': tuned('NGU_PSEUDO_COUNT', 1e-3),
        'ngu_max_similarity': tuned('NGU_MAX_SIMILARITY', 8.0),
        'ngu_memory': int(tuned('NGU_MEMORY', 30000, int)),
        'ngu_lifelong_max': tuned('NGU_LIFELONG_MAX', 5.0),
    }
    r2d2_algo.check_config(config)
    for key in ('dist_atoms', 'dist_v_min', 'dist_v_max', 'r2d2_head'):
        del config[key]
    if config['ngu_mixtures'] < 1:
        raise ValueError('SNEK_NGU_MIXTURES must be at least 1')
    if config['ngu_assign'] not in ASSIGNMENTS:
        raise ValueError('SNEK_NGU_ASSIGN={0!r} is not one of {1} (bandit is Agent57\'s, F2, not built here)'.format(
            config['ngu_assign'], ASSIGNMENTS))
    if config['ngu_assign'] == 'fixed' and config['collect_envs'] != config['ngu_mixtures']:
        raise ValueError('SNEK_NGU_ASSIGN=fixed holds one mixture per lane: SNEK_COLLECT_ENVS={0} must equal '
                         'SNEK_NGU_MIXTURES={1}'.format(config['collect_envs'], config['ngu_mixtures']))
    if config['ngu_beta_max'] < 0.0:
        raise ValueError('SNEK_NGU_BETA_MAX must be at least 0')
    if not 0.0 < config['ngu_gamma_min'] <= config['discount'] < 1.0:
        raise ValueError('SNEK_NGU_GAMMA_MIN={0} and SNEK_DISCOUNT={1} must satisfy 0 < min <= discount < 1'.format(
            config['ngu_gamma_min'], config['discount']))
    if not 0.0 < config['ngu_retrace_lambda'] <= 1.0:
        raise ValueError('SNEK_NGU_RETRACE_LAMBDA={0} must be in (0, 1]; the n-step fallback is not built'.format(
            config['ngu_retrace_lambda']))
    if not 0.0 <= config['ngu_target_epsilon'] < 1.0:
        raise ValueError('SNEK_NGU_TARGET_EPSILON must be in [0, 1)')
    if config['ngu_input'] not in network.INPUTS:
        raise ValueError('SNEK_NGU_INPUT={0!r} is not one of {1}'.format(config['ngu_input'], network.INPUTS))
    if config['ngu_min_windows'] < config['batch_size']:
        raise ValueError('SNEK_NGU_MIN_WINDOWS must be at least SNEK_BATCH_SIZE')
    if config['ngu_min_windows'] > config['r2d2_windows']:
        raise ValueError('SNEK_NGU_MIN_WINDOWS={0} exceeds SNEK_R2D2_WINDOWS={1}'.format(config['ngu_min_windows'], config['r2d2_windows']))
    for key in ('ngu_embed_dim', 'ngu_embed_hidden', 'ngu_embed_steps', 'ngu_rnd_hidden', 'ngu_rnd_out', 'ngu_knn_k', 'ngu_memory'):
        if config[key] < 1:
            raise ValueError('SNEK_{0} must be at least 1'.format(key.upper()))
    if config['ngu_lifelong_max'] < 1.0:
        raise ValueError('SNEK_NGU_LIFELONG_MAX must be at least 1')
    for key in ('ngu_kernel_epsilon', 'ngu_pseudo_count'):
        if config[key] <= 0.0:
            raise ValueError('SNEK_{0} must be positive'.format(key.upper()))
    return config


def arch_fields(config):
    """`recurrent` and `head` as E2's, plus `conditioning` (the ladders' endpoints and the layout) and `intrinsic`
    (every shape and constant behind `r^i`), so a checkpoint rebuilds the policy's inputs bit for bit."""
    return {'recurrent': {'type': config['r2d2_recurrent'], 'hidden': int(config['r2d2_hidden']),
                          'stream_width': int(config['r2d2_stream_width']), 'prev_input': bool(config['r2d2_prev_input'])},
            'head': {'type': 'scalar'},
            'conditioning': {'mixtures': int(config['ngu_mixtures']), 'beta_max': float(config['ngu_beta_max']),
                             'gamma_top': float(config['discount']), 'gamma_min': float(config['ngu_gamma_min']), 'split': 'none'},
            'intrinsic': {'input': config['ngu_input'], 'embed_dim': int(config['ngu_embed_dim']),
                          'embed_hidden': int(config['ngu_embed_hidden']), 'rnd_hidden': int(config['ngu_rnd_hidden']),
                          'rnd_out': int(config['ngu_rnd_out']), 'knn_k': int(config['ngu_knn_k']),
                          'kernel_epsilon': float(config['ngu_kernel_epsilon']), 'cluster_distance': float(config['ngu_cluster_distance']),
                          'pseudo_count': float(config['ngu_pseudo_count']), 'max_similarity': float(config['ngu_max_similarity']),
                          'memory': int(config['ngu_memory']), 'lifelong_max': float(config['ngu_lifelong_max'])}}


def reportable(config):
    return dict(config)


def rss_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 ** 3) if sys.platform == 'darwin' else peak / (1024.0 ** 2)


class NguAlgo(object):

    step_granularity = 1

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        c = config
        block = c['r2d2_seq_length'] - c['r2d2_burn_in']
        self.agent = NguAgent(arch, learning_rate=c['learning_rate'], adam_epsilon=c['adam_epsilon'],
                              gradient_clipping=c['gradient_clipping'], target_update_period=c['target_update_period'],
                              burn_in=c['r2d2_burn_in'], block=block, rescale_targets=c['r2d2_rescale'],
                              rescale_eps=c['r2d2_rescale_eps'], priority_eta=c['r2d2_priority_eta'],
                              retrace_lambda=c['ngu_retrace_lambda'], target_epsilon=c['ngu_target_epsilon'],
                              embed_lr=c['ngu_embed_lr'], embed_l2=c['ngu_embed_l2'], embed_steps=c['ngu_embed_steps'],
                              seed=c['seed'], device=device)
        self.buffer = NguReplay(c['r2d2_windows'], arch['obs_len'], c['collect_envs'], self.agent.state_width,
                                board_len=BOARD_CELLS if c['ngu_input'] == 'board' else 0,
                                seq_length=c['r2d2_seq_length'], burn_in=c['r2d2_burn_in'], lookahead=c['n_step_update'],
                                alpha=c['priority_exponent'], beta=c['r2d2_is_beta'], eta=c['r2d2_priority_eta'], seed=c['seed'])
        self.novelty = LaneNovelty(self.agent.net, c['collect_envs'], device=device, training=True)
        mixtures = self._initial_mixtures()
        self.collector = NguCollector(VecSnake(c['collect_envs'], seed=c['seed'], shaping_discount=c['discount']),
                                      self.agent, self.buffer, self._epsilons(0), self.novelty, mixtures,
                                      assign=c['ngu_assign'], stride=c['r2d2_stride'], seed=c['seed'])
        self.moves = 0
        self.gradient_debt = 0.0
        self._pace = {'updates': 0, 'seconds': 0.0}

    # ------------------------------------------------------------ exploration and the index

    def _initial_mixtures(self):
        c = self.config
        if c['ngu_assign'] == 'fixed':
            return np.arange(c['collect_envs']) % c['ngu_mixtures']
        return np.random.default_rng(c['seed']).integers(0, c['ngu_mixtures'], size=c['collect_envs'])

    def _epsilons(self, moves):
        c = self.config
        if c['epsilon_schedule'] == 'apex':
            ladder = np.asarray(schedules.apex_epsilons(c['collect_envs'], c['initial_epsilon'], c['apex_alpha']))
            return ladder[permuted_rungs(c['collect_envs'], c['ngu_epsilon_permute'])]
        shared = schedules.linear_epsilon(moves, c['initial_epsilon'], c['min_epsilon'], c['epsilon_anneal_steps'])
        return [shared] * c['collect_envs']

    @property
    def epsilon(self):
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
        net = self.agent.net
        parts = ['ngu', '{0} lane(s) on {1} mixture(s), assign {2}'.format(c['collect_envs'], c['ngu_mixtures'], c['ngu_assign']),
                 'beta 0 .. {0:.3g}, gamma {1} .. {2}'.format(c['ngu_beta_max'], c['discount'], c['ngu_gamma_min']),
                 'LSTM {0}'.format(c['r2d2_hidden']) if c['r2d2_recurrent'] == 'lstm' else 'dense {0}'.format(c['r2d2_hidden']),
                 'dueling streams {0}'.format(c['r2d2_stream_width']),
                 'cell inputs: index one-hot' + (', prev action, prev r_e, prev r_i' if c['r2d2_prev_input'] else ''),
                 'intrinsic on the {0} ({1} values): embed {2} via {3}, RND {4}->{5}, k {6}, memory {7:,}, L {8}'.format(
                     c['ngu_input'], net.intrinsic_len, c['ngu_embed_dim'], c['ngu_embed_hidden'], c['ngu_rnd_hidden'],
                     c['ngu_rnd_out'], c['ngu_knn_k'], c['ngu_memory'], c['ngu_lifelong_max']),
                 'Retrace lambda {0} target-eps {1}'.format(c['ngu_retrace_lambda'], c['ngu_target_epsilon']),
                 'windows {0} = {1} burn-in + {2} loss, stride {3}, lookahead {4}'.format(
                     c['r2d2_seq_length'], c['r2d2_burn_in'], c['r2d2_seq_length'] - c['r2d2_burn_in'], c['r2d2_stride'], c['n_step_update']),
                 'replay {0:,} windows ({1:,} before learning)'.format(c['r2d2_windows'], c['ngu_min_windows']),
                 'batch {0}'.format(c['batch_size']), 'replay ratio {0:.5g}'.format(c['replay_ratio']),
                 'PER alpha {0} beta {1} eta {2}'.format(c['priority_exponent'], c['r2d2_is_beta'], c['r2d2_priority_eta']),
                 'rescale h() eps {0}'.format(c['r2d2_rescale_eps']) if c['r2d2_rescale'] else 'no rescaling',
                 'Adam lr {0} eps {1}; aux Adam {2} L2 {3} on {4} pairs a window'.format(
                     c['learning_rate'], c['adam_epsilon'], c['ngu_embed_lr'], c['ngu_embed_l2'], c['ngu_embed_steps']),
                 'target copy every {0:,} updates'.format(c['target_update_period']), 'clip {0}'.format(c['gradient_clipping'])]
        if c['epsilon_schedule'] == 'apex':
            ladder = schedules.apex_epsilons(c['collect_envs'], c['initial_epsilon'], c['apex_alpha'])
            parts.append('epsilon ladder {0:.3g} .. {1:.3g} (alpha {2}), {3} over the lanes'.format(
                ladder[0], ladder[-1], c['apex_alpha'], 'permuted' if c['ngu_epsilon_permute'] else 'in order'))
        else:
            parts.append('epsilon linear {0} -> {1} over {2:,} moves'.format(c['initial_epsilon'], c['min_epsilon'], c['epsilon_anneal_steps']))
        return ', '.join(parts)

    # ------------------------------------------------------------ the loop

    def prefill(self):
        """Moves until `initial_collect_steps` rows are banked **and** `ngu_min_windows` windows exist (the paper's
        6,250 sequences before learning)."""
        banked = 0
        need = max(self.config['ngu_min_windows'], self.config['batch_size'])
        while self.buffer.rows < self.config['initial_collect_steps'] or self.buffer.n_windows < need:
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
            started = time.time()
            priorities, _ = self.agent.update(batch, weights)
            self._pace['seconds'] += time.time() - started
            self._pace['updates'] += 1
            self.buffer.update_priorities(slots, priorities)

    # ------------------------------------------------------------ the row

    def fields(self):
        snap = self.collector.snapshot()
        pace = self._pace
        out = {'epsilon': round(self.epsilon, 5),
               'ngu': {'train_step': int(self.agent.train_step), 'windows': int(self.buffer.n_windows),
                       'pending_windows': int(snap['pending_windows']),
                       'intrinsic_mean': round(float(snap['intrinsic_mean']), 5),
                       'memory_mean': round(float(snap['memory_mean']), 1),
                       'updates_per_s': round(pace['updates'] / pace['seconds'], 3) if pace['seconds'] > 0 else 0.0,
                       'rss_gb': round(rss_gb(), 3)}}
        for key in ('episodic', 'alpha'):
            if key in snap:
                out['ngu'][key] = round(float(snap[key]), 5)
        for key in ('td_loss', 'grad_norm', 'embed_loss', 'embed_acc', 'rnd_loss'):
            if key in self.agent.last:
                out['ngu'][key] = round(float(self.agent.last[key]), 5)
        self._pace = {'updates': 0, 'seconds': 0.0}
        return out

    def on_eval(self, eval_rows, measured):
        return {'epsilon': round(self.epsilon, 5)}

    def log_note(self, row):
        return 'eps {0:<7}'.format(row.get('epsilon', '?'))

    def log_extra(self, row):
        r = row.get('ngu') or {}
        if not r:
            return []
        return ['           ngu updates {0:,} ({1}/s)  td {2}  grad {3}  r_i {4} (epi {5} x alpha {6})  embed acc {7}  rnd {8}  '
                'windows {9:,} (+{10} pending)  memory {11}  rss {12} GB'.format(
                    r.get('train_step', 0), r.get('updates_per_s', '?'), r.get('td_loss', '?'), r.get('grad_norm', '?'),
                    r.get('intrinsic_mean', '?'), r.get('episodic', '?'), r.get('alpha', '?'), r.get('embed_acc', '?'),
                    r.get('rnd_loss', '?'), r.get('windows', 0), r.get('pending_windows', 0), r.get('memory_mean', '?'),
                    r.get('rss_gb', '?'))]

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
    return NguAlgo(config, arch, device=device)
