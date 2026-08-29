"""Train one arm. `python train.py <policy> [--max-steps N]`

    python train.py b1a-baseline
    SNEK_SEED=2 SNEK_FORK_BRANCHES=4 python train.py b1b-fork4

**What lives here is what is not algorithm-specific**: the environment-variable config, seeding, the
`arch.json` sidecar, the checkpoint cadence, the stage-A self-eval, the epsilon and shield schedules'
call sites, the progress chart, the run report and the step cap. The DDQN loop itself is
`dqn/collect.py` plus `dqn/agent.py`, and a later `ppo/` gets its own collector rather than bending
this one — an on-policy rollout and a replay-driven step do not share a loop.

## The two intervals that are not knobs

The self-eval **is** stage A of the measurement (see [`docs/protocol.md`](docs/protocol.md)): its
100 episodes decide which checkpoints a stage-B wave measures at 500. So the eval interval must equal
the checkpoint interval — otherwise a checkpoint exists that no screen can select — and the episode
count must be 100, because the gate is literally "95 of 100". Both are constants below rather than
`tuned()` calls, and `SNEK_GRAPH_EVAL_EPISODES` exists only to make a smoke test cheap.

## Cost, stated plainly

Stage A dominates an arm's wall clock and that is now correct rather than wasteful: a single
checkpoint's 100 episodes drain the measurement engine's lanes with nothing to refill them, so it
runs at ~17 ep/s against ~96 ep/s streamed. A 3M-step arm is ~5 hours, ~90% of it evaluating.
[`docs/runs.md`](docs/runs.md) carries the batching idea that would recover most of it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from dqn import collect
from dqn import schedules
from dqn.agent import DdqnAgent
from dqn.replay import PrioritizedReplay
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import progress_chart
from tools import run_report
from vectorized import engine
from vectorized.vec_env import VecSnake

# ---------------------------------------------------------------- config

def tuned(name, default, cast=float):
    """Reads a hyperparameter from `SNEK_<NAME>`, printing any override.

    Sweeps run several arms side by side from this one file, so what varies between them comes from
    the environment rather than an edit. Everything read here lands in `run_config`, so
    `runs/<arm>.md` records what the run actually used — and `grep 'hyperparameter override:'` on a
    log confirms an arm received its config, which is why this prints rather than returning quietly.
    """
    raw = os.environ.get('SNEK_' + name)
    if raw is None:
        return default
    value = cast(raw)
    print('hyperparameter override: {0} = {1} (default {2})'.format(name, value, default))
    return value


# ---------------------------------------------------------------- the pinned intervals

# **One knob sets both intervals, and that is the point.** The self-eval is stage A of the
# measurement, so a checkpoint written at a step no eval screens is a checkpoint that can never be
# measured. Expressing the two as one value makes them equal by construction rather than by a comment
# asking the next person not to change one of them. It is settable only so a smoke test need not
# train 1,000 steps to reach its first eval; an arm leaves it at 1,000.
EVAL_INTERVAL = int(tuned('EVAL_INTERVAL', 1000, int))
CHECKPOINT_INTERVAL = EVAL_INTERVAL

# 100 because the stage-B gate is literally "95 of 100"; a different denominator is a different gate.
# Lower it for a smoke test and nothing else.
EVAL_EPISODES = int(tuned('GRAPH_EVAL_EPISODES', 100, int))

# The rolling resume state: net, optimiser, step, and the replay buffer beside it. Ten times the
# checkpoint interval because it is ~20 MB against a checkpoint's ~190 KB and it only warm-starts the
# next run, where a checkpoint is evidence.
RESUME_INTERVAL = 10 * EVAL_INTERVAL
RESUME_FILENAME = 'resume.pt'

# The trailing window the checkpoint gate and the chart read. Evals, not steps.
TRAILING_WINDOW = 30

# One log line per this many evals, plus the first and every new best.
QUIET_EVAL_INTERVAL = 25

# The report and the chart are rewritten on their own cadence: both rewrite the whole file from the
# whole series, so doing it per eval would spend a growing fraction of the run on I/O.
REPORT_INTERVAL = 10 * EVAL_INTERVAL


def build_config():
    """Every knob, read once. The returned dict is also what the run report prints.

    **Each key is its `SNEK_` variable, lowercased, without exception.** So a `| priority_exponent |
    0.6 |` row in `runs/<arm>.md` tells you the knob is `SNEK_PRIORITY_EXPONENT` with no lookup — and
    a report row that cannot be grepped back to a knob is a config nobody can reproduce.
    `tests/test_train.py` pins the correspondence.
    """
    fork = collect.ForkConfig(branches=int(tuned('FORK_BRANCHES', 4, int)),
                              prob=tuned('FORK_PROB', 0.5),
                              min_length=int(tuned('FORK_MIN_LENGTH', 85, int)),
                              max_steps=int(tuned('FORK_MAX_STEPS', 60, int)))
    config = {
        'algo': 'dqn',
        'seed': int(tuned('SEED', 1, int)),
        'torch_threads': int(tuned('TORCH_THREADS', 1, int)),
        'max_steps': int(tuned('MAX_STEPS', 10000000, int)),
        # A single hidden layer of 320, which is what every record-holding snek2 arm used — see the
        # `arch.json` beside the imported champions. Phase 3 is a reproduction, so the architecture
        # is not the thing being varied.
        'fc_layers': tuple(int(width) for width in
                           str(tuned('FC_LAYERS', '320', str)).split(',')),
        'learning_rate': tuned('LEARNING_RATE', 1e-5),
        'adam_epsilon': tuned('ADAM_EPSILON', 1e-7),
        'batch_size': int(tuned('BATCH_SIZE', 128, int)),
        'discount': tuned('DISCOUNT', 0.99),
        'n_step_update': int(tuned('N_STEP_UPDATE', 1, int)),
        'target_update_period': int(tuned('TARGET_UPDATE_PERIOD', 8, int)),
        'target_update_tau': tuned('TARGET_UPDATE_TAU', 1.0),
        'gradient_clipping': tuned('GRADIENT_CLIPPING', 0.0),
        'initial_epsilon': tuned('INITIAL_EPSILON', 0.4),
        'min_epsilon': tuned('MIN_EPSILON', 0.002),
        'guided_fraction': tuned('GUIDED_FRACTION', 0.8),
        'collect_envs': int(tuned('COLLECT_ENVS', 1, int)),
        'replay_ratio': tuned('REPLAY_RATIO', 1.0),
        'replay_buffer_max_length': int(tuned('REPLAY_BUFFER_MAX_LENGTH', 100000, int)),
        'initial_collect_steps': int(tuned('INITIAL_COLLECT_STEPS', 2000, int)),
        'priority_exponent': tuned('PRIORITY_EXPONENT', 0.6),
        'is_beta': tuned('IS_BETA', 0.4),
        'is_beta_final': tuned('IS_BETA_FINAL', 1.0),
        'beta_anneal_steps': int(tuned('BETA_ANNEAL_STEPS', 300000, int)),
        'is_weights': bool(int(tuned('IS_WEIGHTS', 1, int))),
        'graph_eval_episodes': EVAL_EPISODES,
        'eval_interval': EVAL_INTERVAL,
        'min_checkpoint_score': constants.MIN_CHECKPOINT_SCORE,
        'fork': fork,
    }
    if config['min_epsilon'] < schedules.EPSILON_HARD_FLOOR:
        raise ValueError('SNEK_MIN_EPSILON={0} is below the hard floor {1}'.format(
            config['min_epsilon'], schedules.EPSILON_HARD_FLOOR))
    if config['replay_ratio'] <= 0.0:
        raise ValueError('SNEK_REPLAY_RATIO={0} must be positive'.format(config['replay_ratio']))
    return config


def reportable(config):
    """`run_config` for the report: the fork object flattened, everything else as-is."""
    out = {key: value for key, value in config.items() if key != 'fork'}
    fork = config['fork']
    out['fork_branches'] = fork.branches
    if fork.enabled:
        out.update({'fork_prob': fork.prob, 'fork_min_length': fork.min_length,
                    'fork_max_steps': fork.max_steps})
    return out


# ---------------------------------------------------------------- the self-eval

def eval_seed(seed, step):
    """A fresh seed per eval, derived from the arm's seed and the step.

    Derived rather than fixed: a fixed eval seed measures the same 100 boards every time, and a
    policy that improves only on those boards would look like a policy that improved. Derived rather
    than random so that a resumed run re-measuring the step it stopped on gets the same boards.
    """
    return int(np.random.SeedSequence([int(seed), int(step)]).generate_state(1, dtype=np.uint32)[0])


def self_eval(agent, step, seed, episodes=EVAL_EPISODES):
    """Stage A: `episodes` greedy episodes, summarised.

    Every field comes from this one sample, including the extremes. snek2 tracked min and max across
    the whole training interval instead and printed them on the same table line as this eval's mean,
    which describes two different things in one row.
    """
    held = engine.measure(agent.policy_fn, episodes, seed=eval_seed(seed, step))
    scores = held['scores']
    return {'avg_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'avg_reward': float(np.mean(held['rewards'])),
            'perfect': float(np.mean(held['perfect']))}


def build_eval_row(step, measured, trailing, epsilon, guided_fraction, steps_per_second,
                   fork_counters=None):
    row = {'step': int(step),
           'avg_score': round(measured['avg_score'], 2),
           'trailing_avg_score': round(trailing, 2),
           'min_score': round(measured['min_score'], 1),
           'max_score': round(measured['max_score'], 1),
           'avg_reward': round(measured['avg_reward'], 3),
           'perfect_percent': round(100.0 * measured['perfect'], 1),
           'epsilon': round(float(epsilon), 5),
           'guided_fraction': round(float(guided_fraction), 3),
           'steps_per_second': round(float(steps_per_second), 1)}
    if fork_counters is not None:
        row['fork'] = fork_counters
    return row


# ---------------------------------------------------------------- the run

class Trainer(object):
    """Owns the loop's state. One instance per process; `run()` returns when the cap is reached."""

    def __init__(self, policy, config, device='cpu'):
        self.policy = policy
        self.config = config
        self.device = device
        self.policy_dir = os.path.join(constants.POLICY_DIR, policy)
        self.history_path = run_report.history_path(constants.RUNS_DIR, policy)
        self.report_path = os.path.join(constants.RUNS_DIR, policy + '.md')
        self.graph_path = progress_chart.chart_path(policy)

        self.arch = self._sidecar()
        self.agent = DdqnAgent(self.arch,
                               learning_rate=config['learning_rate'],
                               adam_epsilon=config['adam_epsilon'],
                               target_update_period=config['target_update_period'],
                               target_update_tau=config['target_update_tau'],
                               gradient_clipping=config['gradient_clipping'],
                               use_is_weights=config['is_weights'],
                               seed=config['seed'], device=device)
        self.buffer = PrioritizedReplay(config['replay_buffer_max_length'], self.arch['obs_len'],
                                        alpha=config['priority_exponent'],
                                        initial_beta=config['is_beta'],
                                        final_beta=config['is_beta_final'],
                                        beta_anneal_steps=config['beta_anneal_steps'],
                                        seed=config['seed'])
        width = config['collect_envs'] * config['fork'].branches
        self.collector = collect.Collector(
            VecSnake(width, seed=config['seed']), self.agent, self.buffer,
            discount=config['discount'], n_step=config['n_step_update'],
            collect_envs=config['collect_envs'], fork=config['fork'],
            guided_fraction=0.0, seed=config['seed'])

        self.step = 0
        self.epsilon = config['initial_epsilon']
        self.eval_rows = []
        self.resume_steps = []
        self.skipped_checkpoints = 0
        self.gradient_debt = 0.0
        self._resume()

    # ------------------------------------------------------------ setup and resume

    def _sidecar(self):
        """The `arch.json` beside the checkpoints: written on a fresh arm, checked on a resume.

        Checked rather than rewritten, because a resume that silently adopted a new
        `SNEK_FC_LAYER_PARAMS` would load the old weights into a different network — which torch
        would refuse, but only after the run had started and only with a shape error.
        """
        built = arch_tools.build_arch(self.config['fc_layers'], constants.NUM_ACTIONS,
                                      constants.OBS_LEN, constants.OBS_ERA, algo=self.config['algo'])
        if os.path.exists(arch_tools.arch_path(self.policy_dir)):
            existing = arch_tools.read_arch(self.policy_dir)
            arch_tools.assert_same_network(built, existing, '<config>', self.policy_dir)
            return existing
        arch_tools.write_arch(self.policy_dir, built)
        return built

    def _resume_path(self):
        return os.path.join(self.policy_dir, RESUME_FILENAME)

    def _resume(self):
        """Picks up net, optimiser, step, buffer and eval history, or starts fresh.

        The buffer is restored under the same condition as the weights and never on its own: a
        resume that paired old weights with a much newer buffer would train the restored network on
        experience it never generated.
        """
        path = self._resume_path()
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.agent.load_state_dict(payload['agent'])
        self.step = int(payload['step'])
        self.epsilon = float(payload['epsilon'])
        self.collector.set_guided_fraction(float(payload.get('guided_fraction', 0.0)))
        loaded = self.buffer.load(self.policy_dir)
        self.eval_rows, resumes = run_report.load_history(self.history_path)
        self.resume_steps = list(resumes) + [self.step]
        print('resumed at step {0:,}: {1} eval row(s), buffer {2}'.format(
            self.step, len(self.eval_rows), 'restored' if loaded else 'empty'), flush=True)

    # ------------------------------------------------------------ the loop

    def run(self):
        cap = self.config['max_steps']
        if self.step >= cap:
            print('already at or past the {0:,}-step cap — nothing to train. Raise SNEK_MAX_STEPS '
                  'to continue this arm.'.format(cap), flush=True)
            return
        self._prefill()
        print('{0}: training to {1:,} steps, {2} lane(s), replay ratio {3}'.format(
            self.policy, cap, self.collector.vec.n, self.config['replay_ratio']), flush=True)

        window_start, window_step = time.time(), self.step
        while self.step < cap:
            transitions = self.collector.step(self.epsilon)
            self.step += 1
            self._learn(transitions)
            if self.step % EVAL_INTERVAL == 0:
                elapsed = max(time.time() - window_start, 1e-9)
                self._evaluate((self.step - window_step) / elapsed)
                window_start, window_step = time.time(), self.step
            if self.step % RESUME_INTERVAL == 0:
                self._save_resume()
            if self.step % REPORT_INTERVAL == 0:
                self._write_report()
        self._save_resume()
        self._write_report()
        print('done at step {0:,}'.format(self.step), flush=True)

    def _prefill(self):
        """Random-ish experience before the first gradient step.

        At the agent's own initial epsilon rather than a uniform policy: with a shielded arm the
        difference is exactly the shield, and pre-filling unshielded would seed the buffer with the
        deaths the shield exists to avoid.
        """
        target = self.config['initial_collect_steps']
        if self.buffer.size >= target:
            return
        while self.buffer.size < target:
            self.collector.step(1.0)

    def _learn(self, transitions):
        """`replay_ratio` gradient steps per *transition*, carrying the fraction across iterations.

        Per transition rather than per iteration, so the ratio holds whether forking is on or off and
        whatever `collect_envs` is. The debt accumulator is what makes a ratio below 1.0 mean what it
        says instead of rounding to zero every iteration.
        """
        self.gradient_debt += transitions * self.config['replay_ratio']
        while self.gradient_debt >= 1.0:
            self.gradient_debt -= 1.0
            drawn = self.buffer.sample(self.config['batch_size'], self.agent.train_step)
            if drawn is None:
                # Nothing to sample yet. The debt already spent stays spent rather than accruing: a
                # buffer that fills on step 3 must not owe three batches at once.
                continue
            batch, indexes, weights = drawn
            td_errors, _ = self.agent.update(batch, weights)
            self.buffer.update_priorities(indexes, td_errors)
            self.agent.maybe_update_target()

    # ------------------------------------------------------------ the eval

    def _evaluate(self, steps_per_second):
        measured = self_eval(self.agent, self.step, self.config['seed'])

        # Both schedules read the *same* reward signal, computed before this row is merged so the
        # window is the last N evals including this one and no row is counted twice. Same signal so
        # the shield switches on at exactly the eval the bootstrap phase hands over on.
        reward_signal = schedules.trailing_reward(self.eval_rows, measured['avg_reward'])
        perfect_rate = schedules.trailing_perfect_rate(self.eval_rows, measured['perfect'])
        self.epsilon = schedules.epsilon_for(reward_signal, perfect_rate,
                                             self.config['initial_epsilon'],
                                             self.config['min_epsilon'])
        guided = schedules.guided_fraction_for(reward_signal, self.config['initial_epsilon'],
                                               self.config['guided_fraction'])
        self.collector.set_guided_fraction(guided)

        trailing = schedules.trailing_mean(self.eval_rows, 'avg_score', measured['avg_score'],
                                           TRAILING_WINDOW)
        keep = self._maybe_checkpoint(measured['avg_score'], trailing)
        row = build_eval_row(self.step, measured, trailing, self.epsilon, guided, steps_per_second,
                             self.collector.snapshot() if self.config['fork'].enabled else None)
        run_report.merge_eval_row(self.eval_rows, row)
        summary = run_report.save_history(self.history_path, self.eval_rows, self.resume_steps)
        self._log(row, summary, keep)

    def _maybe_checkpoint(self, avg_score, trailing):
        """Writes `ckpt-<step>.pt` unless the arm is not worth keeping.

        Gated on `max(this eval, trailing)` rather than trailing alone: the best checkpoints in this
        project are outliers that spike above their neighbourhood, so a trailing-only test could skip
        exactly the one worth keeping while an arm recovers. Either signal clearing the bar is enough.
        """
        keep = max(avg_score, trailing) >= self.config['min_checkpoint_score']
        if keep:
            checkpoints.save(self.policy_dir, self.step, self.agent.net)
        else:
            self.skipped_checkpoints += 1
        return keep

    def _save_resume(self):
        """Net, optimiser, step, schedules — and the buffer beside it, under the same call."""
        payload = {'agent': self.agent.state_dict(), 'step': int(self.step),
                   'epsilon': float(self.epsilon),
                   'guided_fraction': float(self.collector.guided_fraction)}
        path = self._resume_path()
        os.makedirs(self.policy_dir, exist_ok=True)
        staging = path + '.partial'
        torch.save(payload, staging)
        os.replace(staging, path)
        self.buffer.save(self.policy_dir)

    def _write_report(self):
        progress_chart.render(self.eval_rows, self.graph_path, name=self.policy,
                              resume_steps=self.resume_steps)
        run_report.write_run_report(self.report_path, self.policy, reportable(self.config),
                                    self.eval_rows, os.path.basename(self.graph_path),
                                    self.resume_steps)

    def _log(self, row, summary, keep):
        """One line per `QUIET_EVAL_INTERVAL` evals, plus the first and every new best-30.

        `> 0` on the best-30 value matters: with no perfect games yet `best_perfect30` is 0.0 and its
        step falls back to the current step, which would mark every eval as a new best.
        """
        index = self.step // EVAL_INTERVAL
        is_best = (summary['best_perfect30']['value'] > 0
                   and summary['best_perfect30']['step'] == self.step)
        if not (index % QUIET_EVAL_INTERVAL == 0 or is_best or index <= 1):
            return
        note = '  <- best so far' if is_best else ('' if keep else '  no ckpt')
        print('{0:>9,}  score {1:>5.1f}  trail {2:>5.1f}  pf {3:>3.0f}%  best30 {4:>4.1f}%  '
              'eps {5:<7} {6:>5.0f} st/s{7}'.format(
                  row['step'], row['avg_score'], row['trailing_avg_score'], row['perfect_percent'],
                  summary['best_perfect30']['value'], row['epsilon'], row['steps_per_second'], note),
              flush=True)
        if row.get('fork'):
            fork = row['fork']
            print('           forks {0:,}  live {1}  ended {2:,}/trunc {3:,}  '
                  'eligible {4:,}  no slot {5:,}'.format(
                      fork['forks'], fork['live_branches'], fork['terminated'], fork['truncated'],
                      fork['eligible'], fork['skipped_full']), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy', help='arm name; the directory is savedPolicies/<policy>')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='overrides SNEK_MAX_STEPS for this launch')
    parser.add_argument('--device', default='cpu', help='torch device (default cpu)')
    args = parser.parse_args(argv)

    config = build_config()
    if args.max_steps is not None:
        config['max_steps'] = int(args.max_steps)
    # **One thread by default, and it is a measured 1.4x rather than a guess.** The net is
    # 30 -> 320 -> 3; every op is far too small to amortise a fork-join, so torch's default of one
    # thread per core spends more time synchronising than computing — 950 gradient steps/s at 10
    # threads against 1,314 at one. It compounds on the laptop, where four arms run side by side.
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(config['seed'])
    Trainer(args.policy, config, device=args.device).run()


if __name__ == '__main__':
    main()
