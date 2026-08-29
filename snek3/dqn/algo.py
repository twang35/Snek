"""DQN behind the seam `train.py` drives, so the trainer needs to know nothing about DQN.

`train.py` owns what is not algorithm-specific — the config plumbing, seeding, the `arch.json`
sidecar, the checkpoint cadence, stage A and its queue, the chart, the report, the cap. Everything
that is DQN is here or in the four modules this one composes: `agent.py`, `replay.py`, `collect.py`
and `schedules.py`. Nothing in those changed to make this file exist.

**The point of the seam is that PPO reuses the measurement path rather than forking it.** An arm's
numbers are only comparable across algorithms if the same code screened the checkpoints, ran the same
100 episodes, wrote the same rows and drew the same chart — so the one thing that must not be
duplicated for a second algorithm is `train.py`. See [`../plans/ppo.md`](../plans/ppo.md).

## The seam

    algo.step_granularity          the smallest step increment. 1 here; a rollout for PPO
    algo.prefill()          -> n   transitions banked before the first gradient step
    algo.advance()          -> (steps, transitions)   one iteration: collect, then learn
    algo.fields()           -> dict  what this step ran under, captured at the step
    algo.on_eval(rows, measured) -> dict  advance the schedule; return what it just set
    algo.net                       the module a checkpoint holds
    algo.policy_fn                 greedy, for `engine.measure`
    algo.state_dict() / load_state_dict()   what `resume.pt` holds
    algo.save_side_state(dir) / load_side_state(dir)   the replay buffer, which is not a tensor dict
    algo.describe()         -> str  the run's opening line, after "training to N steps, "
    algo.log_note(row)      -> str  the algorithm's column in a progress line
    algo.log_extra(row)     -> [str]  extra lines under it

**`fields()` and `on_eval()` return the same keys and both round at source.** They are the two
spellings of the same quantities — what the arm *ran under*, and what this eval has just *set* — and
`train.py` chooses between them by mode. Rounding here rather than in the row builder is what lets
`train.py` merge an algorithm's fields without knowing that an epsilon wants five decimals.
"""


from dqn import collect
from dqn import schedules
from dqn.agent import DdqnAgent
from dqn.replay import PrioritizedReplay
from vectorized.vec_env import VecSnake

NAME = 'dqn'


def build_config(tuned):
    """The DQN knobs, read through the `tuned` the trainer hands over.

    **`tuned` is a parameter and not an import.** `train.py` imports this module, so importing
    `train.tuned` back would be a cycle; and passing it keeps every `hyperparameter override:` line
    coming out of the one function the docs tell you to grep for.

    Each key is its `SNEK_` variable lowercased, without exception —
    `tests/test_train.py::test_every_config_key_is_its_knob_lowercased` reads this module's source as
    well as `train.py`'s to check it.
    """
    fork = collect.ForkConfig(branches=int(tuned('FORK_BRANCHES', 4, int)),
                              prob=tuned('FORK_PROB', 0.5),
                              min_length=int(tuned('FORK_MIN_LENGTH', 85, int)),
                              max_steps=int(tuned('FORK_MAX_STEPS', 60, int)))
    config = {
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
        'fork': fork,
    }
    if config['min_epsilon'] < schedules.EPSILON_HARD_FLOOR:
        raise ValueError('SNEK_MIN_EPSILON={0} is below the hard floor {1}'.format(
            config['min_epsilon'], schedules.EPSILON_HARD_FLOOR))
    if config['replay_ratio'] <= 0.0:
        raise ValueError('SNEK_REPLAY_RATIO={0} must be positive'.format(config['replay_ratio']))
    return config


def reportable(config):
    """The DQN half of `run_config`, with the fork object flattened into rows a table can print."""
    out = {key: value for key, value in config.items() if key != 'fork'}
    fork = config['fork']
    out['fork_branches'] = fork.branches
    if fork.enabled:
        out.update({'fork_prob': fork.prob, 'fork_min_length': fork.min_length,
                    'fork_max_steps': fork.max_steps})
    return out


def build(config, arch, device='cpu'):
    """The algorithm object `train.py` drives. One factory name per algorithm module, so the trainer
    picks an algorithm out of `ALGOS` without naming a class."""
    return DqnAlgo(config, arch, device=device)


class DqnAlgo(object):
    """Double DQN with prioritised replay, the forking collector and the epsilon/shield schedules."""

    # One counted step is one `collector.step()`. **It is not one game move** — every lane advances,
    # so at the default `fork_branches=4` it is four of them; `train.py` records both.
    step_granularity = 1

    def __init__(self, config, arch, device='cpu'):
        self.config = config
        self.arch = arch
        self.device = device
        self.agent = DdqnAgent(arch,
                               learning_rate=config['learning_rate'],
                               adam_epsilon=config['adam_epsilon'],
                               target_update_period=config['target_update_period'],
                               target_update_tau=config['target_update_tau'],
                               gradient_clipping=config['gradient_clipping'],
                               use_is_weights=config['is_weights'],
                               seed=config['seed'], device=device)
        self.buffer = PrioritizedReplay(config['replay_buffer_max_length'], arch['obs_len'],
                                        alpha=config['priority_exponent'],
                                        initial_beta=config['is_beta'],
                                        final_beta=config['is_beta_final'],
                                        beta_anneal_steps=config['beta_anneal_steps'],
                                        seed=config['seed'])
        width = config['collect_envs'] * config['fork'].branches
        # **`shaping_discount` is the agent's `discount`, and passing it is not optional.**
        # Potential-based shaping pays `c * (gamma * Phi(s') - Phi(s))`, and the invariance theorem
        # that makes `c` free holds only when that gamma is the one the agent discounts with. Left at
        # `VecSnake`'s default of 1.0 — which is what b1 and b2 ran — the telescope does not close:
        # every step keeps `c * (1 - gamma) * Phi(s')` of un-cancelled potential, which at b2's
        # `c=0.1` and `gamma=0.9975` is 2.5e-4 a step. That is the same order as
        # `FOOD_DISTANCE_REWARD` and it is a standing bonus for being alive in a high-potential
        # state, so the shaping was no longer policy-invariant.
        #
        # **It is a no-op for an arm with shaping off**, which is the default and every b1-class arm:
        # `VecSnake.step` only reaches `_shaping_reward` when a coefficient is non-zero, so nothing
        # multiplies this and a shaping-free arm stays bit-identical. Verified by fingerprinting three
        # arms before and after: the two shaping arms moved and the default one did not.
        self.collector = collect.Collector(
            VecSnake(width, seed=config['seed'], shaping_discount=config['discount']),
            self.agent, self.buffer,
            discount=config['discount'], n_step=config['n_step_update'],
            collect_envs=config['collect_envs'], fork=config['fork'],
            guided_fraction=0.0, seed=config['seed'])

        self.epsilon = config['initial_epsilon']
        self.gradient_debt = 0.0

    # ------------------------------------------------------------ what a checkpoint and an eval see

    @property
    def net(self):
        return self.agent.net

    @property
    def policy_fn(self):
        return self.agent.policy_fn

    def describe(self):
        return '{0} lane(s), replay ratio {1}'.format(self.collector.vec.n,
                                                      self.config['replay_ratio'])

    # ------------------------------------------------------------ the loop

    def prefill(self):
        """Random-ish experience before the first gradient step. Returns the transitions banked.

        At the agent's own initial epsilon rather than a uniform policy: with a shielded arm the
        difference is exactly the shield, and pre-filling unshielded would seed the buffer with the
        deaths the shield exists to avoid.
        """
        target = self.config['initial_collect_steps']
        banked = 0
        while self.buffer.size < target:
            banked += self.collector.step(1.0)
        return banked

    def advance(self):
        """One collect step and the gradient steps it bought. Returns `(steps, transitions)`.

        Two numbers rather than one because they are different quantities here: a counted step is one
        `collector.step()` and a transition is one game move, and at the default width they differ by
        four. `train.py` caps on the first and charts both.
        """
        transitions = self.collector.step(self.epsilon)
        self._learn(transitions)
        return 1, transitions

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
            # No `maybe_update_target()` here: `agent.update` already calls it, and the agent is the
            # only thing that knows `train_step`, which is what the period is counted in. Calling it
            # from both places was a no-op at the default `tau` of 1.0 — a second hard copy of weights
            # that were just copied — but at `tau < 1.0` it applied the Polyak step twice at the same
            # train_step, so a requested 0.05 ran at 1 - (1 - 0.05)^2 = 0.0975. The period itself was
            # never affected, because `maybe_update_target` gates on `train_step` rather than counting
            # its own calls.

    # ------------------------------------------------------------ the row, and the schedules

    def fields(self):
        """What the arm was **using** while it collected the step being evaluated.

        With the stage-A queue on this is the whole reason a row can still be built correctly several
        intervals later: by then the schedule has moved, and reading these at merge time would label a
        step with an epsilon it never ran under. The fork counters are the same kind of thing — a
        window of training that is over and cannot be reconstructed.
        """
        return {'epsilon': round(float(self.epsilon), 5),
                'guided_fraction': round(float(self.collector.guided_fraction), 3),
                'fork': self.collector.snapshot() if self.config['fork'].enabled else None}

    def on_eval(self, eval_rows, measured):
        """Moves the epsilon and shield schedules on this eval, and returns what they became.

        Both schedules read the *same* reward signal, computed before this row is merged so the
        window is the last N evals including this one and no row is counted twice. Same signal so the
        shield switches on at exactly the eval the bootstrap phase hands over on.

        Called in **both** queue modes, because it is what advances the schedule; which of its values
        reaches the row is `train.py`'s decision and is documented there.
        """
        reward_signal = schedules.trailing_reward(eval_rows, measured['avg_reward'])
        perfect_rate = schedules.trailing_perfect_rate(eval_rows, measured['perfect'])
        self.epsilon = schedules.epsilon_for(reward_signal, perfect_rate,
                                             self.config['initial_epsilon'],
                                             self.config['min_epsilon'])
        guided = schedules.guided_fraction_for(reward_signal, self.config['initial_epsilon'],
                                               self.config['guided_fraction'])
        self.collector.set_guided_fraction(guided)
        return {'epsilon': round(float(self.epsilon), 5),
                'guided_fraction': round(float(guided), 3)}

    # ------------------------------------------------------------ the log

    def log_note(self, row):
        return 'eps {0:<7}'.format(row['epsilon'])

    def log_extra(self, row):
        if not row.get('fork'):
            return []
        fork = row['fork']
        return ['           forks {0:,}  live {1}  ended {2:,}/trunc {3:,}  '
                'eligible {4:,}  no slot {5:,}'.format(
                    fork['forks'], fork['live_branches'], fork['terminated'], fork['truncated'],
                    fork['eligible'], fork['skipped_full'])]

    # ------------------------------------------------------------ persistence

    def state_dict(self):
        return {'agent': self.agent.state_dict(),
                'epsilon': float(self.epsilon),
                'guided_fraction': float(self.collector.guided_fraction)}

    def load_state_dict(self, state):
        """Restores the agent and both schedules.

        **Reads a pre-seam `resume.pt` too.** Those files hold `agent`, `epsilon` and
        `guided_fraction` at the top level rather than under `algo`, and `train.py` hands whichever
        it found straight here. Refusing them would strand b1's and b2's resume files for no gain;
        a wrong *silent* read is what this cannot do, and it cannot, because the key names are the
        same in both layouts.
        """
        self.agent.load_state_dict(state['agent'])
        self.epsilon = float(state['epsilon'])
        self.collector.set_guided_fraction(float(state.get('guided_fraction', 0.0)))

    def save_side_state(self, policy_dir):
        """The replay buffer, which is a numpy archive rather than a tensor dict.

        Saved under the same call as the weights and never on its own: a resume that paired old
        weights with a much newer buffer would train the restored network on experience it never
        generated.
        """
        self.buffer.save(policy_dir)

    def load_side_state(self, policy_dir):
        """Returns whether a buffer came back, for the resume line the trainer prints."""
        return self.buffer.load(policy_dir)
