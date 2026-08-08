"""Tests for ForkingCollector's scheduling, which is where the bugs in this feature live.

Run against fakes, not the real game: the collector's job is bookkeeping — who steps next, what gets
snapshotted, which stream a trajectory belongs to, when a branch retires — and a real `Game` would
make every one of those assertions depend on snake geometry. `test_game_snapshot.py` covers the
game-state half.

Two invariants matter more than the rest and are called out in their own tests:

  * **exactly one counted environment step per `run()` call**, or the 1 collect : 1 gradient ratio
    the training loop assumes is silently broken and the arm is uninterpretable;
  * **a terminating branch takes its boundary step before retiring**, because with a two-frame
    window the terminal transition's item is `[terminal, boundary]` — retire early and no branch's
    death or win is ever stored, which would gut the feature while leaving it looking like it works.
"""
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories.time_step import StepType, TimeStep

import forking_collector
from forking_collector import ForkingCollector, validate_config
from prioritized_replay_buffer import DEFAULT_STREAM


def make_time_step(step_type, observation, reward=0.0, discount=0.99):
    return TimeStep(step_type=np.int32(step_type), reward=np.float32(reward),
                    discount=np.float32(discount),
                    observation=np.asarray(observation, dtype=np.float32))


def observation(length=90, safety=(1, 1, 1), tag=0.0):
    """A 30-value vector carrying what the collector reads, plus a tag identifying the environment.

    The tag rides at index 0 — a food observation the collector never looks at — so a fake policy
    can tell *which* branch is asking it for an action. Without that, a test cannot distinguish
    "the branch kept its own shield state" from "every branch happened to agree".
    """
    values = np.zeros(30, dtype=np.float32)
    for index, flag in enumerate(safety):
        values[6 + index] = flag
    values[22] = length / 100.0
    values[0] = tag
    return values


class FakeEnv:
    """Emits MID steps, optionally dying after a set number and auto-resetting like the real one."""

    def __init__(self, name='env', length=90, safety=(1, 1, 1), die_after=None, tag=0.0):
        self.name = name
        self.length = length
        self.safety = safety
        self.die_after = die_after
        self.tag = tag
        self.steps = 0          # since the last restore, which is what `die_after` counts against
        self.total_steps = 0    # over the fake's whole life, so pool reuse does not lose the count
        self.actions = []
        self.restored_from = None
        self.snapshots_taken = 0

    def observation(self):
        return observation(self.length, self.safety, self.tag)

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        self.actions.append(int(np.asarray(action).item()))
        if self.die_after is not None:
            if self.steps == self.die_after:
                return make_time_step(StepType.LAST, self.observation(), reward=-5.0, discount=0.0)
            if self.steps == self.die_after + 1:
                # SnakeEnvironment._step returns self.reset() once the game is finished.
                return make_time_step(StepType.FIRST, self.observation())
        return make_time_step(StepType.MID, self.observation())

    def snake_length(self):
        return self.length

    def snapshot(self):
        self.snapshots_taken += 1
        return ('snapshot-of', self.name, self.steps)

    def restore_from_snapshot(self, snapshot, time_step):
        # Counters reset, because a restore really does replace the game: without this the fake
        # keeps counting across pool reuse and a `die_after` branch dies only once per process,
        # which quietly turns any churn test into a single-branch test.
        self.restored_from = snapshot
        self.steps = 0
        self.actions = []
        return time_step


class FakePolicy:
    """A fixed-action policy. `chosen`, not `action` — the method is called `action`."""

    def __init__(self, chosen=2, guided_flag=None, draw_on_first=None):
        self.chosen = chosen
        self.guided_flag = guided_flag
        self.draw_on_first = draw_on_first
        self.seen_guided = []
        self.seen_by_tag = []
        self.calls = 0

    def action(self, time_step, policy_state=()):
        self.calls += 1
        if self.guided_flag is not None:
            # Mimics ShieldedEpsilonGreedyPolicy._resolve_guided, which redraws the per-episode
            # flag on any FIRST step and otherwise leaves it alone.
            if self.draw_on_first is not None and bool(time_step.is_first()):
                self.guided_flag.assign(self.draw_on_first)
            self.seen_guided.append(bool(self.guided_flag.numpy()))
            self.seen_by_tag.append((float(np.asarray(time_step.observation)[0]),
                                     bool(self.guided_flag.numpy())))
        return policy_step.PolicyStep(action=np.int32(self.chosen), state=policy_state, info=())


class FakeBuffer:
    def __init__(self):
        self.adds = []
        self.forks = []
        self.closed = []

    def add(self, traj, stream=DEFAULT_STREAM):
        self.adds.append((stream, traj))

    def fork_stream(self, parent, child):
        self.forks.append((parent, child))

    def close_stream(self, stream):
        self.closed.append(stream)


def build(max_branches=4, fork_prob=1.0, fork_min_length=85, fork_max_steps=0, length=90,
          safety=(1, 1, 1), die_after=None, policy_action=2, guided_flag=None,
          draw_on_first=None, seed=0, branch_die_after=None):
    main_env = FakeEnv('main', length=length, safety=safety, die_after=die_after, tag=0.0)
    branch_envs = [FakeEnv('branch{0}'.format(index), length=length, safety=safety,
                           die_after=branch_die_after, tag=index + 1.0)
                   for index in range(max(0, max_branches - 1))]
    policy = FakePolicy(policy_action, guided_flag=guided_flag, draw_on_first=draw_on_first)
    buffer = FakeBuffer()
    collector = ForkingCollector(main_env, branch_envs, policy, buffer, max_branches, fork_prob,
                                 fork_min_length, fork_max_steps, guided_flag=guided_flag,
                                 seed=seed)
    start = make_time_step(StepType.MID, observation(length, safety))
    return collector, main_env, branch_envs, policy, buffer, start


def drive(collector, start, calls):
    time_step = start
    for _ in range(calls):
        time_step = collector.run(time_step)
    return time_step


# ------------------------------------------------------------------------- validate_config

def test_validate_config_is_off_by_default():
    # One branch means "the main line only", i.e. today's behaviour and no collector at all.
    assert validate_config(1, 0.5, 85, 60) is False
    assert validate_config(4, 0.5, 85, 60) is True


def test_validate_config_rejects_bad_values():
    cases = {
        'zero branches': (0, 0.5, 85, 60, ()),
        'probability above one': (4, 1.5, 85, 60, ()),
        'negative probability': (4, -0.1, 85, 60, ()),
        'length gate off the board': (4, 0.5, 200, 60, ()),
        'length gate below the opening length': (4, 0.5, 1, 60, ()),
        'negative step cap': (4, 0.5, 85, -1, ()),
        'safety flags ablated': (4, 0.5, 85, 60, (6, 7, 8)),
        'one safety flag ablated': (4, 0.5, 85, 60, (7,)),
    }
    problems = []
    for label, arguments in cases.items():
        try:
            validate_config(*arguments)
        except SystemExit:
            continue
        except Exception as error:
            problems.append('{0}: {1} not SystemExit'.format(label, type(error).__name__))
            continue
        problems.append('{0}: accepted'.format(label))
    assert not problems, problems


def test_validate_config_ignores_an_ablation_outside_the_safety_flags():
    # 26-29 is the documented ablation for the 2026-08-03 blocks and has nothing to do with forking.
    assert validate_config(4, 0.5, 85, 60, (26, 27, 28, 29)) is True


def test_an_off_config_is_not_checked_further():
    # With the feature off, a nonsense probability must not stop an arm launching — the value is
    # unused, and refusing would make the off path harder to reach than the on path.
    assert validate_config(1, 99.0, 0, -5, (6, 7, 8)) is False


# ------------------------------------------------------------------------- eligibility

def test_no_fork_below_the_length_threshold():
    collector, main_env, _, _, buffer, start = build(length=84, fork_min_length=85, fork_prob=1.0)
    drive(collector, start, 5)
    assert buffer.forks == []
    assert main_env.snapshots_taken == 0
    assert collector.counters()['eligible'] == 0


def test_the_length_threshold_is_inclusive():
    collector, _, _, _, buffer, start = build(length=85, fork_min_length=85, fork_prob=1.0)
    drive(collector, start, 1)
    assert buffer.forks, 'length 85 must qualify at a gate of 85'


def test_no_fork_when_only_one_action_is_non_fatal():
    # The common endgame state: one corridor out. Forking needs an *alternative*, so a single safe
    # action is not a branch point — and `> 1` rather than `>= 1` is what says so.
    collector, _, _, _, buffer, start = build(safety=(1, 0, 0), fork_prob=1.0)
    drive(collector, start, 5)
    assert buffer.forks == []
    assert collector.counters()['eligible'] == 0


def test_no_fork_when_no_action_is_safe():
    collector, _, _, _, buffer, start = build(safety=(0, 0, 0), fork_prob=1.0)
    drive(collector, start, 3)
    assert buffer.forks == []


def test_no_fork_on_the_boundary_step():
    """The step after the episode ended is not a decision point, and the game is a corpse.

    A self-collision death leaves the head on top of the body and an out-of-bounds death leaves it
    off the board, so a snapshot taken here describes a state no live game is ever in — measured
    zero overlapping bodies in 2743 live steps against 149 self-collision deaths in 300 episodes.
    Forking here crashed a smoke run inside 10k steps against `validate_snapshot`.

    Asserted by counting snapshots at the boundary specifically, since the fork would raise in
    production but a fake env happily hands out a nonsense snapshot.
    """
    collector, main_env, _, _, buffer, start = build(max_branches=4, die_after=1, fork_prob=1.0)
    time_step = start
    # Call 1: main line acts on a MID step and forks. Call 2 onward includes the boundary step,
    # since the main env returns LAST on its first step here.
    time_step = collector.run(time_step)
    snapshots_after_first = main_env.snapshots_taken
    forks_after_first = len(buffer.forks)
    assert forks_after_first > 0, 'the fixture must fork at least once while alive'

    for _ in range(6):
        time_step = collector.run(time_step)
    assert main_env.snapshots_taken > snapshots_after_first, \
        'the main line should fork again once its new episode is under way'

    # The specific check: no snapshot was taken while the main line's TimeStep was LAST.
    replay = ForkingCollector(FakeEnv('m', die_after=1), [FakeEnv('b1'), FakeEnv('b2'),
                                                          FakeEnv('b3')],
                             FakePolicy(), FakeBuffer(), max_branches=4, fork_prob=1.0,
                             fork_min_length=85, fork_max_steps=0, seed=0)
    replay._main.time_step = make_time_step(StepType.LAST, observation())
    replay._maybe_fork(replay._main, chosen_action=2)
    assert replay._main.env.snapshots_taken == 0, 'a boundary step must not be snapshotted'


def test_a_zero_probability_never_forks_but_still_counts_the_point_as_eligible():
    collector, _, _, _, buffer, start = build(fork_prob=0.0)
    drive(collector, start, 4)
    assert buffer.forks == []
    assert collector.counters()['eligible'] == 4


# ------------------------------------------------------------------------- forking

def test_fork_creates_one_branch_per_untaken_safe_action():
    # Three safe actions, the policy takes 2, so 0 and 1 are handed to branches.
    collector, main_env, branch_envs, _, buffer, start = build(max_branches=4, policy_action=2)
    collector.run(start)
    assert len(buffer.forks) == 2
    assert [parent for parent, _ in buffer.forks] == [DEFAULT_STREAM, DEFAULT_STREAM]
    assert main_env.snapshots_taken == 2, 'one snapshot per fork, taken before the parent steps'
    assert {env.restored_from for env in branch_envs if env.restored_from} == {
        ('snapshot-of', 'main', 0)}
    pending = sorted(branch.pending_action for branch in collector._schedule if not branch.is_main)
    assert pending == [0, 1]


def test_the_untaken_action_set_excludes_the_chosen_one_even_when_it_is_unsafe():
    # The shield never overrides a *greedy* action, so the policy can pick a fatal move. Every safe
    # action is then an alternative worth branching.
    collector, _, _, _, buffer, start = build(max_branches=4, safety=(1, 1, 0), policy_action=2)
    collector.run(start)
    assert len(buffer.forks) == 2
    pending = sorted(branch.pending_action for branch in collector._schedule if not branch.is_main)
    assert pending == [0, 1]


def test_the_cap_counts_the_main_line():
    # max_branches=2 leaves room for exactly one fork alongside the main line.
    collector, _, _, _, buffer, start = build(max_branches=2)
    collector.run(start)
    assert len(buffer.forks) == 1
    assert len(collector._schedule) == 2


def test_no_fork_once_the_cap_is_reached():
    collector, _, _, _, buffer, start = build(max_branches=2)
    drive(collector, start, 8)
    assert len(buffer.forks) == 1, 'forked past the cap'
    assert collector.counters()['skipped_full'] > 0, 'a blocked fork must be counted'


def test_truncated_alternatives_are_not_taken_in_action_order():
    """With room for one of two alternatives, both must occur across seeds.

    `alternatives[:room]` without the shuffle would take them in ACTIONS order every time, which
    over-collects 'left' and under-collects 'right' — a silent statistical bias in exactly the data
    the feature exists to gather.
    """
    seen = set()
    for seed in range(12):
        collector, _, _, _, _, start = build(max_branches=2, policy_action=2, seed=seed)
        collector.run(start)
        seen.update(branch.pending_action for branch in collector._schedule if not branch.is_main)
    assert seen == {0, 1}, seen


def test_a_branch_does_not_itself_fork():
    """Depth 1: a branch of a branch deepens correlation between trajectories rather than coverage.

    The cap must leave headroom or this asserts nothing — at `max_branches=4` with three safe
    actions the main line saturates the schedule on its first call, so a branch would be refused for
    lack of room and the depth rule would never be exercised.
    """
    collector, _, _, _, buffer, start = build(max_branches=12)
    drive(collector, start, 40)
    assert len(buffer.forks) > 3, 'need repeated forking for this to mean anything'
    assert all(parent == DEFAULT_STREAM for parent, _ in buffer.forks), buffer.forks


def test_the_policy_is_not_consulted_for_a_forced_action():
    """A branch's first move is the one it was forked for, taken without asking the policy.

    Asking would defeat the whole mechanism — the branch would re-pick the greedy action and be a
    duplicate of its parent.
    """
    collector, _, _, policy, buffer, start = build(max_branches=4)
    collector.run(start)                      # main forks twice
    calls_before = policy.calls
    forks_before = len(buffer.forks)
    collector.run(collector._main.time_step)  # a branch consumes its pending action
    assert len(buffer.forks) == forks_before
    assert policy.calls == calls_before, 'the policy was consulted for a forced action'


def test_a_branch_takes_its_forced_action_first():
    collector, _, branch_envs, _, _, start = build(max_branches=2, policy_action=2)
    collector.run(start)
    forced = [b.pending_action for b in collector._schedule if not b.is_main][0]
    collector.run(collector._main.time_step)
    stepped = [env for env in branch_envs if env.actions]
    assert len(stepped) == 1
    assert stepped[0].actions == [forced]


# ------------------------------------------------------------------------- scheduling

def test_exactly_one_counted_step_per_run_call():
    """The invariant the training loop rests on: one collect step, one gradient step.

    Counted across a run with branches constantly being created and retiring. If children were
    stepped at fork time, or a retiring branch consumed the iteration, the ratio would drift and no
    comparison against a control arm would mean anything.

    The fixture deliberately makes branches *die* (before the step cap) and makes the main line
    cross an episode boundary, so both free-step paths are exercised — an earlier version capped
    branches below their death step, so no boundary step ever happened and the interesting half of
    the invariant went untested.
    """
    collector, main_env, branch_envs, _, _, start = build(max_branches=4, fork_max_steps=8,
                                                          branch_die_after=2, die_after=15)
    calls = 60
    drive(collector, start, calls)
    counters = collector.counters()
    assert counters['main_steps'] + counters['branch_steps'] == calls
    total_env_steps = main_env.total_steps + sum(env.total_steps for env in branch_envs)
    assert total_env_steps >= calls, 'boundary steps are extra, never fewer'
    assert total_env_steps > calls, 'the fixture must exercise at least one boundary step'


def test_round_robin_advances_each_live_branch_in_turn():
    collector, main_env, branch_envs, _, _, start = build(max_branches=3, fork_prob=1.0)
    drive(collector, start, 9)
    stepped = [main_env.steps] + sorted(env.steps for env in branch_envs if env.steps)
    assert main_env.steps < 9, 'the main line must share the budget'
    assert min(stepped) >= 2, 'every live branch must get turns: {0}'.format(stepped)
    assert max(stepped) - min(stepped) <= 2, 'round robin is unfair: {0}'.format(stepped)


def test_run_returns_the_main_branch_time_step():
    """Whichever branch advanced, the loop gets the main line's TimeStep back.

    `training.py` threads the return value into the next call, so returning a branch's step would
    make the main environment and the loop's idea of it diverge, and the main line would quietly
    start following a fork.
    """
    collector, main_env, _, _, _, start = build(max_branches=4)
    returned = collector.run(start)
    assert returned is collector._main.time_step
    # The main line advanced on this first call, so its TimeStep is the env's latest.
    assert main_env.steps == 1

    before = collector._main.time_step
    returned = collector.run(before)
    assert returned is collector._main.time_step
    assert main_env.steps == 1, 'a branch should have taken this turn'
    assert returned is before, 'the main TimeStep must not change when a branch moves'


# ------------------------------------------------------------------------- retirement

def test_a_terminating_branch_takes_its_boundary_step_then_retires():
    """A dying branch stores its terminal transition *and* the boundary that completes the item.

    Retiring one step early is the single most damaging plausible bug here: everything still runs,
    the buffer still fills, and not one branch death or win is ever learned from.
    """
    collector, _, branch_envs, _, buffer, start = build(max_branches=2, branch_die_after=1)
    time_step = start
    for _ in range(8):
        time_step = collector.run(time_step)
        if buffer.closed:
            break

    assert buffer.closed, 'the branch never retired'
    stream = buffer.closed[0]
    branch_adds = [traj for added_stream, traj in buffer.adds if added_stream == stream]
    step_types = [int(np.asarray(traj.step_type).item()) for traj in branch_adds]
    assert StepType.LAST in step_types, 'the boundary trajectory was never added'
    terminal = [traj for traj in branch_adds
                if int(np.asarray(traj.next_step_type).item()) == StepType.LAST]
    assert terminal, 'the terminal transition was never added'
    assert float(np.asarray(terminal[0].discount).item()) == 0.0
    assert collector.counters()['terminated'] == 1
    assert collector.counters()['retired'] == 1


def test_a_retired_branch_returns_its_environment_to_the_pool():
    # A leaked env means the pool runs dry and _fork raises IndexError partway into a long run.
    collector, _, _, _, buffer, start = build(max_branches=2, branch_die_after=1)
    drive(collector, start, 30)
    assert len(buffer.closed) > 1, 'several branches should have come and gone'
    assert len(collector._pool) + len(collector._schedule) - 1 == 1


def test_the_step_cap_truncates_a_long_lived_branch():
    collector, _, _, _, buffer, start = build(max_branches=2, fork_max_steps=2)
    drive(collector, start, 20)
    assert collector.counters()['truncated'] > 0
    assert collector.counters()['terminated'] == 0
    assert buffer.closed


def test_a_zero_step_cap_lets_a_branch_run_on():
    collector, _, branch_envs, _, _, start = build(max_branches=2, fork_max_steps=0)
    drive(collector, start, 20)
    assert collector.counters()['truncated'] == 0
    assert max(env.steps for env in branch_envs) > 5


def test_the_main_line_never_retires():
    # It *is* train_py_env, and the schedule emptying would hang the training loop.
    collector, _, _, _, buffer, start = build(max_branches=2, die_after=2)
    drive(collector, start, 20)
    assert DEFAULT_STREAM not in buffer.closed
    assert collector._main in collector._schedule


def test_the_main_line_crosses_an_episode_boundary_like_pydriver_does():
    # Its boundary step is free, so the call that crosses it still produces one counted step.
    collector, main_env, _, _, _, start = build(max_branches=1, die_after=2, fork_prob=1.0)
    drive(collector, start, 5)
    assert main_env.steps == 6, 'the boundary step is extra, not counted'
    assert collector.counters()['main_steps'] == 5


def test_stream_ids_are_never_reused():
    collector, _, _, _, buffer, start = build(max_branches=3, branch_die_after=1)
    drive(collector, start, 60)
    children = [child for _, child in buffer.forks]
    assert len(children) > 3, 'need churn for this to mean anything'
    assert len(set(children)) == len(children), children
    assert DEFAULT_STREAM not in children


# ------------------------------------------------------------------------- the shield's flag

def test_the_collector_seeds_the_main_line_from_the_shields_own_flag():
    """Not assumed False. The first call can land mid-episode on a resume, and `_apply_guided`
    writes the branch's value back before the policy reads it — so a wrong initial guess would
    unshield an episode that was meant to be shielded."""
    for initial in (True, False):
        flag = tf.Variable(initial, dtype=tf.bool, trainable=False)
        collector, _, _, policy, _, start = build(max_branches=2, guided_flag=flag)
        collector.run(start)
        assert policy.seen_guided == [initial], policy.seen_guided


def test_a_fork_inherits_the_parent_guided_flag():
    """A fork is the same episode taking a different action, so it is guided iff its parent was.

    Driven far enough that the branch actually *asks* the policy: its first turn consumes the forced
    action without a policy call, so a three-call version of this test never exercised the branch's
    flag at all and a mutation dropping the inheritance survived it.
    """
    flag = tf.Variable(True, dtype=tf.bool, trainable=False)
    collector, _, branch_envs, policy, _, start = build(max_branches=2, guided_flag=flag)
    drive(collector, start, 8)
    assert max(env.steps for env in branch_envs) >= 2, 'the branch never got a policy turn'
    assert policy.seen_guided, 'the policy never read the flag'
    assert all(policy.seen_guided), 'a fork must inherit its parent guided state'


def test_a_branch_keeps_its_own_guided_value_after_the_main_line_redraws():
    """Two live branches disagreeing about the shield is the case that needs per-branch state.

    The main line dies early and redraws the flag to False, while a fork created before that is
    still live and still guided. The flag is a single Variable, so the only thing keeping them apart
    is `_apply_guided` writing each branch's value back before the policy reads it — and this
    asserts it by *tag*, because an earlier version only checked that both values appeared somewhere
    and passed with `_apply_guided` disabled entirely.
    """
    flag = tf.Variable(True, dtype=tf.bool, trainable=False)
    collector, _, _, policy, _, start = build(max_branches=2, guided_flag=flag,
                                              draw_on_first=False, die_after=1)
    drive(collector, start, 10)

    main_sequence = [guided for tag, guided in policy.seen_by_tag if tag == 0.0]
    branch_values = {guided for tag, guided in policy.seen_by_tag if tag != 0.0}
    assert branch_values, 'the branch never asked the policy for an action'
    assert branch_values == {True}, 'the branch lost its inherited value: {0}'.format(
        policy.seen_by_tag)

    assert False in main_sequence, 'the main line never picked up its redrawn value'
    # And it must *keep* it. The redraw is the policy's; the collector has to read it back, or the
    # next step writes the pre-episode value over the top and the main line runs shielded against
    # its own coin flip — invisible except as a guided_fraction that stops mattering.
    after_redraw = main_sequence[main_sequence.index(False):]
    assert len(after_redraw) >= 2, 'the main line needs turns after its redraw: {0}'.format(
        main_sequence)
    assert set(after_redraw) == {False}, 'the redrawn value was overwritten: {0}'.format(
        main_sequence)


def test_a_collector_without_a_shield_works():
    # guided_fraction=0 arms use a plain epsilon-greedy policy with no flag to manage.
    collector, _, _, _, buffer, start = build(max_branches=3, guided_flag=None)
    drive(collector, start, 10)
    assert buffer.forks


# ------------------------------------------------------------------------- wiring

def test_too_few_branch_environments_is_refused_at_construction():
    # Better than an IndexError out of _fork thousands of steps into a run.
    try:
        ForkingCollector(FakeEnv('main'), [FakeEnv('only')], FakePolicy(), FakeBuffer(),
                         max_branches=4, fork_prob=0.5, fork_min_length=85, fork_max_steps=60)
    except ValueError:
        return
    raise AssertionError('a short branch-env pool was accepted')


def test_counters_report_the_branch_share():
    collector, _, _, _, _, start = build(max_branches=3)
    drive(collector, start, 20)
    counters = collector.counters()
    assert counters['main_steps'] + counters['branch_steps'] == 20
    assert 0.0 < counters['branch_share'] < 1.0
    assert counters['live_high_water'] >= 2
    assert counters['forks'] >= 1


def test_the_minimum_gate_length_is_the_opening_length():
    # A gate below it could never be met before the snake has grown, and a gate at it always is.
    assert forking_collector.MIN_GATE_LENGTH == 5
