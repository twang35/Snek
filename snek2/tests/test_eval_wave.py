"""Fixtures for the wave controller: the parts a live run cannot reach.

`eval_wave.py`'s flat path is verified against `eval_checkpoints.py`'s own output on the same
explicit step list — same payload keys, same key order, same row keys — and that comparison is the
real gate on the arithmetic. What it cannot reach is the three-stage screened protocol: every arm in
`savedPolicies/` with a current observation era is a *continuation* arm whose entire selection sits
in the mandatory full-length tier, so `top<N>` on real data never produces a screen to confirm. The
screened path, the confirm barrier and the failure handling therefore live here, driven by a fake
pool.

Two properties are worth stating because they are what the design rests on:

  * **The controller is the only writer.** A lane hands back a finished measurement and never
    touches an `Arm`, so `test_a_lane_never_writes_to_the_arm` pins the boundary rather than trusting
    it.
  * **The confirm barrier must advance on failure too.** A screen that raises would otherwise leave
    its arm waiting forever for a unit that will never arrive, and the whole wave hangs with its
    other lanes idle — the exact failure the design exists to prevent, arriving through the error
    path.
"""
import json
import os
import shutil
import tempfile
import time

import eval_plan
import eval_wave
import policy_arch

ARCH = policy_arch.build_arch([64, 64], 4, 30, 'b09c616')


class FakePool:
    """Stands in for `IndependentWorkerPool`: same two calls, no TensorFlow and no subprocesses.

    `rates` maps a step to the fraction of its episodes that are perfect, so a fixture can say
    "this checkpoint is the best one" and then assert the confirm stage picked it.
    """

    def __init__(self, rates=None, fail_on=None, delay=0.0):
        self.rates = rates or {}
        self.fail_on = fail_on or set()
        # A measurement that takes no time at all is not a useful model of a lane: the first thread
        # to start drains the queue before the others reach it, so a fixture about *sharing* has to
        # make a unit take long enough for a second lane to claim one.
        self.delay = delay
        self.loads = []
        self.runs = []

    def load(self, step, ckpt_dir=None):
        self.loads.append((step, ckpt_dir))
        return step

    def run(self, episodes, on_progress=None, should_abandon=None):
        step = self.loads[-1][0]
        if step in self.fail_on:
            raise RuntimeError('fake worker failure on {0}'.format(step))
        self.runs.append((step, episodes))
        if self.delay:
            time.sleep(self.delay)
        rate = self.rates.get(step, 0.5)
        perfect = [1 if index < round(rate * episodes) else 0 for index in range(episodes)]
        scores = [95 if flag else 40 for flag in perfect]
        if on_progress is not None:
            on_progress(1, 1, sum(perfect), episodes, [sum(perfect)])
        return scores, perfect, [float(s) for s in scores], 1.0, False

    def close(self):
        pass


class Bench:
    """A temp `savedPolicies/` and `runs/`, with `eval_wave` and `eval_plan` pointed at them.

    Both modules have to be redirected: `Arm` builds its paths from `eval_wave`'s globals while
    `load_finished_results` and the selectors read `eval_plan`'s. Patching one and not the other is
    how a fixture ends up writing into the real `runs/`.
    """

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix='wavetest-')
        self.policies = os.path.join(self.root, 'savedPolicies') + os.sep
        self.runs = os.path.join(self.root, 'runs')
        self.evals = os.path.join(self.root, 'evals')
        for path in (self.policies, self.runs, self.evals):
            os.makedirs(path, exist_ok=True)
        self.saved = {}

    def __enter__(self):
        for module, name, value in ((eval_wave, 'POLICY_DIR', self.policies),
                                    (eval_wave, 'RUNS_DIR', self.runs),
                                    (eval_wave, 'EVALS_DIR', self.evals),
                                    (eval_plan, 'RUNS_DIR', self.runs),
                                    (eval_plan, 'EVALS_DIR', self.evals)):
            self.saved[(module, name)] = getattr(module, name)
            setattr(module, name, value)
        return self

    def __exit__(self, *exc):
        for (module, name), value in self.saved.items():
            setattr(module, name, value)
        shutil.rmtree(self.root, ignore_errors=True)
        return False

    def policy(self, name, steps=(1000, 2000, 3000), arch=None):
        path = os.path.join(self.policies, name)
        policy_arch.write_arch(path, arch or ARCH)
        for step in steps:
            open(os.path.join(path, 'ckpt-{0}.index'.format(step)), 'w').close()
        return name

    def arm(self, name, requested, selected_by=None, num_episodes=100, screen_episodes=20,
            confirm_count=2, num_workers=4, min_achievable=0.0):
        arm = eval_wave.Arm(name, '', num_episodes, num_workers, screen_episodes, confirm_count,
                            min_achievable, 20, None, None)
        arm.finalise_plan(list(requested), selected_by or {
            step: {'selected_by': 'graph', 'single_eval': 90.0} for step in requested})
        return arm


def graph_meta(step, single):
    return {'selected_by': 'graph', 'single_eval': single, 'surrounding': single}


# ---------------------------------------------------------------- argv and naming

def test_parse_selector_reads_the_three_forms():
    assert eval_wave.parse_selector(['top50', 'b44'])[:2] == ('top', 50)
    assert eval_wave.parse_selector(['above:98', 'b44'])[:2] == ('above', 98.0)
    kind, value, rest = eval_wave.parse_selector(['1000', '2000', 'b44a-x'])
    assert (kind, value, rest) == ('explicit', [1000, 2000], ['b44a-x'])


def test_parse_selector_defaults_match_the_protocol_constants():
    # Spelled bare, both selectors mean what the close-out and the HOF pass mean by them.
    assert eval_wave.parse_selector(['top'])[1] == eval_plan.DEFAULT_COUNT
    assert eval_wave.parse_selector(['above'])[1] == eval_plan.DEFAULT_ABOVE_THRESHOLD


def test_batch_of_does_not_confuse_b4_with_b44():
    # `'b44a-x'.startswith('b4')` is true, which is the trap already fixed once in
    # chart_viewer.live_arms. Matching on the batch id is the fix in both places.
    assert eval_wave.batch_of('b44a-lowlr7-b29b') == 'b44'
    assert eval_wave.batch_of('b4a-uniform') == 'b4'
    assert eval_wave.batch_of('champion_x') == 'champion_x'


def test_arms_for_prefix_expands_a_batch_and_excludes_its_neighbours():
    with Bench() as bench:
        for name in ('b44a-x', 'b44b-y', 'b4a-old', 'b43a-other'):
            bench.policy(name)
        assert eval_wave.arms_for_prefix('b44') == ['b44a-x', 'b44b-y']
        assert eval_wave.arms_for_prefix('b4') == ['b4a-old']


def test_resolve_policies_takes_names_batches_and_rejects_flags():
    with Bench() as bench:
        bench.policy('b44a-x')
        bench.policy('b44b-y')
        assert eval_wave.resolve_policies(['b44a-x']) == ['b44a-x']
        assert eval_wave.resolve_policies(['b44']) == ['b44a-x', 'b44b-y']
        # A batch and one of its arms must not name the same arm twice.
        assert eval_wave.resolve_policies(['b44', 'b44a-x']) == ['b44a-x', 'b44b-y']
        for bad in (['--top', '50'], ['b99']):
            try:
                eval_wave.resolve_policies(bad)
            except SystemExit:
                continue
            raise AssertionError('accepted {0!r}'.format(bad))


# ---------------------------------------------------------------- eligibility

def test_lane_key_ignores_a_missing_algo_and_the_win_reward():
    # b43's own sidecars are non-uniform: three predate `algo`/`perfect_game_reward` and one has
    # both. A raw `.get('algo')` split one batch into two lane groups and halved the balancing;
    # `restore_signature` normalises through `algo_of`, and the win reward is an objective rather
    # than a property of the weights, so it is excluded.
    with Bench() as bench:
        old = dict(ARCH)
        old.pop('algo', None)
        old.pop('perfect_game_reward', None)
        rich = dict(ARCH, perfect_game_reward=10)
        bench.policy('b43a-x', arch=old)
        bench.policy('b43b-y', arch=rich)
        a, b = bench.arm('b43a-x', [1000]), bench.arm('b43b-y', [1000])
        assert eval_wave.lane_key(a) == eval_wave.lane_key(b)


def test_lane_key_separates_a_different_observation_length():
    with Bench() as bench:
        bench.policy('b43a-x')
        bench.policy('b40a-y', arch=policy_arch.build_arch([64, 64], 4, 26, '450e66e'))
        a, b = bench.arm('b43a-x', [1000]), bench.arm('b40a-y', [1000])
        assert eval_wave.lane_key(a) != eval_wave.lane_key(b)


def test_lane_key_separates_arms_measured_under_different_shaping():
    # Not about restoring — a shaped and an unshaped arm restore identically. It is about
    # `avg_reward`: measured under the other arm's shaping it is a number that has quietly stopped
    # meaning what it says.
    with Bench() as bench:
        bench.policy('b43a-x')
        arm = bench.arm('b43a-x', [1000])
        before = eval_wave.lane_key(arm)
        os.environ['SNEK_CHASE_SAFE_SHAPING'] = '0.10'
        try:
            assert eval_wave.lane_key(arm) != before
        finally:
            del os.environ['SNEK_CHASE_SAFE_SHAPING']


# ---------------------------------------------------------------- the queue

def test_the_queue_hands_out_full_length_work_before_screens():
    with Bench() as bench:
        bench.policy('b44a-x')
        # The full-tier step is deliberately the *highest* one: ordered by step alone this comes
        # out screen-screen-full, so a fixture whose full tier is also its lowest step would pass
        # with the stage ordering removed entirely.
        arm = bench.arm('b44a-x', [1000, 2000, 3000], selected_by={
            1000: graph_meta(1000, 90.0), 2000: graph_meta(2000, 90.0),
            3000: graph_meta(3000, 100.0)})
        queue = eval_wave.WaveQueue()
        queue.add(arm.initial_units())
        taken = []
        while True:
            unit = queue.take(eval_wave.lane_key(arm))
            if unit is None:
                break
            taken.append((unit.stage, unit.step))
        assert taken == [('full', 3000), ('screen', 1000), ('screen', 2000)], taken


def test_the_queue_rotates_over_the_arms_within_a_stage():
    # b43's HOF selected 166 / 607 / 133 / 83 checkpoints. Arm-major order spends its first hours
    # inside one arm, so three of four panels read "nothing measured yet" for hours and an early stop
    # leaves three arms unmeasured -- when comparing the arms is the whole point of the batch.
    with Bench() as bench:
        arms = []
        for name in ('b44a-x', 'b44b-y', 'b44c-z'):
            bench.policy(name)
            arms.append(bench.arm(name, [1000, 2000, 3000]))
        queue = eval_wave.WaveQueue()
        for arm in arms:
            queue.add(arm.initial_units())
        taken = []
        while True:
            unit = queue.take(eval_wave.lane_key(arms[0]))
            if unit is None:
                break
            taken.append((unit.arm.policy_name, unit.step))
        assert [name for name, _ in taken] == ['b44a-x', 'b44b-y', 'b44c-z'] * 3, taken
        assert [step for _, step in taken[:3]] == [1000, 1000, 1000], taken


def test_rotation_never_reorders_the_stages():
    # The one ordering that is not free: a confirmation ranks the screens it follows.
    with Bench() as bench:
        for name in ('b44a-x', 'b44b-y'):
            bench.policy(name)
        a = bench.arm('b44a-x', [1000, 3000], selected_by={
            1000: graph_meta(1000, 90.0), 3000: graph_meta(3000, 100.0)})
        b = bench.arm('b44b-y', [1000, 3000], selected_by={
            1000: graph_meta(1000, 90.0), 3000: graph_meta(3000, 100.0)})
        queue = eval_wave.WaveQueue()
        queue.add(a.initial_units())
        queue.add(b.initial_units())
        stages = []
        while True:
            unit = queue.take(eval_wave.lane_key(a))
            if unit is None:
                break
            stages.append(unit.stage)
        assert stages == ['full', 'full', 'screen', 'screen'], stages


def test_rotation_does_not_depend_on_the_order_the_arms_were_added():
    with Bench() as bench:
        for name in ('b44a-x', 'b44b-y', 'b44c-z'):
            bench.policy(name)
        made = [bench.arm(n, [1000, 2000]) for n in ('b44a-x', 'b44b-y', 'b44c-z')]

        def drain(order):
            queue = eval_wave.WaveQueue()
            for arm in order:
                queue.add(arm.initial_units())
            out = []
            while True:
                unit = queue.take(eval_wave.lane_key(made[0]))
                if unit is None:
                    return out
                out.append((unit.arm.policy_name, unit.step))

        assert drain(made) == drain(list(reversed(made)))


def test_interleave_is_stable_however_the_units_arrive():
    # The ranking runs over a sorted copy, not over insertion order. With the current callers the
    # two agree -- `add` is handed each arm's units in step order -- so this tests the function's
    # contract directly rather than through the queue, where the difference is invisible until a
    # caller changes.
    with Bench() as bench:
        for name in ('b44a-x', 'b44b-y', 'b44c-z'):
            bench.policy(name)
        arms = [bench.arm(n, [1000, 2000, 3000]) for n in ('b44a-x', 'b44b-y', 'b44c-z')]
        units = [u for arm in arms for u in arm.initial_units()]
        shape = lambda seq: [(u.arm.policy_name, u.step) for u in eval_wave.interleave_by_arm(seq)]
        assert shape(list(reversed(units))) == shape(units)
        assert [step for _, step in shape(list(reversed(units)))[:3]] == [1000, 1000, 1000]


def test_the_queue_never_hands_out_the_same_unit_twice():
    with Bench() as bench:
        bench.policy('b44a-x')
        arm = bench.arm('b44a-x', [1000, 2000])
        units = arm.initial_units()
        queue = eval_wave.WaveQueue()
        queue.add(units)
        queue.add(units)                        # a re-issue, e.g. a retried confirm barrier
        assert len(queue) == len(units)


def test_a_lane_only_takes_work_it_is_eligible_for():
    with Bench() as bench:
        bench.policy('b43a-x')
        bench.policy('b40a-y', arch=policy_arch.build_arch([64, 64], 4, 26, '450e66e'))
        mine, theirs = bench.arm('b43a-x', [1000]), bench.arm('b40a-y', [1000])
        queue = eval_wave.WaveQueue()
        queue.add(theirs.initial_units())
        assert queue.take(eval_wave.lane_key(mine)) is None
        assert len(queue) == 1                  # still there for a lane that can run it
        assert queue.take(eval_wave.lane_key(theirs)) is not None


# ---------------------------------------------------------------- top-ups

def test_a_partial_sample_is_topped_up_not_restarted():
    with Bench() as bench:
        bench.policy('b44a-x')
        arm = bench.arm('b44a-x', [1000], selected_by={1000: graph_meta(1000, 100.0)})
        arm.samples[1000] = {'scores': [95] * 60, 'perfect': [1] * 60, 'rewards': [1.0] * 60,
                             'seconds': 1.0}
        units = arm.initial_units()
        assert [(u.step, u.episodes) for u in units] == [(1000, 40)]


def test_a_sample_already_at_full_length_produces_no_unit():
    with Bench() as bench:
        bench.policy('b44a-x')
        arm = bench.arm('b44a-x', [1000], selected_by={1000: graph_meta(1000, 100.0)})
        arm.samples[1000] = {'scores': [95] * 100, 'perfect': [1] * 100, 'rewards': [1.0] * 100,
                             'seconds': 1.0}
        assert arm.initial_units() == []


# ---------------------------------------------------------------- the wave, end to end

def run_wave(bench, arms, lanes=2, pools=None):
    """Drives a real `Wave` with `FakePool` lanes, then returns it.

    Deliberately the real `Wave.run` loop and the real `Lane` threads — the scheduling, the barrier
    and the writing are what these fixtures are about, and a hand-rolled loop would test itself.
    """
    wave = eval_wave.Wave(arms, lanes, 4)
    for index in range(lanes):
        pool = (pools or [None] * lanes)[index] or FakePool()
        wave.lanes.append(eval_wave.Lane(index, pool, arms[0].policy_name, wave.queue,
                                         eval_wave.lane_key(arms[0]), wave.outbox))
    wave.chart['off'] = True                    # no matplotlib in a unit test
    wave.enqueue_initial()
    wave.run()
    wave.finish()
    return wave


def test_a_screened_arm_runs_all_three_stages_and_confirms_the_best_screens():
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000, 3000, 4000))
        arm = bench.arm('b44a-x', [1000, 2000, 3000, 4000], confirm_count=2, selected_by={
            1000: graph_meta(1000, 100.0), 2000: graph_meta(2000, 90.0),
            3000: graph_meta(3000, 90.0), 4000: graph_meta(4000, 90.0)})
        pool = FakePool(rates={1000: 1.0, 2000: 0.2, 3000: 0.9, 4000: 1.0})
        wave = run_wave(bench, [arm], lanes=1, pools=[pool])
        assert not wave.failures
        payload = json.load(open(arm.out_path))
        assert payload['complete'] is True
        assert payload['stages']['full'] == {'planned': 1, 'done': 1}
        assert payload['stages']['screen'] == {'planned': 3, 'done': 3}
        assert payload['stages']['confirm']['done'] == 2
        # The two best screens, not the two lowest steps: 2000 screened at 20% and must lose.
        confirmed = {step for step, episodes in pool.runs if episodes == 80}
        assert confirmed == {3000, 4000}, confirmed
        depths = {row['step']: row['episodes'] for row in payload['results']}
        assert depths == {1000: 100, 2000: 20, 3000: 100, 4000: 100}, depths


def test_confirm_units_are_withheld_until_every_screen_has_landed():
    # The barrier `pick_finalists` needs: it ranks the screened rows against each other, so a
    # confirm issued early would rank against a partial field. Checked by holding one screen back
    # rather than by reading the code.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000, 3000))
        arm = bench.arm('b44a-x', [1000, 2000, 3000], confirm_count=1, selected_by={
            step: graph_meta(step, 90.0) for step in (1000, 2000, 3000)})
        wave = eval_wave.Wave([arm], 1, 4)
        wave.chart['off'] = True
        wave.enqueue_initial()
        assert arm.screens_expected == 3
        # 19/20, not 20/20: a perfect screen is confirmed *regardless* of the quota, so two of
        # those would issue two confirm units and the count below would not be the quota's.
        for step in (1000, 2000):
            wave.on_done(eval_wave.Unit(arm, step, 20, 'screen'),
                         [95] * 19 + [40], [1] * 19 + [0], [1.0] * 20, 1.0, False, 0, 0)
            assert not arm.confirm_issued, 'confirmed after {0} of 3 screens'.format(step)
        before = wave.issued
        wave.on_done(eval_wave.Unit(arm, 3000, 20, 'screen'),
                     [40] * 20, [0] * 20, [0.0] * 20, 1.0, False, 0, 0)
        # The screens were folded in directly rather than taken, so they are still pending —
        # `issued` is what the confirm stage moves, not the queue's length.
        assert arm.confirm_issued and wave.issued == before + 1


def test_a_failed_screen_still_advances_the_confirm_barrier():
    # Otherwise the arm waits forever for a unit that will never arrive and the wave hangs with its
    # other lanes idle — the exact failure this design exists to prevent, through the error path.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000))
        arm = bench.arm('b44a-x', [1000, 2000], confirm_count=1, selected_by={
            step: graph_meta(step, 90.0) for step in (1000, 2000)})
        wave = eval_wave.Wave([arm], 1, 4)
        wave.chart['off'] = True
        wave.enqueue_initial()
        wave.on_done(eval_wave.Unit(arm, 1000, 20, 'screen'),
                     [95] * 20, [1] * 20, [1.0] * 20, 1.0, False, 0, 0)
        wave.on_error(eval_wave.Unit(arm, 2000, 20, 'screen'), 0, 'boom')
        assert arm.confirm_issued


def test_an_arm_whose_lane_failed_is_not_marked_complete():
    # `select_checkpoints_above` refuses to select from an incomplete file, which is what stops a
    # HOF pass reading a truncated close-out as if it were the whole arm.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000))
        arm = bench.arm('b44a-x', [1000, 2000], screen_episodes=0, selected_by={
            step: graph_meta(step, 100.0) for step in (1000, 2000)})
        run_wave(bench, [arm], lanes=1, pools=[FakePool(fail_on={2000})])
        assert json.load(open(arm.out_path))['complete'] is False


def test_two_arms_share_the_lanes_and_one_lane_switches_arms():
    # The point of the whole design: lanes are not tied to arms, so a one-checkpoint arm does not
    # own a lane while a ten-checkpoint arm queues behind its own.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        bench.policy('b44b-y', steps=tuple(range(1000, 9000, 1000)))
        small = bench.arm('b44a-x', [1000], screen_episodes=0,
                          selected_by={1000: graph_meta(1000, 100.0)})
        big = bench.arm('b44b-y', list(range(1000, 9000, 1000)), screen_episodes=0,
                        selected_by={s: graph_meta(s, 100.0) for s in range(1000, 9000, 1000)})
        wave = run_wave(bench, [small, big], lanes=2,
                        pools=[FakePool(delay=0.02), FakePool(delay=0.02)])
        assert not wave.failures
        assert wave.done == 9 and wave.issued == 9
        assert sum(lane.units_run for lane in wave.lanes) == 9
        # Both lanes worked, and at least one crossed from one arm to the other — which is the
        # whole design: a one-checkpoint arm does not hold a lane while an eight-checkpoint arm
        # queues behind its own.
        assert all(lane.units_run for lane in wave.lanes), [l.units_run for l in wave.lanes]
        assert sum(lane.switches for lane in wave.lanes) >= 1
        for arm in (small, big):
            assert json.load(open(arm.out_path))['complete'] is True


def test_a_lane_never_writes_to_the_arm():
    # One writer per file, by construction. A lane that folded its own result in would race the
    # controller's `build_row`, and the symptom would be a corrupt row rather than an exception.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        arm = bench.arm('b44a-x', [1000], screen_episodes=0,
                        selected_by={1000: graph_meta(1000, 100.0)})
        wave = eval_wave.Wave([arm], 1, 4)
        wave.chart['off'] = True
        lane = eval_wave.Lane(0, FakePool(), 'b44a-x', wave.queue,
                              eval_wave.lane_key(arm), wave.outbox)
        wave.enqueue_initial()
        unit = wave.queue.take(lane.key)
        lane._measure(unit)
        assert arm.samples.get(1000, {}).get('scores') in (None, [])
        assert not os.path.exists(arm.out_path + '.partial')
        kind, got, index, payload = wave.outbox.get_nowait()      # the round report
        assert kind == 'round'
        kind, got, index, payload = wave.outbox.get_nowait()
        assert kind == 'done' and got is unit


# ---------------------------------------------------------------- the payload

IN_FLIGHT_KEYS = {'step', 'round', 'rounds_total', 'perfect_so_far', 'episodes_so_far',
                  'episodes_this_pass', 'running_percent', 'per_round_perfect', 'started_at'}


def test_the_in_flight_block_carries_exactly_the_keys_eval_progress_reads():
    # `eval_progress.py` draws the live ETA from these by name, so an extra key is as wrong as a
    # missing one — the controller's own bookkeeping (`queued`, the `_`-prefixed tallies) must not
    # reach the file.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        arm = bench.arm('b44a-x', [1000], screen_episodes=0,
                        selected_by={1000: graph_meta(1000, 100.0)})
        wave = eval_wave.Wave([arm], 1, 4)
        wave.chart['off'] = True
        wave.enqueue_initial()
        wave.on_round(eval_wave.Unit(arm, 1000, 100, 'flat'), 3, 25, 2, 12, [1, 1, 0])
        block = json.load(open(arm.out_path))['in_flight']
        assert set(block) == IN_FLIGHT_KEYS, sorted(set(block) ^ IN_FLIGHT_KEYS)
        assert block['episodes_this_pass'] == 12 and block['running_percent'] == 16.7


def test_a_topped_up_checkpoint_reports_its_whole_sample_while_running():
    # A confirm pass counts only its own 80 episodes, but the rate a reader wants is over all 100.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        arm = bench.arm('b44a-x', [1000], selected_by={1000: graph_meta(1000, 90.0)})
        arm.samples[1000] = {'scores': [95] * 20, 'perfect': [1] * 20, 'rewards': [1.0] * 20,
                             'seconds': 1.0}
        wave = eval_wave.Wave([arm], 1, 4)
        wave.chart['off'] = True
        wave.on_round(eval_wave.Unit(arm, 1000, 80, 'confirm'), 5, 20, 20, 20, [4] * 5)
        block = arm.payload(complete=False)['in_flight']
        assert block['episodes_so_far'] == 40 and block['perfect_so_far'] == 40
        assert block['episodes_this_pass'] == 20


def test_a_mandatory_perfect_screen_raises_the_plan_rather_than_reporting_past_100():
    # `plan_stages` cannot know the number — it depends on how the screens came out — so the
    # correction happens when the finalists are picked. Without it `measurements_done` exceeds
    # `measurements_planned` and the chart's ETA goes negative.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000, 3000))
        arm = bench.arm('b44a-x', [1000, 2000, 3000], confirm_count=1, selected_by={
            step: graph_meta(step, 90.0) for step in (1000, 2000, 3000)})
        planned = arm.spec.measurements_planned
        wave = run_wave(bench, [arm], lanes=1,
                        pools=[FakePool(rates={1000: 1.0, 2000: 1.0, 3000: 0.1})])
        payload = json.load(open(arm.out_path))
        assert payload['measurements_planned'] > planned
        assert payload['measurements_done'] <= payload['measurements_planned']
        assert payload['stages']['confirm'] == {'planned': 2, 'done': 2}


# ---------------------------------------------------------------- lane allocation

def test_lanes_are_split_across_eligibility_groups_in_proportion_to_their_work():
    with Bench() as bench:
        bench.policy('b43a-x', steps=tuple(range(1000, 10000, 1000)))
        bench.policy('b40a-y', steps=(1000,),
                     arch=policy_arch.build_arch([64, 64], 4, 26, '450e66e'))
        big = bench.arm('b43a-x', list(range(1000, 10000, 1000)), screen_episodes=0)
        small = bench.arm('b40a-y', [1000], screen_episodes=0)
        split = eval_wave.Wave([big, small], 4, 4).lane_split()
        assert [count for _, count in split] == [3, 1], split
        assert sum(count for _, count in split) == 4


def test_every_eligibility_group_gets_at_least_one_lane():
    # Otherwise its work is unrunnable and the wave hangs on units nobody can take.
    with Bench() as bench:
        bench.policy('b43a-x', steps=tuple(range(1000, 40000, 1000)))
        bench.policy('b40a-y', steps=(1000,),
                     arch=policy_arch.build_arch([64, 64], 4, 26, '450e66e'))
        big = bench.arm('b43a-x', list(range(1000, 40000, 1000)), screen_episodes=0)
        small = bench.arm('b40a-y', [1000], screen_episodes=0)
        split = eval_wave.Wave([big, small], 2, 4).lane_split()
        assert [count for _, count in split] == [1, 1], split


def test_one_group_takes_every_lane():
    with Bench() as bench:
        bench.policy('b43a-x', steps=(1000, 2000))
        bench.policy('b43b-y', steps=(1000, 2000))
        arms = [bench.arm('b43a-x', [1000, 2000], screen_episodes=0),
                bench.arm('b43b-y', [1000, 2000], screen_episodes=0)]
        split = eval_wave.Wave(arms, 4, 4).lane_split()
        assert len(split) == 1 and split[0][1] == 4


# ---------------------------------------------------------------- --chain

def test_parse_options_reads_chain_and_leaves_the_selector_alone():
    assert eval_wave.parse_options(['--chain', 'top50', 'b44']) == (True, ['top50', 'b44'])
    assert eval_wave.parse_options(['top50', 'b44']) == (False, ['top50', 'b44'])
    # Idempotent, because the daemon builds its argv from a list and a duplicated flag is a typo
    # rather than a request for two chains.
    assert eval_wave.parse_options(['--chain', '--chain', 'top50']) == (True, ['top50'])


def test_hof_settings_fix_the_recipe_and_carry_the_launch_over():
    base = {'suffix': '', 'source_suffix': '', 'num_episodes': 100, 'num_workers': 6,
            'screen_episodes': 20, 'screen_requested': None, 'confirm_count': 100,
            'min_achievable': 97.0, 'abandon_floor': 20, 'resume': '1'}
    hof = eval_plan.hof_settings(base)
    # The recipe.
    assert hof['num_episodes'] == 500 and hof['screen_episodes'] == 0
    assert hof['min_achievable'] == eval_plan.HOF_GATE == 98.0
    assert hof['suffix'] == '_hof500' and hof['source_suffix'] == ''
    # Not the recipe: how the wave was launched carries over.
    assert hof['num_workers'] == 6 and hof['resume'] == '1'
    # Stage A is untouched — a shared dict would have stage B's gate applied to stage A's arms.
    assert base['num_episodes'] == 100 and base['min_achievable'] == 97.0


def test_hof_settings_keep_a_throwaway_suffix_distinct():
    # A verification wave launched with EVAL_OUT_SUFFIX must not have its HOF stage land on the real
    # `_hof500` file, so the suffixes compose rather than replace.
    hof = eval_plan.hof_settings({'suffix': '_check', 'num_episodes': 100, 'screen_episodes': 20,
                                  'screen_requested': None, 'num_workers': 4, 'confirm_count': 1,
                                  'min_achievable': 97.0, 'abandon_floor': 20, 'resume': None})
    assert hof['suffix'] == '_check_hof500' and hof['source_suffix'] == '_check'


def test_the_gate_ordering_assert_fires_at_import_time():
    # The single definition of the invariant, and it is an `assert` at module scope rather than a
    # test-only check: a close-out gate at or above the HOF gate abandons exactly the rows the
    # re-measure needs, and the failure is silent — a HOF pass that measures nothing.
    assert eval_plan.DEFAULT_MIN_ACHIEVABLE < eval_plan.HOF_GATE
    source = open(eval_plan.__file__).read()
    assert 'assert DEFAULT_MIN_ACHIEVABLE < HOF_GATE' in source


def test_completed_policies_reads_the_flag_and_skips_what_it_cannot_trust():
    with Bench() as bench:
        for name, payload in (('a-done', {'complete': True}),
                              ('b-partial', {'complete': False}),
                              ('c-nokey', {'results': []})):
            path = os.path.join(bench.runs, '{0}_checkpoint_evals.json'.format(name))
            json.dump(payload, open(path, 'w'))
        open(os.path.join(bench.runs, 'd-broken_checkpoint_evals.json'), 'w').write('{ not json')
        got = eval_wave.completed_policies(
            ['a-done', 'b-partial', 'c-nokey', 'd-broken', 'e-missing'], '')
        assert got == ['a-done'], got


def test_stage_b_selects_out_of_stage_as_file_not_its_own():
    # The suffix pair is the whole mechanism: `above:98` has to read the close-out's 100-episode
    # rows, while writing somewhere that cannot clobber them.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000, 2000))
        rows = [{'step': 1000, 'episodes': 100, 'perfect_games': 99, 'perfect_percent': 99.0},
                {'step': 2000, 'episodes': 100, 'perfect_games': 80, 'perfect_percent': 80.0}]
        json.dump({'policy_name': 'b44a-x', 'complete': True, 'results': rows},
                  open(os.path.join(bench.runs, 'b44a-x_checkpoint_evals.json'), 'w'))
        hof = eval_plan.hof_settings(
            {'suffix': '', 'num_episodes': 100, 'screen_episodes': 20, 'screen_requested': None,
             'num_workers': 4, 'confirm_count': 1, 'min_achievable': 97.0, 'abandon_floor': 20,
             'resume': None})
        arms = eval_wave.build_arms(['b44a-x'], 'above', eval_plan.HOF_GATE, hof)
        assert len(arms) == 1
        assert arms[0].requested_steps == [1000]        # 2000 read 80%, below the gate
        assert arms[0].out_path.endswith('b44a-x_checkpoint_evals_hof500.json')
        assert arms[0].num_episodes == 500 and arms[0].screen_episodes == 0


def test_stage_b_produces_no_arm_when_nothing_reached_the_gate():
    # The normal outcome for most arms, and it must be a clean exit rather than a failure — the
    # desktop marks the job done off the exit code, and a retry would measure nothing forever.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        json.dump({'policy_name': 'b44a-x', 'complete': True,
                   'results': [{'step': 1000, 'episodes': 100, 'perfect_games': 90,
                                'perfect_percent': 90.0}]},
                  open(os.path.join(bench.runs, 'b44a-x_checkpoint_evals.json'), 'w'))
        hof = eval_plan.hof_settings(
            {'suffix': '', 'num_episodes': 100, 'screen_episodes': 20, 'screen_requested': None,
             'num_workers': 4, 'confirm_count': 1, 'min_achievable': 97.0, 'abandon_floor': 20,
             'resume': None})
        code, arms = eval_wave.run_stage('stage B', 'above', eval_plan.HOF_GATE, ['b44a-x'],
                                         hof, 1)
        assert (code, arms) == (0, [])


def test_stage_b_reads_a_non_default_close_out_suffix():
    # The same property as the fixture above, but with a suffix that is not the empty string —
    # which is what `select_checkpoints_above` defaults to, so a version that dropped
    # `source_suffix` entirely would still pass there and fail here.
    with Bench() as bench:
        bench.policy('b44a-x', steps=(1000,))
        json.dump({'policy_name': 'b44a-x', 'complete': True,
                   'results': [{'step': 1000, 'episodes': 100, 'perfect_games': 99,
                                'perfect_percent': 99.0}]},
                  open(os.path.join(bench.runs, 'b44a-x_checkpoint_evals_check.json'), 'w'))
        hof = eval_plan.hof_settings(
            {'suffix': '_check', 'num_episodes': 100, 'screen_episodes': 20,
             'screen_requested': None, 'num_workers': 4, 'confirm_count': 1,
             'min_achievable': 97.0, 'abandon_floor': 20, 'resume': None})
        arms = eval_wave.build_arms(['b44a-x'], 'above', eval_plan.HOF_GATE, hof)
        assert [arm.requested_steps for arm in arms] == [[1000]]
        assert arms[0].out_path.endswith('b44a-x_checkpoint_evals_check_hof500.json')
