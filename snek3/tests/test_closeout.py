"""The close-out: one process per batch, both boxes.

`tools/closeout.py` is what the desktop daemon dispatches and what an agent types on the laptop, so
these fixtures are about the parts that only show up when something goes wrong — a failing arm, a
raising arm — which is exactly where the two hand-written sequencers it replaced differed from each
other. The window is the scheduler's now (`tools/window.py`), so nothing here is about one.
"""

import os

import pytest

from tools import closeout


class Waves(object):
    """Stands in for `eval_wave.ArmWave`: records the arms in the order they were resolved and
    started, returns canned finish statuses, and can raise at construction (a bad arm).

    `candidates` is each arm's step count, 1 by default; `ticks` is how many polls a launched shard
    stays alive, 1 by default, so pooling is observable through `peak` -- the most shards alive at
    once -- without any real process."""

    def __init__(self, codes=None, candidates=None, ticks=1):
        self.codes, self.candidates, self.ticks = codes or {}, candidates or {}, ticks
        self.calls, self.started, self.finished, self.settings = [], [], [], {}
        self.alive_now, self.peak = 0, 0

    def __call__(self, policy, selector='screen', episodes=500, shards=4, label=None, width=None,
                 seed=0, resume=True, merge=True):
        self.calls.append(policy)
        self.settings[policy] = {'selector': selector, 'episodes': episodes, 'shards': shards,
                                 'label': label, 'seed': seed}
        code = self.codes.get(policy, 0)
        if isinstance(code, Exception):
            raise code
        return _FakeArm(self, policy, self.candidates.get(policy, 1), shards, code)


class _FakeArm(object):
    def __init__(self, waves, policy, candidates, pool, code):
        self.waves, self.policy, self.code = waves, policy, code
        self.steps = list(range(candidates))
        self.shards = min(pool, candidates)
        self.pending = list(range(self.shards))
        self.running = []          # remaining ticks per launched shard

    def announce(self, requested=None):
        pass

    def start_shard(self):
        if not self.pending:
            return False
        self.pending.pop(0)
        self.running.append(self.waves.ticks)
        if self.policy not in self.waves.started:
            self.waves.started.append(self.policy)
        self.waves.alive_now = sum(1 for arm in _live_arms(self.waves) for t in arm.running if t > 0)
        self.waves.peak = max(self.waves.peak, self.waves.alive_now)
        return True

    def exhausted(self):
        return not self.pending

    def alive(self):
        return sum(1 for t in self.running if t > 0)

    def finished(self):
        return self.exhausted() and self.alive() == 0

    def tick(self):
        self.running = [t - 1 for t in self.running]

    def counts(self):
        return [1 for t in self.running if t <= 0]

    def terminate(self):
        self.running = [0 for _ in self.running]

    def finish(self):
        self.waves.finished.append(self.policy)
        return self.code


_ARMS = []


def _live_arms(waves):
    return [arm for arm in _ARMS if arm.waves is waves]


@pytest.fixture
def waves(monkeypatch):
    def install(codes=None, candidates=None, ticks=1):
        stub = Waves(codes, candidates, ticks)
        real_call = stub.__call__

        def make(*args, **kwargs):
            arm = real_call(*args, **kwargs)
            _ARMS.append(arm)
            return arm
        monkeypatch.setattr(closeout.eval_wave, 'ArmWave', make)
        monkeypatch.setattr(closeout.stage_b_chart, 'redraw', lambda *a, **k: (None, []))
        # a poll is a tick: every launched shard ages by one, no real sleeping
        monkeypatch.setattr(closeout.time, 'sleep', lambda s: [arm.tick() for arm in _live_arms(stub)])
        monkeypatch.setattr(closeout.eval_wave, 'POLL_S', 0)
        del _ARMS[:]
        return stub
    return install


def test_every_arm_is_measured_in_the_order_the_batch_named_them(waves):
    stub = waves()
    assert closeout.run(['b1a', 'b1b', 'b1c'], shards=8) == 0
    assert stub.calls == ['b1a', 'b1b', 'b1c'], 'resolved in order'
    assert stub.started == ['b1a', 'b1b', 'b1c'], 'started in order'
    assert sorted(stub.finished) == ['b1a', 'b1b', 'b1c']


def test_one_arm_failing_does_not_drop_the_arms_behind_it(waves):
    """The `;`-not-`&&` rule of the shell chain this replaced. The incident is snek2's."""
    stub = waves({'b1a': 1})
    assert closeout.run(['b1a', 'b1b', 'b1c']) == 1
    assert stub.calls == ['b1a', 'b1b', 'b1c'], 'a bad first arm must not end the batch'


def test_an_arm_that_raises_does_not_drop_the_arms_behind_it_either(waves):
    """A restore failure is one bad arm, not a bad batch — and it used to kill the whole pass."""
    stub = waves({'b1b': RuntimeError('no arch.json')})
    assert closeout.run(['b1a', 'b1b', 'b1c']) == 1
    assert stub.calls == ['b1a', 'b1b', 'b1c']
    assert stub.started == ['b1a', 'b1c']


def test_the_status_is_a_failure_even_when_the_last_arm_succeeds(waves):
    waves({'b1a': 2})
    assert closeout.run(['b1a', 'b1b']) != 0


def test_the_protocol_defaults_are_the_close_outs_own(waves):
    """The daemon passes a selector and an episode count only if a spec named them."""
    stub = waves()
    closeout.run(['b1a'])
    assert stub.settings['b1a'] == {'selector': 'screen', 'episodes': 500, 'shards': 4,
                                    'label': None, 'seed': 0}


# --- the pool --------------------------------------------------------------------------------------

def test_arms_with_few_candidates_run_side_by_side_under_one_shard_budget(waves):
    """The hof30k case: eight arms with one candidate each used to be eight sequential single-shard
    waves. Pooled, all eight run at once and the wave takes one checkpoint's time."""
    arms = ['b14{0}'.format(letter) for letter in 'abcdefgh']
    stub = waves(ticks=3)
    assert closeout.run(arms, shards=12) == 0
    assert stub.peak == 8, 'every arm\'s single shard alive at once'
    assert stub.started == arms


def test_the_pool_is_never_exceeded(waves):
    arms = ['b14{0}'.format(letter) for letter in 'abcdefgh']
    stub = waves(candidates={arm: 5 for arm in arms}, ticks=2)
    closeout.run(arms, shards=12)
    assert stub.peak <= 12
    assert stub.peak == 12, 'and it is kept full'


def test_an_arm_gets_no_more_shards_than_it_has_candidates(waves):
    stub = waves(candidates={'b1a': 3, 'b1b': 40})
    closeout.run(['b1a', 'b1b'], shards=12)
    arms = {arm.policy: arm for arm in _ARMS}
    assert (len(arms['b1a'].running), len(arms['b1b'].running)) == (3, 12)


def test_an_arm_with_no_candidates_is_closed_at_once_and_is_not_a_failure(waves):
    """Its empty pass file is what the next pass selects from."""
    stub = waves(candidates={'b1a': 0})
    assert closeout.run(['b1a', 'b1b']) == 0
    assert stub.finished[0] == 'b1a', 'closed before any shard is launched'
    assert stub.started == ['b1b']


def test_the_next_arm_starts_as_soon_as_a_slot_frees_not_when_the_arm_ends(waves):
    """Pool 4, first arm 6 candidates: its last two shards launch as the first four finish, and the
    second arm's shards fill whatever is free alongside them."""
    stub = waves(candidates={'b1a': 6, 'b1b': 6}, ticks=2)
    closeout.run(['b1a', 'b1b'], shards=4)
    assert stub.peak == 4
    assert stub.started == ['b1a', 'b1b']


def test_a_single_checkpoint_runs_no_wave(waves, monkeypatch):
    stub = waves()
    monkeypatch.setattr(closeout, 'measure_one', lambda *a, **k: None)
    assert closeout.run(['b1a'], selector='one', episodes=3000) == 0
    assert stub.calls == [], 'the `one` selector is measured in this process, not by a wave'


def test_the_close_out_opens_no_window_of_its_own():
    """Since 2026-09-05 the scheduler owns the box's window and points it at the pass's charts; a
    close-out that opened one beside it was the two-windows-and-a-stand-by-loop design."""
    import inspect
    source = inspect.getsource(closeout)
    assert 'eval_window' not in source and 'chart_window' not in source


# --- the redraw ------------------------------------------------------------------------------------

def test_the_drawer_redraws_once_more_after_the_wave_finishes():
    """The last word is the merged file's, not the pooled shard files' a mid-pass redraw saw."""
    drawn = []
    drawer = closeout.Drawer('b1a', interval=60.0)
    drawer.draw_once = lambda: drawn.append('b1a')
    with drawer:
        pass
    assert drawn == ['b1a'], 'no periodic redraw fired in the interval, and exactly one at the end'


def test_the_drawer_thread_is_a_daemon_so_it_cannot_outlive_the_pass():
    """A `stage_b_chart --watch` subprocess could be orphaned; a daemon thread cannot."""
    drawer = closeout.Drawer('b1a', interval=60.0)
    drawer.draw_once = lambda: None
    with drawer:
        assert drawer.thread.daemon and drawer.thread.is_alive()
    assert not drawer.thread.is_alive()


def test_a_redraw_failure_is_swallowed(monkeypatch, capsys):
    """A chart is a readout. A pass that died because a PNG could not be written would be backwards.

    The redraw is made to raise rather than merely fail: an arm with no rows yet returns `(None, [])`
    quite legitimately, so a fixture built on a missing policy would pass with the `except` deleted.
    """
    def explode(*args, **kwargs):
        raise IOError('disk full')
    monkeypatch.setattr(closeout.stage_b_chart, 'redraw', explode)
    assert closeout.Drawer('b1a').draw_once() is None
    assert 'chart redraw failed for b1a' in capsys.readouterr().err


def test_a_redraw_failure_does_not_fail_the_arm(monkeypatch, waves):
    def explode(*args, **kwargs):
        raise IOError('disk full')
    stub = waves()
    monkeypatch.setattr(closeout.stage_b_chart, 'redraw', explode)
    assert closeout.run(['b1a', 'b1b']) == 0
    assert stub.calls == ['b1a', 'b1b']


# --- the passes, by name ---------------------------------------------------------------------------

def test_the_chain_is_three_passes_each_selecting_from_the_one_before():
    assert closeout.CHAIN == ('stageb', 'hof5000', 'hof30k')
    assert closeout.FOLLOW_ON == {'stageb': 'hof5000', 'hof5000': 'hof30k'}
    assert closeout.PASSES['hof5000']['selector'] == 'above:99'
    assert closeout.PASSES['hof30k']['selector'] == 'above:99:hof5000', 'reads the hof5000 file'


def test_stage_b_is_the_default_pass_and_the_close_outs_own_defaults():
    """A command that names no pass is unchanged: `screen:97` at 500, unlabelled, seed 0."""
    assert closeout.pass_settings('stageb') == {'selector': 'screen', 'episodes': 500,
                                                'label': None, 'seed': 0}
    args = closeout.build_parser().parse_args(['b1a'])
    assert args.pass_name == 'stageb'


def test_a_hof_pass_is_labelled_so_it_never_overwrites_what_it_selected_from():
    """The output path is `runs/<arm>_checkpoint_evals[_<label>].json`; unlabelled, the 5,000-episode
    rows would replace the 500-episode file `above:99` reads. The hof-remeasure skill calls omitting
    the label 'destroying the input'; the preset makes it impossible to omit."""
    assert closeout.pass_settings('hof5000') == {'selector': 'above:99', 'episodes': 5000,
                                                 'label': 'hof5000', 'seed': 0}
    assert closeout.pass_settings('hof30k') == {'selector': 'above:99:hof5000', 'episodes': 30000,
                                                'label': 'hof30k', 'seed': 7}


def test_an_explicit_flag_wins_over_the_preset_but_none_never_unsets_it():
    chosen = closeout.pass_settings('hof5000', episodes=2000, seed=3)
    assert (chosen['episodes'], chosen['seed'], chosen['label']) == (2000, 3, 'hof5000')
    assert closeout.pass_settings('hof5000', label=None)['label'] == 'hof5000'
    with pytest.raises(ValueError):
        closeout.pass_settings('hof9000')


def test_main_hands_the_pass_to_run(monkeypatch):
    seen = {}

    def run(policies, selector, episodes, shards, label, width, seed, resume, merge):
        seen.update(policies=policies, selector=selector, episodes=episodes, shards=shards,
                    label=label, seed=seed)
        return 0
    monkeypatch.setattr(closeout, 'run', run)
    assert closeout.main(['b1a', 'b1b', '--pass', 'hof30k', '--shards', '12']) == 0
    assert seen == {'policies': ['b1a', 'b1b'], 'selector': 'above:99:hof5000', 'episodes': 30000,
                    'shards': 12, 'label': 'hof30k', 'seed': 7}
    closeout.main(['b1a'])
    assert (seen['selector'], seen['episodes'], seen['label'], seen['seed']) == ('screen', 500, None, 0)
