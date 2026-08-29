"""The training loop's own pieces: config, the intervals, the gradient budget, the sidecar.

Three of these are silent when broken and would cost a whole arm:

- **The eval and checkpoint intervals must be equal.** The self-eval is stage A of the measurement,
  so a checkpoint written at a step no eval screens can never be measured at all.
- **The replay ratio must hold across iterations.** A ratio below 1.0 that rounded to zero every
  iteration would train nothing while reporting a full run.
- **A resume must not adopt a new architecture.** `arch.json` is what stops old weights being loaded
  into a differently shaped network.
"""

import json
import os
import re

import time

import numpy as np
import pytest

import train
from vectorized import config as reward_config
from dqn import collect
from tools import arch as arch_tools
from tools import checkpoints
from tools import eval_queue
from tools import run_report


class StubBuffer(object):
    """Counts samples and hands back a fixed batch. No tree, no torch."""

    def __init__(self, size=1000):
        self.size = size
        self.samples = 0
        self.updates = 0

    def sample(self, batch_size, step):
        self.samples += 1
        return {'obs': None}, np.arange(2), np.ones(2)

    def update_priorities(self, indexes, td_errors):
        self.updates += 1

    def add(self, *args):
        self.size += 1


class StubAgent(object):
    """Mirrors the real agent's contract, including that **`update` refreshes the target itself.**

    That detail is not incidental to the stub: while `update` was a no-op on this point, a `_learn`
    that also called `maybe_update_target` looked correct here and doubled the Polyak step in
    production. See `test_a_gradient_step_refreshes_the_target_once_and_not_twice`.
    """

    def __init__(self):
        self.train_step = 0
        self.updates = 0
        self.target_updates = 0

    def update(self, batch, weights=None):
        self.updates += 1
        self.train_step += 1
        self.maybe_update_target()
        return np.zeros(2), {}

    def maybe_update_target(self):
        self.target_updates += 1


# --- the intervals ----------------------------------------------------------------------------

def test_the_eval_interval_equals_the_checkpoint_interval():
    # One knob sets both, so this cannot drift — but it is the invariant, so it is asserted.
    assert train.EVAL_INTERVAL == train.CHECKPOINT_INTERVAL


def test_the_resume_and_report_intervals_are_whole_multiples_of_the_eval_interval():
    # Otherwise a resume or a report is written at a step no eval produced a row for.
    assert train.RESUME_INTERVAL % train.EVAL_INTERVAL == 0
    assert train.REPORT_INTERVAL % train.EVAL_INTERVAL == 0


# --- tuned ------------------------------------------------------------------------------------

def test_tuned_returns_the_default_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv('SNEK_MADE_UP_KNOB', raising=False)
    assert train.tuned('MADE_UP_KNOB', 7, int) == 7


def test_tuned_reads_and_casts_an_override(monkeypatch, capsys):
    monkeypatch.setenv('SNEK_MADE_UP_KNOB', '11')
    assert train.tuned('MADE_UP_KNOB', 7, int) == 11
    # Printed, because this project greps `hyperparameter override:` on a log to confirm an arm
    # received the config it was launched with.
    assert 'hyperparameter override: MADE_UP_KNOB = 11' in capsys.readouterr().out


# --- build_config -----------------------------------------------------------------------------

def test_the_default_architecture_is_the_one_every_record_holder_used():
    # A single hidden layer of 320. Phase 3 is a reproduction; the architecture is not the variable.
    assert train.build_config()['fc_layers'] == (320,)


def test_every_config_key_is_its_knob_lowercased():
    """So a row in `runs/<arm>.md` can be grepped straight back to the variable that set it.

    Checked by reading the `tuned(...)` calls out of this module's own source, which is the only way
    to catch a key that was renamed without its knob or the reverse.
    """
    import re
    source = open(train.__file__).read()
    knobs = {name.lower() for name in re.findall(r"tuned\('([A-Z_]+)'", source)}
    keys = set(train.build_config())
    # `algo` and `min_checkpoint_score` are not `tuned()` calls — one is fixed, the other is read in
    # `env/constants.py` so the eval workers inherit it.
    derived = {'algo', 'min_checkpoint_score', 'fork'}
    assert keys - derived <= knobs, sorted(keys - derived - knobs)


def test_the_documented_defaults_are_the_ones_the_code_uses():
    """`docs/running.md` is the contract, so a drift between it and this file is a real bug.

    Checked against a hand-copied table rather than by parsing the doc: a parser would follow the
    doc wherever it went, and the point is to notice when the two disagree.
    """
    documented = {'max_steps': 10000000, 'learning_rate': 1e-5, 'adam_epsilon': 1e-7,
                  'batch_size': 128, 'discount': 0.99, 'target_update_period': 8,
                  'gradient_clipping': 0.0, 'n_step_update': 1, 'initial_epsilon': 0.4,
                  'min_epsilon': 0.002, 'guided_fraction': 0.8,
                  'replay_buffer_max_length': 100000, 'priority_exponent': 0.6, 'is_beta': 0.4,
                  'is_beta_final': 1.0, 'beta_anneal_steps': 300000, 'is_weights': True,
                  'replay_ratio': 1.0, 'collect_envs': 1, 'fc_layers': (320,),
                  'eval_interval': 1000, 'graph_eval_episodes': 100, 'torch_threads': 1}
    config = train.build_config()
    assert {key: config[key] for key in documented} == documented


def test_a_min_epsilon_below_the_hard_floor_is_rejected(monkeypatch):
    monkeypatch.setenv('SNEK_MIN_EPSILON', '1e-9')
    with pytest.raises(ValueError, match='hard floor'):
        train.build_config()


def test_a_non_positive_replay_ratio_is_rejected(monkeypatch):
    # Zero would train nothing while reporting a complete run.
    monkeypatch.setenv('SNEK_REPLAY_RATIO', '0')
    with pytest.raises(ValueError, match='must be positive'):
        train.build_config()


def test_forking_is_on_by_default_as_the_docs_say():
    # Four branches, matching snek2's default, so a bare launch is a forking arm.
    fork = train.build_config()['fork']
    assert (fork.branches, fork.prob, fork.min_length, fork.max_steps) == (4, 0.5, 85, 60)
    assert fork.enabled


def test_the_fork_knobs_reach_the_fork_config(monkeypatch):
    monkeypatch.setenv('SNEK_FORK_BRANCHES', '4')
    monkeypatch.setenv('SNEK_FORK_MIN_LENGTH', '70')
    fork = train.build_config()['fork']
    assert fork.branches == 4 and fork.min_length == 70 and fork.enabled


def test_the_report_config_flattens_the_fork_object():
    # The report writes `| key | value |` rows, so an object would print as its repr.
    config = train.build_config()
    config['fork'] = collect.ForkConfig(branches=4, prob=0.25, min_length=90, max_steps=30)
    out = train.reportable(config)
    assert 'fork' not in out
    assert out['fork_branches'] == 4 and out['fork_prob'] == 0.25
    for value in out.values():
        assert isinstance(value, (int, float, str, tuple, bool, type(None))), value


def test_a_control_arm_does_not_report_fork_knobs_it_never_used():
    config = train.build_config()
    config['fork'] = collect.ForkConfig(branches=1)
    out = train.reportable(config)
    assert out['fork_branches'] == 1
    assert 'fork_prob' not in out, 'a stale probability in the report reads as a forking arm'


# --- the eval seed ----------------------------------------------------------------------------

def test_the_eval_seed_is_derived_and_repeatable():
    # Repeatable so a resumed run re-measuring the step it stopped on gets the same boards.
    assert train.eval_seed(1, 1000) == train.eval_seed(1, 1000)


def test_the_eval_seed_moves_with_the_step_and_with_the_arm():
    # A fixed eval seed measures the same 100 boards forever, and a policy that improved only on
    # those boards would be indistinguishable from one that improved.
    steps = {train.eval_seed(1, step) for step in range(1000, 21000, 1000)}
    assert len(steps) == 20
    assert train.eval_seed(1, 1000) != train.eval_seed(2, 1000)


# --- the eval row -----------------------------------------------------------------------------

def measured(avg_score=42.0, perfect=0.3):
    return {'avg_score': avg_score, 'min_score': 1.0, 'max_score': 95.0,
            'avg_reward': 12.5, 'perfect': perfect}


def test_the_row_carries_every_column_the_report_prints():
    row = train.build_eval_row(1000, measured(), 40.0, 0.01, 0.0, 900.0)
    for key, _ in run_report.EVAL_COLUMNS:
        assert key in row, key
        assert row[key] is not None, key


def test_the_row_summarises_this_evals_own_episodes():
    # Not a mix of this eval's mean with a training interval's extremes, which is what snek2 printed.
    row = train.build_eval_row(1000, measured(avg_score=42.0), 40.0, 0.01, 0.0, 900.0)
    assert row['avg_score'] == 42.0
    assert row['min_score'] == 1.0 and row['max_score'] == 95.0


def test_the_perfect_column_is_a_percentage_not_a_fraction():
    # Stored on 0-100 because every reader — the chart, the summary, the schedules — assumes it.
    row = train.build_eval_row(1000, measured(perfect=0.37), 40.0, 0.01, 0.0, 900.0)
    assert row['perfect_percent'] == 37.0


def test_the_row_feeds_the_summary_builder_without_a_key_error():
    rows = [train.build_eval_row(step, measured(), 40.0, 0.01, 0.0, 900.0)
            for step in (1000, 2000, 3000)]
    summary = run_report.build_summary(rows)
    assert summary['step'] == 3000 and summary['evals'] == 3


def test_a_control_arm_writes_no_fork_field():
    row = train.build_eval_row(1000, measured(), 40.0, 0.01, 0.0, 900.0)
    assert 'fork' not in row


# --- the gradient budget ----------------------------------------------------------------------

def learner(ratio, batch_size=8):
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.config = {'replay_ratio': ratio, 'batch_size': batch_size}
    trainer.buffer = StubBuffer()
    trainer.agent = StubAgent()
    trainer.gradient_debt = 0.0
    return trainer


def test_a_ratio_of_one_is_one_gradient_step_per_transition():
    trainer = learner(1.0)
    for _ in range(10):
        trainer._learn(3)
    assert trainer.agent.updates == 30


def test_a_ratio_below_one_accumulates_across_iterations_instead_of_rounding_away():
    """The property that makes a fractional ratio mean what it says.

    Rounded per iteration, `0.25` with one transition per iteration would floor to zero every time
    and the arm would collect for hours without a single gradient step.
    """
    trainer = learner(0.25)
    for _ in range(100):
        trainer._learn(1)
    assert trainer.agent.updates == 25


def test_a_ratio_above_one_does_several_gradient_steps_per_transition():
    trainer = learner(3.0)
    trainer._learn(2)
    assert trainer.agent.updates == 6


def test_every_gradient_step_feeds_its_priorities_back():
    # A sampled batch whose priorities are never updated keeps its old weight forever, which turns
    # prioritised replay into a slow uniform buffer.
    trainer = learner(1.0)
    trainer._learn(5)
    assert trainer.buffer.samples == 5 == trainer.buffer.updates


def test_a_gradient_step_refreshes_the_target_once_and_not_twice():
    """`agent.update` owns the target refresh; `_learn` must not call it again.

    `_learn` did until 2026-08-29, and at the default `tau` of 1.0 nothing could see it — a hard copy
    of weights just copied is idempotent. At `tau < 1.0` it applied the Polyak step twice at one
    `train_step`, so a requested 0.05 ran at 1 - (1 - 0.05)^2 = 0.0975. The update *period* was never
    affected, because `maybe_update_target` gates on `train_step` rather than counting its own calls,
    which is why this is a call-count assertion and not a period one.
    """
    trainer = learner(1.0)
    trainer._learn(5)
    assert trainer.agent.updates == 5
    assert trainer.agent.target_updates == 5, 'the target is being refreshed twice per gradient step'


def test_an_empty_buffer_does_not_bank_debt_for_later():
    """A buffer that fills on step 3 must not owe three batches at once.

    Otherwise the first gradient steps of every run come in a burst against the smallest, least
    diverse buffer the run will ever have.
    """
    trainer = learner(1.0)
    trainer.buffer.sample = lambda batch_size, step: None
    trainer._learn(10)
    assert trainer.agent.updates == 0
    assert trainer.gradient_debt < 1.0


# --- the sidecar ------------------------------------------------------------------------------

def make_trainer(tmp_path, policy='t1', monkeypatch=None, **overrides):
    """A Trainer rooted in `tmp_path`, with the intervals shrunk so a test can reach an eval.

    **`eval_queue` is pinned off here rather than inherited from `build_config`**, so that a test about
    the in-loop path keeps testing the in-loop path. It was inherited until 2026-08-29, when the
    default flipped to on and five tests began asserting against `eval_rows` that the queue had not
    filled yet — failing for a reason unrelated to what any of them was about. A test that cares about
    the queue passes `eval_queue=True` explicitly, which is also how it reads.
    """
    config = train.build_config()
    config.update({'replay_buffer_max_length': 2000, 'initial_collect_steps': 50, 'max_steps': 40,
                   'batch_size': 8, 'seed': 3, 'eval_queue': False})
    config.update(overrides)
    monkeypatch.setattr(train.constants, 'POLICY_DIR', str(tmp_path / 'savedPolicies'))
    monkeypatch.setattr(train.constants, 'RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setattr(train, 'EVAL_INTERVAL', 10)
    monkeypatch.setattr(train, 'CHECKPOINT_INTERVAL', 10)
    monkeypatch.setattr(train, 'EVAL_EPISODES', 4)
    monkeypatch.setattr(train, 'RESUME_INTERVAL', 20)
    monkeypatch.setattr(train, 'REPORT_INTERVAL', 20)
    monkeypatch.setattr(train.progress_chart, 'chart_path',
                        lambda name: str(tmp_path / 'runs' / (name + '.png')))
    return train.Trainer(policy, config)


def test_a_fresh_arm_writes_its_arch_sidecar(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch)
    written = json.load(open(arch_tools.arch_path(trainer.policy_dir)))
    assert written['fc_layer_params'] == list(trainer.config['fc_layers'])


def test_a_resume_keeps_the_sidecar_it_already_had(tmp_path, monkeypatch):
    make_trainer(tmp_path, monkeypatch=monkeypatch)
    again = make_trainer(tmp_path, monkeypatch=monkeypatch)
    assert again.arch['fc_layer_params'] == [320]


def test_a_resume_that_asks_for_a_different_network_is_refused(tmp_path, monkeypatch):
    """Refused here rather than by torch, which would only notice at the first `load_state_dict`.

    A run that started, trained and then died on a shape error has already overwritten the resume
    state it would need to recover from.
    """
    make_trainer(tmp_path, monkeypatch=monkeypatch)
    with pytest.raises(arch_tools.ArchMismatch):
        make_trainer(tmp_path, monkeypatch=monkeypatch, fc_layers=(64, 64))


# --- the checkpoint gate ----------------------------------------------------------------------

def test_a_worthless_checkpoint_is_skipped_and_counted(tmp_path, monkeypatch):
    """The gate that stopped snek2 evicting its best checkpoint behind 4.5M dead steps.

    Counted rather than silent, because "no checkpoints in this directory" and "this arm never
    cleared the bar" look identical from outside otherwise.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=40.0)
    trainer.step = 10
    assert trainer._settle_checkpoint(trainer.step, 1.0, 1.0) is False
    assert trainer.skipped_checkpoints == 1
    assert not os.path.exists(os.path.join(trainer.policy_dir, 'ckpt-10.pt'))


def test_either_this_eval_or_the_trailing_mean_clearing_the_bar_is_enough(tmp_path, monkeypatch):
    """The best checkpoints in this project are outliers that spike above their neighbourhood.

    A trailing-only gate would skip exactly the one worth keeping while an arm is recovering.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=40.0)
    # The step is an argument now rather than read off the trainer, which is what lets a queued row
    # settle a checkpoint several intervals after the loop moved past it.
    assert trainer._settle_checkpoint(20, 80.0, 1.0) is True, 'a spike must be kept'
    assert trainer._settle_checkpoint(30, 1.0, 80.0) is True, 'a good neighbourhood must be kept'
    for step in (20, 30):
        assert os.path.exists(os.path.join(trainer.policy_dir, 'ckpt-{0}.pt'.format(step)))


# --- end to end -------------------------------------------------------------------------------

def test_a_short_run_produces_checkpoints_a_history_and_a_report(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    trainer.run()
    assert trainer.step == 40
    steps = [row['step'] for row in trainer.eval_rows]
    assert steps == [10, 20, 30, 40]
    assert os.path.exists(trainer.report_path)
    assert os.path.exists(trainer.history_path)
    assert os.path.exists(os.path.join(trainer.policy_dir, train.RESUME_FILENAME))
    rows, resumes = run_report.load_history(trainer.history_path)
    assert len(rows) == 4 and resumes == []


def test_a_resumed_run_continues_the_step_count_and_the_same_curve(tmp_path, monkeypatch):
    """The graph has to continue rather than restart, and the resume has to be recorded.

    A resume that started a fresh series at step 0 would draw two arms on one chart and make every
    step number in the report ambiguous.
    """
    first = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    first.run()
    second = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0,
                          max_steps=60)
    assert second.step == 40, 'the resume did not pick up the step count'
    assert len(second.eval_rows) == 4, 'the resume did not pick up the curve'
    assert second.resume_steps == [40]
    second.run()
    assert [row['step'] for row in second.eval_rows] == [10, 20, 30, 40, 50, 60]


def test_a_run_already_past_its_cap_does_nothing_rather_than_looping(tmp_path, monkeypatch, capsys):
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    trainer.run()
    again = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    again.run()
    assert again.step == 40
    assert 'nothing to train' in capsys.readouterr().out


# --- the eval, driven with scripted measurements ----------------------------------------------

def scripted(monkeypatch, *measurements):
    """Replaces the self-eval with a fixed script, so the loop's reactions can be asserted.

    Necessary rather than convenient: a randomly initialised net scores ~0 on every eval, so a real
    self-eval can never move the epsilon schedule off its ceiling and every fixture that depends on
    the schedule reacting would pass vacuously.
    """
    script = list(measurements)
    calls = []

    def fake(agent, step, seed, episodes=None):
        calls.append(step)
        return script[min(len(calls) - 1, len(script) - 1)]

    monkeypatch.setattr(train, 'self_eval', fake)
    return calls


def test_the_epsilon_schedule_is_applied_at_every_eval(tmp_path, monkeypatch):
    """Or the arm explores at its initial epsilon for its whole life.

    Silent in the log — the printed epsilon is the same variable — and silent in the chart, which
    would just show an arm that never refines.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    assert trainer.epsilon == trainer.config['initial_epsilon']
    scripted(monkeypatch, measured(avg_score=90.0, perfect=0.0))
    trainer.step = 10
    trainer._evaluate(900.0)
    # An avg_reward of 12.5 clears the first two bootstrap rungs and no more, so the ceiling halves
    # twice: this pins that the schedule ran, not merely that epsilon changed.
    assert trainer.epsilon == pytest.approx(train.schedules.epsilon_for(
        12.5, 0.0, trainer.config['initial_epsilon'], trainer.config['min_epsilon']))
    assert trainer.epsilon < trainer.config['initial_epsilon']


def test_the_shield_fraction_reaches_the_collector(tmp_path, monkeypatch):
    """A configured shield that never arrives leaves an unshielded arm reported as shielded.

    The reward here stands the bootstrap phase down, which is the condition the shield switches on
    under — one rule, "shielded iff refining".
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0,
                           guided_fraction=0.8)
    assert trainer.collector.guided_fraction == 0.0, 'the shield is off during bootstrap'
    scripted(monkeypatch, dict(measured(), avg_reward=25.0))
    trainer.step = 10
    trainer._evaluate(900.0)
    assert trainer.collector.guided_fraction == pytest.approx(0.8)


def test_the_trailing_score_is_a_window_and_not_this_eval_alone(tmp_path, monkeypatch):
    # The checkpoint gate and the chart both read it, and an arm's trailing curve is the main thing
    # a progress report is read for.
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    scripted(monkeypatch, measured(avg_score=10.0), measured(avg_score=20.0),
             measured(avg_score=30.0))
    for step in (10, 20, 30):
        trainer.step = step
        trainer._evaluate(900.0)
    assert [row['avg_score'] for row in trainer.eval_rows] == [10.0, 20.0, 30.0]
    assert [row['trailing_avg_score'] for row in trainer.eval_rows] == [10.0, 15.0, 20.0]


def test_re_evaluating_a_step_replaces_its_row_rather_than_adding_one(tmp_path, monkeypatch):
    """A killed run resumes from a step the history already has rows past.

    Appending would put two points at the same x and draw a vertical segment through the chart, and
    every step number in the report would be ambiguous.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, min_checkpoint_score=0.0)
    scripted(monkeypatch, measured(avg_score=10.0), measured(avg_score=99.0))
    trainer.step = 10
    trainer._evaluate(900.0)
    trainer._evaluate(900.0)
    assert [row['step'] for row in trainer.eval_rows] == [10]
    assert trainer.eval_rows[0]['avg_score'] == 99.0, 'the newer measurement must win'


# --- prefill and the resume payload -----------------------------------------------------------

def test_the_buffer_is_prefilled_before_the_first_gradient_step(tmp_path, monkeypatch):
    """Otherwise the opening gradient steps run against a buffer of one or two transitions.

    Every early batch would then be 128 copies of the same handful of frames, at the highest
    learning pressure of the run.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, initial_collect_steps=120)
    assert trainer.buffer.size == 0
    trainer._prefill()
    assert trainer.buffer.size >= 120


def test_a_prefilled_buffer_is_not_filled_twice(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, initial_collect_steps=50)
    trainer._prefill()
    size = trainer.buffer.size
    trainer._prefill()
    assert trainer.buffer.size == size, 'a resume must not re-run the prefill'


def test_the_resume_state_carries_the_buffer_beside_the_weights(tmp_path, monkeypatch):
    """Saved together and restored together, never one without the other.

    A resume that paired restored weights with a much newer buffer would train the network on
    experience it never generated — and a resume with no buffer at all silently re-runs the prefill
    at whatever epsilon the schedule has reached, which is not the same experience.
    """
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, initial_collect_steps=60)
    trainer._prefill()
    trainer.step = 10
    trainer._save_resume()
    assert os.path.exists(os.path.join(trainer.policy_dir, 'replay.npz'))

    again = make_trainer(tmp_path, monkeypatch=monkeypatch, initial_collect_steps=60)
    assert again.step == 10
    assert again.buffer.size >= 60, 'the buffer did not come back with the weights'


def test_the_resume_state_carries_the_schedules(tmp_path, monkeypatch):
    # Or a resumed arm restarts exploring at its initial epsilon, undoing the refinement it earned.
    trainer = make_trainer(tmp_path, monkeypatch=monkeypatch, guided_fraction=0.8)
    trainer.epsilon = 0.0125
    trainer.collector.set_guided_fraction(0.8)
    trainer.step = 10
    trainer._save_resume()
    again = make_trainer(tmp_path, monkeypatch=monkeypatch, guided_fraction=0.8)
    assert again.epsilon == pytest.approx(0.0125)
    assert again.collector.guided_fraction == pytest.approx(0.8)


# --- the config an arm actually got ---------------------------------------------------------------
#
# `grep 'hyperparameter override:'` is what the docs name as the way to confirm an arm got its
# config, and it is silent on every knob `env/constants.py` reads at import — which is the shaping
# set, and therefore exactly what a shaping batch is about. b2's `SNEK_CHASE_SAFE_SHAPING=0.1` had to
# be confirmed by reading `/proc/<pid>/environ` on the desktop. `train.py` prints
# `vectorized.config.describe()` at startup so it never has to be done that way again.

CONSTANTS_KNOB_RE = re.compile(r"_num\('([A-Z_]+)'")

# Read by the trainer's own config rather than describing the reward, so it already appears as a
# `hyperparameter override:` line and `describe()` is the wrong place for it.
KNOBS_NOT_IN_DESCRIBE = {'MIN_CHECKPOINT_SCORE'}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def constants_env_knobs():
    """Every `SNEK_*` knob `env/constants.py` reads through `_num`."""
    source = open(os.path.join(PROJECT_ROOT, 'env', 'constants.py')).read()
    return set(CONSTANTS_KNOB_RE.findall(source))


def describe_with(monkeypatch, knob, value):
    """`describe()` as it would read in a process started with `SNEK_<knob>=<value>`.

    Both modules are reloaded because these knobs are read **at import**, which is the whole reason
    they print no override line.
    """
    import importlib
    from env import constants as constants_module
    monkeypatch.setenv('SNEK_' + knob, str(value))
    importlib.reload(constants_module)
    importlib.reload(reward_config)
    return reward_config.describe()


@pytest.fixture(autouse=True)
def _restore_constants():
    """Puts both modules back, so a reload here cannot leak into another test."""
    yield
    import importlib
    from env import constants as constants_module
    importlib.reload(constants_module)
    importlib.reload(reward_config)


@pytest.mark.parametrize('knob', sorted(constants_env_knobs() - KNOBS_NOT_IN_DESCRIBE))
def test_describe_reflects_every_shaping_knob_the_overrides_miss(knob, monkeypatch):
    """Asserted by *changing* the knob, not by matching its name.

    `describe()` prints short labels — `dist`, `perfect`, `gate=` — so a name match would fail while
    the code was right, which is the fixture trap this project already has a rule about. Changing the
    value is the property that matters: a knob whose value cannot move the line is a knob nobody
    reading the log can confirm.

    The failure this prevents is a shaping batch that silently trained without its shaping.
    """
    before = reward_config.describe()
    after = describe_with(monkeypatch, knob, 7)
    assert after != before, '{0} does not appear in the reward-config line'.format(knob)


def test_the_trainer_prints_the_reward_config_not_just_the_overrides():
    assert 'reward_config.describe()' in open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'train.py')).read()


def test_describe_shows_a_shaping_term_that_is_on(monkeypatch):
    """The point of the line: a `c=0.1` arm must be distinguishable from a `c=0.0` one in the log."""
    assert 'chase_safe c=0.0' in reward_config.describe(), 'default is off'


# ---------------------------------------------------------------- the stage-A queue

def _run_arm(tmp_path, policy, monkeypatch, **overrides):
    """Trains one short arm to its cap and returns its eval rows. `SNEK_CHART_WINDOW` is already off."""
    trainer = make_trainer(tmp_path, policy=policy, monkeypatch=monkeypatch, **overrides)
    trainer.run()
    return trainer.eval_rows


MEASURED_FIELDS = ('step', 'avg_score', 'trailing_avg_score', 'min_score', 'max_score',
                   'avg_reward', 'perfect_percent')


def test_a_depth_zero_queue_reproduces_an_unqueued_arm_measurement_for_measurement(
        tmp_path, monkeypatch):
    """**The fixture the whole queue rests on.** At depth 0 the trainer reclaims each checkpoint in the
    same drain that offered it, so the row is built from the just-written `ckpt-<step>.pt` at the same
    seed with the schedule applied at the same point — every difference between the two modes is
    removed *except* the assembly path itself. So this exercises, end to end and with a bit-exact
    expectation: the two halves of a row meeting in the trainer, the step-ordered merge, the inverted
    checkpoint gate, `summarise` being one function, and the reclaim.

    `steps_per_second` is excluded because it is wall clock. `epsilon` and `guided_fraction` are
    excluded because they are deliberately a *different quantity* in the two modes — see the table in
    `Trainer._record` — and the next fixture pins that shift rather than waving at it.
    """
    unqueued = _run_arm(tmp_path / 'off', 'q1', monkeypatch, eval_queue=False)
    queued = _run_arm(tmp_path / 'on', 'q1', monkeypatch,
                      eval_queue=True, eval_queue_depth=0, eval_workers=0)
    assert len(queued) == len(unqueued) > 1
    for expected, actual in zip(unqueued, queued):
        assert ({key: expected[key] for key in MEASURED_FIELDS}
                == {key: actual[key] for key in MEASURED_FIELDS})


# A measurement good enough to clear a bootstrap reward threshold, so the schedule actually moves.
# **A fixture where epsilon never changes proves nothing about which epsilon a row reports**, and the
# end-to-end version of this test was exactly that: four evals of a freshly initialised net all sat at
# `initial_epsilon`, so asserting a one-row shift passed with the two columns identical. A mutation
# swapping the queued branch for the unqueued one survived it.
STRONG = {'avg_score': 30.0, 'min_score': 10.0, 'max_score': 40.0, 'avg_reward': 25.0,
          'perfect': 0.0}


def _fields(epsilon, guided=0.0):
    return {'epsilon': epsilon, 'guided_fraction': guided, 'steps_per_second': 100.0, 'fork': None}


def test_a_queued_row_reports_the_epsilon_the_arm_ran_under_not_the_one_just_computed(
        tmp_path, monkeypatch):
    """Queued, the column governs the interval *ending* at that step — see the table in `_record`.

    Driven through `_record` with a measurement that provably moves the schedule, so the two candidate
    values are different numbers and the assertion has something to distinguish.
    """
    trainer = make_trainer(tmp_path, policy='q2', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=8, eval_workers=0,
                           min_checkpoint_score=1e9)
    trainer._record(10, STRONG, _fields(epsilon=0.4, guided=0.0))
    row = trainer.eval_rows[-1]
    assert trainer.epsilon != 0.4, 'the measurement must move the schedule or this proves nothing'
    assert row['epsilon'] == 0.4, 'a queued row reports what the arm ran under'
    assert row['guided_fraction'] == 0.0


def test_an_unqueued_row_reports_the_epsilon_its_own_eval_just_set(tmp_path, monkeypatch):
    """The mirror, so the two modes' meanings are both pinned rather than only their difference."""
    trainer = make_trainer(tmp_path, policy='q2b', monkeypatch=monkeypatch,
                           eval_queue=False, min_checkpoint_score=1e9)
    trainer._record(10, STRONG, _fields(epsilon=0.4))
    row = trainer.eval_rows[-1]
    assert trainer.epsilon != 0.4
    assert row['epsilon'] == trainer.epsilon, 'an unqueued row reports what it just set'


def test_the_queued_epsilon_column_is_the_previous_rows_unqueued_value(tmp_path, monkeypatch):
    """The same difference end to end, over an arm whose epsilon is *checked* to be non-constant."""
    unqueued = _run_arm(tmp_path / 'off', 'q2c', monkeypatch, eval_queue=False,
                        initial_epsilon=0.4, max_steps=60)
    queued = _run_arm(tmp_path / 'on', 'q2c', monkeypatch,
                      eval_queue=True, eval_queue_depth=0, eval_workers=0,
                      initial_epsilon=0.4, max_steps=60)
    columns = [row['epsilon'] for row in unqueued]
    if len(set(columns)) == 1:
        pytest.skip('this arm never moved epsilon, so the shift is not observable here')
    assert [row['epsilon'] for row in queued][1:] == columns[:-1]


def test_every_landed_row_merges_without_waiting_for_an_earlier_one(tmp_path, monkeypatch):
    """**‡ The opposite of what this fixture first asserted, and the reversal is the finding.**

    Strict step order looked like a correctness requirement — it makes the trailing window hold exactly
    what an unqueued arm's would. But a streamed round does not complete in step order and cannot:
    `measure_stream` keeps every queued checkpoint resident and interleaves their episodes across 1024
    lanes, so completion order follows lane availability. Waiting for the front serialised the arm
    behind the slowest member of a round — measured at step 9,000 for 55 s with 2 of 9 rows unmerged.

    So a gap at the front must **not** hold anything back. Set up exactly that: 20 and 30 have landed,
    10 has not.
    """
    trainer = make_trainer(tmp_path, policy='q8', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=8, eval_workers=0)
    trainer.pending_evals = [10, 20, 30]
    scores = {10: 10.0, 20: 20.0, 30: 30.0}

    def land(step):
        held = {'scores': [scores[step]], 'perfect': [0], 'rewards': [1.0]}
        eval_queue.enqueue('q8', step, _fields(0.4), 1)
        eval_queue.claim('q8', step)
        eval_queue.complete('q8', step, held, _fields(0.4))

    land(20)
    land(30)
    assert trainer._merge_landed() == 2, 'a gap at the front must not hold up what is ready'
    assert trainer.pending_evals == [10], 'only the unmeasured step stays outstanding'

    land(10)
    assert trainer._merge_landed() == 1
    # **The file is still in step order** — that is what `merge_eval_row` is for, and it is the
    # property downstream readers actually depend on.
    assert [row['step'] for row in trainer.eval_rows] == [10, 20, 30]


def test_the_depth_bound_counts_work_still_being_measured(tmp_path, monkeypatch):
    """Otherwise the bound would count rows waiting to merge, and `_merge_landed` takes those every
    pass — so the number it bounded would be a different quantity from the schedule's blind spot."""
    trainer = make_trainer(tmp_path, policy='q8b', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=8, eval_workers=0)
    trainer.pending_evals = [10, 20]
    eval_queue.enqueue('q8b', 20, _fields(0.4), 1)
    eval_queue.claim('q8b', 20)
    eval_queue.complete('q8b', 20, {'scores': [1.0], 'perfect': [0], 'rewards': [1.0]},
                        _fields(0.4))
    trainer._merge_landed()
    assert trainer.pending_evals == [10]


def test_a_queued_arm_leaves_the_same_checkpoints_on_disk_as_an_unqueued_one(tmp_path, monkeypatch):
    """The gate is inverted, not removed: with the queue the file must exist for a worker to restore
    it, so `_settle_checkpoint` prunes instead of skipping. The set that survives has to be identical,
    or the queue would quietly change which checkpoints a stage-B wave can even select.
    """
    _run_arm(tmp_path / 'off', 'q3', monkeypatch, eval_queue=False)
    _run_arm(tmp_path / 'on', 'q3', monkeypatch,
             eval_queue=True, eval_queue_depth=0, eval_workers=0)
    kept = lambda root: sorted(checkpoints.steps(str(root / 'savedPolicies' / 'q3')))
    assert kept(tmp_path / 'on') == kept(tmp_path / 'off')


def test_a_queued_arm_finishes_with_no_rows_owed(tmp_path, monkeypatch):
    """The final drain. An arm that stopped with its queue full would otherwise close out missing rows
    for its last `depth` intervals — the newest region, which is the part a close-out reads.

    Run with a depth deeper than the arm has evals, so *every* row is still outstanding when the loop
    reaches the cap and only the final drain can produce them.
    """
    trainer = make_trainer(tmp_path, policy='q4', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=99, eval_workers=0)
    trainer.run()
    assert trainer.pending_evals == []
    assert [row['step'] for row in trainer.eval_rows] == [10, 20, 30, 40]


def test_a_queued_arm_makes_progress_with_no_workers_at_all(tmp_path, monkeypatch):
    """The forward-progress guarantee, stated as a test rather than as a comment.

    No worker is ever started here, so every row exists only because the trainer took the work back on
    reaching its bound. If the reclaim were removed this hangs instead of failing, which is why the
    step cap is small.
    """
    rows = _run_arm(tmp_path, 'q5', monkeypatch,
                    eval_queue=True, eval_queue_depth=1, eval_workers=0)
    assert [row['step'] for row in rows] == [10, 20, 30, 40]


def test_a_resume_adopts_the_rows_the_previous_run_never_merged(tmp_path, monkeypatch):
    """A killed arm's outstanding requests are owed rows, and the gap would land at the resume
    boundary — the worst place, since the resume step is re-measured and would sit beside holes.
    """
    first = make_trainer(tmp_path, policy='q6', monkeypatch=monkeypatch,
                         eval_queue=True, eval_queue_depth=99, eval_workers=0)
    first.run()
    # Put the arm back where a kill would leave it: history saved, three steps still owed.
    for step in (20, 30, 40):
        eval_queue.enqueue('q6', step, {'epsilon': 0.4, 'guided_fraction': 0.0,
                                        'steps_per_second': 1.0, 'fork': None}, 4)
    first.eval_rows = [row for row in first.eval_rows if row['step'] < 20]
    run_report.save_history(first.history_path, first.eval_rows, first.resume_steps)

    again = make_trainer(tmp_path, policy='q6', monkeypatch=monkeypatch,
                         eval_queue=True, eval_queue_depth=99, eval_workers=0)
    assert again.pending_evals == [20, 30, 40]


def test_a_resume_does_not_re_adopt_a_step_it_already_has_a_row_for(tmp_path, monkeypatch):
    """Otherwise a stale request would make the arm re-measure a step it has already published, and
    `merge_eval_row` would replace a good row with one measured from different weights."""
    first = make_trainer(tmp_path, policy='q7', monkeypatch=monkeypatch,
                         eval_queue=True, eval_queue_depth=99, eval_workers=0)
    first.run()
    eval_queue.enqueue('q7', 20, {'epsilon': 0.4, 'guided_fraction': 0.0,
                                  'steps_per_second': 1.0, 'fork': None}, 4)
    again = make_trainer(tmp_path, policy='q7', monkeypatch=monkeypatch,
                         eval_queue=True, eval_queue_depth=99, eval_workers=0)
    assert 20 not in again.pending_evals


def test_a_front_a_live_worker_is_measuring_is_left_alone(tmp_path, monkeypatch):
    """**The flaw that cost the whole speed-up, pinned.** A worker's round completes its checkpoints in
    whatever order lanes free up, so the front of the trainer's queue is often among the last to land
    while later steps pile up behind it as `.done`. The trainer then hit its bound and re-measured the
    front itself — 100 duplicated episodes per eval, and the worker's own result discarded.

    This process stands in for the worker: it holds the claim and is alive.
    """
    trainer = make_trainer(tmp_path, policy='q9', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=1, eval_workers=0)
    eval_queue.enqueue('q9', 10, _fields(0.4), 1)
    eval_queue.claim('q9', 10)
    assert trainer._leave_to_worker(10, waiting_since=time.time()) is True


def test_the_wait_for_a_worker_is_a_clock_and_not_a_queue_length(tmp_path, monkeypatch):
    """**The length version deadlocked and this is the fixture that says why.**

    "Wait while fewer than 4x depth are outstanding" can never fire: `_drain` blocks before returning
    to the training step that would grow the queue, so the length it waits on is frozen. A clock always
    advances, so a spent one must hand the work back even to a live holder.
    """
    trainer = make_trainer(tmp_path, policy='q9b', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=1, eval_workers=0)
    eval_queue.enqueue('q9b', 10, _fields(0.4), 1)
    eval_queue.claim('q9b', 10)
    spent = time.time() - train.QUEUE_WAIT_PATIENCE - 1.0
    assert trainer._leave_to_worker(10, waiting_since=spent) is False


def test_an_unclaimed_front_is_taken_immediately(tmp_path, monkeypatch):
    """No worker has it, so there is nothing to wait for — this is the no-workers-at-all path."""
    trainer = make_trainer(tmp_path, policy='q9c', monkeypatch=monkeypatch,
                           eval_queue=True, eval_queue_depth=1, eval_workers=0)
    eval_queue.enqueue('q9c', 10, _fields(0.4), 1)
    assert trainer._leave_to_worker(10, waiting_since=time.time()) is False


def test_a_dead_workers_claim_does_not_hold_the_queue(tmp_path, monkeypatch):
    """The other side of the same check: a claim is only worth waiting for while its owner exists.

    A pid that cannot exist stands in for a killed worker. Reading the pid out of the filename is what
    makes this answerable at all — a lock file with no pid in it could only be aged out.
    """
    queue = str(tmp_path / 'runs')
    eval_queue.enqueue('qa', 10, _fields(0.4), 1, queue)
    folder = eval_queue.policy_directory('qa', queue)
    os.rename(os.path.join(folder, '10.req'), os.path.join(folder, '10.claim.999999999'))
    assert eval_queue.held_by_live_worker('qa', 10, queue) is False
