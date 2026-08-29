"""Tests for the pure helpers in eval_checkpoints.py.

Only the file-and-arithmetic helpers are covered here. Anything that builds an agent or a
ParallelPyEnvironment needs TensorFlow and 20 subprocesses, which does not belong in a unit
test; those paths are exercised by running a real eval with a throwaway EVAL_OUT_SUFFIX.

Importing eval_checkpoints pulls in TensorFlow, which is slow but unavoidable — the helpers
live in the same module as the eval loop on purpose, so the docstring that explains the
protocol sits next to the code implementing it.
"""
import json
import os

import eval_checkpoints
import eval_plan
from snake_constants import RUNS_DIR


def write_result_file(policy_name, suffix, rows, complete=True, screen_episodes='omit'):
    """Puts a result file where load_finished_results() will look for it.

    `screen_episodes` records which protocol produced the file, which is what the resume guard
    reads. `'omit'` leaves the key out entirely, standing in for files written before it existed.
    """
    path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy_name, suffix))
    payload = {'policy_name': policy_name, 'complete': complete, 'results': rows}
    if screen_episodes != 'omit':
        payload['screen_episodes'] = screen_episodes
    with open(path, 'w') as handle:
        json.dump(payload, handle)
    return path


def row(step, episodes=100, perfect=70):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1)}


# ---------------------------------------------------------------- resume_suffixes

def test_resume_off_by_default():
    # Unset is the normal case and must never silently skip work.
    assert eval_checkpoints.resume_suffixes(None, '_top20') == []


def test_resume_off_for_falsey_spellings():
    for spec in ('', '0', 'false', 'False'):
        assert eval_checkpoints.resume_suffixes(spec, '_top20') == [], spec


def test_resume_1_means_this_runs_own_output_file():
    # The case that matters: relaunch the identical command with EVAL_RESUME=1.
    assert eval_checkpoints.resume_suffixes('1', '_top20') == ['_top20']
    assert eval_checkpoints.resume_suffixes('true', '') == ['']


def test_resume_takes_an_explicit_suffix_list():
    assert eval_checkpoints.resume_suffixes('_a,_b', '_top20') == ['_a', '_b']


def test_resume_list_tolerates_whitespace_and_empty_entries():
    assert eval_checkpoints.resume_suffixes(' _a , , _b ', '_x') == ['_a', '_b']


# --------------------------------------------------- resolve_screen_episodes

def test_screening_is_on_by_default():
    # The default is the decision this pins: a close-out with nothing set should screen.
    assert eval_checkpoints.resolve_screen_episodes(None, 100) == (20, None)


def test_screening_default_survives_an_empty_env_var():
    # `EVAL_SCREEN_EPISODES=` in a launch script must not be read as int('').
    assert eval_checkpoints.resolve_screen_episodes('', 100) == (20, None)


def test_screening_can_be_turned_off_explicitly():
    assert eval_checkpoints.resolve_screen_episodes('0', 100) == (0, None)


def test_screening_honours_an_explicit_length():
    assert eval_checkpoints.resolve_screen_episodes('40', 100) == (40, None)


def test_screening_default_stands_down_on_a_short_run():
    # EVAL_EPISODES=20 to sanity-check one checkpoint must not fail because the default screen is
    # also 20. It reports and carries on flat.
    screen, note = eval_checkpoints.resolve_screen_episodes(None, 20)
    assert screen == 0 and note and 'screening off' in note


def test_screening_rejects_an_explicit_length_that_cannot_confirm():
    try:
        eval_checkpoints.resolve_screen_episodes('100', 100)
    except SystemExit as error:
        assert 'must be below' in str(error)
    else:
        raise AssertionError('a screen as long as the full measurement should be rejected')


# ------------------------------------------------------- load_finished_results

def test_the_protocol_is_read_from_the_source_not_inferred():
    """The batch-18 guard.

    `b18a` and `b18d` were resumed from full-length rows that were the *stage-1 tier* of the
    three-stage protocol — not evidence of a flat run. The old code inferred "flat" from the depth of
    those rows and turned screening off, costing ~3x the episodes and leaving both arms without a
    `pooled_equal_effort` while their siblings had one.
    """
    keep, depth = eval_checkpoints.protocol_from_sources({20})
    assert keep is True and depth == 20


def test_a_recorded_flat_run_still_turns_screening_off():
    keep, depth = eval_checkpoints.protocol_from_sources({0})
    assert keep is False and depth == 0


def test_an_unrecorded_protocol_is_unknown_rather_than_flat():
    """Files predating the field must not be silently classified either way."""
    keep, _ = eval_checkpoints.protocol_from_sources({None})
    assert keep is None


def test_no_sources_at_all_is_unknown():
    keep, _ = eval_checkpoints.protocol_from_sources(set())
    assert keep is None


def test_any_screened_source_wins_over_a_flat_one():
    # An arm holding both is already inconsistent; continuing screened is the recoverable choice,
    # because a screen can be topped up to full length and a full row cannot be un-measured.
    keep, depth = eval_checkpoints.protocol_from_sources({0, 20})
    assert keep is True and depth == 20


def test_the_deepest_recorded_screen_depth_is_used():
    keep, depth = eval_checkpoints.protocol_from_sources({20, 30})
    assert keep is True and depth == 30


def test_none_alongside_a_real_depth_does_not_defeat_the_guard():
    keep, depth = eval_checkpoints.protocol_from_sources({None, 20})
    assert keep is True and depth == 20


def test_load_reports_the_protocol_of_files_it_actually_used():
    policy = 'unittest-protocol-src'
    path = write_result_file(policy, '_t', [row(100, episodes=100)], screen_episodes=20)
    try:
        _, _, screens, _ = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert screens == {20}, screens
    finally:
        os.remove(path)


def test_a_file_contributing_no_rows_does_not_report_its_protocol():
    """Otherwise a stale flat file with nothing usable in it could flip a screened arm."""
    policy = 'unittest-protocol-empty'
    flat = write_result_file(policy, '_flat', [row(100, episodes=20)], screen_episodes=0)
    screened = write_result_file(policy, '_screened', [row(200, episodes=100)],
                                 screen_episodes=20)
    try:
        _, steps, screens, _ = eval_checkpoints.load_finished_results(
            policy, ['_flat', '_screened'], 100)
        assert steps == {200}, steps
        assert screens == {20}, 'the flat file contributed no rows, so it should not vote'
    finally:
        os.remove(flat)
        os.remove(screened)


def test_load_returns_nothing_when_there_is_no_file():
    rows, steps, _, _ = eval_checkpoints.load_finished_results('no-such-policy-xyz', [''], 100)
    assert rows == [] and steps == set()


def test_load_skips_only_checkpoints_measured_to_the_full_episode_count():
    # The point of the >= test: a killed run leaves partial rows behind, and topping those up
    # would mean pooling summary statistics. Re-measuring is cheaper than getting that wrong.
    policy = 'unittest-resume-a'
    path = write_result_file(policy, '_t', [row(100, episodes=100), row(200, episodes=20),
                                            row(300, episodes=140)])
    try:
        rows, steps, _, _ = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert steps == {100, 300}, 'the 20-episode row must be re-measured, not skipped'
        assert [r['step'] for r in rows] == [100, 300]
    finally:
        os.remove(path)


def test_load_returns_rows_in_step_order_whatever_the_file_order():
    policy = 'unittest-resume-b'
    path = write_result_file(policy, '_t', [row(500), row(100), row(300)])
    try:
        rows, _, _, _ = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert [r['step'] for r in rows] == [100, 300, 500]
    finally:
        os.remove(path)


def test_load_takes_a_repeated_step_once_from_the_first_file_listed():
    # These are alternative records of one frozen checkpoint, not extra samples: loading both
    # would put the same step in `results` twice and double-count it in the pooled rate.
    policy = 'unittest-resume-c'
    first = write_result_file(policy, '_first', [row(100, perfect=80)])
    second = write_result_file(policy, '_second', [row(100, perfect=20), row(200)])
    try:
        rows, steps, _, _ = eval_checkpoints.load_finished_results(policy, ['_first', '_second'], 100)
        assert steps == {100, 200}
        assert len(rows) == 2, 'step 100 must appear once'
        assert [r for r in rows if r['step'] == 100][0]['perfect_games'] == 80, 'first file wins'
    finally:
        os.remove(first)
        os.remove(second)


def test_load_reads_an_incomplete_file():
    # The reason resume exists at all is a run that was killed part-way, so its file says
    # complete: false. Refusing to read that would make the feature useless.
    policy = 'unittest-resume-d'
    path = write_result_file(policy, '_t', [row(100), row(200)], complete=False)
    try:
        _, steps, _, _ = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert steps == {100, 200}
    finally:
        os.remove(path)


def test_load_ignores_a_missing_file_among_present_ones():
    policy = 'unittest-resume-e'
    path = write_result_file(policy, '_t', [row(100)])
    try:
        _, steps, _, _ = eval_checkpoints.load_finished_results(policy, ['_absent', '_t'], 100)
        assert steps == {100}
    finally:
        os.remove(path)


# -------------------------------------------------------------------- build_row

def held(perfect_flags, scores=None, rewards=None, seconds=1.0):
    n = len(perfect_flags)
    return {'perfect': list(perfect_flags),
            'scores': list(scores if scores is not None else range(n)),
            'rewards': list(rewards if rewards is not None else [1.0] * n),
            'seconds': seconds}


def test_build_row_counts_perfect_games_and_the_rate():
    r = eval_checkpoints.build_row(500, held([True] * 7 + [False] * 3))
    assert r['step'] == 500
    assert r['episodes'] == 10 and r['perfect_games'] == 7 and r['perfect_percent'] == 70.0


def test_build_row_recomputes_from_the_whole_sample_after_a_top_up():
    # The point of keeping raw episodes: a screen of 3/10 topped up with 14/20 must read 17/30,
    # and the median must come from all 30 rather than being averaged from two summaries.
    screen = held([True] * 3 + [False] * 7, scores=[1] * 10)
    screen['perfect'].extend([True] * 14 + [False] * 6)
    screen['scores'].extend([99] * 20)
    screen['rewards'].extend([1.0] * 20)
    r = eval_checkpoints.build_row(1, screen)
    assert r['episodes'] == 30 and r['perfect_games'] == 17
    assert r['perfect_percent'] == 56.7
    assert r['median_score'] == 99.0, 'median must come from all 30 episodes'
    assert r['min_score'] == 1.0 and r['max_score'] == 99.0


def test_build_row_carries_the_selection_metadata_through():
    r = eval_checkpoints.build_row(7, held([True]), {'selected_by': 'threshold90',
                                                     'single_eval': 90.0, 'surrounding': 61.5})
    assert r['selected_by'] == 'threshold90'
    assert r['graph_single_eval'] == 90.0 and r['graph_surrounding'] == 61.5


def test_build_row_defaults_metadata_for_an_explicit_step():
    r = eval_checkpoints.build_row(7, held([True]))
    assert r['selected_by'] == 'explicit'
    assert r['graph_single_eval'] is None and r['graph_surrounding'] is None


def test_build_row_ci_widens_as_the_sample_shrinks():
    wide = eval_checkpoints.build_row(1, held([True] * 7 + [False] * 3))
    narrow = eval_checkpoints.build_row(1, held([True] * 70 + [False] * 30))
    assert wide['perfect_percent'] == narrow['perfect_percent'] == 70.0
    assert (wide['perfect_ci95'][1] - wide['perfect_ci95'][0]) > \
           (narrow['perfect_ci95'][1] - narrow['perfect_ci95'][0])


# ------------------------------------------------------------ RowCache / WriteGate

def test_a_cached_row_is_not_rebuilt_on_the_next_pass():
    """The point of the cache. Both writers assemble the whole row list on every write, so a rebuild
    per pass is a rebuild per round — 97 ms x 125 on a 513-row arm."""
    cache = eval_checkpoints.RowCache()
    samples = {1000: held([True] * 7 + [False] * 3)}
    builds = []
    # Patched in `eval_plan`, not in `eval_checkpoints`: `RowCache.rows` resolves `build_row` in the
    # namespace it is defined in, and the re-export is a separate binding to the same function.
    real = eval_plan.build_row

    def counting(step, sample, meta=None):
        builds.append(step)
        return real(step, sample, meta)

    eval_plan.build_row = counting
    try:
        first = cache.rows(samples, {}, lambda step: None)
        second = cache.rows(samples, {}, lambda step: None)
    finally:
        eval_plan.build_row = real
    assert builds == [1000], builds
    # By reference, which is what makes the second pass free. Safe only because build_row copies the
    # episode_* lists out of the sample.
    assert second[0] is first[0]


def test_a_topped_up_sample_needs_a_put_or_its_row_freezes():
    """The invariant, stated as the failure it prevents: a screened row topped up to full length
    keeps reporting its screen depth if the mutator forgets to `put`."""
    cache = eval_checkpoints.RowCache()
    sample = held([True] * 15 + [False] * 5)
    samples = {1000: sample}
    assert cache.rows(samples, {}, lambda step: None)[0]['episodes'] == 20
    # A top-up, the way `measure` and `Wave.on_done` both do it: extend in place, then re-put.
    sample['perfect'].extend([True] * 80)
    sample['scores'].extend(range(80))
    sample['rewards'].extend([1.0] * 80)
    cache.put(1000, eval_checkpoints.build_row(1000, sample))
    assert cache.rows(samples, {}, lambda step: None)[0]['episodes'] == 100


def test_clear_drops_rows_cached_against_a_replaced_samples_dict():
    cache = eval_checkpoints.RowCache()
    samples = {1000: held([True] * 4)}
    assert cache.rows(samples, {}, lambda step: None)[0]['episodes'] == 4
    cache.clear()
    samples[1000] = held([True] * 9)
    assert cache.rows(samples, {}, lambda step: None)[0]['episodes'] == 9


def test_a_step_still_being_measured_has_no_row_yet():
    """An empty sample is the checkpoint in flight; its state travels in `in_flight`, not as a row
    with no episodes in it."""
    cache = eval_checkpoints.RowCache()
    samples = {1000: held([True] * 4), 2000: held([])}
    assert [r['step'] for r in cache.rows(samples, {}, lambda step: None)] == [1000]


def test_rows_come_out_in_step_order_whatever_the_dict_order():
    cache = eval_checkpoints.RowCache()
    samples = {3000: held([True]), 1000: held([True]), 2000: held([True])}
    resumed = {500: {'step': 500, 'episodes': 100}}
    assert [r['step'] for r in cache.rows(samples, resumed, lambda step: None)] == \
        [500, 1000, 2000, 3000]


def test_the_write_gate_admits_one_write_per_interval():
    gate = eval_checkpoints.WriteGate(interval=2.0)
    gate.record(now=100.0)
    assert not gate.due(now=101.9)
    assert gate.due(now=102.0)
    # Pure: asking does not consume the slot, so a caller that asks twice and writes once is honest.
    assert gate.due(now=102.0)
    gate.record(now=102.0)
    assert not gate.due(now=103.0)


def test_the_write_gate_defaults_to_the_shared_constant():
    # Read late rather than as a default argument, so the constant is patchable and so the class can
    # be defined above it in the file.
    assert eval_checkpoints.WriteGate().interval == eval_checkpoints.WRITE_MIN_INTERVAL


# --------------------------------------------------------------- skips_screening

def test_graph_100_percent_goes_straight_to_full_length():
    assert eval_checkpoints.skips_screening({'selected_by': 'threshold90', 'single_eval': 100.0})


def test_graph_90_percent_is_screened_first():
    # 90% is still in select_top_checkpoints' mandatory tier, so it is measured — just not at full
    # length unless it earns a confirmation slot. This is the line the whole split turns on.
    assert not eval_checkpoints.skips_screening({'selected_by': 'threshold90', 'single_eval': 90.0})


def test_lower_graph_points_are_screened_first():
    for single in (60.0, 70.0, 80.0):
        assert not eval_checkpoints.skips_screening({'single_eval': single}), single


def test_an_explicitly_named_step_goes_straight_to_full_length():
    # Naming a step is a request to measure it. The docs promise explicit steps bypass the
    # selection thresholds; leaving one at 20 episodes would quietly break that.
    assert eval_checkpoints.skips_screening({'selected_by': 'explicit'})
    assert eval_checkpoints.skips_screening({'selected_by': 'explicit', 'single_eval': None})
    assert eval_checkpoints.skips_screening(None)


# ---------------------------------------------------------------- plan_stages

def graph(single):
    return {'selected_by': 'threshold90', 'single_eval': single}


def batch10ish(n_full=140, n_rest=480):
    """Roughly b10d's shape: a large uncapped 100% tier and a much larger 90% tier."""
    steps = list(range(1, n_full + n_rest + 1))
    selected_by = {s: graph(100.0 if i < n_full else 90.0) for i, s in enumerate(steps)}
    return steps, selected_by


def test_plan_splits_the_full_tier_from_the_screened_one():
    steps, selected_by = batch10ish(3, 7)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    assert plan['full'] == [1, 2, 3]
    assert plan['screened'] == [4, 5, 6, 7, 8, 9, 10]
    assert plan['confirmed'] == 7, 'confirm count is capped by the screened pool, not padded'


def test_plan_prices_each_stage_correctly():
    steps, selected_by = batch10ish(2, 10)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 3, 100, 10)
    # 2 full x 100 + 10 screened x 20 + 3 confirmed x 80 = 200 + 200 + 240
    assert plan['episodes_planned'] == 640
    assert plan['flat_episodes'] == 1200
    # one measurement per full, per screen, and per confirmation
    assert plan['measurements_planned'] == 2 + 10 + 3


def test_plan_rounds_episodes_up_to_whole_rounds():
    # 12 workers cannot run exactly 100 or exactly 20 episodes: it runs 108 and 24.
    steps, selected_by = batch10ish(1, 1)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 1, 100, 12)
    # 1 x 108 (full) + 1 x 24 (screen) + 1 x 84 (80 rounded up) = 216
    assert plan['episodes_planned'] == 216


def test_plan_counts_resumed_rows_as_already_done_work():
    steps, selected_by = batch10ish(1, 1)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 1, 100, 10, resumed=5)
    assert plan['measurements_planned'] == 5 + 1 + 1 + 1
    assert plan['episodes_planned'] == 5 * 100 + 100 + 20 + 80
    assert plan['flat_episodes'] == 7 * 100


def test_plan_with_screening_off_is_one_flat_pass():
    steps, selected_by = batch10ish(3, 7)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 0, 30, 100, 10)
    assert plan['full'] == steps and plan['screened'] == [] and plan['confirmed'] == 0
    assert plan['episodes_planned'] == plan['flat_episodes'] == 1000


def test_plan_handles_an_arm_that_is_all_100_percent():
    # A very strong arm can leave nothing to screen. There is then no stage 3, and the plan is the
    # flat one — correctly, since every checkpoint is being measured at full length anyway.
    steps, selected_by = batch10ish(6, 0)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    assert plan['screened'] == [] and plan['confirmed'] == 0
    assert plan['episodes_planned'] == plan['flat_episodes'] == 600


def test_plan_handles_an_arm_with_no_100_percent_checkpoints():
    steps, selected_by = batch10ish(0, 8)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 2, 100, 10)
    assert plan['full'] == [] and len(plan['screened']) == 8
    assert plan['episodes_planned'] == 8 * 20 + 2 * 80


def test_default_confirm_count_is_100_and_still_pays_for_itself():
    """Pins the confirm count, and the reason it can be this high.

    100 rather than 30 because 30 recovered b10d's best non-100% checkpoint only 57% of the time
    against 97% at 100 — a coin flip on the headline number. The guard on the ratio is the other
    half of that decision: the count can be raised until screening stops being worth doing, and
    below about 2x a flat pass is simpler and gives every checkpoint a real measurement, so a
    future increase should have to justify itself here.
    """
    assert eval_checkpoints.DEFAULT_CONFIRM_COUNT == 100
    steps, selected_by = batch10ish(146, 514)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20,
                                        eval_checkpoints.DEFAULT_CONFIRM_COUNT, 100, 10)
    assert plan['confirmed'] == 100
    ratio = plan['flat_episodes'] / plan['episodes_planned']
    assert ratio > 2.0, 'screening must still beat a flat pass by a clear margin: %.2fx' % ratio


def test_plan_on_a_real_arm_shape_beats_a_flat_pass():
    # b10d's actual shape. The saving is smaller than the 3.6x of a pure screen-everything
    # protocol, because the uncapped 100% tier is 146 checkpoints at full length.
    steps, selected_by = batch10ish(146, 514)
    plan = eval_checkpoints.plan_stages(steps, selected_by, 20, 30, 100, 10)
    ratio = plan['flat_episodes'] / plan['episodes_planned']
    assert 2.3 < ratio < 2.5, ratio


# ------------------------------------------------------------ equal_effort_pooled

def sample(perfect_flags):
    return {'perfect': list(perfect_flags), 'scores': [1] * len(perfect_flags),
            'rewards': [1.0] * len(perfect_flags), 'seconds': 1.0}


def test_equal_effort_pooled_truncates_every_checkpoint_to_the_same_prefix():
    # The deep checkpoint is a perfect 100/100 and the shallow one 5/20. Pooling the rows would
    # give 105/120 = 87.5%; equal effort gives 25/40 = 62.5%, which is the arm-level rate.
    samples = {1: sample([True] * 100), 2: sample([True] * 5 + [False] * 15)}
    perfect, episodes, count = eval_checkpoints.equal_effort_pooled(samples, 20)
    assert (perfect, episodes, count) == (25, 40, 2)


def test_equal_effort_pooled_uses_the_prefix_not_the_best_part():
    # Order matters: the first 20 must be taken as they came, not sorted or sampled favourably.
    samples = {1: sample([False] * 20 + [True] * 80)}
    assert eval_checkpoints.equal_effort_pooled(samples, 20) == (0, 20, 1)


def test_equal_effort_pooled_skips_a_checkpoint_with_too_few_episodes():
    # A checkpoint interrupted mid-screen would otherwise be pooled at a different weight.
    samples = {1: sample([True] * 20), 2: sample([True] * 7)}
    assert eval_checkpoints.equal_effort_pooled(samples, 20) == (20, 20, 1)


def test_equal_effort_pooled_counts_the_full_tier_too():
    # The 100%-tier checkpoints never have a screening stage, so a snapshot taken at the end of
    # one could not include them. Truncating can.
    samples = {1: sample([True] * 100), 2: sample([True] * 100), 3: sample([True] * 10 + [False] * 10)}
    perfect, episodes, count = eval_checkpoints.equal_effort_pooled(samples, 20)
    assert count == 3 and episodes == 60 and perfect == 50


# --------------------------------------------------------------- pick_finalists

def screened(step, perfect, episodes=20, surrounding=None):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1),
            'graph_surrounding': surrounding}


def test_pick_finalists_takes_the_best_screen_rates():
    rows = [screened(1, 10), screened(2, 18), screened(3, 4), screened(4, 15)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 2)] == [2, 4]


def test_pick_finalists_returns_fewer_than_asked_rather_than_padding():
    rows = [screened(1, 10), screened(2, 18)]
    assert len(eval_checkpoints.pick_finalists(rows, 30)) == 2


def test_pick_finalists_ranks_on_the_rate_not_the_raw_count():
    # A checkpoint measured over more episodes must not win on volume alone.
    rows = [screened(1, 12, episodes=20), screened(2, 30, episodes=100)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 1)] == [1]


def test_pick_finalists_breaks_screen_ties_on_the_surrounding_rate():
    # A 20-episode screen only has 21 possible values, so ties are the common case, and the
    # surrounding graph rate is the measured-better tie-break (+0.48 vs +0.10).
    rows = [screened(1, 15, surrounding=40.0),
            screened(2, 15, surrounding=75.0),
            screened(3, 15, surrounding=60.0)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 2)] == [2, 3]


def test_pick_finalists_tolerates_a_missing_surrounding_rate():
    rows = [screened(1, 15, surrounding=None), screened(2, 15, surrounding=10.0)]
    assert [r['step'] for r in eval_checkpoints.pick_finalists(rows, 1)] == [2]


def test_pick_finalists_accepts_a_set_of_excluded_steps():
    # main() passes `set(resumed) | set(full_steps)`, not a dict, so the membership test has to
    # work for both.
    rows = [screened(1, 20), screened(2, 15), screened(3, 14)]
    picked = eval_checkpoints.pick_finalists(rows, 2, already_full={1})
    assert [r['step'] for r in picked] == [2, 3]


def test_a_perfect_screen_is_confirmed_even_past_the_quota():
    """20/20 is the strongest signal screening can give; the quota must not discard it.

    The confirm count rations a large middling pool. Cutting a perfect screen because the pool was
    full is selection working directly against the point of the close-out — the checkpoint most
    likely to be the arm's best never gets the episodes that would show it.
    """
    rows = [screened(1, 20), screened(2, 19), screened(3, 18), screened(4, 20), screened(5, 17)]
    picked = eval_checkpoints.pick_finalists(rows, 1)
    steps = [r['step'] for r in picked]
    # Quota of 1 takes step 1; step 4 also went 20/20 and comes along anyway.
    assert steps == [1, 4], steps


def test_the_quota_still_binds_on_everything_short_of_perfect():
    rows = [screened(1, 19), screened(2, 19), screened(3, 18), screened(4, 17)]
    assert len(eval_checkpoints.pick_finalists(rows, 2)) == 2


def test_perfect_screens_do_not_get_added_twice_when_inside_the_quota():
    rows = [screened(1, 20), screened(2, 20), screened(3, 15)]
    picked = eval_checkpoints.pick_finalists(rows, 3)
    assert [r['step'] for r in picked] == [1, 2, 3]
    assert len(picked) == len(set(r['step'] for r in picked))


def test_perfect_means_every_screened_episode_not_a_fixed_count():
    """A screen depth other than 20 still counts as perfect at full marks.

    The 10/10 row has to rank *below* the quota for this to test anything: an earlier version put it
    first, so it was chosen normally and a `perfect_games == 20` implementation passed.
    """
    rows = [screened(1, 20, surrounding=99.0),          # takes the only slot
            screened(2, 10, episodes=10, surrounding=1.0),  # perfect, but ranked last of the two
            screened(3, 18)]
    picked = eval_checkpoints.pick_finalists(rows, 1)
    steps = [r['step'] for r in picked]
    assert steps == [1, 2], 'a 10/10 screen is perfect too, and must be promoted past the quota'


def test_a_perfect_screen_already_measured_at_full_length_is_not_reconfirmed():
    rows = [screened(1, 19), screened(2, 20)]
    picked = eval_checkpoints.pick_finalists(rows, 1, already_full={2: rows[1]})
    assert [r['step'] for r in picked] == [1], 'step 2 already has the measurement'


def test_mandatory_confirms_keep_the_ranked_order():
    rows = [screened(1, 20, surrounding=10.0), screened(2, 20, surrounding=90.0),
            screened(3, 20, surrounding=50.0), screened(4, 12)]
    picked = eval_checkpoints.pick_finalists(rows, 1)
    # Ranked by surrounding rate within the 20/20 tie: 2, then 3, then 1.
    assert [r['step'] for r in picked] == [2, 3, 1], [r['step'] for r in picked]


def test_pick_finalists_excludes_steps_already_measured_at_full_length():
    # A resumed row already has the measurement this stage would buy, so spending a slot on it
    # would spend it on finished work.
    rows = [screened(1, 20), screened(2, 15), screened(3, 14)]
    picked = eval_checkpoints.pick_finalists(rows, 2, already_full={1: rows[0]})
    assert [r['step'] for r in picked] == [2, 3]


# ------------------------------------------------------------- wilson_interval

def test_wilson_interval_brackets_the_point_estimate():
    low, high = eval_checkpoints.wilson_interval(70, 100)
    assert low < 0.70 < high


def test_wilson_interval_stays_in_range_at_the_extremes():
    # Wilson's interval is asymmetric and never reaches the boundary exactly, which is the
    # point of using it here: 0/100 still gets a non-zero upper bound rather than collapsing
    # to a useless [0, 0] the way the normal approximation does.
    low, high = eval_checkpoints.wilson_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.1
    low, high = eval_checkpoints.wilson_interval(100, 100)
    assert 0.9 < low < 1.0 and high <= 1.0
    assert eval_checkpoints.wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_tightens_with_more_episodes():
    # This is the whole argument for spending 500 episodes on a finalist instead of 100.
    narrow = eval_checkpoints.wilson_interval(350, 500)
    wide = eval_checkpoints.wilson_interval(70, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0]) / 2


# ------------------------------------------- early abandonment: EVAL_MIN_ACHIEVABLE

def test_achievable_percent_assumes_every_remaining_episode_is_perfect():
    # 30 perfect of 40 run, 60 left: the ceiling is (30 + 60) / 100.
    assert eval_checkpoints.achievable_percent(30, 40, 100) == 90.0
    # Nothing left to run, so the ceiling is the actual rate.
    assert eval_checkpoints.achievable_percent(72, 100, 100) == 72.0
    # A perfect start cannot exceed 100%.
    assert eval_checkpoints.achievable_percent(10, 10, 100) == 100.0


def test_achievable_percent_is_monotone_non_increasing():
    # The whole basis of the stopping rule: once the ceiling drops below the gate it can never
    # come back, so stopping is safe. Any counter-example would make the rule discard a
    # checkpoint that could still recover.
    for perfect_rate in (0.0, 0.4, 0.85, 1.0):
        ceilings = [eval_checkpoints.achievable_percent(int(round(done * perfect_rate)), done, 100)
                    for done in range(10, 101, 10)]
        assert all(b <= a + 1e-9 for a, b in zip(ceilings, ceilings[1:])), ceilings


def test_abandons_only_when_the_gate_is_arithmetically_out_of_reach():
    test = eval_checkpoints.make_abandon_test(85.0, 100, 20)
    # 4 perfect of 20: ceiling is 84%, below the gate by one episode. Stop.
    assert test(4, 20) is True
    # 5 perfect of 20: ceiling is exactly 85%, still reachable. Keep going.
    assert test(5, 20) is False
    # Deep in the run: 74 of 90 can still reach 84%, 75 can reach 85%.
    assert test(74, 90) is True
    assert test(75, 90) is False


def test_a_checkpoint_that_would_reach_the_gate_is_never_abandoned():
    # The property that makes this free rather than a trade-off. Walk every prefix of a run that
    # finishes at exactly the gate, and at 100%: the rule must never fire on either.
    test = eval_checkpoints.make_abandon_test(85.0, 100, 20)
    for final in (85, 100):
        for done in range(10, 101, 10):
            # Worst case ordering for the rule: every failure lands as early as possible.
            failures = min(100 - final, done)
            perfect = done - failures
            assert test(perfect, done) is False, (final, done, perfect)


def test_an_abandoned_row_can_never_outrank_a_kept_one():
    # An abandoned row's own rate is always strictly below the gate, so a truncated row cannot
    # win a best-checkpoint comparison it did not earn.
    test = eval_checkpoints.make_abandon_test(85.0, 100, 20)
    for done in range(20, 100, 10):
        for perfect in range(0, done + 1):
            if test(perfect, done):
                assert 100.0 * perfect / done < 85.0, (done, perfect)


def test_the_floor_protects_the_equal_effort_pool():
    # equal_effort_pooled truncates to screen_episodes and *skips* rows shorter than that, so
    # abandoning below the floor would silently delete checkpoints from the arm-level figure.
    #
    # At the shipped defaults the floor is slack rather than load-bearing: reaching a ceiling below
    # 85% of 100 needs more than 15 failures, which cannot happen in the first 10 episodes. It
    # binds as soon as either knob moves, so test it where it bites — a 95% gate can fire on the
    # very first round, and the floor is what stops it.
    ungated = eval_checkpoints.make_abandon_test(95.0, 100, 0)
    assert ungated(4, 10) is True
    floored = eval_checkpoints.make_abandon_test(95.0, 100, 20)
    assert floored(4, 10) is False
    assert floored(4, 20) is True

    # Same story with a deeper screen, which is how the floor is actually raised at startup.
    deep = eval_checkpoints.make_abandon_test(85.0, 100, 30)
    assert eval_checkpoints.make_abandon_test(85.0, 100, 20)(0, 20) is True
    assert deep(0, 20) is False
    assert deep(0, 30) is True

    # And a row abandoned at the floor is exactly long enough to still count in the pool.
    samples = {1: {'perfect': [1] * 5 + [0] * 15}, 2: {'perfect': [1] * 18 + [0] * 82}}
    perfect, episodes, count = eval_checkpoints.equal_effort_pooled(samples, 20)
    assert (perfect, episodes, count) == (5 + 18, 40, 2)
    # A row one episode shorter is dropped entirely — the failure mode the floor prevents.
    samples[1]['perfect'] = samples[1]['perfect'][:19]
    assert eval_checkpoints.equal_effort_pooled(samples, 20) == (18, 20, 1)


def test_a_completed_checkpoint_is_not_abandoned():
    test = eval_checkpoints.make_abandon_test(85.0, 100, 20)
    assert test(0, 100) is False
    assert test(0, 120) is False


def test_the_gate_can_be_turned_off():
    assert eval_checkpoints.make_abandon_test(0, 100, 20) is None
    assert eval_checkpoints.make_abandon_test(None, 100, 20) is None


def test_the_gate_scales_with_a_shorter_target():
    # EVAL_EPISODES=40 with the same gate: 4 of 20 leaves a ceiling of (4 + 20) / 40 = 60%.
    test = eval_checkpoints.make_abandon_test(85.0, 40, 20)
    assert test(4, 20) is True
    # 14 of 20 leaves a ceiling of exactly 85%, which is reachable, so it survives — the boundary
    # is inclusive on the keep side at every target length.
    assert eval_checkpoints.achievable_percent(14, 20, 40) == 85.0
    assert test(14, 20) is False
    assert test(13, 20) is True


def test_the_default_gate_is_97_and_stops_a_100_episode_run_after_4_failures():
    # The gate is the bar a checkpoint has to clear to be worth keeping at all. Pinned as
    # arithmetic rather than as a bare constant, because what matters downstream is where the rule
    # fires, not the number itself.
    #
    # Raised 95 -> 97 on 2026-08-19, with the selection tiers and the 20-episode graph eval. The
    # old form of this test read "after 6 failures"; at 97 it is 4, which is the point of the
    # change — a tighter gate abandons sooner and is most of the close-out saving on arms whose
    # checkpoints sit just under the bar.
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE == 97.0
    test = eval_checkpoints.make_abandon_test(eval_checkpoints.DEFAULT_MIN_ACHIEVABLE, 100,
                                             eval_checkpoints.DEFAULT_ABANDON_FLOOR)
    # 3 failures still allow exactly 97%, so the run continues; the 4th makes 97% unreachable.
    assert eval_checkpoints.achievable_percent(47, 50, 100) == 97.0
    assert test(47, 50) is False
    assert test(46, 50) is True
    # Every looser gate tolerates more failures and would keep that going, so a revert to any of
    # them cannot pass silently. 95 is included because that is the value this replaced.
    for looser in (95.0, 90.0, 85.0):
        assert eval_checkpoints.make_abandon_test(looser, 100, 20)(46, 50) is False
    # And the gate must stay strictly below the HOF selection gate, or the re-measure starves.
    assert eval_checkpoints.DEFAULT_MIN_ACHIEVABLE < eval_checkpoints.DEFAULT_ABOVE_THRESHOLD

    # The floor still binds first: 97% of 100 is unreachable after 4 failures, which can happen
    # inside the first 20 episodes, so the floor is what keeps a row long enough for
    # equal_effort_pooled. The tighter gate makes the floor bind more often, not less.
    assert test(4, 10) is False, 'the floor must suppress a check below 20 episodes'
    assert test(4, 20) is True


# ------------------------------------------- best checkpoint under a gate that can truncate everything

def _row(step, perfect_percent, episodes):
    return {'step': step, 'perfect_percent': perfect_percent, 'episodes': episodes}


def test_best_checkpoint_ignores_shallow_screens_when_a_full_length_row_exists():
    # A 20-episode screen can read 100% on luck across hundreds of tries, so it must never beat an
    # honestly-measured row.
    rows = [_row(1000, 100.0, 20), _row(2000, 96.0, 100), _row(3000, 91.0, 100)]
    assert eval_checkpoints.best_full_length_row(rows, 100)['step'] == 2000


def test_best_checkpoint_falls_back_to_deep_rows_not_to_every_row():
    # The case the 90% gate makes reachable: no checkpoint cleared the gate, so every 100-episode
    # target was abandoned short and there is no full-length row at all. Falling back to *all* rows
    # would crown the 20-episode screen on its lucky 20/20 — which is what the old
    # `or results` did, and is exactly what the deep-row rule exists to prevent.
    rows = [_row(1000, 100.0, 20), _row(2000, 88.0, 92), _row(3000, 86.0, 89)]
    best = eval_checkpoints.best_full_length_row(rows, 100)
    assert best['step'] == 2000, 'a 20-episode screen must not win by default'
    assert best['episodes'] == 92


def test_best_checkpoint_relaxes_all_the_way_only_when_nothing_is_deep():
    # If every row really is shallow there is nothing better to report, so it still returns the
    # best of what exists rather than None — a missing line would read as "the arm failed".
    rows = [_row(1000, 70.0, 20), _row(2000, 80.0, 20)]
    assert eval_checkpoints.best_full_length_row(rows, 100)['step'] == 2000
    assert eval_checkpoints.best_full_length_row([], 100) is None


def test_best_checkpoint_treats_half_depth_as_the_relaxed_boundary():
    # Half of num_episodes, matching eval_progress.deep_rows, so the two agree on what is rankable.
    # 50 is in, 49 is out; a row abandoned under a 90% gate has ~89 episodes so it lands well inside.
    rows = [_row(1000, 100.0, 49), _row(2000, 60.0, 50)]
    assert eval_checkpoints.best_full_length_row(rows, 100)['step'] == 2000


# ------------------------------------- per-episode storage, and screens that survive a resume

def raw_held(scores, perfect, rewards=None, seconds=1.0):
    """A `held` sample in (scores, perfect) order.

    Named apart from this file's older `raw_held(perfect_flags, scores=...)` helper, which takes
    its arguments the other way round — the collision was a TypeError, not a stale test.
    """
    return {'scores': list(scores), 'perfect': list(perfect),
            'rewards': list(rewards if rewards is not None else [1.0] * len(scores)),
            'seconds': seconds}


def test_build_row_stores_the_raw_episodes():
    r = eval_checkpoints.build_row(1000, raw_held([95, 61, 95], [1, 0, 1], [180.0, 40.0, 181.0]))
    assert r['episode_scores'] == [95, 61, 95]
    assert r['episode_perfect'] == [1, 0, 1]
    assert r['episode_rewards'] == [180.0, 40.0, 181.0]
    assert r['episodes'] == 3 and r['perfect_games'] == 2


def test_a_row_round_trips_through_held_from_row():
    original = raw_held([95, 61, 88], [1, 0, 0], [180.0, 40.0, 90.0], seconds=12.5)
    row = eval_checkpoints.build_row(1000, original)
    back = eval_checkpoints.held_from_row(row)
    assert back['scores'] == [95, 61, 88]
    assert back['perfect'] == [1, 0, 0]
    assert back['rewards'] == [180.0, 40.0, 90.0]
    assert back['seconds'] == 12.5


def test_topping_up_a_restored_sample_matches_measuring_it_in_one_go():
    """The property that makes resuming a screen safe: pooling raw episodes is exact.

    The median is the reason this needs raw episodes rather than summary statistics — it cannot be
    pooled from two summaries, so a topped-up row rebuilt from summaries would be quietly wrong.
    """
    first = [95, 61, 88, 90, 12]
    second = [95, 95, 40, 77, 95]
    flags_first, flags_second = [1, 0, 0, 0, 0], [1, 1, 0, 0, 1]

    # measured in one pass
    one_go = eval_checkpoints.build_row(
        1000, raw_held(first + second, flags_first + flags_second))
    # screened, stored, restored, then topped up
    screen_row = eval_checkpoints.build_row(1000, raw_held(first, flags_first))
    restored = eval_checkpoints.held_from_row(screen_row)
    restored['scores'].extend(second)
    restored['perfect'].extend(flags_second)
    restored['rewards'].extend([1.0] * len(second))
    topped = eval_checkpoints.build_row(1000, restored)

    for key in ('episodes', 'perfect_games', 'perfect_percent', 'avg_score', 'median_score',
                'min_score', 'max_score', 'perfect_ci95'):
        assert topped[key] == one_go[key], '{0}: {1} != {2}'.format(key, topped[key], one_go[key])


def test_held_from_row_returns_none_for_a_row_without_episode_data():
    assert eval_checkpoints.held_from_row(row(100, episodes=100)) is None


def test_held_from_row_rejects_a_row_whose_lists_disagree_with_its_count():
    r = eval_checkpoints.build_row(1000, raw_held([95, 61], [1, 0]))
    r['episodes'] = 5          # hand-edited or truncated
    assert eval_checkpoints.held_from_row(r) is None, 'a mismatched row must not be pooled'


def test_resume_carries_a_completed_screen_instead_of_discarding_it():
    """The b18a incident: 192 screens and 7,534 episodes were thrown away by a resume."""
    policy = 'unittest-carry-screen'
    screen = eval_checkpoints.build_row(200, raw_held([95] * 20, [1] * 20))
    full = eval_checkpoints.build_row(100, raw_held([95] * 100, [1] * 100))
    path = write_result_file(policy, '_t', [full, screen], screen_episodes=20)
    try:
        rows, steps, _, partial = eval_checkpoints.load_finished_results(policy, ['_t'], 100)
        assert steps == {100}, 'the full-length row is still a plain skip'
        assert set(partial) == {200}, 'the 20-episode screen should be carried, not dropped'
        assert len(partial[200]['scores']) == 20
    finally:
        os.remove(path)


def test_a_full_length_row_supersedes_a_partial_one_for_the_same_step():
    policy = 'unittest-carry-supersede'
    partial_row = eval_checkpoints.build_row(300, raw_held([95] * 20, [1] * 20))
    full_row = eval_checkpoints.build_row(300, raw_held([95] * 100, [1] * 100))
    first = write_result_file(policy, '_a', [partial_row], screen_episodes=20)
    second = write_result_file(policy, '_b', [full_row], screen_episodes=20)
    try:
        _, steps, _, partial = eval_checkpoints.load_finished_results(policy, ['_a', '_b'], 100)
        assert steps == {300}
        assert partial == {}, 'a step measured to full length must not also be carried as partial'
    finally:
        os.remove(first)
        os.remove(second)


def test_the_deeper_partial_wins_when_two_files_hold_the_same_step():
    policy = 'unittest-carry-deeper'
    shallow = eval_checkpoints.build_row(400, raw_held([95] * 20, [1] * 20))
    deeper = eval_checkpoints.build_row(400, raw_held([95] * 60, [1] * 60))
    first = write_result_file(policy, '_a', [shallow], screen_episodes=20)
    second = write_result_file(policy, '_b', [deeper], screen_episodes=20)
    try:
        _, _, _, partial = eval_checkpoints.load_finished_results(policy, ['_a', '_b'], 100)
        assert len(partial[400]['scores']) == 60, 'more episodes is strictly more information'
    finally:
        os.remove(first)
        os.remove(second)


# --------------------------------------------------------- select_checkpoints_above (HOF)

def test_select_above_takes_only_checkpoints_at_or_above_the_threshold():
    pol = '_hoftest_above'
    path = write_result_file(pol, '', [
        row(1000, perfect=97),   # 97.0% -> below 98, excluded
        row(2000, perfect=98),   # 98.0% -> at the bar, included
        row(3000, perfect=100),  # 100%  -> included
    ])
    try:
        steps, meta = eval_checkpoints.select_checkpoints_above(pol, {1000, 2000, 3000}, 98)
    finally:
        os.remove(path)
    assert steps == [2000, 3000], steps
    assert meta[2000]['closeout_percent'] == 98.0
    assert meta[2000]['selected_by'] == 'above98'
    # No single_eval key -> the step skips screening and goes straight to full length.
    assert 'single_eval' not in meta[2000]


def test_select_above_skips_abandoned_and_missing_checkpoints():
    pol = '_hoftest_above2'
    rows = [row(1000, perfect=99),
            row(2000, episodes=30, perfect=30),   # 100% but its checkpoint file is gone
            {'step': 3000, 'episodes': 40, 'perfect_games': 40, 'perfect_percent': 100.0,
             'abandoned': True}]                   # abandoned rows never qualify, even at 100%
    path = write_result_file(pol, '', rows)
    try:
        # 3000 IS available, so its exclusion is the abandoned filter, not a missing checkpoint;
        # 2000 is absent from `available`, standing in for an evicted checkpoint.
        steps, _ = eval_checkpoints.select_checkpoints_above(pol, {1000, 3000}, 98)
    finally:
        os.remove(path)
    assert steps == [1000], steps


def test_select_above_returns_empty_when_nothing_qualifies():
    pol = '_hoftest_above3'
    path = write_result_file(pol, '', [row(1000, perfect=90), row(2000, perfect=95)])
    try:
        steps, meta = eval_checkpoints.select_checkpoints_above(pol, {1000, 2000}, 98)
    finally:
        os.remove(path)
    assert steps == [] and meta == {}


def test_select_above_missing_file_is_an_error():
    try:
        eval_checkpoints.select_checkpoints_above('_hoftest_no_such_policy', {1}, 98)
        assert False, 'expected SystemExit'
    except SystemExit:
        pass
