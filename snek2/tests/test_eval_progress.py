"""Tests for eval_progress.py's aggregation, which has to read two protocols' output.

`summarize()` sees result files from the flat one-pass protocol and from the screening protocol
at the same time — a resumed arm can even produce both — so the fallbacks matter as much as the
new fields do.
"""
import os
import shutil
import tempfile
import time

import eval_progress


def result(step, episodes=100, perfect=70, seconds=100.0):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1),
            'perfect_ci95': [0.0, 100.0], 'avg_score': 90.0, 'seconds': seconds}


def run(results, complete=True, **extra):
    payload = {'suffix': '_t', 'results': results, 'complete': complete,
               'updated_at': 1e12, 'mtime': 1e12}
    payload.update(extra)
    # None means "not recorded", which is a real case (files predating the field), so it has to be
    # representable rather than silently dropped by dict.update.
    return {k: v for k, v in payload.items() if not (k == 'episodes_per_checkpoint' and v is None)}


# ------------------------------------------------------------------- summary panel geometry

def test_fit_summary_panel_gives_the_slack_to_the_charts():
    """The summary block's height is the worst case (16 lines); a flat HOF pass writes about 10, and
    the leftover rendered as 324 px of blank -- 17% of the frame -- under the last line of text. In
    `chart_viewer`'s 2x2 that is the band between the rows."""
    # bottom-up: summary at 2.75in, two charts at 1.68in, 0.71in gaps, 8.56in frame
    h = 8.56
    boxes = [(0.02, 2.75 / h), (0.02 + (2.75 + 0.71) / h, 1.68 / h),
             (0.02 + (2.75 + 0.71 + 1.68 + 0.71) / h, 1.68 / h)]
    fitted = eval_progress.fit_summary_panel(boxes, 1.75, h)
    (t0, th), (c2_0, c2h), (c1_0, c1h) = fitted
    assert abs(th * h - 1.75) < 1e-9                       # the block is exactly what it needs
    assert t0 == boxes[0][0]                               # anchored at the bottom margin
    freed = 2.75 - 1.75
    assert abs(c2h * h - (1.68 + freed / 2)) < 1e-9         # split evenly between the charts
    assert abs(c1h * h - (1.68 + freed / 2)) < 1e-9
    # the gaps and the top of the upper chart are exactly where they were
    assert abs((c2_0 - (t0 + th)) * h - 0.71) < 1e-9
    assert abs((c1_0 - (c2_0 + c2h)) * h - 0.71) < 1e-9
    assert abs((c1_0 + c1h) - (boxes[2][0] + boxes[2][1])) < 1e-12


def test_fit_summary_panel_never_grows_the_block():
    """The gridspec height is the capacity, and a block that wants more than that overflows into the
    bottom margin rather than stealing from the charts -- which is what the margin is left for."""
    h = 8.56
    boxes = [(0.02, 2.75 / h), (0.4, 1.68 / h), (0.7, 1.68 / h)]
    assert eval_progress.fit_summary_panel(boxes, 3.5, h) == list(boxes)
    assert eval_progress.fit_summary_panel(boxes, 2.75, h) == list(boxes)
    # degenerate inputs are returned untouched rather than raising
    assert eval_progress.fit_summary_panel(boxes, 1.0, 0) == list(boxes)
    assert eval_progress.fit_summary_panel([(0.0, 0.0), (0.4, 0.1), (0.7, 0.1)], 1.0, h) == \
        [(0.0, 0.0), (0.4, 0.1), (0.7, 0.1)]


def test_the_rendered_frame_has_almost_no_trailing_blank():
    """The property, measured on a real render rather than inferred from the geometry: the blank
    below the last line of text is what lands between the rows of the viewer's grid. It was 324 of
    1883 px (17%); at three slack lines it was still 233, because the gap a reader sees also includes
    the figure's bottom margin, the last line's own descender space and the next panel's top margin.
    """
    import numpy as np
    import matplotlib.image as mpimg

    state = eval_progress.summarize([run(
        [result(step, episodes=500, perfect=490, seconds=250.0) for step in (1000, 2000, 3000)],
        complete=False, episodes_per_checkpoint=500, measurements_planned=10,
        measurements_done=3, session_measurements=3, session_episodes=1500,
        session_seconds=750.0)])
    out = os.path.join(tempfile.mkdtemp(prefix='chartfit-'), 'frame.png')
    try:
        eval_progress.render('b43a-lowlr-b29b', state, out)
        image = mpimg.imread(out)[:, :, :3]
        height = image.shape[0]
        background = image[2, 2]
        ink = np.where(np.abs(image - background).max(axis=(1, 2)) > 0.02)[0]
        blank = height - 1 - int(ink.max())
        # 6%: the fit renders 47 of 941 px (5.0%) at the default 110 dpi, and the figure's own
        # bottom margin is the other half of this gap -- 0.045 instead of 0.02 puts it at ~70 px, so
        # a looser bar here would let the margin drift back without failing anything.
        assert blank < 0.06 * height, '{0} of {1} px blank below the last line'.format(blank, height)
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)


# ------------------------------------------------------------------- best_of

def test_best_of_is_plain_max_when_every_row_has_the_same_length():
    rows = [result(1, perfect=70), result(2, perfect=85), result(3, perfect=60)]
    assert eval_progress.best_of(rows)['step'] == 2


def test_best_of_ignores_a_lucky_short_screen():
    # The failure this exists to prevent: a 20-episode screen going 19/20 is 95% and would be
    # crowned over a checkpoint honestly measured at 88/100.
    rows = [result(1, episodes=100, perfect=88), result(2, episodes=20, perfect=19)]
    assert eval_progress.best_of(rows)['step'] == 1


def test_best_of_tolerates_a_worker_count_that_rounds_episodes_up():
    # 12 workers turn a 100-episode request into 108. Those rows are still full measurements and
    # must stay eligible, which is why the floor is half the deepest rather than an exact match.
    rows = [result(1, episodes=108, perfect=70), result(2, episodes=100, perfect=90)]
    assert eval_progress.best_of(rows)['step'] == 2


def test_best_of_falls_back_to_screens_when_nothing_is_confirmed_yet():
    rows = [result(1, episodes=20, perfect=12), result(2, episodes=20, perfect=16)]
    assert eval_progress.best_of(rows)['step'] == 2


def test_best_of_handles_no_rows():
    assert eval_progress.best_of([]) is None


# ----------------------------------------------------------------- summarize

def test_summarize_counts_checkpoints_for_the_flat_protocol():
    state = eval_progress.summarize([run([result(1), result(2)], checkpoints_requested=4)])
    assert state['done'] == 2 and state['requested'] == 4
    assert state['percent_done'] == 50.0
    assert state['unit'] == 'checkpoints'


def test_summarize_counts_measurements_for_the_screening_protocol():
    # Stage 1 finishing must not read as 100% done: the confirmations are still to come.
    state = eval_progress.summarize([run(
        [result(1, episodes=20, perfect=14), result(2, episodes=20, perfect=12)],
        complete=False, checkpoints_requested=2, screen_episodes=20, confirm_count=1,
        measurements_planned=3, measurements_done=2,
        episodes_planned=120, episodes_done=40)])
    assert state['done'] == 2 and state['requested'] == 3
    assert state['unit'] == 'measurements'
    assert round(state['percent_done']) == 67


def test_summarize_prices_the_eta_in_episodes_not_checkpoints():
    # Two rows of 100s cover 40 episodes, so 5s an episode; 80 episodes remain => 400s on one
    # live process. Counting checkpoints instead would call it 1 remaining x 100s = 100s, a 4x
    # underestimate, because the confirmation pass is four times longer than the screen it
    # follows.
    state = eval_progress.summarize([run(
        [result(1, episodes=20, perfect=14, seconds=100.0),
         result(2, episodes=20, perfect=12, seconds=100.0)],
        complete=False, screen_episodes=20, measurements_planned=3, measurements_done=2,
        episodes_planned=120, episodes_done=40,
        in_flight={'step': 3, 'round': 1, 'rounds_total': 2, 'perfect_so_far': 1,
                   'episodes_so_far': 10, 'running_percent': 10.0,
                   'per_round_perfect': [1], 'started_at': 1e12})])
    assert abs(state['eta_seconds'] - 400.0) < 0.001


def test_the_eta_is_deflated_by_the_gates_observed_savings():
    """`remaining_episodes` prices the plan ahead at full length and said so, calling the missing
    prediction of future abandonment a small upward bias. On a HOF pass at gate 98 it is the dominant
    term: `b43d-lowlr-b29c` had abandoned 23 of 23 at a mean of 262 of 500 episodes, so the chart read
    4h25m against a real ~2h."""
    rows = [result(step, episodes=250, perfect=200, seconds=125.0) for step in range(1, 5)]
    payload = run(rows, complete=False, episodes_per_checkpoint=500,
                  measurements_planned=8, measurements_done=4,
                  session_measurements=4, session_episodes=1000, session_seconds=500.0)
    state = eval_progress.summarize([payload])
    # 4 rows x 250 of a 500 target => half of every planned episode is actually run.
    assert abs(eval_progress.expected_run_fraction([payload]) - 0.5) < 1e-9
    # 4 ahead x 500 planned = 2000 episodes, deflated to 1000, at 0.5 s/episode => 500 s.
    assert abs(state['eta_seconds'] - 500.0) < 0.001
    assert abs(state['run_fraction'] - 0.5) < 1e-9


def test_the_deflator_ignores_the_session_counters_a_resume_breaks():
    """The tempting source is `session_episodes + episodes_saved`, the identity `eval_checkpoints`
    prints. It is wrong across a resume: a resumed row the gate re-abandons from its stored samples
    runs ~0 new episodes while still reporting the whole shortfall as saved. On b43a mid-resume that
    read **0.025**, which would have turned a 4-hour estimate into six minutes."""
    rows = [result(step, episodes=250, perfect=200, seconds=125.0) for step in range(1, 5)]
    payload = run(rows, complete=False, episodes_per_checkpoint=500,
                  measurements_planned=8, measurements_done=4,
                  session_measurements=2, session_episodes=40, session_seconds=20.0,
                  episodes_saved=1560)
    assert abs(eval_progress.expected_run_fraction([payload]) - 0.5) < 1e-9


def test_the_deflator_stands_down_without_evidence_and_on_a_staged_run():
    one = run([result(1, episodes=250, perfect=200)], complete=False,
              episodes_per_checkpoint=500, measurements_planned=4, measurements_done=1)
    assert eval_progress.expected_run_fraction([one]) == 1.0     # too few rows to trust
    staged = run([result(step, episodes=20, perfect=19) for step in range(1, 6)], complete=False,
                 episodes_per_checkpoint=100, screen_episodes=20,
                 stages={'full': {'planned': 0, 'done': 0},
                         'screen': {'planned': 5, 'done': 5},
                         'confirm': {'planned': 2, 'done': 0}})
    # A screen and a full-length row are different targets and the payload does not say which a row
    # was, so there is no honest per-row target: 1.0, which is conservative.
    assert eval_progress.expected_run_fraction([staged]) == 1.0


def test_the_controllers_own_eta_beats_this_files_arithmetic():
    """An arm of a wave shares lanes, so its own remaining work over "one process" is remaining
    lane-time and nothing in its file says what share of the box it is getting: b43b read 37.9 h on a
    wave with ~13 h left. `eval_wave` measures the share as wall clock between completions and stamps
    the result, so the stamp wins wherever it exists."""
    rows = [result(step, episodes=250, perfect=200, seconds=125.0) for step in range(1, 5)]
    payload = run(rows, complete=False, episodes_per_checkpoint=500,
                  measurements_planned=8, measurements_done=4,
                  session_measurements=4, session_episodes=1000, session_seconds=500.0,
                  arm_eta_seconds=600.0, arm_eta_window=10,
                  wave_eta_seconds=1234.0, wave_lanes=4, wave_arms=4)
    state = eval_progress.summarize([payload])
    assert state['eta_seconds'] == 600.0, 'the arm ETA is the arm\'s own, not the wave\'s'
    assert state['eta_window'] == 10
    # The wave total is carried alongside rather than in place of it -- a different question, "when
    # is the box free", and the only one an arm's file cannot answer for itself.
    assert state['wave_eta_seconds'] == 1234.0 and state['wave_arms'] == 4
    # Before the second completion there is no interval to average, and the episode arithmetic that
    # the stamp exists to replace is still better than no ETA at all.
    del payload['arm_eta_seconds']
    fallback = eval_progress.summarize([payload])
    assert fallback['eta_seconds'] not in (None, 1234.0) and fallback['eta_window'] is None
    # ...and a single-policy run carries neither field.
    del payload['wave_eta_seconds']
    assert eval_progress.summarize([payload])['wave_eta_seconds'] is None


def test_the_wave_total_is_named_on_the_chart_and_the_arm_eta_is_not_relabelled():
    """The label read `wave ETA` while the wave's number was replacing the arm's. Now the left
    column's ETA is always this arm's, and the wave's total is a line in the right column -- so
    neither number can be read as the other."""
    rows = [result(step, episodes=250, perfect=200, seconds=125.0) for step in range(1, 5)]
    payload = run(rows, complete=False, episodes_per_checkpoint=500,
                  measurements_planned=8, measurements_done=4,
                  session_measurements=4, session_episodes=1000, session_seconds=500.0,
                  arm_eta_seconds=600.0, arm_eta_window=10,
                  wave_eta_seconds=36000.0, wave_lanes=4, wave_arms=4)
    left, right = eval_progress.summary_columns('b43b-lowlr-b29a',
                                                eval_progress.summarize([payload]))
    assert 'ETA 10m' in left and 'wave ETA' not in left, left
    assert 'wave of 4: all done in 10h00m' in right, right
    # No wave, no line: a single-policy close-out says nothing about waves.
    del payload['wave_eta_seconds']
    _, alone = eval_progress.summary_columns('arm', eval_progress.summarize([payload]))
    assert 'wave of' not in alone, alone


def test_summarize_falls_back_to_per_checkpoint_eta_for_older_files():
    # Files written before episodes_planned existed must still get an ETA.
    state = eval_progress.summarize([run(
        [result(1, seconds=100.0), result(2, seconds=100.0)],
        complete=False, checkpoints_requested=4,
        in_flight={'step': 3, 'round': 1, 'rounds_total': 10, 'perfect_so_far': 1,
                   'episodes_so_far': 10, 'running_percent': 10.0,
                   'per_round_perfect': [1], 'started_at': 1e12})])
    assert state['eta_seconds'] == 200.0
    assert state['unit'] == 'checkpoints'


def test_summarize_paces_the_eta_on_this_session_not_resumed_rows():
    # The b10b case: 300 rows on file at the old slow pace, 3 measured this session at the new
    # one. Averaging both put the ETA out by nearly 3x. Session pace here is 20s/100 episodes =
    # 0.2s each, so 1000 remaining episodes is 200s — not the 5s/episode the old rows imply.
    state = eval_progress.summarize([run(
        [result(s, seconds=500.0) for s in range(300)],
        complete=False, screen_episodes=None,
        measurements_planned=310, measurements_done=303,
        session_measurements=3, session_episodes=300, session_seconds=60.0,
        episodes_planned=31000, episodes_done=30000,
        in_flight={'step': 999, 'round': 1, 'rounds_total': 10, 'perfect_so_far': 1,
                   'episodes_so_far': 10, 'running_percent': 10.0,
                   'per_round_perfect': [1], 'started_at': 1e12})])
    assert abs(state['eta_seconds'] - 200.0) < 0.001, state['eta_seconds']
    assert abs(state['mean_seconds'] - 20.0) < 0.001, 'pace must be this session\'s too'


def test_summarize_divides_remaining_work_across_live_processes():
    flight = {'step': 9, 'round': 1, 'rounds_total': 10, 'perfect_so_far': 1,
              'episodes_so_far': 10, 'running_percent': 10.0,
              'per_round_perfect': [1], 'started_at': 1e12}
    one = eval_progress.summarize([run([result(1, seconds=100.0)], complete=False,
                                       checkpoints_requested=5, in_flight=flight)])
    two = eval_progress.summarize([run([result(1, seconds=100.0)], complete=False,
                                       checkpoints_requested=5, in_flight=flight),
                                   run([result(2, seconds=100.0)], complete=False,
                                       checkpoints_requested=0, in_flight=flight)])
    assert two['eta_seconds'] < one['eta_seconds']


# ------------------------------------------------------------ flight_workers

def flight(round_index=3, episodes_so_far=50, **extra):
    payload = {'step': 7, 'round': round_index, 'rounds_total': 8, 'perfect_so_far': 30,
               'episodes_so_far': episodes_so_far, 'running_percent': 60.0,
               'per_round_perfect': [7] * round_index, 'started_at': 1e12}
    payload.update(extra)
    return payload


def test_flight_workers_reads_the_static_count_from_the_run():
    # The live b11a case: round 3 of a top-up, 20 screen episodes already on file plus 30 from
    # this pass. Inferring gives 50 // 3 = 16 and changes every round; the run says 10.
    assert eval_progress.flight_workers(run([], num_workers=10), flight()) == 10


def test_flight_workers_is_constant_across_a_topped_up_checkpoints_rounds():
    # The actual symptom. Without the recorded count this walks 30, 20, 16, 15, 14, 13, 12, 12.
    counts = {eval_progress.flight_workers(
        run([], num_workers=10), flight(round_index=r, episodes_so_far=20 + 10 * r))
        for r in range(1, 9)}
    assert counts == {10}


def test_flight_workers_prefers_the_pass_local_count_when_deriving():
    # No num_workers, but episodes_this_pass shares a denominator with `round`, so 30 // 3 = 10
    # even though the checkpoint's cumulative sample is 50.
    assert eval_progress.flight_workers(
        run([]), flight(episodes_this_pass=30)) == 10


def test_flight_workers_still_derives_something_for_the_oldest_files():
    # Neither field present. Exact for a fresh checkpoint, which is all those files ever had.
    assert eval_progress.flight_workers(run([]), flight(round_index=3,
                                                        episodes_so_far=30)) == 10


def test_flight_workers_never_returns_zero():
    # Round 1 of a fresh checkpoint writes progress before any episode lands in some orderings;
    # a zero here would divide by zero in the running-rate line.
    assert eval_progress.flight_workers(run([]), flight(round_index=1,
                                                        episodes_so_far=0)) == 1


# ------------------------------------------------------------- stage reporting

def staged(full=(48, 48), screen=(181, 0), confirm=(100, 0), **extra):
    """A run payload carrying the three-stage counts, as (planned, done) pairs."""
    return run([], complete=False,
               screen_episodes=20, confirm_count=confirm[0],
               stages={'full': {'planned': full[0], 'done': full[1]},
                       'screen': {'planned': screen[0], 'done': screen[1]},
                       'confirm': {'planned': confirm[0], 'done': confirm[1]}},
               **extra)


def test_stage_summary_is_none_for_a_flat_run():
    # Every protocol before screening. The chart must fall back to its old single-bar form.
    assert eval_progress.stage_summary([run([])]) is None


def test_stage_summary_reports_the_first_unfinished_stage():
    s = eval_progress.stage_summary([staged(full=(48, 20))])
    assert s['current'] == 'full'
    s = eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 90))])
    assert s['current'] == 'screen'
    s = eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 181), confirm=(100, 3))])
    assert s['current'] == 'confirm'


def test_stage_summary_says_done_when_every_stage_is_finished():
    s = eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 181), confirm=(100, 100))])
    assert s['current'] == 'done'


def test_stage_summary_skips_a_stage_with_nothing_planned():
    # An arm with no 100% graph points has an empty full tier and should read as screening from the
    # start, not as "stage 1, 0 of 0" forever.
    s = eval_progress.stage_summary([staged(full=(0, 0), screen=(87, 4))])
    assert s['current'] == 'screen'


def test_stage_summary_sums_across_parallel_processes():
    two = [staged(full=(20, 20), screen=(50, 10)), staged(full=(30, 30), screen=(60, 5))]
    s = eval_progress.stage_summary(two)
    assert s['full'] == {'planned': 50, 'done': 50}
    assert s['screen'] == {'planned': 110, 'done': 15}


def test_stage_lines_hide_the_percentage_once_a_stage_is_finished():
    # The specific request: a finished stage's progress percent is noise once the next is running.
    lines = eval_progress.stage_lines(
        eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 90))]))
    full_line = [l for l in lines if l.strip().startswith('full')][0]
    screen_line = [l for l in lines if l.strip().startswith('screen')][0]
    assert 'done' in full_line and '%' not in full_line
    assert '%' in screen_line and '#' in screen_line, screen_line


def test_stage_lines_report_how_much_the_screen_cut():
    # Answers "what percentage was screened out": 181 screened, 100 promoted, 45% left at 20. The
    # cut *count* came off the line on 2026-08-19 -- at 78 characters this was the widest line the
    # left column produced and it overlapped the right one, and 81 is `181 - 100`, both of which are
    # still on the line.
    lines = eval_progress.stage_lines(
        eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 181), confirm=(100, 5))]))
    screen_line = [l for l in lines if l.strip().startswith('screen')][0]
    assert '100 promoted' in screen_line and '45% cut' in screen_line, screen_line
    assert len(screen_line) <= 54, 'too wide for the left column: {0}'.format(len(screen_line))


def test_stage_lines_mark_a_future_stage_pending_not_zero_percent():
    lines = eval_progress.stage_lines(
        eval_progress.stage_summary([staged(full=(48, 20))]))
    confirm_line = [l for l in lines if l.strip().startswith('confirm')][0]
    assert 'pending' in confirm_line, confirm_line


def test_stage_lines_name_the_current_stage_and_its_position():
    lines = eval_progress.stage_lines(
        eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 181), confirm=(100, 5))]))
    assert lines[0].strip().startswith('stage 3 of 3'), lines[0]


def test_stage_lines_are_empty_for_a_flat_run():
    assert eval_progress.stage_lines(None) == []


def test_text_summary_includes_the_stage_block():
    state = eval_progress.summarize([staged(full=(48, 48), screen=(181, 90))])
    text = eval_progress.text_summary('arm', state)
    assert 'stage 2 of 3' in text and 'screening the rest' in text, text


def test_top_five_excludes_lucky_screens_like_the_best_line_does():
    """The list and the `best` line above it must agree.

    Caught by eyeballing a rendered chart: an unfiltered top 5 showed five 20-episode screens at
    100.0% sitting directly under a `best` of 95%, because across a few hundred screens several
    land on 20/20 by luck. Both now rank over full-length rows only.
    """
    rows = ([result(1000 + i, episodes=20, perfect=20) for i in range(8)]
            + [result(9000, episodes=100, perfect=95)])
    state = eval_progress.summarize([run(rows, checkpoints_requested=9)])
    # Scoped to the ranking block since 2026-08-19, not the whole summary: the metrics column
    # legitimately prints '100.0%' -- the drift line pools the eight screens and the early half of
    # this fixture really is 20/20. Asserting over the whole block made this test fail for a
    # sentence that was correct, which is a test that has stopped describing its own subject.
    text = '\n'.join(eval_progress.ranking_lines(state))
    assert state['best']['step'] == 9000
    assert '100.0%' not in text, text
    assert 'full-length rows only' in text


def test_top_five_has_no_caveat_when_every_row_is_full_length():
    rows = [result(1000 + i, episodes=100, perfect=70 + i) for i in range(6)]
    state = eval_progress.summarize([run(rows, checkpoints_requested=6)])
    assert 'full-length rows only' not in eval_progress.text_summary('arm', state)


def test_top_five_does_not_claim_full_length_when_the_gate_abandoned_every_row():
    """The caveat must not promise a depth no row has.

    Batch 20's close-out is the case: at gate 95 every row in all four arms was abandoned, the
    deepest at 58 of 100 episodes, and the summary still printed "(full-length rows only)" over
    it — which reads as a 100-episode measurement and is how an 89.7% on 58 episodes gets quoted
    as if it were confirmed.
    """
    rows = ([result(1000 + i, episodes=20, perfect=20) for i in range(8)]
            + [result(9000, episodes=58, perfect=52)])
    state = eval_progress.summarize(
        [run(rows, checkpoints_requested=9, episodes_per_checkpoint=100)])
    text = eval_progress.text_summary('arm', state)
    assert 'none full' in text and 'full-length rows only' not in text, text
    # Wording shortened 2026-08-19 -- the long form crossed into the right column on the chart -- so
    # the depth is asserted as the pair it now prints rather than as prose.
    assert 'deepest 58/100 ep' in text, text


def test_top_five_flags_shallow_rows_even_when_none_were_filtered_out():
    """The worst case, and the one the filtering test misses.

    b20b's rows were 20-36 episodes: every one short of the 100 target, but all within half the
    deepest, so nothing was excluded and the caveat never fired — an 83.3% on 36 episodes
    printed with no qualifier at all. Absence of a note has to mean full length.
    """
    rows = [result(1000 + i, episodes=20 + 2 * i, perfect=15 + i) for i in range(9)]
    state = eval_progress.summarize(
        [run(rows, checkpoints_requested=9, episodes_per_checkpoint=100)])
    text = eval_progress.text_summary('arm', state)
    assert len(eval_progress.deep_rows(rows)) == len(rows), 'fixture must filter nothing'
    assert 'none full' in text, text
    assert 'deepest 36/100 ep' in text, text


def test_top_five_still_says_full_length_when_the_deep_rows_reached_the_target():
    """The mixed-depth case the caveat was written for still reports it as full length."""
    rows = ([result(1000 + i, episodes=20, perfect=20) for i in range(8)]
            + [result(9000, episodes=100, perfect=95)])
    state = eval_progress.summarize(
        [run(rows, checkpoints_requested=9, episodes_per_checkpoint=100)])
    assert 'full-length rows only' in eval_progress.text_summary('arm', state)


def test_top_five_caveat_falls_back_when_the_target_was_never_recorded():
    """Files predating episodes_per_checkpoint have no target, so the deepest row is the target
    and the old wording is still the honest one."""
    rows = ([result(1000 + i, episodes=20, perfect=20) for i in range(8)]
            + [result(9000, episodes=100, perfect=95)])
    state = eval_progress.summarize([run(rows, checkpoints_requested=9)])
    assert state['target_episodes'] is None
    assert 'full-length rows only' in eval_progress.text_summary('arm', state)


def test_deep_rows_is_every_row_under_one_uniform_episode_count():
    rows = [result(1, episodes=100), result(2, episodes=100)]
    assert eval_progress.deep_rows(rows) == rows
    assert eval_progress.deep_rows([]) == []


def test_pooled_prefers_the_writers_equal_effort_figure():
    """Pooling the rows is not an arm rate once depths differ, and the chart used to show it.

    Here the deep row is a perfect 100/100 and the shallow one 5/20. Pooling gives 105/120 = 87.5%,
    which reads high purely because the deep rows are the arm's best by construction. The writer's
    equal-effort figure (every checkpoint truncated to its first 20) is 62.5%.
    """
    rows = [result(1, episodes=100, perfect=100), result(2, episodes=20, perfect=5)]
    state = eval_progress.summarize([run(rows, screen_episodes=20, pooled_equal_effort=62.5)])
    assert state['pooled'] == 62.5
    assert state['pooled_is_equal_effort'] is True
    assert 'equal effort' in eval_progress.text_summary('arm', state)


def test_pooled_falls_back_to_row_pooling_for_a_flat_run():
    # With one uniform depth, pooling the rows already gives equal effort, and every arm measured
    # before screening existed reports it that way.
    rows = [result(1, episodes=100, perfect=70), result(2, episodes=100, perfect=80)]
    state = eval_progress.summarize([run(rows, checkpoints_requested=2)])
    assert abs(state['pooled'] - 75.0) < 0.001
    assert state['pooled_is_equal_effort'] is False
    assert 'equal effort' not in eval_progress.text_summary('arm', state)


# ------------------------------------------------- the gate line on the perfect-% charts

def test_summarize_exposes_the_gate_from_the_payload():
    # Read from the file, not the environment: a chart rendered later, or by `report` over several
    # arms at once, must show the gate each file was *measured* under. Batches 11 and 13 have none,
    # 14-15 have 90, 16 on have 95, and a chart is often drawn long after the run.
    rows = [result(1, perfect=70)]
    assert eval_progress.summarize([run(rows, min_achievable=95.0)])['min_achievable'] == 95.0
    assert eval_progress.summarize([run(rows, min_achievable=90.0)])['min_achievable'] == 90.0


def test_summarize_reports_no_gate_for_an_ungated_file():
    # `min_achievable` is written as None when abandonment is off, and absent entirely in files
    # that predate it. Both must read as "no gate" so no line is drawn.
    rows = [result(1, perfect=70)]
    assert eval_progress.summarize([run(rows)])['min_achievable'] is None
    assert eval_progress.summarize([run(rows, min_achievable=None)])['min_achievable'] is None


def test_draw_threshold_marks_the_gate_and_labels_it_with_the_measured_value():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for gate in (95.0, 90.0):
        figure, axis = plt.subplots()
        eval_progress.draw_threshold(axis, {'min_achievable': gate})
        lines = axis.get_lines()
        assert len(lines) == 1, 'expected exactly one gate line'
        # A horizontal line at the gate, so both ends sit at the same y and that y is the gate.
        assert lines[0].get_ydata()[0] == gate
        assert lines[0].get_ydata()[-1] == gate
        assert lines[0].get_linestyle() != '-', 'the gate line must be dashed, not solid'
        assert '{0:.0f}%'.format(gate) in lines[0].get_label()
        plt.close(figure)


def test_draw_threshold_draws_nothing_without_a_gate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for state in ({}, {'min_achievable': None}, {'min_achievable': 0}):
        figure, axis = plt.subplots()
        eval_progress.draw_threshold(axis, state)
        assert axis.get_lines() == [], state
        plt.close(figure)


def test_the_in_flight_line_reports_the_worker_count():
    # The chart's worker axis came out on 2026-08-07, so this line and the x-axis label are the only
    # places the count survives — and it is what turns "round 3/10" into an episode count.
    flight = {'step': 5000, 'round': 3, 'rounds_total': 10, 'running_percent': 80.0,
              'started_at': 1e12, 'per_round_perfect': [8, 8, 8]}
    payload = run([result(1)], complete=False, num_workers=10)
    state = eval_progress.summarize([payload])
    state['active'] = [(payload, flight)]
    assert 'round 3/10 x 10w' in eval_progress.text_summary('arm', state)


# ------------------------------------------------- the in-flight running-rate annotation

def render_flight(per_round, workers, rounds_total=10):
    """Renders a chart with one in-flight checkpoint and returns the top axes' annotations.

    render() closes its own figure, so the axes are captured by patching plt.subplot-creation the
    way test_progress_chart does — less invasive than making production hand a figure back.
    """
    import os
    import tempfile

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    live = run([result(1000, perfect=90)], complete=False, num_workers=workers, suffix='_p1',
               in_flight=flight(round_index=len(per_round), per_round_perfect=list(per_round),
                                rounds_total=rounds_total),
               measurements_planned=20, measurements_done=1)
    state = eval_progress.summarize([live])
    captured = {}
    real_figure = plt.figure

    def capture(*args, **kwargs):
        fig = real_figure(*args, **kwargs)
        captured['figure'] = fig
        return fig

    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    plt.figure = capture
    try:
        eval_progress.render('arm', state, path)
    finally:
        plt.figure = real_figure
        for leftover in (path, path + '.partial.png'):
            if os.path.exists(leftover):
                os.remove(leftover)
    figure = captured['figure']
    top = figure.axes[0]
    texts = [t.get_text() for t in top.texts]
    plt.close(figure)
    return texts


def test_the_in_flight_panel_writes_out_the_current_running_rate():
    """Reading the rate off a 2.1in panel with 20pp gridlines is the thing this fixes."""
    # 3 rounds x 4 workers = 12 episodes, 9 perfect -> 75%
    texts = render_flight([4, 3, 2], workers=4)
    assert '75%' in texts, texts


def test_the_written_rate_is_the_running_total_not_the_last_round():
    # Last round alone is 1/4 = 25%; the running total is 9/12 = 75%. The running one is the useful
    # number and the one the y axis plots.
    texts = render_flight([4, 4, 1], workers=4)
    assert '75%' in texts, texts
    assert '25%' not in texts, texts


def test_the_rate_tracks_the_worker_count():
    # Same per-round counts, different workers: 9 perfect of 3x10=30 is 30%, not 75%.
    texts = render_flight([4, 3, 2], workers=10)
    assert '30%' in texts, texts


# ------------------------------------------------- a layout that does not move

def render_figure(active_flight):
    """Renders with or without in-flight work and returns (figsize, n_axes, top-axis texts)."""
    import os
    import tempfile

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if active_flight:
        live = run([result(1000, perfect=90)], complete=False, num_workers=4, suffix='_p1',
                   in_flight=flight(round_index=3, per_round_perfect=[4, 3, 2], rounds_total=10),
                   measurements_planned=20, measurements_done=1)
    else:
        live = run([result(1000, perfect=90)], complete=True, num_workers=4, suffix='_p1',
                   measurements_planned=20, measurements_done=20)
    state = eval_progress.summarize([live])

    captured = {}
    real_figure = plt.figure

    def capture(*args, **kwargs):
        fig = real_figure(*args, **kwargs)
        captured['figure'] = fig
        return fig

    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    plt.figure = capture
    try:
        eval_progress.render('arm', state, path)
        size = os.path.getsize(path)
    finally:
        plt.figure = real_figure
        for leftover in (path, path + '.partial.png'):
            if os.path.exists(leftover):
                os.remove(leftover)
    figure = captured['figure']
    out = (tuple(figure.get_size_inches()), len(figure.axes),
           [t.get_text() for t in figure.axes[0].texts])
    plt.close(figure)
    return out


def test_the_figure_is_the_same_size_whether_or_not_anything_is_in_flight():
    """The window jumped several times a minute because the panel count changed."""
    with_flight = render_figure(True)
    without = render_figure(False)
    assert with_flight[0] == without[0], (with_flight[0], without[0])
    assert with_flight[1] == without[1], 'panel count changed: {0} vs {1}'.format(
        with_flight[1], without[1])


def test_the_in_flight_panel_is_kept_and_labelled_when_idle():
    _, axes, texts = render_figure(False)
    assert axes == 3, 'the in-flight panel should still be there, empty'
    assert any('no checkpoint in flight' in t for t in texts), texts


def test_the_idle_panel_keeps_the_same_y_scale_so_the_eye_does_not_have_to_readjust():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import tempfile

    limits = {}
    for label, has in (('flight', True), ('idle', False)):
        if has:
            live = run([result(1000, perfect=90)], complete=False, num_workers=4, suffix='_p1',
                       in_flight=flight(round_index=3, per_round_perfect=[4, 3, 2],
                                        rounds_total=10),
                       measurements_planned=20, measurements_done=1)
        else:
            live = run([result(1000, perfect=90)], complete=True, num_workers=4, suffix='_p1',
                       measurements_planned=20, measurements_done=20)
        state = eval_progress.summarize([live])
        captured = {}
        real_figure = plt.figure

        def capture(*args, **kwargs):
            fig = real_figure(*args, **kwargs)
            captured['figure'] = fig
            return fig

        handle, path = tempfile.mkstemp(suffix='.png')
        os.close(handle)
        plt.figure = capture
        try:
            eval_progress.render('arm', state, path)
        finally:
            plt.figure = real_figure
            for leftover in (path, path + '.partial.png'):
                if os.path.exists(leftover):
                    os.remove(leftover)
        limits[label] = captured['figure'].axes[0].get_ylim()
        plt.close(captured['figure'])
    # 50, not 0, since 2026-08-19: the panel's whole job is telling 93% from 98%, and at 0-100 that
    # difference was 5% of the axis. The floor is fixed rather than data-driven precisely so these
    # two are equal -- an adaptive floor would move the moment a checkpoint ran badly.
    # The literal 50 on purpose: written as `eval_progress.PERFECT_AXIS_FLOOR` this asserted the
    # constant equals itself and passed with the floor put back to 0.
    assert limits['flight'] == limits['idle'] == (50, 100), limits


# ------------------------------------------------- solid means full length, nothing less

def marker_split(rows, target_episodes=100):
    """Renders panel 2 and returns (solid_count, hollow_count) from its two scatter series."""
    import os
    import tempfile

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    live = run(rows, complete=True, num_workers=4, suffix='_p1',
               episodes_per_checkpoint=target_episodes,
               measurements_planned=len(rows), measurements_done=len(rows))
    state = eval_progress.summarize([live])
    captured = {}
    real_figure = plt.figure

    def capture(*args, **kwargs):
        fig = real_figure(*args, **kwargs)
        captured['figure'] = fig
        return fig

    handle, path = tempfile.mkstemp(suffix='.png')
    os.close(handle)
    plt.figure = capture
    try:
        eval_progress.render('arm', state, path)
    finally:
        plt.figure = real_figure
        for leftover in (path, path + '.partial.png'):
            if os.path.exists(leftover):
                os.remove(leftover)
    figure = captured['figure']
    middle = figure.axes[1]
    solid = hollow = 0
    for coll in middle.collections:
        n = len(coll.get_offsets())
        size = coll.get_sizes()[0] if len(coll.get_sizes()) else 0
        # Matched on marker size, which is what distinguishes the three scatters: 9 for the
        # partial series, 22 for the full one, and 90 for the single hollow red ring marking the
        # best checkpoint. An earlier version of this helper counted that ring as a partial point.
        if size == 9:
            hollow += n
        elif size == 22:
            solid += n
    plt.close(figure)
    return solid, hollow


def test_only_full_length_rows_are_solid():
    """The bug: the split was at half the deepest row, so a row abandoned at 60/100 drew solid."""
    rows = [result(1000, episodes=100, perfect=90),
            result(2000, episodes=60, perfect=50),    # abandoned past half of 100 - must be hollow
            result(3000, episodes=20, perfect=19)]
    solid, hollow = marker_split(rows)
    assert solid == 1, 'only the 100-episode row should be solid, got {0}'.format(solid)
    assert hollow == 2, 'the 60- and 20-episode rows should both be hollow, got {0}'.format(hollow)


def test_a_row_just_short_of_the_target_is_hollow():
    rows = [result(1000, episodes=100, perfect=95), result(2000, episodes=96, perfect=90)]
    solid, hollow = marker_split(rows)
    assert (solid, hollow) == (1, 1), (solid, hollow)


def test_every_row_at_the_target_is_solid_and_there_is_no_hollow_series():
    rows = [result(1000, episodes=100, perfect=95), result(2000, episodes=100, perfect=90)]
    solid, hollow = marker_split(rows)
    assert (solid, hollow) == (2, 0), (solid, hollow)


def test_rows_deeper_than_the_target_still_count_as_full():
    # A worker count that does not divide the request rounds episodes up; 108 of 100 is full.
    rows = [result(1000, episodes=108, perfect=95), result(2000, episodes=20, perfect=19)]
    solid, hollow = marker_split(rows)
    assert (solid, hollow) == (1, 1), (solid, hollow)


def test_without_a_recorded_target_the_deepest_row_defines_full():
    """Files predating episodes_per_checkpoint must still render sensibly."""
    rows = [result(1000, episodes=100, perfect=95), result(2000, episodes=40, perfect=30)]
    solid, hollow = marker_split(rows, target_episodes=None)
    assert (solid, hollow) == (1, 1), (solid, hollow)


def test_an_arm_where_nothing_reached_full_length_has_no_solid_points():
    """The real b16c/b16d case: under a 95 gate their deepest rows were 40 and 30 episodes.

    `deepest` is then 40, not 100, so any rule keyed on the deepest row would draw a 40-episode
    abandoned measurement as a confirmed one — on an arm whose headline number this file already
    marks `[truncated]`. The target has to come from the payload.
    """
    rows = [result(1000, episodes=40, perfect=34), result(2000, episodes=30, perfect=24),
            result(3000, episodes=20, perfect=15)]
    solid, hollow = marker_split(rows, target_episodes=100)
    assert solid == 0, 'nothing reached 100 episodes, so nothing should be solid'
    assert hollow == 3, hollow


# ------------------------------------------------------------- in-flight title

def test_format_step_k_groups_thousands():
    assert eval_progress.format_step_k(1320000) == '1,320k'
    assert eval_progress.format_step_k(777000) == '777k'


def test_in_flight_title_names_the_single_checkpoint():
    # The whole point of the change: the title must say *which* checkpoint, not just a count.
    active = [({'suffix': '_t'}, {'step': 1320000})]
    title = eval_progress.in_flight_title(active)
    assert '1 checkpoint @1,320k' in title, title
    assert '1 process' in title, title


def test_in_flight_title_lists_distinct_steps_sorted_for_several_processes():
    active = [({'suffix': '_a'}, {'step': 1320000}),
              ({'suffix': '_b'}, {'step': 450000})]
    title = eval_progress.in_flight_title(active)
    assert '@450k/1,320k' in title, title  # sorted ascending, joined with '/'
    assert '2 checkpoints' in title and '2 processes' in title, title


def test_in_flight_title_collapses_duplicate_steps():
    # A shard set: two processes on the same checkpoint -> the step appears once, not '900k/900k'.
    active = [({'suffix': '_r0'}, {'step': 900000}),
              ({'suffix': '_r1'}, {'step': 900000})]
    title = eval_progress.in_flight_title(active)
    assert '@900k' in title and '900k/900k' not in title, title


# --------------------------------------------------- load_runs: which files count as "this job"
# These write real result files and point RUNS_DIR at a temp dir, because the behaviour under test
# is the glob-and-filter in load_runs, not summarize()'s handling of run dicts.
import json
import os
import tempfile


def _write_result_file(directory, policy, suffix, results, mtime):
    path = os.path.join(directory, '{0}_checkpoint_evals{1}.json'.format(policy, suffix))
    with open(path, 'w') as handle:
        json.dump({'results': results, 'complete': True, 'updated_at': mtime}, handle)
    os.utime(path, (mtime, mtime))
    return path


def _with_temp_runs_dir(body):
    directory = tempfile.mkdtemp()
    real = eval_progress.RUNS_DIR
    eval_progress.RUNS_DIR = directory
    try:
        body(directory)
    finally:
        eval_progress.RUNS_DIR = real
        for name in os.listdir(directory):
            os.remove(os.path.join(directory, name))
        os.rmdir(directory)


def test_load_runs_suffix_scope_excludes_a_recent_other_eval_the_window_would_merge():
    # The HOF bug, reproduced: a close-out (many rows, no suffix) finished 44 min before an
    # _hof500 re-measurement started. Both are inside the 1h mtime window, so the unscoped view
    # merges them and the live chart reads ~1700 checkpoints. Passing the running job's own
    # suffix set drops the close-out and leaves only the nine rows it has actually measured.
    def body(directory):
        closeout = [result(s) for s in range(1000, 1000 + 1646)]
        hof = [result(s) for s in range(2000, 2009)]
        base = 1_000_000.0
        _write_result_file(directory, 'b24b', '', closeout, base)            # the close-out
        _write_result_file(directory, 'b24b', '_hof500', hof, base + 2653)   # 44 min later

        merged = eval_progress.load_runs('b24b', window=3600)
        assert sorted(r['suffix'] for r in merged) == ['(none)', '_hof500'], \
            'the mtime window should merge both, which is the misfire being fixed'

        scoped = eval_progress.load_runs('b24b', suffixes={'_hof500'})
        assert [r['suffix'] for r in scoped] == ['_hof500']
        assert sum(len(r['results']) for r in scoped) == 9, \
            'only the HOF file counts once scoped to its suffix'
    _with_temp_runs_dir(body)


def test_load_runs_suffix_scope_keeps_the_jobs_own_parallel_shards():
    # The scope must not throw the baby out: a job split across two suffixed workers passes both
    # of its own suffixes and both shards still load, even with an unrelated file on the arm.
    def body(directory):
        base = 1_000_000.0
        _write_result_file(directory, 'b24b', '_w0', [result(1)], base)
        _write_result_file(directory, 'b24b', '_w1', [result(2)], base)
        _write_result_file(directory, 'b24b', '', [result(9)], base)  # unrelated, must be dropped
        scoped = eval_progress.load_runs('b24b', suffixes={'_w0', '_w1'})
        assert sorted(r['suffix'] for r in scoped) == ['_w0', '_w1']
    _with_temp_runs_dir(body)


# ---------------------------------------------------- render resolution (HiDPI crispness)

def test_render_dpi_defaults_to_110_and_follows_the_env_knob():
    """chart_viewer only magnifies this PNG, so the source dpi is what decides crispness on a
    HiDPI panel. The default must stay 110 (the desktop's 1x display and the standalone window
    rely on it); SNEK_EVAL_CHART_DPI raises it, and a higher dpi means more source pixels at the
    same figsize -- which is how eval_checkpoints makes the laptop charts crisp without resizing
    the window."""
    from PIL import Image
    state = eval_progress.summarize([run([result(1000), result(2000)])])
    directory = tempfile.mkdtemp(prefix='snek-dpi-test-')
    saved = os.environ.get('SNEK_EVAL_CHART_DPI')
    try:
        os.environ.pop('SNEK_EVAL_CHART_DPI', None)
        low = os.path.join(directory, 'low.png')
        eval_progress.render('b30e-chase10fc200x100x100seed1', state, low)
        assert round(Image.open(low).info['dpi'][0]) == 110

        os.environ['SNEK_EVAL_CHART_DPI'] = '220'
        high = os.path.join(directory, 'high.png')
        eval_progress.render('b30e-chase10fc200x100x100seed1', state, high)
        assert round(Image.open(high).info['dpi'][0]) == 220
        # Same figsize, double the dpi -> ~double the pixels, i.e. a crisper source, not a bigger
        # chart in inches. This is the property that keeps chart_viewer's window size unchanged.
        assert Image.open(high).size[0] > Image.open(low).size[0]
    finally:
        if saved is None:
            os.environ.pop('SNEK_EVAL_CHART_DPI', None)
        else:
            os.environ['SNEK_EVAL_CHART_DPI'] = saved
        shutil.rmtree(directory, ignore_errors=True)


# ------------------------------------- the bands, the plateau and the drift (2026-08-19)

def test_the_bands_are_half_open_ranges_not_exact_integers():
    # At 100 episodes "99-100%" can only be 99/100, which is why exact tests survived review the
    # first time. The HOF re-measure runs 500 episodes: 498/500 is 99.6% and belongs in the same
    # band, and 500/500 is the only thing that may count as 100%.
    rows = [result(1, episodes=500, perfect=500), result(2, episodes=500, perfect=498),
            result(3, episodes=500, perfect=492), result(4, episodes=500, perfect=400)]
    assert eval_progress.band_counts(rows) == [
        ('= 100%', 1), ('99-100%', 1), ('98-99%', 1), ('below 98%', 1)]


def test_the_bands_partition_the_rows():
    rows = [result(i, perfect=p) for i, p in enumerate([100, 100, 99, 98, 97, 40])]
    assert sum(count for _, count in eval_progress.band_counts(rows)) == len(rows)


def test_the_longest_run_is_broken_by_an_abandoned_row():
    # The point of the metric: five 98s in a row is a policy that holds, and the same five with an
    # abandoned checkpoint in the middle is two short stretches. Abandonment *is* evidence -- the
    # gate only stops a row it has proved cannot reach the close-out threshold.
    rows = [result(step, perfect=99) for step in (1000, 2000, 3000, 4000, 5000)]
    assert eval_progress.longest_band_run(rows, 100) == (5, 1000, 5000)
    # Abandoned *and* short, which is what an abandoned row always is -- so a depth test on its own
    # would skip it and bridge the two stretches it separates.
    rows[2] = dict(rows[2], episodes=44, perfect_games=43, perfect_percent=97.7, abandoned=True)
    assert eval_progress.longest_band_run(rows, 100)[0] == 2


def test_the_longest_run_breaks_on_an_abandoned_row_that_reads_above_the_band():
    # An abandoned row's rate is always strictly below the gate it was abandoned under, so at this
    # project's gates (<= 98) the percentage test alone would already catch it and the abandonment
    # test looks redundant. It is not: the HOF path takes EVAL_MIN_ACHIEVABLE as an argument, and at
    # a gate of 99 a 500-episode row cut at 464/470 reads 98.7% -- above the band, and still a proof
    # that the checkpoint is not a 99% one.
    rows = [result(1000, episodes=500, perfect=500),
            dict(result(2000, episodes=470, perfect=464), abandoned=True),
            result(3000, episodes=500, perfect=500)]
    assert eval_progress.longest_band_run(rows, 500)[0] == 1


def test_the_longest_run_ignores_the_order_rows_were_measured_in():
    # The close-out measures stage 1 before stage 2, so the file is not in step order.
    rows = [result(3000, perfect=99), result(1000, perfect=99), result(2000, perfect=99)]
    assert eval_progress.longest_band_run(rows, 100) == (3, 1000, 3000)


def test_the_longest_run_counts_only_consecutive_measured_rows():
    # Unselected steps are not gaps in the policy -- nobody looked at them. A stretch of measured
    # rows at 1000 and 9000 with nothing measured between is still a stretch of two.
    rows = [result(1000, perfect=99), result(9000, perfect=99)]
    assert eval_progress.longest_band_run(rows, 100) == (2, 1000, 9000)


def test_the_longest_run_neither_counts_nor_breaks_on_a_screen():
    # A 20-episode screen is not evidence: 19/20 fits a 98% policy and a 90% one equally. Counting
    # it would invent a plateau out of one lucky screen; breaking on it would erase a real one.
    lucky = [result(1000, perfect=99), result(2000, episodes=20, perfect=20),
             result(3000, perfect=99)]
    assert eval_progress.longest_band_run(lucky, 100) == (2, 1000, 3000)
    unlucky = [result(1000, perfect=99), result(2000, episodes=20, perfect=17),
               result(3000, perfect=99)]
    assert eval_progress.longest_band_run(unlucky, 100) == (2, 1000, 3000)


def test_the_drift_split_weights_a_deep_row_over_a_screen():
    # A 20-episode screen at 100% in the late half must not outweigh 100 episodes at 90%: pooling
    # is by episode, not by row, or one lucky screen reverses the sign of the trend.
    rows = [result(1000, episodes=100, perfect=95), result(2000, episodes=100, perfect=95),
            result(3000, episodes=100, perfect=90), result(4000, episodes=20, perfect=20)]
    early, late = eval_progress.half_split_pooled(rows)
    assert early == 95.0
    assert round(late, 1) == round(100.0 * 110 / 120, 1)
    assert late < early, 'the deep row has to dominate the screen'


def test_the_drift_split_declines_on_too_few_rows():
    assert eval_progress.half_split_pooled([result(1), result(2)]) == (None, None)


def test_the_metrics_block_counts_the_bands_over_full_length_rows_only():
    # A 20-episode screen going 20/20 is not a 100% checkpoint, and the bands are what a promotion
    # decision reads. Same rule as `best_of`, applied to the distribution.
    rows = [result(1000, episodes=100, perfect=98), result(2000, episodes=20, perfect=20)]
    state = eval_progress.summarize([run(rows, episodes_per_checkpoint=100)])
    text = '\n'.join(eval_progress.metrics_lines(state))
    assert 'full-length (100 ep)' in text
    assert '= 100%         0' in text, text
    assert '98-99%         1' in text, text


def test_the_metrics_block_says_so_when_no_row_reached_full_length():
    # At a 97 gate a weak arm has no full-length row at all. Four zeroes would be a lie about the
    # arm; the fallback ranks the deep rows and labels the depth.
    rows = [dict(result(1000, episodes=40, perfect=30), abandoned=True),
            dict(result(2000, episodes=40, perfect=28), abandoned=True)]
    state = eval_progress.summarize([run(rows, episodes_per_checkpoint=100)])
    text = '\n'.join(eval_progress.metrics_lines(state))
    assert 'deep rows' in text and 'deepest 40 of 100 ep, none full length' in text, text
    assert '2 rows measured = 0 full + 2 partial' in text, text


def test_the_metrics_block_reports_no_hof_candidate_rather_than_an_empty_range():
    state = eval_progress.summarize([run([result(1000, perfect=70)], episodes_per_checkpoint=100)])
    text = '\n'.join(eval_progress.metrics_lines(state))
    assert 'no hall-of-fame candidate' in text, text


def test_the_gate_savings_line_prices_the_skipped_episodes_in_wall_clock():
    # 4000 episodes never run at the measured 0.5s each is a little over half an hour, and that is
    # the form the number is useful in -- "episodes_saved: 4000" is not a decision.
    rows = [dict(result(1000, episodes=40, perfect=30, seconds=20.0), abandoned=True)]
    state = eval_progress.summarize([run(rows, episodes_per_checkpoint=100,
                                         episodes_saved=4000,
                                         session_measurements=1, session_episodes=40,
                                         session_seconds=20.0)])
    text = '\n'.join(eval_progress.metrics_lines(state))
    assert 'gate cut 1 rows, saved 4,000 ep ~33m' in text, text


def test_the_drift_line_does_not_print_a_negative_zero():
    rows = [result(1000, episodes=1000, perfect=940), result(2000, episodes=1000, perfect=940),
            result(3000, episodes=1000, perfect=940), result(4000, episodes=1000, perfect=939)]
    text = '\n'.join(eval_progress.metrics_lines(
        eval_progress.summarize([run(rows, episodes_per_checkpoint=1000)])))
    assert '-0.0 pp' not in text, text


# ------------------------------------- two columns, and no in-flight text (2026-08-19)

def test_the_chart_text_is_two_columns_split_by_kind():
    state = eval_progress.summarize([run([result(1000, perfect=98)],
                                         episodes_per_checkpoint=100,
                                         measurements_planned=1, measurements_done=1)])
    left, right = eval_progress.summary_columns('arm', state)
    assert 'top 5' in left and 'best' in left
    assert 'perfect % of' in right and '= 100%' in right
    assert 'top 5' not in right and 'perfect % of' not in left


def test_the_chart_text_drops_the_in_flight_block_the_top_panel_already_draws():
    live = run([result(1000, perfect=90)], complete=False, num_workers=4,
               in_flight=flight(round_index=3, per_round_perfect=[4, 3, 2], rounds_total=10),
               measurements_planned=20, measurements_done=1)
    state = eval_progress.summarize([live])
    left, right = eval_progress.summary_columns('arm', state)
    assert 'in flight' not in left + right, (left, right)
    # ...and `report`, which has no panel, still prints it.
    assert 'in flight' in eval_progress.text_summary('arm', state)


def test_a_stale_process_is_still_reported_on_the_chart():
    # Staleness is measured against the wall clock, so the fixture has to use it -- the far-future
    # `updated_at` the other fixtures pass is what makes a run read as fresh.
    stale = run([result(1000, perfect=90)], complete=False, updated_at=time.time() - 600)
    state = eval_progress.summarize([stale], stale_after=180)
    _, right = eval_progress.summary_columns('arm', state)
    assert 'STALE' in right, right


def test_the_terminal_summary_carries_the_metrics_too():
    state = eval_progress.summarize([run([result(1000, perfect=98)],
                                         episodes_per_checkpoint=100)])
    assert 'perfect % of' in eval_progress.text_summary('arm', state)


def test_points_below_the_axis_floor_are_counted_in_the_panel():
    """The floor hides data, so the count is drawn where the data would have been."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    eval_progress.note_clipped(axis, [99.0, 40.0, 12.0, None], 'rows')
    assert [t.get_text() for t in axis.texts] == ['2 rows below 50%']
    plt.close(figure)

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    eval_progress.note_clipped(axis, [99.0, 51.0], 'rows')
    assert list(axis.texts) == [], 'nothing to say when nothing is hidden'
    plt.close(figure)


def render_panel_heights(state, policy='arm'):
    """The three panels' heights in inches, top-down, off a real render."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    captured = {}
    real_figure = plt.figure

    def capture(*args, **kwargs):
        fig = real_figure(*args, **kwargs)
        captured['figure'] = fig
        return fig

    directory = tempfile.mkdtemp()
    plt.figure = capture
    try:
        eval_progress.render(policy, state, os.path.join(directory, 'c.png'))
    finally:
        plt.figure = real_figure
        shutil.rmtree(directory, ignore_errors=True)
    figure = captured['figure']
    size = figure.get_size_inches()
    heights = [axis.get_position().height * size[1] for axis in figure.axes]
    plt.close(figure)
    return heights, (round(float(size[0]), 3), round(float(size[1]), 3))


def test_the_charts_take_back_whatever_the_summary_block_does_not_need():
    """Was `the two charts are a fifth shorter than the text panel is tall`, which pinned the
    gridspec's 1.68/1.68/2.75 as the *final* layout. Those ratios are now the **capacity**: the
    summary block is fitted to its own text and hands the difference back to the two charts, so a
    flat run renders charts taller than their share and a text panel shorter than its own. What still
    has to hold is that the charts stay equal, that they only ever grow, and that the frame does not
    move -- the constant-size property the viewer's window depends on.
    """
    flat = eval_progress.summarize([run([result(1000, perfect=90)], complete=True, num_workers=4,
                                        measurements_planned=1, measurements_done=1)])
    heights, size = render_panel_heights(flat)
    assert size == (9.5, 8.56), size
    assert round(heights[0], 2) == round(heights[1], 2), heights
    # 2.10in each before the 20% cut of 2026-08-19, 1.73in as the gridspec now hands them out.
    share = 1.73
    assert heights[0] > share, 'the charts did not take the block\'s slack: {0}'.format(heights)
    # The block is fitted to its lines, so a flat run's is well under the worst-case capacity.
    lines = 8
    assert heights[2] < lines * eval_progress.SUMMARY_LINE_IN + 0.2, heights

    # ...and the busier the block, the more of it comes back off the charts.
    busy, _ = render_panel_heights(busy_state('confirm'))
    assert busy[2] > heights[2] + 0.5, (busy, heights)
    assert busy[0] < heights[0] - 0.2 and busy[0] >= share - 0.02, (busy, heights)


# ------------------------------------- the text columns must not collide (2026-08-19)

def render_text_extents(state, policy='b43a-lowlr-b29b'):
    """(left bbox, right bbox, bottom-panel height in inches), bboxes in axes fraction.

    Measured off the real renderer rather than computed from character counts: the columns are
    positioned in axes fraction and sized by font metrics, so the only honest test of "do these two
    blocks touch" is to draw them and ask.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    captured = {}
    real_figure = plt.figure

    def capture(*args, **kwargs):
        fig = real_figure(*args, **kwargs)
        captured['figure'] = fig
        return fig

    directory = tempfile.mkdtemp()
    plt.figure = capture
    try:
        eval_progress.render(policy, state, os.path.join(directory, 'c.png'))
    finally:
        plt.figure = real_figure
        shutil.rmtree(directory, ignore_errors=True)
    figure = captured['figure']
    bottom = figure.axes[2]
    renderer = figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer=renderer).transformed(bottom.transAxes.inverted())
             for t in bottom.texts]
    inches = bottom.get_position().height * figure.get_size_inches()[1]
    plt.close(figure)
    return boxes[0], boxes[1], inches


def busy_state(stage='full'):
    """A state with the widest lines both columns can produce.

    Eight-digit steps, five-digit measurement counts, a full top five and every band populated. The
    `stage` argument matters: at `'confirm'` the screen line carries its `done — N promoted, X% cut`
    detail, which was the widest line in the whole block and is invisible while screening is still
    running -- so a fixture that only ever tested stage 1 missed it, and it overlapped the right
    column at 8.5pt and 10pt alike.
    """
    rows = ([dict(result(10000000 - 1000 * i, episodes=100, perfect=100), min_score=95.0)
             for i in range(5)]
            + [dict(result(9000000 - 1000 * i, episodes=100, perfect=99 - i % 3), min_score=0.0)
               for i in range(20)]
            + [dict(result(8000000 - 1000 * i, episodes=44, perfect=42), abandoned=True)
               for i in range(10)])
    stages = {'order': ['full', 'screen', 'confirm'], 'current': stage,
              'full': {'planned': 12791, 'done': 12791 if stage != 'full' else 12000},
              'screen': {'planned': 10506, 'done': 10506 if stage == 'confirm' else 0},
              'confirm': {'planned': 2137, 'done': 48 if stage == 'confirm' else 0}}
    return eval_progress.summarize([run(
        rows, complete=False, num_workers=4, episodes_per_checkpoint=100, min_achievable=97.0,
        screen_episodes=20, confirm_count=100, episodes_saved=14330,
        session_measurements=35, session_episodes=3000, session_seconds=1500.0,
        measurements_planned=25434, measurements_done=12048, stages=stages)])


def test_the_left_column_keeps_clear_of_the_right_one():
    """The font size is bounded by width, not height, and this is the bound.

    Raising it from 8.5 to 10pt widened every line by 18% and needed the stage label and the
    progress bar shortened to fit. A couple of characters of clearance is the margin: enough for a
    step number that grows a digit, not enough for a new line wider than the ones already there.
    """
    for stage in ('full', 'screen', 'confirm'):
        left, right, _ = render_text_extents(busy_state(stage))
        assert left.x1 <= eval_progress.RIGHT_COLUMN_X - 0.02, (
            'at stage {0} the left column reaches {1:.3f}, right starts at {2}'.format(
                stage, left.x1, eval_progress.RIGHT_COLUMN_X))
        assert right.x1 <= 1.0, 'at stage {0} the right column runs off at {1:.3f}'.format(
            stage, right.x1)


def test_the_right_column_fits_when_no_row_reached_full_length():
    """The wording that was clipped. `b43d`'s live chart drew `perfect % of 23 deep rows (deepest 488
    ep, none full)` -- 53 characters -- and the frame cut it after `none full`, losing exactly the
    caveat the line exists to carry. Every fixture above has full-length rows, so the branch that
    produces this wording was never measured; this is the one that measures it."""
    shallow = busy_state('confirm')
    # Every row abandoned short of the target, which at a 98 gate is the normal case for a weak arm.
    for row in shallow['completed']:
        row['episodes'] = 488
        row['abandoned'] = True
    shallow['target_episodes'] = 500
    _, right = eval_progress.summary_columns('b43d-lowlr-b29c', shallow)
    assert 'none full length' in right, right
    left_box, right_box, _ = render_text_extents(shallow, policy='b43d-lowlr-b29c')
    assert right_box.x1 <= 1.0, 'the right column runs off the frame at {0:.3f}'.format(right_box.x1)
    assert left_box.x1 <= eval_progress.RIGHT_COLUMN_X - 0.02, left_box.x1


def test_both_columns_fit_inside_the_text_panel():
    left, right, _ = render_text_extents(busy_state('confirm'))
    assert left.y0 >= 0.0, 'left column overflows the panel: y0 {0:.3f}'.format(left.y0)
    assert right.y0 >= 0.0, 'right column overflows the panel: y0 {0:.3f}'.format(right.y0)


def test_the_text_panel_is_no_taller_than_the_block_it_draws():
    """Was `holds sixteen lines at the chosen size`, which pinned the panel's *capacity*: the box was
    a fixed gridspec ratio sized for the documented worst case, so the only question was whether the
    worst case fit. Since `fit_summary_panel` (2026-08-19) the panel is sized **to** the block it
    actually draws, and the failure mode flipped -- a panel taller than its text is the 324 px of
    blank the fit exists to remove. `test_both_columns_fit_inside_the_text_panel` is the other side;
    this is the one that would regress if the fit were reverted or bypassed.

    Measured against the drawn extents rather than a line count, because a literal 16 would now pin
    how many lines `busy_state` happens to produce instead of the layout rule.
    """
    left, right, inches = render_text_extents(busy_state())
    line = eval_progress.SUMMARY_FONTSIZE * 1.2 / 72.0
    drawn = max(left.height, right.height) * inches   # axes fraction of the panel -> inches
    slack = inches - drawn
    assert slack >= 0, 'the block overflows its panel by {0:.2f}in'.format(-slack)
    # 2.5 lines, as a literal: one is the slack the layout means to leave, and the rest is the
    # difference between a text artist's ink box and its line box. Reading SUMMARY_SLACK_LINES here
    # instead would move the bar with the setting, so raising the slack could not fail this.
    allowed = 2.5 * line
    assert slack <= allowed, '{0:.2f}in of blank under a {1:.2f}in block (allowed {2:.2f}in)'.format(
        slack, drawn, allowed)


def test_the_summary_font_is_larger_than_the_axis_labels():
    # Raised from 8.5 on request. The chart titles are 10pt, so this is now the same size as them
    # and clearly above the 7-8pt tick and axis labels.
    assert eval_progress.SUMMARY_FONTSIZE >= 10.0


# ------------------------------- the ETA prices the plan ahead, not the shortfall (2026-08-19)

def b43c_state(**overrides):
    """`b43c-lowlr-b40b`'s real numbers mid-confirm, the run that surfaced the 3.3x ETA error.

    803 full rows and 522 screens done, 52 of 137 confirmations done; 101,700 episodes planned,
    77,883 run, 16,697 saved by the abandonment gate; 38,466s of session time.
    """
    payload = dict(
        complete=False, num_workers=4, episodes_per_checkpoint=100, screen_episodes=20,
        confirm_count=100, min_achievable=96.0, abandoned=525, episodes_saved=16697,
        measurements_planned=1462, measurements_done=1377,
        episodes_planned=101700, episodes_done=77883,
        session_measurements=1377, session_episodes=77883, session_seconds=38466.1,
        stages={'order': ['full', 'screen', 'confirm'], 'current': 'confirm',
                'full': {'planned': 803, 'done': 803},
                'screen': {'planned': 522, 'done': 522},
                'confirm': {'planned': 137, 'done': 52}})
    payload.update(overrides)
    rows = [result(1000 * (i + 1), episodes=100, perfect=95) for i in range(20)]
    return eval_progress.summarize([run(rows, **payload)])


def test_the_eta_counts_the_work_ahead_not_the_planned_minus_done_gap():
    """The 3.3x error: `episodes_planned - episodes_done` counts the gate's savings as work to come.

    b43c read **3h16m** with 85 confirmations left. Each is an 80-episode top-up, so the real
    remainder was 85 x 80 = 6,800 episodes at 0.494s = **56m**. The 16,697-episode difference is
    exactly `episodes_saved` -- rows the gate abandoned early, whose unrun episodes sat in
    `planned - done` as though they were still ahead.
    """
    state = b43c_state()
    assert state['episodes_ahead'] == 6800
    assert 3300 < state['eta_seconds'] < 3500, state['eta_seconds']
    # What the old arithmetic would have said, for the record.
    assert (101700 - 77883) * state['seconds_per_episode'] > 11000


def test_abandonment_does_not_inflate_the_eta():
    """Same work ahead, wildly different amounts of work the gate declined to run: same ETA.

    This is the property the old formula lacked. Holding the stage plan fixed, an arm the gate cut
    hard and an arm it never touched have the same amount left to do.
    """
    cut = b43c_state()
    untouched = b43c_state(abandoned=0, episodes_saved=0, episodes_done=94580,
                           session_episodes=94580, session_seconds=46718.0)
    assert cut['episodes_ahead'] == untouched['episodes_ahead']
    assert abs(cut['eta_seconds'] - untouched['eta_seconds']) < 60, (
        cut['eta_seconds'], untouched['eta_seconds'])


def test_the_eta_prices_a_confirmation_as_a_top_up_not_a_full_pass():
    # A confirmation adds the difference between the full length and the screen it already has, the
    # same split plan_stages uses. Pricing it at 100 would read 25% high.
    state = b43c_state(stages={'order': ['full', 'screen', 'confirm'], 'current': 'confirm',
                               'full': {'planned': 1, 'done': 1},
                               'screen': {'planned': 1, 'done': 1},
                               'confirm': {'planned': 10, 'done': 0}})
    assert state['episodes_ahead'] == 10 * 80


def test_the_eta_rounds_each_item_up_to_whole_rounds():
    # 12 workers cannot run 100 episodes; it runs 108, and 20 becomes 24. An estimate that ignored
    # the rounding reads ~7% low.
    assert eval_progress.whole_rounds(100, 12) == 108
    assert eval_progress.whole_rounds(20, 12) == 24
    assert eval_progress.whole_rounds(100, 4) == 100
    assert eval_progress.whole_rounds(100, None) == 100
    state = b43c_state(num_workers=12,
                       stages={'order': ['full', 'screen', 'confirm'], 'current': 'full',
                               'full': {'planned': 10, 'done': 0},
                               'screen': {'planned': 10, 'done': 0},
                               'confirm': {'planned': 0, 'done': 0}})
    assert state['episodes_ahead'] == 10 * 108 + 10 * 24


def test_the_flat_protocol_eta_counts_whole_measurements():
    # No stages and no screen: every measurement is a full pass, so the remaining count times the
    # full length is exact -- and both counts include resumed rows, so a resume is handled too.
    state = b43c_state(screen_episodes=None, stages=None,
                       measurements_planned=30, measurements_done=20)
    assert state['episodes_ahead'] == 10 * 100


def test_the_eta_falls_back_when_the_file_predates_the_plan_fields():
    # Files with episodes_planned but no episodes_per_checkpoint keep the old estimate, minus the
    # gate's savings.
    state = b43c_state(episodes_per_checkpoint=None, stages=None, screen_episodes=None,
                       episodes_planned=100000, episodes_saved=10000)
    assert state['episodes_ahead'] is None
    # `state['episodes']` is the row sum, which is what the fallback measures against -- the payload's
    # `episodes_done` is the same number on a real file, but this fixture's rows are synthetic.
    expected = (100000 - state['episodes'] - 10000) * state['seconds_per_episode']
    assert abs(state['eta_seconds'] - expected) < 1.0, (state['eta_seconds'], expected)


def test_the_eta_line_shows_the_episode_count_it_is_priced_on():
    """`remaining x pace` disagrees with the ETA whenever the remaining work is not average work.

    At the end of a screening close-out every remaining measurement is an 80-episode confirmation
    while `pace` is blended over 20-episode screens too, so the obvious check reads ~30% low and the
    ETA looks broken. The episode count is what reconciles them.
    """
    text = '\n'.join(eval_progress.ranking_lines(b43c_state()))
    assert '6,800 ep left' in text, text


def test_a_flat_exact_quota_run_prices_its_remaining_episodes_exactly():
    """`num_workers` is a *round size*, not a parallelism figure, and conflating them broke an ETA.

    `remaining_episodes` multiplies every checkpoint still ahead by `whole_rounds(episodes,
    num_workers)`, which is correct for the batched TF path: `evaluate` runs one episode per worker
    per round and cannot stop mid-round, so 100 episodes on 12 workers really runs 108. The
    vectorised driver runs an exact quota instead, and it reported its 1024-lane batch width in this
    field -- so 100 episodes rounded up to a whole 1024-episode "round" and every b45 arm's chart
    read a 6-8 h ETA against a true ~50 min, an exact 10.24x.

    Both halves are pinned here, because the fix is *not* to make `whole_rounds` stop rounding -- that
    would silently under-price a real batched run.
    """
    def run(workers):
        return [{'episodes_per_checkpoint': 100, 'num_workers': workers,
                 'measurements_planned': 3222, 'measurements_done': 2072,
                 'stages': {'full': {'planned': 3222, 'done': 2072}}, 'screen_episodes': None}]

    exact = eval_progress.remaining_episodes(run(None))
    assert exact == (3222 - 2072) * 100 == 115000, exact

    # The batched-path contract, kept deliberately: a round size larger than the request really does
    # cost a whole round, so this number is right for a run that measures in rounds.
    rounded = eval_progress.remaining_episodes(run(1024))
    assert rounded == (3222 - 2072) * 1024 == 1177600, rounded
    assert rounded == pytest_approx_ratio(exact, 10.24), (rounded, exact)


def pytest_approx_ratio(base, ratio):
    """The inflation factor spelled out, so the 10.24x in the docstring above is asserted."""
    return int(round(base * ratio))
