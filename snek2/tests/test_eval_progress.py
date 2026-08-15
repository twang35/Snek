"""Tests for eval_progress.py's aggregation, which has to read two protocols' output.

`summarize()` sees result files from the flat one-pass protocol and from the screening protocol
at the same time — a resumed arm can even produce both — so the fallbacks matter as much as the
new fields do.
"""
import os
import shutil
import tempfile

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
    # Answers "what percentage was screened out": 181 screened, 100 promoted, 81 left at 20.
    lines = eval_progress.stage_lines(
        eval_progress.stage_summary([staged(full=(48, 48), screen=(181, 181), confirm=(100, 5))]))
    screen_line = [l for l in lines if l.strip().startswith('screen')][0]
    assert '100 promoted' in screen_line and '81 (45%)' in screen_line, screen_line


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
    text = eval_progress.text_summary('arm', state)
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
    assert 'none full length' in text and 'full-length rows only' not in text, text
    assert '58 of 100 episodes' in text, text


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
    assert 'none full length' in text, text
    assert '36 of 100 episodes' in text, text


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
    assert limits['flight'] == limits['idle'] == (0, 100), limits


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
