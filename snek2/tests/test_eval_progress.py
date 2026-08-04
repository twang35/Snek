"""Tests for eval_progress.py's aggregation, which has to read two protocols' output.

`summarize()` sees result files from the flat one-pass protocol and from the screening protocol
at the same time — a resumed arm can even produce both — so the fallbacks matter as much as the
new fields do.
"""
import eval_progress


def result(step, episodes=100, perfect=70, seconds=100.0):
    return {'step': step, 'episodes': episodes, 'perfect_games': perfect,
            'perfect_percent': round(100.0 * perfect / episodes, 1),
            'perfect_ci95': [0.0, 100.0], 'avg_score': 90.0, 'seconds': seconds}


def run(results, complete=True, **extra):
    payload = {'suffix': '_t', 'results': results, 'complete': complete,
               'updated_at': 1e12, 'mtime': 1e12}
    payload.update(extra)
    return payload


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
