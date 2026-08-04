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
