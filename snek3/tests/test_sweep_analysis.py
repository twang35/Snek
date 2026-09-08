"""The sweep reducer: the numbers `docs/sweep.md` and `viewer/sweep.html` both read."""

import json
import os

from tools import sweep_analysis as sa


def _write(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def _evals(perfects, per_eval=16384, kl=0.002):
    return [{'step': (i + 1) * per_eval, 'transitions': (i + 1) * per_eval, 'perfect_percent': p, 'avg_score': 90.0,
             'ppo': {'value_loss': 1.0, 'explained_variance': 0.9, 'approx_kl': kl if i != 3 else 0.5,
                     'clip_fraction': 0.02, 'entropy': 0.2, 'epochs_run': 4}}
            for i, p in enumerate(perfects)]


def _arm(runs, policy, perfects, rows=None, hof=None, h30=None):
    best = max(perfects)
    _write(os.path.join(runs, policy + '_evals.json'), {
        'summary': {'step': len(perfects) * 16384, 'evals': len(perfects), 'trailing_now': perfects[-1],
                    'strong_eval_fraction': 50.0, 'best_perfect30': {'value': best, 'step': 7}},
        'evals': _evals(perfects), 'resumes': []})
    if rows is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals.json'),
               {'policy': policy, 'rows': [{'step': 3_000_000 * (i + 1), 'perfect_percent': p} for i, p in enumerate(rows)]})
    if hof is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals_hof5000.json'),
               {'policy': policy, 'rows': [{'step': i, 'perfect_percent': p} for i, p in enumerate(hof)]})
    if h30 is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals_hof30k.json'),
               {'policy': policy, 'rows': [{'step': 10 * i, 'perfect_percent': p} for i, p in enumerate(h30)]})


def test_bin_trace_keeps_the_collapse_as_the_bin_min():
    # 15 evals at 95 and one at 0 in one 250k bin: the mean smooths the collapse to ~89, the min shows it
    perfects = [95] * 15 + [0]
    trace = sa.bin_trace(_evals(perfects), bin_size=250_000, horizon=250_000)
    assert trace['perfect'] == [round(sum(perfects) / 16, 2)]
    assert trace['perfect_min'] == [0]
    assert trace['steps'] == [250_000]


def test_bin_trace_leaves_empty_bins_none():
    trace = sa.bin_trace(_evals([95, 96]), bin_size=16384, horizon=4 * 16384)
    assert trace['perfect'] == [95.0, 96.0, None, None]
    assert trace['perfect_min'] == [95, 96, None, None]
    assert trace['kl'][0] == 0.002


def test_stage_b_density_bins_rows_and_rows98():
    rows = [{'step': 1_000_000, 'perfect_percent': 97.0}, {'step': 2_000_000, 'perfect_percent': 98.2},
            {'step': 4_000_000, 'perfect_percent': 99.0}]
    d = sa.stage_b_density(rows, bin_size=2_500_000, horizon=5_000_000)
    assert d['rows'] == [2, 1] and d['rows98'] == [1, 1]


def test_scalars_onsets_and_post_onset_statistics():
    # 40 evals: climb 0..78, then 40 at/above 80 with one collapse to 20
    climb = [0, 20, 40, 60, 78]
    plateau = [92] * 20 + [20] + [96] * 19
    perfects = climb + plateau
    stage_a = {'summary': {'step': 1, 'evals': len(perfects), 'trailing_now': 96, 'strong_eval_fraction': 1,
                           'best_perfect30': {'value': 95.0, 'step': 5}}, 'evals': _evals(perfects)}
    s = sa.scalars(stage_a, None, None, None)
    assert s['onset80'] == 6 * 16384                       # the first eval >= 80
    assert s['onset90'] is not None and s['onset90'] > s['onset80']   # trailing-30 >= 90 arrives later
    assert s['worst_post'] == 20
    assert s['drawdown50'] == round(100 * 1 / 40, 2)
    assert s['drawdown80'] == round(100 * 1 / 40, 2)
    assert s['stage_a_98'] == 0.0
    assert s['end_drop'] == -1.0                            # trailing_now above best30 here, by construction
    assert s['kl_p99'] is not None and s['kl_late'] == 0.002
    assert s['rows'] is None and s['hof_rows'] is None      # no stage B, no hof pass


def test_scalars_stage_b_and_hof():
    s = sa.scalars({'summary': {}, 'evals': []}, {'rows': [{'perfect_percent': p} for p in [97, 98, 98.4, 99.2]]},
                   {'rows': [{'perfect_percent': p} for p in [98.8, 99.1]]},
                   {'rows': [{'step': 5, 'perfect_percent': 99.3}]})
    assert (s['rows'], s['density98'], s['density99'], s['cands99'], s['best_row']) == (4, 75.0, 25.0, 1, 99.2)
    assert (s['hof_rows'], s['hof_best'], s['hof_9873']) == (2, 99.1, 2)
    assert (s['hof30k_best'], s['hof30k_best_step']) == (99.3, 5)


def test_mann_whitney_exact_floor_and_tie():
    assert sa.mann_whitney_exact([5, 6, 7, 8], [1, 2, 3, 4]) == 0.0286      # every seed past every seed
    assert sa.mann_whitney_exact([1, 2, 3, 4], [5, 6, 7, 8]) == 0.0286      # two-sided
    assert sa.mann_whitney_exact([1, 8, 2, 7], [3, 4, 5, 6]) > 0.5           # interleaved
    assert sa.mann_whitney_exact([], [1]) is None


def test_layout_cells_numeric_sorted_then_categorical_with_reference_at_its_value():
    batch = {'batch': 'b17', 'control_value': {'SNEK_PPO_CLIP': '0.2'}, 'cells': [
        {'slug': 'clip03', 'env': {'SNEK_PPO_CLIP': '0.3'}},
        {'slug': 'clip005', 'env': {'SNEK_PPO_CLIP': '0.05'}},
        {'slug': 'clipanneal', 'env': {'SNEK_PPO_CLIP': '0.2', 'SNEK_PPO_CLIP_FINAL': '0.02'}},
        {'slug': 'lranneal', 'env': {'SNEK_PPO_LEARNING_RATE': '3e-4', 'SNEK_PPO_LEARNING_RATE_FINAL': '0'}},
    ]}
    cells = sa.layout_cells(batch, {'arms': ['b7aa-fc320-seed1'], 'value': '0.2', 'after': 'clip015'})
    assert [c['slug'] for c in cells] == ['clip005', 'reference', 'clip03', 'clipanneal', 'lranneal']
    assert [c['label'] for c in cells] == ['0.05', '0.2', '0.3', 'clipanneal', 'lranneal']
    assert [c['x'] for c in cells] == [0, 1, 2, 3, 4]


def test_layout_cells_two_control_keys_are_all_categorical_and_reference_follows_after():
    batch = {'batch': 'b21', 'control_value': {'SNEK_CHASE_SAFE_SHAPING': '0.1', 'SNEK_CHASE_SAFE_GATE': '75'},
             'cells': [{'slug': 'shape0', 'env': {'SNEK_CHASE_SAFE_SHAPING': '0.0'}},
                       {'slug': 'shape005', 'env': {'SNEK_CHASE_SAFE_SHAPING': '0.05'}},
                       {'slug': 'gate60', 'env': {'SNEK_CHASE_SAFE_GATE': '60'}}]}
    cells = sa.layout_cells(batch, {'arms': ['x'], 'value': '0.1', 'after': 'shape005'})
    assert [c['slug'] for c in cells] == ['shape0', 'shape005', 'reference', 'gate60']
    assert all(c['value'] is None for c in cells)


def test_layout_cells_switches_reference_goes_first_without_after():
    batch = {'batch': 'b19', 'control_value': {'A': '1', 'B': 'huber'},
             'cells': [{'slug': 'mse', 'env': {'B': 'mse'}}]}
    cells = sa.layout_cells(batch, {'arms': ['x'], 'value': 'base'})
    assert [c['slug'] for c in cells] == ['reference', 'mse']


def test_build_groups_arms_into_cells_and_compares_to_the_reference(tmp_path):
    runs = str(tmp_path / 'runs'); os.makedirs(runs)
    manifest = {'batches': [{'batch': 'b9', 'knob': 'GAE lambda', 'control_value': {'SNEK_PPO_GAE_LAMBDA': '0.98'},
                             'cells': [{'slug': 'lam0', 'env': {'SNEK_PPO_GAE_LAMBDA': '0.0'}, 'prediction': 'slow'}]}]}
    refs = {'b9': {'arms': ['b7aa-fc320-seed1', 'b7ab-fc320-seed2'], 'label': 'ref', 'value': '0.98'}}
    _write(str(tmp_path / 'm.json'), manifest); _write(str(tmp_path / 'r.json'), refs)
    for p, rows in (('b9a-lam0-seed1', [98, 98, 97]), ('b9b-lam0-seed2', [98, 99, 97])):
        _arm(runs, p, [85] * 40, rows=rows)
    for p, rows in (('b7aa-fc320-seed1', [97, 97, 97]), ('b7ab-fc320-seed2', [98, 97, 97])):
        _arm(runs, p, [85] * 40, rows=rows)
    _arm(runs, 'b9c-lam0-seed3_checkpoint', [1])          # a stray name that must not become an arm
    sweep = sa.build(runs, str(tmp_path / 'm.json'), str(tmp_path / 'r.json'))
    b9 = sweep['batches'][0]
    assert [c['slug'] for c in b9['cells']] == ['lam0', 'reference']
    assert b9['cells'][0]['arms'] == ['b9a-lam0-seed1', 'b9b-lam0-seed2']
    assert b9['cells'][1]['arms'] == ['b7aa-fc320-seed1', 'b7ab-fc320-seed2']
    st = b9['cells'][0]['stats']['density98']
    assert st['values'] == [66.7, 66.7] and st['median'] == 66.7 and st['n'] == 2
    assert st['p'] == sa.mann_whitney_exact([66.7, 66.7], [0.0, 33.3])
    assert set(sweep['arms']) == {'b9a-lam0-seed1', 'b9b-lam0-seed2', 'b7aa-fc320-seed1', 'b7ab-fc320-seed2'}
    table = sa.peaks_markdown(sweep, keys=('density98',))
    assert '`0.0` 66.7' in table


def test_build_appends_the_extra_manifests_batches_after_the_plans_and_shrugs_at_a_missing_file(tmp_path):
    """`plans/sweep-extra.json` carries batches added to the view after the sweep (b26 first, 2026-09-07);
    the plan's file stays the plan. No extra file, or an explicit None-path of '' -- the plan alone."""
    runs = str(tmp_path / 'runs'); os.makedirs(runs)
    plan = {'batches': [{'batch': 'b9', 'knob': 'GAE lambda', 'control_value': {'SNEK_PPO_GAE_LAMBDA': '0.98'},
                         'cells': [{'slug': 'lam0', 'env': {'SNEK_PPO_GAE_LAMBDA': '0.0'}}]}]}
    extra = {'batches': [{'batch': 'b26', 'knob': 'step penalty', 'control_value': {'SNEK_STEP_PENALTY': '0'},
                          'cells': [{'slug': 'pen0', 'env': {'SNEK_STEP_PENALTY': '0'}},
                                    {'slug': 'pen01', 'env': {'SNEK_STEP_PENALTY': '0.01'}}]}]}
    _write(str(tmp_path / 'm.json'), plan); _write(str(tmp_path / 'x.json'), extra); _write(str(tmp_path / 'r.json'), {})
    _arm(runs, 'b26a-pen01-seed1', [85] * 40, rows=[99, 98])
    sweep = sa.build(runs, str(tmp_path / 'm.json'), str(tmp_path / 'r.json'), extra_path=str(tmp_path / 'x.json'))
    assert [b['batch'] for b in sweep['batches']] == ['b9', 'b26']
    b26 = sweep['batches'][1]
    assert [c['slug'] for c in b26['cells']] == ['pen0', 'pen01'] and b26['cells'][1]['arms'] == ['b26a-pen01-seed1']
    alone = sa.build(runs, str(tmp_path / 'm.json'), str(tmp_path / 'r.json'), extra_path=str(tmp_path / 'absent.json'))
    assert [b['batch'] for b in alone['batches']] == ['b9']
    assert [b['batch'] for b in sa.manifest_batches(str(tmp_path / 'm.json'), extra_path='')] == ['b9']


def test_write_and_load_round_trip(tmp_path):
    sweep = {'generated': 'now', 'metrics': [], 'batches': [], 'arms': {}}
    json_path, js_path = sa.write(sweep, str(tmp_path / 's.json'), str(tmp_path / 's.js'))
    assert sa.load(json_path) == sweep
    with open(js_path) as handle:
        assert handle.read().startswith('window.SNEK_SWEEP = {')
