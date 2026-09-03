"""The chart viewer's manifest: the numbers it shows are the docs tables' definitions, and an arm
without a chart is not a row."""

import json
import os

from tools import viewer_manifest as vm


def _write(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def _arm(runs, policy, evals=None, rows=None, stage_b_png=False):
    open(os.path.join(runs, policy + '.png'), 'wb').close()
    if evals is not None:
        best = max(evals, default=0)
        _write(os.path.join(runs, policy + '_evals.json'), {
            'summary': {'step': 100, 'evals': len(evals), 'trailing_now': 1.0,
                        'strong_eval_fraction': 50.0,
                        'best_perfect30': {'value': best, 'step': 7}},
            'evals': [{'step': i, 'perfect_percent': p} for i, p in enumerate(evals)]})
    if rows is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals.json'),
               {'policy': policy, 'rows': [{'step': i, 'perfect_percent': p} for i, p in enumerate(rows)]})
    if stage_b_png:
        open(os.path.join(runs, policy + '_checkpoint_evals.png'), 'wb').close()


def test_names_split_into_batch_knob_and_seed():
    assert (vm.batch_of('b9ce-lam999-seed1'), vm.knob_of('b9ce-lam999-seed1'),
            vm.seed_of('b9ce-lam999-seed1')) == ('b9', 'lam999', 1)
    assert (vm.batch_of('b3k-fc200x100'), vm.knob_of('b3k-fc200x100'), vm.seed_of('b3k-fc200x100')) == \
        ('b3', 'fc200x100', None)
    assert vm.batch_of('ppo-smoke') == 'ppo'
    assert vm.knob_of('b4a-fc200x100ep8-seed1') == 'fc200x100ep8'


def test_drawdown_is_the_post_onset_share_below_the_threshold():
    evals = [{'perfect_percent': p} for p in [10, 40, 85, 30, 90, 70, 99]]
    # onset at 85; post = [85, 30, 90, 70, 99]: one below 50, two below 80
    assert vm.drawdown(evals, 50) == 20.0
    assert vm.drawdown(evals, 80) == 40.0
    assert vm.drawdown([{'perfect_percent': 10}], 50) is None


def test_build_reduces_each_arm_to_the_docs_numbers(tmp_path):
    runs = str(tmp_path)
    _arm(runs, 'b9ce-lam999-seed1', evals=[0, 85, 40, 99], rows=[97.0, 98.0, 98.6, 99.2], stage_b_png=True)
    _arm(runs, 'b9cf-lam999-seed2', evals=[0, 90])                # trained, no stage B yet
    _arm(runs, 'b9cg-lam999-seed3')                               # a chart and nothing else
    open(os.path.join(runs, 'b9ce-lam999-seed1_checkpoint_evals_hof5000.png'), 'wb').close()  # derived, not an arm
    _write(os.path.join(runs, 'orphan_evals.json'), {'summary': {}})   # no chart -> not a row
    manifest = vm.build(runs)
    by = {a['policy']: a for a in manifest['arms']}
    assert list(by) == ['b9ce-lam999-seed1', 'b9cf-lam999-seed2', 'b9cg-lam999-seed3']
    a = by['b9ce-lam999-seed1']
    assert (a['rows'], a['density98'], a['cands985'], a['best_row']) == (4, 75.0, 2, 99.2)
    assert (a['best30'], a['drawdown50'], a['stage_b_png']) == (99, 33.33, True)
    b = by['b9cf-lam999-seed2']
    assert b['rows'] is None and b['density98'] is None and b['best30'] == 90
    c = by['b9cg-lam999-seed3']
    assert c['best30'] is None and c['rows'] is None and c['batch'] == 'b9'


def test_render_is_a_script_that_defines_the_global():
    text = vm.render({'generated': 'now', 'arms': []})
    assert text.startswith('window.SNEK_MANIFEST = {') and text.rstrip().endswith('};')
    assert json.loads(text[len('window.SNEK_MANIFEST = '):].rstrip().rstrip(';')) == {'generated': 'now', 'arms': []}
