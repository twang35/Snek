"""The chart viewer's manifest: the numbers it shows are the docs tables' definitions, and an arm
without a chart is not a row."""

import json
import os

from tools import viewer_manifest as vm


def _write(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def _arm(runs, policy, evals=None, rows=None, stage_b_png=False, hof=None, hof30k=None):
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
    if hof is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals_hof5000.json'),
               {'policy': policy, 'rows': [{'step': i, 'perfect_percent': p} for i, p in enumerate(hof)]})
        open(os.path.join(runs, policy + '_checkpoint_evals_hof5000.png'), 'wb').close()
    if hof30k is not None:
        _write(os.path.join(runs, policy + '_checkpoint_evals_hof30k.json'),
               {'policy': policy, 'rows': [{'step': 10 * i, 'perfect_percent': p} for i, p in enumerate(hof30k)]})
        open(os.path.join(runs, policy + '_checkpoint_evals_hof30k.png'), 'wb').close()


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
    _arm(runs, 'b9ce-lam999-seed1', evals=[0, 85, 40, 99], rows=[97.0, 98.0, 98.6, 99.2], stage_b_png=True,
         hof=[98.2, 98.8, 99.4], hof30k=[98.9, 99.3])
    _arm(runs, 'b9cf-lam999-seed2', evals=[0, 90])                # trained, no stage B yet
    _arm(runs, 'b9cg-lam999-seed3')                               # a chart and nothing else
    open(os.path.join(runs, 'b9ce-lam999-seed1_checkpoint_evals_hof5000.png'), 'wb').close()  # derived, not an arm
    _write(os.path.join(runs, 'orphan_evals.json'), {'summary': {}})   # no chart -> not a row
    manifest = vm.build(runs)
    by = {a['policy']: a for a in manifest['arms']}
    assert list(by) == ['b9ce-lam999-seed1', 'b9cf-lam999-seed2', 'b9cg-lam999-seed3']
    a = by['b9ce-lam999-seed1']
    assert (a['rows'], a['density98'], a['cands99'], a['best_row']) == (4, 75.0, 1, 99.2)
    assert (a['best30'], a['drawdown50'], a['stage_b_png']) == (99, 33.33, True)
    assert (a['hof_png'], a['hof_rows'], a['hof_mean'], a['hof_best'], a['hof_9873']) == (True, 3, 98.8, 99.4, 2)
    assert (a['hof30k_png'], a['hof30k_rows'], a['hof30k_mean'], a['hof30k_best'], a['hof30k_best_step']) == (True, 2, 99.1, 99.3, 10)
    b = by['b9cf-lam999-seed2']
    assert b['rows'] is None and b['density98'] is None and b['best30'] == 90
    assert b['hof_png'] is False and b['hof_rows'] is None and b['hof30k_png'] is False and b['hof30k_rows'] is None
    c = by['b9cg-lam999-seed3']
    assert c['best30'] is None and c['rows'] is None and c['batch'] == 'b9'


def test_render_is_a_script_that_defines_the_global():
    text = vm.render({'generated': 'now', 'arms': []})
    assert text.startswith('window.SNEK_MANIFEST = {') and text.rstrip().endswith('};')
    assert json.loads(text[len('window.SNEK_MANIFEST = '):].rstrip().rstrip(';')) == {'generated': 'now', 'arms': [], 'charts_dir': '../runs/'}
    assert '"charts_dir":"charts/"' in vm.render({'generated': 'now', 'arms': []}, charts_dir='charts/')


def test_references_attach_only_arms_that_exist(tmp_path):
    runs = str(tmp_path / 'runs'); os.makedirs(runs)
    _arm(runs, 'b7aa-fc320-seed1'); _arm(runs, 'b9ce-lam999-seed1')
    refs = tmp_path / 'references.json'
    refs.write_text(json.dumps({'_comment': 'x', 'b9': {'arms': ['b7aa-fc320-seed1', 'b7zz-missing-seed9'], 'label': 'ref', 'after': 'lam97'},
                                'b10': {'arms': ['b7aa-fc320-seed1'], 'label': 'ref too'}}))
    m = vm.build(runs, str(refs))
    assert m['references'] == {'b9': {'arms': ['b7aa-fc320-seed1'], 'label': 'ref', 'after': 'lam97'},
                               'b10': {'arms': ['b7aa-fc320-seed1'], 'label': 'ref too', 'after': None}}
    assert vm.build(runs, str(tmp_path / 'absent.json'))['references'] == {}


def test_live_desktop_snapshot_stands_in_until_the_close_out_file_exists(tmp_path):
    (tmp_path / 'b30aa-x-seed1.png').write_bytes(b'')
    live = tmp_path / '.live' / 'desktop'
    live.mkdir(parents=True)
    evals = [{'step': 1000 * i, 'perfect_percent': 90} for i in range(10)]
    (live / 'b30aa-x-seed1_evals.json').write_text(json.dumps(
        {'summary': {'step': 9000, 'best_perfect30': {'value': 97.0, 'step': 8000}, 'strong_eval_fraction': 65.2},
         'evals': evals}))
    (live / 'b30aa-x-seed1_checkpoint_evals.json').write_text(json.dumps({'rows': [{'perfect_percent': 98.2, 'step': 1}]}))
    record = vm.arm_record('b30aa-x-seed1', str(tmp_path))
    assert record['best30'] == 97.0 and record['sef'] == 65.2 and record['drawdown50'] == 0.0
    assert record['rows'] == 1 and record['density98'] == 100.0
    (tmp_path / 'b30aa-x-seed1_evals.json').write_text(json.dumps(
        {'summary': {'step': 50000, 'best_perfect30': {'value': 98.5, 'step': 40000}}, 'evals': evals}))
    assert vm.arm_record('b30aa-x-seed1', str(tmp_path))['best30'] == 98.5     # the real file wins


# ---- where each view's pass stands, so a missing panel says why
def test_pass_state_ranks_files_over_liveness_over_candidates():
    assert vm.pass_state(True, True, 0, True, 'queued') == 'done'          # the file settles it
    assert vm.pass_state(False, True, 0, False, None) == 'running'         # shards on disk
    assert vm.pass_state(False, False, 3, True, None) == 'running'         # a running desktop job names the arm
    assert vm.pass_state(False, False, 3, False, 'queued') == 'queued'
    assert vm.pass_state(False, False, 3, False, 'running') == 'queued'    # the batch's job runs, not yet on this arm
    assert vm.pass_state(False, False, 3, False, 'done') == 'pending'      # ledger says done but no file: still owed
    assert vm.pass_state(False, False, 0, False, None) == 'none'
    assert vm.pass_state(False, False, None, False, None) == 'upstream'


def test_shard_files_keep_labelled_passes_apart(tmp_path):
    for name in ('b30aa-x-seed1_checkpoint_evals-s1of8.json', 'b30aa-x-seed1_checkpoint_evals-s2of8.json',
                 'b30aa-x-seed1_checkpoint_evals_hof5000-s1of8.json', 'b30aa-x-seed1_checkpoint_evals.json'):
        (tmp_path / name).write_text('{}')
    assert len(vm.shard_files(str(tmp_path), 'b30aa-x-seed1')) == 2
    assert len(vm.shard_files(str(tmp_path), 'b30aa-x-seed1', 'hof5000')) == 1
    assert vm.shard_files(str(tmp_path), 'b30aa-x-seed1', 'hof30k') == []


def test_status_follows_the_files_when_nothing_is_live(tmp_path):
    runs = str(tmp_path)
    _arm(runs, 'b30aa-x-seed1', evals=[90, 99], rows=[99.2, 97.0], hof=[99.1, 98.0], hof30k=[98.9])
    _arm(runs, 'b30ab-x-seed2', evals=[90, 99], rows=[99.2, 97.0], hof=[98.0])       # hof5000 found nothing >=99
    _arm(runs, 'b30ac-x-seed3', evals=[90, 99], rows=[97.0])                          # nothing >=99 /500
    _arm(runs, 'b30ad-x-seed4', evals=[90, 99], rows=[99.5])                          # hof5000 owed
    _arm(runs, 'b30ae-x-seed5', evals=[90, 99])                                       # stage B owed
    (tmp_path / 'b30ae-x-seed5_checkpoint_evals-s1of4.json').write_text('{"rows": []}')
    _arm(runs, 'b30af-x-seed6', evals=[90, 99])
    by = {a['policy']: a for a in vm.build(runs)['arms']}
    assert by['b30aa-x-seed1']['status'] == {'a': 'done', 'b': 'done', 'h': 'done', 'k': 'done'}
    assert by['b30aa-x-seed1']['hof_99'] == 1
    assert by['b30ab-x-seed2']['status'] == {'a': 'done', 'b': 'done', 'h': 'done', 'k': 'none'}
    assert by['b30ac-x-seed3']['status'] == {'a': 'done', 'b': 'done', 'h': 'none', 'k': 'none'}
    assert by['b30ad-x-seed4']['status'] == {'a': 'done', 'b': 'done', 'h': 'pending', 'k': 'upstream'}
    assert by['b30ae-x-seed5']['status'] == {'a': 'done', 'b': 'running', 'h': 'upstream', 'k': 'upstream'}
    assert by['b30af-x-seed6']['status'] == {'a': 'done', 'b': 'pending', 'h': 'upstream', 'k': 'upstream'}


def test_status_reads_the_desktop_ledger_snapshot(tmp_path):
    runs = str(tmp_path)
    live = tmp_path / '.live' / 'desktop'
    live.mkdir(parents=True)
    _arm(runs, 'b31aa-x-seed1', evals=[90, 99], rows=[99.5])       # closed; its hof passes are queued
    _arm(runs, 'b31ab-x-seed2', evals=[90, 99], rows=[99.5])
    _arm(runs, 'b32aa-x-seed1')                                     # training on the box: snapshot, no file
    (live / 'b32aa-x-seed1_evals.json').write_text(json.dumps({'summary': {'step': 5}, 'evals': []}))
    _arm(runs, 'b32ab-x-seed2')                                     # queued on the box
    _arm(runs, 'b33aa-x-seed1', evals=[90, 99])                     # in the running stage-B job
    _arm(runs, 'b33ab-x-seed2', evals=[90, 99])                     # same batch, later wave
    (live / 'status.json').write_text(json.dumps({
        'iso': '2026-09-04T12:00:00',
        'running': [{'id': 'b33-stageb', 'type': 'eval', 'policies': ['b33aa-x-seed1']},
                    {'id': 'b31-hof5000', 'type': 'eval', 'policies': ['b31ab-x-seed2']}],
        'ledger': {'b31-hof5000': 'running', 'b31-hof30k': 'queued', 'b32aa-x-seed1': 'running',
                   'b32ab-x-seed2': 'queued', 'b33-stageb': 'running', 'b33aa-x-seed1': 'done',
                   'b33ab-x-seed2': 'done'}}))
    manifest = vm.build(runs)
    assert manifest['desktop_iso'] == '2026-09-04T12:00:00'
    by = {a['policy']: a for a in manifest['arms']}
    assert by['b31aa-x-seed1']['status'] == {'a': 'done', 'b': 'done', 'h': 'queued', 'k': 'queued'}
    assert by['b31ab-x-seed2']['status']['h'] == 'running'
    assert by['b32aa-x-seed1']['status'] == {'a': 'running', 'b': 'upstream', 'h': 'upstream', 'k': 'upstream'}
    assert by['b32ab-x-seed2']['status']['a'] == 'queued'
    assert by['b33aa-x-seed1']['status']['b'] == 'running'
    assert by['b33ab-x-seed2']['status']['b'] == 'queued'
    assert vm.build(runs)['arms'][0]['status']                         # and without a snapshot nothing breaks
    (live / 'status.json').unlink()
    assert vm.build(runs)['desktop_iso'] is None
