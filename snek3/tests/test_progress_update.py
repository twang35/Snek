"""tools/progress_update.py: the tables, the doc surgery and the shared-queue reading, on synthetic data."""

import datetime
import json
import os

from tools import progress_update as pu
from tools import viewer_manifest


def _arm(policy, rows=None, best30=97.0, sef=80.0, evals=None):
    return {
        'policy': policy, 'batch': viewer_manifest.batch_of(policy), 'knob': viewer_manifest.knob_of(policy),
        'seed': viewer_manifest.seed_of(policy), 'stage_b_png': bool(rows), 'rows': len(rows or []),
        'density98': (100.0 * sum(r >= 98 for r in rows) / len(rows)) if rows else 0.0,
        'cands99': sum(r >= 99 for r in rows or []), 'best_row': max(rows) if rows else None,
        'best30': best30, 'best30_step': 40e6, 'sef': sef, 'drawdown50': 1.0, 'drawdown80': 5.0,
        'stage_a_98': 20.0, 'onset_step': 3e6, 'evals': 100,
    }


ARMS = [_arm('b20aa-k1-seed1', [97, 98, 99]), _arm('b20ab-k1-seed2', [96, 96]),
        _arm('b20ac-k2-seed1', [98, 98, 98, 99]), _arm('b20ad-k2-seed2', []),
        _arm('b19zz-ref-seed1', [98, 97], best30=98.0)]
ENVS = {'b20aa-k1-seed1': {'SNEK_KNOB': '0.5', 'SNEK_SEED': '1', 'SNEK_OTHER': 'x', '_max_steps': 5e7},
        'b20ab-k1-seed2': {'SNEK_KNOB': '0.5', 'SNEK_SEED': '2', 'SNEK_OTHER': 'x', '_max_steps': 5e7},
        'b20ac-k2-seed1': {'SNEK_KNOB': '0.9', 'SNEK_SEED': '1', 'SNEK_OTHER': 'x', '_max_steps': 5e7},
        'b20ad-k2-seed2': {'SNEK_KNOB': '0.9', 'SNEK_SEED': '2', 'SNEK_OTHER': 'x', '_max_steps': 5e7}}
MANIFEST = {'arms': ARMS, 'references': {}}
REF = {'arms': ['b19zz-ref-seed1'], 'label': 'reference: b19', 'value': '0.7'}


def test_knob_key_is_the_one_env_var_that_varies_apart_from_seed():
    assert pu.knob_key(ENVS) == 'SNEK_KNOB'
    assert pu.knob_key({'a': {'SNEK_SEED': '1'}, 'b': {'SNEK_SEED': '2'}}) is None


def test_batch_table_groups_by_spec_value_pools_rows_and_slots_the_reference_numerically():
    table = pu.batch_table('b20', MANIFEST, ENVS, REF, ENVS)
    values = [g['value'] for g in table['groups']]
    assert values == ['0.5', '0.7', '0.9']
    assert [g['reference'] for g in table['groups']] == [False, True, False]
    k1 = table['groups'][0]
    assert k1['rows'] == 5 and k1['density98'] == 40.0          # 2 of 5 rows, pooled not averaged
    assert k1['per_seed'] == '66.7 0.0' and k1['cands99'] == 1 and k1['best_row'] == 99
    k2 = table['groups'][2]
    assert k2['per_seed'] == '100.0 –' and k2['rows'] == 4       # the rowless seed shows as a dash


def test_knob_value_falls_back_to_the_name_when_there_is_no_spec():
    assert pu.knob_value(None, 'SNEK_KNOB', 'b20aa-k1-seed1') == 'k1'
    assert pu.knob_value({'SNEK_KNOB': '0.5'}, 'SNEK_KNOB', 'b20aa-k1-seed1') == '0.5'


def test_group_table_md_marks_the_reference_row():
    md = pu.group_table_md(pu.batch_table('b20', MANIFEST, ENVS, REF, ENVS))
    assert '| **0.7** (reference) |' in md and md.startswith('| knob |')


CHARTS = """# Charts

## Watching them live

words

## Batch b20 — hand-written

**In flight**: wave 1 training.

| knob | best30 |
|---|---|
| 0.5 | 97 |

**k1** — `b20aa`-`b20ab`:

![b20aa-k1-seed1](../runs/b20aa-k1-seed1.png)

**What to look at.** The reading.

## Batch b19 — older

untouched
"""


def _table():
    return pu.batch_table('b20', MANIFEST, ENVS, REF, ENVS)


def test_hand_written_section_is_left_alone_unless_adopted():
    assert pu.update_charts_md(CHARTS, _table(), 'Batch b20 — t', 'status') == CHARTS


def test_adoption_keeps_every_prose_paragraph_and_drops_tables_panels_and_group_headings():
    out = pu.update_charts_md(CHARTS, _table(), 'Batch b20 — t', 'status', adopt=True)
    reading = out[out.index(pu.READING):out.index(pu.END_READING)]
    assert '**In flight**: wave 1 training.' in reading and '**What to look at.** The reading.' in reading
    assert '| 0.5 | 97 |' not in reading and '![' not in reading and '**k1** —' not in reading
    assert out.count('## Batch b20') == 1 and '## Batch b19 — older\n\nuntouched' in out
    assert pu.MARK.format('b20') in out and pu.END_MARK.format('b20') in out
    assert '![b20ac-k2-seed1](https://twang35.github.io/Snek/charts/b20ac-k2-seed1.png)' in out
    assert '![b20ac-k2-seed1 stage B](https://twang35.github.io/Snek/charts/b20ac-k2-seed1_checkpoint_evals.png)' in out
    assert '![b20ad-k2-seed2 stage B]' not in out                  # no stage-B panel without rows
    assert '**knob 0.7** — reference: b19:' in out and '![b19zz-ref-seed1](https://twang35.github.io/Snek/charts/b19zz-ref-seed1.png)' in out


def test_regeneration_preserves_the_reading_verbatim_and_replaces_everything_else():
    once = pu.update_charts_md(CHARTS, _table(), 'Batch b20 — t', 'status', adopt=True)
    edited = once.replace(pu.END_READING, 'A sentence written by hand.\n' + pu.END_READING)
    again = pu.update_charts_md(edited, _table(), 'Batch b20 — new title', 'new status')
    assert 'A sentence written by hand.' in again and '## Batch b20 — new title' in again
    assert 'Batch b20 — t\n' not in again and again.count(pu.MARK.format('b20')) == 1
    assert again.index('## Batch b20') < again.index('## Batch b19')


def test_new_batch_goes_above_the_first_batch_section():
    text = CHARTS.replace('b20', 'b21')
    out = pu.update_charts_md(text, _table(), 'Batch b20 — t', 'status')
    assert out.index(pu.MARK.format('b20')) < out.index('## Batch b21')
    assert out.index('## Watching them live') < out.index(pu.MARK.format('b20'))
    # and above the existing section's own start marker, not between that marker and its heading
    # (which is where b13 landed on 2026-09-04, stacking the markers and displacing two readings)
    if pu.MARK.format('b21') in text:
        assert out.index(pu.END_MARK.format('b20')) < out.index(pu.MARK.format('b21'))
    generated = pu.update_charts_md(out, dict(_table(), batch='b22'), 'Batch b22 — t', 'status')
    assert generated.index(pu.END_MARK.format('b22')) < generated.index(pu.MARK.format('b20'))


def test_results_skeleton_is_inserted_once_above_the_first_batch():
    results = '# Results\n\nintro\n\n## Batch b19 — old\n\nold\n'
    out, added = pu.insert_results_skeleton(results, _table(), 'Batch b20 — closed', 'facts')
    assert added and out.index('## Batch b20') < out.index('## Batch b19')
    assert '### Every arm' in out and '| `b20aa-k1-seed1` | 0.5 |' in out and pu.READING in out
    again, added = pu.insert_results_skeleton(out, _table(), 'Batch b20 — closed', 'facts')
    assert not added and again == out
    # a second closed batch goes above the first one's start marker, not between that marker and its heading
    more, added = pu.insert_results_skeleton(out, dict(_table(), batch='b21'), 'Batch b21 — closed', 'facts')
    assert added and more.index(pu.END_MARK.format('b21')) < more.index(pu.MARK.format('b20'))


def _view(specs, records, published=(), running=None):
    """A `tools.claims.gather` view on dicts: specs on ops, claims, published ids, running ids."""
    return {'specs': {s['id']: s for s in specs}, 'records': list(records), 'published': {p: 'desktop' for p in published},
            'running': dict(running or {}), 'ledger': {}}


def _train(policy):
    return {'id': policy, 'type': 'train', 'policy': policy, 'max_steps': 10, 'priority': 100}


B20 = ['b20aa-k1-seed1', 'b20ab-k1-seed2', 'b20ac-k2-seed1', 'b20ad-k2-seed2']
VIEW = _view([_train(p) for p in B20 + ['b19zz-ref-seed1']],
             [{'kind': 'wave', 'batch': 'b20', 'wave': 1, 'box': 'desktop', 'arms': B20[:2]},
              {'kind': 'wave', 'batch': 'b20', 'wave': 2, 'box': 'laptop', 'arms': B20[2:3]},
              {'kind': 'wave', 'batch': 'b19', 'wave': 1, 'box': 'laptop', 'arms': ['b19zz-ref-seed1']}],
             published=B20[:2] + ['b20-stageb', 'b20-hof5000', 'b20-hof30k', 'b19zz-ref-seed1', 'b19-stageb',
                                  'b19-hof5000', 'b19-hof30k'],
             running={B20[2]: 'laptop'})


def test_batch_counts_span_both_boxes_and_count_the_unclaimed_as_queued():
    arms, waves = pu.batch_counts(VIEW, 'b20')
    assert dict(arms) == {'done': 2, 'running': 1, 'queued': 1} and dict(waves) == {'done': 1, 'queued': 1}


def test_eta_uses_the_fallback_cadence_when_no_wave_has_closed(monkeypatch):
    monkeypatch.setattr(pu, 'wave_close_times', lambda batch: [])
    now = datetime.datetime(2026, 9, 3, 12, 0)
    remaining, per_wave, finish = pu.eta(VIEW, 'b20', now=now, fallback_seconds=3600)
    assert remaining == 1 and per_wave == 3600 and finish == now + datetime.timedelta(hours=1)
    monkeypatch.setattr(pu, 'wave_close_times', lambda batch: [0, 7200, 14400])
    assert pu.eta(VIEW, 'b20', now=now)[1] == 7200


def test_live_batches_are_those_with_anything_left_on_either_box():
    assert pu.live_batches(VIEW) == ['b20']
    assert pu.desktop_live_batches(VIEW) == [], 'b20 has nothing open on the desktop; its w1 is closed'


def test_state_line_reads_wave_by_wave_across_the_boxes(monkeypatch):
    monkeypatch.setattr(pu, 'wave_close_times', lambda batch: [])
    assert pu.state_line(VIEW, 'b19').startswith('Closed: w1 laptop closed')
    line = pu.state_line(VIEW, 'b20', {'running': [{'id': B20[2], 'step': 4, 'max_steps': 10}]})
    assert line.startswith('In flight: w1 desktop closed, w2 laptop training (0 of 1 at cap, live at 40%), 1 arm unclaimed; ')
    assert '1 training wave(s) left' in line
    assert pu.state_line(_view([], []), 'b99').startswith('b99: not on ops')


def test_import_skips_shard_files_and_existing_ones(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pu.subprocess, 'run', lambda argv, **kw: calls.append(argv) or type(
        'R', (), {'stdout': b'{}', 'returncode': 0})())
    (tmp_path / 'b20aa-k1-seed1.png').write_bytes(b'x')
    tree = ['results/b20-stageb-w1/b20aa-k1-seed1.png', 'results/b20-stageb-w1/b20aa-k1-seed1_checkpoint_evals.json',
            'results/b20-stageb-w1/b20aa-k1-seed1_checkpoint_evals-s1of4.json',
            'results/b20aa-k1-seed1/b20aa-k1-seed1_evals.json',          # the arm's own job: imported too
            'results/b20ab-k1-seed2/b20ab-k1-seed2_evals.json']          # on the laptop's feed
    copied = pu.import_closed_waves([('results', p) for p in tree[:4]] + [('laptop-results', tree[4])], str(tmp_path))
    assert copied == 3 and (tmp_path / 'b20aa-k1-seed1_checkpoint_evals.json').exists()
    assert (tmp_path / 'b20aa-k1-seed1_evals.json').exists() and (tmp_path / 'b20ab-k1-seed2_evals.json').exists(), \
        'a job on a feed is finished by construction, whichever feed'
    assert not (tmp_path / 'b20aa-k1-seed1_checkpoint_evals-s1of4.json').exists()
    assert any('origin/laptop-results:' in argv[2] for argv in calls)


def test_import_takes_a_done_hof_pass_too(tmp_path, monkeypatch):
    # b11's hof5000/hof30k finished on the desktop 2026-09-04 and nothing imported them.
    monkeypatch.setattr(pu.subprocess, 'run', lambda argv, **kw: type('R', (), {'stdout': b'{}', 'returncode': 0})())
    tree = ['results/b11-hof30k/b11ae-lr1e4-seed1_checkpoint_evals_hof30k.json',
            'results/b11-hof30k/b11ae-lr1e4-seed1_checkpoint_evals_hof5000-s1of8.json',
            'results/b11-hof5000/b11ae-lr1e4-seed1_checkpoint_evals_hof5000.json']
    tree.append('results/p2-hof5000/p2a-ep8-seed1_checkpoint_evals_hof5000.json')   # b5 under its old name: stays on results
    copied = pu.import_closed_waves([('results', p) for p in tree], str(tmp_path))
    assert copied == 2 and (tmp_path / 'b11ae-lr1e4-seed1_checkpoint_evals_hof30k.json').exists()
    assert not (tmp_path / 'p2a-ep8-seed1_checkpoint_evals_hof5000.json').exists()


def test_superseded_snapshots_are_dropped_once_the_close_out_file_is_in_runs(tmp_path):
    live = tmp_path / '.live' / 'desktop'
    live.mkdir(parents=True)
    (live / 'a_evals.json').write_text('{}')
    (live / 'b_evals.json').write_text('{}')
    (tmp_path / 'a_evals.json').write_text('{}')
    assert pu.drop_superseded_snapshots(str(tmp_path)) == 1
    assert not (live / 'a_evals.json').exists() and (live / 'b_evals.json').exists()


def test_read_spec_falls_back_to_a_local_scheduler_spec(tmp_path, monkeypatch):
    # b13 was dequeued from ops and run by scheduler from logs/b13specs/; the tool must still find its knob.
    monkeypatch.setattr(pu, 'LOCAL_SPECS', str(tmp_path))
    monkeypatch.setattr(pu, '_OPS_SPECS', None)
    monkeypatch.setattr(pu.subprocess, 'run', lambda argv, **kw: type('R', (), {'returncode': 128, 'stdout': ''})())
    assert pu.read_spec('b13aa-mb32-seed1') is None
    (tmp_path / 'b13specs').mkdir()
    (tmp_path / 'b13specs' / 'b13aa-mb32-seed1.json').write_text(json.dumps(
        {'env': {'SNEK_PPO_MINIBATCH': '32'}, 'max_steps': 100, 'notes': 'Prediction: slow'}))
    assert pu.spec_envs(['b13aa-mb32-seed1']) == {'b13aa-mb32-seed1': {'SNEK_PPO_MINIBATCH': '32', '_max_steps': 100}}
    assert pu.spec_notes('b13aa-mb32-seed1') == 'Prediction: slow'


def test_ops_specs_reads_every_pending_spec_in_one_cat_file_batch(monkeypatch):
    # One `ls-tree` and one `cat-file --batch` for all arms, not a `git show` per arm (540 processes, 106 s).
    specs = {'b30aa-x-seed1': json.dumps({'env': {'SNEK_X': '1'}, 'notes': 'Prediction: fast'}),
             'b30ab-x-seed2': json.dumps({'env': {'SNEK_X': '2'}})}
    calls = []

    def run(argv, **kw):
        calls.append(argv[1])
        if argv[1] == 'ls-tree':
            out = ''.join('snek3/desktop/queue/pending/{0}.json\n'.format(p) for p in specs) + 'snek3/desktop/queue/pending/README.md\n'
        else:
            assert argv[1:3] == ['cat-file', '--batch']
            assert kw['input'].splitlines() == ['origin/ops:snek3/desktop/queue/pending/{0}.json'.format(p) for p in specs]
            out = ''.join('{0} blob {1}\n{2}\n'.format('a' * 40, len(t), t) for t in specs.values())
        return type('R', (), {'returncode': 0, 'stdout': out})()

    monkeypatch.setattr(pu, '_OPS_SPECS', None)
    monkeypatch.setattr(pu.subprocess, 'run', run)
    assert pu.spec_envs(list(specs)) == {'b30aa-x-seed1': {'SNEK_X': '1', '_max_steps': None},
                                         'b30ab-x-seed2': {'SNEK_X': '2', '_max_steps': None}}
    assert pu.spec_notes('b30aa-x-seed1') == 'Prediction: fast'
    assert pu.read_spec('b30zz-x-seed9') is None
    assert calls == ['ls-tree', 'cat-file']            # every read after the first is served from the cache


def test_ops_specs_skips_a_missing_blob(monkeypatch):
    def run(argv, **kw):
        if argv[1] == 'ls-tree':
            return type('R', (), {'returncode': 0, 'stdout': 'snek3/desktop/queue/pending/a-x-seed1.json\nsnek3/desktop/queue/pending/b-x-seed1.json\n'})()
        return type('R', (), {'returncode': 0, 'stdout': 'origin/ops:snek3/desktop/queue/pending/a-x-seed1.json missing\n' + 'b' * 40 + ' blob 2\n{}\n'})()
    monkeypatch.setattr(pu, '_OPS_SPECS', None)
    monkeypatch.setattr(pu.subprocess, 'run', run)
    assert pu.ops_specs() == {'b-x-seed1': '{}'}


def test_save_desktop_status_lands_where_the_manifest_reads_it(tmp_path):
    from tools import progress_update, viewer_manifest
    path = progress_update.save_desktop_status({'iso': 'now', 'ledger': {'b9-hof30k': 'queued'}, 'running': []},
                                               runs_dir=str(tmp_path), view=dict(VIEW, ledger={'b20ac-k2-seed1': 'running'}))
    assert path == str(tmp_path / viewer_manifest.DESKTOP_STATUS)
    ledger = viewer_manifest.desktop_ledger(str(tmp_path))
    assert ledger['iso'] == 'now' and ledger['jobs'] == {'b9-hof30k': 'queued', 'b20ac-k2-seed1': 'running'}
    assert ledger['running']['a'] == {'b20ac-k2-seed1'}, "the laptop's running arm reaches the viewer too"
    assert viewer_manifest.boxes(str(tmp_path)) == {p: 'desktop' for p in B20[:2] + ['b19zz-ref-seed1']}
    # nothing in runs/ shares its name, so the superseded-snapshot sweep leaves it alone
    assert progress_update.drop_superseded_snapshots(str(tmp_path)) == 0 and os.path.exists(path)


def test_knob_key_is_none_for_a_switches_batch_and_anneal_cells_are_their_own_group():
    from tools import progress_update as pu
    # b19: four on/off switches, no key with more than two values -> the name token is the cell
    switches = {'b19a-noadvnorm-seed1': {'SNEK_PPO_NORMALIZE_ADV': '0', 'SNEK_SEED': '1'},
                'b19e-mse-seed1': {'SNEK_PPO_VALUE_LOSS': 'mse', 'SNEK_SEED': '1'},
                'b19i-vf01-seed1': {'SNEK_PPO_VF_COEF': '0.1', 'SNEK_SEED': '1'}}
    assert pu.knob_key(switches) is None
    # b15: the knob is the entropy coefficient; an anneal cell shares its start value with a fixed cell but
    # carries a `_FINAL` the fixed cells lack, so it is grouped by name, after the numeric values
    envs = {'b15ai-ent003-seed1': {'SNEK_PPO_ENTROPY_COEF': '0.003', 'SNEK_SEED': '1'},
            'b15aq-ent02-seed1': {'SNEK_PPO_ENTROPY_COEF': '0.02', 'SNEK_SEED': '1'},
            'b15bg-entanneal01-seed1': {'SNEK_PPO_ENTROPY_COEF': '0.01', 'SNEK_PPO_ENTROPY_COEF_FINAL': '0.001', 'SNEK_SEED': '1'},
            'b15bk-entanneal01to0-seed1': {'SNEK_PPO_ENTROPY_COEF': '0.01', 'SNEK_PPO_ENTROPY_COEF_FINAL': '0.0', 'SNEK_SEED': '1'}}
    assert pu.knob_key(envs) == 'SNEK_PPO_ENTROPY_COEF'
    arms = [{'policy': p} for p in envs]
    assert [v for v, _ in pu.group_arms(arms, envs, 'SNEK_PPO_ENTROPY_COEF')] == ['0.003', '0.02', 'entanneal01', 'entanneal01to0']


def test_a_batch_off_the_bus_is_closed_when_every_arm_has_its_three_pass_files(tmp_path):
    from tools import progress_update
    arms = ['b19a-noadvnorm-seed1', 'b19b-noadvnorm-seed2']
    for arm in arms:
        for suffix in progress_update.PASS_SUFFIXES[:2]:
            (tmp_path / (arm + suffix)).write_text('{}')
    assert not progress_update.files_closed(arms, str(tmp_path)), 'hof30k still owed'
    for arm in arms:
        (tmp_path / (arm + progress_update.PASS_SUFFIXES[2])).write_text('{}')
    assert progress_update.files_closed(arms, str(tmp_path))
    assert not progress_update.files_closed([], str(tmp_path)), 'no arms is not a closed batch'
