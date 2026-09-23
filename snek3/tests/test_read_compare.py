"""`tools/read_compare.py`: a reads pass paired against its control, on hand-written result files."""

import os

from env import constants
from tools import read_compare
from tools import results


def a_row(step, perfect_flags, terminals):
    """One stage-B row: `perfect_flags` per episode and, for each, how it ended (0 for a perfect game,
    -5.0 a death, -0.5 a starve), summarised the way `eval_plan.build_row` writes them."""
    return {'step': step, 'episodes': len(perfect_flags), 'perfect_games': sum(perfect_flags),
            'perfect_percent': 100.0 * sum(perfect_flags) / len(perfect_flags),
            'deaths': sum(1 for t in terminals if t == -5.0), 'starves': sum(1 for t in terminals if t == -0.5)}


def write_pass(arm, label, rows):
    results.write(results.stage_b_path(arm, label), {'policy': arm, 'label': label, 'rows': rows})


def test_reads_are_paired_by_step_and_outcomes_read_off_the_row(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, 'RUNS_DIR', str(tmp_path))
    write_pass('armA', 'reads-meanfixed', [a_row(1000, [1, 1, 0, 0], [0, 0, -5.0, -0.5]),
                                          a_row(2000, [1, 1, 1, 0], [0, 0, 0, -5.0])])
    # leastneg: fewer deaths, more starves, one checkpoint down and one level; a third step the control lacks.
    write_pass('armA', 'reads-leastneg', [a_row(1000, [1, 0, 0, 0], [0, -0.5, -0.5, -0.5]),
                                         a_row(2000, [1, 1, 1, 0], [0, 0, 0, -0.5]),
                                         a_row(3000, [0, 0, 0, 0], [-0.5] * 4)])
    # a shard file with the prefix must not be read as a label
    results.write(results.stage_b_path('armA', 'reads-leastneg', shard=0, shards=2), {'rows': []})
    table = {entry['read']: entry for entry in read_compare.compare(['armA'], 'reads-', 'meanfixed')}
    assert set(table) == {'meanfixed', 'leastneg'}
    control, least = table['meanfixed'], table['leastneg']
    assert control['checkpoints'] == 2 and least['checkpoints'] == 3 and least['paired'] == 2
    assert least['delta'] == -12.5                      # (25 - 50 + 75 - 75) / 2
    assert least['wins'] == 0 and least['losses'] == 1
    assert control['death_pct'] == 25.0 and control['starve_pct'] == 12.5
    assert least['death_pct'] == 0.0 and least['starve_pct'] == 100.0 * 8 / 12
    rendered = read_compare.render(list(table.values()), 'meanfixed')
    assert '`leastneg`' in rendered and '-12.50' in rendered


def test_rows_without_the_counts_report_the_perfect_rate_only(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, 'RUNS_DIR', str(tmp_path))
    write_pass('armA', 'reads-meanfixed', [{'step': 1000, 'perfect_percent': 90.0}])
    write_pass('armA', 'reads-mix07', [{'step': 1000, 'perfect_percent': 80.0}])
    table = {entry['read']: entry for entry in read_compare.compare(['armA'], 'reads-', 'meanfixed')}
    assert table['mix07']['delta'] == -10.0 and table['mix07']['death_pct'] is None
    assert table['mix07']['interval'] is None          # one pair: no bootstrap
