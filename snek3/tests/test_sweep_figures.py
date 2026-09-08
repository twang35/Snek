"""The sweep figures draw from a tiny reduced sweep without a real runs/ directory."""

import os

from tools import sweep_analysis as sa
from tools import sweep_figures as sf
from tests.test_sweep_analysis import _arm, _write


def _sweep(tmp_path):
    runs = str(tmp_path / 'runs'); os.makedirs(runs)
    manifest = {'batches': [
        {'batch': 'b9', 'knob': 'GAE lambda', 'control_value': {'SNEK_PPO_GAE_LAMBDA': '0.98'},
         'cells': [{'slug': 'lam0', 'env': {'SNEK_PPO_GAE_LAMBDA': '0.0'}}, {'slug': 'lam100', 'env': {'SNEK_PPO_GAE_LAMBDA': '1.0'}}]},
        {'batch': 'b19', 'knob': 'switches', 'control_value': {'A': '1', 'B': 'huber'},
         'cells': [{'slug': 'mse', 'env': {'B': 'mse'}}]}]}
    refs = {'b9': {'arms': ['b7aa-fc320-seed1', 'b7ab-fc320-seed2'], 'label': 'ref', 'value': '0.98'},
            'b19': {'arms': ['b7aa-fc320-seed1', 'b7ab-fc320-seed2'], 'label': 'ref', 'value': 'base'}}
    _write(str(tmp_path / 'm.json'), manifest); _write(str(tmp_path / 'r.json'), refs)
    plateau = [0, 50] + [90] * 38
    for p, rows in (('b9a-lam0-seed1', [98, 97]), ('b9b-lam0-seed2', [99, 97]), ('b9c-lam100-seed1', [98, 98]),
                    ('b9d-lam100-seed2', [99, 98]), ('b19a-mse-seed1', [98]), ('b19b-mse-seed2', [99]),
                    ('b7aa-fc320-seed1', [97, 98]), ('b7ab-fc320-seed2', [97, 97])):
        _arm(runs, p, plateau, rows=rows, hof=[98.9], h30=[99.1] if p.endswith('seed1') else None)
    return sa.build(runs, str(tmp_path / 'm.json'), str(tmp_path / 'r.json'), extra_path='')   # the real extra manifest stays out


def test_every_figure_renders(tmp_path):
    sweep = _sweep(tmp_path)
    out = str(tmp_path / 'figs')
    for batch in sweep['batches']:
        sf._save(sf.curve_figure(sweep, batch), os.path.join(out, batch['batch'] + '-curve.png'))
        sf._save(sf.traces_figure(sweep, batch), os.path.join(out, batch['batch'] + '-traces.png'))
    sf._save(sf.cell_figure(sweep, sweep['batches'][0], 'lam100'), os.path.join(out, 'cell.png'))
    sf._save(sf.levers_figure(sweep), os.path.join(out, 'levers.png'))
    sf._save(sf.peaks_figure(sweep), os.path.join(out, 'peaks.png'))
    fig, rho = sf.scatter_figure(sweep, 'ev_late', 'density98')
    sf._save(fig, os.path.join(out, 'scatter.png'))
    assert len(os.listdir(out)) == 8
    assert all(os.path.getsize(os.path.join(out, f)) > 1000 for f in os.listdir(out))


def test_cell_colours_ramp_for_ordered_and_categorical_for_switches(tmp_path):
    sweep = _sweep(tmp_path)
    b9, b19 = sweep['batches']
    assert list(sf.cell_colours(b9).values()) == [sf.RAMP[0], sf.RAMP[-1]]     # light -> dark by axis position
    assert list(sf.cell_colours(b19).values()) == [sf.CATEGORICAL[0]]
    assert sf.ramp_colours(3)[1] == sf.RAMP[len(sf.RAMP) // 2] or len(sf.ramp_colours(3)) == 3


def test_spearman_ranks_with_ties():
    assert sf._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert sf._spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert abs(sf._spearman([1, 1, 2, 3], [1, 2, 3, 4])) < 1.0
