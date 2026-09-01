"""What counts as a `hallOfFame/` entry, and what `hof` resolves to.

`hallOfFame/` holds entry directories *and* a `gifs/` directory beside them, so "every
subdirectory" is the wrong definition and was the shipped one: with no entries it made `hof`
resolve to `gifs`, and with two entries it reported three. An entry is a directory with an
`arch.json` in it — the same test `tools/restore.policy_dir` uses, so anything listed here loads.
"""

import json
import os

import pytest

import record_gif
from tools import arch as arch_tools


def an_entry(root, name, step):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(arch_tools.arch_path(d), 'w') as f:
        json.dump({'algo': 'ppo', 'fc_layer_params': [320], 'num_actions': 3,
                   'obs_era': 'b09c616', 'obs_len': 30}, f)
    open(os.path.join(d, 'ckpt-%d.pt' % step), 'wb').close()
    return d


@pytest.fixture
def hof(tmp_path, monkeypatch):
    root = str(tmp_path / 'hallOfFame')
    os.makedirs(root)
    monkeypatch.setattr(record_gif, 'HOF_DIR', root)
    return root


def test_a_sibling_directory_without_arch_json_is_not_an_entry(hof):
    # `gifs/` is the real one. A blacklist would have fixed only this name.
    os.makedirs(os.path.join(hof, 'gifs'))
    os.makedirs(os.path.join(hof, 'some-future-sibling'))
    an_entry(hof, 'b5h-ep8-seed8-ckpt9027584', 9027584)
    assert record_gif.hof_entries() == ['b5h-ep8-seed8-ckpt9027584']


def test_a_lone_gifs_directory_leaves_the_folder_empty(hof):
    os.makedirs(os.path.join(hof, 'gifs'))
    assert record_gif.hof_entries() == []


def test_entries_are_listed_sorted(hof):
    an_entry(hof, 'b6b-fc200x100-seed2-ckpt133120000', 133120000)
    an_entry(hof, 'b5h-ep8-seed8-ckpt9027584', 9027584)
    assert record_gif.hof_entries() == ['b5h-ep8-seed8-ckpt9027584',
                                        'b6b-fc200x100-seed2-ckpt133120000']


def test_hof_resolves_to_the_named_record_even_with_several_entries(hof, monkeypatch):
    an_entry(hof, 'b5h-ep8-seed8-ckpt9027584', 9027584)
    an_entry(hof, 'b6b-fc200x100-seed2-ckpt133120000', 133120000)
    monkeypatch.setattr(record_gif, 'HOF_RECORD', 'b5h-ep8-seed8-ckpt9027584')
    ckpt_dir, step, _ = record_gif.resolve_policy('hof', None)
    assert os.path.basename(ckpt_dir) == 'b5h-ep8-seed8-ckpt9027584'
    assert step == 9027584


def test_hof_is_ambiguous_when_several_entries_and_no_record_named(hof, monkeypatch):
    an_entry(hof, 'b5h-ep8-seed8-ckpt9027584', 9027584)
    an_entry(hof, 'b6b-fc200x100-seed2-ckpt133120000', 133120000)
    monkeypatch.setattr(record_gif, 'HOF_RECORD', None)
    with pytest.raises(SystemExit) as e:
        record_gif.resolve_policy('hof', None)
    # It has to name them, or the error tells you nothing you can act on.
    assert 'b6b-fc200x100-seed2-ckpt133120000' in str(e.value)


def test_the_shipped_record_is_an_entry_that_exists():
    # HOF_RECORD is a hand-edited constant; a typo in it turns `hof` into a resolution failure.
    if record_gif.HOF_RECORD is not None:
        assert record_gif.HOF_RECORD in record_gif.hof_entries()
