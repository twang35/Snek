"""`docs/` is rebuilt from `viewer/` and `runs/`: the page, a manifest pointing at `charts/`, exactly
the charts the manifest refers to, and nothing stale."""

import json
import os

from tools import publish_pages, viewer_manifest


def _touch(path, content=b'x'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(content)


def test_publish_writes_the_page_the_manifest_and_only_the_referenced_charts(tmp_path):
    runs, viewer, docs = str(tmp_path / 'runs'), str(tmp_path / 'viewer'), str(tmp_path / 'docs')
    _touch(os.path.join(viewer, 'index.html'), b'<title>Snek charts</title>')
    _touch(os.path.join(runs, 'b9aa-lam0-seed1.png'))
    _touch(os.path.join(runs, 'b9aa-lam0-seed1_checkpoint_evals.png'))
    _touch(os.path.join(runs, 'b9aa-lam0-seed1_checkpoint_evals_hof5000.png'))
    _touch(os.path.join(runs, 'b9aa-lam0-seed1_eval_progress.png'))       # derived, not shown
    _touch(os.path.join(docs, 'charts', 'gone-arm.png'))                    # stale from an earlier publish
    copied, removed, total = publish_pages.publish(runs, viewer, docs)
    assert (copied, removed, total) == (3, 1, 3)
    assert sorted(os.listdir(os.path.join(docs, 'charts'))) == [
        'b9aa-lam0-seed1.png', 'b9aa-lam0-seed1_checkpoint_evals.png',
        'b9aa-lam0-seed1_checkpoint_evals_hof5000.png']
    assert open(os.path.join(docs, 'index.html'), 'rb').read() == b'<title>Snek charts</title>'
    assert os.path.exists(os.path.join(docs, '.nojekyll'))
    text = open(os.path.join(docs, 'manifest.js')).read()
    payload = json.loads(text[len('window.SNEK_MANIFEST = '):].rstrip().rstrip(';'))
    assert payload['charts_dir'] == 'charts/'
    assert [a['policy'] for a in payload['arms']] == ['b9aa-lam0-seed1']
    # a second run with nothing new copies nothing
    assert publish_pages.publish(runs, viewer, docs)[:2] == (0, 0)


def test_chart_files_follows_the_manifest_flags():
    manifest = {'arms': [{'policy': 'p', 'stage_b_png': True, 'hof_png': False, 'hof30k_png': True}]}
    assert publish_pages.chart_files(manifest) == ['p.png', 'p_checkpoint_evals.png', 'p_checkpoint_evals_hof30k.png']
