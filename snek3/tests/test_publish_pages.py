"""The site directory is rebuilt from `viewer/` and `runs/`: the page, a manifest pointing at `charts/`, exactly
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
    _touch(os.path.join(viewer, 'pages', 'atlas.html'), b'<title>Deep RL Atlas</title>')
    _touch(os.path.join(viewer, 'pages', 'fonts', 'plex-sans-latin.woff2'), b'wOF2')
    _touch(os.path.join(viewer, 'pages', '.DS_Store'), b'junk')
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
    assert open(os.path.join(docs, 'atlas.html'), 'rb').read() == b'<title>Deep RL Atlas</title>'
    assert open(os.path.join(docs, 'fonts', 'plex-sans-latin.woff2'), 'rb').read() == b'wOF2'
    assert not os.path.exists(os.path.join(docs, '.DS_Store'))
    assert publish_pages.standalone_pages(viewer) == ['atlas.html', os.path.join('fonts', 'plex-sans-latin.woff2')]
    text = open(os.path.join(docs, 'manifest.js')).read()
    payload = json.loads(text[len('window.SNEK_MANIFEST = '):].rstrip().rstrip(';'))
    assert payload['charts_dir'] == 'charts/'
    assert [a['policy'] for a in payload['arms']] == ['b9aa-lam0-seed1']
    # a second run with nothing new copies nothing
    assert publish_pages.publish(runs, viewer, docs)[:2] == (0, 0)


def test_publish_without_a_pages_directory_writes_only_the_viewer(tmp_path):
    runs, viewer, docs = str(tmp_path / 'runs'), str(tmp_path / 'viewer'), str(tmp_path / 'docs')
    _touch(os.path.join(viewer, 'index.html'), b'<title>Snek charts</title>')
    _touch(os.path.join(runs, 'b9aa-lam0-seed1.png'))
    publish_pages.publish(runs, viewer, docs)
    assert sorted(n for n in os.listdir(docs) if n.endswith('.html')) == ['index.html']
    assert publish_pages.standalone_pages(viewer) == []


def test_chart_files_follows_the_manifest_flags():
    manifest = {'arms': [{'policy': 'p', 'stage_b_png': True, 'hof_png': False, 'hof30k_png': True}]}
    assert publish_pages.chart_files(manifest) == ['p.png', 'p_checkpoint_evals.png', 'p_checkpoint_evals_hof30k.png']
