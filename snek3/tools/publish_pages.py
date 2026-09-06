"""Builds the GitHub Pages site -- `index.html`, `manifest.js`, `charts/*.png`, `.nojekyll` -- from `viewer/` and a runs directory.
Pages serves the **`site` branch**, which `tools/site_build.py` builds on the desktop from both boxes'
results feeds and pushes as a snapshot; `publish(runs_dir, viewer_dir, site_dir, manifest)` is the writer,
a library function with no default target. Until 2026-09-05 the site was `master`'s `/docs`, committed by
the progress update; that local build is gone -- to see the page locally, regenerate the manifest with
`python -m tools.viewer_manifest` and open `viewer/index.html`, which reads `../runs/` directly.
| in the site | from |
|---|---|
| `index.html` | `viewer/index.html`, byte for byte |
| `manifest.js` | the same manifest `viewer/manifest.js` gets, with `charts_dir` set to `charts/` |
| `charts/<policy>*.png` | every chart the manifest refers to -- stage A, stage B, hof5000, hof30k |
| `.nojekyll` | so Pages serves the files as they are |
Files under `charts/` that no arm refers to any more are removed, so the folder never grows past what
the page can show. A copy happens only when size or mtime differ, so a run with nothing new changes
nothing.
"""
import os
import shutil

from env import constants
from tools import viewer_manifest

VIEWER_DIR = os.path.join(constants.ROOT, 'viewer')
CHARTS_SUBDIR = 'charts'
SITE_URL = 'https://twang35.github.io/Snek/'
SITE_CHARTS_URL = SITE_URL + CHARTS_SUBDIR + '/'      # where the docs link a picture: the site, not master

SUFFIXES = {'stage_b_png': '_checkpoint_evals.png', 'hof_png': '_checkpoint_evals_hof5000.png',
            'hof30k_png': '_checkpoint_evals_hof30k.png'}


def chart_files(manifest):
    """The PNG basenames the page can show, given a manifest."""
    names = []
    for arm in manifest['arms']:
        names.append(arm['policy'] + '.png')
        for flag, suffix in SUFFIXES.items():
            if arm.get(flag):
                names.append(arm['policy'] + suffix)
    return names


def _same(src, dst):
    try:
        a, b = os.stat(src), os.stat(dst)
    except OSError:
        return False
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def publish(runs_dir, viewer_dir, site_dir, manifest=None):
    """Rewrites `site_dir` from the sources. Returns `(copied, removed, total)` chart counts."""
    viewer_dir = viewer_dir or VIEWER_DIR
    manifest = manifest or viewer_manifest.build(runs_dir)
    charts_dir = os.path.join(site_dir, CHARTS_SUBDIR)
    os.makedirs(charts_dir, exist_ok=True)

    shutil.copyfile(os.path.join(viewer_dir, 'index.html'), os.path.join(site_dir, 'index.html'))
    with open(os.path.join(site_dir, 'manifest.js'), 'w') as handle:
        handle.write(viewer_manifest.render(manifest, charts_dir=CHARTS_SUBDIR + '/'))
    open(os.path.join(site_dir, '.nojekyll'), 'a').close()

    wanted = chart_files(manifest)
    copied = 0
    for name in wanted:
        src, dst = os.path.join(runs_dir, name), os.path.join(charts_dir, name)
        if not _same(src, dst):
            shutil.copy2(src, dst)
            copied += 1
    removed = 0
    for name in os.listdir(charts_dir):
        if name not in set(wanted):
            os.remove(os.path.join(charts_dir, name))
            removed += 1
    return copied, removed, len(wanted)

