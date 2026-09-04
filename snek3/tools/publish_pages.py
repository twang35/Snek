"""Builds the GitHub Pages site into the repository's top-level `docs/`, from `viewer/` and `runs/`.

    PYTHONPATH=. python -m tools.publish_pages          # -> ../docs/: index.html, manifest.js, charts/*.png

GitHub Pages serves `master`'s `/docs` folder (source switched from the branch root on 2026-09-03: the
whole tree is 0.98 GB against Pages' 1 GB cap, and the site needs only the viewer and ~24 MB of charts).
`docs/` is therefore **generated output, never edited by hand**: this rewrites it completely from its
sources on every run and the progress-update skill commits it. What goes in:

| in `docs/` | from |
|---|---|
| `index.html` | `viewer/index.html`, byte for byte |
| `manifest.js` | the same manifest `viewer/manifest.js` gets, with `charts_dir` set to `charts/` |
| `charts/<policy>*.png` | every chart the manifest refers to — stage A, stage B, hof5000, hof30k |
| `.nojekyll` | so Pages serves the files as they are |

Files under `charts/` that no arm refers to any more are removed, so the folder never grows past what
the page can show. A copy happens only when size or mtime differ, so a run with nothing new changes
nothing and `git status` stays quiet.
"""

import os
import shutil
import sys

from env import constants
from tools import viewer_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(constants.RUNS_DIR))
VIEWER_DIR = os.path.join(os.path.dirname(constants.RUNS_DIR), 'viewer')
DOCS_DIR = os.path.join(REPO_ROOT, 'docs')
CHARTS_SUBDIR = 'charts'

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


def publish(runs_dir=None, viewer_dir=None, docs_dir=None, manifest=None):
    """Rewrites `docs_dir` from the sources. Returns `(copied, removed, total)` chart counts."""
    runs_dir = runs_dir or constants.RUNS_DIR
    viewer_dir = viewer_dir or VIEWER_DIR
    docs_dir = docs_dir or DOCS_DIR
    manifest = manifest or viewer_manifest.build(runs_dir)
    charts_dir = os.path.join(docs_dir, CHARTS_SUBDIR)
    os.makedirs(charts_dir, exist_ok=True)

    shutil.copyfile(os.path.join(viewer_dir, 'index.html'), os.path.join(docs_dir, 'index.html'))
    with open(os.path.join(docs_dir, 'manifest.js'), 'w') as handle:
        handle.write(viewer_manifest.render(manifest, charts_dir=CHARTS_SUBDIR + '/'))
    open(os.path.join(docs_dir, '.nojekyll'), 'a').close()

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


def main(argv=None):
    manifest = viewer_manifest.build()
    # The source viewer's own manifest, so the page works opened from `viewer/` too.
    with open(viewer_manifest.MANIFEST_PATH, 'w') as handle:
        handle.write(viewer_manifest.render(manifest))
    copied, removed, total = publish(manifest=manifest)
    print('{0}: {1} arms -> {2}: {3} charts ({4} copied, {5} removed)'.format(
        manifest['generated'], len(manifest['arms']), os.path.relpath(DOCS_DIR), total, copied, removed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
