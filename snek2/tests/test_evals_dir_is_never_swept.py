"""A tripwire: **starting an eval must not move any chart out of `evals/`.**

This file replaces `test_eval_chart_archiving.py`, which pinned the opposite behaviour. Until
2026-08-24, every eval entry point called `eval_plan.archive_existing_eval_pngs()` before it did
anything else, sweeping every PNG at the top of `evals/` into `evals/archive/<timestamp>/` so the
folder showed "only the most recently completed work".

**It cost far more than the tidiness was worth**, over eight months and repeatedly:

* A one-checkpoint verification eval displaced a whole batch's finished panels — batch 11's four
  charts, then batch 13's, then b43's and k1000's twelve on 2026-08-24 while this very removal was
  being written and tested.
* `EVAL_OUT_SUFFIX` protected the *results* file and not the charts, because the chart path has no
  suffix in it and the sweep ran before any setup, so there was no flag that made an eval harmless.
* A batch measured as several waves erased its own earlier waves. `keep_batches` was added to patch
  that, which made the rule "an eval archives every chart except the ones belonging to batches this
  particular process happens to be measuring" — a rule nobody could hold in their head, and one that
  still displaced any *other* batch's charts.
* Restoring was lossless but manual, and had to be remembered.

The behaviour it was protecting turns out to be free without it: **every arm rewrites its own chart
by name**, so `evals/` is self-correcting. It accumulates instead of resetting, and stale entries are
stale charts of real arms rather than missing charts of current ones — which is the strictly better
failure. `evals/archive/` stays on disk (it is in CLAUDE.md's never-delete table) and nothing writes
there any more.

So what is asserted here is an *absence*, and it is asserted three ways, because an absence is easy to
undo by accident and none of these would fail loudly on their own.
"""

import ast
import os
import sys

SNEK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SNEK_DIR)

import eval_plan

# Every entry point that measures checkpoints. `vec_eval.py` is deliberately included even though it
# never archived: the point is that no eval anywhere acquires the habit.
EVAL_ENTRY_POINTS = ('eval_checkpoints.py', 'eval_wave.py',
                     os.path.join('vectorized', 'vec_eval.py'),
                     os.path.join('vectorized', 'vec_wave.py'))


def test_eval_plan_no_longer_defines_an_archiver():
    """The function itself is gone, so nothing can call it by importing `eval_plan`."""
    assert not hasattr(eval_plan, 'archive_existing_eval_pngs'), (
        'archive_existing_eval_pngs is back — see this module\'s docstring for why it was removed')


def test_no_eval_entry_point_moves_a_chart_out_of_evals():
    """Source-level, because the call was always the *first* statement of `main` and a behavioural
    test would have to run a whole eval to reach it.

    Looks for the act rather than the old name: a reintroduction is much more likely to be a fresh
    `shutil.move` of something under `evals/` than a call to a function that no longer exists.
    """
    for entry in EVAL_ENTRY_POINTS:
        path = os.path.join(SNEK_DIR, entry)
        source = open(path).read()
        assert 'archive_existing_eval_pngs' not in source, entry
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ''
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            assert name not in ('move', 'rmtree'), (
                '{0} calls {1}() — an eval must not relocate or delete chart files'.format(
                    entry, name))


def test_the_archive_directory_is_still_named_but_unused():
    """`EVALS_ARCHIVE_DIR` stays: the directory holds every chart the old sweep took, and it is in
    CLAUDE.md's never-delete table. What must not come back is anything *writing* to it."""
    import snake_constants
    assert snake_constants.EVALS_ARCHIVE_DIR.endswith(os.path.join('evals', 'archive'))
    # `eval_plan` was the only module that ever wrote there, and it no longer imports the name.
    assert not hasattr(eval_plan, 'EVALS_ARCHIVE_DIR'), (
        'eval_plan imports EVALS_ARCHIVE_DIR again — the only reason it ever did was to sweep '
        'charts into it')


def test_two_arms_of_a_batch_can_write_their_charts_in_either_order():
    """The property that makes the sweep unnecessary: charts are keyed by policy name, so an eval
    writing one arm's chart cannot affect another's, whatever order they land in.

    Written as a real filesystem exercise rather than an assertion about code, because this is the
    load-bearing claim — if it were false, `evals/` really would need resetting.
    """
    import shutil
    import tempfile

    root = tempfile.mkdtemp()
    try:
        names = ['b45a-x_eval_progress.png', 'b45b-y_eval_progress.png', 'b44a-old_eval_progress.png']
        for name in names:
            open(os.path.join(root, name), 'wb').write(b'first')
        # A second wave re-measures one arm of one batch. Nothing else may change.
        open(os.path.join(root, 'b45a-x_eval_progress.png'), 'wb').write(b'second')
        assert sorted(os.listdir(root)) == sorted(names), 'a chart went missing'
        assert open(os.path.join(root, 'b45a-x_eval_progress.png'), 'rb').read() == b'second'
        assert open(os.path.join(root, 'b45b-y_eval_progress.png'), 'rb').read() == b'first'
        # And the previous batch's chart survives, which is exactly what the sweep used to destroy.
        assert open(os.path.join(root, 'b44a-old_eval_progress.png'), 'rb').read() == b'first'
    finally:
        shutil.rmtree(root, ignore_errors=True)
