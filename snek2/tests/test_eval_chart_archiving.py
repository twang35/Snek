"""`archive_existing_eval_pngs` and the batch it must not sweep away.

`evals/` top level is meant to hold only current work, so every eval archives what it finds there
before writing anything of its own. That is right across batches and wrong within one: a batch whose
close-out arrives as several waves used to archive its own earlier waves' charts, so a finished arm's
panel went blank and nothing ever rewrote it. `keep_batches` is the exemption.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_plan


class _Dirs(object):
    """Point eval_plan's module-level EVALS_DIR/EVALS_ARCHIVE_DIR at a scratch tree."""

    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix='evalchart-')
        self.evals = os.path.join(self.root, 'evals')
        self.archive = os.path.join(self.evals, 'archive')
        os.makedirs(self.evals)
        self._saved = (eval_plan.EVALS_DIR, eval_plan.EVALS_ARCHIVE_DIR)
        eval_plan.EVALS_DIR, eval_plan.EVALS_ARCHIVE_DIR = self.evals, self.archive
        return self

    def __exit__(self, *exc):
        eval_plan.EVALS_DIR, eval_plan.EVALS_ARCHIVE_DIR = self._saved
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, *names):
        for name in names:
            with open(os.path.join(self.evals, name), 'w') as fh:
                fh.write('x')

    def top_level(self):
        return sorted(n for n in os.listdir(self.evals) if n.endswith('.png'))

    def archived(self):
        out = []
        for stamp in sorted(os.listdir(self.archive)) if os.path.isdir(self.archive) else []:
            out += sorted(os.listdir(os.path.join(self.archive, stamp)))
        return sorted(out)


def test_with_no_exemption_everything_is_archived():
    with _Dirs() as d:
        d.write('b44a-x_eval_progress.png', 'b45a-x_eval_progress.png')
        eval_plan.archive_existing_eval_pngs()
        assert d.top_level() == []
        assert d.archived() == ['b44a-x_eval_progress.png', 'b45a-x_eval_progress.png']


def test_the_batch_being_measured_keeps_its_charts():
    # b45's closeout wave 2 measures b45b; a's and c's finished charts must stay put.
    with _Dirs() as d:
        d.write('b45a-x_eval_progress.png', 'b45b-x_eval_progress.png',
                'b45c-x_eval_progress.png')
        eval_plan.archive_existing_eval_pngs(keep_batches={'b45'})
        assert d.top_level() == ['b45a-x_eval_progress.png', 'b45b-x_eval_progress.png',
                                 'b45c-x_eval_progress.png']
        assert d.archived() == []


def test_an_earlier_batch_is_still_archived_alongside_an_exemption():
    # The exemption is per batch, not a blanket "keep everything": b44's charts are last batch's
    # work and evals/ is meant to show only current work.
    with _Dirs() as d:
        d.write('b44a-x_eval_progress.png', 'b45a-x_eval_progress.png')
        eval_plan.archive_existing_eval_pngs(keep_batches={'b45'})
        assert d.top_level() == ['b45a-x_eval_progress.png']
        assert d.archived() == ['b44a-x_eval_progress.png']


def test_exempting_the_only_batch_present_does_not_create_an_archive_dir():
    # Nothing to move must stay a no-op -- an empty timestamped directory per wave would be litter
    # in the one place the docs promise is never pruned.
    with _Dirs() as d:
        d.write('b45a-x_eval_progress.png')
        eval_plan.archive_existing_eval_pngs(keep_batches={'b45'})
        assert not os.path.isdir(d.archive)


def test_a_chart_that_is_not_an_eval_progress_png_is_still_archived():
    # `keep_batches` filters on the policy parsed out of `<policy>_eval_progress.png`; anything
    # else in evals/ has no batch and is swept as before.
    with _Dirs() as d:
        d.write('b45a-x_eval_progress.png', 'scratch.png')
        eval_plan.archive_existing_eval_pngs(keep_batches={'b45'})
        assert d.top_level() == ['b45a-x_eval_progress.png']
        assert d.archived() == ['scratch.png']
