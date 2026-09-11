"""One feed per box of the work its scheduler finished: `results/<job-id>/<files>` on a single-writer branch.

    python -m tools.results_feed <job-id> <file>...     # publish by hand, e.g. after a failed push

The desktop's branch is `results` and the laptop's `laptop-results`, the same layout on both, so one
reader (`tools/site_build.py`, and `tools/progress_update.py`'s importer) serves either. The scheduler
publishes at the three moments a picture becomes final (and, below, every ten minutes while one is not) -- an arm reaching its cap, a pass writing its
merged files, an eval spec finishing -- through `Publisher.publish`, which never raises: the run is the
arms, a github outage is a line in the log, and the commit is local until the next publish's push
carries it (`gitbus._commit_and_push` pushes `HEAD`, earlier commits included).

| moment | job id | files |
|---|---|---|
| an arm at its cap | the policy | `<policy>.md`, `.png`, `_evals.json` |
| a pass exits 0 | `<batch>-<pass>[-wN]` | each arm's `_checkpoint_evals[_<label>].json` and `.png` |
| an eval spec exits 0 | its id | every `runs/` file of its policies |
| every ten minutes | each live arm's policy | its `.png` and `.md`, one commit for the wave |

An arm's `_evals.json` is published once, with the arm, rather than again with each of its three
passes as the daemon did until 2026-09-05 (1.7 MB x 3 passes x 8 arms per wave; the `results` pack
had reached 372 MB).

Which branch and worktree: `SNEK_RESULTS_BRANCH` and `SNEK_RESULTS_WORKTREE`, which the daemon sets
for the desktop's scheduler from `host.env`; the laptop's defaults are below.

**Live pictures too, every ten minutes, on both boxes.** The site showed a laptop arm only once it
reached its cap, while a desktop arm's picture moved every ten minutes because the desktop builds the
site from its own `runs/`. Now every scheduler publishes each live arm's `.png` and `.md` through
`Publisher.publish_live` with its ten-minute status republish, into the same `results/<policy>/` the arm's
final files land in later -- the final simply overwrites the live picture at the same path, and the
desktop's own pictures on `results` make the bus complete rather than a shortcut. Never `_evals.json`:
3.6 MB per arm per ten minutes, and the site only draws the `.png`. What keeps this cheap is that the
branch is a snapshot: `gitbus._commit_and_push` writes one parentless commit per publish (a picture per
arm per ten minutes on a branch with history would have been ~30 MB a day).
"""

import os
import sys

from desktop.daemon import gitbus
from tools import closeout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
REMOTE = 'origin'
BRANCH = os.environ.get('SNEK_RESULTS_BRANCH', 'laptop-results')
WORKTREE = os.environ.get('SNEK_RESULTS_WORKTREE', os.path.expanduser('~/.snek3-laptop/results'))


def host(repo=REPO, worktree=WORKTREE, branch=BRANCH, remote=REMOTE):
    """The host dict `gitbus.publish_results` needs, for this box."""
    return {'REPO_PATH': repo, 'RESULTS_WORKTREE': worktree, 'RESULTS_BRANCH': branch, 'GIT_REMOTE': remote}


def arm_files(policy, runs_dir):
    """The arm's own three files, those that exist."""
    return _existing(runs_dir, ['{0}.md'.format(policy), '{0}.png'.format(policy), '{0}_evals.json'.format(policy)])


def live_files(policies, runs_dir):
    """Each live arm's picture and report, those that exist, keyed by policy -- what the live snapshot
    carries. Never `_evals.json`: 3.6 MB per arm per ten minutes, and the site only draws the `.png`."""
    out = {}
    for policy in policies:
        paths = _existing(runs_dir, ['{0}.png'.format(policy), '{0}.md'.format(policy)])
        if paths:
            out[policy] = paths
    return out


def pass_files(policies, pass_name, runs_dir):
    """Each arm's merged file and picture for one pass of the chain, those that exist."""
    label = closeout.PASSES[pass_name]['label']
    stem = '_checkpoint_evals' + ('_' + label if label else '')
    names = []
    for policy in policies:
        names += [policy + stem + '.json', policy + stem + '.png']
    return _existing(runs_dir, names)


def every_file(policies, runs_dir):
    """Every `runs/` file of these policies -- what an eval spec, whose command is opaque, publishes."""
    try:
        names = sorted(os.listdir(runs_dir))
    except OSError:
        return []
    out = []
    for policy in policies:
        out += [os.path.join(runs_dir, name) for name in names
                if name == policy + '.md' or name.startswith(policy + '.') or name.startswith(policy + '_')]
    return out


def _existing(runs_dir, names):
    paths = [os.path.join(runs_dir, name) for name in names]
    return [path for path in paths if os.path.exists(path)]


class _Job(object):
    def __init__(self, job_id):
        self.id = job_id


class Publisher(object):
    """Publishes one job's files to this box's results branch. `publish` returns True if the push landed
    and never raises. Every process call goes through an attribute so a test can stand in."""

    def __init__(self, host_config=None, publish_results=gitbus.publish_results,
                 publish_jobs=gitbus.publish_jobs, ensure=None, log=None):
        self.host = host_config or host()
        self._publish_results, self._publish_jobs = publish_results, publish_jobs
        self._ensure = ensure or (lambda h: gitbus.ensure_worktree(
            h['REPO_PATH'], h['RESULTS_WORKTREE'], h['RESULTS_BRANCH'], h['GIT_REMOTE']))
        self._ready = False
        self.log = log or (lambda message: sys.stderr.write(message + '\n'))

    def publish(self, job_id, paths):
        if not paths:
            self.log('{0}: nothing to publish (no files)'.format(job_id))
            return False
        try:
            if not self._ready:
                self._ensure(self.host)
                self._ready = True
            landed = self._publish_results(self.host, _Job(job_id), list(paths))
            self.log('{0}: {1} file(s) published to {2}{3}'.format(
                job_id, len(paths), self.host['RESULTS_BRANCH'], '' if landed else ' (push failed; the commit is local)'))
            return landed
        except Exception as error:      # noqa: BLE001 -- "could not publish", never fatal to the run
            self.log('{0}: publish to {1} failed (the scheduler continues): {2}'.format(
                job_id, self.host['RESULTS_BRANCH'], error))
            return False

    def publish_live(self, files_by_policy):
        """The live arms' pictures, `{policy: [paths]}` (`live_files`), as one commit. Nothing to send is
        not an error and not a push. Never raises."""
        if not files_by_policy:
            return False
        try:
            if not self._ready:
                self._ensure(self.host)
                self._ready = True
            landed, copied = self._publish_jobs(self.host, files_by_policy, 'live pictures of {0} arm(s)'.format(
                len(files_by_policy)))
            self.log('live: {0} file(s) of {1} arm(s) published to {2}{3}'.format(
                copied, len(files_by_policy), self.host['RESULTS_BRANCH'], '' if landed else ' (push failed; the commit is local)'))
            return landed
        except Exception as error:      # noqa: BLE001
            self.log('live: publish to {0} failed (the scheduler continues): {1}'.format(self.host['RESULTS_BRANCH'], error))
            return False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.split('\n\n')[0]); print('usage: python -m tools.results_feed <job-id> <file>...')
        return 2
    landed = Publisher().publish(argv[0], argv[1:])
    return 0 if landed else 1


if __name__ == '__main__':
    sys.exit(main())
