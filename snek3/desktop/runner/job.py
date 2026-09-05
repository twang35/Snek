"""Job specs: the JSON the laptop drops into `queue/pending/` on the `ops` branch.

One file, one job. The daemon reads them **read-only** from the branch and reconciles them against
its own ledger by job id — it never writes back to `ops`, which is what keeps that branch
single-writer and conflict-free.

## `project` is required, and that is the point of this file

`ops` still holds ~150 stale **snek2** specs in `queue/pending/`, left there when snek2's daemon was
retired. They are valid JSON, they have plausible ids, and their `script` would resolve to a
TensorFlow trainer that does not exist in this tree. A spec with no `project`, or one naming any
project but this daemon's, is **rejected and recorded** rather than skipped quietly, so an accidental
dispatch is impossible and a future snek4 inherits the same guard.

There is no default. A missing field is the exact case being guarded against, so defaulting it to
`snek3` would defeat the guard on precisely the specs it exists for.

## `label` and `notes` are folded to ASCII, and that is a display fix with one cause

`status.json` is published with `json.dumps`, whose default `ensure_ascii=True` renders every
non-ASCII character as a `\\uXXXX` escape **in the file a human reads**. An em dash in a label came
back as `"b8 \\u2014 b8: kl02, seed 1 of 4 \\u2014 wave 2 of 2"`, which is correct JSON and unreadable
prose. Rejecting the spec would stop real work over a cosmetic problem, so the text is folded here
instead: the punctuation an agent reaches for has an ASCII spelling (`--`, `->`, `>=`, `"`), and
anything without one becomes `?` rather than an escape.

Folding at parse time rather than at publish time means the ledger, the logs and `at_a_glance` all
carry the same string, and a spec already sitting on `ops` is fixed without rewriting the branch.
`runner.py` also publishes with `ensure_ascii=False` now, so a non-ASCII *policy name* — which is not
folded, because it is a path — reads as itself rather than as an escape.

## An eval job carries `policies`, not one `policy`

A stage-B pass is a **wave**: one `tools/closeout.py` process owning every arm of a batch, so the
spec names them all and the daemon dispatches one job where it would otherwise dispatch four. That is
the same process an agent runs on the laptop, which is why the sequencing is there and not here. `policy` is
sugar for the single-policy case, and neither field is ever None where the other is set — whichever
is given fills in the other, so a caller that knows only about `policy` still works.
"""

import json
import re

PROJECT = 'snek3'
# `deploy` and `restart` are **actions**, not work: the daemon runs them itself on the poll that sees
# them, ahead of dispatch and regardless of what is running or of a pause, so a deploy never needs ssh
# or sudo -- the box has `Restart=always`, and a restart is the daemon recording the action, publishing,
# and exiting (2026-09-05). Named actions only, never arbitrary shell: anything on `ops` runs as `claw`.
ACTION_TYPES = ('deploy', 'restart')
JOB_TYPES = ('train', 'smoke', 'benchmark', 'eval') + ACTION_TYPES
_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')

# The characters an agent writing prose actually reaches for, and their ASCII spellings. Anything
# else non-ASCII falls through to `?`, which is visible rather than silent.
_ASCII_FOLD = {
    0x2014: '--', 0x2013: '-', 0x2212: '-',            # em dash, en dash, minus
    0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"',  # curly quotes
    0x2026: '...', 0x00a0: ' ', 0x2009: ' ', 0x202f: ' ',  # ellipsis, the spaces
    0x2265: '>=', 0x2264: '<=', 0x2260: '!=', 0x00d7: 'x', 0x00b1: '+/-',
    0x2192: '->', 0x2190: '<-', 0x2021: '!', 0x2020: '+',
}


def to_ascii(text):
    """`text` with its non-ASCII folded to ASCII. See the module docstring for why, not just what."""
    if not text:
        return text
    folded = text.translate(_ASCII_FOLD)
    return folded.encode('ascii', 'replace').decode('ascii')


class JobError(Exception):
    """A spec that cannot be run. Recorded against the spec, never raised into the loop."""


class Job(object):
    """One unit of work the daemon dispatches."""

    def __init__(self, id, type, project=PROJECT, policy=None, env=None, max_steps=None,
                 eval_shards=None, eval_args=None, priority=100, notes='', label='',
                 policies=None, selector=None, episodes=None, restart=None):
        self.id = id
        self.type = type
        self.project = project
        # Runs in both directions: `_StubJob` in runner.py rebuilds a Job from a ledger record after
        # a daemon restart, with no spec in hand.
        self.policies = [name for name in (list(policies) if policies else [policy]) if name]
        self.policy = policy or (self.policies[0] if self.policies else None)
        self.env = env or {}
        self.max_steps = max_steps
        # How many shard processes take checkpoints in parallel. None means "whatever the live
        # runtime config says", which is the normal case.
        self.eval_shards = eval_shards
        # Which checkpoints, and how deeply. None on either means the protocol's default, which is
        # `screen:97` at 500 episodes (95 until 2026-08-30) — see `tools/closeout.py`.
        self.selector = selector
        self.episodes = episodes
        self.eval_args = eval_args or []
        self.priority = priority
        self.notes = notes
        # A short human description for `status.json`'s at-a-glance summary, e.g.
        # 'b1: free space + chase-safe shaping, gate=75, c=0.10'. Optional.
        self.label = label
        # deploy only: True forces a restart after the merge, False forbids one, None (the default)
        # restarts iff the merge changed `desktop/runner/` or `desktop/systemd/`.
        self.restart = restart

    @property
    def category(self):
        """Which concurrency pool the job draws from: 'eval' or 'trainer' -- or 'action' for a
        deploy/restart, which draws from none and runs in the poll that sees it."""
        if self.type in ACTION_TYPES:
            return 'action'
        return 'eval' if self.type == 'eval' else 'trainer'


def parse_job(text, source='<job>', project=PROJECT):
    """Parses and validates one spec.

    Raises `JobError` with a message naming the source, so a bad spec is recorded and skipped rather
    than crashing the loop. Every check is on the spec alone — nothing here touches the filesystem or
    the git bus, which is what makes this the one module worth unit-testing exhaustively.
    """
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as error:
        raise JobError('{0}: not valid JSON: {1}'.format(source, error))
    if not isinstance(raw, dict):
        raise JobError('{0}: must be a JSON object'.format(source))

    job_project = raw.get('project')
    if job_project != project:
        raise JobError(
            '{0}: project must be "{1}", got {2!r}. `ops` still carries retired snek2 specs in '
            'queue/pending/; this daemon runs {1} jobs only.'.format(source, project, job_project))

    job_id = raw.get('id')
    if not isinstance(job_id, str) or not _ID_RE.match(job_id):
        raise JobError('{0}: id must match {1}'.format(source, _ID_RE.pattern))

    job_type = raw.get('type')
    if job_type not in JOB_TYPES:
        raise JobError('{0}: type must be one of {1}'.format(source, JOB_TYPES))

    env = raw.get('env', {})
    if not isinstance(env, dict):
        raise JobError('{0}: env must be an object'.format(source))
    # Stringified because these go straight into a subprocess environment, where a JSON number would
    # otherwise arrive as `1` from an int and `1.0` from a float and change an arm's config.
    env = {str(key): str(value) for key, value in env.items()}

    policy = raw.get('policy')
    policies = raw.get('policies')
    if policies is not None:
        if (not isinstance(policies, list) or not policies
                or any(not isinstance(name, str) or not name for name in policies)):
            raise JobError('{0}: policies must be a non-empty list of strings'.format(source))
        if job_type != 'eval':
            raise JobError('{0}: only eval jobs take "policies" (a wave); {1} jobs take '
                           '"policy"'.format(source, job_type))
    if job_type in ('train', 'eval') and not (policy or policies):
        raise JobError('{0}: {1} jobs need a "policy" (or "policies" for an eval wave)'.format(
            source, job_type))
    if policy is not None and not isinstance(policy, str):
        raise JobError('{0}: policy must be a string'.format(source))

    def optional_int(key):
        value = raw.get(key)
        # `isinstance(True, int)` is True in Python, and `eval_shards: true` would otherwise become
        # one shard rather than an error.
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise JobError('{0}: {1} must be an integer'.format(source, key))
        return value

    max_steps = optional_int('max_steps')
    eval_shards = optional_int('eval_shards')
    episodes = optional_int('episodes')

    selector = raw.get('selector')
    if selector is not None and not isinstance(selector, str):
        raise JobError('{0}: selector must be a string, e.g. "screen:97"'.format(source))
    if selector is not None and job_type != 'eval':
        raise JobError('{0}: only eval jobs take "selector"'.format(source))

    eval_args = raw.get('eval_args', [])
    if not isinstance(eval_args, list) or any(not isinstance(arg, str) for arg in eval_args):
        raise JobError('{0}: eval_args must be a list of strings'.format(source))

    priority = raw.get('priority', 100)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise JobError('{0}: priority must be an integer'.format(source))

    restart = raw.get('restart')
    if restart is not None and not isinstance(restart, bool):
        raise JobError('{0}: restart must be true or false'.format(source))
    if restart is not None and job_type != 'deploy':
        raise JobError('{0}: only deploy jobs take "restart"'.format(source))
    if job_type in ACTION_TYPES and (policy or policies or env or max_steps is not None):
        raise JobError('{0}: a {1} action takes no policy, env or max_steps'.format(source, job_type))

    return Job(id=job_id, type=job_type, project=job_project, policy=policy, policies=policies,
               env=env, max_steps=max_steps, eval_shards=eval_shards, selector=selector,
               episodes=episodes, eval_args=eval_args, priority=priority, restart=restart,
               notes=to_ascii(str(raw.get('notes', ''))),
               label=to_ascii(str(raw.get('label', ''))))
