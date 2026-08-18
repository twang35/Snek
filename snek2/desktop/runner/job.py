"""Job specs: the JSON files the laptop drops into `queue/pending/` on the `ops`
branch. One file, one job. The daemon reads them read-only from the branch and
reconciles them against its own ledger by job id -- it never writes back to `ops`,
which is what keeps that branch single-writer and conflict-free."""
import json
import re

JOB_TYPES = ('train', 'smoke', 'benchmark', 'eval')
_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')


class JobError(Exception):
    pass


class Job:
    def __init__(self, id, type, policy=None, env=None, max_steps=None,
                 eval_workers=None, eval_args=None, priority=100, notes='', label=''):
        self.id = id
        self.type = type
        self.policy = policy
        self.env = env or {}
        self.max_steps = max_steps
        self.eval_workers = eval_workers
        self.eval_args = eval_args or []
        self.priority = priority
        self.notes = notes
        # A short human description for status.json's at-a-glance summary, e.g.
        # 'b40: free space + chase-safe shaping, gate=75, c=0.10'. Optional; unset -> ''.
        self.label = label

    @property
    def category(self):
        """Which concurrency pool the job draws from: 'eval' or 'trainer'."""
        return 'eval' if self.type == 'eval' else 'trainer'


def parse_job(text, source='<job>'):
    """Parses and validates one job spec. Raises JobError with a clear message so
    a bad spec is recorded and skipped rather than crashing the loop."""
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as e:
        raise JobError('{0}: not valid JSON: {1}'.format(source, e))
    if not isinstance(raw, dict):
        raise JobError('{0}: must be a JSON object'.format(source))

    jid = raw.get('id')
    if not isinstance(jid, str) or not _ID_RE.match(jid):
        raise JobError('{0}: id must match {1}'.format(source, _ID_RE.pattern))

    jtype = raw.get('type')
    if jtype not in JOB_TYPES:
        raise JobError('{0}: type must be one of {1}'.format(source, JOB_TYPES))

    env = raw.get('env', {})
    if not isinstance(env, dict):
        raise JobError('{0}: env must be an object'.format(source))
    env = {str(k): str(v) for k, v in env.items()}  # SNEK_* go to the environment

    policy = raw.get('policy')
    if jtype in ('train', 'eval') and not policy:
        raise JobError('{0}: {1} jobs need a "policy"'.format(source, jtype))
    if policy is not None and not isinstance(policy, str):
        raise JobError('{0}: policy must be a string'.format(source))

    def opt_int(key):
        v = raw.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise JobError('{0}: {1} must be an integer'.format(source, key))
        return v

    max_steps = opt_int('max_steps')
    eval_workers = opt_int('eval_workers')

    eval_args = raw.get('eval_args', [])
    if not isinstance(eval_args, list) or any(not isinstance(a, str) for a in eval_args):
        raise JobError('{0}: eval_args must be a list of strings'.format(source))

    priority = raw.get('priority', 100)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise JobError('{0}: priority must be an integer'.format(source))

    return Job(id=jid, type=jtype, policy=policy, env=env, max_steps=max_steps,
               eval_workers=eval_workers, eval_args=eval_args, priority=priority,
               notes=str(raw.get('notes', '')), label=str(raw.get('label', '')))
