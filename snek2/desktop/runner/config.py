"""The runner's two config tiers.

`host.env` -- set once at setup, holds machine identity (paths, git branches) and
the HARD safety ceilings. Never changed at runtime.

`runtime.json` -- tracked on the `ops` branch, re-read every poll. Holds the knobs
you tune live (concurrency, threads, poll interval, pause/drain). A malformed
`runtime.json` is rejected and the last-known-good config is kept, because in
normal operation the box has no SSH backstop -- a bad commit must never be able to
wedge or crash the daemon. Values are also clamped to the host ceilings, so no
commit can push the box past `HARD_MAX_*` or below the poll floor.
"""
import json

# Used until a valid runtime.json is read, and to fill in any missing keys.
RUNTIME_DEFAULTS = {
    'max_trainers': 2,        # concurrent train/smoke/benchmark jobs
    'max_evals': 1,           # concurrent eval jobs
    'eval_workers': 10,       # EVAL_WORKERS per eval job
    'poll_seconds': 30,
    'tf_intraop_threads': 0,  # 0 = leave TensorFlow's default
    'omp_num_threads': 0,     # 0 = leave oneDNN's default
    'nice': 0,
    'disk_min_gb': 5,         # refuse to launch below this much free space
    'paused': False,          # finish running jobs, start nothing new
    'drain': False,           # alias of paused, kept separate for intent
    'viewer': True,           # keep a decoupled chart viewer up while trainers run
    'auto_closeout': True,    # a finished training auto-queues its closeout eval (runs next)
}

_INT_KEYS = ('max_trainers', 'max_evals', 'eval_workers', 'poll_seconds',
             'tf_intraop_threads', 'omp_num_threads', 'nice', 'disk_min_gb')
_BOOL_KEYS = ('paused', 'drain', 'viewer', 'auto_closeout')

_REQUIRED_HOST = ('REPO_PATH', 'SNEK_DIR', 'PYTHON_BIN', 'GIT_REMOTE',
                  'OPS_BRANCH', 'STATUS_BRANCH', 'RESULTS_BRANCH',
                  'STATUS_WORKTREE', 'RESULTS_WORKTREE', 'LEDGER_PATH', 'LOG_DIR',
                  'HARD_MAX_TRAINERS', 'HARD_MAX_EVALS', 'MIN_POLL_SECONDS')


class ConfigError(Exception):
    pass


def load_host_config(path):
    """Reads host.env (KEY=VALUE lines, `#` comments) into a dict, with the
    numeric ceilings coerced to int. Raises ConfigError on a missing key."""
    cfg = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ConfigError('bad host.env line (no "="): ' + raw.rstrip())
            key, _, val = line.partition('=')
            cfg[key.strip()] = val.strip()
    missing = [k for k in _REQUIRED_HOST if k not in cfg]
    if missing:
        raise ConfigError('host.env missing keys: ' + ', '.join(missing))
    for k in ('HARD_MAX_TRAINERS', 'HARD_MAX_EVALS', 'MIN_POLL_SECONDS'):
        try:
            cfg[k] = int(cfg[k])
        except ValueError:
            raise ConfigError('{0} must be an integer, got {1!r}'.format(k, cfg[k]))
    return cfg


def clamp_runtime(cfg, host):
    """Clamps a runtime dict in place to the host ceilings. Returns a list of
    human-readable notes for any value that was clamped (not an error -- a
    request for 10 trainers is honoured as HARD_MAX_TRAINERS and noted)."""
    notes = []

    def clamp(key, lo, hi):
        v = cfg[key]
        nv = max(lo, min(hi, v))
        if nv != v:
            notes.append('{0} {1} clamped to {2}'.format(key, v, nv))
        cfg[key] = nv

    clamp('max_trainers', 0, host['HARD_MAX_TRAINERS'])
    clamp('max_evals', 0, host['HARD_MAX_EVALS'])
    clamp('eval_workers', 1, 64)
    clamp('poll_seconds', host['MIN_POLL_SECONDS'], 3600)
    clamp('tf_intraop_threads', 0, 64)
    clamp('omp_num_threads', 0, 64)
    clamp('nice', 0, 19)
    clamp('disk_min_gb', 0, 100000)
    return notes


def parse_runtime_config(text, host):
    """Parses runtime.json and clamps it.

    Returns (config, notes) on success, or (None, errors) when the text is not a
    trustworthy config. The caller keeps its last-known-good config whenever the
    first element is None and surfaces the errors in status.json.
    """
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as e:
        return None, ['runtime.json is not valid JSON: {0}'.format(e)]
    if not isinstance(raw, dict):
        return None, ['runtime.json must be a JSON object']

    errors = []
    cfg = dict(RUNTIME_DEFAULTS)
    for key, val in raw.items():
        if key not in RUNTIME_DEFAULTS:
            errors.append('unknown key: {0}'.format(key))
            continue
        if key in _BOOL_KEYS and not isinstance(val, bool):
            errors.append('{0} must be true/false'.format(key))
        elif key in _INT_KEYS and (isinstance(val, bool) or not isinstance(val, int)):
            errors.append('{0} must be an integer'.format(key))
        else:
            cfg[key] = val
    if errors:
        # A single bad field means we do not trust the file at all.
        return None, errors

    notes = clamp_runtime(cfg, host)
    return cfg, notes
