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
    'eval_workers': 4,        # EVAL_WORKERS per eval lane
    # EVAL_LANES per eval job. An eval job is now a *wave*: one `eval_wave.py` process owning
    # every arm of a batch, with `eval_lanes` worker pools taking whichever checkpoint is next
    # regardless of which arm it belongs to. So `max_evals` normally reads 1 and this is the knob
    # that fills the box. 4 x 4 is the measured throughput point (~12.7 of 14 cores busy).
    'eval_lanes': 4,
    'poll_seconds': 30,
    'tf_intraop_threads': 0,  # 0 = leave TensorFlow's default
    'omp_num_threads': 0,     # 0 = leave oneDNN's default
    'nice': 0,
    'disk_min_gb': 5,         # refuse to launch below this much free space
    'paused': False,          # finish running jobs, start nothing new
    'drain': False,           # alias of paused, kept separate for intent
    'viewer': True,           # keep a decoupled chart viewer up while trainers run
    'auto_closeout': True,    # a finished training auto-queues its closeout eval (runs next)
    # A closeout wave also runs the 500-episode HOF re-measure of its >=98% checkpoints, in the same
    # process, via `eval_wave.py --chain`. Deliberately still spelled `auto_hof` although the
    # mechanism is now a chained stage rather than a queued job: `parse_runtime_config` rejects the
    # whole file on an unknown key and keeps the last-known-good config, so renaming it would make a
    # deploy that landed before the `ops` edit reject every config until someone noticed.
    'auto_hof': True,
    # Which engine measures a checkpoint. `vec` is `vectorized/vec_wave.py`, which runs one wide
    # numpy env serving many checkpoints at once and measures ~40x faster than the TF path; `scalar`
    # is `eval_wave.py`, spawned TF workers, one board each. The default moved to `vec` on
    # 2026-08-24, after a 24-checkpoint x 500-episode head-to-head agreed to -0.058 pp (z = -0.28).
    # Kept as a knob because it is the only way to reproduce a pre-switch measurement, and because a
    # regression in the new engine has to be answerable without a deploy.
    'eval_engine': 'vec',
    # `VEC_WAVE_PROCS` for a vec wave: how many `vec_eval.py` shards fill the box. 0 leaves it to
    # `vec_wave.DEFAULT_PROCS`, which is cores minus two -- derived rather than pinned, because one
    # shard saturates about one core and the two hosts do not have the same number of them.
    'vec_wave_procs': 0,
}

_INT_KEYS = ('max_trainers', 'max_evals', 'eval_workers', 'eval_lanes', 'poll_seconds',
             'tf_intraop_threads', 'omp_num_threads', 'nice', 'disk_min_gb', 'vec_wave_procs')
_BOOL_KEYS = ('paused', 'drain', 'viewer', 'auto_closeout', 'auto_hof')
# Keys whose value must be one of a fixed set. A typo here would otherwise reach `build_command` and
# take down every eval dispatch with a ValueError, one job at a time, instead of being rejected with
# the rest of the file.
_ENUM_KEYS = {'eval_engine': ('vec', 'scalar')}

_REQUIRED_HOST = ('REPO_PATH', 'SNEK_DIR', 'PYTHON_BIN', 'GIT_REMOTE',
                  'OPS_BRANCH', 'STATUS_BRANCH', 'RESULTS_BRANCH',
                  'STATUS_WORKTREE', 'RESULTS_WORKTREE', 'LEDGER_PATH', 'LOG_DIR',
                  'HARD_MAX_TRAINERS', 'HARD_MAX_EVALS', 'MIN_POLL_SECONDS')


# The most spawned eval workers this box may hold at once, across every lane of every eval job.
# See the clamp in `clamp_runtime` for the memory arithmetic behind the number.
MAX_EVAL_WORKERS = 32


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
    clamp('eval_lanes', 1, host['HARD_MAX_EVALS'])
    clamp('poll_seconds', host['MIN_POLL_SECONDS'], 3600)
    clamp('tf_intraop_threads', 0, 64)
    clamp('omp_num_threads', 0, 64)
    clamp('nice', 0, 19)
    clamp('disk_min_gb', 0, 100000)
    notes += clamp_eval_workers(cfg)
    return notes


def clamp_eval_workers(cfg):
    """Holds `max_evals x eval_lanes x eval_workers` at or under `MAX_EVAL_WORKERS`.

    **All three multiply, and forgetting `max_evals` is how this box gets OOM-killed.** An eval job
    used to be one policy with `eval_workers` behind it; it is now a *wave* with `eval_lanes` pools
    of `eval_workers` each, so a `max_evals` of 4 that used to mean 16 spawned workers means 64.
    Memory, not cores, is the binding constraint: a spawned worker carries its own TensorFlow arena
    at ~230 MB, and 15,030 MB total puts the OOM ceiling near 40 -- so 32 is the operating band,
    chosen to keep >=3 GB of headroom.

    `eval_workers` gives way first and `eval_lanes` only if that is not enough, because a lane that
    does not exist cannot pick up another arm's work -- which is the entire point of a wave -- while
    a lane with fewer workers merely measures more slowly. `max_evals` is never touched: it is a
    scheduling decision, and quietly running fewer waves than asked would be a surprising way to
    honour a memory limit.
    """
    def total(lanes, workers):
        return max(1, cfg['max_evals']) * lanes * workers

    lanes, workers = cfg['eval_lanes'], cfg['eval_workers']
    if total(lanes, workers) <= MAX_EVAL_WORKERS:
        return []
    per_job = max(1, MAX_EVAL_WORKERS // max(1, cfg['max_evals']))
    cfg['eval_workers'] = max(1, per_job // lanes)
    if total(lanes, cfg['eval_workers']) > MAX_EVAL_WORKERS:
        cfg['eval_lanes'] = max(1, per_job // cfg['eval_workers'])
    return ['max_evals {0} x eval_lanes {1} x eval_workers {2} = {3} spawned workers exceeds {4}; '
            'reduced to {5} lanes x {6} workers'.format(
                cfg['max_evals'], lanes, workers, total(lanes, workers), MAX_EVAL_WORKERS,
                cfg['eval_lanes'], cfg['eval_workers'])]


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
        if key in _ENUM_KEYS and val not in _ENUM_KEYS[key]:
            errors.append('{0} must be one of {1}'.format(key, ' / '.join(_ENUM_KEYS[key])))
        elif key in _BOOL_KEYS and not isinstance(val, bool):
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
