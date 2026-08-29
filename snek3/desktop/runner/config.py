"""The runner's two config tiers.

| tier | where | changed | holds |
|---|---|---|---|
| `host.env` | on the box, set once at setup | never at runtime | machine identity — paths, branches — and the **hard** ceilings |
| `runtime.json` | tracked on the `ops` branch | re-read every network cycle | the live knobs: concurrency, threads, poll interval, pause/drain |

**A malformed `runtime.json` is rejected whole and the last known-good config is kept.** In normal
operation the box has no ssh backstop, so a bad commit must never be able to wedge or crash the
daemon — and a *partially* applied config is worse than a rejected one, because it looks like it
worked. Values are then clamped to the host ceilings, so no commit can push the box past
`HARD_MAX_*` or below the poll floor.

## The binding constraint moved from memory to cores

snek2's ceiling arithmetic was about memory: a spawned TF eval worker carried its own ~230 MB arena,
`max_evals x eval_lanes x eval_workers` all multiplied, and 15,030 MB put the OOM cliff near 40
workers. **None of that carries.** Measured 2026-08-28 on this code:

| process | peak RSS | of which torch's import |
|---|---:|---:|
| one eval shard, width 1024, 100 episodes | **202 MB** | 193 MB |
| one trainer, 100k-transition buffer, 3k steps | **290 MB** | 191 MB |

A full box — 4 trainers and 16 eval shards — is **4.4 GB of 15,030**, 29%. The per-process cost is
almost entirely a fixed torch import; the actual work is 8 MB for a shard. So there is one ceiling
here rather than snek2's three-way clamp, and it is set by **threads, not bytes**: the box is a Ryzen
7 9700X with 8 physical cores and 16 SMT threads, its measured vec-eval optimum is 16 shards at one
intra-op thread each, and 18 shards loses 6-10%.

Which is also why `torch_threads` and `omp_num_threads` default to **1** rather than 0. Sixteen
shards each taking a thread per core is a 16x oversubscription, and the nets here are far too small
for a fork-join to pay for itself anyway — measured 1.4x *slower* at torch's default.
"""

import json

# Used until a valid `runtime.json` is read, and to fill in any key a valid one omits.
RUNTIME_DEFAULTS = {
    'max_trainers': 2,          # concurrent train/smoke/benchmark jobs
    'max_evals': 1,             # concurrent eval jobs. A job is a whole wave, so 1 is normal
    # Shard processes per eval job. This is the knob that fills the box, not `max_evals`: one wave
    # owns every arm of a batch and its shards take whichever checkpoint is next regardless of which
    # arm it belongs to, so widening the wave beats running two.
    'eval_shards': 16,
    # How often the loop runs its LOCAL half: reap, read the already-fetched `ops` ref, dispatch.
    # Cheap and off-network, so it stays fast — a close-out queued by a training that just finished
    # should not wait for the next git cycle to launch.
    'poll_seconds': 30,
    # How often the loop runs its NETWORK half: one fetch of the three bus branches, one status
    # push, and a retry of any local-only commit. Separate from `poll_seconds` because at 30 s the
    # box made ~2,880 fetches and ~2,880 pushes to github a day — enough sustained machine-shaped
    # traffic from a home connection to be worth not making. 600 s cuts both to 144 while costing
    # nothing locally, and `trigger` forces a cycle when a batch should start now rather than within
    # ten minutes. **So a `status.json` up to 10 minutes old is a healthy daemon.** 0 restores one
    # network cycle per poll.
    'git_seconds': 600,
    'torch_threads': 1,         # SNEK_TORCH_THREADS. See the header: 1 is measured, not cautious
    'omp_num_threads': 1,       # oneDNN/BLAS. Same reason
    'nice': 0,
    'disk_min_gb': 5,           # refuse to launch below this much free space
    'paused': False,            # finish running jobs, start nothing new
    'drain': False,             # alias of paused, kept separate for intent
    # A finished training auto-queues its stage-B wave. Spelled for the protocol snek3 actually has:
    # snek2's `auto_closeout` + `auto_hof` were two hops of a tiered close-out, and there is one
    # stage now, so carrying the old names would be archaeology rather than continuity.
    'auto_stage_b': True,
}

_INT_KEYS = ('max_trainers', 'max_evals', 'eval_shards', 'poll_seconds', 'git_seconds',
             'torch_threads', 'omp_num_threads', 'nice', 'disk_min_gb')
_BOOL_KEYS = ('paused', 'drain', 'auto_stage_b')

_REQUIRED_HOST = ('REPO_PATH', 'SNEK_DIR', 'PYTHON_BIN', 'GIT_REMOTE',
                  'OPS_BRANCH', 'STATUS_BRANCH', 'RESULTS_BRANCH',
                  'STATUS_WORKTREE', 'RESULTS_WORKTREE', 'LEDGER_PATH', 'LOG_DIR',
                  'QUEUE_DIR', 'RUNTIME_PATH',
                  'HARD_MAX_TRAINERS', 'HARD_MAX_EVALS', 'HARD_MAX_EVAL_SHARDS',
                  'MIN_POLL_SECONDS')

_HOST_INT_KEYS = ('HARD_MAX_TRAINERS', 'HARD_MAX_EVALS', 'HARD_MAX_EVAL_SHARDS',
                  'MIN_POLL_SECONDS')


class ConfigError(Exception):
    """`host.env` is unusable. Fatal at startup, because there is nothing to fall back to."""


def load_host_config(path):
    """Reads `host.env` — `KEY=VALUE` lines, `#` comments — with the ceilings coerced to int."""
    config = {}
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ConfigError('bad host.env line (no "="): ' + raw.rstrip())
            key, _, value = line.partition('=')
            config[key.strip()] = value.strip()
    missing = [key for key in _REQUIRED_HOST if key not in config]
    if missing:
        raise ConfigError('host.env missing keys: ' + ', '.join(missing))
    for key in _HOST_INT_KEYS:
        try:
            config[key] = int(config[key])
        except ValueError:
            raise ConfigError('{0} must be an integer, got {1!r}'.format(key, config[key]))
    return config


def clamp_runtime(config, host):
    """Clamps a runtime dict in place. Returns notes for anything clamped.

    Notes rather than errors: a request for 10 trainers is *honoured* as `HARD_MAX_TRAINERS` and
    noted in `status.json`, because refusing the whole file over an ambitious number would be a
    worse failure than running fewer arms than asked.
    """
    notes = []

    def clamp(key, low, high):
        value = config[key]
        clamped = max(low, min(high, value))
        if clamped != value:
            notes.append('{0} {1} clamped to {2}'.format(key, value, clamped))
        config[key] = clamped

    clamp('max_trainers', 0, host['HARD_MAX_TRAINERS'])
    clamp('max_evals', 0, host['HARD_MAX_EVALS'])
    clamp('eval_shards', 1, host['HARD_MAX_EVAL_SHARDS'])
    clamp('poll_seconds', host['MIN_POLL_SECONDS'], 3600)
    # Not floored at MIN_POLL_SECONDS: 0 is the meaningful opt-out — a network cycle every poll,
    # which is what the daemon did before this knob existed. Ceiling is a day.
    clamp('git_seconds', 0, 86400)
    clamp('torch_threads', 1, 64)
    clamp('omp_num_threads', 1, 64)
    clamp('nice', 0, 19)
    clamp('disk_min_gb', 0, 100000)
    notes += clamp_total_shards(config, host)
    return notes


def clamp_total_shards(config, host):
    """Holds `max_evals x eval_shards` at or under `HARD_MAX_EVAL_SHARDS`.

    **Both multiply, and this is the whole ceiling** — snek2 needed a three-way version of this
    because a worker cost 230 MB of TensorFlow arena and memory ran out first. Here 16 shards is
    3.2 GB of 15, so the limit is the box's 16 SMT threads: past `cpu_count` throughput *falls*, and
    two waves of 16 is a 2x oversubscription that makes both slower than one.

    `eval_shards` gives way, never `max_evals`: how many waves run is a scheduling decision, and
    quietly running fewer than asked would be a surprising way to honour a thread limit. A narrower
    wave merely measures more slowly.
    """
    ceiling = host['HARD_MAX_EVAL_SHARDS']
    jobs = max(1, config['max_evals'])
    total = jobs * config['eval_shards']
    if total <= ceiling:
        return []
    reduced = max(1, ceiling // jobs)
    note = ('max_evals {0} x eval_shards {1} = {2} shard processes exceeds {3}; reduced to {4} '
            'shards'.format(config['max_evals'], config['eval_shards'], total, ceiling, reduced))
    config['eval_shards'] = reduced
    return [note]


def parse_runtime_config(text, host):
    """Parses `runtime.json` and clamps it.

    `(config, notes)` on success, or `(None, errors)` when the text is not a trustworthy config. The
    caller keeps its last known-good config whenever the first element is None, and surfaces the
    errors in `status.json` — which is the only channel a box with no ssh has.
    """
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as error:
        return None, ['runtime.json is not valid JSON: {0}'.format(error)]
    if not isinstance(raw, dict):
        return None, ['runtime.json must be a JSON object']

    errors = []
    config = dict(RUNTIME_DEFAULTS)
    for key, value in raw.items():
        if key not in RUNTIME_DEFAULTS:
            # Rejected rather than ignored, because an unknown key is usually a *renamed* one and
            # ignoring it would silently run the default while the file says otherwise. The cost is
            # that renaming a key here requires the deploy to land before the `ops` edit.
            errors.append('unknown key: {0}'.format(key))
            continue
        if key in _BOOL_KEYS and not isinstance(value, bool):
            errors.append('{0} must be true/false'.format(key))
        elif key in _INT_KEYS and (isinstance(value, bool) or not isinstance(value, int)):
            errors.append('{0} must be an integer'.format(key))
        else:
            config[key] = value
    if errors:
        # One bad field means the file is not trusted at all.
        return None, errors

    return config, clamp_runtime(config, host)
