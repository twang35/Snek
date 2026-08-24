"""A wave of vectorised evals: `eval_wave.py`'s CLI, sharded across processes and merged.

    PYTHONPATH=. python -u vectorized/vec_wave.py top50 <policy> [<policy> ...]
    PYTHONPATH=. python -u vectorized/vec_wave.py top50 <batch>            # b44 -> its four arms
    PYTHONPATH=. python -u vectorized/vec_wave.py --chain top50 <batch>    # close-out, then HOF

**Every argv token is parsed by `eval_wave`'s own functions**, imported rather than reimplemented, so
the two entry points cannot drift about what `top50`, `above:98`, `--chain` or a bare batch id mean.
`eval_wave` imports no TensorFlow, so this costs nothing.

## Why a supervisor and not a backend inside `eval_wave.py`

`eval_wave` runs `EVAL_LANES` threads and each measures **one checkpoint at a time**
(`pool.run(unit.episodes, ...)`). That shape is right for spawned TF workers and wrong for this
engine, whose speed comes from keeping one wide batch full by streaming many checkpoints through it:
a lane measuring a single 100-episode checkpoint can fill at most 100 of 1024 lanes, so a drop-in
`VecPool` would run the batch at ~10% utilisation and throw away most of the gain. Sharding whole
checkpoint lists across processes reaches the same end `eval_wave`'s lane migration reaches -- no core
idle while any arm still has work -- without either engine adopting the other's control flow.

## Why one process and not N launched by the caller

The desktop's runner tracks a job as a single pid and publishes per arm when it exits, so a job that
fanned out into twelve pids would need the ledger, the reaper and `_publish_results` all changed.
This keeps a job a process.

## What it writes, and where

The **canonical** paths: `runs/<policy>_checkpoint_evals<EVAL_OUT_SUFFIX>.json` and
`evals/<policy>_eval_progress.png`, exactly as `eval_wave` does. `vec_eval.py`'s own defaults
(`_vec`, `evals/vec/`) exist so a hand-run probe can never overwrite a TF result, and they stay as
they are -- but this *is* the close-out now, so it has to land where `eval_progress.best_of`,
`select_checkpoints_above`, `refresh_charts.sh`, the desktop's publish globs and every tuning doc
already look. Both are passed to the children explicitly, so neither tool's behaviour depends on the
other's default.

Nothing is moved out of `evals/` to make room. An arm rewrites its own chart by name, so the folder
is self-correcting; the sweep that used to run at every eval's startup is gone (2026-08-24).

## Every arm comes here, c51 included

This file used to split a wave in two and hand the categorical arms to `eval_wave.py`, because
`vec_eval` refused them. **It no longer refuses** (2026-08-24) -- and the interesting part is that no
atom arithmetic had to be written for that, because the engine never reads a Q head. It builds through
`eval_agent.build_eval_agent`, which selects the agent class from `arch.json`, and asks
`policy.action(...)`; a categorical agent's greedy policy reduces over its own support internally. So
the split, the fallback and the "which engine can measure this arm" question are all gone, and
`vec_wave` is now the one close-out path.

Validated the same way the scalar comparison was: six `b38a-c51fc320eps3125seed1` checkpoints spanning
35-96%, 200 episodes per checkpoint per engine, pooled **78.33% (vec) against 78.50% (scalar)** --
**-0.17 pp, z = -0.10**. That is the same agreement as the ddqn head-to-head (-0.06 pp, z = -0.28).
"""

import contextlib
import io
import json
import os
import subprocess
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import chart_viewer
import eval_plan
import eval_wave
from snake_constants import EVALS_DIR, POLICY_DIR, RUNS_DIR

# **Cores minus two.** One process saturates about one core -- the observation build is
# single-threaded numpy and is most of a step -- so this tracks core count and nothing about the
# model, which is why it is derived rather than a constant that would be wrong on the other host.
#
# The measured point on the laptop's 14 (10 performance + 4 efficiency): **12 processes x width 1024
# held 347-350 episodes/s at 2-6% CPU idle**, against 168 at 4 processes (59% idle) and 280 at 16 --
# 0% idle and therefore *slower*, which is the whole reason for the -2. The two spare cores are what
# the parent, the chart writer and the OS run in; taking them is what tipped 16 into oversubscription.
DEFAULT_PROCS = max(1, (os.cpu_count() or 14) - 2)

VEC_EVAL = os.path.join('vectorized', 'vec_eval.py')


def selector_string(kind, value):
    """`eval_wave`'s parsed `(kind, value)` back into the one argument `vec_eval.py` takes.

    Reassembled rather than passed through, because `vec_eval` takes the selection as a **single**
    argv token: handing it `above:98` as two tokens, or a step list as many, has it measure one
    checkpoint and report a short selection rather than an error.
    """
    if kind == 'top':
        return 'top{0}'.format(value)
    if kind == 'above':
        return 'above:{0:g}'.format(value)
    return ','.join(str(step) for step in value)


def plan_shards(sizes, procs):
    """`{arm: shard_count}`, allocated in proportion to each arm's selection size.

    Proportional rather than equal, because a wave's arms differ by an order of magnitude -- b45's
    HOF selections were 1568 / 1264 / 1173 / 298 -- and equal shares would leave the small arm's
    processes finished and its cores idle for most of the run. Largest-remainder, so the rounding
    error is spread rather than all landing on one arm.

    **Every arm with work gets at least one shard, and that floor can push the total above `procs`**
    when an arm's share is under one process's worth (b45's 298 against 12 processes is not, but a
    30-checkpoint arm beside a 3000-checkpoint one would be). The overshoot is bounded by the arm
    count and is the right failure: dropping an arm to hold a process budget would silently not
    measure it.
    """
    live = {arm: count for arm, count in sizes.items() if count > 0}
    if not live:
        return {}
    procs = max(len(live), int(procs))
    total = sum(live.values())
    exact = {arm: procs * count / float(total) for arm, count in live.items()}
    # Never more shards than checkpoints -- an empty shard is a wasted process, and `vec_eval` exits
    # cleanly on one, so the loss would be invisible.
    out = {arm: max(1, min(int(exact[arm]), live[arm])) for arm in live}
    order = sorted(live, key=lambda arm: (-(exact[arm] % 1), -live[arm], arm))
    index = 0
    while sum(out.values()) < procs and any(out[arm] < live[arm] for arm in live):
        arm = order[index % len(order)]
        if out[arm] < live[arm]:
            out[arm] += 1
        index += 1
    return out


def stitch_payload(policy, suffix, pieces):
    """Give the merged file the top-level fields `merge_checkpoint_evals` does not carry.

    **The merge is a row combiner, not a payload builder.** It writes seven keys -- policy, episodes
    per checkpoint, requested count, `complete`, `merged_from`, `incomplete_sources`, `results` -- and
    drops every protocol and progress field `build_payload` emits. That is fine for the ad-hoc "put my
    four hand-run pieces together" job it was written for, and not fine here: it would make an arm's
    close-out file depend on **how many processes measured it**, so a one-shard arm and a
    twelve-shard arm in the same wave would publish different-shaped files, and `requested_steps` --
    the record of what the arm's selection actually was -- would exist for one and not the other.

    So the payload is rebuilt through `eval_plan.build_payload`, the one definition, with the rows the
    merge produced and the protocol read back out of the shards rather than restated here. A field
    added to `build_payload` therefore reaches a merged file for free, which is the property that
    keeps this from becoming the third copy of the schema.
    """
    shards = []
    for piece in pieces:
        path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy, piece))
        with open(path) as handle:
            shards.append(json.load(handle))
    out_path = os.path.join(RUNS_DIR, '{0}_checkpoint_evals{1}.json'.format(policy, suffix))
    with open(out_path) as handle:
        merged = json.load(handle)

    def summed(key):
        return sum(shard.get(key) or 0 for shard in shards)

    first = shards[0]
    spec = eval_plan.PayloadSpec(
        policy_name=policy,
        num_episodes=first.get('episodes_per_checkpoint'),
        # The union of the shards' own selections, which reconstructs the arm's selection exactly:
        # `shard_steps` is a stride of one list, so the shards partition it with nothing shared and
        # nothing dropped.
        all_steps=sorted({step for shard in shards for step in shard.get('requested_steps') or ()}),
        # Read back, not restated: whether this engine screens, gates or rounds its episodes is
        # `vec_eval`'s to say, and a literal here would be a second opinion that could go stale.
        num_workers=first.get('num_workers'),
        screen_episodes=first.get('screen_episodes'),
        confirm_count=first.get('confirm_count'),
        min_achievable=first.get('min_achievable'),
        abandon_floor=first.get('abandon_floor'),
        measurements_planned=summed('measurements_planned'),
        episodes_planned=summed('episodes_planned'),
        full_planned=summed('measurements_planned'), screen_planned=0, confirm_planned=0)
    progress = {'measurements': len(merged['results']),
                'session_measurements': summed('session_measurements'),
                'session_episodes': summed('session_episodes'),
                # The wall clock the *wave* spent, not the sum of the shards' -- they ran at the same
                # time, so summing would report twelve times the elapsed time as if it were serial.
                'session_seconds': max([shard.get('session_seconds') or 0 for shard in shards]),
                'episodes_saved': summed('episodes_saved'), 'abandoned': summed('abandoned'),
                'stage': 'full', 'full_done': len(merged['results']),
                'screen_done': 0, 'confirm_done': 0,
                # A finished wave has no ETA and no lane in flight. `wave_lanes` is per shard, so the
                # arm's figure is the width times the shards that measured it.
                'wave_lanes': (first.get('wave_lanes') or 0) * len(shards) or None,
                'wave_arms': None}
    payload = eval_plan.build_payload(spec, progress, {}, merged['results'],
                                      complete=merged['complete'])
    # Provenance the rebuild cannot know, kept from the merge: which files this arm was assembled
    # from, and which of them were short. Without them a truncated shard is invisible in the result.
    payload['merged_from'] = merged.get('merged_from')
    payload['incomplete_sources'] = merged.get('incomplete_sources')
    eval_plan.write_payload(out_path, payload)
    return payload

def selection_size(policy, selector, source_suffix):
    """How many checkpoints `selector` picks for `policy` -- the weight in the shard plan.

    `vec_eval` is imported lazily because it pulls in TensorFlow, and the import is worth a good three
    seconds that a wave which exits early -- nothing selected, a bad selector -- should not pay. A
    failure that is not a `SystemExit` returns 1 rather than propagating: the shard plan is an
    optimisation, and an arm whose size cannot be read still has to be measured.
    """
    from vectorized import vec_eval
    try:
        # Sized silently. `select_checkpoints_above` prints its candidate list, and every shard prints
        # the same list again when it resolves the selection for itself -- so letting the parent print
        # too buries the wave's own four lines under N+1 copies of a listing that says nothing new.
        with contextlib.redirect_stdout(io.StringIO()):
            steps, _ = vec_eval.resolve_selection(
                policy, os.path.join(POLICY_DIR, policy), selector, source_suffix)
        return len(steps)
    except SystemExit:
        raise
    except Exception as error:
        print('    could not size {0} ({1}: {2}) - assuming one shard'.format(
            policy, type(error).__name__, error))
        return 1


def run_stage(title, policies, selector, episodes, suffix, source_suffix, procs):
    """Spawn one stage's shards, wait for all of them, then merge each arm's pieces.

    Returns an exit code. Every shard is waited on even after one fails, because a surviving sibling
    still holds a slice of the arm's checkpoints and killing the wave would throw away work that is
    already paid for -- the merge simply reports what is missing.
    """
    print('\n{0}'.format(title))
    sizes = {policy: selection_size(policy, selector, source_suffix) for policy in policies}
    for policy in policies:
        print('    {0}: {1} checkpoints selected'.format(policy, sizes[policy]))
    shards = plan_shards(sizes, procs)
    if not shards:
        print('    nothing selected for any arm - nothing to measure')
        return 0
    print('    {0} processes: {1}'.format(
        sum(shards.values()),
        ', '.join('{0}x{1}'.format(shards[arm], arm) for arm in sorted(shards))))

    children, failed = [], []
    for policy in sorted(shards):
        count = shards[policy]
        for index in range(1, count + 1):
            env = child_env(episodes, suffix, source_suffix)
            if count > 1:
                env['VEC_EVAL_SHARD'] = '{0}/{1}'.format(index, count)
            children.append((policy, index, count, subprocess.Popen(
                [sys.executable, '-u', VEC_EVAL, policy, selector], env=env)))
    for policy, index, count, child in children:
        if child.wait() != 0:
            failed.append('{0} shard {1}/{2} exited {3}'.format(
                policy, index, count, child.returncode))

    for policy in sorted(shards):
        count = shards[policy]
        if count == 1:
            continue                    # a single shard writes the arm's file directly
        pieces = ['{0}-s{1}of{2}'.format(suffix, i, count) for i in range(1, count + 1)]
        try:
            eval_plan.merge_checkpoint_evals(policy, suffixes=pieces, out_suffix=suffix)
            payload = stitch_payload(policy, suffix, pieces)
            print('    merged {0} shards into {1}_checkpoint_evals{2}.json ({3} checkpoints, '
                  '{4} episodes, complete={5})'.format(
                      count, policy, suffix or '', len(payload['results']),
                      payload['episodes_done'], payload['complete']))
        except SystemExit as error:
            failed.append('merge {0}: {1}'.format(policy, error))
    if failed:
        print('\n{0} failure(s) in this stage:'.format(len(failed)))
        for line in failed:
            print('    ' + line)
        return 1
    return 0


def child_env(episodes, suffix, source_suffix):
    """The environment for one `vec_eval.py` shard.

    Three groups of settings are handled here rather than left to inheritance, and each one has bitten
    something in this project already:

    * **The output paths are forced to the canonical ones.** `vec_eval`'s `_vec` / `evals/vec/`
      defaults are for a hand-run probe; a close-out has to land where every reader looks.
    * **The scalar protocol's knobs are dropped.** There is no screen tier and no abandon gate here,
      so `EVAL_MIN_ACHIEVABLE` and friends have nothing to act on -- and a value silently ignored is
      how someone concludes a wave was gated when it was not. `EVAL_WORKERS`/`EVAL_LANES` go too:
      they size TF worker processes, and this engine has none.
    * **The children never open a window.** One owner per wave, spawned by this process; twelve
      claimants racing one lock is the documented way to end up with two windows.
    """
    env = dict(os.environ)
    for key in ('EVAL_SCREEN_EPISODES', 'EVAL_MIN_ACHIEVABLE', 'EVAL_ABANDON_FLOOR',
                'EVAL_CONFIRM_COUNT', 'EVAL_WORKERS', 'EVAL_LANES', 'VEC_EVAL_SHARD'):
        env.pop(key, None)
    env['VEC_EVAL_EPISODES'] = str(episodes)
    env['EVAL_OUT_SUFFIX'] = suffix
    env['VEC_EVAL_SOURCE'] = source_suffix
    env['VEC_EVAL_CHART_DIR'] = EVALS_DIR
    env['SNEK_CHART_VIEWER'] = '0'
    return env


def main(argv):
    chain, tokens = eval_wave.parse_options(list(argv[1:]))
    kind, value, rest = eval_wave.parse_selector(tokens)
    policies = eval_wave.resolve_policies(rest)
    if not policies:
        raise SystemExit('no policies given')
    selector = selector_string(kind, value)
    procs = int(os.environ.get('VEC_WAVE_PROCS', DEFAULT_PROCS))
    episodes = int(os.environ.get('EVAL_EPISODES', 100))
    suffix = os.environ.get('EVAL_OUT_SUFFIX', '')

    if sys.platform == 'darwin':
        # HiDPI: chart_viewer only magnifies the PNG, so 110 dpi looks soft on a Retina panel.
        os.environ.setdefault('SNEK_EVAL_CHART_DPI', '220')
        try:
            # `vec_eval.py <policy>` puts a policy name right after the script on every shard's
            # command line, so the prefix pattern matches all twelve -- the window stays up until the
            # last shard of the last arm exits. Matching `vec_wave.py` instead would be one process
            # and would close the window on a stage boundary.
            chart_viewer.spawn_for_eval(policies[0], watch='vec_eval.py {prefix}')
        except Exception as error:
            print('chart viewer skipped ({0}: {1})'.format(type(error).__name__, error))

    print('vectorised wave: {0} arm(s), {1}, {2} episodes each, flat, no abandon gate'.format(
        len(policies), selector, episodes))
    code = run_stage('stage A: {0}'.format(eval_wave.describe_selector(kind, value)),
                     policies, selector, episodes, suffix, '', procs)
    if chain:
        # `hof_settings` is the only definition of the stage-B recipe -- 500 episodes, `above:98` out
        # of stage A's own file, into `_hof500`. Derived from stage A's settings rather than written
        # out here, so the two cannot disagree about which file the candidates come from.
        recipe = eval_plan.hof_settings({'suffix': suffix})
        ready = eval_wave.completed_policies(policies, suffix)
        if not ready:
            print('\nno arm produced a complete close-out - nothing to re-measure')
        else:
            hof_code = run_stage(
                'stage B: above:{0:g} at {1} episodes, flat, into {2}'.format(
                    eval_plan.HOF_GATE, eval_plan.HOF_EPISODES,
                    recipe['suffix'] or '<no suffix>'),
                ready, 'above:{0:g}'.format(eval_plan.HOF_GATE), recipe['num_episodes'],
                recipe['suffix'], recipe['source_suffix'], procs)
            code = code or hof_code
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
