"""A batch's stage-B pass: every arm of a wave, its shards pooled, each arm merged as it ends.

    PYTHONPATH=. python -u -m tools.closeout b6a b6b b6c --shards 12                  # stage B
    PYTHONPATH=. python -u -m tools.closeout b6a b6b b6c --pass hof5000 --shards 12   # its follow-on

**This is the close-out, and both boxes run this file.** The desktop's daemon dispatches one `eval`
job per batch and that job is this command; on the laptop the same command is typed by hand. Before
2026-08-30 the desktop built a `sh -c` string that chained one `evaluate.py` per arm and the laptop
used whatever an agent had written that day — two sequencers, one of which was a scratch shell script
whose `| tail` swallowed the wave's progress lines. There is one now.

**The shards are pooled across the arms** (2026-09-04; before that the arms went one at a time, the
2026-08-29 decision, each arm a full-width wave). `--shards` is the pool: every arm's candidates are
resolved up front, each arm gets `min(candidates, pool)` shards, and the close-out keeps the pool full
in arm order — when an arm's shards start draining, the next arm's start. Per-arm waves were fine for
stage B, hundreds of candidates per arm, and cost only an ~11 s lane drain per arm; they were the
bottleneck for the hof passes, where an arm has 0-30 candidates: a hof30k wave was one busy core and
eleven shards exiting at once, arm after arm, ~40 min for eight arms that pooled take ~6. Each arm is
still merged on its own, under the same refusals, and an arm's shard count depends on its candidates
alone, so a rerun resumes every shard from its own file.

**A failing arm never drops the arms behind it.** Every arm runs; the exit status is the last
failure's. That is the `;`-not-`&&` rule of the shell version it replaces, and the snek2 incident
behind it is a wave that lost a whole batch's measurement to one bad arm. An arm whose selector
raises — no checkpoints, no upstream file — is reported and skipped the same way.

**Progress is visible three ways**, and none of them is this file doing work per episode: the wave's
own `[ 2167/3222] 4 shard(s) alive, 9m elapsed  eta 5m` line, each arm's stage-B PNG — redrawn here
off the shard files while the wave runs — and the box's chart window, which the **scheduler** points at
those PNGs while this pass runs (2026-09-05; this file opened its own window before that, and
`tools/window.py` says why it no longer does). A close-out typed by hand gets no window; queue it as an
eval spec instead.
"""

import argparse
import os
import sys
import threading
import time

from tools import eval_wave
from tools import stage_b_chart

# How often an arm's PNG is rebuilt from its shard files. A shard writes a row every ~10 s and the
# window redraws on its own clock; faster than this is CPU taken off the measurement.
REDRAW_SECONDS = 20

# **The protocol's passes, by name — the one place their numbers live.** Every batch gets all three,
# in this order, and each selects from the pass before it: stage B screens every checkpoint at
# >=97/100 in stage A and measures it at 500; `hof5000` takes the rows at >=99/500 to 5,000; `hof30k`
# takes the rows at >=99/5,000 to 30,000 on seed 7, a seed no selecting pass used, so a row there is a
# confirmed rate rather than a selected high. The labels are the files: `runs/<arm>_checkpoint_evals
# [_<label>].json`, so a pass never overwrites the one it read from — omitting `hof5000`'s label
# would replace the 500-episode file `above:99` selects from with 5,000-episode rows.
#
# `--pass <name>` is how both boxes ask for one: the desktop daemon dispatches
# `tools.closeout <arms> --pass hof5000` and carries none of these numbers (see
# `desktop/runner/launch.py` for why it must not), and `tools/scheduler.py` runs the same command.
# `stageb` is the close-out's own defaults, so a command that names no pass is unchanged.
PASSES = {
    'stageb': {'selector': 'screen', 'episodes': 500, 'label': None, 'seed': 0},
    'hof5000': {'selector': 'above:99', 'episodes': 5000, 'label': 'hof5000', 'seed': 0},
    'hof30k': {'selector': 'above:99:hof5000', 'episodes': 30000, 'label': 'hof30k', 'seed': 7},
}
# The chain, in the order the passes run. `FOLLOW_ON[pass]` is what a finished pass earns.
CHAIN = ('stageb', 'hof5000', 'hof30k')
FOLLOW_ON = {CHAIN[i]: CHAIN[i + 1] for i in range(len(CHAIN) - 1)}


def pass_settings(name, selector=None, episodes=None, label=None, seed=None):
    """`{'selector', 'episodes', 'label', 'seed'}` for a named pass, with any explicit value winning.

    Explicit means "the caller typed it": `None` is the not-given value for every field, including
    `label`, whose preset for stage B *is* None — so a caller cannot un-label a hof pass by accident,
    and the hand-typed `--selector above:99 --episodes 5000 --label hof5000` still spells the same
    pass `--pass hof5000` does.
    """
    if name not in PASSES:
        raise ValueError('unknown pass {0!r}; known: {1}'.format(name, ', '.join(CHAIN)))
    settings = dict(PASSES[name])
    for key, value in (('selector', selector), ('episodes', episodes), ('label', label),
                       ('seed', seed)):
        if value is not None:
            settings[key] = value
    return settings


class Drawer(object):
    """Keeps one arm's stage-B PNG fresh while its wave runs, on a daemon thread.

    A thread and not a subprocess, deliberately: a `stage_b_chart --watch` child would have to be
    reaped and could outlive a killed close-out, and this project has spent hours on orphaned
    workers. A daemon thread cannot outlive its process.

    **Every failure here is swallowed.** A chart is a readout; a pass that dies because a PNG could
    not be written would be the tail wagging the dog.
    """

    def __init__(self, policy, label=None, interval=REDRAW_SECONDS):
        self.policy, self.label, self.interval = policy, label, interval
        self.stop = threading.Event()
        self.thread = None

    def draw_once(self):
        try:
            return stage_b_chart.redraw(self.policy, self.label)[0]
        except Exception as error:                    # noqa: BLE001 - a readout, never the pass
            sys.stderr.write('chart redraw failed for {0}: {1}\n'.format(self.policy, error))
            return None

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.draw_once()

    def start(self):
        self.thread = threading.Thread(target=self._loop, name='redraw', daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        # The last word on the arm is drawn from the merged file the wave has just written, not from
        # the shard files a mid-pass redraw pooled.
        self.draw_once()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.finish()
        return False


def measure_one(policy, episodes=500, width=None, seed=0):
    """A single checkpoint, in this process — the `one` selector.

    `one` is not a close-out; it is the spot check and the record re-measurement, and it goes through
    here only so that **every eval the desktop can dispatch has one entry point**. The alternative was
    a second branch in `desktop/runner/launch.py` choosing between two scripts, which is the shape
    this change exists to remove.

    Imported where it is used rather than at the top: `evaluate.py` is a script, and a `tools/` module
    importing one is the wrong direction to make permanent — this is a delegation, not a dependency.
    """
    import evaluate
    return evaluate.measure_one(policy, None, episodes, width, seed)


def run(policies, selector='screen', episodes=500, shards=4, label=None, width=None, seed=0,
        resume=True, merge=True, redraw_interval=REDRAW_SECONDS):
    """Measures every arm, shards pooled across them. Returns the last failing arm's status, or 0.

    The keyword defaults are `evaluate.py`'s, which are the protocol — `screen:97` at 500 episodes.
    Nothing here has an opinion about them; a caller that wants the protocol passes nothing.
    """
    started = time.time()
    print('=== close-out: {0} arm(s), selector {1}, {2} episodes, {3} shard(s) pooled'.format(
        len(policies), selector, episodes, shards), flush=True)
    if selector == 'one':
        status = 0
        for index, policy in enumerate(policies, 1):
            print('\n--- [{0}/{1}] {2}  {3}'.format(
                index, len(policies), policy, time.strftime('%H:%M:%S')), flush=True)
            try:
                measure_one(policy, episodes, width, seed)
            except Exception as error:                # noqa: BLE001 - see the class docstring
                status = 1
                print('--- {0} raised {1}: {2}'.format(policy, type(error).__name__, error), flush=True)
    else:
        status = run_pool(policies, selector, episodes, shards, label, width, seed, resume, merge,
                          redraw_interval)
    print('\n=== close-out done in {0:.1f}m, status {1}'.format(
        (time.time() - started) / 60.0, status), flush=True)
    return status


def run_pool(policies, selector, episodes, pool, label, width, seed, resume, merge,
             redraw_interval=REDRAW_SECONDS, poll=None):
    """Keeps up to `pool` shards running across the arms, in arm order, merging each arm as it ends.

    Every arm is resolved first, so a bad arm is reported at the top rather than an hour in, and the
    total is known for the progress line. Then the loop: fill the free slots from the first arm that
    still has shards to launch, sleep, finish the arms whose shards have all exited, print one line.
    An arm's chart is redrawn from its shard files while any of them run (`Drawer`), and once more
    from the merged file when it is finished.
    """
    poll = eval_wave.POLL_S if poll is None else poll
    status = 0
    waves = []
    for index, policy in enumerate(policies, 1):
        try:
            wave = eval_wave.ArmWave(policy, selector, episodes, pool, label, width, seed, resume,
                                     merge)
        except Exception as error:                    # noqa: BLE001 - one bad arm, not a bad batch
            status = 1
            print('--- [{0}/{1}] {2} raised {3}: {4}; the arms behind it still run'.format(
                index, len(policies), policy, type(error).__name__, error), flush=True)
            continue
        print('--- [{0}/{1}] '.format(index, len(policies)), end='')
        wave.announce(requested=pool)
        waves.append(wave)
    total = sum(len(wave.steps) for wave in waves)
    print('{0} step(s) across {1} arm(s), {2} shard(s) pooled\n'.format(total, len(waves), pool),
          flush=True)

    queue = list(waves)          # arms with shards still to launch, in order
    active = []                  # arms with shards launched and not yet finished
    drawers = {}
    started = time.time()
    baseline = sum(sum(wave.counts()) for wave in waves)
    banked = [0]        # rows of finished arms: the merge deletes the shard files they were read from

    def close(wave):
        banked[0] += sum(wave.counts())
        try:
            code = wave.finish()
        except Exception as error:                    # noqa: BLE001 - see the class docstring
            code = 1
            print('--- {0} raised {1}: {2}'.format(wave.policy, type(error).__name__, error),
                  flush=True)
        if wave in drawers:
            drawers.pop(wave).finish()
        if code:
            print('--- {0} exited {1}; the arms behind it still run'.format(wave.policy, code),
                  flush=True)
        return code

    # Arms with nothing to measure are closed at once: their empty file is what the next pass reads.
    for wave in [wave for wave in queue if not wave.shards]:
        queue.remove(wave)
        status = close(wave) or status

    try:
        while queue or active:
            free = pool - sum(wave.alive() for wave in active)
            while free > 0 and queue:
                wave = queue[0]
                if wave not in active:
                    active.append(wave)
                    drawers[wave] = Drawer(wave.policy, label, redraw_interval).start()
                    print('--- {0} starts  {1}'.format(wave.policy, time.strftime('%H:%M:%S')),
                          flush=True)
                wave.start_shard()
                free -= 1
                if wave.exhausted():
                    queue.pop(0)
            if not active:
                break
            time.sleep(poll)
            for wave in [wave for wave in active if wave.finished()]:
                active.remove(wave)
                status = close(wave) or status
            done = banked[0] + sum(sum(wave.counts()) for wave in active)
            elapsed = time.time() - started
            fresh = done - baseline
            eta = ''
            if fresh >= eval_wave.ETA_MIN_ROWS and elapsed > 0:
                eta = '  eta {0:.0f}m'.format((total - done) / (fresh / elapsed) / 60.0)
            print('[{0:>5}/{1}] {2} shard(s) alive on {3} arm(s), {4} arm(s) waiting, {5:.0f}m '
                  'elapsed{6}'.format(done, total, sum(wave.alive() for wave in active),
                                      len(active), len(queue), elapsed / 60.0, eta), flush=True)
    except KeyboardInterrupt:
        print('\nstopping shards; their rows are already on disk and a rerun resumes them',
              flush=True)
        for wave in active:
            wave.terminate()
        for drawer in drawers.values():
            drawer.finish()
        raise
    return status


def build_parser():
    """This module's command-line contract, as an object.

    Extracted for the same reason `evaluate.build_parser` is: the desktop daemon runs on base python
    and cannot import this file, so `desktop/runner/launch.py` spells this interface out by hand.
    That duplication has drifted before — a `--selector` flag against a positional, and several
    policies against a single one, which made every wave the box dispatched exit 2 — so
    `tests/test_desktop_runner.py` hands what the daemon builds to this parser.

    The selector is a **flag** here and positional in `evaluate.py`. It has to be: `policies` is
    `nargs='+'`, and a trailing positional behind it is ambiguous.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policies', nargs='+')
    parser.add_argument('--pass', dest='pass_name', default='stageb', choices=CHAIN,
                        help='which pass of the protocol: its selector, depth, label and seed. '
                             'stageb is the default; hof5000 and hof30k select from the pass before')
    parser.add_argument('--selector', default=None,
                        help='a step selector; the default is the pass\'s, screen:97 for stage B')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--label', default=None, help='names the pass, so two do not collide')
    parser.add_argument('--width', type=int, default=None, help='games in lockstep, per process')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pass_ = pass_settings(args.pass_name, args.selector, args.episodes, args.label, args.seed)
    return run(args.policies, pass_['selector'], pass_['episodes'], args.shards, pass_['label'],
               args.width, pass_['seed'], not args.no_resume, not args.no_merge)


if __name__ == '__main__':
    sys.exit(main())
