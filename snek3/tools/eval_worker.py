"""One process draining the stage-A queue for every arm on this box.

    PYTHONPATH=. python -u -m tools.eval_worker --slot 0

**Rounds, not a single endless stream, and that is what keeps `engine.py` untouched.** A round takes
whatever is queued right now, measures it in one `measure_stream` call, and starts again. The
alternative — a `next_job` that blocks waiting for the trainers to produce more — would hold live
lanes idle inside the engine and trip its own stall guard, which raises when no episode completes in
`MAX_STARVE_BUDGET * PERFECT_SCORE` steps precisely so that a loop which is not advancing says so
instead of spinning.

Rounds also make the throughput self-correcting, which is the property the whole design rests on. A
worker that falls behind finds a deeper queue on its next round, and a deeper round is *cheaper per
checkpoint*: 0.44 utilisation at 10 resident checkpoints against 0.70 at 30. So the queue converts lag
into efficiency until it reaches the trainer's depth bound, at which point the trainer takes the work
back. Nothing has to measure its own rate or tune anything.

**Arms are interleaved, not drained one at a time.** Each arm's schedule lags by its own queue depth,
and the bound is per arm, so emptying one arm's backlog while another sits at its bound would stall
that other arm on work this process was holding. `_round_steps` takes from every arm in turn.

**A checkpoint that has gone is skipped, not an error.** The trainer prunes checkpoints its eval put
below `MIN_CHECKPOINT_SCORE`, and it takes work back when it is blocked — so by the time a round
reaches a step, that step may have been measured and retired already. That is the normal operation of
the queue rather than a race to close.
"""

import argparse
import os
import sys
import time

import torch

from tools import checkpoints
from tools import eval_queue
from tools import restore
from vectorized import engine

# Resident checkpoints per round. Above the engine's own derived `max_live` there is nothing to gain —
# a round is bounded by what the trainers have queued, which at four arms and a depth of 8 is 32 — and
# a cap keeps one very deep round from holding every arm's work at once.
MAX_ROUND = 64

# How long to wait before looking again when the queue is empty. Short relative to the ~3.3 s between
# one arm's evals, so a worker picks up work promptly, and cheap because the check is one `listdir`
# per arm.
POLL_SECONDS = 1.0


class _NetPool:
    """Nets sized for one arch, handed out and returned as checkpoints come and go.

    Copied from `tools/shard.py` for the same reason it exists there: `measure_stream` keeps several
    checkpoints live at once, so a single net cannot serve them, and a `30 -> 320 -> 3` net is 45 KB
    so pooling is free. One pool per arch, because a box may run arms of different widths.
    """

    def __init__(self, arch, device='cpu'):
        self._arch = arch
        self._device = device
        self._free = []

    def take(self):
        if self._free:
            return self._free.pop()
        return restore.build_net(self._arch, device=self._device)

    def give_back(self, net):
        self._free.append(net)


def _round_steps(queued, limit=MAX_ROUND):
    """`[(policy, step), ...]` for one round, taking from each arm in turn.

    Round robin over arms rather than concatenating them, so no arm's backlog is drained at the
    expense of another's — see the module docstring. Within an arm the order is oldest first, because
    a trainer merges strictly in step order and a gap at the front holds up everything behind it.
    """
    lanes = [(policy, list(steps)) for policy, steps in sorted(queued.items())]
    taken = []
    while lanes and len(taken) < limit:
        for policy, steps in list(lanes):
            if not steps:
                lanes.remove((policy, steps))
                continue
            taken.append((policy, steps.pop(0)))
            if len(taken) >= limit:
                break
    return taken


def _same_depth(candidates, episodes_of):
    """Narrows a round to one episode count: the most common among `candidates`, ties to the largest.

    **`measure_stream` takes one episode count for the whole call**, so a round cannot mix depths —
    and forcing them into one would give an arm rows at a denominator it never asked for, which
    `docs/invariants.md` invariant 8 says are not comparable. Whatever is left over is simply picked
    up by the next round, so a box running a 10-episode smoke test beside 100-episode arms just does
    a couple more rounds.

    The most common rather than the first, so one smoke test cannot halve the throughput of three
    real arms by winning the round-robin's first slot.
    """
    depths = {}
    for policy, step in candidates:
        depth = episodes_of(policy, step)
        if depth:
            depths.setdefault(depth, []).append((policy, step))
    if not depths:
        return None, []
    chosen = max(depths, key=lambda depth: (len(depths[depth]), depth))
    return chosen, depths[chosen]


class Worker(object):
    """Drains the queue until it has been idle for `idle_exit`, then exits.

    Owns its slot file and nothing else. Every result goes to its own `<step>.done`; nothing here ever
    opens a `runs/<policy>_evals.json`, which is what keeps that file single-writer.
    """

    def __init__(self, slot, runs_dir=None, episodes=100, width=None,
                 idle_exit=eval_queue.IDLE_EXIT_SECONDS, device='cpu'):
        self.slot = int(slot)
        self.runs_dir = runs_dir
        self.episodes = int(episodes)
        self.width = width
        self.idle_exit = float(idle_exit)
        self.device = device
        self.arches = {}
        self.pools = {}
        self.measured = 0
        self.rounds = 0

    # ------------------------------------------------------------ per-arm state

    def _arch_for(self, policy):
        """The arm's `arch.json`, cached. None when the policy directory has gone.

        Read from the sidecar rather than from a knob, which is the rule the whole project runs on:
        a checkpoint restored into a differently-shaped net loads silently and plays like a beginner.
        """
        if policy not in self.arches:
            try:
                # `policy_arch` takes the *directory*; `policy_dir` resolves a bare arm name against
                # `savedPolicies/` and passes a hallOfFame-style path through untouched.
                self.arches[policy] = restore.policy_arch(restore.policy_dir(policy))
            except Exception:               # a directory that vanished, or a sidecar not yet written
                self.arches[policy] = None
        return self.arches[policy]

    def _pool_for(self, policy, arch):
        key = (policy, tuple(arch['fc_layer_params']), arch['obs_len'], arch['num_actions'])
        if key not in self.pools:
            self.pools[key] = _NetPool(arch, device=self.device)
        return self.pools[key]

    # ------------------------------------------------------------ one round

    def run_round(self):
        """Claims what is queued and measures it in one stream. Returns how many checkpoints landed."""
        taken = _round_steps(eval_queue.pending(self.runs_dir))
        if not taken:
            return 0
        # The depth comes from the requests, never from this process: `--episodes` is only the
        # fallback for a request written before the field existed. A worker that imposed its own
        # count silently gave a `SNEK_GRAPH_EVAL_EPISODES=10` arm 100-episode rows.
        episodes, taken = _same_depth(
            taken, lambda policy, step: eval_queue.episodes_of(policy, step, self.runs_dir)
                                        or self.episodes)
        if not taken:
            return 0

        jobs = []
        for policy, step in taken:
            arch = self._arch_for(policy)
            if arch is None:
                continue
            payload = eval_queue.claim(policy, step, self.runs_dir)
            if payload is None:
                continue                    # another worker won it, or the trainer took it back
            jobs.append((policy, step, arch, payload.get('fields') or {}))
        if not jobs:
            return 0

        pending = iter(jobs)
        live = {}
        landed = [0]

        def next_job():
            """The next claimed checkpoint, as `(key, policy_fn)`, or None to end the round.

            A checkpoint whose file has gone is skipped rather than raised on: the trainer prunes the
            ones its eval rejected, and it retires the ones it measured itself. Skipping keeps the
            round going, which matters because one missing file would otherwise abandon every
            checkpoint queued behind it.
            """
            for policy, step, arch, fields in pending:
                pool = self._pool_for(policy, arch)
                net = pool.take()
                try:
                    checkpoints.load(checkpoints.path(restore.policy_dir(policy), step), net,
                                     device=self.device)
                except Exception as error:
                    pool.give_back(net)
                    sys.stderr.write('eval worker {0}: skipping {1} @{2} ({3})\n'.format(
                        self.slot, policy, step, error))
                    continue
                key = (policy, step)
                live[key] = (net, pool, fields)
                return key, restore.policy_fn_for(arch, net, device=self.device)
            return None

        def on_complete(key, held):
            policy, step = key
            net, pool, fields = live.pop(key)
            pool.give_back(net)
            # The sample as the engine produced it, plus the trainer's half copied straight through.
            # No row is built here: a row needs both halves and this process has no business
            # interpreting either — inventing a default epsilon would put a number in the arm's
            # history that describes nothing that happened.
            eval_queue.complete(policy, step, _sample(held), fields, self.runs_dir)
            landed[0] += 1

        # The seed mixes the clock, this process and the round number. The clock alone is not enough:
        # two workers starting a round in the same second would draw the *same* food streams, so two
        # arms' rows would share their boards and their sampling error would be correlated — which is
        # exactly the kind of hidden coupling that makes a between-arm comparison read as a real
        # effect. There is nothing to reproduce here either way: a queued row's boards are not
        # derivable from the arm's seed (see `tools/eval_queue.py`), which is the cost the queue is
        # pre-registered as having.
        seed = (int(time.time() * 1000) ^ (os.getpid() << 16) ^ (self.rounds * 2654435761)) & 0x7FFFFFFF
        engine.measure_stream(next_job, on_complete, episodes, width=self.width, seed=seed)
        self.rounds += 1
        self.measured += landed[0]
        return landed[0]

    # ------------------------------------------------------------ the loop

    def run(self):
        torch.set_num_threads(1)
        print('eval worker {0}: draining {1}, episode count per request, exits after {2:.0f}s '
              'idle'.format(self.slot, eval_queue.directory(self.runs_dir), self.idle_exit),
              flush=True)
        idle_since = time.time()
        try:
            while True:
                if self.run_round():
                    idle_since = time.time()
                    continue
                if time.time() - idle_since >= self.idle_exit:
                    break
                time.sleep(POLL_SECONDS)
        finally:
            # The slot goes even on a traceback, so the next arm to start can replace this worker
            # rather than seeing a slot held by a pid that is about to disappear.
            eval_queue.release_slot(self.slot, self.runs_dir)
        print('eval worker {0}: {1} checkpoint(s) in {2} round(s), exiting idle'.format(
            self.slot, self.measured, self.rounds), flush=True)


def _sample(held):
    """The measured sample, JSON-ready and trimmed to what a stage-A row needs.

    The per-episode arrays are dropped. Stage B keeps them because a row there has to be poolable and
    its median exact, but a stage-A row stores only the five summary fields — so carrying 300 numbers
    per checkpoint through the queue would be ~1 KB of JSON per eval to be thrown away by the reader.
    """
    return {'scores': [float(value) for value in held['scores']],
            'perfect': [int(value) for value in held['perfect']],
            'rewards': [float(value) for value in held['rewards']]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--slot', type=int, default=0, help='which worker slot this process holds')
    parser.add_argument('--runs-dir', default=None)
    parser.add_argument('--episodes', type=int, default=100,
                        help='fallback only; a request carries its own episode count')
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--idle-exit', type=float, default=eval_queue.IDLE_EXIT_SECONDS)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    Worker(args.slot, runs_dir=args.runs_dir, episodes=args.episodes, width=args.width,
           idle_exit=args.idle_exit, device=args.device).run()


if __name__ == '__main__':
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    main()
