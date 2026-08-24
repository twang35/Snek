"""Decoupled live chart viewer — shows a grid of policy chart PNGs and refreshes.

Runs as its **own process**, entirely separate from training: it only *reads* the
PNGs training writes every eval (`runs/<policy>.png`). So if the viewer dies — an X
error, or the OS OOM-kills it — training is untouched; and when training stops, the
viewer notices via `--watch` and exits on its own. This is the safe replacement for
the in-process cv2 window whose fatal XIO error took down all four desktop arms at
once on 2026-08-09.

Memory is trivial (matplotlib plus a few small PNGs, ~100 MB) regardless of how many
arms there are, so it can neither cause nor be worsened by an OOM.

On the laptop a training launch starts one of these itself — `snek2.main()` calls
`spawn_for_policy()`, so an arm shows its chart without anyone remembering to open a
window. See that function for the gate and the dedupe.

Usage:
  python chart_viewer.py --arms b20            # the b20 arms that are actually running
  python chart_viewer.py runs/b20e-*.png ...   # explicit files, or
  python chart_viewer.py --glob 'runs/b20*.png'   # a glob, re-read each refresh
    [--watch 'snek2.py b20']   pgrep -f pattern; exit once nothing matches (training stopped)
    [--interval 1]             refresh seconds
    [--scale 2]                window size multiplier
    [--title 'snek training']  window title
"""
import argparse
import glob as globmod
import os
import re
import signal
import subprocess
import sys
import tempfile
import time


def _scan_all_commands(pattern, mine):
    """Command lines matching `pattern` read from a full `ps -Ao pid=,command=` scan.

    The second opinion `_matching_commands` asks before it reports nothing running. `ps -A` lists
    every process with the same argv text `ps -p` gives, so it answers the same question without
    going through pgrep at all; the pattern is applied with Python's `re`, which is why an
    unparseable one raises here rather than reading as "nothing found".

    A non-zero exit is unanswerable, not empty. Same filters as the caller: our own pid, and any
    other viewer, since the watched pattern is an argument on a viewer's own command line.
    """
    ps = subprocess.run(['ps', '-Ao', 'pid=,command='],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if ps.returncode != 0:
        raise RuntimeError('ps -A failed with status {0}'.format(ps.returncode))
    try:
        matcher = re.compile(pattern)
    except re.error as error:
        raise RuntimeError('pattern {0!r} is not a usable regex: {1}'.format(pattern, error))
    found = []
    for line in ps.stdout.decode('utf-8', 'replace').splitlines():
        fields = line.split()
        if not fields or fields[0] == mine or 'chart_viewer.py' in line:
            continue
        if matcher.search(line):
            found.append(line)
    return found


def _matching_commands(pattern):
    """Command lines of processes matching `pattern` (pgrep -f), excluding viewers.

    Our own pid and every other `chart_viewer.py` are dropped, because the watched pattern
    is an *argument* on the viewer's own command line -- a naive pgrep matches the viewer
    itself. Uses `ps` for the cmdline text so it works on macOS (no /proc) as well as Linux.
    Raises on failure; callers decide what an unanswerable check means, which differs
    between "is training alive" (keep showing) and "which arms are live" (show none new).

    **That last paragraph was a lie until 2026-08-19, and it is the whole point of the
    returncode check below.** `pgrep` exits 0 for a match, **1** for no match and **>= 2** for an
    error — bad usage, an unparseable pattern, or a failure to enumerate processes (verified on this
    machine: 1 for no match, 2 for both a bad flag and a bad regex). Every one of those produces
    empty stdout, and `subprocess.run` without `check=True` does not raise, so a pgrep that *failed*
    read as **"nothing is running"** — the strongest possible answer — from the weakest possible
    evidence.

    What that costs: `_training_alive` returns False, and six consecutive False readings close the
    window. The `b43` eval window did exit at 13:59 on 2026-08-19 while `b43b-lowlr-b29a` was still
    running, in the same minute a sibling arm and its four spawned workers were tearing down. **That
    is a plausible cause, not a proven one** — the check passes on the same pattern afterwards, so it
    was never reproduced, and the more obvious suspect (a sibling pid dying between the `pgrep` and
    the `ps`) was tested and *falsified*: a recently dead pid still returns the live ones. The bug
    below is real regardless of whether it is that bug.

    `live_arms` shares the blind spot, and there it means a panel never appearing — the same shape as
    the historical 3-of-4 window bugs. `running_policies` shared it too, until the panel status tag
    it fed was deleted for being wrong about `eval_wave.py`.

    The `ps` call is deliberately *not* given the same treatment: it exits 1 both when every listed
    pid is gone (a real answer, and the common one during a teardown) and on a bad pid, so there is
    nothing to distinguish. Its empty output is trusted, which is safe because a pid list that came
    from a successful pgrep and then emptied really does mean the processes ended.

    **‡ The returncode check was not enough, and a no-match is now corroborated (2026-08-19).** Three
    `eval_wave.py` waves in a row opened a window that exited within ~10 s -- `0 panels`, then
    `watched training gone`, while the wave it was watching ran for hours afterwards. The mechanism
    was never found: `pgrep -f 'eval_wave.py .*b43'` matches that process every time it is run by
    hand, 80/80 probes, including from a process whose own argv contains the pattern. So the fix
    stops trying to explain pgrep and stops *trusting* it in one direction: a **match** is taken at
    face value, an **absence** is re-checked against a full `ps -Ao pid=,command=` scan with Python's
    own `re`. Only when both agree is the answer empty.

    That asymmetry is the whole idea, and it is cheap -- the corroboration runs only on the answer
    that closes windows, never on the common one. It also removes any dependence on pgrep's regex
    dialect for the negative case. What it cannot do is invent evidence: a genuinely finished wave
    is absent from `ps` too, so a window still exits when it should.
    """
    res = subprocess.run(['pgrep', '-f', pattern],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if res.returncode > 1:
        raise RuntimeError('pgrep -f {0!r} failed with status {1}'.format(
            pattern, res.returncode))
    mine = str(os.getpid())
    pids = [p for p in res.stdout.decode().split() if p and p != mine]
    if not pids:
        return _scan_all_commands(pattern, mine)
    args = ['ps', '-o', 'pid=,command=']
    for p in pids:
        args += ['-p', p]
    ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return [line for line in ps.stdout.decode('utf-8', 'replace').splitlines()
            if line.strip() and 'chart_viewer.py' not in line]


def live_arms(prefix):
    """Policy names of running trainers belonging to batch `prefix`.

    This is what lets a window show *the arms that were launched* rather than every chart
    the batch has ever produced. Globbing `runs/b20*.png` was the alternative and it ages
    badly: by wave 2 that glob matched eight finished arms plus four live ones, which at 2x
    scale is a window taller than the screen.

    Membership is `batch_prefix(policy) == prefix`, not `startswith` -- `'b20a'.startswith('b2')`
    is True, so a prefix test would quietly merge b2 into b20.

    A name is only taken from something that is actually **running python**: the line must
    contain a `python` token before the script, and the script token must be `snek2.py` or
    `.../snek2.py`. Merely mentioning the file is not enough, because plenty of commands do --
    `git ls-files snek2/snek2.py b20i-fc200x50seed1` would otherwise donate its next argument
    as an arm name, and Airbnb git telemetry fires curl processes whose payload lists repo
    paths. That class of false positive has already misread trainer counts twice here.
    """
    arms = []
    try:
        for line in _matching_commands('snek2.py'):
            parts = line.split()
            for i, token in enumerate(parts):
                if not (token == 'snek2.py' or token.endswith('/snek2.py')):
                    continue
                if not any('python' in earlier for earlier in parts[:i]):
                    break       # not a python invocation — this line names no arm
                if i + 1 < len(parts):
                    policy = parts[i + 1]
                    if batch_prefix(policy) == prefix and policy not in arms:
                        arms.append(policy)
                    break
    except Exception:
        return []       # unknown means "nothing new to add", never "drop what we have"
    return sorted(arms)


def arms_path(prefix):
    """Where batch `prefix`'s trainers register themselves. Beside the claim lock, same TTL story."""
    return os.path.join(LOCK_DIR, 'snek_chart_viewer_{0}.arms'.format(prefix))


def register_arm(policy_name):
    """Record that `policy_name` is an arm of its batch, for the viewer to read.

    **This is the fix for a window that opened on 3 of 4 arms**, 2026-08-14. `live_arms` is a
    *snapshot* of the process list, and every arm's panel depended on that snapshot landing after
    the arm's `exec` — for a wave of four launched inside the same second, with the viewer spawned
    by whichever arm reached `main()` first, that is a race with no repair path: an arm the scan
    missed only reappears if a later scan sees it, and a `pgrep`/`ps` hiccup or an arm still in
    Python's import phase can lose it for good. The registry removes the timing dependence — each
    trainer states its own name, once, before any scan happens.

    Append-only with `O_APPEND` so four simultaneous trainers cannot clobber each other: each line
    is a single small write, which POSIX makes atomic at the end of the file. Every trainer calls
    this, not only the one that wins the viewer lock — the winner opens the window, but all four
    have to be *in* it.

    Lines are `<epoch>\\t<policy>`; the timestamp is what lets a later wave of the same batch
    prefix age the previous one out (see `registered_arms`). Failures are swallowed: a registry
    write is never worth a training run.
    """
    try:
        with open(arms_path(batch_prefix(policy_name)), 'a') as handle:
            handle.write('{0:.0f}\t{1}\n'.format(time.time(), policy_name))
    except Exception:
        pass


# Backstop only: a line older than this is dropped whatever else is true, so the file cannot grow
# without bound. **It is not what separates one wave from the next — liveness is** (see
# `registered_arms`). A TTL alone cannot do that job, and shipping one that tried is why batch 30's
# relaunch opened a window with **eight panels**: the arms it replaced were 71 minutes old, well
# inside 12 h, so the registry offered all eight and the union took them.
ARM_REGISTRY_TTL = 12 * 3600
# Grace period in which a registered arm needs no live process to keep its panel. This is the whole
# reason the registry exists: a trainer registers before `exec` is visible to `ps` and while Python
# is still importing TensorFlow, so for the first seconds of a wave the process scan genuinely
# cannot see arms that are starting. 120 s is ~20x the observed gap between registration and a
# scan-visible process, and short enough that a dead arm from an earlier launch never reaches it.
ARM_REGISTRY_GRACE = 120
# Hard cap on panels, whatever the registry says. A window taller than the screen shows nothing.
MAX_WAVE_PANELS = 8


def registered_arms(prefix, now=None, alive=None):
    """Arms `prefix`'s trainers registered that are **either just-registered or still running**.

    Oldest first, deduped. Two admission rules, and the second is the one that separates a relaunch
    from the wave it replaced:

    - **age <= `ARM_REGISTRY_GRACE`** — a starting arm, which the process scan cannot see yet. This
      is the registry's purpose and it needs no corroboration.
    - **the policy has a live trainer** — an older entry that is still real. After the grace period
      the process list is the authority on "running", so the registry stops asserting it alone.

    **Age alone was not enough, and batch 30 proved it within an hour.** The original rule admitted
    anything inside a 12 h TTL, so relaunching `b30` 71 minutes after killing its first wave offered
    the four dead arms plus the four new ones and the window opened with eight panels — the exact
    failure the registry was introduced to prevent, arriving by a different route.

    **This does not weaken the sticky-panel property**, which is the reason the naive fix ("drop
    anything not running") looks wrong: stickiness does not live here. `wave_files` accumulates into
    its caller's `known` list and never prunes it, so an arm that finishes mid-wave was already
    admitted and keeps its panel for the life of the viewer. What this rule drops is an arm that was
    never in *this* viewer's `known` to begin with.

    Unreadable or malformed lines are skipped rather than raised on: this is a convenience index
    rebuilt by every launch, so a corrupt line costs one panel and never the window.
    """
    now = time.time() if now is None else now
    alive = live_arms(prefix) if alive is None else alive
    found = []
    try:
        with open(arms_path(prefix)) as handle:
            lines = handle.read().splitlines()
    except Exception:
        return []
    for line in lines:
        stamp, _, policy = line.partition('\t')
        if not policy:
            continue
        try:
            age = now - float(stamp)
        except ValueError:
            continue
        if age > ARM_REGISTRY_TTL or age < -60:
            continue        # stale, or a clock that ran backwards further than a skew
        if age > ARM_REGISTRY_GRACE and policy not in alive:
            continue        # a previous wave's arm: registered long ago, not running now
        if batch_prefix(policy) == prefix and policy not in found:
            found.append(policy)
    return found


def wave_files(prefix, known):
    """PNG paths for batch `prefix`'s arms, adding any newly-seen arm to `known`.

    Two sources, unioned, because each covers the other's blind spot: the **registry** is what
    the trainers wrote about themselves, so it cannot miss an arm that launched (the 3-of-4 bug),
    and the **process scan** still catches an arm started by hand or resumed without going through
    `spawn_for_policy`.

    Sticky on purpose: `known` is mutated and never pruned, so an arm that reaches its step
    cap keeps its panel while its siblings run. That is the point of watching a wave — the
    finished curve is what the others get compared against — and it also means a transient
    `ps` failure cannot blank the window. It is also where *all* the stickiness lives, which is what
    lets `registered_arms` require liveness of anything past its grace period.

    One scan, passed to both readers. Two scans would be two chances to disagree — an arm appearing
    between them would be admitted by the second and absent from the first, which is the class of
    race this file has already been bitten by twice.
    """
    alive = live_arms(prefix)
    for policy in registered_arms(prefix, alive=alive) + alive:
        if policy not in known:
            known.append(policy)
    del known[MAX_WAVE_PANELS:]
    return [os.path.join('runs', policy + '.png') for policy in sorted(known)]


def policy_from_png(path):
    """The policy a chart PNG belongs to: `runs/b20q-fc25.png` -> `b20q-fc25`, and
    `evals/b20q-fc25_eval_progress.png` -> `b20q-fc25`. The two layouts are the trainer's
    chart and the eval's chart; both encode the policy as the stem so a panel can be tied
    back to the process that produces it."""
    base = os.path.basename(path)
    if base.endswith('_eval_progress.png'):
        return base[:-len('_eval_progress.png')]
    if base.endswith('.png'):
        return base[:-len('.png')]
    return base


# There is no per-panel title, and that is deliberate (2026-08-19). Every chart this viewer shows
# already carries its own title *inside the image* — an eval PNG says `<policy> — eval progress
# <time>`, a training PNG names itself — so a matplotlib title printed the policy name a second time
# directly above the first, and a four-panel window spent two lines of every panel's height on it.
# `tight_layout` pays for that by shrinking the images, which is the visible symptom: smaller charts
# and a smaller font in exactly the window whose whole job is to be readable across the room.
#
# It also carried a status tag, and the tag is the reason this is a hard rule rather than a
# preference. `(completed)` was decided by asking whether the arm's name appeared on a running
# `snek2.py` or `eval_checkpoints.py` command line — so the moment a wave began running under
# `eval_wave.py` instead, **every panel of a live four-arm eval read `(completed)`**. A status
# derived from a process scan is wrong whenever the launcher changes, and it is wrong silently.
# The image itself already shows liveness: a live arm's curve moves every refresh and a finished
# one's is frozen.
#
# So panels get no title. If a chart is genuinely missing the panel is blank, which is honest — and
# `--glob`/`--arms` only ever names files that exist or arms that launched.


def _training_alive(pattern):
    """True while any process *other than this viewer* matches `pattern` (pgrep -f).

    The watched pattern is itself part of the viewer's own command line -- it is an
    argument -- so a naive pgrep matches the viewer and it would never exit. Drop our
    own pid and any chart_viewer.py process from the match. Uses `ps` for the cmdline
    check so it works on macOS (no /proc) as well as Linux. If we cannot tell, keep
    showing rather than exit prematurely."""
    if not pattern:
        return True
    try:
        return bool(_matching_commands(pattern))
    except Exception:
        return True


BATCH_RE = re.compile(r'^(b\d+)[a-z]*-')


def batch_prefix(policy_name):
    """The batch an arm belongs to: `b20a-fc50seed1` -> `b20`.

    One window per *batch*, not per arm, because a wave launches four arms within
    seconds of each other and four windows would be unreadable. A name that is not a
    `b<n><letters>-` arm (`train`, `eval`) is its own prefix and gets a window of its own.

    The arm suffix is `[a-z]*`, not a single letter: a batch with more than 26 arms rolls
    into double letters (`b20aa-`, `b20ab-`, ...), and those must still group under `b20`,
    not be read as their own batch. Batch 20 does exactly this — nine shapes at four seeds is
    36 arms."""
    m = BATCH_RE.match(policy_name)
    return m.group(1) if m else policy_name


def viewer_enabled():
    """On by default on the laptop (darwin), off elsewhere; `SNEK_CHART_VIEWER` overrides.

    Off elsewhere because the desktop's runner daemon owns the viewer there
    (`desktop/runner/runner.py::_ensure_viewer`) — it has to inject the graphical
    session's DISPLAY/XAUTHORITY, which a trainer launched by systemd does not have, so a
    trainer-side launch would fail there anyway."""
    default = '1' if sys.platform == 'darwin' else '0'
    return os.environ.get('SNEK_CHART_VIEWER', default) not in ('0', '', 'false', 'False')


def viewer_running_for(glob_pattern):
    """True if some viewer process is already showing `glob_pattern`.

    Matches on the glob rather than on `chart_viewer.py` alone, so a second batch on the
    same host still gets its own window while the first one's stays up. If the check
    itself fails we answer True: skipping a window is recoverable by hand, whereas
    stacking one per arm on the display is what this dedupe exists to prevent."""
    try:
        res = subprocess.run(['pgrep', '-f', 'chart_viewer.py'],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pids = [p for p in res.stdout.decode().split() if p and p != str(os.getpid())]
        # Zombies drop out here for the same reason they do in `pid_alive`: an exited viewer keeps
        # its argv — `--arms b30` and all — until its parent trainer reaps it, so a substring match
        # on `ps` output reports the window of a viewer that closed hours ago. This site was missed
        # when `pid_alive` was added and it blocked the b30 restart on its own. State-only, not
        # `pid_alive`: the pid came from `pgrep` a moment ago, so `kill` adds nothing here.
        pids = [p for p in pids if not zombie(pid_state(int(p)))]
        if not pids:
            return False
        args = ['ps', '-o', 'command=']
        for p in pids:
            args += ['-p', p]
        ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return any(glob_pattern in line
                   for line in ps.stdout.decode('utf-8', 'replace').splitlines())
    except Exception:
        return True


# Where the per-batch claim lock lives. A module constant rather than an inline
# `tempfile.gettempdir()` so tests can point it somewhere private: they exercise real
# prefixes like `b20`, and a test that wrote the real lock could suppress or steal the
# window of a wave that is actually training.
LOCK_DIR = tempfile.gettempdir()


def lock_path(prefix):
    return os.path.join(LOCK_DIR, 'snek_chart_viewer_{0}.lock'.format(prefix))


def pid_state(pid):
    """First word of `ps -o stat=` for `pid`: a state like `S`, `R` or `ZN`.

    `''` means ps knows nothing about the pid (it is gone), `None` means ps itself could not be
    run and the caller has to decide what that means. Split out from `pid_alive` because the two
    callers want different fallbacks — see each."""
    try:
        res = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    text = res.stdout.decode('utf-8', 'replace').strip()
    return text.split()[0] if text else ''


def zombie(state):
    """True if `state` is a ps state word for a zombie. `None`/`''` are not zombies."""
    return bool(state) and state.startswith('Z')


def pid_alive(pid):
    """True if `pid` is a process that could still be showing a window — **zombies are dead.**

    `os.kill(pid, 0)` alone is not the right test here, and that shipped broken: a viewer is
    spawned by a trainer that never `wait()`s for it, so when the viewer exits it stays a zombie
    for as long as its parent trainer runs — hours. `kill(pid, 0)` succeeds on a zombie, so the
    claim lock read "a live viewer owns this batch's window" for the rest of the wave, and no
    trainer could ever reopen it. That is exactly the "nothing suppresses the window forever"
    property `claim_viewer_slot` claims to have. Found 2026-08-14 while restarting b30's window:
    the killed viewer sat in state `ZN` and `kill -0` still reported it alive.

    An unanswerable `ps` falls back to the `kill` result: refusing to open a second window is the
    safer error, since a duplicate window is the thing the lock exists to prevent.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    state = pid_state(pid)
    if state is None:
        return True         # cannot tell — keep the claim rather than risk a second window
    if state == '':
        return False        # ps knows nothing about it; it exited between the two checks
    return not zombie(state)


def claim_viewer_slot(prefix):
    """Atomically claim the right to open batch `prefix`'s window. True if we won it.

    **A check-then-spawn dedupe is not enough, and this shipped broken.** A wave's four
    trainers launch in the same second, so all four ran `viewer_running_for` before any of
    them had spawned anything, all four saw nothing, and all four opened a window. `O_EXCL`
    is the fix because the create and the test are one operation.

    A lock left behind by a dead holder is taken over rather than honoured, so nothing
    suppresses the window forever. The pid written here is only a placeholder — the caller
    replaces it with the *viewer's* pid via `hold_viewer_slot` once the spawn succeeds, and
    calls `release_viewer_slot` if it fails. That ordering matters: the question the lock
    answers is "is a viewer up for this batch", so pointing it at a trainer would keep the
    claim alive after the viewer had been killed, and drop it when the trainer merely
    finished."""
    lock = lock_path(prefix)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        pass
    except Exception:
        return True     # cannot lock (odd tmpdir) -> fall back to the pgrep check alone
    try:
        with open(lock) as fh:
            holder = int(fh.read().strip() or 0)
    except Exception:
        holder = 0
    if holder and pid_alive(holder):
        return False                # a live claimant owns this batch's window
    try:                            # stale: the claimant is gone, so take it over
        with open(lock, 'w') as fh:
            fh.write(str(os.getpid()))
        return True
    except Exception:
        return False


def hold_viewer_slot(prefix, pid):
    """Point the claim at the viewer we just started, so liveness tracks the window."""
    try:
        with open(lock_path(prefix), 'w') as fh:
            fh.write(str(pid))
    except Exception:
        pass        # the claim still holds under our own pid; not worth failing over


def release_viewer_slot(prefix):
    """Give the claim back, so another arm of the same wave can try after a failed spawn."""
    try:
        os.remove(lock_path(prefix))
    except Exception:
        pass


def spawn_for_policy(policy_name):
    """Best-effort live chart window for a training launch. Returns the Popen, or None.

    A *separate* process in its own session — never a child of training in any way that
    matters — so it cannot take an arm down, which is the whole reason the in-process cv2
    window was removed. Every failure path is swallowed with a note on stdout: a chart is
    never worth a training run. Smoke runs are skipped, since they are verification, not
    an arm anyone watches."""
    if not viewer_enabled() or policy_name.startswith('smoke'):
        return None
    prefix = batch_prefix(policy_name)
    # Before the dedupe, and unconditionally: the three trainers that *lose* the lock still have
    # to appear in the window the winner opens. Registering after the `return None` below is how
    # a wave ends up showing only the arms a process scan happened to catch.
    register_arm(policy_name)
    pattern = '--arms {0}'.format(prefix)
    # Both checks are needed: pgrep catches a viewer started by hand or by an earlier wave
    # whose lock has since been cleaned out of tmp, and the lock closes the race pgrep loses.
    if viewer_running_for(pattern) or not claim_viewer_slot(prefix):
        return None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # Not under runs/ or evals/: both are protected from deletion, and a log is
        # throwaway. tempdir keeps it findable without adding clutter to either.
        log_path = os.path.join(tempfile.gettempdir(),
                                'snek_chart_viewer_{0}.log'.format(prefix))
        log = open(log_path, 'ab')
        argv = [sys.executable, '-u', os.path.join(here, 'chart_viewer.py'),
                '--arms', prefix,
                '--watch', 'snek2.py {0}'.format(prefix),
                '--title', 'snek {0} — live'.format(prefix)]
        proc = subprocess.Popen(argv, cwd=here, stdout=log, stderr=log,
                                start_new_session=True, close_fds=True)
        hold_viewer_slot(prefix, proc.pid)
        print('chart viewer: pid {0} showing the live {1} arms '
              '(log {2}; SNEK_CHART_VIEWER=0 to disable)'.format(proc.pid, prefix, log_path))
        return proc
    except Exception as error:
        release_viewer_slot(prefix)
        print('chart viewer launch failed ({0}: {1}) — training continues'
              .format(type(error).__name__, error))
        return None


def is_verification_policy(policy_name):
    """Throwaway evals nobody watches: smokes, `champion_*` and `bench-*` verification runs.

    The same set the never-delete table treats as disposable, so none of them earns a window."""
    return (policy_name.startswith('smoke') or policy_name.startswith('champion')
            or policy_name.startswith('bench'))


def spawn_for_eval(policy_name, watch='eval_checkpoints.py {prefix}', chart_dir='evals',
                   slot_suffix='eval'):
    """Best-effort live chart window for an eval (close-out / HOF) launch. Returns the Popen or None.

    The training counterpart is `spawn_for_policy`; this differs only in what it points the viewer
    at. An eval writes `evals/<policy>_eval_progress.png` (not `runs/<policy>.png`), a wave of evals
    is a set of *files* rather than live trainers, and there is no arm registry to read — so the
    viewer is launched with `--glob evals/<prefix>*_eval_progress.png` and
    `--watch eval_checkpoints.py <prefix>`, which shows every arm of the wave and exits when the last
    eval process stops. Verification evals (smoke / champion_* / bench-*) get no window.

    `watch` is a `pgrep -f` pattern with `{prefix}` in it, and it has to be the caller's to choose:
    the pattern is matched against a whole command line, so it depends on where the policy names sit
    in that line. `eval_checkpoints.py b44a-x` puts one right after the script, while
    `eval_wave.py top50 b44a-x b44b-y` puts the selector in between — hence `eval_wave.py .*{prefix}`
    from the wave controller. A pattern that cannot match reads as "the eval stopped" and the window
    closes within six checks.

    `chart_dir` and `slot_suffix` exist for the vectorised eval, which writes its charts to
    `evals/vec/` and must not share a lock namespace with the TF eval. Both matter: pointing two
    viewers at the same directory would give each the other's panels, and sharing the `-eval` slot
    would let whichever launched first suppress the other's window entirely — the two are expected
    to run side by side during validation, since that comparison is the whole point.

    Darwin-only via `viewer_enabled()` — the same gate as training — so this never fires on the
    desktop, where the runner daemon owns the viewer (it injects the graphical session's
    DISPLAY/XAUTHORITY; two owners would open two windows per wave). Every failure path is swallowed:
    a chart is never worth an eval."""
    if not viewer_enabled() or is_verification_policy(policy_name):
        return None
    prefix = batch_prefix(policy_name)
    # A lock namespace of its own (`<prefix>-eval`), so a training viewer's leftover claim never
    # suppresses an eval window, nor this one a later trainer's. The glob is relative and resolves
    # against the viewer's cwd (the snek2 dir), exactly as the training path's `runs/` glob does.
    slot = '{0}-{1}'.format(prefix, slot_suffix)
    glob_arg = os.path.join(chart_dir, '{0}*_eval_progress.png'.format(prefix))
    # pgrep catches a viewer opened by an earlier arm of the same wave whose lock has aged out of
    # tmp; the O_EXCL claim closes the race four arms starting in one second would otherwise lose.
    watch_pattern = watch.format(prefix=prefix)
    if viewer_running_for('--glob {0}'.format(glob_arg)) or not claim_viewer_slot(slot):
        return None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(tempfile.gettempdir(),
                                'snek_chart_viewer_{0}-{1}.log'.format(prefix, slot_suffix))
        log = open(log_path, 'ab')
        argv = [sys.executable, '-u', os.path.join(here, 'chart_viewer.py'),
                '--glob', glob_arg,
                '--watch', watch_pattern,
                '--title', 'snek {0} {1} — live'.format(prefix, slot_suffix)]
        proc = subprocess.Popen(argv, cwd=here, stdout=log, stderr=log,
                                start_new_session=True, close_fds=True)
        hold_viewer_slot(slot, proc.pid)
        print('chart viewer: pid {0} showing the live {1} eval charts '
              '(log {2}; SNEK_CHART_VIEWER=0 to disable)'.format(proc.pid, prefix, log_path))
        return proc
    except Exception as error:
        release_viewer_slot(slot)
        print('chart viewer launch failed ({0}: {1}) — eval continues'
              .format(type(error).__name__, error))
        return None


def exit_now(plt, code=0, exit_fn=None):
    """Close the Tk windows, then leave *without* running interpreter shutdown.

    Both halves matter. Tk delivers events for as long as its windows exist and each one
    can call back into Python; if the interpreter has already cleared its thread state,
    that callback aborts the process — `_Py_FatalError_TstateNULL` under `PythonCmd` inside
    `Tk_HandleEvent`, which macOS reports as "python quit unexpectedly" with a crash-report
    dialog. Seen twice on 2026-08-09 (pids 30193, 76128). Closing the figures first removes
    the windows while the interpreter is fully alive, and `os._exit` then skips the
    finalisation window in which a late event could arrive. It is race-dependent, so a
    clean exit on one run does not mean the ordering is safe.

    Skipping cleanup costs nothing here: the viewer owns no state, it only reads PNGs.
    """
    try:
        plt.close('all')
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()   # os._exit does not flush, and the exit line matters
        except Exception:
            pass
    (exit_fn or os._exit)(code)


def install_signal_exit(plt):
    """Route SIGTERM/SIGINT through `exit_now`.

    **Call this after the first figure exists, and after every figure rebuild.** Tk installs
    its own OS-level SIGTERM handler when it creates its interpreter — which happens inside
    the first `subplots()`, not at import — and that overwrites ours. Tcl's handler runs
    `Tcl_Exit` straight off the signal trampoline: it finalises the Tcl thread, destroys the
    windows, fires their `<Destroy>` bindings, and those call back into Python with no thread
    state, which is the abort described in `exit_now`. Installing before the figure looks
    right and does nothing — measured, 5 of 5 kills still aborted."""
    def handle(_sig, _frame):
        exit_now(plt)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle)
        except (ValueError, OSError):   # not the main thread — not worth failing over
            pass
    return handle


# A window taller (or wider) than the screen opens with its lower rows *below* the display,
# which reads as missing charts — a 2x2 wave at scale 2.0 is 12in = 1200px tall and a laptop's
# built-in Retina panel is ~900 usable points, so the bottom row of a four-arm wave was clipped
# and looked like three charts. `fit_figure_to_screen` is the real guard against that: it shrinks
# to the display the window actually opened on. These budgets (inches at matplotlib's 100 dpi) set
# the *target* size on a roomy screen, where the screen fit does not bind: a 2x2 lands at 10.4in =
# 1040px. A built-in panel is smaller than that, so there the screen fit clamps it down to ~88% of
# the panel height instead. A single panel is under the budgets, so its size comes from the scale.
MAX_FIG_W_IN = 18.2
MAX_FIG_H_IN = 13.0

# These budgets and the default scale below were raised 30% on request (from 14.0/8.0 and 2.0); an
# earlier 8.0in was itself a fix for a 9.0in that still clipped a built-in panel.
#
# The height went 10.4 -> 13.0 on 2026-08-19, when the panel box started matching the image's aspect
# (`figure_dims`). A near-square eval chart at the old cap made the *height* the binding constraint
# on a 2x2 grid, so the window came out narrower than the screen with the charts no larger -- the
# whitespace was gone but none of it had been given back. 13.0 hands the decision to
# `fit_figure_to_screen`, which reads the real display and only ever shrinks, so this is a ceiling
# for the case where that probe fails rather than a size anyone sees.


def clamp_dims(w, h, max_w, max_h):
    """Shrink (w, h) uniformly so both fit within (max_w, max_h); aspect preserved, never grown."""
    shrink = min(1.0, max_w / w, max_h / h)
    return w * shrink, h * shrink


def grid_shape(n):
    """(rows, cols) for `n` panels: a single column for one, otherwise two columns.

    Pulled out of the render loop so the rebuild decision can be tested without a display."""
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    return rows, cols


# Fallback panel aspect (width / height) for a grid built before any image has been read. 1.4 was
# the only aspect until 2026-08-19, hardcoded as 4.2 x 3.0 inches per panel.
DEFAULT_PANEL_ASPECT = 1.4


def image_aspect(images, default=DEFAULT_PANEL_ASPECT):
    """Width / height of the first readable image, or `default` when there is none.

    One aspect for the whole grid rather than one per panel: the panels are the same kind of chart
    written by the same code, and a per-axes aspect would need a `GridSpec` with unequal rows for no
    real gain.
    """
    for image in images or ():
        if image is None:
            continue
        try:
            height, width = image.shape[0], image.shape[1]
        except (AttributeError, IndexError):
            continue
        if height and width:
            return float(width) / float(height)
    return default


def figure_dims(rows, cols, scale, aspect=DEFAULT_PANEL_ASPECT):
    """Panel-grid size in inches, shrunk uniformly (aspect preserved) to the laptop-safe budget.

    Uniform, not per-axis: clamping width and height independently would distort the charts.
    The shrink is 1.0 whenever the requested size already fits, so single panels and the
    desktop's own --scale are unchanged unless they would overflow.

    **`aspect` is the panel's, and it has to come from the image** (2026-08-19). Each panel was a
    fixed 4.2 x 3.0 inches, aspect 1.4, while `imshow` preserves the image's own aspect inside that
    box -- so every panel was letterboxed by whatever the two disagreed about, and a 2x2 grid showed
    the difference twice with the slack piling up between the rows. The eval chart is 1.11 and the
    training chart 1.62, so both were wrong, in opposite directions. Matching the box to the image
    means the chart fills its panel and the window's whole area is chart, which is what makes the
    text inside it bigger.
    """
    aspect = aspect or DEFAULT_PANEL_ASPECT
    panel_w = 4.2 * scale
    return clamp_dims(cols * panel_w, rows * panel_w / aspect, MAX_FIG_W_IN, MAX_FIG_H_IN)


def apply_tight_grid(fig, gap=0.01):
    """Push the panels out to the figure edges with a hairline between them.

    Replaces `tight_layout`, which reserves room for the titles, labels and ticks a panel of bare
    `imshow` axes does not have -- roughly 8% of the height on a 2x2 grid, all of it visible as a
    band between the rows. `gap` is in axes-height/width units, so 0.01 is a seam rather than a
    margin.
    """
    try:
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=gap, hspace=gap)
    except Exception:
        pass


def fit_figure_to_screen(fig, w_in, h_in):
    """Best-effort second shrink to the display the window actually opened on.

    `figure_dims` is a fixed laptop-safe fallback; this reads the real screen from Tk and shrinks
    to 95% of its width and 88% of its height (leaving the menu/title bars), so a small screen is
    handled and a large external one keeps the fallback size. Only ever shrinks. Wrapped in a bare
    except because it touches the Tk backend, which is not guaranteed present, and a failure here
    must not stop the viewer from drawing."""
    try:
        win = fig.canvas.manager.window
        dpi = fig.get_dpi()
        screen_w = win.winfo_screenwidth() / dpi
        screen_h = win.winfo_screenheight() / dpi
        fit_w, fit_h = clamp_dims(w_in, h_in, screen_w * 0.95, screen_h * 0.88)
        if (fit_w, fit_h) != (w_in, h_in):
            fig.set_size_inches(fit_w, fit_h)
    except Exception:
        pass


# Darwin's built-in panels are 2x Retina, but the conda Tk build reports `device_pixel_ratio` 1
# (Tk scaling 1.0) — so matplotlib renders the figure at 1x and the macOS compositor upscales it,
# which is what makes every laptop chart look soft however sharp the source PNG is. Rendering the
# figure at 2x dpi gives it the pixels the panel actually has, so it is crisp at the same perceived
# (logical) size. `fit_figure_to_screen` already divides screen px by the real dpi, so the window
# still fits. Linux keeps 100: its 1x display shows the figure pixel-for-pixel and 2x would only
# waste pixels and shrink the window. Measured 2026-08-15: dpr=1 on the built-in Retina panel.
VIEWER_HIDPI_DPI = 200


def viewer_dpi(platform, override=None):
    """The dpi to render the viewer's figure at. 2x on darwin (see `VIEWER_HIDPI_DPI`), 100
    elsewhere; an explicit `--dpi` wins either way. Pure so the platform rule is testable without
    a display."""
    if override:
        return override
    return VIEWER_HIDPI_DPI if platform == 'darwin' else 100


def make_figure(plt, rows, cols, scale, title, dpi=None,
                aspect=DEFAULT_PANEL_ASPECT):
    """Build the chart grid, then (re)install the signal handler — one operation, in that
    order, because separating them silently breaks the clean exit. See
    `install_signal_exit`: Tk overwrites the OS-level handler while creating this window,
    so any install that does not follow a `subplots()` call is dead code."""
    dims = figure_dims(rows, cols, scale, aspect)
    fig, grid = plt.subplots(rows, cols, squeeze=False, figsize=dims,
                             dpi=viewer_dpi(sys.platform, dpi))
    fit_figure_to_screen(fig, *dims)
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    install_signal_exit(plt)
    return fig, [ax for row in grid for ax in row]


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='PNG files to show (or use --glob)')
    ap.add_argument('--glob', default=None, help='glob for PNGs, re-evaluated each refresh')
    ap.add_argument('--watch', default=None, help='pgrep -f pattern; exit when no match remains')
    ap.add_argument('--arms', default=None, metavar='PREFIX',
                    help='show only the arms of batch PREFIX that are actually running '
                         '(re-checked each refresh, and remembered once seen)')
    # 1s refresh and 2x size are the laptop defaults: a chart is rewritten once per eval,
    # so re-reading every second costs a stat plus a small PNG decode and shows a new point
    # as soon as it exists. The desktop runner passes its own --interval/--scale.
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--scale', type=float, default=2.6, help='multiply the window size')
    ap.add_argument('--title', default='snek charts')
    ap.add_argument('--dpi', type=int, default=None,
                    help='figure render dpi; default 200 on darwin (Retina), 100 elsewhere')
    return ap


def main():
    args = build_parser().parse_args()

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.ion()

    fig = None
    fignum = None
    axes = []
    wave_arms = []
    rendered_files = None
    rendered_aspect = None
    absent_checks = 0
    # Startup grace: training (and its first PNG) may not exist for a few refreshes,
    # so require the watch pattern to be absent on several consecutive checks first.
    GRACE = 6

    while True:
        if args.arms:
            files = wave_files(args.arms, wave_arms)
        else:
            files = sorted(args.files) if args.files else (
                sorted(globmod.glob(args.glob)) if args.glob else [])
        # Always log a *change* to the panel set, not only under SNEK_VIEWER_DEBUG. The 3-of-4
        # window on 2026-08-14 could not be diagnosed after the fact because the log was empty:
        # one line per change is nothing (a wave produces two or three), and it is the only record
        # of what the window actually showed. The env var still adds a line per refresh.
        if files != rendered_files or os.environ.get('SNEK_VIEWER_DEBUG'):
            sys.stderr.write('{0:.0f} viewer refresh: {1} panels {2}\n'.format(
                time.time(), len(files), [policy_from_png(f) for f in files]))
            sys.stderr.flush()
        n = max(1, len(files))
        rows, cols = grid_shape(n)
        # Decoded *before* the figure, because the panel box is sized to the image's own aspect --
        # a grid built first would have to be thrown away and rebuilt on the first refresh.
        images = []
        for path in files:
            try:
                images.append(mpimg.imread(path))
            except Exception:
                images.append(None)   # a half-written PNG: blank panel, the next refresh has it
        aspect = image_aspect(images)
        # Rebuild when the panel *set* changes, not only when the axis count does: a late arm can
        # arrive without changing rows*cols (3 -> 4 both want a 2x2), and rebuilding on the set
        # guarantees the new arm gets a panel rather than relying on positional reuse. The aspect
        # joins that test because the first refresh with a readable image is usually the one that
        # learns it, and a stale box would letterbox every panel until the set next changed.
        if (fig is None or len(axes) != rows * cols or files != rendered_files
                or aspect != rendered_aspect):
            if fig is not None:
                plt.close(fig)
            fig, axes = make_figure(plt, rows, cols, args.scale, args.title, args.dpi,
                                    aspect=aspect)
            fignum = fig.number
        rendered_files, rendered_aspect = files, aspect
        for ax in axes:
            ax.clear()
            ax.axis('off')
        for i, image in enumerate(images):
            if image is not None:
                axes[i].imshow(image)
        try:
            apply_tight_grid(fig)
            fig.canvas.draw_idle()
        except Exception as error:
            # A draw failure never matters to anything but this window.
            sys.stderr.write('viewer draw error ({0}: {1})\n'.format(type(error).__name__, error))

        # Closed by hand: without this the loop would spin on a destroyed figure, and
        # every redraw is another chance to abort in Tk. Closing the window means done.
        if fignum is not None and not plt.fignum_exists(fignum):
            print('chart_viewer: window closed, exiting')
            exit_now(plt)

        if args.watch:
            if _training_alive(args.watch):
                absent_checks = 0
            else:
                absent_checks += 1
                if absent_checks >= GRACE:
                    break

        # Pump GUI events across the interval with flush_events rather than plt.pause. plt.pause
        # starts a *nested* Tk event loop, which under a four-arm launch's CPU load has wedged the
        # whole refresh — freezing the panel set at whatever incomplete state startup produced, so a
        # late arm never got its panel (the recurring missing-chart bug). flush_events only drains
        # the pending events, so a stall in one drain cannot stop the next refresh; the short sleeps
        # keep the window responsive without blocking event processing for the full interval.
        deadline = time.time() + max(1.0, float(args.interval))
        while time.time() < deadline:
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            time.sleep(0.1)

    print('chart_viewer: watched training gone, exiting')
    exit_now(plt)


if __name__ == '__main__':
    main()
