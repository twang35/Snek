# One scheduler, and it owns the window — design proposal (2026-09-05)

Status: **approved 2026-09-05; phase 1 and phase 3 built the same day (laptop), phase 2 in progress (desktop).** Section 0 is the argument, 1–3 are the
design, 4 is the phasing, 5 is what is deliberately left alone, 6 is the open questions.

## 0. What is wrong, in one sentence each

| symptom | root cause |
|---|---|
| windows that never open (2026-09-03: a chart-less window held the eval slot 15 h; 2026-09-04: the previous pass's window lingered 60 s and the next pass ran an hour with none) | the window is opened by the *work* — every `train.py` and every `closeout.py` asks for it — so "should there be a window" is answered by N processes each knowing only about itself, and every gap between them needed another rule: a flock, a pid watch, three negative checks, a stand-by loop, a zombie check |
| more than one window (2026-08-29: five on the desktop, 6.6 mean per 8-arm launch) | same: N racers, one slot, a claim protocol. The flock fixed *that* race and left the lifecycle ones above |
| a hand-relaunched window that never closes and holds the slot | a second opener with its own rules, beside the trainers' rules |
| seven slot-0 eval workers (2026-09-03) | the same pattern one layer down: every arm races to start the box's six shared workers, through a slot-claim protocol |
| two boxes drift (pass naming pinned equal by a *test*; `training_env` vs `build_command`; `run_pass` vs `eval_command`; two `at_a_glance` builders until 2026-09-05) | two schedulers: `desktop/runner/runner.py` (1,282 lines) and `tools/laptop_batch.py` (546), each with waves, the chain, launch env, status |

The common cause is the one the user named: **the process that knows what is running is not the
process that owns the things that depend on what is running.** On both boxes there already *is* one
process that knows — the daemon on the desktop, the queue driver on the laptop, and the `laptop-run`
skill already forbids launching arms outside the driver. So the fix is not a better peer protocol. It
is to give the scheduler the window, and then to have one scheduler.

## 1. The window: owned by the scheduler, one per box

**The scheduler spawns the viewer, tells it what to draw, and closes it. Nothing else opens one.**

| today | proposed |
|---|---|
| every arm and every close-out spawns a viewer; an `flock` in `runs/.live/.window` / `.evalwindow` picks the survivor | the scheduler spawns **one** viewer per box and holds its `Popen`. "Is the window up" is `popen.poll() is None`. No lock, no slot file, no pid file to read |
| the viewer decides what to draw from the `runs/.live/` registry (training) or from a path list on argv (stage B), with sticky-panel and new-wave rules of its own | the scheduler writes `runs/.live/status.json` — the file it already produces for status publishing — with a `panels` list of PNG paths: every arm of the current wave while training (finished ones included, so a batch with one arm left still shows all of them), every arm's stage-B chart while a pass runs. The viewer reads that file every refresh and draws exactly what it says. **One writer, one reader** |
| the viewer decides when to close: registry idle 300 s, or watched pids gone 3 times, or a branch that forgot to check | the scheduler closes it: `popen.terminate()` when the box has been idle for the grace period, and on its own exit. The viewer also exits if the status file says `"window": "close"` or if its parent's pid is gone — so a scheduler that dies takes no window with it into eternity |
| two slots, because on the laptop a training and a close-out can overlap | **one window.** Under the scheduler a box runs a training wave or a pass, never both; the panels just change. A hand `closeout` beside a running batch is the thing the queue exists to prevent |
| `python -m tools.chart_window` / `tools.eval_window --watch-pid` relaunch by hand, each with its own close rules | `python -m tools.scheduler --reopen-window` drops a trigger file (the daemon's own `unlink`-is-the-test pattern); the scheduler kills its viewer if any and spawns a fresh one on its next poll. There is one opener, so a hand reopen cannot leave a second window or a window with the wrong close rule |
| a window you close by hand stays closed until the next arm starts | same rule, now explicit: the scheduler (re)spawns only when it **launches** a job and the viewer is not up. Closing it mid-wave keeps it closed until the next launch |
| the desktop forwards `DISPLAY`/`XAUTHORITY` into every job so its viewer can reach the monitor | forwarded into the one viewer instead. `SNEK_CHART_WINDOW=0` and `runtime.json`'s `viewer` are read by the scheduler, once |

**On a scheduler restart** (a deploy, a crash): the scheduler kills the pid its previous life wrote
into `status.json` (a pid it wrote itself — no pattern) and spawns fresh if anything is running. One
flicker per deploy; one code path. Adopting the old window by pid is possible and is what the daemon
does for jobs, but it is a second path for a cosmetic saving.

**What this deletes.** `chart_viewer.py` loses `take_window_slot`, `stand_by_for_slot`, `watch_step`,
`pids_alive`, `idle_close`, `wave_panels`, `live_panels`, the `--runs-dir`/`--slot`/`--watch-pid`
modes — roughly 250 of its 661 lines; what remains is a grid of PNGs that follows a file.
`eval_window.py` (138) and `chart_window.py` (214) go entirely, their `sizing`/`wanted` reads moving
into the ~80-line window owner. `live_runs.py` keeps the arm registry (see §3) and loses the lock
paths and `zombie`. `tests/test_chart_window.py` (691) and most of `test_chart_viewer.py` (300) are
replaced by tests of two pure things: the `panels` list for a given scheduler state, and "spawn once
per launch, never while up". `train.py` and `closeout.py` stop opening windows.

## 2. One scheduler; the daemon becomes a bus adapter

**`tools/scheduler.py` — grown from `laptop_batch.py`, run identically on both boxes.**

Its input is a **queue directory of batch directories** of desktop-format specs, which is the laptop's
`logs/laptop-queue/<batch>/` today. Its state is **the filesystem**, as the laptop's already is: an arm
is finished when its `_evals.json` step is at `max_steps`; a pass is done when its merged file exists;
an arm already live on the box (registry) is waited for, not relaunched. "Rerun the same command" stays
the whole recovery. It owns:

| responsibility | today lives in | note |
|---|---|---|
| waves, the barrier, the chain (stageb → hof5000 → hof30k), pass ids | both, duplicated | one copy |
| trainer launch env, close-out argv, smoke/benchmark defaults | `launch.build_command` and `laptop_batch.training_env`/`run_pass` | one copy; the `parse_job`-vs-`closeout.build_parser` pinning test stays, now trivially |
| the chart window | nobody, really | §1 |
| the trainer cap, shard pool size, threads, nice, paused/drain | `runtime.json` (desktop) / flags (laptop) | one local `runs/.live/scheduler.json`, re-read each poll. On the desktop the daemon writes it from `runtime.json`; on the laptop a human does. Single writer per box |
| `status.json`: running, queued (waves and owed passes, from the files), `at_a_glance`, `panels` | `runner._publish` + `anticipated_queue` (700 lines of forecast) / `Driver.jobs` (20 lines, because state is files) | the laptop's version, because the queue *is* the forecast when state is files |
| failed pass: run once, mark, surface, do not loop | ledger `failed` / in-memory `ran` set | a marker file `<batch>/.failed-<pass>` written by the scheduler; delete to retry |
| lifetime | driver exits when idle; daemon is forever | scheduler exits when the queue has no work. On the desktop the daemon respawns it when it drops new specs; on the laptop the skill starts it |

**`desktop/runner/` shrinks to what only the desktop has: the git bus.** Still stdlib-only on base
python, still shelling out — now to one scheduler instead of to N jobs.

| keeps | drops |
|---|---|
| `gitbus.py` (fetch, publish, unpushed retry, stale locks), `config.py` (runtime parse + clamp), `job.parse_job` (spec validation at the bus edge), `deploy.py`, `trigger.py`, the `deploy`/`restart` actions | `_dispatch`, `_auto_jobs`, `_measured_policies`, `mint_pass_id`, `_pass_id`, `anticipated_queue`, `_reattach`, `_StubJob`, boot-id logic, `launch.build_command`/`eval_command`, `RunningJob`, `update_throughput`, the ledger's job states, `build_at_a_glance` and friends (move to `tools/`) |

Its loop becomes: fetch → validate pending specs and **materialise** them into
`desktop/queue-local/<batch>/<id>.json` (gitignored; grouped by `batch_of(id)`, eval specs included) →
write `scheduler.json` from `runtime.json` → spawn the scheduler if the queue has work and none is
running → **publish results** for every arm and pass whose files say finished and that it has not
published yet (a small `published.json`, its own single-writer file; idempotent, restart-safe) → publish
`ops-status` = the scheduler's `status.json` + box extras (disk, load, head, unpushed, failed markers
under `attention`) + the laptop's `laptop-status` folded in. `laptop_status.py` becomes "push the
scheduler's `status.json` to the `laptop-status` branch".

Roughly: `runner.py` 1,282 → ~350; `laptop_batch.py` 546 → `scheduler.py` ~650; `test_desktop_runner.py`
1,209 → ~300, with the deleted assertions re-homed as scheduler tests where they still mean something.

**The constraint that shaped the old daemon still holds and still fits.** It stays stdlib-only so it
can start before the conda env exists and cannot be broken by a trainer change; the scheduler runs on
env python as the jobs always did. A broken scheduler deploy shows as "scheduler exited N twice with
work pending" under `attention`, and `ssh the-claw-den 'Snek/snek3/desktop/deploy'` is the backstop it
did not have when the rule was written.

## 3. Shards and single-writer files: mostly already right

**Stage B is the design the user described, and it should not change.** The close-out resolves every
arm's candidates up front, gives each shard a fixed slice of steps, each shard writes only its own
file after every row, and the controller does no work per episode: it counts rows, redraws the PNG,
merges at the end. One writer per file, no lock, no claim, and the snek2 bottleneck (a controller
banking every lane's episodes, 58 s of bookkeeping per 46 s of work) cannot recur because the
controller touches no rows until the merge. The cost of static slicing is a tail: a shard whose slice
drew longer episodes finishes late, and pooling across arms already fills those cores from the next
arm. A pull queue (shards taking steps from a shared list) would trade that tail for a claim protocol
— a rename-based one, like stage A's — and is not worth it for a few minutes per wave.

**Stage A's queue is a thread pool with a directory as the queue**, which is the shape the user is
used to: the trainer writes `<step>.req`, a worker takes it with one atomic `rename`, writes
`<step>.done`, and the trainer alone merges into `_evals.json`. No lock files. The one peer race in it
is **who starts the six shared workers** — every arm calls `ensure_workers`, guarded by the `os.link`
slot claim that replaced the `O_EXCL` one after the seven-workers incident. It has held since
2026-09-03, so it is not urgent, but it is the window pattern again and the same fix applies: the
scheduler calls `ensure_workers` **before** launching a wave. The trainers' own calls stay for bare
launches and become no-ops under the scheduler, so the race has one contender. Phase 3.

**The arm registry (`runs/.live/<policy>` = pid) stays.** It is one writer per file, the pid is the
truth, and it is what lets the scheduler adopt an arm a killed predecessor left running and count a
hand-launched trainer against the cap. It stops being the window's input.

## 4. Phasing — laptop first, one implementation of the window

| phase | what lands | where | when |
|---|---|---|---|
| **1** | `tools/scheduler.py` (from `laptop_batch`) with the window owner; `chart_viewer --follow`; `train.py`/`closeout.py` stop opening windows; delete `eval_window.py`, `chart_window.py`; `laptop-run`, `stop-run`, docs updated | laptop | at a batch boundary — the driver exits between batches, the scheduler starts on the next one. The desktop keeps today's flock window until phase 2; the known remedy still applies there |
| **2** | daemon → bus adapter; spec materialisation; results publish from files; `at_a_glance` moves to `tools/`; `laptop_status` simplified; `desktop-deploy`, `desktop-batch`, `progress-update` skills updated | desktop | `drain: true`, wait for the wave and its passes, deploy, `drain: false`. The desktop gets the unified scheduler and the window fix in one deploy |
| **3** (optional) | scheduler starts the stage-A workers before each wave | both | any time after 2 |

Phase 1 is ordered first because the window is the acute pain and because building the owner once,
inside the shared scheduler, is what keeps it from being written twice. Putting a window owner into
today's daemon *and* today's driver would be exactly the two-copies rule this repo already has.

**Migration notes for phase 2.** The box's ledger is not read by the new daemon: finished arms and
passes are read from `desktop/runs/` (present for every closed batch), so nothing re-runs; a pass the
ledger marks `failed` will be tried once more and then marked with the new file. `status.json` loses the
`ledger` block (hundreds of ids) — see open question 4. The `ops` branch and spec format are unchanged,
so nothing about queueing from the laptop changes.

## 5. Left alone, deliberately

- the git bus, its four single-writer branches, `--force-with-lease`, the unpushed retry
- stage-B sharding and merging; `results.py`'s refusals
- the stage-A queue's request/claim/done files and the trainer as sole writer of `_evals.json`
- spec format, batch naming, pass naming, the chain's numbers in `closeout.PASSES`
- `deploy`, `trigger`, the `deploy`/`restart` actions
- `progress_update`, `publish_pages`, the viewer site

## 6. Decisions (user, 2026-09-05)

| question | decision |
|---|---|
| one window per box, or two | **one**, panels switching between training and stage-B charts |
| window on a scheduler restart | **kill and respawn**; one flicker per deploy |
| window closed by hand | **stays closed** until the next launch |
| the `ledger` block in `status.json` | **dropped.** Not read; the docs record what has run, and where it ran does not matter |
| a gitignored local copy of the desktop's specs as batch directories | **fine** |
| wait for a boundary, or pause and restart | **either box takes a restart at any point**: arms run in their own session and the registry lets the new scheduler adopt them; shards resume on rerun; results publish from files. No pause needed. A **wave** boundary is still the tidy moment, because arms started by the old `train.py` hold the old flock window until they finish, and a scheduler started mid-wave would open its window beside it. A batch boundary is not required |

**Build hazard.** The laptop driver launches every arm from the working tree, so phase 1 is developed
in a separate git worktree and merged at the switch; a running batch must never see a half-edited
`train.py` or `closeout.py`. The desktop's `deploy` is a fast-forward, so the same holds there by
construction.
