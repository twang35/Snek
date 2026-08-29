# One wave, one controller: `eval_wave.py`

**Status:** proposed 2026-08-19. Scope and the `--chain` decision approved in conversation the same
day. **Phases 0-4 are written and their gates met.** Phases 0 and 1 are committed (`eval_plan.py` extracted, 29 definitions moved byte-identically; the cross-arm
`load` guard landed, and its premise is confirmed by measurement — **a lane switches arms in
0.00 s**). `eval_wave.py`, `--chain` and their 33 fixtures are **uncommitted pending review**; suite
**31 modules / 780 tests / 0 failed** and the desktop's **100 / 0 failed**, with **18 of 18** and
**19 of 19** mutants killed respectively. Phase 5's code is written and uncommitted; its gate needs an
idle box, and b44's close-outs were still running as of 17:57. Phase 6 not started, and its docs
should land with the deploy rather than ahead of it.

**Read the ‡ correction below before quoting this plan's speedup.** The measured makespan gain is
1.29x, not the 1.7-2.8x derived from lane-idle time.

**Headline claim corrected 2026-08-19 — see the ‡ note below:** the measured makespan gain is
**1.29x** (1.12x episode-normalised), not the 1.7-2.8x derived from lane-idle time. The case for
building it stands on the other grounds.

**One line:** replace the four independent `eval_checkpoints.py` processes — and the *two* separate
things that orchestrate them, a bash script on the laptop and the runner daemon on the desktop —
with one controller that owns a whole wave of policies, hands whole-checkpoint units to a pool of
worker lanes, and is launched identically on both hosts.

Two problems, one shape. The wave has no representation in code today: on the desktop it is four
ledger rows, on the laptop a bash loop. Everything below follows from giving it one.

## Why

### ‡ Correction (2026-08-19): the idle is real, but recovering it is worth ~1.1-1.3x, not 1.7-2.8x

The table below measures **lane-idle time** — how much of a wave's lane-seconds a four-process
close-out leaves unused — and reads it as the speedup a wave would recover. That step is wrong, and
a head-to-head benchmark says so.

Four b43 arms, one 8-checkpoint step list, seeded so 8 / 4 / 2 / 1 checkpoints were left to measure
(the shape a HOF wave really has, produced the way it really arises — some arms already measured).
40 episodes flat, gate 97, 4 workers per process or per lane, so both sides held 16 spawned workers:

| | wall clock | episodes | throughput |
|---|---|---|---|
| 4 x `eval_checkpoints.py` | **98 s** | 516 | 5.27 ep/s |
| 1 x `eval_wave.py`, 4 lanes | **76 s** | 448 | 5.89 ep/s |

**1.29x on the clock, 1.12x once normalised for the episodes each side happened to run** (the two
differ because abandonment is stochastic per checkpoint). The lanes came out 3/4/4/4 with 7 arm
switches, so the balancing itself worked exactly as designed.

**What the lane-idle arithmetic misses is that the baseline is not a fixed partition — the straggler
inherits the whole box.** The 8-checkpoint arm's per-unit time in the four-process run:

```
22.0s  14.3s   7.4s  11.3s  10.2s   7.6s   8.6s   6.5s      <- alone from about here
```

It runs **~2.9x faster** once its three siblings have finished, because 16 spawned workers on a
10-core laptop are heavily oversubscribed and the freed cores go straight to the one arm left. The
wave, by contrast, holds every unit at 12-22 s from start to finish — it is *always* saturated,
which is the point, but it means the comparison is 16-workers-shared against a baseline that ends up
running 4-workers-exclusive. The box recovers most of the theoretical loss on its own.

Modelled forward onto the worst real shape in the record — b40's HOF wave, 63/16/9/2 candidates —
with the two measured rates (18.6 s per unit at four processes, 6.5 s alone, interpolated as
`6.5 x n^0.76`), the baseline finishes in ~523 s against the wave's ~419 s: **~1.25x**. That is a
model, not a measurement, and it is offered only to say that the extreme shape does not restore the
old number either.

**Is it still worth building? Yes, and the reasons are mostly not speed.** ~1.25x on a 15-hour
continuation close-out is still three hours, and the rest of the case is unchanged and does not
depend on the ratio: one job where the desktop dispatched four plus four HOF follow-ons, no
`hof: pending` marker chain to lose results down, one definition of the eval protocol instead of
three, the 200-line bash reimplementation deleted, and lane utilisation reported for the first time
(a four-process wave cannot report it, because none of the four can see the others' idle time).
**But the plan should not have claimed a speedup it had derived rather than measured**, and the
sections below are left as written with this correction standing in front of them.

### The idle is measured, not assumed

Row-seconds summed per arm ≈ that process's wall clock, so `makespan / (total / 4)` is the speedup a
perfect balancer buys. Computed over every result file in `runs/`:

| pass | batch | rows per arm | makespan | balanced | gain |
|---|---|---|---|---|---|
| HOF-500 | b40 | 16, 63, 9, 2 | 1.04 h | 0.37 h | **2.84x** |
| HOF-500 | b37 | 43, 16 | 0.53 h | 0.19 h | **2.74x** |
| HOF-500 | b29 | 59, 64, 9, 1 | 1.53 h | 0.71 h | **2.14x** |
| HOF-500 | b42 | 17, 27, 6, 48 | 0.97 h | 0.56 h | 1.74x |
| close-out | b19 | 37, 22, 4, 490 | 1.86 h | 0.62 h | **3.01x** |
| close-out | b17 | 20, 306, 117, 131 | 4.82 h | 2.12 h | **2.28x** |
| close-out | b26 | 735, 1071, 907, 151 | 3.59 h | 2.57 h | 1.40x |
| close-out | b43 | 1297, 1305, 1325, 1378 | 14.28 h | 12.82 h | 1.11x |
| close-out | b44 | 251, 242, 253, 286 | 2.67 h | 2.66 h | 1.00x |

**HOF is structurally imbalanced** — `above:98` yields 1 to 64 candidates depending on the arm, so
three lanes routinely idle while one works. **Close-outs are bimodal:** `top50` selects similar
counts when arms behave similarly (b43, b44: 1.0-1.1x) and wildly different counts when they do not
(b17, b19, b26: 1.4-3.0x). Even at 1.11x, b43 is 1.5 hours.

This idle is real and not absorbed elsewhere, because `_dispatch` opens with
`if self.running or not desired: return` — a **hard wave barrier**, documented as *"NOTHING new
starts until the whole wave finishes … the box idles on the stragglers"*. Trainings and evals never
overlap either, so eval wall clock directly delays the next batch.

### The same logic exists twice

`hyperparamTuning/scripts/chain_closeout_after_training.sh` is a 200-line bash reimplementation of
the desktop chain:

| logic | desktop | laptop |
|---|---|---|
| gate constants | `HOF_EVAL_ENV`, `CLOSEOUT_THRESHOLD` | `CLOSEOUT_GATE`/`HOF_GATE`, with the comment *"copied from runner.py … if that changes, change this too"* |
| gate ordering invariant | `assert CLOSEOUT_THRESHOLD < HOF_THRESHOLD` | `if [ "$CLOSEOUT_GATE" -ge "$HOF_GATE" ]` |
| close-out finished before HOF selects from it | the `hof: pending` marker is set only on reap | an inline heredoc that re-reads the file for `complete` |
| launch N arms, collect per-arm exit codes | `launch.spawn` + the ledger | `pids="$pids $!:$a"` + `wait` |
| which arms exist | the ledger | `ls -d savedPolicies/<prefix>[a-z]-*` |
| the three `pgrep` traps | not applicable | all three documented in that script's header |

None of that is inherent. A controller that knows its own children needs no `pgrep`, and a chain
inside one process needs no marker protocol and no file re-read.

## Non-goals

- **No daemon on the laptop.** `eval_wave.py` is a job: it starts, works, exits. No election, no
  broker, no lock file, no persistent state. An agent launches it; when the wave is done the process
  is gone.
- **Training launch is untouched.** Trainings stay four independent `snek2.py` processes. The
  symmetric `train_wave.py` — whose payoff would be deleting `register_arm`, `ARM_REGISTRY_GRACE`,
  `wave_files`'s registry-union-scan and the claim lock — is a separate follow-on, listed under
  [Deferred](#deferred-deliberately).
- **No change to selection, gates, episode counts or the abandonment rule.** The unit is a whole
  checkpoint, so `EVAL_MIN_ACHIEVABLE`'s arithmetic is untouched and the property that *no ranking
  among rows that reach the gate can change* survives by construction.
- **No episode-level parallelism.** Considered and rejected — see
  [Rejected alternatives](#rejected-alternatives).
- **`EVAL_RENDER` and the batched `ParallelPyEnvironment` path stay in `eval_checkpoints.py`**, which
  remains for single-policy and by-hand work. The new controller supports independent workers only,
  which keeps the riskier path out of the new program.

## Design

### Process shape

```
eval_wave.py  (controller — no TensorFlow)
  ├── lane 0  thread ──> IndependentWorkerPool(EVAL_WORKERS) ──> 4 spawned TF workers
  ├── lane 1  thread ──> IndependentWorkerPool(EVAL_WORKERS) ──> 4 spawned TF workers
  ├── lane 2  thread ──> IndependentWorkerPool(EVAL_WORKERS) ──> 4 spawned TF workers
  └── lane 3  thread ──> IndependentWorkerPool(EVAL_WORKERS) ──> 4 spawned TF workers
```

`EVAL_LANES` (default 4) x `EVAL_WORKERS` (default 4) = the same 16 spawned workers the standing
eval-cost instruction already asks for, and the same ~3.7 GB. **A lane is a thread, not a process**,
because `IndependentWorkerPool` already *is* the "group of independent workers with `load`/`run`"
abstraction — so no new IPC protocol is needed at all. The thread only ever blocks on a
`multiprocessing.Queue`, so the GIL is irrelevant.

Two consequences worth stating. The controller imports no TensorFlow (selection, `plan_stages`,
`build_row` and the charts need only `json`, `numpy` and `matplotlib`), so **17 processes replace
today's 20** and ~0.9 GB of duplicate TF arena in the four parents goes away. And the controller must
**not** use `system_multiprocessing.handle_main` — that is what would drag TF back in; a plain
`if __name__ == '__main__'` guard is correct, since `IndependentWorkerPool` uses its own
`get_context('spawn')`.

**‡ That guard is mandatory rather than stylistic, and omitting it does not fail cleanly.**
`spawn` re-imports `__main__` in every worker, so an unguarded module-level pool construction starts
building pools *inside* the workers: measured 2026-08-19 as a **2-minute hang leaving 8 orphaned
`spawn_main` helpers**, with the `RuntimeError` about bootstrapping arriving only from the children.
So `eval_wave.py` builds nothing at module scope.

### The unit and the dispatch rule

A unit is `(policy, step, episodes, stage)` — one checkpoint, one stage, measured start to finish by
one lane, with the gate applied inside the lane exactly as `measure()` does today. Per-unit cost is
**~86 s for HOF** (b29b: 1.53 h / 64) and **~35 s for a close-out** (b43a: 12.65 h / 1297), so greedy
claiming lands within one unit of perfect balance — a worst-case tail of ~90 seconds.

Dispatch is stage-major, then policy, then step: `full` (0) before `screen` (1) before `confirm` (2).
A free lane takes the first unit it is eligible for. Stage-major order keeps each policy's result
file filling in the same order it does today, which matters only for how the live chart reads.

### Cross-arm work, and why it needs no affinity heuristic

I expected lanes to need policy affinity, because `IndependentWorkerPool(policy_name, ckpt_dir, n)`
binds `ckpt_dir` at construction and each worker pays ~8 s of TF import and graph build. It does not.
Reading `_worker_main`: the worker builds env and agent **once**, then serves `load`/`run`, and `load`
is `checkpoint.restore(ckpt_dir/ckpt-<step>)` into the same variables — already what happens per unit.
`SnakeEnvironment`'s `policy_name` only reaches `Game` (cosmetic); the network shape comes from
`arch.json`.

So the change is `load(step, ckpt_dir=None)`, and a lane can switch arms with **no rebuild at all**
provided the target's `arch.json` matches what it built. The controller reads every policy's
`arch.json` at startup and groups policies by arch signature (`fc_layer_params`, `num_actions`,
`obs_len`, `obs_era`, plus the categorical fields). Lanes belong to an arch group and only take units
from it; when arms differ, lanes are allocated across groups in proportion to estimated work. Within
a batch the arms always share an arch, so in practice every lane can take anything.

**Measured 2026-08-19, and the premise holds.** A real 2-worker pool built for
`b43a-lowlr-b29b`, loading its own checkpoint and then `b43b-lowlr-b29a`'s through the same workers:
**own 0.01 s, sibling 0.00 s, back 0.00 s, against 3.2 s to build the pool** — and both arms played
`[95, 95]`, i.e. filled boards, so the weights genuinely work in a network built for the sibling.
Cross-arm work is free, so the affinity heuristic this section hedged about is unnecessary and the
controller's scheduler can be plain greedy.

**‡ But the sidecars are not uniform, and a naive signature refuses arms of the same batch.**
`b43a/b/d` predate the `algo` and `perfect_game_reward` fields and carry neither key; `b43c` records
both. A raw `arch.get('algo')` therefore reads `None` against `'ddqn'` and splits one batch into two
lane groups. So `restore_signature` goes through `algo_of`, which already reads an absent `algo` as
the default — and `perfect_game_reward` is deliberately **excluded**: it is the objective the arm
trained toward, not a property of the weights, the greedy action is an argmax over one checkpoint's
own Q values, and it legitimately differs between arms measured side by side (batch 33 trained at
10). With that, all four b43 arms group into one lane pool.

**‡ Amendment 2026-08-19: lane eligibility is `(restore_signature, eval-relevant env)`, not the
signature alone.** A worker builds **one** `SnakeEnvironment` from its process's `SNEK_*` at startup.
Today each policy's eval is its own process and the desktop passes that policy's own training env
(`rec['env']`); a wave is one process and can carry only one. Of the knobs the env reads:

| knob | reaches a measurement? | wave rule |
|---|---|---|
| `SNEK_ZERO_OBS` | **yes** — zeroes observation indices, so the policy sees something else and behaves differently | **hard-fail** a wave that mixes disagreeing arms |
| `SNEK_CHASE_SAFE_SHAPING`/`_GATE`, `SNEK_FREE_SPACE_SHAPING`/`_GATE`, `SNEK_FOOD_DISTANCE_REWARD`, `SNEK_PERFECT_GAME_REWARD` | reward only. `perfect_percent` is immune (a perfect game is identified by **score**), but a row's `avg_reward` would be computed under the lane's config rather than the arm's | **partition lanes** — cheap, since arms of one batch always agree |
| `SNEK_TILE_PIXELS` | no — cosmetic, verified by a fixed-seed observation hash | ignore |
| `SNEK_V_MAX` | no — the support comes from `arch.json` (`support_from_arch`), not the env | ignore |

This only bites on a hand-assembled cross-batch wave; a batch launched by one script agrees by
construction. Enforcing it is better than remembering it.

**Sidecar vintage is *not* part of this.** `b43a/b/d` carry no `algo` and no `perfect_game_reward`
while `b43c` carries both, because each arm's sidecar came from the checkpoint it was seeded from
(b43c from b40, the others from b29). It is notational: `DEFAULT_PERFECT_GAME_REWARD = 100`, so
b43c's recorded `100.0` *is* a/b/d's absent value. Verified live — a pool built on b43a's old-style
sidecar loaded b43c and played filled boards, 0.014 s cold and 0.004 s once the per-directory check
is cached. **Do not "fix" this with `backfill_arch.py`**: writing those fields into an old sidecar
asserts a fact about how that arm trained, where absence is already read correctly by `algo_of` and
`reward_scale_of`.

**The guard is the important half.** The worker compares the target directory's `arch.json` against
its built arch and raises `ArchMismatch` rather than restoring — the same discipline
`policy_arch.assert_restorable` already enforces at build time, extended to the switch. CLAUDE.md's
`game_over` incident is exactly this failure: a checkpoint restores whenever the vector is the same
*length*, and nothing else checks that the values still mean what they meant.

### Output files: the invariant

**Every artifact stays exactly where it is, per policy, byte-compatible.**
`runs/<policy>_checkpoint_evals<suffix>.json` keeps every payload key it has today;
`evals/<policy>_eval_progress.png` is still one PNG per policy. That is what keeps `eval_progress`,
`select_checkpoints_above`, `run_report`, `refresh_charts.sh`, the desktop's publish globs and every
tuning doc untouched — and it is what makes the change reviewable, because the numbers must not move.

The controller is the single writer of each policy's file, calling the same `write_results` payload
builder after every completed unit, under a per-policy lock.

Three details change inside the payload:

- **`session_*` becomes per policy, and a `wave` block is added** (`policies`, `lanes`,
  `episodes_planned`, `episodes_done`, `session_seconds`). A policy's own row-seconds no longer
  predict its finish time, because lanes are shared — so `eval_progress` should prefer the wave block
  for the ETA when it is present. Getting this wrong is a documented failure mode: averaging over
  resumed rows once put b10b's ETA out by 3x.
- **`archive_existing_eval_pngs()` runs once**, not once per process. Today four processes race it and
  can leave up to four `evals/archive/<timestamp>/` directories for one wave.
- **`chart_viewer.spawn_for_eval(prefix)` is called once.** The 2026-08-19 double-window incident was
  four processes racing `claim_viewer_slot` inside one second; with one controller the eval path has
  no race to lose. The lock itself stays — training still needs it.

### The confirm barrier

The close-out's third stage depends on every screen for that policy, since it confirms the best
`EVAL_CONFIRM_COUNT` *of those screened*. The controller holds all the rows, so it picks finalists and
enqueues confirm units directly — cleaner than today, where the logic is inline in a single-policy
loop. Confirm units are therefore **published lazily**, after that policy's screens land.

Cost: late in a *single-policy* wave, lanes can idle at that barrier. With four policies the other
three keep them fed, and per CLAUDE.md a continuation batch's bill is almost entirely the uncapped
full-length tier, which is known up front. This is why close-out gains land nearer the measured
1.0-1.2x than the HOF 2.8x.

### `--chain`

```
eval_wave.py --chain top50 <policy> [<policy> ...]
```

Stage A runs `top50` for every policy at `EVAL_MIN_ACHIEVABLE=97`. Stage B, for each policy whose file
came out `complete`, selects `above:98` from that file and runs 500 episodes flat at gate 98 into
`_hof500`. One place asserts `CLOSEOUT_GATE < HOF_GATE`, and the runner imports those constants rather
than defining its own — which is what removes both copies. A policy with no qualifying checkpoint
contributes no units and is not an error; that is the normal outcome for most arms.

**As built** (2026-08-19), three details are worth having written down:

- **The recipe is a function of stage A's settings, not a constant dict.** `eval_plan.hof_settings`
  overrides what the recipe fixes — 500 episodes, flat, gate 98, `_hof500` — and *carries over* what
  it does not: the worker count and `EVAL_RESUME`, which belong to how the wave was launched rather
  than to the protocol. `runner.py`'s `HOF_EVAL_ENV` is a constant dict and therefore silently
  discards both.
- **The two suffixes compose rather than replace**, and `source_suffix` is a separate field from
  `suffix`. Stage B reads `<suffix>` and writes `<suffix>_hof500`, so a verification wave launched
  with `EVAL_OUT_SUFFIX=_check` cannot land on the real `_hof500` file. A mutant that spells stage
  B's source as its own output survived one fixture and was only killed by a second one written with
  a non-empty suffix — the empty string is also `select_checkpoints_above`'s default, so the obvious
  test cannot tell the two apart.
- **Stage B's arm list comes from the *files*, not from stage A's arms in memory.** An arm the wave
  skipped because everything was already measured is finished, not failed, and is owed its
  re-measure all the same; an arm whose file is missing, unparseable or `complete: false` is skipped,
  which is the same flag `select_checkpoints_above` refuses to read from.

### Resume

`EVAL_RESUME` and `load_finished_results` work unchanged, per policy, because the file format and the
one-writer property are unchanged. A killed wave loses only the units in flight — the same exposure as
today, since `write_results` already rewrites after every checkpoint. **No spool directory and no
claim files are needed**, which is the main thing the controller buys over the peer-to-peer design
that was considered first.

## The edits

### New and moved

| file | change |
|---|---|
| `eval_plan.py` **(new)** | the TF-free half of `eval_checkpoints.py`, moved verbatim: `select_top_checkpoints`, `select_checkpoints_above`, `plan_stages`, `pick_finalists`, `skips_screening`, `build_row`, `held_from_row`, `wilson_interval`, `equal_effort_pooled`, `make_abandon_test`, `achievable_percent`, `best_full_length_row`, `resolve_screen_episodes`, `load_finished_results`, `protocol_from_sources`, `resume_suffixes`, `backup_previous_results`, `archive_existing_eval_pngs`, `merge_checkpoint_evals`, and the threshold constants |
| `eval_checkpoints.py` | re-imports those names so it and its **90 tests** are unaffected; keeps `main()`, the batched path and `EVAL_RENDER` |
| `eval_wave.py` **(new)** | the controller: arg parsing, `--arms <prefix>`, arch grouping, lanes, greedy dispatch, per-policy writing and charts, `--chain` |
| `eval_workers.py` | `load(step, ckpt_dir=None)` plus the worker-side arch compare; `('load', step, ckpt_dir)` on the command queue |

### Desktop runner (phase 5, in full)

**`--chain` deletes the daemon's chain rather than changing it.** Stage B runs inside the wave
process, so the box's job graph becomes `training wave -> one eval wave job`:

| retires | why |
|---|---|
| `_auto_hof_jobs()` | the wave job runs the HOF stage itself |
| `wants_hof`, `AUTO_HOF_PRIORITY`, `HOF_EVAL_ENV`, `HOF_EVAL_ARGS`, `HOF_THRESHOLD` | recipe and gate move to `eval_plan.py`, imported by both hosts |
| the `hof: pending` marker and its `_reap` branch | no marker to set |
| `hof_for()` in `anticipated_queue` | nothing to forecast separately |
| `auto_hof` runtime flag | becomes `chain_hof`, passed as `--chain` |

What changes:

| file | change |
|---|---|
| `job.py` | `policies: [...]` with `policy` kept as sugar (`policies or [policy]`); `parse_job` validates it the same way. **`_StubJob` needs the same fallback** — it re-adopts a running job from a ledger record after a daemon restart, with no spec in hand, which is what breaks on deploy if missed |
| `runner.py` | `_auto_closeout_jobs` groups the pending markers into **one** job carrying all their policies. Safe because the wave barrier returns early while anything runs, so the marker set is closed before any dispatch |
| `launch.py` | `argv = [py, '-u', 'eval_wave.py', '--chain', selector] + policies`, plus `EVAL_LANES` beside today's `EVAL_WORKERS` |
| `runner.py` | `_publish_results` unions the globs over `policies` but **publishes per policy with per-policy error capture** — see the risk below |
| `runner.py` | `anticipated_queue` loses its HOF hop; `build_at_a_glance` gets simpler, since it already groups by `(batch, phase)` and four rows collapsing to one is that grouping finally being real. Optional: `phase_of` can read `stage` from the policies' result files so a wave shows `closeout eval` then `hof` as it moves |
| `config.py` | `max_evals` stops meaning "concurrent eval jobs" (it would read 1); `eval_lanes` joins `RUNTIME_DEFAULTS`, clamped to 4, with `lanes * workers <= 32`. `viewer_png_paths` / `sticky_wave_pngs` expand `policies` |

**The job id is the one genuinely fiddly decision.** `<policy>-closeout` is idempotent by
construction today, and the guard `if prior.get('state') in TERMINAL or eval_id in self.running`
keys on it. `<batch>-closeout` is not enough: **b20 ran 36 arms under one prefix, in nine waves of
four.** Recommended: `<batch>-closeout`, with a `-w<k>` suffix where `k` counts that batch's already
terminal closeout wave records — a pure function of the ledger that does not churn while a wave is
pending. **The hazard to pin with a fixture is id churn before dispatch**: `_scan_pending` runs every
poll, so an id that shifted as more markers appeared would let a partly-finished wave relaunch under
a new id and redo the work. Grouping only at dispatch, where the barrier guarantees the marker set is
closed, is what prevents it.

**Per-policy outcomes live in one record.** An arm can fail alone, so the record carries per-policy
state, and the authority for "did this arm's close-out finish" is the flag the bash script already
checks by hand: `complete: true` in `runs/<policy>_checkpoint_evals.json`. With `--chain` that check
moves *inside* the controller, where it gates stage B per arm.

`interrupted` recovery is unchanged and still per policy: relaunch with `EVAL_RESUME=1`, stage A
finds everything measured and exits in seconds, stage B resumes under its own suffix via
`resume_suffixes`. A reboot during stage B must not redo stage A — fixture.

**‡ Pre-flight: no outstanding `hof: pending` marker may be left in the ledger.** Deleting
`_auto_hof_jobs` deletes the only thing that reads that marker, so any pending HOF work becomes
invisible — no error, no queued row, just a re-measure that never happens. **This is live as of
2026-08-19:** the queue was paused mid-b44 to build this, so b44's four close-outs will reap and set
exactly those four markers. The resume is therefore *not* flipping `paused` back — it is queueing one
explicit `above:98` wave job for b44 first, which is also the new controller's first real exercise
(b44's HOF is the imbalanced shape the whole plan is aimed at). **The marker is not visible in `status.json`** — that payload flattens each ledger record to a
bare state string, so the `hof` field never reaches it. Check either from the box
(`ssh the-claw-den 'grep -c "\"hof\": \"pending\"" <LEDGER_PATH>'`) or, conservatively and from
anywhere, by looking for any `-hof` row in a freshly fetched `at_a_glance.queued` / ledger view.

**Test impact is the reason this phase is last.** `desktop/tests/test_runner.py` has **83 tests** and
roughly 30 are touched: all six `test_auto_closeout_*`, the seven `test_auto_hof_*` plus the two reap
ones, five `anticipated_queue` HOF tests, both `test_build_eval_command_*`, the two
`viewer_png_paths` ones, and `test_publish_status_folds_the_queue_into_the_ledger_in_order`. About
ten of those are deletions. New fixtures: id stability across polls, one wave job publishing four
policies with one of them failing, per-policy `complete` gating stage B, and `_StubJob` re-adoption
from an old-style record.

### Laptop

`chain_closeout_after_training.sh` is **deleted**, along with its duplicated constants, its heredoc
completeness check and its three `pgrep` traps. Its one irreducible feature — waiting for a batch's
trainers to drain so an overnight batch measures itself — becomes `--wait-for-trainers <prefix>`, a
bounded poll inside the job.

**Arm discovery is a positional batch id, not `--arms <prefix>`** (as built): a trailing token that
is not a policy directory is expanded through `arms_for_prefix`, so `eval_wave.py top50 b45` and
`eval_wave.py top50 b45a-… b45b-…` mean the same thing. One argument form instead of two, and the
same spelling works on both hosts — which is the property the whole plan is for. It matches on the
batch id rather than `startswith`, because `'b44a-x'.startswith('b4')` is true.

```
# what an agent runs on the laptop
PYTHONPATH=. python -u eval_wave.py --chain top50 b45

# what the daemon runs on the desktop
PYTHONPATH=. python -u eval_wave.py --chain top50 b45a-… b45b-… b45c-… b45d-…
```

### ‡ Three deviations in phase 5, and why

**1. The daemon carries no protocol numbers, and does not import them either.** The table above says
"the runner imports those constants rather than defining its own". It cannot: the daemon runs on
**base** miniconda python with `desktop/` as its working directory, deliberately, so that it can
start before the `snek` conda env exists — and `eval_plan` needs numpy. The rule is inverted
instead. `inherited_eval_env` *strips* `EVAL_EPISODES`, `EVAL_SCREEN_EPISODES`,
`EVAL_MIN_ACHIEVABLE`, `EVAL_ABANDON_FLOOR`, `EVAL_CONFIRM_COUNT` and `EVAL_OUT_SUFFIX` from the env
a close-out inherits, and lets the tool's own defaults decide. That removes **five** numbers and one
duplicated assert from `runner.py` rather than the two the plan counted, with no import at all.

**2. The runtime flag stays spelled `auto_hof`, not `chain_hof`.** `parse_runtime_config` rejects the
whole file on an unknown key and keeps the last-known-good config, so a deploy that landed before the
`ops` edit would reject every config until someone noticed — and after a daemon restart
"last-known-good" is the built-in defaults. The rename buys accuracy and costs a deploy hazard on a
box with no SSH backstop. The comment on the key says what the mechanism now is.

**3. The job id is `<batch>-closeout`, with `-w<k>` from the second wave on**, resolved as the first
free `k`. Two cases collapse into that one rule: b20's nine waves under one prefix, and a single
dispatch that splits a batch in two because its arms disagree about their eval-relevant env. The
second case is why `_closeout_id` also takes the ids claimed **in this same pass** — the ledger
cannot know about them yet, and without it both groups are handed `<batch>-closeout` and the second
silently replaces the first. That mutant is killed by a fixture.

## Phases

| phase | content | gate to the next |
|---|---|---|
| **0 done** | extract `eval_plan.py`, re-export from `eval_checkpoints` | **met**: the 29 moved and 3 kept definitions all byte-identical to HEAD (`ast.get_source_segment` diff); suite **30 modules / 740 tests / 0 failed**; 4 dead imports trimmed |
| **1 done** | `load(step, ckpt_dir)` + arch compare | **met**: live swap on one pool costs 0.00 s and plays filled boards; 7 new `policy_arch` fixtures, cross-arch and moved-support swaps raise; suite **747 tests / 0 failed** |
| **2 done** | `eval_wave.py`, one policy, one lane | **met**: on `b43a` steps 1447000-1449000 at 4 episodes, the wave's file and `eval_checkpoints.py`'s have identical key order, identical row-key order and identical values for all 20 protocol fields (only the stochastic rates differ) |
| **3 done** | multi-policy, multi-lane, greedy dispatch | **met, and it revises the headline claim**: measured makespan **98 s -> 76 s (1.29x)** on an 8/4/2/1 imbalance, 15 units at 4 lanes x 4 workers, lanes running 3/4/4/4 with 7 arm switches. Episode-normalised the gain is **1.12x**, and the reason it is not the 1.7-2.8x this plan predicted is below |
| 3a | the `wave` block in the payload | not written yet: the per-arm payload is byte-compatible today, and the wave-level ETA is an addition to it |
| **4 done** (laptop half) | `--chain` | **met**: the recipe and the gate are one definition, `eval_plan.hof_settings` + the module-scope `assert DEFAULT_MIN_ACHIEVABLE < HOF_GATE`; a no-candidate arm returns 0 with no arms built. `runner.py` still carries its own copies — deleting them is phase 5, because the daemon needs `snek2/` on its import path and a deploy |
| 5 **code written, deploy outstanding** | runner deltas + `desktop/tests/test_runner.py` | a queued b46 close-out dispatches as one job and publishes four policies. Code and 100 desktop fixtures are done (19 of 19 mutants killed); the gate needs an **idle box**, and b44's close-outs were still running. Three deviations from the table above, all recorded below |
| 6 | delete the bash script; update `CLAUDE.md`, `hyperparamTuning.md`, `desktop/README.md` | — |

## Tests

New fixtures in `tests/test_eval_wave.py`, plus additions to `tests/test_eval_workers.py` and
`desktop/tests/test_runner.py`.

### ‡ The screened protocol cannot be tested against real checkpoints, and that is why the fixtures exist

The flat path is verified against `eval_checkpoints.py`'s own output — the phase-2 gate above — and
that comparison is the real check on the arithmetic. **The three-stage screened path has no such
check available**, because every policy in `savedPolicies/` with a current observation era is a
*continuation* arm: `select_top_checkpoints('b43a-lowlr-b29b', …, 6)` returns **791 checkpoints, all
791 in the mandatory full-length tier and 0 screened**, whatever count is asked for. So no live
`top<N>` run on this machine's data produces a screen to confirm, and the screened path, the confirm
barrier, the plan correction and the error handling are pinned by fixtures driven by a fake pool
instead.

25 fixtures, and each of the 12 mutants below fails at least one of them:

| mutant | killed by |
|---|---|
| `on_error` does not advance the barrier | the arm waits forever for a screen that will never arrive |
| confirm on the first screen instead of the last | `pick_finalists` ranks against a partial field |
| `lane_key` compares the raw sidecar dict | b43's non-uniform `algo` splits one batch into two lane groups |
| no top-up: always ask for the whole target | a checkpoint stopped at 60 of 100 is restarted |
| the payload leaks the controller's own bookkeeping | `queued` / `_already` reach `in_flight` |
| no overshoot correction | `measurements_done` exceeds `measurements_planned` |
| `lane_split` may starve a group | its work becomes unrunnable and the wave hangs |
| `take` ignores eligibility | an arm is measured by a lane built for a different observation |
| `add` does not dedupe | a re-issued unit is measured twice |
| `finish` always marks complete | a HOF pass selects out of a truncated close-out |
| `batch_of` keeps the whole head | `b4` and `b44` become the same batch |
| the queue is not stage-major | screens run before the full tier |

**One of those fixtures was worthless on its first draft** and is worth recording, because it is the
failure mode CLAUDE.md warns about. The stage-order test used steps 1000 (full), 2000 and 3000
(screened) — ordered by step alone that is *also* full-then-screen, so it passed with the stage
ordering deleted entirely. The fix is that the full-tier step is now the **highest** one.

- **Dispatch is exhaustive and disjoint.** Every planned unit is measured exactly once across lanes,
  for a plan with uneven per-policy counts (1, 2, 9, 64 — b29's real shape).
- **Stage order holds.** No confirm unit is issued for a policy before all its screens are in.
- **Arch grouping.** Lanes never take a unit from another arch group; a mismatched `arch.json` raises
  rather than restoring.
- **Row construction is unchanged.** A recorded score/perfect/reward list replayed through the lane
  accounting builds a row byte-identical to `build_row`'s on the same list — the exact-equality check
  that a live A/B cannot give.
- **Gate arithmetic is unchanged.** `make_abandon_test` fires on the same episode index whether driven
  by the old loop or a lane, at 100 and at 500 episode targets.
- **One writer per policy.** Two policies completing units in the same instant produce two intact
  files with no interleaved write.
- **Chain gating.** Stage B runs only for a policy whose stage-A file is `complete`; a no-candidate
  policy yields no units and exit code 0.
- **`--arms` prefix matching** uses `batch_prefix`-style equality, never `startswith` — `b4` must not
  match `b44` (the trap already fixed once in `chart_viewer.live_arms`).

**Mutants that must fail a test** (a passing suite is not coverage — `group_obs` took a third
signature with all 24 tests green):

| mutation | test that must fail |
|---|---|
| drop the arch compare in `load` | cross-arch swap fixture |
| ignore stage order in dispatch | confirm-before-screen fixture |
| let two lanes take the same unit | exhaustive-and-disjoint fixture |
| remove the per-policy write lock | concurrent-write fixture |
| gate on the lane's episodes instead of the unit's target | gate arithmetic fixture |
| mark a policy `complete` with unmeasured units | chain gating fixture |

`tests/test_chart_viewer.py`'s existing rule applies to any new spawn path: **stub `Popen` as well as
`run`**, so a wrong assertion cannot open real windows.

## Verifying the numbers do not move

Split by what is deterministic.

**Exact, in tests.** Selection (`select_top_checkpoints`, `select_checkpoints_above`), the stage plan
(`plan_stages`), row construction and the gate are pure functions of their inputs. Run a finished
arm's real `_evals.json` and close-out file through both paths and assert identical requested steps,
identical stage plans and identical rows from identical samples.

**Statistical, on a live A/B.** Episode outcomes are not deterministic — and note
`EVAL_WORKER_THREADS` changes floating-point reduction order, which can flip an argmax on a near-tie.
So on one finished arm with ~12 explicit steps, run `eval_checkpoints.py` and `eval_wave.py` at the
same episodes and gate, then compare: same steps, same `episodes` per row, same payload keys, same
`min_achievable` / `screen_episodes` / `abandon_floor` / `selected_by`, and pooled rates inside each
other's Wilson intervals. A directional surprise here is a prompt to measure again, not evidence —
the 800-episode independent-worker scare reversed on a 1,000-episode paired rerun.

**Makespan, on a real batch.** b44's four close-outs are the recorded baseline (2.67 h makespan,
2.66 h balanced, so ~1.00x available) and any HOF wave is the interesting one. Report makespan and
total lane-busy time, not a speedup claim from one arm.

## Risks

| risk | mitigation |
|---|---|
| a lane crash silently drops units | `TAG_ERROR` already exists; the controller returns the unit to the queue, retries once, then fails **that policy** and continues the others — and never writes `complete: true` for a policy with unmeasured units, because HOF selection reads that flag |
| building pools from a non-main thread | all pools are constructed on the main thread at startup; threads only drive `load`/`run` |
| worker count creep past the memory band | refuse `EVAL_LANES * EVAL_WORKERS > 32`; spawned workers are ~230 MB each and ~40 hit the OOM killer |
| the controller becomes a second place that knows the protocol | it does not define constants or thresholds — `eval_plan.py` is the only definition, imported by the controller, `eval_checkpoints.py` and `runner.py` alike |
| ledger shape change orphans in-flight desktop work | ship phase 5 while the box is idle between batches, and add the `done`-record migration guard above |
| one failed push now withholds four arms, not one | `publish_results` has no retry and the box's DNS for github.com flaps — b40 sat unpublished for hours with a 98.2%/500 checkpoint in it. So publish per policy in a loop with per-policy error capture, and a partial publish still lands the other three |
| one wave job hides per-arm failure in `status.json` | the wave job reports per-policy state in its ledger record, and `at_a_glance` shows the batch label it already groups by |

**What would falsify the premise:** if a lane switching arms turns out to cost a pool rebuild in
practice — an arch mismatch within a batch, or a restore that does not land cleanly across
directories — cross-arm sharing costs ~8 s per switch. That is still cheap against an 86 s unit, but
it would make affinity worth implementing rather than skipping. Phase 1 answers it before anything
else is built.

## Rejected alternatives

**Static equal split of the checkpoint list.** Cheapest, and it fails on the axis with the most
variance: `EVAL_MIN_ACHIEVABLE` makes a checkpoint cost 20 to 500 episodes, a **~17x spread inside one
arm** (b40b's HOF: 63 rows, 62 abandoned). An equal *count* split is not an equal *time* split, and it
cannot move work across arms, which is where the HOF gain lives.

**Peer-to-peer work stealing between four equal processes.** Needs an `O_EXCL` claim protocol, and
CLAUDE.md documents the exact bug: `claim_viewer_slot` writes the pid as a second step after the
create, so an empty-but-present lock reads as *stale* and two claimants win. A controller deletes that
problem rather than fixing it, and it also owns the confirm barrier naturally, which the peer design
made awkward.

**Controller distributing episodes within a checkpoint.** Best theoretical balance; it damages two
properties this codebase paid for. The gate stops being arithmetic — 500 episodes at ">10 failures"
becomes four shards at ">2 of 125", a different and stricter rule, unless failure counts are shared
live across processes. And restores go 4x. What it buys is the ~90 s tail that whole-checkpoint
claiming already leaves, in exchange for a new IPC protocol on the path that produces every number in
this investigation. The degenerate case it would help — an arm with one candidate — is also the case
where the whole pass takes 36 seconds (b29d HOF: 1 row, 0.01 h).

## Deferred deliberately

- **`train_wave.py`.** Four independent trainers are why `register_arm`, `registered_arms`,
  `ARM_REGISTRY_GRACE`, `wave_files`'s registry-union-scan, the sticky panel rule and the claim lock
  all exist — every one of them reconstructs "what is this wave" from outside. A parent would know it.
  The desktop is already immune to that whole class precisely because its daemon *is* such a parent
  and passes explicit PNG paths. Payoff is simplification, not throughput; it adds a supervisor to
  multi-hour runs. Worth doing after the eval wave is proven.
- **`merge_checkpoint_evals` fidelity.** It drops `episode_scores` and the protocol fields
  (`min_achievable`, `screen_episodes`), so a merged file cannot be pooled or resumed correctly. Not
  on this path — the controller writes one file per policy and never merges — but the function is
  still reachable by hand and should either be fixed or documented as lossy.
- **The `above:<threshold>` selector for close-outs.** CLAUDE.md notes the remaining close-out lever
  is the selector, not the worker count. Orthogonal to this plan and it changes what gets measured,
  so it needs its own argument.
