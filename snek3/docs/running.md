# Running things

`tuned()` is the authoritative knob list and it is read in **two** files — `train.py` for what is not
algorithm-specific, and the algorithm's own `algo.py` (today only [`dqn/algo.py`](../dqn/algo.py)) for
the rest. This file is a summary of both.

```
conda activate snek3       # or /opt/miniconda3/envs/snek3/bin/python directly
cd snek3

PYTHONPATH=. python -u train.py <policy>                # train
PYTHONPATH=. python -u evaluate.py <policy> [selector]  # a stage-B wave
PYTHONPATH=. python -u evaluate.py <policy> one         # one checkpoint, in this process
PYTHONPATH=. python -u watch.py <policy> [step]         # a live window
PYTHONPATH=. python -u record_gif.py <policy|hof>       # -> gifs/, throwaway
PYTHONPATH=. python -m tools.chart_window               # the box's chart window, if it is not up
python -m pytest -q                                     # the suite; conftest.py handles the path
```

**A training opens the chart window itself** — one per box, every running arm in it, nothing to launch
(`../CLAUDE.md`). The command above is only for putting it back after closing it; killing it, closing
it and relaunching it are all free, because no training reads it, waits on it or reopens it.

**`conda run` buffers stdout**, even with `python -u` — a backgrounded run's log can stay empty for
90+ seconds while the process is fine, and `kill -9` then discards the buffer permanently. Call the
env's python directly for anything backgrounded.

**Always pass `smoke` for verification runs.** The policy name doubles as the checkpoint directory
under `savedPolicies/`, so `smoke` keeps throwaway output isolated and safe to delete.

## Knobs

Every hyperparameter comes from a `SNEK_*` environment variable, so variants run side by side without
editing files. Each override prints a `hyperparameter override:` line at startup — **that grep is how
a misconfigured control arm gets caught**, and it has been.

**It does not cover every knob, and the gap is the shaping set.** `hyperparameter override:` covers
everything read through `tuned()`, in `train.py` and in the algorithm's module alike — they share the
one function, which is why the split cost the grep nothing. The reward and shaping knobs are read by
`env/constants.py` **at import**, before the trainer's config exists, so they print no override line
— `SNEK_CHASE_SAFE_SHAPING`, `SNEK_CHASE_SAFE_GATE`, `SNEK_FREE_SPACE_*`,
`SNEK_FOOD_DISTANCE_REWARD`, `SNEK_PERFECT_GAME_REWARD`, `SNEK_ZERO_OBS`. For those, grep
**`reward config:`**, one line printed at startup from `vectorized/config.describe()`. Before it
existed, b2's shaping dose had to be confirmed by reading `/proc/<pid>/environ` on the desktop.

### Run control

| knob | default | notes |
|---|---|---|
| `SNEK_ALGO` | `dqn` | which algorithm to train. An unknown value is **refused by name** rather than defaulting, so an arm launched as something this build has no code for cannot quietly train DQN and be reported as the other thing |
| `SNEK_SEED` | 1 | seeds the network initialisation, the exploration coins, the replay sampler, the env's food and every eval. Recorded in `runs/<policy>.md`, so two arms of the same config are the same arm |
| `SNEK_MAX_STEPS` | 10,000,000 | **absolute**, not "run this many more" — `global_step` is restored on resume. An arm at its cap prints so and exits after its opening eval. **Counted steps, not game moves**: at the default `fork_branches=4` a DQN step is four moves, and every eval row carries `transitions` for that reason |
| `SNEK_MIN_CHECKPOINT_SCORE` | 40 | below this no checkpoint is written, so a short smoke run writes none and cannot resume. Set 0 to test resume |
| `SNEK_DEBUG` | 0 | verbose logging. For debugging, not status |
| `SNEK_TORCH_THREADS` | 1 | **measured 1.4x faster than one-per-core**: a 30 -> 320 -> 3 net has no op large enough to amortise a fork-join. Compounds when four arms share the laptop |
| `SNEK_CHART_WINDOW` | 1 | 0 opens no window. The test suite sets it (a suite that ran the loop opened real windows), the desktop sets it for benchmarks, and `runtime.json`'s `viewer: false` sets it for every job on the box |
| `SNEK_CHART_WINDOW_SCALE` | 1.0 | fraction of the screen the window fills. There is one window per box, so the default is the whole screen |

### Network and optimiser

| knob | default |
|---|---|
| `SNEK_FC_LAYERS` | `320` — one hidden layer, which is what every record-holding snek2 arm used. snek2's *code* default was `50,100,50`; no champion ran it |
| `SNEK_LEARNING_RATE` | 1e-5 |
| `SNEK_ADAM_EPSILON` | 1e-7 |
| `SNEK_BATCH_SIZE` | 128 |
| `SNEK_DISCOUNT` | 0.99 |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 |
| `SNEK_TARGET_UPDATE_TAU` | 1.0 (a hard copy). **Below 1.0 was applying twice per gradient step until 2026-08-29** — `train.py` called `maybe_update_target()` after `agent.update()`, which already calls it, so a requested 0.05 ran at 1 - (1 - 0.05)² = 0.0975. Invisible at 1.0, where a second hard copy is idempotent, so no arm ever run was affected. The *period* was never wrong |
| `SNEK_GRADIENT_CLIPPING` | 0 (off) |
| `SNEK_N_STEP_UPDATE` | 1 |

### Exploration

| knob | default | notes |
|---|---|---|
| `SNEK_INITIAL_EPSILON` | 0.4 | |
| `SNEK_MIN_EPSILON` | 0.002 | **exactly 0 is rejected, not clamped.** A fully greedy collect policy makes the buffer a closed loop on its own behaviour |
| `SNEK_GUIDED_FRACTION` | 0.8 | share of refinement-phase episodes where the epsilon coin's random move is drawn from non-fatal actions. **Never the greedy action** |

### Replay

| knob | default |
|---|---|
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100,000 |
| `SNEK_PRIORITY_EXPONENT` | 0.6 |
| `SNEK_IS_BETA`, `SNEK_IS_BETA_FINAL`, `SNEK_BETA_ANNEAL_STEPS` | 0.4, 1.0, 300,000 |
| `SNEK_IS_WEIGHTS` | 1 |
| `SNEK_INITIAL_COLLECT_STEPS` | 2,000 — collected before the first gradient step, at the initial epsilon rather than uniformly |
| `SNEK_REPLAY_RATIO` | 1.0 — gradient steps per **transition banked**, so the ratio is exact whether forking is on or off, and it already matches snek2's 1 gradient step per transition. **Do not lower it to reproduce a snek2 arm** — that makes snek3 *less* data-efficient than snek2 ever was; `SNEK_MAX_STEPS` is the comparability knob. As a dynamics change it is worth ~2x at batch 512 |
| `SNEK_COLLECT_ENVS` | 1 — collect lanes per iteration. **It is a width knob, not a speed knob, and it makes an arm slower.** Transitions/s rises ~1.46x at 8 lanes, but `SNEK_MAX_STEPS` counts *counted steps* and each one now banks 8x the transitions — so time to the same cap rises ~5.5x. Spend it only alongside a cap lowered by the same factor. It is also a dynamics change: 8 concurrent episodes feeding one buffer |

### Collection

| knob | default | notes |
|---|---|---|
| `SNEK_FORK_BRANCHES` | 4 | 1 is off. Advances one of several branches of the same game, so the buffer holds the consequence of the *untaken* endgame action too |
| `SNEK_FORK_PROB` | 0.5 | |
| `SNEK_FORK_MIN_LENGTH` | 85 | |
| `SNEK_FORK_MAX_STEPS` | 60 | |

### Rewards and shaping

`SNEK_PERFECT_GAME_REWARD` (100), `SNEK_FOOD_DISTANCE_REWARD` (0.001),
`SNEK_CHASE_SAFE_SHAPING` + `SNEK_CHASE_SAFE_GATE` (0, 85),
`SNEK_FREE_SPACE_SHAPING` + `SNEK_FREE_SPACE_GATE` (0, 85).

**Changing any of these changes the objective**, so they are the loudest thing an arm can carry.
`SNEK_PERFECT_GAME_REWARD` cannot be moved without re-deriving `SNEK_DISCOUNT` — see
[`invariants.md`](invariants.md) invariant 6.

### Stage A, inside training

| knob | default | notes |
|---|---|---|
| `SNEK_GRAPH_EVAL_EPISODES` | 100 | **pinned.** The stage-A gate is literally "95 of 100"; a different denominator is a different gate |
| `SNEK_EVAL_INTERVAL` | 1000 | **sets the checkpoint interval too, from the same value.** They must be equal — a checkpoint at a step no eval screens can never be measured — so there is one knob rather than two that can disagree. Lower it for a smoke test and nothing else. **Rounded up to a whole algorithm step**, which changes nothing for DQN (granularity 1) and is what lets an algorithm whose step is a whole rollout keep the equality; `runs/<policy>.md` records the interval the arm actually ran |
| `SNEK_EVAL_QUEUE` | **1** | hands stage A to shared worker processes instead of measuring it in the training loop. **Measured 3.83x** end to end at the tuned defaults (9.12 h -> 2.38 h per arm to 3M, four arms, laptop). It **changes the training** — bounded schedule lag, and stage-A rows are no longer bit-reproducible from the arm's seed — so `0` is what an arm being diffed against b1 or b2 must use. See below |
| `SNEK_EVAL_QUEUE_DEPTH` | **16** | how many checkpoints may be **unmeasured**. **This is both the schedule's blind spot and the throughput lever**: at 16 the epsilon and shield schedules read a measurement up to 16,000 counted steps old. 16 is where an arm stops being eval-bound — all 19 swept configurations at depth 8 sat pinned against their cap with the trainer waiting, while at 16 the queue drains and the arm reaches 94% of its unblocked rate. 24 regresses, 12 gets 63% of the gain. **0 is the verification mode** — the trainer measures each checkpoint before resuming, which reproduces an unqueued arm bit for bit and recovers nothing |
| `SNEK_EVAL_WORKERS` | **6** | worker processes **per box**, shared by every arm on it. Unlike depth this turns over: at four arms the sweep measured 4.84 h (2w), 3.94 h (4w), **3.21 h (6w)**, 3.27 h (8w), 3.55 h (10w) — past six, workers starve the trainers, whose unblocked rate falls 380 -> 306 st/s, and the box reaches 4% idle to run *slower*. **Idle CPU is not the target**; the fastest configuration leaves ~20% of the box free. Starting none is safe — the arm measures its own and runs at today's speed |

**Where the 1.67x comes from, and why it is not more.** The saving is the *streaming efficiency* —
a checkpoint measured inside a sustained round costs ~1.1-1.9 s against ~4.2 s drained — plus whatever
of the remainder overlaps with training. It is **not** free parallelism: total CPU work is roughly
conserved, and the laptop test was core-bound, with the arms' own training rate falling from 426 to
270 counted steps/s while two workers ran. A box with spare cores overlaps more; a saturated one gets
only the efficiency.

#### What the queue changes about the training, which is two things and not one

Both are pre-registered rather than slipped in as a speed-up, because `perfect_percent` steers
exploration ([`invariants.md`](invariants.md) invariant 2) and is a feedback loop rather than a readout.

- **The schedule's feedback lags by up to `DEPTH` evals**, bounded — the trainer takes the work back
  at the bound, so the worst case is today's behaviour and never an unbounded drift.
- **Stage-A rows stop being bit-reproducible from the arm's seed.** A queued measurement shares one
  wide `VecSnake` with the other checkpoints in its round and lanes migrate between them, which is
  exactly why streaming is faster — so a job cannot own its RNG and does not see the boards
  `eval_seed(seed, step)` would draw. The two instruments agree to 0.09 standard errors over 3,222
  rows (phase 2), so nothing is lost statistically, but **a queued arm cannot be diffed byte for byte
  against an unqueued one.**

**A queued row's trailing window is "the last N rows that have landed", not "the last N steps".** A
streamed round completes in lane order, so rows arrive out of step order; holding them back to fix that
serialised the arm behind the slowest member of a round and was worth nothing (measured: stalled at
step 9,000 for 55 s with 2 of 9 rows unmerged). The file itself stays in step order, and every row's
own content is independent of when its neighbours arrive.

**One column shifts, and nothing selects on it.** A queued row's `epsilon` and `guided_fraction` are
what the arm *ran under* — governing the interval **ending** at that step — where an unqueued row's are
what its own eval just set, governing the interval **starting** there. Exactly one row apart, measured.
The measurement half of every row — score, reward, perfect rate, trailing mean — is bit-identical, and
`screen:95` reads only that half.

**Nothing can deadlock.** A trainer at its depth bound measures the oldest checkpoint itself. In the
steady state it never asks whether a worker exists or is keeping up; reaching the bound *is* the
signal. So killing every worker mid-batch slows the arms down and cannot stop them. (The tail drain is
the one place that checks, and only to decide how long to wait for a backlog that is landing.)

**The desktop needs no changes to run it** — the workers are started by the first arm that wants one,
exactly as the chart window is, so a job spec turns it on with nothing else:

```json
{"project": "snek3", "id": "...", "type": "train", "policy": "...",
 "env": {"SNEK_SEED": "1", "SNEK_EVAL_QUEUE": "1"}}
```

Wave-barrier scheduling means trainings and evals never overlap on the box, so four trainers plus two
workers is six processes on sixteen cores and the 16-shard stage-B wave still comes afterwards.

**A run report's config rows are the knobs.** Every key in `runs/<arm>.md`'s Config table is its
`SNEK_` variable lowercased, so `| priority_exponent | 0.6 |` means `SNEK_PRIORITY_EXPONENT` with no
lookup. `tests/test_train.py` pins both the correspondence and the defaults in this file.

### Stage B — flags, not environment variables

**A deliberate split.** A hyperparameter is an env var so an arm's config travels with its launch and
is recorded; a *measurement* parameter is a flag, so a re-measure is a command you can read rather
than an environment you have to reconstruct. snek2 had `SNEK_EVAL_EPISODES`,
`SNEK_SCREEN_THRESHOLD`, `EVAL_MIN_ACHIEVABLE`, `EVAL_SCREEN_EPISODES`, `EVAL_CONFIRM_COUNT` and
`VEC_WAVE_PROCS`, and a result file recorded none of them.

| flag | default | notes |
|---|---|---|
| `<selector>` | `screen` = `screen:95` | which checkpoints. See [`../tools/step_selectors.py`](../tools/step_selectors.py) |
| `--episodes` | 500 | stage B's depth. 100 when reproducing a snek2 close-out |
| `--shards` | 4 | how parallel, and nothing else. Measured **96 episodes/s per shard** on the laptop with 4 — against 16.9 for a single checkpoint measured on its own, because a shard's lanes are refilled from the next checkpoint and never drain |
| `--label` | none | names the pass, so an A/B does not overwrite the file it is compared with |
| `--width` | derived | games in lockstep per shard. The engine picks; there has been no reason to override it |
| `--seed` | 0 | the food stream. Two seeds are two independent samples of the same policy |
| `--no-resume` | off | re-measure rows already on disk instead of skipping them |

### Diagnostics

| knob | notes |
|---|---|
| `SNEK_ZERO_OBS` | comma-separated observation indices to zero, for ablations. Zeroes rather than deletes, so the checkpoint still loads |
| `SNEK_CHART_SCALE` | 2.0 — the chart PNG's dpi is `100 x` this. A viewer magnifies the PNG ~1.5-2x, so a lower value looks blurry blown up |
| `SNEK_TILE_PIXELS` | window size, cosmetic only — every pixel constant derives from it. **Must be set before `env.render` is imported** |
| `WATCH_FPS` | default 60; drop to 20-30 to follow the moves. 0 is uncapped, which tops out near 180 because of the display flip |
| `SNEK2_PYTHON` | `/opt/miniconda3/envs/snek/bin/python` — the interpreter `tools/import_tf_checkpoint.py` runs its TensorFlow half under |
