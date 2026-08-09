# What is running, and what is next

The live file. Everything here is current state and forward plan — results and
conclusions live elsewhere so this stays short enough to actually keep accurate.

| file | contents |
|---|---|
| `runs.md` (this file) | what is running, what to run next |
| [`completedRuns.md`](completedRuns.md) | every arm that has finished: config, final numbers, verdict |
| [`findings.md`](findings.md) | what is established, what has been falsified |
| [`failureModes.md`](failureModes.md) | the four ways a policy degrades, and how to tell them apart |
| [`hyperparamTuning.md`](hyperparamTuning.md) | the protocol: metrics, how to judge, how to launch |
| [`charts.md`](charts.md) | progress graph per arm |

## Next up: batch 20 — FC layer shapes, on batch 19's base

**Designed 2026-08-08, not yet launched.** Nothing is running on the laptop; the desktop is idle with
an empty queue. Batch 19's arms were stopped at 2.00-2.42M and are the control.

**The question is the one thing nine batches of optimiser knobs have not moved: the ceiling.** Network
architecture has **never been varied in this project** — `FC_LAYERS` has sat at `(50, 100, 50)` since
batch 1 and there is not a single measurement of it in [`findings.md`](findings.md).

### Why batch 19 is the base, even though it lost

At the user's direction, and the reasoning is sound. Batch 19 was falsified on level but it **cut max
drawdown from 55.52 to 8.76, 4/4 seeds** — the strongest anti-forgetting result on record. The premise
of batch 20 is that **consistency and ceiling are separable**: keep the config that learns steadily and
change the function approximator to see whether a different shape lifts the level without giving the
steadiness back.

That makes the target explicit: **a higher ceiling at batch-19 drawdown**. An arm that raises `sef` or
peak trailing while drawdown climbs back toward batch 18's 55 has not answered the question — it has
just walked back to batch 18.

### ‡ Two variables move against batch 19, on purpose — and that shapes the whole design

**β anneal goes to 300k**, the current default, at the user's direction. Batch 19 ran the **1M**
schedule, so **batch 20 differs from batch 19 in two ways, not one**: `FC_LAYERS` and
`BETA_ANNEAL_STEPS`. This is a deliberate choice — 300k puts the full IS correction inside the window
where arms actually learn, which is the whole argument in the `snek2.py` comment — but it has a
consequence that must not be glossed over:

> **No batch-20 vs batch-19 difference can be attributed to architecture alone.** The β schedule moved
> too. What *is* clean is the comparison **among batch-20 shapes**, since they share β=300k and differ
> only in `FC_LAYERS`.

**That is why wave 1 re-baselines.** Running the control shape `50,100,50` at β=300k costs four arms and
buys two things: it isolates the β change against batch 19 (never measured — 300k is a new default that
no arm has run), and it becomes the true seed-matched control every later shape is read against. Without
it, every architecture result in this batch is confounded.

**Pin β explicitly rather than relying on the default.** The default already drifted once between
batch 19's launch and this design, which is exactly how a batch ends up testing something nobody
intended.

```
SNEK_FC_LAYERS=<shape>  SNEK_SEED=1..4
SNEK_BETA_ANNEAL_STEPS=300000
SNEK_TARGET_UPDATE_PERIOD=1000  SNEK_FOOD_DISTANCE_REWARD=0
SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5  SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60
# PER otherwise at defaults, as batch 19 ran it: td_error priority, IS on, beta 0.4 -> 1.0
```

`SNEK_FC_LAYERS` must also be set on **every eval and `watch.py`** for these checkpoints — see the trap
below.

### The shapes, and what each one is for

30 inputs → 3 actions (`left/right/forward`), ReLU, He init. Parameter counts include biases.

| shape | params | vs control | depth | what it isolates |
|---|---|---|---|---|
| `50,100,50` (control) | 11,853 | 1.00x | 3 | batch 19 itself — already run |
| `200,100,50` | 31,503 | **2.66x** | 3 | capacity up, shape preserved |
| `200,50` | 16,403 | 1.38x | 2 | **wide-early**: conjunctions of engineered features |
| `320` | 10,883 | 0.92x | **1** | **depth, at matched capacity** |
| `25,50,25` | 3,428 | **0.29x** | 3 | is capacity binding at all |
| `60,30,30,30,30` | 6,573 | 0.55x | **5** | deep and narrow |
| `93,93` | 11,907 | 1.00x | 2 | fills the iso-param depth ladder |
| `100,200,100` | 43,703 | 3.69x | 3 | escalation, only if `200,100,50` moves |
| `100,50,50` | 10,853 | 0.92x | 3 | lowest priority — a reshuffle at the same depth |

**Wide-early is the one with a mechanism behind it.** The observation is already high-level — per-action
safety triples, tail-following flags, food direction — so what the net needs is *conjunctions* of those
features ("safe left AND food left AND tail not adjacent"), not a deep hierarchy. Conjunctions want
width in the first layer, which is what `200,50` and `200,100,50` supply.

**`320` is the cleanest single finding available.** It holds parameters constant and removes all depth.
If it matches the control, depth is contributing nothing here, and every future architecture question
gets simpler.

**`60,30,30,30,10` was proposed and is deliberately widened to `...,30`.** A 10-unit final layer forces
all three Q-values through 10 ReLU units, where dead units cost a large share of the representation
permanently. Five layers at `lr 1e-5` with no normalisation will also learn slowly early, which lands
on the speed metric regardless of the final level.

### Waves — 4 seeds per shape, two shapes at a time

**Never fewer than 4 seeds per shape.** This domain does not resolve below ~10 pp at n=4, so four
shapes at one seed each would resolve nothing. Laptop (4 trainers) plus desktop (up to 4) runs two
shapes concurrently, ~7h per wave.

| wave | shapes | why this pairing |
|---|---|---|
| **1** | **`50,100,50`** (re-baseline at β=300k) + **`200,100,50`** | the control every later shape needs, run alongside the strongest ceiling candidate. Also measures the β change on its own |
| 2 | `200,50` + `320` | wide-early against depth-1, both read against wave 1's new control |
| 3 | `25,50,25` + `93,93` | is capacity binding at all, and the iso-param depth-2 rung |
| 4 | `60,30,30,30,30` | deep-and-narrow; only worth a wave if 2-3 showed any depth signal |
| — | `100,200,100` | escalate only if `200,100,50` moves the ceiling upward |

**Wave 1 pairs the re-baseline with a ceiling candidate rather than running two candidates**, because
β=300k moved with the architecture and something has to hold it still. It leads on capacity-up rather
than the scientifically cleaner `320` because the stated goal is a higher ceiling, and a smaller or
equal-capacity net is not a plausible route to one.

**If wave 1's `50,100,50` at β=300k beats batch 19 on its own**, that is a result in itself — the β
default change validated — and it also means every later shape has a stronger bar to clear.

### Pre-registration

| item | value |
|---|---|
| **control** | **wave 1's `50,100,50` at β=300k**, seed-matched — the only comparison that isolates `FC_LAYERS`. Batch 19 is a *secondary* reference and differs by β as well |
| horizon | batch 19 reached 2.00-2.42M, so the paired read against it caps at **~2.0M**. Run to `max_steps` 2,500,000 so arms can pass it |
| **co-primary (ceiling)** | **peak trailing** and `best_perfect30`. Batch 19: peak 94.16 mean (94.66 / 94.40 / 92.72 / 94.86); batches 11-18 all sat at 94.8-95.0 |
| **co-primary (must not regress)** | **max drawdown**, running-max definition. Batch 19: 4.94-12.84, mean **8.76**. A shape that raises the ceiling while drawdown exceeds ~25 has not answered the question |
| secondary | `strong_eval_fraction` — batch 19 was **13.82%** at 2.004M against batch 18's 31.60%. Recovering that *while* holding drawdown down is the jackpot outcome |
| test | exact paired permutation over the 16 sign flips, as batches 16-19 |
| abandon | **≥3 of 4** arms not crossing pf30 ≥ 40% by 800k — note `b19c` never crossed at all, so the control itself fails a 2-of-4 rule; or drawdown above ~25 on 3+ arms, which kills the premise |

### ‡ The trap that will silently ruin the evals

**`restore()` is called with `expect_partial()`, so a checkpoint trained at one width rebuilt at another
loads with no error and simply leaves the mismatched layers unpopulated** — the policy then plays like a
beginner. `under_the_hood.eval_fc_layer_params()` exists because `eval_checkpoints.py` used to hardcode
`(50, 100, 50)`.

So **`SNEK_FC_LAYERS` must be set identically on the training job, on every eval of its checkpoints,
and on any `watch.py` run.** This is the same failure class that once took a 90.3% champion down to
scoring 0, 0, 1. Batch 20's checkpoints are a new era: if any of them earns a
[`../hallOfFame/`](../hallOfFame/README.md) entry, the entry has to record its width.

### Launching it — wave 1 is 8 arms

**Naming:** `b20a-fcbaseseed<N>` for the `50,100,50` re-baseline, `b20b-fc200x100x50seed<N>` for the
capacity arm. `x` stands in for the comma so the policy name stays a clean directory name.

| arm | shape | host | seeds |
|---|---|---|---|
| `b20a-fcbaseseed1..4` | `50,100,50` | laptop | 1-4 |
| `b20b-fc200x100x50seed1..4` | `200,100,50` | desktop | 1-4 |

**Laptop** — one command per seed. Only `SNEK_SEED` and the policy name change between the four:

```
cd snek2
SNEK_FC_LAYERS=50,100,50 SNEK_SEED=1 SNEK_BETA_ANNEAL_STEPS=300000 \
SNEK_TARGET_UPDATE_PERIOD=1000 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_DISCOUNT=0.9975 \
SNEK_GUIDED_FRACTION=0.8 SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 \
SNEK_FORK_MIN_LENGTH=85 SNEK_FORK_MAX_STEPS=60 SNEK_MAX_STEPS=2500000 \
/opt/miniconda3/envs/snek/bin/python -u snek2.py b20a-fcbaseseed1 > /tmp/b20a1.log 2>&1 &
```

Setting `SNEK_FC_LAYERS=50,100,50` explicitly on the baseline arm is not redundant — it is what makes
the arm's `hyperparameter override:` startup line prove which width it ran.

**Desktop** — one JSON per arm in `queue/pending/` on the `ops` branch
([how](../desktop/README.md#driving-it-from-the-laptop)), seeds 1-4:

```json
{
  "id": "b20b-fc200x100x50seed1",
  "type": "train",
  "policy": "b20b-fc200x100x50seed1",
  "max_steps": 2500000,
  "priority": 10,
  "env": {
    "SNEK_FC_LAYERS": "200,100,50",
    "SNEK_SEED": "1",
    "SNEK_BETA_ANNEAL_STEPS": "300000",
    "SNEK_TARGET_UPDATE_PERIOD": "1000",
    "SNEK_FOOD_DISTANCE_REWARD": "0",
    "SNEK_DISCOUNT": "0.9975",
    "SNEK_GUIDED_FRACTION": "0.8",
    "SNEK_FORK_BRANCHES": "4",
    "SNEK_FORK_PROB": "0.5",
    "SNEK_FORK_MIN_LENGTH": "85",
    "SNEK_FORK_MAX_STEPS": "60"
  },
  "notes": "batch 20 wave 1 capacity arm. Any eval or watch.py on these checkpoints MUST set SNEK_FC_LAYERS=200,100,50"
}
```

**Confirm each arm got its config** — the one log line worth grepping:

```
grep "hyperparameter override:" /tmp/b20a1.log        # laptop
git show origin/ops-status:status.json                 # desktop: it appears under `running`
```

**Two desktop caveats for whoever launches.** `runtime.json` currently has `max_trainers: 2`, so it must
be raised to 4 on `ops` to fill the box. And **4 concurrent trainers there is unmeasured** — each trainer
spawns 10 parallel eval-env processes on top of its own, and only *evals* were benchmarked (4 × 10 workers
peaked at 12.8 GB of 14). Start at 2, sample RAM, then raise:

```
ssh -i ~/.ssh/snek_desktop claw@the-claw-den 'free -m | awk "NR==2{print \$3\" MB used\"}"'
```

## Record status

**~94%, and the ceiling has not moved in seven batches.** The record is `b17b-forkseed2` @1190000 —
**94.24%** over 5,120 fresh episodes (CI 93.6-94.8), level with the old record (`b14a` 93.5%/200,
`b15b` 93.0%/300) once measured properly. **The one real gain is speed:** it reached the frontier at
**1.19M steps** where `b15b` needed 3.2M and `b14a` 3.7M. Read the speed, not the level.

**A high selected reading is mostly sampling luck** — `b17b`'s 99/100 close-out rows re-measure to
~93%, and a *blind* grid over the same region reads **84%** against the selected rows' 96%. The rule:
an arm's selected full-length rows describe the checkpoints the screen liked, never the region — any
regional claim needs a position-chosen sample. Full derivation and the selection-reasoning error:
[`archive/runs-archive.md`](archive/runs-archive.md); measurement caveats:
[`findings.md`](findings.md#three-measurement-caveats).

**Batch 19 makes it nine batches flat.** Peak trailing 94.66 / 94.40 / 92.72 / 94.86 at ~2M, so
standard PER did not move the ceiling either — it moved how much of the time an arm sits near it, and
downward. Batch 18's peaks were 94.92-94.96.

**Outstanding, highest-value, in order:**

1. **Re-measure `b18b-tgt1000seed2` @1588000 over fresh episodes.** Its close-out row is
   **98.0% over 100** with 9 of 9 full-length rows ≥95% — the highest selected reading since `b17b`'s
   99/100. Selected highs have shrunk every single time (`b17b` 99→~93, `b15b` 97→93.0, `b11b`
   96→~94), so this is a candidate and not a record until it is measured against the standing
   94.24%/5,120. See [`completedRuns.md`](completedRuns.md#batch-18--target_update_period-1000-the-strongest-speed-result-and-20-rows-95).
2. ~~Promote `b17b-forkseed2` @1190000 to `../hallOfFame/`~~ — **already done 2026-08-08**, and this
   entry was stale. It is the folder's current-record entry with a full write-up:
   [`../hallOfFame/README.md`](../hallOfFame/README.md#the-current-record-942-over-5120-episodes-b17b-forkseed2-ckpt1190000).

## Closed batches (11-19)

One line each; full write-ups and per-seed numbers in [`completedRuns.md`](completedRuns.md),
superseded detail in [`archive/runs-archive.md`](archive/runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **19** | standard PER (`td_error` + IS on, β→1.0) | **falsified** — worse on all 5 pooled metrics, 4/4 seeds, p=0.125. Drawdown 55.5 → 8.8, but at a lower level |
| 18 | `TARGET_UPDATE_PERIOD` 8 → 1000 | primary moved: 102k faster to pf30≥40%, 4/4 seeds; drawdown improved. **Close-out done** — tightest eq-effort spread of any batch (74.8-81.9), 20 rows ≥95% |
| 17 | `FORK_BRANCHES=4` forked endgame collection | null on config (one seed carried it), **project record on `b17b`**; dose ~60% of design |
| 16 | `FOOD_DISTANCE_REWARD=0` | **first non-null** — `sef` +11.35 pp at 1.25M; consolidation not ceiling. Needs replication |
| 15 | `N_STEP_UPDATE=3` | falsified on speed — 128k slower; evals null |
| 14 | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13 |
| 13 | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| 12 | eps handover 0.05 | deadlocked, abandoned 4/4 |
| 11 | the 30-value vector itself | +4-5 pp vs 10, not significant |

**The binding constraint is seed variance, not ideas** — n=4 resolves nothing below ~10 pp, and peak
trailing reads 94.7-95.0 flat across batches 11-18. Nothing has raised the ceiling; batch 16 raised
how much of the time an arm sits near it. **Batch 19 is the first batch to move peak trailing at all,
4/4 seeds, and it moved it down** to 94.16 — the invariance is breakable, just not yet upward.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **closed as batch 18** — primary moved, 4/4 seeds; see [`completedRuns.md`](completedRuns.md) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| ~~`FC_LAYERS=128,128`~~ | capacity | **superseded by batch 20**, which sweeps eight shapes rather than one — see [above](#next-up-batch-20--fc-layer-shapes-on-batch-19s-base). Prior was "low", and the nine-batch flat ceiling is why |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**Epsilon ladder shape — rewritten 2026-08-04, and it is now the highest-value untested
change on this page.** The old ladder ran **96.8% of batches 10-11's training steps at epsilon
exactly 0.0**, bottoming out at median step 15000 while 7 of 8 arms were still at 0% perfect
games, because its rungs were calibrated to `avg_reward` values a non-winning policy clears.
Replaced with two phases — `avg_reward` bootstrap to `INITIAL_EPSILON`/8, then a geometric
descent driven by the trailing-30 perfect rate — neither a ratchet, floor 0.002, and exactly 0
rejected at startup. Design and the measured diagnosis:
[`hyperparamTuning.md`](hyperparamTuning.md#the-epsilon-schedule--rewritten-2026-08-04-and-it-breaks-curve-comparability).

Two consequences for planning. **Learning curves are no longer comparable to batches 1-11** —
checkpoints still load, but every earlier arm trained greedily from step ~15k, so graph shapes
are not like-for-like. And because the refinement phase is a pure function of current skill, a
declining arm now automatically explores more (`b11a` would have gone 0.0020 → 0.0087 across its
42pp drawdown), which makes this the first change in the backlog aimed at the post-peak decline
rather than at the ceiling.

## Explicitly not planned

- ~~**Reward changes**~~ — **retracted 2026-08-07, and the stated reason was wrong.** The claim was
  that changing a reward breaks comparability of `avg_score`. It does not: `avg_score` is a count of
  food eaten, so it and every eval metric derived from it are on the same scale whatever the rewards
  are. Only `avg_reward` changes scale. **Retracting it paid off immediately**: `FOOD_DISTANCE_REWARD`
  became tunable and batch 16's ablation of it is the first non-null in six batches. `FOOD_REWARD`,
  `DEATH_REWARD`, `STARVE_REWARD` and `PERFECT_GAME_REWARD` remain fixed, since those *do* rescale
  `avg_reward` enough to move the bootstrap epsilon thresholds a long way — but the batch-16 result is
  a standing argument for looking at the rest of the reward function rather than at another optimiser
  knob. See [`hyperparamTuning.md`](hyperparamTuning.md#available-knobs).
- **Reverting to `PyUniformReplayBuffer`** — cpprb is ~2.4x faster with no measured
  learning cost, so cheaper experiments come from keeping it.
- **An LR schedule** — no evidence of optimization instability; degradation is gradual
  in every arm, not spiky.
- **Adding more epsilon knobs** — the schedule was rewritten 2026-08-04 and already exposes
  `INITIAL_EPSILON` and `MIN_EPSILON`; the thresholds, windows and the 80% target are constants
  in `training.py` on purpose. Measure the new schedule before making any of them tunable, or
  the next batch will vary four things at once.
- **Setting `SNEK_MIN_EPSILON=0`** — rejected at startup. See
  [`findings.md`](findings.md#scope-of-that-falsification-added-2026-08-04-it-was-never-about-the-descent-rate)
  for why the batch-3 result does not license it.
- **`N_STEP_UPDATE=5`** — batch 15 measured n=3 and the predicted mechanism is absent: it reached
  pf30 ≥ 40% **128k later** than its control, 3 of 4 seeds slower, with level a null. The whole case
  for a larger n was that the effect scales with n, so its absence at n=3 is the worst possible sign
  for n=5. Contamination was never the obstacle either — 0.53% of targets at n=3, 1.06% at n=5 — so
  there is no cost to blame the null on. **Closed unless the propagation story is revived by
  something other than n.**
- **Resuming any arm from batch 10 or earlier** — every checkpoint on record from before
  batch 11 was trained on an observation vector this project has since changed (20, 23 or 26
  values against the 30 batch 11 trained on), so none of them load; see
  [`../hallOfFame/README.md`](../hallOfFame/README.md#the-entries-below-predate-2026-08-02-and-do-not-run-on-master).
  **Batches 11 onward are all resumable** — 11-15 share the 30-value vector. The arms worth resuming
  are the ones stopped mid-climb: **`b15a` and `b15d`, both still gaining in their final 500k band at
  5.5-6.0M**, then `b14c`, `b11d` and `b13c`. Resuming needs `SNEK_MAX_STEPS` raised above the arm's
  current step or it exits immediately.

### Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates,
what outcome would mean what — in this file for as long as any of its arms is running.
When the last arm of a batch stops, move that description and its results to
[`completedRuns.md`](completedRuns.md) and delete it here.

The reason to keep the description live rather than only the status table: the design
rationale is what tells a future session whether a surprising result is informative or
just an arm that was never going to answer anything.

Verify what's actually running with `pgrep -fl "python -u snek2.py"`. Not
`grep "[s]nek2.py"` — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count.
