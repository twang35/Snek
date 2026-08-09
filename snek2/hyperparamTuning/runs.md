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

## Batch 20 — FC layer shapes, on batch 19's base

**Wave 1 control done and closed out; capacity half is re-running to 3M (2026-08-09).** Per-arm numbers
and charts:
[`charts.md`](charts.md#batch-20-wave-1--fc-layer-capacity-fc_layers-the-first-architecture-test-stopped-at-173-278m).
Batch 19's close-out also finished on the laptop overnight.

- **Control `50,100,50` (`b20a-d`, laptop) is finished and closed out.** Resumed from ~2.5M and all four
  self-terminated at **3.000M**; close-out ran on the laptop 2026-08-09 in 45 min. It is now the
  seed-matched control every later shape reads against.
- **Capacity `200,100,50` (`b20e-h`, desktop, 2.66× params)** crashed together in the OOM→XIO cascade at
  **~1.75M** — checkpoints preserved, and no arm was on a bad trajectory (all `zero_since` null, still
  climbing). The cause is fixed since: the live chart window is off by default and replaced by the
  decoupled `chart_viewer.py`.

**Provisional read: 2.66× capacity did not move the ceiling.** Peak trailing **94.44** (control) vs
**94.50** (capacity), best-30 64.0 vs 65.6 — both inside the flat 94.7-95.0 band that every batch since
11 has held. But the two hosts stopped ~0.8M apart, so **this is not a matched-horizon comparison** and
`strong_eval_fraction` (a fraction of each arm's own evals) is not comparable across the two columns yet.

> **Treatment rerun RUNNING (launched 2026-08-09, desktop).** The four `200,100,50` arms resumed from
> their ~1.8M checkpoints to **3M** — daemon jobs `b20{e,f,g,h}-fc200seed{1-4}-3m` (new ids, same policies
> so they restore their own checkpoints), one 4-trainer wave. Steady ~4.5 GB, charts on the monitor. ETA
> ~8h from launch. **When done: fetch from the `results` branch into `runs/`, then read the treatment
> against the control at a matched ~2.5M horizon.** Only then decide wave 2 — it stays on hold until this
> control-vs-treatment comparison is settled (the β-moved-with-architecture confound needs it clean).
>
> Memory question that gated this is **answered**: 4 trainers × 10 *forked* self-eval workers cost only
> ~4.2 GB (COW-shared TF), nowhere near the 14 GB ceiling — the overnight OOM was the cv2/XIO cascade, not
> steady memory. Standalone (spawned) eval workers are the heavy ones: ~230 MB each, OOM-killer at ~52,
> safe budget ~40.
>
> **Amended 2026-08-09: wave 2's `200,50` half went ahead anyway, at the user's direction.** The hold
> above was about the *control* being settled, and it now is — `b20a-d` finished at 3M and closed out. The
> treatment rerun only gates the **capacity** verdict (`200,100,50`), not the wide-early one, which reads
> against the control. `320`, wave 2's other half, has **not** been run.

> **Wave 2 `200,50` RUNNING (launched 2026-08-09, laptop).** Four seeds, `b20{i,j,k,l}-fc200x50seed{1-4}`,
> `SNEK_MAX_STEPS=3000000`, all four confirmed on the `hyperparameter override:` line as `FC_LAYERS =
> (200, 50)` with β=300k and the rest identical to `b20a-d`. Logs `/tmp/b20[ijkl]-fc200x50seed*.log`.
> **Read against the control table below at a matched 3M** — both halves now run to 3M, which supersedes
> the pre-registration's 2.5M `max_steps`; the *batch 19* secondary read still caps at ~2.0M.
> 16,403 params, 1.38× the control, depth 2. **Any eval or `watch.py` on these checkpoints needs
> `SNEK_FC_LAYERS=200,50`.**

#### Control at 3M — the numbers every shape is read against

| arm | peak trailing | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20a` | 93.80 | 41.3 | 0.2% | 7.66 | 33.2% | 52.0% @2000k (n=25) |
| `b20b` | 94.84 | 78.3 | 16.2% | 4.26 | 62.7% | 83.3% @1802k (n=36) |
| `b20c` | 94.34 | 56.3 | 2.2% | 5.06 | 52.8% | 68.0% @1919k (n=25) |
| `b20d` | 94.76 | 80.3 | 26.3% | 4.68 | **71.3%** | 89.7% @384k (n=58) |
| mean | **94.44** | 64.05 | 11.2% | **5.41** | 55.0% | — |

**No arm produced a full-length row.** At gate 95 every measurement was abandoned, deepest 58 of 100
episodes, so the best-row column is a *bound* and is not comparable across arms or with earlier batches —
`pooled_equal_effort` is the exact column. Nothing cleared 95%, so no hall-of-fame candidate; the record
stays `b18b @1588000` at 97.6%.

**‡ β=300k is a wash on ceiling and slightly *better* on drawdown** — matched at batch 19's 2.004M
horizon, so the only difference is the β schedule:

| | control (β=300k) | batch 19 (β=1M) |
|---|---|---|
| peak trailing | 94.41 | 94.16 |
| best-30 | 64.05 | 63.27 |
| `sef` | 12.17% | 13.82% |
| max drawdown | **5.27** | 8.76 |

So the β default change is neither validated nor falsified on the ceiling — every gap is well inside
seed noise at n=4. What it does show is that **batch 19's anti-forgetting property survived both the β
change and 800k extra steps**: drawdown is 5.41 at 3M, against batch 19's 8.76 at 2.18M and batch 18's
~57. That was the premise of using batch 19 as the base, and it is holding.

**FC trap — permanent, not just for wave 1:** any eval / `watch.py` / close-out of a `200,100,50` or
`200,50` checkpoint **must** pass the matching `SNEK_FC_LAYERS`, or `restore()` silently mismatches the
net (see below).

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

### Launching a wave — the mechanics (reused every wave)

**Naming:** `b20<letter>-fc<shape>seed<N>`, with `x` standing in for the comma so the policy name stays
a clean directory name — e.g. `b20b-fc200x100x50seed1`. Wave 1 used `b20a-fc50seed<N>` (control) and
`b20e-h-fc200seed<N>` (capacity).

**Laptop** — one command per seed; only `SNEK_FC_LAYERS`, `SNEK_SEED` and the policy name change:

```
cd snek2
SNEK_FC_LAYERS=<shape> SNEK_SEED=1 SNEK_BETA_ANNEAL_STEPS=300000 \
SNEK_TARGET_UPDATE_PERIOD=1000 SNEK_FOOD_DISTANCE_REWARD=0 SNEK_DISCOUNT=0.9975 \
SNEK_GUIDED_FRACTION=0.8 SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 \
SNEK_FORK_MIN_LENGTH=85 SNEK_FORK_MAX_STEPS=60 SNEK_MAX_STEPS=2500000 \
/opt/miniconda3/envs/snek/bin/python -u snek2.py b20<letter>-fc<shape>seed1 > /tmp/b20.log 2>&1 &
```

Set `SNEK_FC_LAYERS` explicitly even on the `50,100,50` control — it is what makes the arm's
`hyperparameter override:` startup line prove which width it ran.

**Desktop** — one JSON per arm in `queue/pending/` on the `ops` branch
([how](../desktop/README.md#driving-it-from-the-laptop)); same env block plus `"SNEK_FC_LAYERS": "<shape>"`,
`max_steps` 2500000, and a `notes` reminder that any eval/`watch.py` on the checkpoints must set the same
`SNEK_FC_LAYERS`. Confirm config landed: `grep "hyperparameter override:" /tmp/b20.log` (laptop) or
`git show origin/ops-status:status.json` (desktop, under `running`).

**Desktop concurrency — corrected by the wave-1 crash.** Wave 1 ran `max_trainers: 4` with the live
chart window **on**, and all four capacity arms died together at ~1.75M: an OOM disrupted the X session,
and the in-process cv2 window raised a fatal XIO error that took the arms with it. Two things changed as
a result — the live window is off by default now (decoupled viewer instead), and the **max-eval-worker
RAM test is still owed** before trusting `max_trainers: 4` again. Sample RAM early on any desktop wave:

```
ssh -i ~/.ssh/snek_desktop claw@the-claw-den 'free -m | awk "NR==2{print \$3\" MB used\"}"'
```

## Record status

**NEW RECORD, 2026-08-09: `b18b-tgt1000seed2` @1588000 — 97.6% over 700 fresh episodes** (683/700,
CI **96.1-98.5**). It beats the previous record, `b17b-forkseed2` @1190000 at 94.24%/5,120
(CI 93.6-94.8), with **non-overlapping intervals** — the first genuine move in the ceiling since the
30-value vector landed. Promoted to
[`../hallOfFame/`](../hallOfFame/README.md#the-current-record-976-over-700-episodes-b18b-tgt1000seed2-ckpt1588000),
copy verified to load and play.

**It is also the first selected high in this project that did not shrink.** Selected at 98/100, it
re-measures at **97.4% over 500** — a 0.6 pp change, against `b17b` 99→94.2, `b15b` 97→93.0,
`b14a` 96→93.5, `b11b` 96→~94. Nine batch-18 checkpoints above 95% were re-measured at 500 episodes
each and the mean shrinkage was **−5.2 pp**, so @1588000 is a clear outlier in the good direction
rather than a loose protocol.

**But it is a narrow peak, not a strong region.** `@1578000` — 10k steps earlier — reads **91.6%**,
5.8 pp lower. The old rule survives the better result: a selected row describes the checkpoint the
screen liked, never its neighbourhood. Any regional claim still needs a position-chosen sample.
Full derivation of that error: [`archive/runs-archive.md`](archive/runs-archive.md); measurement
caveats: [`findings.md`](findings.md#three-measurement-caveats).

### Max progression across batches

Best checkpoint per batch, at most three each, newest first. **Two columns, because they say
different things:** `selected` is the close-out's own best row and reads high by construction;
`re-measured` is a later independent sample and is the only column that can be compared across
batches. `*trunc*` means no full-length row survived the gate, so the figure is shorter and noisier.

| batch | change | best selected | re-measured |
|---|---|---|---|
| **20** w1 | `FC_LAYERS` 2.66x | *no close-out yet* | — |
| **19** | standard PER + IS | 91% @1536k *trunc* · 77% @1485k *trunc* · 76% @937k *trunc* | **none reached full length** |
| **18** | `TARGET_UPDATE_PERIOD` 1000 | **98% @1588k** · 97% @1601k · 96% @1289k | **97.6% /700** ← **record** · 94.7% /700 · 85.4% /500 |
| 17 | forked endgame collection | 99% @1248k · 99% @1205k · 98% @1231k | **94.24% /5120** · region grid 84% |
| 16 | `FOOD_DISTANCE_REWARD=0` | 93% @913k *trunc* · 92% @1203k · 85% @979k *trunc* | — |
| 15 | `N_STEP_UPDATE=3` | 97% @3245k · 95% @4697k · 91% @3671k | **93.0% /300** |
| 14 | `DISCOUNT=0.9975` | 96% @3702k · 93% @2559k · 90% @2261k | **93.5% /200** |
| 13 | eps handover + shield | 95% @986k · 92% @3367k · 91% @1166k | — |
| 12 | eps handover 0.05 | *deadlocked, not measured* | — |
| 11 | the 30-value vector | 96% @855k · 94% @671k · 88% @3507k | **~94%** (shrunk) |
| 10 ‡‡ | `DISCOUNT=0.995` | 93% @1695k · 90% @1501k · 85% @2344k | 74.9% /66000 pooled |
| ≤9 | earlier environments | `b8f` 92%, `b9d` 70%, `b7f` 51%, `b4c` 50% | not comparable |

**Read the re-measured column and the story is short: 94.2% for a year of batches, then 97.6%.**
Batches 11-19 all train on the same 30-value vector so they are comparable to each other; batch 10
(‡‡) and everything below it are earlier environments where the same checkpoint scores differently,
which is why those rows are not a trend line.

**The selected column is nearly flat from batch 11 on** — 93-99% in every batch that produced a
full-length row — which is exactly why it cannot be used to judge progress. Batch 17's three 98-99%
rows re-measured to 94.2%; batch 18's 98% re-measured to 97.6%. Same selected number, 3.4 pp apart in
reality.

**Outstanding, highest-value, in order:**

1. **Decide batch 20 wave 1's horizon** — rerun the capacity arm to 2.5M, or truncate the control to
   ~1.75M and read what exists. Wave 2 is blocked on it. See the decision note above.
2. **Consider a position-chosen grid around `b18b` @1588000.** The record is a narrow peak, so the
   open question is whether `TARGET_UPDATE_PERIOD=1000` produces a better *region* or just got one
   lucky checkpoint. A blind every-10k grid over 1.55-1.62M would settle it, and it is the same test
   that deflated `b17b`'s apparent region to 84%.
3. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which
   took ">95%" literally. ~9 minutes of eval if a fuller picture of the region is wanted.

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
| ~~`FC_LAYERS=128,128`~~ | capacity | **superseded by batch 20**, which sweeps eight shapes rather than one — see [above](#batch-20--fc-layer-shapes-on-batch-19s-base). Prior was "low", and the nine-batch flat ceiling is why |
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
