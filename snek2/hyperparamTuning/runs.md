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

## Batch 24 / 25 — FC width under IS-off (`SNEK_IS_WEIGHTS=0`, otherwise b22)

**b24 is running on the desktop; b25 is queued behind it.** Both take b22's exact IS-off config —
`td_error` priority α=0.6, `SNEK_IS_WEIGHTS=0`, target 1000, disc 0.9975, guided 0.8, fork 4/0.5/85/60,
food-distance off, seeds 1-4, 3M — and vary **only** the network width. That makes them batch 20's width
question re-asked under IS-off prioritisation instead of under the β→1.0 control.

**What each outcome would mean.** Batch 20 answered the width question under β→1.0 and found nothing:
nine shapes, no ceiling movement, and the consolidation columns exposed as seed noise
([`findings.md`](findings.md#network-shape-the-sweep-is-complete--nine-shapes-and-architecture-never-raises-the-ceiling)).
The β ladder then found the real consolidation lever — IS correction, worth +20.7 pooled from β→1.0 to
β→0.1. So batch 24/25 asks the one architecture question batch 20 could not: **does width matter once the
prioritisation is fixed?** A null replicates batch 20's finding under a stronger base and closes width for
good. A gain would mean the two interact — width only pays when the gradient is prioritised — which would
be the first architecture result in the project.

**Read it against b22, not against batch 20's control.** b22 (`50,100,50`, IS off) is the seed-matched
control for both arms: pooled **75.7**, best-30 86.2, `sef` 30.5, peak 94.88.

- **b24 (`b24a-d`, fc 320) — running (desktop), priority 200.** One wide layer, matching batch 20's `320`
  shape, which was that batch's apparent (and later downgraded) best. At **~440-475k of 3M** as of
  2026-08-12 17:15, ~83 min in, all four arms alive.
- **b25 (`b25a-d`, fc 200,100,100) — queued (desktop), priority 210.** 3-layer wide, ~1.6× the control.
  Runs after b24 and its close-outs.

**Beware the pooled comparison this batch is set up to invite.** `320` read +10.1 pooled over the control
in batch 20 and that was noise — an iso-capacity shape (`100,50,50`) landed 18.8 pp on the *other* side of
the control at the same parameter count. So a ~10 pp pooled gap here is **not** a result at n=4; peak
trailing and drawdown are the columns to read, and a consolidation claim needs more seeds.

Scheduler order (desktop `status.json` ledger): b24 training → b24 close-outs → b25 → b25 close-outs, all
auto-closed-out. Check with `git show origin/ops-status:status.json`.

## Batches 20-23 are closed — where their descriptions went

All four batches finished and closed out, so per the bookkeeping rule at the end of this file their
descriptions moved to [`completedRuns.md`](completedRuns.md):

| batch | change | verdict | design + results |
|---|---|---|---|
| **23** | IS β annealed **0→0.1** | **the best point on the β ladder** — pooled **75.7**, +20.7 over the control, higher on all 4 seeds | [write-up](completedRuns.md#batch-23--β-annealed-001-the-best-point-on-the-β-ladder-near-the-no-is-extreme) |
| **22** | IS **off** (`SNEK_IS_WEIGHTS=0`) | **dead heat with β→0.1** — pooled 75.7. The consolidation gain saturates by β→0.1 | [write-up](completedRuns.md#batch-22--is-off-a-dead-heat-with-β01--the-consolidation-gain-saturates) |
| **21** | partial IS (β→**0.5**) | beats the β→1.0 control (pooled 64.3 vs 55.0, 3/4 seeds), well short of no-IS | [write-up](completedRuns.md#batch-21--partial-is-β05-beats-the-β10-control-still-far-behind-no-is) |
| **20** | `FC_LAYERS`, **nine shapes** | **architecture never raises the ceiling**; capacity binds only below ~0.55× | [the sweep's design](completedRuns.md#batch-20--the-design-of-the-nine-shape-sweep-complete-2026-08-12) |

**The β ladder is the live result to build on.** Gradient concentration ESS/N walks down it — **β→1.0
≈1.0** (batch 20's control, near-uniform) → **β→0.5 ≈0.86** (b21) → **β→0.1** (b23, effective exponent
α·(1−β)=0.54) → **IS off ≈0.38** (b22) → b18's no-IS ≈0.21 — and pooled climbs monotonically with it:
**55.0 → 64.3 → {75.7, 75.7} → ~78.8**, then **flattens at the bottom**. Most of the consolidation is
bought by β→0.1; going all the way to IS-off adds nothing measurable. **The ceiling is unmoved throughout**
(peak 94.4-94.9), so the ladder buys time-near-the-ceiling, not a higher one.

`b23b` holds five full-length checkpoints ≥95/100 around 777k (best 97/100). It was **not** a
hall-of-fame candidate after re-measurement — the selected @777k reads 92.4% over 500 fresh episodes, the
worst of its own cluster. The `b23b` 217-242k collapse was investigated in place and is **not** an escape
from a local minimum; all four seeds make the same level shift
([`findings.md`](findings.md#-falsified-a-drawdown-is-not-how-a-policy-escapes-a-local-minimum)).

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
| **20** | `FC_LAYERS` shapes (**all 9 closed**) | 92% @1470k *trunc* (`b20ah`, 100,200,100) · 91% @1435k *trunc* · 91% @2935k *trunc* (`b20g`) | **none reached full length** — 0 of 36 arms |
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

1. **`CHASE_SAFE_SHAPING` is the highest-value untested change, and its hold has expired.** It was
   approved 2026-08-11 and held until the then-running batches closed out; batches 20-23 are all closed and
   only the desktop is busy. It targets endgame food-finding — the modal failure since batch 16 — which is
   the one place a *ceiling* gain could still come from, now that architecture is closed and the β ladder
   has flattened. Plan: [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md).
   The laptop's four slots are free.
2. **Batch 24/25 will not settle the width question at n=4 if it reads on pooled alone** — see the warning
   in the batch 24/25 section. If b24 comes back with a ~10 pp pooled gap and nothing on peak, the honest
   next step is more seeds on b24, not a third width.
3. **Consider a position-chosen grid around `b18b` @1588000.** The record is a narrow peak, so the
   open question is whether `TARGET_UPDATE_PERIOD=1000` produces a better *region* or just got one
   lucky checkpoint. A blind every-10k grid over 1.55-1.62M would settle it, and it is the same test
   that deflated `b17b`'s apparent region to 84%.
4. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which
   took ">95%" literally. ~9 minutes of eval if a fuller picture of the region is wanted.

## Closed batches (11-23)

One line each; full write-ups and per-seed numbers in [`completedRuns.md`](completedRuns.md),
superseded detail in [`archive/runs-archive.md`](archive/runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **23** | IS β annealed 0→**0.1** | **the best point on the β ladder** — pooled 75.7, +20.7 over the control, higher on all 4 seeds |
| **22** | IS **off** | **dead heat with b23** at pooled 75.7 — the consolidation gain saturates by β→0.1 |
| **21** | partial IS (β→**0.5**) | beats the β→1.0 control (64.3 vs 55.0, 3/4 seeds), well short of no-IS |
| **20** | `FC_LAYERS`, nine shapes, 12.7× param range, depths 1-5 | **architecture never raises the ceiling** (peak 93.75-94.69 across the range, 0 full-length rows in 36 arms). Capacity binds only below ~0.55×. Consolidation columns exposed as seed noise |
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
trailing reads 93.8-95.0 flat across batches 11-23. Nothing has raised the ceiling; batches 16 and 21-23
raised how much of the *time* an arm sits near it. **Two things have moved peak trailing downward** —
batch 19's full IS correction (94.16, 4/4 seeds) and batch 20's 0.29× net (93.75, 4/4) — so the invariance
is breakable, just not yet upward. **Thirteen batches of optimiser, PER and architecture knobs have not
raised the ceiling once**, which is the argument for the reward-shaping direction at the top of the
backlog.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `CHASE_SAFE_SHAPING` — potential-based shaping on head/food/tail in one region | endgame food-finding, the modal failure since batch 16 | **the top item, and ready to launch** — full plan in [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md). Approved 2026-08-11 and held until the running batches closed out; **20-23 are now closed and the laptop's four slots are free** |
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **closed as batch 18** — primary moved, 4/4 seeds; see [`completedRuns.md`](completedRuns.md) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| ~~`FC_LAYERS=128,128`~~ | capacity | **closed by batch 20**, which swept nine shapes rather than one: none raised the ceiling — [findings](findings.md#network-shape-the-sweep-is-complete--nine-shapes-and-architecture-never-raises-the-ceiling). Width under IS-off is the one loose end, and batches 24/25 are on it |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |

**`LEARNING_RATE=1e-4` — only after a stability fix.** 1e-5 is very conservative and
the in-code comment already suggests 1e-4. With a stable target it may train several
times faster; on its own with `TARGET_UPDATE_PERIOD=8` it would probably make
instability worse. The order matters.

**Epsilon ladder shape — rewritten 2026-08-04, and still never measured in isolation.** Every batch from
13 on has run the new ladder, so it is baked into all of them and no arm separates it from the config it
shipped with; that is why it stays on this page even though the code change is long done.
The old ladder ran **96.8% of batches 10-11's training steps at epsilon
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

**Verify what is running on both hosts — neither check sees the other**, so a count is meaningless
without naming the box:

| host | check |
|---|---|
| laptop (4 trainers max) | `pgrep -fl "python -u snek2.py"` |
| desktop `the-claw-den` | `git show origin/ops-status:status.json` — read `counts`, `running`, and the `iso` heartbeat |

Not `grep "[s]nek2.py"` on the laptop — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count. And `git fetch` first: `git show origin/ops-status:...` serves whatever
was last fetched, which on 2026-08-12 was **17 hours stale** and showed four finished evals as still
running.
