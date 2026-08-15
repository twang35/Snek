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

## Batch 25 / 26 — more FC shapes under IS-off (b24 done, new record)

**b24 closed with the project's first architecture result — and a new record.** Width `320` under IS-off
pools **87.9** (eq-effort, gate 95) — **+12.2 over the b22 control and higher on all four seeds** — with the
ceiling unmoved at peak 95.00
([write-up](completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record)).
The HOF-500 re-measured all 199 ≥97%/100 checkpoints and **9 held ≥97%/500**: the new record is **`b24d`
@1342k at 98.0%/500** (490/500, CI [96.4,98.9], it *rose* from 97.0/100), tied by **`b24b` @2860k 98.0%/500**;
both are in [`../hallOfFame/`](../hallOfFame/README.md). b25 and b26 carry the width question to two more
shapes under the same IS-off base.

- **b25 (`b25a-d-r2`, fc 200,100,100) — fully evaluated (desktop); the first auto-HOF chain ran end to end.**
  3-layer, ~1.6× the control. Close-out (gate 95) pools a mean **86.0** — **+10.3 over the b22 control's
  75.7, within 1.9 of b24's 87.9** — so **the consolidation lift replicates at a 3-layer shape**: it tracks
  capacity, not width per se. Peak unmoved at 95.0. **The HOF-500 (gate 98) held nothing** — every ≥98%/100
  candidate was abandoned, none reaching 98%/500; the strongest, `b25b` @911k, was still 97.2% when gate-98
  stopped it at 392 episodes. **No b25 checkpoint enters the hall on the auto run**, though `b25b` @911k is a
  plausible ~97%/500 holder a hand gate-97 re-measure could still promote (the auto run's gate is 98).
  Charts in [`charts.md`](charts.md).

| mean of 4 | b22 control (`50,100,50`) | b24 (`320`) | **b25 (`200,100,100`)** |
|---|---|---|---|
| peak trailing | 94.88 | 95.00 | **95.00** |
| `sef` (training) | 30.5 | 66.0 | **63.8** |
| best-30 | 86.2 | 96.2 | **94.3** |
| close-out pooled (eq-effort, gate 95) | 75.7 | 87.9 | **86.0** |

  Per seed — close-out pooled: a 85.6 · b 85.5 · c 87.2 · d 85.9. HOF-500 best partial (all abandoned at
  gate 98): a 92.7% · b **97.2%** · c 95.3% · d 96.4%.

- **b26 (`b26a-d`, fc 100,100) — closed out (desktop); HOF-500 running but empty.** A shallower two-layer
  shape — does the lift survive without the depth/capacity b24 and b25 had? **No: it does not carry it.**
  Close-out pooled mean **79.2** is only **+3.5 over the b22 control's 75.7** (per seed: b 83.8 · c 83.2 ·
  a 80.0 · d 69.6), against b24's +12.2 and b25's +10.3. `b26d` is a weak seed (`sef` 13.8, pooled 69.6) but
  never died. **No ≥98%/100 checkpoint** — best full-length is `b26b`/`b26c` at 97.0% — so the auto-HOF-500
  (gate 98) selects nothing and lands empty; the record stays b24's. Charts (3M) in [`charts.md`](charts.md).

**Read b25/b26 against b22, not batch 20's control.** b22 (`50,100,50`, IS off) is the seed-matched
control: pooled **75.7**, best-30 86.2, `sef` 30.5, peak 94.88. b24 has answered the yes/no; b25 and b26
ask whether the gain is width itself or just more parameters, and whether it survives at shapes other than
one wide layer.

**The pooled gap is still an n=4 signal, not a settled ceiling.** b24's +12.2 is higher on all four
seed-matched controls (p=0.0625, the n=4 floor), which is cleaner than batch 20's within-batch confound —
but `320` once read +10.1 pooled in batch 20 and that was noise. The HOF-500 confirmed the batch owns 9
genuine ≥97%/500 checkpoints (not just inflated /100 highs), which firms up the *consolidation* claim; b25/b26
are the replication that separates width from parameter count. **b25 has now replicated the pooled lift (86.0,
+10.3) at a 3-layer shape — so the gain tracks capacity, not width specifically — but produced no ≥98%/500
checkpoint, so the *record* is still b24's alone.** **b26 (`100,100`) has now answered the shallow case: it
does *not* carry the lift — pooled 79.2, only +3.5 over control — so the gain needs real capacity (one wide
`320` layer or a deep `200,100,100`), not just any net wider than the control.**

Scheduler order (desktop `status.json` ledger): **b26 HOFs (running, empty — no ≥98% checkpoint)**, then the
queue is clear. b25 is fully
done — training → close-out → HOF, all four arms — and was the **first live run of the auto-HOF chain**,
which worked end to end. The HOF step auto-queues a 500-episode, gate-98 re-measure of each close-out's ≥98%
checkpoints (`auto_hof`, on by default, 2026-08-13); it only measures, promotion into `hallOfFame/` stays
manual. Check with `git show origin/ops-status:status.json`.

## Batches 20-24 are closed — where their descriptions went

All five batches finished and closed out, so per the bookkeeping rule at the end of this file their
descriptions moved to [`completedRuns.md`](completedRuns.md):

| batch | change | verdict | design + results |
|---|---|---|---|
| **24** | **fc `320`** under IS-off | **first architecture result + a new record** — pooled **87.9**, +12.2 over the control, all 4 seeds (ceiling unmoved). HOF-500: `b24d` @1342k **98.0%/500**, the new record | [write-up](completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record) |
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

**NEW RECORD, 2026-08-13: `b24d-fc320noisseed4` @1342000 — 98.0% over 500 fresh episodes** (490/500,
CI **96.4-98.9**). It edges the prior record, `b18b-tgt1000seed2` @1588000 at 97.6%/700 (CI 96.1-98.5), on
the point estimate — intervals overlap, so it is a narrow lead, **taken as confirmed under the folder's
500-episode standard** (we are not re-running for the 700-vs-500 difference). It is the first record to come
from a non-default architecture (`fc 320`, not `50,100,50`). `b24b-fc320noisseed2` @2860000 ties it at
98.0%/500. Both promoted to [`../hallOfFame/`](../hallOfFame/README.md), copies verified to load and play.

**Like `b18b` before it, `b24d` @1342k is a checkpoint that did *not* shrink** — it read 97.0/100 in the
close-out and **rose** to 98.0%/500 on re-measurement, the genuine-region signature. That is the exception,
not the rule: of the batch's 199 ≥97%/100 checkpoints only 9 held ≥97%/500, and `b24a`'s two 100%/100 highs
produced zero survivors. Selection inflates a /100 high by ~5-6 pp, which is why every HOF add must clear the
500-episode re-measurement first (see [`../hallOfFame/`](../hallOfFame/README.md)).

**The prior record, `b18b` @1588000 (97.6%/700), still stands as the deepest-measured strong checkpoint** and
was itself the first selected high that did not shrink (98/100 → 97.4%/500, a 0.6 pp change, against `b17b`
99→94.2, `b15b` 97→93.0, `b14a` 96→93.5, `b11b` 96→~94).

**But `b18b` @1588k is a narrow peak, not a strong region.** `@1578000` — 10k steps earlier — reads
**91.6%**, 5.8 pp lower. The old rule survives every better result: a selected row describes the checkpoint
the screen liked, never its neighbourhood — and the same caution applies to `b24d` @1342k, whose neighbours
have not been position-sampled. Any regional claim still needs a position-chosen sample.
Full derivation of that error: [`archive/runs-archive.md`](archive/runs-archive.md); measurement
caveats: [`findings.md`](findings.md#three-measurement-caveats).

### Max progression across batches

Best checkpoint per batch, at most three each, newest first. **Two columns, because they say
different things:** `selected` is the close-out's own best row and reads high by construction;
`re-measured` is a later independent sample and is the only column that can be compared across
batches. `*trunc*` means no full-length row survived the gate, so the figure is shorter and noisier.

| batch | change | best selected | re-measured |
|---|---|---|---|
| **24** | **fc `320`** under IS-off | 100% @1633k /100 (`b24a`) · 100% @2126k /100 (`b24c`) · 99% @1031k /100 (`b24b`) | **98.0% /500** ← **new record** (`b24d` @1342k) · 98.0% /500 (`b24b` @2860k) · 97.4% /500 (`b24c` @2982k) |
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
   only the desktop is busy. **Better supported as of 2026-08-14**: the endgame failure is not failing to
   *go and get* a safe meal but arriving at boards where there is none — eating the reachable food is fatal
   in 54% of losses
   ([`findings.md`](findings.md#-retracted-2026-08-14-the-positions-are-trapped--geom-counts-routes-that-eat-and-die)),
   so safe-chaseability is the right quantity to shape and distance-to-food is not. It targets endgame
   food-finding — the modal failure since batch 16 — which is
   the one place a *ceiling* gain could still come from, now that architecture is closed for the ceiling (b24 raised consolidation, not peak) and the β ladder
   has flattened. Plan: [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md),
   **revised 2026-08-14 and ready to execute** — the arms are now **b24's config** (`fc 320`, IS off, 3M)
   plus the one knob, seed-matched against `b24a-d` at no new compute cost, rather than the batch-23 base
   the plan was first written against. The revision also caught a live defect in the design: the potential
   is per-episode state that `Game.snapshot()` does not carry, and b24's config runs `FORK_BRANCHES=4`
   with forks at length ≥85, so every other endgame fork would have started its shaping from a stale
   value. Both hosts are idle; the desktop is preferred because it chains close-out and HOF-500
   automatically.
2. **b24 landed the pooled gap this warning was about (+12.2, all 4 seeds, nothing on peak), so the honest
   next step is more seeds on `320`, not only more widths.** b25/b26 test a different question — width
   versus parameter count — which is worth it, but they do not firm up the `320` gap. The HOF-500 is done and
   confirmed 9 genuine ≥97%/500 checkpoints (and a new record); what is still owed before the width×IS-off
   interaction is called established is **4 more `320` seeds**, to move the pooled sign-test off its n=4 floor
   (p=0.0625).
3. **Consider a position-chosen grid around `b18b` @1588000.** The record is a narrow peak, so the
   open question is whether `TARGET_UPDATE_PERIOD=1000` produces a better *region* or just got one
   lucky checkpoint. A blind every-10k grid over 1.55-1.62M would settle it, and it is the same test
   that deflated `b17b`'s apparent region to 84%.
4. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which
   took ">95%" literally. ~9 minutes of eval if a fuller picture of the region is wanted.

## Closed batches (11-24)

One line each; full write-ups and per-seed numbers in [`completedRuns.md`](completedRuns.md),
superseded detail in [`archive/runs-archive.md`](archive/runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **24** | **fc `320`** under IS-off | **first architecture result + new record** — pooled 87.9, +12.2 over the control, all 4 seeds (ceiling unmoved, 95.00). HOF-500: `b24d` @1342k **98.0%/500**, the new record |
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
| `CHASE_SAFE_SHAPING` — potential-based shaping on head/food/tail in one region | endgame food-finding, the modal failure since batch 16 | **the top item, and ready to build** — full plan in [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md), approved 2026-08-11, **revised 2026-08-14**: batch 24's config as the base and `b24a-d` as the seed-matched control, plus the `restore_snapshot` fix the fork path needs. **20-26 are closed and both hosts are idle** |
| **Free space in one piece** — as a graded potential and/or an observation | endgame packing, which is what decides whether the food lands somewhere edible | **new 2026-08-14**, and the largest per-policy separation on record: one-piece share at length 90-94 is **92% / 77% / 5%** for `b24d` / `b18b` / `b20d` ([findings](findings.md#-the-packing-property-the-records-keep-their-free-space-in-one-piece-and-it-separates-them-by-87-points)). `count_groups` already runs every step, so both forms are nearly free. **Correlational, n=3 checkpoints** |
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
- **Plasticity interventions — resets, ReDo, shrink-and-perturb, L2-to-init** — **closed 2026-08-14
  before being tried.** These are the standard response to the shape of this project's curves, and the
  mechanism they target is measurably absent: across 9 arms dormancy *falls* from a fresh-net control,
  centred srank ends at 95-99% of it, and a direct fit-a-new-target probe reads **0.96-1.52× a fresh
  net** with a paired 3M change of −0.021 to +0.022. `b20d` collapses 80.3 → 42.7 while its probe fit
  **rises**. See
  [`findings.md`](findings.md#-falsified-2026-08-14-there-is-no-plasticity-loss--the-collapsed-networks-fit-a-new-target-better-than-their-own-peak).
  The one real ageing signature is weight growth (1.4-2.7× init) with movement decay (3-10×), nearly all
  of it inside the first 500k — a **shrinking effective step size**, whose fixes are weight decay or a
  larger late learning rate, not resets. That variant is not planned either while `LEARNING_RATE` and
  `CHASE_SAFE_SHAPING` are ahead of it in the backlog.
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
