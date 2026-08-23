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

## What is running — 2026-08-22

| host | state |
|---|---|
| **laptop** | **idle.** No trainers, no evals. `b43`'s three instruments and the 1000-episode champion re-measurement are all finished |
| **desktop `the-claw-den`** | **`b45`'s close-out, wave 1 of 2.** `b45-closeout` (`b45a`, `b45c`) has been running 21.7 h and is in **stage B**, the chained HOF-500 re-measure, at 625 of 2755 measurements. `b45-closeout-w2` (`b45b`, `b45d`) is **queued** behind it — `max_evals` is 1 |

**`b45b` and `b45d` have not been measured at all yet**, and their absence from the desktop chart window is
that, not a finished pass: the viewer's panel set for an eval wave is every chart in `evals/` whose batch the
wave is measuring, and membership is the file existing on disk. Nothing has published either — `origin/results`
carries training artifacts only for all four arms.

**The two waves exist because the job predates the grouping fix.** `b45-closeout` was dispatched under the old
whole-env key, which split the batch into `{a,c}`, `{b}`, `{d}`; `1f2e9d96f` (2026-08-21 16:35) keys on
`runner.EVAL_RELEVANT_ENV` instead and re-grouped the remainder into one `w2`, so the batch now costs two waves
rather than three. It could not fold `{b,d}` into a wave already running.

**Do not raise `max_evals` to unblock wave 2.** Load average is 18.5 on 14 cores with 4 lanes × 4 workers
already, so a second job splits the same cores and both finish later. The lever on a continuation batch's
measurement bill is the *selector*, not parallelism.

**Verify each host separately — neither check sees the other**, so a count is meaningless without naming the
box. Laptop: `ps -Ao pid=,command= | grep "python -u sne[k]2.py"`. Desktop:
**`git fetch origin ops-status &&`** `git show origin/ops-status:status.json`, then read `counts`, `running` and
the `iso` heartbeat. **The fetch is not optional** — without it you are shown an arbitrarily old local
remote-tracking ref with no sign that it is old, which has produced three false "the daemon is dead" alarms.
Ladder and rationale: [`CLAUDE.md`](../../CLAUDE.md#there-are-two-compute-hosts--say-which-one-you-mean).

## Batches 42-45 — what happens if you keep training a champion — **yes, and lower is better down to `1e-7`: 4 → 187 → 874 rows ≥98%/500 across 1e-5/1e-6/1e-7, and `1e-8` is the frozen floor**

**The question nobody here has asked.** Every record in this project is a checkpoint some 2M-step arm
*passed through* on its way to a worse endpoint. No arm has ever been continued **from its own best
checkpoint**. So: does a champion that keeps training improve, hold, or decay?

**One experiment, now a three-rung learning-rate ladder on one set of checkpoints.**

| | desktop `b42` | laptop `b43` | desktop `b44` | desktop `b45` |
|---|---|---|---|---|
| learning rate | **1e-5** (the default, the rate these checkpoints were trained at) | **1e-6** | **1e-7** | **1e-8** |
| everything else | b29's config verbatim | b29's config verbatim | b29's config verbatim | b29's config verbatim |
| cap | 3M, absolute | 3M, absolute | 3M, absolute | **5M**, absolute — see below |
| state | **stopped at +385-421k — it decays.** Closed out and HOF-500'd — [write-up](completedRuns.md#batch-42--the-same-four-checkpoints-at-the-default-lr-1e-5-stopped-early-it-decays) | **finished on all three instruments** — [write-up in `completedRuns.md`](completedRuns.md#batch-43--continuing-the-four-best-checkpoints-at-lr-1e-6-a-record-region-10x-wider-than-anything-before-it-and-the-best-checkpoint-was-the-wrong-one-to-continue) | **finished on all three instruments** — HOF-500 landed 2026-08-20, all 2235 measurements | **trained to 4.10-4.42M and stopped.** Close-out running on the desktop as **two waves**: `{a,c}` is in its HOF stage, `{b,d}` is queued — see [What is running](#what-is-running--2026-08-22) |
| self-eval | pooled eq-effort mean **92.1**; its only ≥98%/500 rows are within 75k steps of its own seed | holds flat, `sef` 96.5-99.5, one seed hit a 100.0 best-30 window | **wins: 4 of 4 seeds over `b43`**, `sef` 98.7-99.9 | **flat on all four**, drift −0.6 to +0.3 pp over 2.8M steps; best-30 **98.3-100.0** on an equal-episode window (99.2 raw) |
| close-out (/100) | **≥98% on 2.2-14.1%** of its checkpoints | ≥98% on **6.0-38.7%** | **≥98% on 6.9-56.3%** — 3 of 4 seeds ahead of `b43`, and by a lot | **running** |
| HOF (/500) | 4 rows ≥98% in total, all within 75k of a seed | **187 rows ≥98%**, best 99.6% (`b43b` @1661k) | **874 rows ≥98%**, 90 at ≥99%, and **two 500/500s** (`b44a` @2798k, `b44b` @1886k) — [both selection artefacts](findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20) | **running**, wave 1 at 625 of 2755 |

`b44` existed because `b42` and `b43` bracketed the effect on the first try: dropping the rate 10× turned decay
into a hold. It asked where that stops, and **the answer is "not yet at 1e-7"** — see [the write-up](completedRuns.md#batch-44--the-same-four-checkpoints-at-lr-1e-7-the-best-rung-of-the-ladder--874-checkpoints-at-98500-and-it-falsified-its-own-pre-registration),
and the pre-registration it falsified is in
[`archive/runs-archive.md`](archive/runs-archive.md#retired-from-runsmd-2026-08-22--the-closed-rungs-of-the-b42-b45-ladder).

**Where this leaves the ladder.** Banded mean self-eval perfect rate is monotone in the rate across all three
rungs, and the step sizes tell you where the plateau is: `1e-5` → `1e-6` bought **+4.9 pp**, `1e-6` → `1e-7`
bought **+2.0 pp**. Decelerating but not flat, so **`1e-8` is the obvious next rung** and the one thing that
would settle whether this is a plateau or a slope. `b45`'s close-out is what decides it on the 500-episode
instrument; everything above is 10-episode self-eval.

**‡ At 85% of its cap, `1e-8` reads as the plateau — and specifically as the frozen end of it.** The rung is
**level with `1e-7`, not ahead of it**: mean best-30 99.32 vs 99.17 once the 10 → 20 episode graph-eval change is
corrected for (see below). What separates it is that **nothing moves**. Per-0.5M-band drift, first band to last:

| rung | drift, worst → best arm | band-to-band spread |
|---|---|---|
| `1e-6` | **−6.3** → −0.0 | 0.91 - 2.89 |
| `1e-7` | −2.7 → **+1.2** | 0.19 - 1.67 |

| `1e-8` | **−0.6 → +0.3** | **0.19 - 0.37** |

`1e-8` stopped the decay and stopped the gain together — `b44a` climbed +1.2 pp and `b45a`, the *same seed*, drifts
−0.4. Every arm holds within ±0.5 pp of its opening band across 2.8M steps, which is the pre-registered **"flat from
early on"** branch, not the "still climbing at the cap" one. So the ladder's answer is `1e-7`, and the next question
is not a fifth rung downward.

**Two metrics stop working at this rung, and both flatter it.** `sef` (99.8-100.0, the ladder's highest) asks whether
an arm ever fell below 80% perfect, which an arm that never moves gets for free; and **a count of rows ≥98%/500 is
trivially maximised by a frozen arm parked near 98%**, so `b45`'s close-out will probably beat `b44`'s 874 without
being better. **Read `b45` on its best row's rate, not its count.**

**‡ `best_perfect30` is not comparable across the 2026-08-19 boundary, and the size of the gap is now measured.**
`training.num_eval_episodes` went 10 → 20, so `b45` is the first arm set whose best-30 is a maximum over a *less
noisy* window — which lowers it systematically. Recomputing every arm on windows holding the **same 300 episodes**
(30 evals × 10 for `b43`/`b44`, 15 × 20 for `b45`) raises every `b45` arm by **+0.8 to +1.0 pp** and moves the older
rungs by nothing. Uncorrected, `b45` reads 98.55 against `b44`'s 99.17 and looks like a regression; corrected, 99.32
against 99.17. **The whole apparent deficit was the instrument.** Banded means are unaffected and are the metric to
cross the boundary with.

Config is b29's, byte-checked against `b29b`'s own spec with only `SNEK_SEED` substituted: `fc 320`,
chase-safe `c=0.10` gate 75, IS off, target-update 1000, discount 0.9975, food-distance 0, fork-branches 4,
and **no free-space shaping**. `b42` is therefore the seed-matched control for `b43`, and because `b42`'s
configured rate equals the rate its checkpoints carry, the [`SNEK_LEARNING_RATE` resume
fix](findings.md#-snek_learning_rate-was-silently-discarded-by-every-resume--adams-rate-rides-in-the-checkpoint)
is a provable no-op there — the two batches differ in the learning rate and nothing else.

**The four arms are the top 4 of 8 across b29 and b40 by their best 500-episode perfect rate.**

| arm | continues | from | 500-ep rate at that checkpoint | steps it will add |
|---|---|---|---|---|
| `b42a` / `b43a` | `b29b-chase10g75seed2` | @1447000 | **99.0%** (495/500) — the project record | 1.553M |
| `b42b` / `b43b` | `b29a-chase10g75seed1` | @1347000 | **98.4%** | 1.653M |
| `b42c` / `b43c` | `b40b-chasefree10g75seed2` | @1513000 | **98.2%** | 1.487M |
| `b42d` / `b43d` | `b29c-chase10g75seed3` | @1396000 | 97.1% — but over **378** episodes, not 500 | 1.604M |

Three of the four are b29 arms; that is where the measured checkpoints are, not a design choice. `b42d`'s row
was **abandoned by the 98% gate**, so its 97.1% is not strictly comparable with the three full-length rows
above it — it is the rank-4 arm on either reading, since rank 5 (`b40a` @1816k, 96.3%/294) sits behind it with
a heavily overlapping CI.


### Batch 45 — `1e-8`, and the first rung with a longer cap

**Trained on the desktop 2026-08-21, now being measured.** The ladder is monotone and decelerating
(+4.9 then +2.0), which is what approaching a plateau looks like without having arrived; nothing yet
distinguishes "plateau" from "slope", and this rung is the measurement that does.

**The cap is 5M rather than 3M, and it is the one deliberate difference from the other three rungs.** The
`best_perfect30` peak arrives later at every rung — **at** the seed step for `b42`, 1527k-2803k for `b43`,
1814k/2190k/2230k/2460k for `b44` (300-1100k past seed). If that timescale keeps stretching with the rate, a 3M
cap would stop these arms *before* their peak and the batch would report truncation as a null — which is exactly
the "it will just freeze" error that was already wrong once, at 1e-7. 5M gives each arm ~3.5-3.65M steps past
seed against `b44`'s ~1.49-1.65M, for roughly 20 h of wall clock.

**This introduces no confound.** A cap cannot change a trajectory, only where it stops, so every cross-rung
comparison is still made over the matched +1487k window and is unaffected; the extra steps are strictly
additional data.

| reading | probability | what it would mean |
|---|---|---|
| `b45` beats `b44` by a further **+0.5 to +1.5 pp**, 3-4 of 4 seeds | ~45% | the ladder is a slope, not a plateau, and the useful rate is lower still |
| **null against `b44`**, inside ±0.5 pp | ~40% | the plateau is real and sits between 1e-7 and 1e-8. **The outcome to plan around** — it makes 1e-7 the operating point and closes the ladder. **← this is what happened**, on the training instrument at 85% of the cap: +0.15 pp, and flat from the first band |
| `b45` is **worse** than `b44` | ~15% | two mechanisms, and **the peak's timing separates them**: still climbing at the 5M cap means merely too slow (raise the cap again); flat from early on means the step has fallen below the scale that changes the greedy action, and the arm is genuinely frozen — the real floor this ladder has been looking for |

**Read the peak's timing, not only its height.** `best_perfect30`'s *step* is the discriminator between "frozen"
and "still improving", and it is what made `b44`'s result legible. Given the last rung's pre-registration was
wrong in exactly the direction of assuming saturation, the null branch above is the one to hold most loosely.

**‡ In the event the discriminator that worked was neither the height nor the peak's step, but the *drift*.**
`b45`'s peak steps scatter uninformatively — 1616k, 1957k, 2990k, 3040k on the equal-episode window — while the
per-band drift separates the rungs cleanly and in one direction (table above). Peak step is a maximum over a noisy
series, so it moves with the number of draws; band drift is a mean-of-means and does not. **Prefer drift next
time**, and keep the peak's step for the qualitative question of whether an arm was still climbing when it stopped.


### How these arms were started — it is not an ordinary resume

Each policy dir was **pre-seeded by hand**: `arch.json`, exactly one `ckpt-*` pair, a copy of the source
arm's `replay_buffer/buffer.npz`, and a `checkpoint` state file naming only that step. Three reasons, all
load-bearing:

1. `initialize_or_restore()` takes whatever the `checkpoint` file names, which in the source dir is its
   **last** step (2M), not its best. Resuming the source arm in place would have continued the wrong weights.
2. It would also **append this continuation's checkpoints and graph history over the very arm the selection
   was made from** — destroying the evidence.
3. One pre-existing checkpoint means **every later `ckpt-*` in the dir belongs to the new run**, so a
   close-out cannot mix two arms' weights at the same step.

**Carrying the replay buffer is not optional, and it was measured.** A fresh dir holds only the 1000 random
transitions `random_play()` writes, and training samples a batch from those on step 1. A 5k-step smoke test
from `b29b` @1447000: **80% → 50% perfect without the buffer, 90-100% with it.** The seeding script is
[`scripts/seed_from_checkpoint.sh`](scripts/seed_from_checkpoint.sh).


## Record status

**The record is `b29b-chase10g75seed2` @1447000 — 99.0% over 500 fresh episodes** (495/500, CI 97.7-99.6), set
2026-08-16 and promoted to [`../hallOfFame/`](../hallOfFame/README.md) with the copy verified to load and play.
It is the highest /500 point estimate on record; the lead over `b24d`'s 98.0%/500 is inside the CIs, so it is a
*point* lead, taken under the folder's 500-episode standard exactly as `b24d` was taken over `b18b`.

| checkpoint | /500 | status |
|---|---|---|
| `b29b` @1447000 | **99.0%** (495/500) | **record**, in `hallOfFame/` |
| `b40b` @1513000 | 98.2% (491/500) | in `hallOfFame/`, promoted 2026-08-18 — **its batch is a null**, it is preserved on its own number |
| `b24b` @2860000 · `b24d` @1342000 | 98.0% (490/500) | both in `hallOfFame/` |

**Nothing from the b42-b45 ladder has been promoted, and the reason is the winner's curse.** `b43b` @1661k read
99.6%/500 and `b44b` @1886k read 500/500 — then the four best checkpoints in the project were re-measured
together on 1000 fresh episodes and **all four fell**, mean −1.35 pp, `b44b` to 98.2% at p=0.0025. So a selected
/500 maximum is worth about **1.4 pp less than it reads**, and the pooled fresh figure for those four is
**98.40%** — level with the standing record, not above it.
[Full result](findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20).

**‡ b29b's 18-checkpoint ≥98%/500 band was read as the config's property and it is not.** `b37`, byte-identical
on seeds 5-8, held **0 of 4**; `b40` held 1 — on comparable ≥98%/**100** tiers. The 21 record-tier checkpoints
are a **seed**.
[Correction](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading).

**A selected row describes the checkpoint the screen liked, never its neighbourhood.** `b18b` @1588k reads
97.6%/700 and `@1578000` — 10k steps earlier — reads **91.6%**. Any *regional* claim needs a position-chosen
sample, and that has never been run for `b24d` @1342k or `b29b` @1447k either. Historical progression and the
per-batch best-checkpoint table: [`archive/runs-archive.md`](archive/runs-archive.md); measurement caveats:
[`findings.md`](findings.md#three-measurement-caveats).

## Outstanding, highest-value, in order

1. **`b41` finished on the desktop and has no write-up anywhere** — training, close-out and HOF-500 all read
   `done` in the ledger, and there is no section in `completedRuns.md`, `findings.md` or `charts.md`. It is the
   **b29 same-seed determinism probe**: b29's exact config re-run on seeds 1-4 to measure the process-noise
   floor that every seed-matched verdict in this folder sits on top of, since `ParallelPyEnvironment` worker
   ordering and TF's threaded FP reductions make a same-seed re-run diverge. Nobody has pinned that number, and
   **if `b29b`'s 99.0%/500 does not reappear on its own seed it is the strongest confirmation yet that the /500
   record was noise rather than a property of the seed or the config.** Rationale as written at launch:
   [`archive/runs-archive.md`](archive/runs-archive.md).
2. **`b45`'s close-out is the ladder's last measurement.** Wave 1 is in its HOF stage, wave 2 is queued. **Read
   it on its best row's rate, not on its count of ≥98%/500 rows** — a frozen arm parked near 98% maximises that
   count without being better, so `b45` will probably beat `b44`'s 874 while being level with it.
3. **`fc 512` under the b24 config is the strongest untested architecture arm.** b24/b25/b26 order the IS-off
   lift by *widest layer* — 320 → +12.2, 200 → +10.3, 100 → +3.5, 50 → 0 — with parameter count not even
   monotone
   ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
   `512` is the cheap way to ask whether that ordering keeps going or has a knee, and `b24a-d` is already the
   seed-matched control.
4. **4 more `320` seeds are still owed** before the width × IS-off interaction is called established. b24's
   +12.2 is at the n=4 sign-test floor (p=0.0625); b25/b26 answered a different question and do not firm up the
   `320` gap itself.
5. **A position-chosen grid around a record checkpoint.** Every record here is a narrow peak measured by the
   screen that selected it. A blind every-10k grid over 1.55-1.62M (`b18b`) or around `b29b` @1447k would say
   whether the config produces a better *region* or one lucky checkpoint — the same test that deflated `b17b`'s
   apparent region to 84%.
6. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which took
   ">95%" literally. ~9 minutes of eval if a fuller picture of that region is wanted.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the table. **Closed items are
pruned from here** once their result is in [`findings.md`](findings.md) and
[`completedRuns.md`](completedRuns.md) — chase-safe shaping, `TARGET_UPDATE_PERIOD`, `FC_LAYERS=128,128` and
the `eval_wave.py` controller all left this table that way.

| change | targets | prior |
|---|---|---|
| **Free space in one piece** — as a graded potential and/or an observation | endgame packing, which is what decides whether the food lands somewhere edible | **new 2026-08-14**, and the largest per-policy separation on record: one-piece share at length 90-94 is **92% / 77% / 5%** for `b24d` / `b18b` / `b20d` ([findings](findings.md#-the-packing-property-the-records-keep-their-free-space-in-one-piece-and-it-separates-them-by-87-points)). `count_groups` already runs every step, so both forms are nearly free. **Correlational, n=3 checkpoints** |
| ~~`SNEK_ALGO=c51`~~ — distributional RL, a categorical value head instead of a scalar Q | post-peak collapse | **closed 2026-08-18** — [plan](../plans/distributional-c51.md) shipped and ran as the pilot, `b31`, `b32`, `b36`, `b38` and `b39`. **C51 is far behind `ddqn` at the identical shape** (b36 pools 74.8-80.2, b38 71.8-78.5, neither reaching ≥98% once, against b24's 86.0-89.0 with a ≥98% checkpoint in every seed); Adam's `epsilon` is a real −26% churn effect with no dose response, and zero-init loses through action separation. Write-ups in [`completedRuns.md`](completedRuns.md#batch-36--c51-on-fc-320-the-better-c51-shape-and-still-far-behind-ddqn) |
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| **`FC_LAYERS=512`** under the b24 config | consolidation — how far the widest-layer ordering goes | **new 2026-08-14, and the top architecture item.** b24/b25/b26 order the lift by widest layer (320 → +12.2, 200 → +10.3, 100 → +3.5) with parameter count non-monotone ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)). `b24a-d` is already the seed-matched control, so the arm costs four trainings and nothing else |
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


## Batch bookkeeping

Each batch keeps its **description** — why it is shaped that way, what each arm isolates, what outcome would
mean what — in this file for as long as any of its arms is running, **measurement included**: a batch whose
arms have stopped training but whose close-out or HOF pass is still going is still live here. When the last
job of a batch stops, move the description and its results to [`completedRuns.md`](completedRuns.md) and
delete it here.

The reason to keep the description live rather than only the status table: the design rationale is what tells
a future session whether a surprising result is informative or just an arm that was never going to answer
anything.

**Status paragraphs for closed batches are retired to
[`archive/runs-archive.md`](archive/runs-archive.md)**, not deleted — they index a family faster than the
per-batch write-ups do, and a number cited here has to stay traceable to the arm that produced it. The last
sweep was 2026-08-22, which took this file from 1075 lines to ~350 by retiring the batch-27/30 counter-bug
notice, the closed rungs of the b42-b45 ladder, every closed-batch status section from b31 to b41, the gate
ladder summary, the b20-b26 index, the max-progression table and the batch 11-19 one-liners.

**Verifying what is running on each host is [above](#what-is-running--2026-08-22)** — and note that neither
check sees the other box, so a count is meaningless without naming it. Full ladder for a desktop that looks
dead: [`CLAUDE.md`](../../CLAUDE.md#there-are-two-compute-hosts--say-which-one-you-mean).
