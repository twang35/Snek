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

## Running: batch 19 — standard PER (`td_error` + IS on)

**Launched 2026-08-08, four arms `b19a`-`b19d` at seeds 1-4**, alongside the two batch-18 close-out
evals (training runs a bit slower until those finish, by design). Give the textbook PER values a fair
shot now the buffer's consistency issues are fixed. **One knob-group changes against batch 18:** the
priority signal goes `td_loss` → `td_error` (so the priority is `|TD error|`) and importance sampling
turns **on** with β annealing 0.4 → 1.0 over 1M steps. Huber stays the network loss
(`element_wise_huber_loss`, unchanged). Everything else is batch 18 byte-for-byte, so **batch 18 is
the seed-matched control**.

No code edit: `td_error`, `IS_WEIGHTS=1`, `IS_BETA=0.4` and final β 1.0 are all `snek2.py` defaults,
so the launch just *drops* batch 18's two PER overrides — confirmed by the startup logs showing no
`PRIORITY_SIGNAL`/`IS_WEIGHTS` override on any arm.

**β anneal:** these four arms ran the **1M** schedule (the default at launch). The default was changed
to **300k** on 2026-08-08 for future runs — it does not affect b19, which was already at ~1M steps and
fully annealed when the change was made.

| item | value |
|---|---|
| change | `td_loss` → `td_error` priority, IS off → **on** (β 0.4 → 1.0 over 1M); everything else = batch 18 |
| control | **batch 18**, seed-matched — identical but for the two PER knobs |
| isolates | whether standard proportional PER (`\|δ\|` priority + full IS correction) beats the effective-α~1.2 `td_loss`/no-IS config the project has run since batch 5 |
| decided at launch | **LR kept at default 1e-5** (IS weights are mean-normalised, so the tuned LR is preserved — may revisit if β=1 correction misbehaves); **seeds 1-4**; **batch-18 base kept** (`TARGET_UPDATE_PERIOD=1000` + forking) |
| horizon | batch 18's arms reached ~1.4M, so the paired read against the control caps at **~1.4M**; arms may run past that, the user calls the stop |

As launched (`EVAL_WORKERS=10`, LR default):

```
SNEK_SEED=1..4  SNEK_TARGET_UPDATE_PERIOD=1000
SNEK_FOOD_DISTANCE_REWARD=0  SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5  SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60
# PER left at snek2.py defaults: td_error priority, IS on, beta 0.4 -> 1.0 over 1M
```

**What each outcome means.** If standard PER matches or beats batch 18, the long-standing
`td_loss`/no-IS default loses its justification and IS-on stops being a confounded question — `b5c`
paired IS with `td_loss` + alpha 0.8 *and* a fast anneal, so it never isolated IS. If it reads
clearly worse, that reproduces `b5c` cleanly for the first time and closes the
[partial-IS-correction candidate](archive/runs-archive.md#later-candidates) with it.

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

**Outstanding, highest-value:** promote `b17b-forkseed2` @1190000 to
[`../hallOfFame/`](../hallOfFame/README.md) — verify the *copy* loads and plays.

## Closed batches (11-19)

One line each; full write-ups and per-seed numbers in [`completedRuns.md`](completedRuns.md),
superseded detail in [`archive/runs-archive.md`](archive/runs-archive.md).

| batch | change | verdict |
|---|---|---|
| **19** | standard PER (`td_error` + IS on, β→1.0) | **running** — see above |
| 18 | `TARGET_UPDATE_PERIOD` 8 → 1000 | primary moved: 102k faster to pf30≥40%, 4/4 seeds; drawdown improved |
| 17 | `FORK_BRANCHES=4` forked endgame collection | null on config (one seed carried it), **project record on `b17b`**; dose ~60% of design |
| 16 | `FOOD_DISTANCE_REWARD=0` | **first non-null** — `sef` +11.35 pp at 1.25M; consolidation not ceiling. Needs replication |
| 15 | `N_STEP_UPDATE=3` | falsified on speed — 128k slower; evals null |
| 14 | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8` | null vs 13 |
| 13 | eps handover 0.0125 + shield 0.5 | null vs 11 on five metrics |
| 12 | eps handover 0.05 | deadlocked, abandoned 4/4 |
| 11 | the 30-value vector itself | +4-5 pp vs 10, not significant |

**The binding constraint is seed variance, not ideas** — n=4 resolves nothing below ~10 pp, and peak
trailing reads 94.7-95.0 flat across batches 11-16. Nothing has raised the ceiling; batch 16 raised
how much of the time an arm sits near it.

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **closed as batch 18** — primary moved, 4/4 seeds; see [`completedRuns.md`](completedRuns.md) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| `FC_LAYERS=128,128` | capacity | low |
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
