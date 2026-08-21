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

## ⚠ Batches 27 and 30 were relaunched — the perfect-game counter was reward-based (2026-08-14)

**Read this before anything below it.** Every perfect-game counter identified a win by comparing the
episode's final reward with `PERFECT_GAME_REWARD`, and the chase-safe term shifts that reward by `−c`. So
`perfect_percent` read **0 for every eval of b27a-d and b30a-d** while the arms were filling boards from
step 9k, and because `training.epsilon_for` takes the trailing perfect rate as its skill signal, **epsilon
stayed pinned at 0.0125** — the refinement ceiling — instead of annealing. Full account, numbers and fix:
[`findings.md`](findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter).

**The fix shipped as `b72a5a84`** — counting moves to `state_helpers.is_perfect_score(score)`, and
`tests/test_perfect_game_counting.py` pins it (10 tests, both mutants caught, suite 23 modules / 596 tests /
0 failed). Deployed to the desktop the same evening; both boxes now run it.

| batch | void arms | live arms | where |
|---|---|---|---|
| **27** | `b27a-d`, killed at 309-326k | **`b27e-h`** (seeds 1-4), launched 21:31 | desktop, priority 10, close-outs and HOF-500 auto-chained |
| **30** | `b30a-d`, killed at 137-139k | **`b30e-h`** (seeds 1-4), launched 21:40 | laptop, one chart window on `--arms b30` |

**Fresh policies, not resumes.** The void arms' weights and buffers carry 320k steps trained under an
exploration schedule that was never going to descend, which is the one thing a seed-matched comparison
against b24/b25 cannot absorb.

**Batch 30's checkpoints were deleted and the wave relaunched a second time** (21:40), because the first
relaunch's chart window opened with **eight panels**: the viewer's arm registry admitted anything inside a
12 h TTL, and the killed a-d arms were 71 minutes old. `savedPolicies/b30[a-h]` (~330 MB) and the tmp
registry are gone; **`runs/b30a-d*` is kept** — those graphs, reports and eval series are the measured
record of the counter bug, committed at `1cd5a03b`, and `charts/` holds its own copies by design. Say so if
you want them removed as well. The registry rule is now liveness-based
([the mechanism](../../CLAUDE.md#rendering-is-off-by-default--use-watchpy-to-see-a-game)); the desktop was
checked and needs no change, since its daemon passes explicit PNG paths and reads no registry.

**Progress update, 2026-08-16 — all four chase-safe shaping batches (b27-b30) are done, and the gate is the
lever: gate 85 is null at any dose, gate 75 produced a record region.** All sixteen fixed-counter arms
trained with a real perfect rate and a descending epsilon — the two things b27a-d could not produce in 320k
steps — so the counter fix is confirmed end to end. Where each batch landed (full numbers and graphs in
[`charts.md`](charts.md); write-ups in [`completedRuns.md`](completedRuns.md)):

- **b27e-h (desktop): done at 2M, closed out — a null.** Pooled mean **85.2** (eq-effort, gate 95) vs the
  b24 control's ~87.9 (a shade *below*), and **0 of 4** seeds produced a ≥98%/500 checkpoint — best `b27h`
  **97.5%** — against the control's **two** 98.0%/500 records. `c=0.10` on `fc 320` did not reproduce the
  record, let alone beat it.
- **b30e-h (laptop): done at 2M and closed out at 15:05 — also a null, and by the same margin as b27.**
  Training was a dead heat (matched ≤2M best-30 **92.9 vs 93.6**, `sef` **56.9 vs 58.6** against the b25-r2
  control), and the close-out agrees: pooled equal-effort **83.3 mean** (84.3 / 84.3 / 83.8 / 81.0) against
  b25-r2's ~86.1, so **~2.8 behind its seed-matched control** — the same direction and size as b27's 85.2 vs
  87.9 on `fc 320`. **Two architectures, two nulls-or-worse for `c=0.10`.** Ten checkpoints reached ≥98% at
  *100* episodes (6 / 3 / 1 / 0), best 99.0% — but that is not the ≥98%/**500** gate. **The HOF-500 pass has
  since run and is empty**: every one of its ten candidates was abandoned under gate 98 before 500 episodes,
  best partial `b30e` @651k **96.1% at 285 episodes**, then `b30g` 95.3% and `b30f` 90.9%. So the
  shaping×architecture 2×2 is complete and `c=0.10` produced **no record-tier checkpoint on either net** —
  the write-up is in [`completedRuns.md`](completedRuns.md) and
  [`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).
- **b28a-d (desktop, `c=0.20`, gate 85): done at 2M, closed out and HOF-500'd — a null, the dose is not the
  issue.** Pooled mean **85.4** (~2.5 under the b24 control's 87.9) and **0 of 4** seeds held ≥98%/500.
  Doubling the dose changed nothing, which — with b27/b30 — rules out both the net and the dose at gate 85.
- **b29a-d (desktop, `c=0.10`, gate 75): done at 2M, closed out and HOF-500'd — the positive result.**
  Pooled **87.8** (a dead heat with b24) but **21 checkpoints held ≥98%/500 across 2 of 4 seeds**, where the
  record-holding control produced only 2 isolated ones. `b29b` @1447k = **99.0%/500 (495/500)**, the head of
  an 18-checkpoint band — **the new project record**, promoted to `hallOfFame/` (see Record status below).
  **The gate, not the dose or the net, is the lever.**
- **Laptop: `b31a-d` was stopped at 538-569k with no close-out**, for the same measurement that started the
  C51 `epsilon` line — the churn is the learning rate, not C51
  ([`findings.md`](findings.md#-the-c51-arms-chaos-is-the-learning-rate-not-c51--and-the-rate-is-high-because-c51-needs-it)).

**Both hosts as of 2026-08-18 21:24 (fetched). Both are training `b42`/`b43` — the same four
checkpoints, one learning rate each, and both chains are armed. Nothing is owed by hand.**

| host | state | measurement chain |
|---|---|---|
| **laptop** | **idle of trainers.** `b43a-d` are finished on all three instruments. Its last work was the **1000-episode re-measurement of the project's eight best checkpoints** (four `b43`/`b44` continuations, four `hallOfFame/` entries) — 8000 episodes, ~16 min, results in `runs/k1000*_checkpoint_evals_k1000.json`. **All eight fell**, mean −1.35 and −1.45 pp; see [the finding](findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20) | **armed.** `scripts/chain_closeout_after_training.sh b43 120` running detached (reparented to pid 1, log `/tmp/b43_chain.log`): polls until the four arms self-terminate at the 3M cap, then close-out at gate 96, then the HOF-500 re-measure on anything ≥98%. **fired on time at 01:07.** The close-out is at 542-709 of 791-1196 checkpoints per arm after 7.3 h — `b43b` is the long pole at ~8 h remaining, so HOF-500 starts this evening. See the close-out cost warning below: this is expected, not stuck |
| **desktop** | **`b45a-d` training**, ~31% of a 5M cap (1.48-1.66M, +133-142k past seed), 4 trainers, 0 evals. `b42`, `b43` and `b44` are all finished on training, close-out and HOF-500; `b44`'s HOF landed 2026-08-20 with all 2235 measurements and **874 rows ≥98%/500**. `b45`'s close-out is queued behind the training | **explicit, not automatic.** A `kill -9` makes the trainings `failed`, and `auto_closeout` fires only on `ok` — so four `<policy>-closeout` specs were queued by hand *before* killing. They use **exactly the id the daemon would synthesize**, which `_scan_pending` keeps in preference to its own projection, so there is no double-run; and `_hof_owed` keys off the *close-out's* success, so HOF-500 still chains by itself |

**The two hosts now run the same chain, which they did not before 2026-08-18.** The desktop daemon has
chained `training → closeout → HOF` since 2026-08-15; the laptop's `chain_closeout_after_training.sh`
stopped after the close-out, so a laptop batch produced *less* than the same batch on the desktop and the
missing half was the one that decides whether a checkpoint is hall-of-fame material. The script now carries
the HOF stage, copied from the daemon's own `HOF_EVAL_ENV`/`HOF_EVAL_ARGS`. **It also pins the close-out
gate at 96** — it used to inherit `eval_checkpoints`' default of 95 while the desktop pinned 96, so the two
hosts were writing close-outs under different gates. That matters because a file's gate lives in its
payload as `min_achievable` and has to be checked before anything is pooled across files, and because the
HOF pass selects `above:98` *from the close-out file* — a gate at or above 98 would abandon the very rows
it needs. Verified against four finished close-outs: the selector reproduces the desktop's own HOF row
counts exactly (b40b 63, b29b 64, b40d 2, b39a 0).

**`b42`'s arms started ~5 minutes before the learning-rate fix was deployed, and it does not matter.** A
running job keeps the code it launched with, so b42 is on the pre-fix `snek2.py` — but its configured rate
*is* the rate its checkpoints carry (1e-5), so `enforce_learning_rate` would compare equal and assign
nothing. The two batches still differ only in the learning rate. Any *future* desktop batch that retunes
the rate on a resume gets the fix, which is live there from `b8d817fd7`.

**Adam's `epsilon` is settled and b32 is closed** — shared-state-set churn **0.119 → 0.088, −26%, 4 of 4
paired, flat to 1M, no dose effect**. The same measurement found that every previously published per-arm
churn figure was inflated ~2×; both are written up under [batch 32](completedRuns.md#batch-32--adams-epsilon-on-c51-it-works-at-26-churn-and-the-dose-does-not).
**The dose is now closed for good** — b36 vs b38 retried it at 4 seeds a side and pooled 76.77 vs 74.73 at a
matched ≤2M horizon, 3 of 4 favouring `1.5e-4`, p=0.625. `1.5e-4` stays the default on lower seed variance.

**A `failed` training does not publish.** `_publish_results` is gated on the same `ok` flag as
`auto_closeout`, so `b42`'s training artifacts were never going to reach the `results` branch. They were
rsynced across by hand (`rsync -a "the-claw-den:Snek/snek2/runs/b42*" snek2/runs/`) and are what the banded
comparison below is computed from. Re-rsync when the close-out finishes to pick up
`_checkpoint_evals.json`.

**Batch numbering: `b37` is the desktop's b29 replication, so the laptop's dose arm is `b38`.** Worth stating
because both were queued within minutes of each other from different hosts, and `b37` was very nearly used
twice. **`b39` is a C51 zero-init batch (laptop, `launch_b39_zeroinit.sh`); the free-space batch below is
`b40`.**

## Batches 42-45 — what happens if you keep training a champion — **the answer is yes, and lower is better: 4 → 187 → 874 rows ≥98%/500 across 1e-5/1e-6/1e-7; `b45` at 1e-8 is training**

**The question nobody here has asked.** Every record in this project is a checkpoint some 2M-step arm
*passed through* on its way to a worse endpoint. No arm has ever been continued **from its own best
checkpoint**. So: does a champion that keeps training improve, hold, or decay?

**One experiment, now a three-rung learning-rate ladder on one set of checkpoints.**

| | desktop `b42` | laptop `b43` | desktop `b44` | desktop `b45` |
|---|---|---|---|---|
| learning rate | **1e-5** (the default, the rate these checkpoints were trained at) | **1e-6** | **1e-7** | **1e-8** |
| everything else | b29's config verbatim | b29's config verbatim | b29's config verbatim | b29's config verbatim |
| cap | 3M, absolute | 3M, absolute | 3M, absolute | **5M**, absolute — see below |
| state | **stopped at +385-421k — it decays.** Closed out and HOF-500'd — [write-up](completedRuns.md#batch-42--the-same-four-checkpoints-at-the-default-lr-1e-5-stopped-early-it-decays) | **finished on all three instruments** — [write-up in `completedRuns.md`](completedRuns.md#batch-43--continuing-the-four-best-checkpoints-at-lr-1e-6-a-record-region-10x-wider-than-anything-before-it-and-the-best-checkpoint-was-the-wrong-one-to-continue) | **finished on all three instruments** — HOF-500 landed 2026-08-20, all 2235 measurements | **training on the desktop, ~31%** of a 5M cap; close-out queued |
| self-eval | pooled eq-effort mean **92.1**; its only ≥98%/500 rows are within 75k steps of its own seed | holds flat, `sef` 96.5-99.5, one seed hit a 100.0 best-30 window | **wins: 4 of 4 seeds over `b43`**, `sef` 98.7-99.9 | holding at ~31%: `sef` **99.3-100.0**, best-30 95.2-**99.2** |
| close-out (/100) | **≥98% on 2.2-14.1%** of its checkpoints | ≥98% on **6.0-38.7%** | **≥98% on 6.9-56.3%** — 3 of 4 seeds ahead of `b43`, and by a lot | queued |
| HOF (/500) | 4 rows ≥98% in total, all within 75k of a seed | **187 rows ≥98%**, best 99.6% (`b43b` @1661k) | **874 rows ≥98%**, 90 at ≥99%, and **two 500/500s** (`b44a` @2798k, `b44b` @1886k) — [both selection artefacts](findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20) | — |

`b44` existed because `b42` and `b43` bracketed the effect on the first try: dropping the rate 10× turned decay
into a hold. It asked where that stops, and **the answer is "not yet at 1e-7"** — see [what actually
happened](#-b44-at-1e-7-beat-1e-6-on-4-of-4-seeds--the-pre-registered-null-was-wrong) below.

**Where this leaves the ladder.** Banded mean self-eval perfect rate is monotone in the rate across all three
rungs, and the step sizes tell you where the plateau is: `1e-5` → `1e-6` bought **+4.9 pp**, `1e-6` → `1e-7`
bought **+2.0 pp**. Decelerating but not flat, so **`1e-8` is the obvious next rung** and the one thing that
would settle whether this is a plateau or a slope. The close-outs now running are what decide it on the
500-episode instrument; everything above is 10-episode self-eval.

Config is b29's, byte-checked against `b29b`'s own spec with only `SNEK_SEED` substituted: `fc 320`,
chase-safe `c=0.10` gate 75, IS off, target-update 1000, discount 0.9975, food-distance 0, fork-branches 4,
and **no free-space shaping**. `b42` is therefore the seed-matched control for `b43`, and because `b42`'s
configured rate equals the rate its checkpoints carry, the LR fix below is a provable no-op there — the two
batches differ in the learning rate and nothing else.

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

### ⚠ Read the result against selection bias, not against the numbers in that table

The four starting rates are **the maximum of a noisy statistic over 8 arms and hundreds of checkpoints**, so
they are biased upward — the winner's curse. This file already established the size of the effect:
[a ≥98%/100 checkpoint has roughly a 1-in-60 chance of holding at
500](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading),
and `b29b`'s 18-wide band was **a seed, not a config**. So:

- **An arm that continues and later measures ~96-97% has not necessarily decayed.** Regression to the mean
  predicts exactly that, with no contribution from the extra training.
- **The clean comparison is `b42` against `b43`, not either against its own starting rate.** Both start from
  byte-identical checkpoints, so the selection bias is common-mode and cancels.
- **The unbiased per-arm baseline is cheap and has not been run**: re-measure each *starting* checkpoint at
  500 fresh episodes. The close-outs may supply it for free — the seeded dir still contains the start
  checkpoint and it has a graph point, so `top20` can select it — but that is not guaranteed, and it is worth
  one 2000-episode job if the headline reads as a decline.

### What each outcome would mean

| reading | meaning |
|---|---|
| `b42` holds ~99% and extends the band | the record region is reachable by *training longer from inside it*, and every previous arm was stopped early. The most valuable outcome |
| **✅ `b42` decays, `b43` holds — this is what happened, 4 of 4 seeds** | the 1e-5 steps are too large to sit still at a champion — the endgame is a narrow basin and the default rate walks out of it. Makes **low-LR fine-tuning the way to bank a record**, which is a new tool |
| both decay together | the decay is not the step size. Points at the objective or the replay distribution, and says a champion checkpoint is a transient the optimizer does not want to stay in |
| both hold, neither improves | ~99% is the ceiling of this config and the remaining 1% is not a learning problem — consistent with the four null PBRS terms |
| `b43` decays and `b42` does not | would be surprising and is the one reading that suggests a bug; check the reset line is in all four logs first |

### ✅ What happened — `b42` decays, `b43` holds, on 4 of 4 seeds

Banded mean self-eval perfect rate over the **+385k window matched across all eight arms** (the largest window
where every arm has data — `b42` was stopped at +385-421k, `b43` is past +541k, and `sef` is a fraction of each
arm's *own* evals, so bands are the only fair comparison):

| band past seed | `b42` (1e-5) | `b43` (1e-6) | diff | seeds favouring 1e-6 |
|---|---|---|---|---|
| 0-100k | 93.6 | 95.4 | +1.8 | 3 of 4 |
| 100-200k | 91.6 | 96.3 | **+4.7** | **4 of 4** |
| 200-300k | 89.1 | 96.2 | **+7.1** | **4 of 4** |
| 300-385k | 89.5 | 95.6 | **+6.1** | **4 of 4** |

`best_perfect30` and `strong_eval_fraction` agree on all four seeds: **93.3-97.7 / 87.0-97.9** at 1e-5 against
**98.3-100.0 / 97.9-99.8** at 1e-6.

Two refinements on the headline. **The decay is a one-time drop to a lower plateau, not an ongoing collapse** —
1e-5's last two bands are flat at ~89, so the arm re-equilibrates about 5 pp down rather than unravelling.
And **1e-6 is not pure preservation: `b43b` reached a `best_perfect30` of 100.0 at 1667k, 320k steps past its
own seed**, a 30-eval window with no imperfect game, against 97.3 for its byte-identical `b42` twin. So on at
least one seed, continuing a champion at a low rate *improved* it. These are 10-episode self-evals; the
close-outs and HOF-500s decide it.

**⚠ `peak_trailing` cannot judge this family of arms, and it briefly fooled this investigation.** It is
trailing average *score*, which maxes at 95/95 — **all eight `b42`/`b43` arms read exactly 95.0 with the peak
timestamped at their own seed step**, because a policy restored from a 98% checkpoint fills the board on eval
one. That saturation was first read as "no arm ever beat its starting checkpoint", which is false and is
retracted. Use `perfect_percent`, `best_perfect30` or `sef`. The general lesson is the `game_over`/`sef` one
again: **a metric at its ceiling looks like a measurement and carries no information.**

### ‡ b44 at 1e-7 beat 1e-6 on 4 of 4 seeds — the pre-registered null was wrong

Done at the 3M cap. Same four seeded checkpoints, `SNEK_LEARNING_RATE=1e-7` the only change from `b42`.
**The estimate written into the specs before launch put ~65% on a null and ~25% on a win; the 25% branch is
what happened, on every seed.** The readings as pre-registered, with the outcome marked:

| reading | probability | what it would mean |
|---|---|---|
| ~~null against `b43`~~ | ~65% | **did not happen.** The reasoning — that 1e-6 is already doing nothing but failing to damage the policy — was simply wrong: at 1e-7 these arms are still *moving*, and their best-30 peaks arrive 300-1100k steps past seed rather than at it |
| **✅ better than `b43`** | ~25% | **this is what happened, 4 of 4 seeds.** Decay at 1e-6 is still nonzero and the optimum is below it — extend the ladder to **1e-8** |
| ~~worse than `b43`~~ | ~10% | did not happen. The "1e-6 is actively repairing and 1e-7 cannot keep pace" mechanism is not what is going on — **both** rates improve, and the slower one improves further |

**Why the null prediction failed, and it is worth keeping.** The reasoning was that Adam's step is
`lr · m/(√v + ε)`, so lowering the rate scales every step near-uniformly without changing the relative dynamics
— hence 1e-7 should look like 1e-6 at a tenth the movement, and if 1e-6 is already only *preserving*, a tenth of
preserving is still preserving. The scaling argument is fine; **the premise that 1e-6 was only preserving was
not.** It was improving, slowly, and 1e-7 improves *further* because it improves *more slowly and for longer*.
The tell was available before launch and was misread: `b43b`'s 100.0 best-30 window arrived 320k steps past its
seed, which is motion, not stillness. **The general lesson: "this knob has saturated" needs evidence that the
quantity has stopped changing, not that it is changing slowly** — and a peak arriving late in the run is
evidence against saturation.

| band past seed | `b43` 1e-6 | `b44` 1e-7 | diff | seeds `1e-7` ahead |
|---|---|---|---|---|
| 0-250k | 95.9 | 96.3 | +0.4 | 3 of 4 |
| 250-500k | 95.4 | 96.5 | +1.1 | **4 of 4** |
| 500-750k | 92.2 | **96.6** | **+4.4** | **4 of 4** |
| 750-1000k | 93.4 | 97.0 | **+3.6** | **4 of 4** |
| 1000-1250k | 94.8 | 96.6 | +1.8 | 3 of 4 |
| 1250-1487k | 94.5 | 95.2 | +0.7 | 2 of 4 |

Per seed over the whole +1487k common window: **+4.6, +1.0, +1.6, +0.8** — mean **+2.0 pp**, against **+4.9 pp**
for the 1e-5 → 1e-6 step. The gap is widest at 500-1000k because that is where `b43` dips to 92.2 and `b44` does
not: **1e-6 still wanders on a half-million-step timescale, 1e-7 largely does not.** `b44`'s best-30 peaks land
at 1814k / 2190k / 2230k / 2460k, i.e. 300-1100k past seed, against *at the seed step* for `b42`.

### ✅ The 100-episode close-outs agree with the self-evals, and the effect is much larger than the self-evals suggested

All three rungs are closed out (2026-08-19). The comparison below is **the share of a batch's measured
checkpoints that reached ≥98% over 100 episodes** — not the best row, which is a maximum over a noisy statistic
and reads high by construction on whichever arm was measured most.

| starting checkpoint | `b42` 1e-5 | `b43` 1e-6 | `b44` 1e-7 |
|---|---|---|---|
| `b29b` @1447k (the record) | 6.5% (17/261) | 12.8% (166/1297) | **56.3% (853/1516)** |
| `b29a` @1347k | 7.2% (27/373) | 38.7% (607/1568) | **53.7% (867/1616)** |
| `b40b` @1513k | 2.2% (6/279) | 10.0% (133/1325) | **30.1% (415/1377)** |
| `b29c` @1396k | **14.1% (48/340)** | 6.0% (83/1378) | 6.9% (100/1450) |

**Monotone in the rate on 3 of 4 seeds, and the steps are large** — on the record seed the ≥98%/100 share goes
6.5 → 12.8 → **56.3**, i.e. `1e-7` puts more than half of its checkpoints in the record tier where the default
rate puts one in fifteen. The ≥99%/100 shares tell the same story (2.7 → 6.0 → 32.8 on that seed).

**The exception is `b29c`, and it is the weakest of the four starts** (its seed row was 97.1% over 378 episodes,
abandoned by the gate rather than a full-length 500). On that seed the ladder is flat from 1e-6 down and `b42`
is nominally ahead. Consistent with the rest of this file's experience: the rate matters most where there is a
narrow good basin to sit still in, and that seed may not have been in one.

**Two reasons this understates rather than overstates the effect.** The close-out selector measures every
checkpoint whose graph eval cleared the mandatory tier, so the denominators are near-censuses of different
sizes: **65-93% of `b42`'s checkpoints and 84-98% of `b43`/`b44`'s**. `b42`'s pool is therefore the *more*
selected one — its weak checkpoints are missing from the denominator — and it still comes last. And `b44`'s
pool is the least selected of the three (98% of all its checkpoints on two arms), so it carries the most dead
weight and still wins.

**A caveat that applies to the whole table**: these close-outs ran at a gate of 96, so a row below 96% was
abandoned early and is not full length. Every share above counts full-length rows only, which is exactly the
set the gate guarantees, so the comparison is sound — but the *pooled* rate over all rows is not comparable
across arms with different abandonment counts and is deliberately not quoted here.

### ✅ `b43`'s HOF-500 is done, `b44`'s is half done — and its 500/500 did not survive re-measurement

Both re-measures are `above:98` at 500 episodes, flat, and they are the instrument that decides the ladder —
everything above is 100 episodes. `b43`'s finished on the laptop at 09:55 on 2026-08-20 (767 measurements,
12.3 h wall clock, 80% lane utilisation); `b44`'s is at 1176 of 2235 on the desktop, read 10:05.

| arm | ≥98%/500 | ≥99%/500 | best /500 | pass |
|---|---|---|---|---|
| `b43a` (1e-6, from the record) | 16 | 2 | 99.4% @1618000 (497/500) | **done** |
| `b43b` | **170** | **17** | **99.6% @1661000** (498/500), also @1708000 | **done** |
| `b43c` | 1 | 0 | 98.0% @1760000 | **done** |
| `b43d` | 0 | 0 | — (97.7% @1426k over 488, gate-abandoned) | **done** |
| `b44a` (1e-7, from the record) | 105 | — | 99.6% @2207000 | ~53% |
| `b44b` | **155** | — | 100.0% @1886000 (500/500) — **re-measures 98.2%/1000** | ~53% |
| `b44c` | 33 | — | 98.8% @2301000 | ~53% |
| `b44d` | 0 | — | — | ~53% |

**Totals: `b42` 4 rows ≥98%/500 in a completed pass, `b43` 187 in a completed pass, `b44` 293 with roughly half
its pass to go.** The ladder's ordering holds on the deepest instrument available, and the gap is not marginal —
it is 4 → 187 → 293+ from byte-identical starting weights, with learning rate the only difference.

**`b44b` @1886000 scored 500/500, and then failed to replicate.** Re-measured the same day on **1000 fresh
episodes: 982/1000 = 98.2%** (97.2-98.9), a −1.8 pp drop at p=0.0025. The four best checkpoints in the project
were re-measured together and **all four fell**, mean −1.35 pp — so a selected /500 maximum should be treated as
roughly **1.4 pp optimistic**, and the pooled fresh figure for the four is **98.40%**. Nothing is in
`hallOfFame/`, and the re-measurement is why: [full result in `findings.md`](findings.md#-the-winners-curse-measured-four-selected-champions-all-fell-and-the-500500-did-not-reproduce-2026-08-20).

**Two things this pass corrects in the note it replaces.** First, the candidate record was attributed to the
wrong arm: `b43a` was leading at ~25% of the pass with 99.4%, and `b43b` — a *worse* starting checkpoint, 98.4%
against 99.0% — finished with 99.6% and **170 of `b43`'s 187** ≥98% rows against `b43a`'s 16. **So a checkpoint's
own /500 rate does not predict how well it continues**, and continuing the best checkpoint was not the best move.
Second, the note's own advice — *watch the count, not the maximum* — was right, and it is what makes the ladder
readable: the maxima (99.4 / 99.6 / 100.0) are three noisy draws separated by 2-3 episodes, while the counts are
separated by factors of 40 and up.

**The counts are still one seed's story.** `b43b` alone carries 91% of `b43`'s total and `b44b` 53% of `b44`'s so
far, while the `b29c` seed (`b43d`/`b44d`) produced zero on every rung. The rate multiplies whatever the seed
had; it does not create it. That is consistent with [the retired b29 record
region](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading)
and it means **`b45` at `1e-8` should be read on the same four seeds separately**, never pooled into one number.

**`b43` is now closed out of this file.** Its canonical write-up — the per-seed `b42` comparison, pooled
equal-effort figures, the cost breakdown and the two readings that would be wrong — is in
[`completedRuns.md`](completedRuns.md#batch-43--continuing-the-four-best-checkpoints-at-lr-1e-6-a-record-region-10x-wider-than-anything-before-it-and-the-best-checkpoint-was-the-wrong-one-to-continue), and the selection result it produced is in
[`findings.md`](findings.md#-continuing-a-champion-works-and-lower-is-better--but-the-best-checkpoint-was-the-wrong-one-to-continue-b42b43b44-2026-08-20).
The ladder's forward plan stays here until `b45` finishes.

### Batch 45 — `1e-8`, and the first rung with a longer cap

**Queued on the desktop 2026-08-19**, behind `b44`'s close-out and HOF. The ladder is monotone and decelerating
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
| **null against `b44`**, inside ±0.5 pp | ~40% | the plateau is real and sits between 1e-7 and 1e-8. **The outcome to plan around** — it makes 1e-7 the operating point and closes the ladder |
| `b45` is **worse** than `b44` | ~15% | two mechanisms, and **the peak's timing separates them**: still climbing at the 5M cap means merely too slow (raise the cap again); flat from early on means the step has fallen below the scale that changes the greedy action, and the arm is genuinely frozen — the real floor this ladder has been looking for |

**Read the peak's timing, not only its height.** `best_perfect30`'s *step* is the discriminator between "frozen"
and "still improving", and it is what made `b44`'s result legible. Given the last rung's pre-registration was
wrong in exactly the direction of assuming saturation, the null branch above is the one to hold most loosely.

### ⚠ A continuation batch's close-out costs 10-20x a normal one's — budget for it

`top20` measured **791, 1196, 803 and 826** checkpoints on `b43`'s four arms, not 20, and `b43`'s close-out has
been running since 01:07. This is documented behaviour rather than a bug: **"N is a target, not a quota"** —
`select_top_checkpoints` measures every checkpoint whose 10-episode graph eval reached **≥90%**, past N. A normal
arm climbs from 0 so few checkpoints qualify; **an arm continued from a 98% checkpoint spends its whole run above
the mandatory threshold, so nearly every checkpoint qualifies.** `b42` is the contrast from the other side — it
decayed, so only 261-373 of its checkpoints qualified and its close-out finished overnight.

**So do not read a continuation close-out still running after 7 hours as hung.** Measured now that all three are
finished: **`b43` 15 h on the laptop, `b44` 11.2-14.9 h on the desktop, `b42` 2.5 h** — the last one because it
decayed, so its pool was a fifth the size.

**The HOF-500 passes are the same shape one instrument deeper**, and they are the larger bill: `above:98` selects
**166 / 607 / 133 / 83** checkpoints on `b43` and **853 / 867 / 415 / 100** on `b44`, each at 500 episodes. `b43`'s
pass prices at ~10 h and `b44`'s at ~18 h; `b42`'s was 0.2-1.0 h per arm, on 4 qualifying rows. **So a rung of
this ladder costs roughly 4 h of training and 25-30 h of measurement**, and the measurement is on the critical
path for the next rung — see the note on `b45` below.

If a future continuation batch needs a cheaper close-out, the lever is the *selector*, not the worker count —
`above:<threshold>` reads a prior close-out's 100-episode numbers instead of the graph, which is what the HOF
pass already uses.

### ⚠ `b45` is blocked behind `b44`'s HOF-500 (~13 h left as of 2026-08-20 10:05), and that is the wave barrier working as designed

As of 21:50 the desktop is running **one** job — `b44-hof` — and holding `b45`'s four trainings plus their
close-out queued behind it, because nothing new starts while anything is running. `b44-hof` is ~6% done at
128 of 2235 measurements, so on its own pace that is **~18 h of a 4-trainer box running one eval**.

**Nothing is broken and nothing needs doing tonight**, but it is a choice worth making deliberately rather than
by default, and it is the user's call:

| option | cost |
|---|---|
| **let it run** (default) | `b45` starts ~16:00 tomorrow; the ladder's decisive instrument for `b44` lands first |
| **kill `b44-hof` and let `b45` train** | frees the box now; `b44`'s /500 tier is lost until it is requeued, and its 100-episode result already stands |
| **requeue `b44-hof` with a stricter selector** | `above:99` is 1238 of the 2235 (~10 h — the 98-99 band is only the cheaper half); `above:99.5`, i.e. the 100%/100 rows alone, is **374 (~3 h)** |

The third is the one to pick if the box is wanted back quickly, but only at `above:99.5`: a hall-of-fame
promotion needs ≥98%/500, while the *record* question lives entirely in the top band, and `b44` has **160 and
160** checkpoints at 100%/100 on its two best arms — far more than a record needs. `above:99` barely saves
anything, because the 98-99 band is where the rows are cheapest to abandon.

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

### ⚠ This batch only measures anything because of a bug found while setting it up

`SNEK_LEARNING_RATE` was a **no-op on every resume**. Adam's `learning_rate` is a checkpointed `tf.Variable`,
so `initialize_or_restore()` silently restored the saved 1e-5 over the configured 1e-6 — `b43` would have run
four arms identical to `b42` and reported otherwise. Fixed by `training.enforce_learning_rate`; each arm now
prints its reset line at startup, **and that line is the batch's tripwire** — an arm missing it is training at
1e-5. Nothing already measured is invalidated (every prior resume re-used its original rate). Mechanism,
measurement and the two general lessons:
[`findings.md`](findings.md#-snek_learning_rate-was-silently-discarded-by-every-resume--adams-rate-rides-in-the-checkpoint).

## Batch 40 — the free-space term stacked on the record — **closed 2026-08-18: null, and it retires b29's record region**

**Result.** Pooled equal-effort **85.68 / 88.28 / 89.11 / 89.52** (mean **88.15**, a dead heat with b29's 87.83
and b35's 88.20). **Two arms produced a flawless 100.0%/100 checkpoint** (`b40a` @1562k, `b40b` @1424k) and all
four reached the ≥98%/100 tier — 16 / 63 / 9 / 2 = **90 candidates**, close to b29's own 59/64/9/1. **One held
≥98% over 500 episodes**: `b40b` @1513k at **98.2%/500**, third-best /500 on record.

**Verdict: the free-space term is a null**, and read with `b37` (b29's config on fresh seeds, **0 of 4** held) it
also retires the claim that gate 75 produces a *record region* — three batches with indistinguishable /100 tiers
produced 21, 1 and 0 held checkpoints, so that count is seed noise. Full account in
[`findings.md`](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading);
rationale, per-arm rows and charts in [`completedRuns.md`](completedRuns.md) and
[`charts.md`](charts.md#batch-40--chase-safe-plus-a-global-free-space-term--done-on-the-desktop-null-and-it-makes-b29s-record-region-look-like-seed-luck).

**`b40b` @1513k is a HOF-promotion candidate** — 98.2%/500 sits behind `b29b` (99.0) and `b29a` (98.4) and ahead
of `b24b`/`b24d` (98.0). Promotion is still the manual, verified process; it has not been done.

## Batch 41 — b29 re-run on the **same seeds** — **queued on the desktop** (2026-08-18)

**Sharpens the b37 finding by removing the seed as a variable.** b37 showed b29's /500 "region" does not
replicate on *fresh* seeds; b41 re-runs b29's exact config on the **same** seeds 1-4 to ask whether it
replicates even then. The training is not bit-reproducible — `ParallelPyEnvironment` worker ordering and TF's
threaded FP reductions diverge and compound in the RL loop — so a same-seed re-run is *expected* to diverge,
and the gap `b41x` vs `b29x` measures the **process-noise floor** every seed-matched comparison in this folder
sits on top of, a number nobody has pinned.

| | |
|---|---|
| arms | `b41a-d-b29repro-seed{1..4}`, seeds 1-4, **2M**, priority 30, desktop (same host as b29) |
| config | `b29` verbatim, no free-space term (behaviourally identical under current code, which defaults it off) |
| reads | per-seed vs `b29a-d`: pooled / best-30 / `sef`, the >=98%/500 count, and how far each curve tracks its twin before separating |

**What each outcome means.** Curves that track a long time then separate, pooled within a point or two -> the
floor is small and n=4 seed-matched verdicts here are trustworthy. Wide divergence from step ~0 -> the floor
is large, and *every* n=4 verdict in this folder (the b40 null included) carries that much irreducible noise.
Either way, if `b29b`'s 99.0%/500 does not reappear on its own seed, that is the strongest confirmation yet
that the /500 record was noise rather than a property of the seed **or** the config.

**Caveat:** current desktop code (`>=6bdbe7c3`) is not byte-identical to what b29 first ran — the free-space
addition is off/no-op, but intervening commits make this "current code, same seed", not a binary replay. The
FP/threading floor above dominates any such drift.

## ‡ The desktop marked 14 publishes `done` that never reached the `results` branch (2026-08-18)

**Its DNS resolution for `github.com` is flapping**, and `publish_results` has **no retry**: the artifacts are
written, the ledger records `done`, the push fails once, and nothing tries again. 122 `publish_status` failures
and **14 `publish_results` failures** since 2026-08-17, the last at 07:27 — all four `b40` HOF-500 files plus
`b40b`'s whole close-out were sitting on the box unpublished while `status.json` said `done`.

**Nothing was lost, and the recovery is cheap.** A failed push leaves the commit local, so **the next successful
results push carries the backlog with it** — which is why b35's and three of b37's close-outs eventually appeared
and b40's did not: no results job has completed since 07:27, and the queue is now empty. Retrieved by hand with
`rsync the-claw-den:Snek/snek2/runs/<file> runs/`.

**Two things to internalise.** A `done` in the ledger means "the job finished", **not** "the results are
published" — check `git ls-tree --name-only origin/results:results | grep <policy>` before concluding a HOF pass
found nothing, because *no file* and *an empty file* are the same absence over the git bus, and this batch looked
like four empty HOF passes. And **the fix belongs in the daemon**: either retry `publish_results` on the next
poll, or reconcile at idle by pushing whenever the local `results` worktree is ahead of `origin/results`. That is
a code change and has not been made.

## C51 pilot — closed at 600k, and it handed off to batch `b31` by itself (2026-08-15)

Distributional RL, phase 3 of
[`../plans/distributional-c51.md`](../plans/distributional-c51.md). The implementation is committed
(`245cf914`).

| | |
|---|---|
| wave A, 15:06 | `c51pilot-lr1e5seed{1,2}`, `c51pilot-lr5e5seed{1,2}` |
| wave B, 16:41 | `c51pilotB-lr1e4seed{1,2}`, `c51pilotB-lr25e4seed{1,2}` |
| config | b25's verbatim (`fc 200,100,100`, `IS_WEIGHTS=0`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance shaping) plus `ALGO=c51`, 51 atoms over `[-5, 120]`. Seeds matched across all four rates |
| cap | **600k steps** — a screen, not a result |
| launchers | `launch_c51_pilot.sh` (wave A, which waited out b30's close-out), then the generic `launch_c51_wave.sh` (wave B) |

**What it is asking**, in order: does a categorical agent learn this task at all; how many steps to its
first perfect game against b25's ~9k; is the loss scale sane at 1e-5 (a cross-entropy starts at
`ln 51 ≈ 3.93`, where the Huber TD loss starts near 0, so the same learning rate is not obviously the same
step size). **The gate to phase 4 is one learning rate.**

**Eight trainers on a 14-core laptop is deliberate**, and measured before launching: ~2.3 GB per arm
(0.4 GB parent + 1.9 GB across 11 forked self-eval workers) against 36 GB of RAM, and a swap-in rate of
244 pages per 20 s, so the cost is throughput — roughly half the steps/s per arm — and not paging. It is
also the one place this project's "never more than 4 trainers" rule is knowingly suspended; it was the
user's call, for this screen only.

### The handoff to `b31` ran unattended, and worked

**`launch_c51_batch.sh b31` fired at 20:09**, ~3h20m after it was armed: all eight pilot arms reached the
600k cap, it picked `5e-5`, launched `b31a-d` at 2M, regenerated the tables below and pushed the result as
`c45e8a4f` — with nobody watching any of it. Written this way because cron jobs
in this tool are session-only and fire only while the REPL is idle, so nothing scheduled can be relied on
once the session closes — a detached `nohup` can.

| step | what it does | what it refuses to do |
|---|---|---|
| wait | polls until no `snek2.py c51pilot` trainer remains (one substring covers both waves) | it excludes `chart_viewer` from that `pgrep`, or it would wait forever on the window rather than the arms |
| slots | waits for the laptop's 4-trainer limit to be free | if something else is still training after 6 h it **exits without launching** rather than breaking the limit |
| pick | [`pick_c51_lr.py`](pick_c51_lr.py)'s pre-registered rule — mean `best_perfect30` at a **common horizon**, then `sef`, then `peak_trailing` | if fewer than two rates have usable data it refuses, and the launcher falls back to `5e-5` so a batch still starts |
| launch | `b31a-d`, 4 seeds, 2M, the chosen rate, otherwise identical to the pilots — so `b25a-d` is the seed-matched control | staggers the four by 5 s rather than leaning on the chart viewer's claim lock with nobody watching |
| docs | regenerates the marked region in this file and `charts.md`, then commits and pushes | a push failure is logged, not retried — the commit is local and recoverable |

**Three guards in the picker exist because a dry run got the answer wrong.** At an early horizon every arm
reads `best_perfect30` 0.0 *and* `sef` 0.0, and a two-level rule then picked whichever rate came out of a
dict first — it chose the slowest rate over the fastest, so `peak_trailing` is now the third key. An arm
that dies early no longer sets everyone's horizon (it would have judged seven healthy arms at 13k of 600k).
And an arm with no eval series is excluded and named rather than counted as a zero, so a failed launch
cannot vote against its own rate.

<!-- C51-PILOT-STATUS:BEGIN -->
*Generated by `pick_c51_lr.py` at 2026-08-15 20:09, when the last pilot arm stopped — the numbers below are read straight off the eval series, and the prose around this block is hand-written.*

**Compared at a common horizon of 600k steps**, the lowest final step any arm reached, because both metrics accumulate over an arm's own evals and a longer arm would otherwise win on horizon alone.

| lr | seeds | mean best-30 | mean `sef` | mean peak trail |
|---|---|---|---|---|
| 5e-05 **← chosen** | 2 | 69.5 | 12.6 | 92.42 |
| 1e-05 | 2 | 56.5 | 3.6 | 89.89 |
| 0.0001 | 2 | 39.0 | 5.3 | 88.19 |
| 0.00025 | 2 | 4.0 | 0.0 | 68.79 |

| arm | lr | seed | step | best-30 | `sef` | peak trail | first perfect |
|---|---|---|---|---|---|---|---|
| `c51pilot-lr1e5seed1` | 1e-05 | 1 | 600k | 85.3 | 7.3 | 93.56 | 141k |
| `c51pilot-lr5e5seed2` | 5e-05 | 2 | 600k | 71.7 | 13.0 | 93.30 | 20k |
| `c51pilot-lr5e5seed1` | 5e-05 | 1 | 600k | 67.3 | 12.1 | 91.54 | 15k |
| `c51pilotB-lr1e4seed2` | 0.0001 | 2 | 600k | 66.3 | 10.6 | 90.80 | 46k |
| `c51pilot-lr1e5seed2` | 1e-05 | 2 | 600k | 27.7 | 0.0 | 86.22 | 92k |
| `c51pilotB-lr1e4seed1` | 0.0001 | 1 | 600k | 11.7 | 0.0 | 85.58 | 8k |
| `c51pilotB-lr25e4seed1` | 0.00025 | 1 | 600k | 5.7 | 0.0 | 70.82 | 49k |
| `c51pilotB-lr25e4seed2` | 0.00025 | 2 | 600k | 2.3 | 0.0 | 66.76 | 59k |

**Chosen: `5e-05`** — best_perfect30 69.5 against 56.5 for the next rate (1e-05).

**Batch `b31` launched at 2026-08-15 20:09** on this rate, 4 seeds, 2M cap, `fc 200,100,100`, otherwise b25's config — so `b25a-d` is the seed-matched control.
<!-- C51-PILOT-STATUS:END -->

**`SNEK_CHART_VIEWER=0` on all four, one window opened by hand.** `chart_viewer.batch_prefix` groups only
`b<n><letters>-` names, so four `c51pilot-*` arms would open four windows; the launcher uses the
`--glob`/`--watch` form an eval wave already uses. The pilot deliberately does **not** claim `b31` — `fc 512`
and the four owed `320` seeds are ahead of C51 in the backlog below.

## Batch 39 — **C51 initialised at expected Q = 0** — **closed 2026-08-18 at the 3M cap: it loses on every metric**

**Result.** Matched at ≤1.87M and seed-paired against `b36`, **−9.4 pp** best-30 and **−7.1 pp** `sef`, 4 of 4
seeds down; pooled **70.18 vs 76.76**, also 4 of 4. **All 650 close-out rows were abandoned under the 95% gate**
— the batch produced no measurable checkpoint at all, where b36 produced 4 and b38 5. Pre-registered **H2
confirmed, H1 falsified**.

**But the predicted mechanism was wrong**, and that is the transferable part: zero-init converged its value
*level* **faster** (half-life 163-202k vs b36a's 304k) from a *larger* initial error, so calibration is not the
channel. The channel is **action separation** — b36a reaches a 12.18 action gap by **8k steps**, b39 sits at
**1.72** and needs ~600k to reach 8.90. **Judge a categorical init by the spread it leaves available, not by how
close its mean is to the truth.** `SNEK_C51_ZERO_INIT` stays off, now on measured grounds. Full account in
[`findings.md`](findings.md#-zero-init-loses-and-the-channel-is-action-separation-not-calibration--b39-closed-at-3m);
pre-registration, per-arm rows and charts in [`completedRuns.md`](completedRuns.md) and
[`charts.md`](archive/charts-archive.md#batch-39--c51-initialised-at-expected-q--0-instead-of-the-grid-midpoint--closed-at-the-3m-cap-it-loses-on-every-metric-through-the-heads-capacity-rather-than-its-calibration).

## Batch 38 — b36's config at the **other** Adam epsilon (`3.125e-4`) — **closed 2026-08-17: dead heat, the dose question is settled**

**All four hit the 3M cap and self-terminated**, then closed out at gate 95. At a **matched ≤2M horizon**
b38 pools **74.73 against b36's 76.77** — 3 of 4 seeds favour `1.5e-4`, mean **−2.04 pp**, sign test
**p=0.625**. Best-30 80.0-88.3 vs 84.0-86.7 and best checkpoint 93.4-96.0 vs 91.6-97.0 agree. **No arm
≥98%, so no HOF-500.** So `1.5e-4` stays the default as the lower-variance reference and **the dose is
closed for good** — b32 could not separate the two at n=2, and n=4 says there is nothing to separate.

**Two by-products worth more than the dose answer itself.**

1. **C51 does not benefit from running past ~2M.** Pooling all rows against ≤2M only, **3 of 4 arms got
   worse** past 2M; `b38a` is the exception and holds the batch's best checkpoint at **2355k**. So future
   C51 batches can stop at ~2M — the horizon question b36's launcher raised, answered.
2. **`pooled_equal_effort` is exactly recomputable at any horizon** from each row's stored
   `episode_perfect` flags truncated to the screen depth. Verified by reproducing all 8 published figures
   to the decimal. That removes the horizon caveat that made the first b38-vs-b36 reading unquotable, and
   it is the method to use whenever two arms stopped at different steps.

Per-arm table in
[`archive/charts-archive.md`](archive/charts-archive.md#batch-38--adam-ε-3125e-4-on-b36s-fc-320--closed-the-dose-is-a-dead-heat-at-n4-as-pre-registered)
— batch 38's chart section was retired there on 2026-08-18 to make room for 42/43.

**`launch_b38_eps3125.sh`, identical to b36 with `SNEK_ADAM_EPSILON=3.125e-4` the only change**, seeds 1-4,
3M cap. `b36a-d` is therefore an exact seed-matched control and this is a clean one-variable dose comparison.
**Launched automatically** by `chain_after_evals.sh`, which polls for `eval_checkpoints.py` to drain and then
runs the launcher — log at `/tmp/chain-b38.log`.

**This is the dose question b32 could not answer, retried at 4 seeds a side instead of 2.** b32's shared-set
churn put `1.5e-4` at **0.0865** and `3.125e-4` at **0.0895** — nothing, on n=2, exactly as pre-registered.
b36 + b38 is **the first configuration in this project with 4 seeds per side on one architecture**, so it is
the first that can say anything about the dose at all.

**Read churn first, and only with `--states-from`** — the same reference every C51 reading now uses, so the
numbers stay comparable across batches:

```
PYTHONPATH=. python hyperparamTuning/perDiagnostics/c51_stability.py \
  --policy b36a-c51fc320seed1 --policy b38a-c51fc320eps3125seed1 ... \
  --states 1500 --stride 5000 --points 10 --end 2000000 \
  --states-from hallOfFame/b29b-chase10g75seed2-ckpt1447000
```

| outcome | reading |
|---|---|
| lower churn at `3.125e-4`, best-30 held | the higher dose becomes the C51 default, and the response is still climbing — worth one more rung |
| lower churn, **worse** best-30 | `epsilon` acting as a smaller learning rate in disguise, the known failure mode. This is why best-30 is read *alongside*, not after |
| dead heat at n=4 a side | the dose question closes for good; `1.5e-4` stays the default as the lower-variance reference config |

**b36 stopped at 2M, so match there** rather than at b38's 3M cap.

## Batch 36 (C51 on `fc 320`) and Batch 38 (its Adam-ε dose) — **both closed 2026-08-17**

**Full narratives moved to [`completedRuns.md`](completedRuns.md#batch-36--c51-on-fc-320-the-better-c51-shape-and-still-far-behind-ddqn), per the runs.md/completedRuns.md split.** The verdicts:

- **`fc 320` is the better C51 shape** — best-30 84.0-86.7 against `b32`'s 77.0/63.0, seed spread 14.0 → 2.7 pp.
- **C51 is still far behind `ddqn` at the identical shape.** `b24a-d` pools 85.97-89.03 with a ≥98% checkpoint
  in every seed; b36 pools 74.77-80.19 and b38 71.79-78.51, **neither batch reaching ≥98% once**.
- **The Adam-ε dose is closed for good** at 4 seeds a side: matched ≤2M, 76.77 vs 74.73, p=0.625.
- **C51 gains nothing past ~2M**, so future C51 batches can stop there.
- **Init optimism is excluded** as the remaining suspect, which is what `b39` now tests directly.

## Batch 32 — Adam's `epsilon` on C51 — **closed 2026-08-16**

**Full narrative moved to [`completedRuns.md`](completedRuns.md#batch-32--adams-epsilon-on-c51-it-works-at-26-churn-and-the-dose-does-not).** Shared-state-set churn **0.119 → 0.088 (−26%)**, 4 of 4 paired, flat 600k→1M, dose a dead heat — and the dose
stayed a dead heat when b36+b38 retried it at 4 seeds a side. Its chart section is in
[`archive/charts-archive.md`](archive/charts-archive.md); **b32a-d were never closed out**, so they have no rows
in the canonical table.

## Batch 31 — C51 at `lr 5e-5`, 2M — **stopped at 538-569k, no close-out** (2026-08-15)

Launched 20:09 by the pilot's handoff, **killed 23:10 at the user's call** after
[`c51_stability.py`](perDiagnostics/c51_stability.py) showed the chaos was the learning rate rather than
C51, which made a 2M run at a rate chosen under the old reading not worth the slots. **No close-out was
run** — deliberately, not pending.

Reached 538-569k in 2h44m, all four healthy (no zero stretch), best-30 **21.0 / 53.3 / 66.7 / 71.7** — a
**50.7 pp spread at one config**, which is the n=4 noise problem restated rather than a result. Graphs in
[`charts.md`](charts.md); the arms are in [`completedRuns.md`](completedRuns.md) as void.

## Batch 34 — chase-safe `c=0.10`, **gate 70** — *done on the desktop: null*

**Closed 2026-08-16 (training + close-out + HOF-500), and the pre-registered "gate 70 < gate 75" outcome
landed: gate 75 is a narrow sweet spot.** Identical to b29 with `SNEK_CHASE_SAFE_GATE=70` the only change.
Pooled equal-effort **86.4** (~1.5 under the b24 control and just under b29's 87.8) and **0 of 4 seeds held
any ≥98%/500 checkpoint**, against b29's 21 across two seeds — so a single 5-length step off 75 already
collapses the record region. All four healthy throughout (peak 95.00, no zero stretch). This makes the gate a
**band around 75**, not a threshold: 85 null, 75 records, 70 null again. Full numbers and per-arm table in
[`completedRuns.md`](completedRuns.md#batch-34--chase-safe-c010-gate-70-null--gate-75-is-a-narrow-sweet-spot-not-a-threshold);
finding: [`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## Batch 37 — **b29 replication on fresh seeds 5-8** — **closed 2026-08-18: the /100 band replicates, the /500 record does not**

**Result.** Pooled equal-effort **80.72 / 82.19 / 87.88 / 90.50** (mean **85.32**; `b37b`'s 90.50 is the highest
single arm of the chase-safe family). The ≥98%/100 tier reproduces b29's **2-of-4** shape — 43 and 16 candidates
in two seeds, none in the other two — and **0 of 4 seeds held ≥98% over 500 episodes**, where b29 held 21; the
two best were abandoned under gate 98 at ~360 episodes (97.0%, 96.9%).

**Verdict: the outcome the pre-registration called "b29's region was lucky seeds".** Read with `b40`, three
batches with indistinguishable /100 tiers produced held counts of 21, 1 and 0, so **the ≥98%/500 count is seed
noise** and this family must be judged on pooled. The correction is in
[`findings.md`](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading);
rationale, per-arm rows and charts in [`completedRuns.md`](completedRuns.md) and
[`charts.md`](charts.md#batch-37--b29-replicated-on-fresh-seeds-5-8--done-on-the-desktop-the-100-band-replicates-the-500-record-does-not).

## Batch 35 — chase-safe `c=0.10`, **gate 40** — *done on the desktop: null*

**Closed 2026-08-17, and the pre-registered "gate 40 < gate 70/75" outcome landed — mid-game shaping is a
null on records.** The ladder's deep rung: `b29`'s config with only `SNEK_CHASE_SAFE_GATE=40` (per-flip dose
held at the 0.10 clamp; total episode dose ~2.5× gate 85). **0 of the 3 measured seeds held any ≥98%/500
checkpoint** (`b35c`'s HOF-500 was still running at check time), best partials abandoned at 96-97%. The twist:
gate 40 posts the **highest pooled equal-effort of any shaped batch (88.2**, above b29's 87.8 and the b24
control's 87.9) — so it grades the mid-game into a healthier *average* board without ever reaching the
record-tier endgame. **Consolidation and the record tier are decoupled**, and across four gates (85, 75, 70, 40)
**only 75 records** — the sweet spot is a narrow, isolated band. Full numbers and per-arm table:
[`completedRuns.md`](completedRuns.md#batch-35--chase-safe-c010-gate-40-null--the-sweet-spot-at-75-is-isolated-not-a-plateau);
finding: [`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## The chase-safe gate ladder is complete (b27-30, 34, 35, 37, 40) — 75 leads on the /100 tier, and its /500 "region" was seed noise

**Six batches walked the length gate from 85 down to 40, and only 75 records.** All are closed, so per the
bookkeeping rule their descriptions moved to [`completedRuns.md`](completedRuns.md). The shaping adds
`c·(γΦ(s′) − Φ(s))` with Φ = 1 iff the head and tail share a free region holding the food and the snake is
≥ gate long — potential-based, optimal policy unchanged ([plan](../plans/chase-safe-reward-shaping.md);
Phase 0 Φ calibration in
[`findings.md`](findings.md#-measured-the-chase-safe-potential-is-nearly-static-for-a-record-policy-and-busy-for-a-bad-one)).
All arms are **b24's config plus the one knob** (`fc 320` for b27/b28/b29/b34/b35/b37/b40, `fc 200,100,100` for
b30), so `b24a-d`/`b25a-d` are the seed-matched controls at zero extra compute. Cap 2M.

**‡‡‡ Closed 2026-08-18, and the headline is qualified.** `b37` re-ran gate 75 on fresh seeds 5-8 and held
**0 of 4** at ≥98%/500; `b40` added a free-space term and held **1**; b29 held **21** — on
indistinguishable ≥98%/**100** tiers and tied pooled. So **the /500 count is seed noise**, gate 75's lead lives
on the /100 tier and on pooled, and the ladder's nulls at 85/70/40 stand (they are null on *both* tiers). Read the
[correction](findings.md#-corrected-2026-08-18-the-record-region-does-not-replicate--the-98500-count-is-seed-noise-and-pooled-is-the-only-metric-of-this-family-worth-reading)
before quoting the 21.

| batch | `c` | gate | net | verdict | write-up |
|---|---|---|---|---|---|
| **27** | 0.10 | 85 | `320` | **null** — pooled 85.2 vs b24 87.9, 0 of 4 ≥98%/500 | [Batch 30 2×2](completedRuns.md#batch-30--chase-safe-shaping-on-fc-200100100-c010-null-and-it-completes-the-shapingarchitecture-22) |
| **30** | 0.10 | 85 | `200,100,100` | **null** — pooled 83.3 vs b25 86.0, 0 of 10 held | [Batch 30](completedRuns.md#batch-30--chase-safe-shaping-on-fc-200100100-c010-null-and-it-completes-the-shapingarchitecture-22) |
| **28** | **0.20** | 85 | `320` | **null** — pooled 85.4, 0 of 4 held; the dose is not the issue | [Batches 28-29](completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region) |
| **29** | 0.10 | **75** | `320` | **records** — pooled 87.8, **21 held ≥98%/500 in 2 seeds**, best `b29b` @1447k **99.0%/500** | [Batches 28-29](completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region) |
| **34** | 0.10 | **70** | `320` | **null** — pooled 86.4, 0 of 4 held; a 5-length step off 75 loses it | [Batch 34](completedRuns.md#batch-34--chase-safe-c010-gate-70-null--gate-75-is-a-narrow-sweet-spot-not-a-threshold) |
| **35** | 0.10 | **40** | `320` | **null** — 0 of 3 held (b35c pending), *highest pooled 88.2*; consolidation ≠ records | [Batch 35](completedRuns.md#batch-35--chase-safe-c010-gate-40-null--the-sweet-spot-at-75-is-isolated-not-a-plateau) |

**The lever is the gate, and 75 is an isolated sweet spot.** Gate 85 is null on `fc 320` (b27), on
`fc 200,100,100` (b30) and at doubled dose (b28); gate 70 (b34) and gate 40 (b35) are null too; only gate 75 (b29) matches the
control's pooled *and* produces a record region the control never did. The Φ calibration is why: the potential carries ~0 at lengths 98-99, so a gate-85
term grades the flat final approach, while gate 75 turns it on ten meals earlier, in the packing decisions
that decide whether the endgame is winnable. Full conclusion:
[`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## Batches 20-26 are closed — where their descriptions went

All seven batches finished and closed out, so per the bookkeeping rule at the end of this file their
descriptions moved to [`completedRuns.md`](completedRuns.md):

| batch | change | verdict | design + results |
|---|---|---|---|
| **26** | **fc `100,100`** under IS-off | **does not carry the lift** — pooled 79.2, only +3.5 over the control, at **more** parameters than b24's `320`. The arm that showed the lift is width, not size | [write-up](completedRuns.md#batch-26--fc-100100-under-is-off-the-shallow-shape-does-not-carry-the-lift) |
| **25** | **fc `200,100,100`** under IS-off | **the lift replicates** — pooled 86.0, +10.3, all 4 seeds; peak unmoved. No hall entry: the auto chain's gate-98 abandoned every candidate, `b25b` @911k still 97.2% at 392 episodes | [write-up](completedRuns.md#batch-25--fc-200100100-under-is-off-the-lift-replicates-at-a-second-shape--but-no-record) |
| **24** | **fc `320`** under IS-off | **first architecture result + a new record** — pooled **87.9**, +12.2 over the control, all 4 seeds (ceiling unmoved). HOF-500: `b24d` @1342k **98.0%/500**, the new record | [write-up](completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record) |
| **23** | IS β annealed **0→0.1** | **the best point on the β ladder** — pooled **75.7**, +20.7 over the control, higher on all 4 seeds | [write-up](completedRuns.md#batch-23--β-annealed-001-the-best-point-on-the-β-ladder-near-the-no-is-extreme) |
| **22** | IS **off** (`SNEK_IS_WEIGHTS=0`) | **dead heat with β→0.1** — pooled 75.7. The consolidation gain saturates by β→0.1 | [write-up](completedRuns.md#batch-22--is-off-a-dead-heat-with-β01--the-consolidation-gain-saturates) |
| **21** | partial IS (β→**0.5**) | beats the β→1.0 control (pooled 64.3 vs 55.0, 3/4 seeds), well short of no-IS | [write-up](completedRuns.md#batch-21--partial-is-β05-beats-the-β10-control-still-far-behind-no-is) |
| **20** | `FC_LAYERS`, **nine shapes** | **architecture never raises the ceiling**; capacity binds only below ~0.55× | [the sweep's design](completedRuns.md#batch-20--the-design-of-the-nine-shape-sweep-complete-2026-08-12) |

**‡ The width result now has three shapes behind it, and "it tracks capacity" is retracted.** Pooled lift
over the b22 control orders by **widest layer**, not by size: `320` **+12.2** at 0.94× the control's
parameters, `200,100,100` **+10.3** at 3.09×, `100,100` **+3.5** at 1.14×. The smallest of the four nets
wins and the second-largest gets almost nothing, so parameter count is not even monotone with the result
([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
**The architecture arm this implies is `fc 512` under the b24 config** — a wider first layer, not a bigger
net — and it is now the strongest remaining consolidation direction after the shaping batches.

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

**‡ Candidate, 2026-08-19, pass still running: `b43a-lowlr-b29b` @1618000 — 99.4%/500 (497/500).** It is
`b29b`'s record checkpoint continued at `lr 1e-6` for another 171k steps, so if it holds it is both a new record
and the ladder's headline. **Not claimed yet**: its HOF-500 pass is ~25% done, 497/500 against the standing
495/500 is well inside the CI, and it is the maximum over 59 measured checkpoints — the winner's curse that has
already been corrected twice in this file. See
[the HOF-500 note](#-b43s-hof-500-is-done-b44s-is-half-done--and-its-500500-did-not-survive-re-measurement).

**NEW RECORD, 2026-08-16: `b29b-chase10g75seed2` @1447000 — 99.0% over 500 fresh episodes** (495/500, CI
97.7-99.6). It is the highest /500 point estimate on record, above `b24d`'s 98.0%/500; the lead is inside the
500-episode CIs, so it is a *narrow* point lead — taken as the record under the folder's 500-episode standard,
exactly as `b24d` was taken over `b18b`. b29b carries an
**18-checkpoint ≥98%/500 band** (1446k-1529k) and its sibling `b29a` holds 3 more — 21 record-tier checkpoints
across 2 seeds. **‡‡‡ Corrected 2026-08-18: that region was read as the config's property and it is not.** `b37`,
byte-identical on seeds 5-8, held **0 of 4**, and `b40` held 1, on comparable ≥98%/100 tiers — so the 21 is a
**seed**, and the record stands as a *point* lead like every one before it. **Promoted to [`../hallOfFame/`](../hallOfFame/README.md) on 2026-08-16** — the checkpoint was rsynced
off the desktop and the copy re-measured 98/100 on fresh laptop episodes (loads and plays like a champion).
Write-up: [Batches 28-29](completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region).

**Promoted 2026-08-18: `b40b-chasefree10g75seed2` @1513000 — 491/500 = 98.2%** (CI 96.6-99.1), now in
[`../hallOfFame/`](../hallOfFame/README.md#-b40b-1513000--third-best-measured-and-the-one-entry-whose-batch-is-a-null).
Third-best /500 in the folder, behind `b29b` (99.0) and ahead of `b24b`/`b24d` (98.0); it *rose* from 98.0/100.
Checkpoint rsynced off the desktop and **the copy re-measured 97/100 on fresh laptop episodes** (avg score
94.39, min 64). **Its batch is a null** — it is preserved on its own number, not as evidence for the free-space
term. It also carries a ≥97%/500 cluster at 1509k-1545k that the auto pass's gate 98 abandoned.

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

1. **`CHASE_SAFE_SHAPING`: all four batches (b27-b30) closed. Gate 85 is null at any dose or net; `gate 75`
   (b29) produced a 21-checkpoint ≥98%/500 region, the gate is the lever.** `b29b` @1447k (99.0%/500) is
   promoted to [`../hallOfFame/`](../hallOfFame/README.md) as the new record (2026-08-16, copy verified).
   **Two gate-ladder follow-ups on the desktop (both `c=0.10`, control `b24a-d`): `b34a-d` gate 70 running,
   `b35a-d` gate 40 queued behind it** (2026-08-16). b34 is one rung below b29; b35 is a bigger step into the
   mid-game to test whether chase-safety shaping helps outside the endgame at all. A monotone gate response
   (40 ≥ 70 ≥ 75) would say the causal horizon reaches deep; a knee would locate the useful gate. Both are
   `c=0.10` so per-flip dose is fixed and only the gate moves (total episode dose rises ~2.5× at gate 40).
   Sections at the top of this file. Design and Phase 0:
   [`../plans/chase-safe-reward-shaping.md`](../plans/chase-safe-reward-shaping.md); full b29 result in
   [`completedRuns.md`](completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region).
2. **`fc 512` under the b24 config is now the strongest untested architecture arm.** b25/b26 turned the
   width result into an ordering on the *widest layer* — 320 → +12.2, 200 → +10.3, 100 → +3.5, 50 → 0 —
   with parameter count not even monotone
   ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)).
   `512` is the only cheap way to ask whether that ordering keeps going or has a knee, and it needs no new
   control: `b24a-d` is seed-matched. Order it behind the shaping batches, which own the queue.
3. **4 more `320` seeds are still owed before the width×IS-off interaction is called established.** b24's
   +12.2 is at the n=4 sign-test floor (p=0.0625). b25/b26 answered a different question (which property
   carries the lift) and do not firm up the `320` gap itself.
4. **Consider a position-chosen grid around `b18b` @1588000.** The record is a narrow peak, so the
   open question is whether `TARGET_UPDATE_PERIOD=1000` produces a better *region* or just got one
   lucky checkpoint. A blind every-10k grid over 1.55-1.62M would settle it, and it is the same test
   that deflated `b17b`'s apparent region to 84%.
5. **The 11 batch-18 checkpoints at exactly 95.0%** were excluded from the 500-episode sweep, which
   took ">95%" literally. ~9 minutes of eval if a fuller picture of the region is wanted.

## Closed batches (11-19)

One line each; **batches 20-26 are in the table above**, with write-up links. Full write-ups and
per-seed numbers in [`completedRuns.md`](completedRuns.md), superseded detail in
[`archive/runs-archive.md`](archive/runs-archive.md).

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
trailing reads 93.8-95.0 flat across batches 11-23. Nothing has raised the ceiling; batches 16 and 21-23
raised how much of the *time* an arm sits near it. **Two things have moved peak trailing downward** —
batch 19's full IS correction (94.16, 4/4 seeds) and batch 20's 0.29× net (93.75, 4/4) — so the invariance
is breakable, just not yet upward. **Fifteen batches of optimiser, PER and architecture knobs have not
raised the ceiling once** — peak trailing reads 94.84-95.00 across `50,100,50`, `100,100`,
`200,100,100` and `320` — which is the argument behind the reward-shaping batches now running (b27-b29).

## Standing backlog

Untested, ordered by expected value. Rationale for the ones that need it follows the
table.

| change | targets | prior |
|---|---|---|
| ~~`CHASE_SAFE_SHAPING`~~ — potential-based shaping on head/food/tail in one region | endgame food-finding, the modal failure since batch 16 | **closed 2026-08-16** across b27-b30: gate 85 null at any dose/net, **`gate 75` (b29) produced a record region** ([finding](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever)). The live follow-up is gate-75 replication + promoting `b29b` — see Outstanding #1 above |
| **Free space in one piece** — as a graded potential and/or an observation | endgame packing, which is what decides whether the food lands somewhere edible | **new 2026-08-14**, and the largest per-policy separation on record: one-piece share at length 90-94 is **92% / 77% / 5%** for `b24d` / `b18b` / `b20d` ([findings](findings.md#-the-packing-property-the-records-keep-their-free-space-in-one-piece-and-it-separates-them-by-87-points)). `count_groups` already runs every step, so both forms are nearly free. **Correlational, n=3 checkpoints** |
| **`SNEK_ALGO=c51`** — distributional RL, a categorical value head instead of a scalar Q | post-peak collapse, which has no established mechanism now that plasticity loss is ruled out | **proposed 2026-08-15, awaiting review** — [plan](../plans/distributional-c51.md). The first change to the *loss* rather than a reward, width or schedule, and the only backlog item that needs a code change before it can be launched by env var. A feasibility probe already ran: the shield, the buffer spec and the network all work unchanged, and `tf_agents`' own C51 loss **silently drops IS weights and returns no PER priority**, both of which the plan fixes. Honest prior: ~40% null on `best_perfect30` |
| `LEARNING_RATE=1e-4` | training speed | high, but order it after a stability fix |
| ~~`TARGET_UPDATE_PERIOD`~~ | early learning speed, target stability | **closed as batch 18** — primary moved, 4/4 seeds; see [`completedRuns.md`](completedRuns.md) |
| `TARGET_UPDATE_TAU=0.005`, period 1 | smoothness (soft target updates) | medium |
| ~~`FC_LAYERS=128,128`~~ | capacity | **closed by batch 20**, which swept nine shapes rather than one: none raised the ceiling — [findings](findings.md#network-shape-the-sweep-is-complete--nine-shapes-and-architecture-never-raises-the-ceiling). Width under IS-off is the one loose end, and batches 24/25 are on it |
| **`FC_LAYERS=512`** under the b24 config | consolidation — how far the widest-layer ordering goes | **new 2026-08-14, and the top architecture item.** b24/b25/b26 order the lift by widest layer (320 → +12.2, 200 → +10.3, 100 → +3.5) with parameter count non-monotone ([finding](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count)). `b24a-d` is already the seed-matched control, so the arm costs four trainings and nothing else |
| ~~epsilon ladder *shape*~~ | exploration schedule | **done 2026-08-04** — rewritten, needs measuring |
| `REPLAY_BUFFER_MAX_LENGTH=1000000` | experience diversity | low — the 500k result was ambiguous |
| **`eval_wave.py`** — one controller owns a wave of evals, on both hosts | eval wall clock, and the laptop/desktop asymmetry. **Infrastructure, not a training change** | **proposed 2026-08-19, scope approved** — [plan](../plans/eval-wave-controller.md). Whole-checkpoint units handed to shared lanes, so a wave stops idling on its stragglers: measured **1.7-2.8x on a HOF wave** (b40 16/63/9/2 candidates), 1.0-1.2x on a recent close-out and up to 3.0x on the uneven ones (b19, b17). Also deletes `chain_closeout_after_training.sh` and the gate constants duplicated between it and `runner.py` |

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
| desktop `the-claw-den` | **`git fetch origin ops-status &&`** `git show origin/ops-status:status.json` — read `counts`, `running`, and the `iso` heartbeat |

Not `grep "[s]nek2.py"` on the laptop — git telemetry `curl` processes carry `snek2/snek2.py` in their
payload and inflate the count.

**And `git fetch` first — this is the most-repeated mistake in this project's dealings with the desktop.**
`git show origin/ops-status:…` reads a *local* remote-tracking ref, so with no fetch you get an old
snapshot **and no sign that it is old**; because the payload carries a timestamp, it reads as a dead
daemon. Three false alarms so far: 2026-08-12 (**17 hours stale**, four finished evals reported as still
running) and twice on 2026-08-17, the second of which reported a 10-hour-dead daemon and a batch that had
"failed to dispatch" while the box was healthy, had finished that batch *and* its close-outs *and* its
HOF-500s, and had moved on to the next wave. **A stale-looking `iso` is your own ref until you have
fetched and re-read it**, and `ssh the-claw-den -o ConnectTimeout=8` settles reachability in one command
rather than by inference. Ladder and rationale:
[CLAUDE.md](../../CLAUDE.md#there-are-two-compute-hosts--say-which-one-you-mean).
