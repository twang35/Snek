# Findings

What is established, what is falsified, and what is still open. Organized by topic rather than
by when it was discovered.

**Read this before proposing an experiment** — several questions here are closed, and a few were
closed *narrowly* in ways worth checking before reopening them.

| | |
|---|---|
| what is running, what is next | [`runs.md`](runs.md) |
| per-arm numbers and verdicts | [`completedRuns.md`](completedRuns.md) |
| how to measure and judge | [`hyperparamTuning.md`](hyperparamTuning.md) |
| the four degradation patterns | [`failureModes.md`](failureModes.md) |
| superseded findings, batches 1-10 | [`archive/`](archive/) — history only, don't load it |

## Status at a glance

Only current and load-bearing findings. Results about observation vectors this project has
replaced (20, 21, 23, 26 values) and per-batch config results that later batches settled live in
[`archive/findings-superseded.md`](archive/findings-superseded.md).

**Environment and checkpoints**

| finding | status |
|---|---|
| The vector is **30 values**; only batch 11+ checkpoints load on `master` (`450e66e` = 26, `e4514a8` = 20) | **breaking** |
| A same-width observation change loads silently and plays like a beginner — 90.3% → scoring 0 | **standing hazard** |
| Index 29 (food-space) reads 1 in **99.95%** of states, so its weights are barely trained | **hazard**, don't repurpose it |
| ~~Nothing in the vector distinguishes snake lengths 50 to 99~~ | **fixed 2026-08-02** — index 22 is linear board-fill, so 50 and 99 differ |
| The 2026-08-03 observations gave +4 to +5 pp on three metrics, none significant | **open**, n=4, p 0.14-0.24 |

**Records and the horizon**

| finding | status |
|---|---|
| **The record is 96%**, held jointly by `b11b` @855k and `b14a` @3702k; both correct to ~94% | **measured**, 96/100 each; `b14a` re-measured 187/200 |
| Two of the four record jumps came from the **horizon** and the **env audit**, not hyperparameters | **established** |
| An arm has a lifetime: peak ~2.5-3M steps, dead by ~7M | **established**, 2 arms to the end |
| Arms peak by ~3.4M | **falsified** — 2 of batch 14's 4 peaked past 3.5M, and 2 of batch 15's were still gaining at **5.5-6.0M**; the old rule tracked where humans stopped arms |
| The horizon was the binding constraint — records live past 2.5M, early arms stopped at ~1.06M | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Arms recover from long zero stretches — `b8g` came back from 1.2M steps at zero | **established** |

**Config**

| finding | status |
|---|---|
| `DISCOUNT=0.995` matches the best ceiling and survives 3 of 3 seeds | **measured**, ~2.3x expected value |
| Higher discount is monotonically better | **falsified** — 0.999 died 2 of 2 |
| `0.995` vs `0.9975` on the current environment | **falsified as a difference** — batch 14 null vs 13, `pooled_equal_effort` +0.01 pp, n=4 paired |
| `td_loss` + alpha 0.8 + no IS is effectively alpha 1.6 | **established** — arithmetic |
| No prioritization setting tested so far survives reliably | **established**, 7 seeds |
| `GRADIENT_CLIPPING=10` on 0.995 helps | **falsified** — 1 of 3 seeds, no ceiling gain |
| n-step returns help | **falsified on speed** — batch 15 at n=3 reached pf30 >= 40% **128k later** than its control, 3 of 4 seeds slower; level is a null (+4.05 pp `sef`, p=0.625). Evals pending |
| A larger replay buffer prevents the collapse | **not settled** — opposite results twice |
| Epsilon reaching 0.0 causes the collapse | **falsified**, *but only at 0.001 vs 0.0* |
| **96.8% of batches 10-11's steps ran at epsilon exactly 0.0**, the ladder bottoming out at ~15k | **measured**, 8 arms, 31.1M steps |
| Elevated exploration (handover 0.05) helps | **falsified** — batch 12 deadlocked, 0% perfect 4 of 4 |
| Elevated exploration (handover 0.0125) helps | **falsified** — batch 13 null on **five** metrics, n=4 paired |
| A one-step exploration shield helps | **open**, confounded twice — nothing to fix at 0.0125, and `GUIDED_FRACTION` 0.8 moved with the discount in batch 14 |
| A seed number is a stable unit of quality across configs | **falsified** — batch 11's best seed became batch 13's worst |
| The epsilon *ratchet* was a real defect | **standing**, on mechanism: no recovery from a collapse |

**Measurement**

| finding | status |
|---|---|
| **`fraction of evals >= 80%` has the lowest between-seed variance** of the candidate metrics | **measured**, sd 5.8 vs 8.6 for best-30 |
| Abandoning a checkpoint eval early is not worth it | **falsified for an arithmetic rule** — 30% saved at an 85% achievability gate, 48% at the 90% gate shipped 2026-08-06; only the *predictive* version was 14% |
| **n=4 cannot resolve an effect below ~10 pp**; 5 pp needs n≈17-37 depending on the metric | **established** |
| 100-episode measurement reproduces within binomial noise | **established**, 51 repeats |
| The max of N noisy measurements is upward-biased — shrink it before quoting it | **established** — `b14a`'s 96/100 re-measured 91/100, pooling to 93.5% |
| The graph-100% tier is comparable across arms and batches | **falsified under a gate** — `EVAL_MIN_ACHIEVABLE` censors it from below; reads +15.6 pp on batch 14 as pure artifact |
| A 100% single graph eval is the only graph value with a usable floor | **measured**, 9 of 9 above 64% |
| A high single 10-episode eval predicts a good checkpoint; smoothing is anti-predictive | **established**, +0.64 vs −0.40 |
| Policy quality changes materially within 1000 training steps | **established**, up to 27 points |
| Checkpoint-to-checkpoint variance is large, and it is not sampling noise | **established** |
| The graph misranks arms badly — `b5c` is 2nd by graph, last by measurement | **established** |
| This domain is very noisy: the same config has produced 62.5 and 18.0 | **established** |

---

## The discount: an optimum near 0.995-0.9975, and now a closed question

The one hyperparameter that has reliably helped. Not monotone — the sweep has a peak and falls off
hard on both sides:

| discount | eff horizon | outcome | verdict |
|---|---|---|---|
| 0.99 | ~100 | 12.0% measured, dies 2 of 4 seeds | too short |
| **0.995** | ~200 | 38.8% measured, **survived 3 of 3**; the current default | best expected value |
| **0.9975** | ~400 | held the record twice (92%, then 69.3% best-30), but 1 of 2 on survival | best ceiling |
| 0.999 | ~1000 | **dead 2 of 2**, at 452k and 398k, peak trailing 63.1 / 31.8 | too long |

**`0.995`'s gain is reliability, not ceiling.** Priced for survival it is ~2.3x the best previous
config: 28.2% mean level at 3 of 3 surviving, against `b4c`'s 37.1% at 1 of 3 (expected value 12.4%).
The ceiling claim made for it originally would have been wrong — `0.9975` beats it there.

**`0.995` vs `0.9975` is settled on the current vector, and the answer is "no difference".**
Batch 14 ran 0.9975 against batch 13's 0.995 at n=4 paired and came back null on every metric that
survives its abandonment gate — `pooled_equal_effort` **72.08% against 72.07%**, best checkpoint
+2.75 pp at p=1.000, `best_perfect30` +1.08 pp at p=0.625. Write-up:
[`completedRuns.md`](completedRuns.md#batch-14--disc-09975-at-guided-08-and-the-widest-seed-spread-yet).

That closes the question batch 9 left open at n=2, and it closes it against a specific hypothesis
worth recording as dead: **the 2026-08-03 endgame observations do not need a longer horizon to be
usable.** The argument for re-asking was that following-tail (26-28), food-space (29) and
reachable-tail (9-14) all describe structure 300+ steps out, while 0.995's effective horizon is ~200
against a perfect game's ~1780. Measured, it makes no difference. **Do not re-ask at n=4.**

Batch 9's reason for staying open still describes the shape of the problem: the two values won
*different* things there (0.995 better expected value, one 0.9975 seed dead at 328k; 0.9975 the
steadier single arm), and **the seed spread exceeded the effect** — batch 9's two 0.995 seeds were 18
points apart on best checkpoint. Batch 14 is the same story with more arms: -16.2 to +24.8 pp per
seed on the primary around a +2.05 pp mean.

**Stop sweeping the discount.** Above 0.9975 is measured dead, 0.995 and 0.9975 are measured
indistinguishable, and an interior point like 0.996 cannot plausibly differ from either by more than
the ~10 pp n=4 can resolve. This is now a closed question, not a narrower one.

Two process notes that came out of batch 9 and still apply:

- **A partial close-out is not a small version of a complete one.** `b9d` at 12 of 17 checkpoints
  reported numbers that moved materially once the rest landed.
- **The best checkpoint and the trailing peak are in different places** — `b9a`'s best is at 1735k
  while its trailing peaked at 3277k. Confirmed again across batches 10-11, where the gap ran to 3M
  steps on `b10b`. Do not use a graph peak to decide where to look for a checkpoint.

Per-batch tables: [`archive/findings-superseded.md`](archive/findings-superseded.md) and
[`archive/batches1-11.md`](archive/batches1-11.md).

## Graph evals are a filter, not a ranker

A graph point is 10 episodes, so `perfect_percent` only takes values 0, 10, … 100. What that
signal can and cannot do, from 88 checkpoints measured on 2026-07-30 — the largest sample here:

| question | answer | evidence |
|---|---|---|
| Does a high single eval beat a smoothed one for *selecting*? | Yes, decisively | +0.64 vs **−0.40** correlation; outlier picks measured 41.3% vs 27.1%, CIs disjoint |
| Among already-high checkpoints, does the graph value rank them? | No | +0.10; 90% and 80% points are indistinguishable (57.9% vs 58.6%) and both span ~50 points |
| Does the *surrounding* rate rank them? | Yes | **+0.48** |
| Is any graph value trustworthy on its own? | Only 100% | 9 of 9 measured at ≥64%, mean 72.5%; 90%/80% reach into the 20s |

**The −0.40 and the +0.48 are not a contradiction**, and it took a while to see why: the first
compares *selecting* on smoothed vs raw across a wide range, the second asks whether — among
checkpoints that already spiked — the region rate picks the best. Both are true. Range
restriction attenuates every correlation in the second column.

So: **measure 100% points first** (rare and reliably top-decile), measure the whole ≥90% tier
because there is no way to tell the 88% from the 33% in advance, and use the surrounding rate as
the tiebreak. That is what `eval_checkpoints.py` implements.

**A 100% graph point is not a shortlist of champions**, though — 6 of the 8 arms measured across
batches 10-11 found their best checkpoint in the 90% tier, which is ~4x larger. See
[`hyperparamTuning.md`](hyperparamTuning.md).

**`b6b-alpha06` and `b6a-alpha04` were measured with the old smoothed selector and are
underestimates.** They cannot be fixed by re-measuring — `b6a`'s best graph point in 1415 evals is
50%, so the current thresholds yield nothing from it. The alpha comparison needs new seeds.

## Three measurement caveats

**‡ An abandonment gate silently invalidates every pooled figure except `pooled_equal_effort`.**
Learned on batch 14, the first batch measured under `EVAL_MIN_ACHIEVABLE=90`. The gate stops a
checkpoint once its ceiling drops below 90%, which means every surviving full-length row is a
*winner* — so any statistic pooled over "the rows that reached full length" is censored from below.
Two casualties:

| statistic | what the gate does to it |
|---|---|
| **graph-100% tier** | reads 90.3% on batch 14 against batch 13's 74.6%, a +15.6 pp artifact. Tier sizes fell from 31-114 checkpoints per arm to 1-28. **Unusable.** |
| **winner's-curse shrinkage** | was fitted on each arm's *unselected* graph-100% rows; under a gate those rows are abandoned and biased low by optional stopping. **Not computable.** |
| `pooled_equal_effort` | **exact at any gate** — it truncates every checkpoint to the 20-episode screen depth, and abandonment cannot fire before the floor |

The reason this was easy to miss is that the gate's safety argument is about *rankings* — an
abandoned row can never outrank a kept one, which is true — and says nothing about *pooling*. Check
`min_achievable` in a payload before pooling anything from it, and never compare a gated figure to an
ungated one. The substitute for shrinkage is a second independent 100-episode run on the champion,
which is stronger evidence anyway: `b14a`'s 96/100 re-measured **91/100**, pooling to 93.5%.

**Pooled rates only compare when the selection rule matches.** The rule has changed several times
and the checkpoint count now varies per arm (1 to 660), so pooling over 16 checkpoints and over 1
are not the same statistic. **Use best checkpoint for cross-arm comparison**, and read pooled as a
within-arm consistency check: a config whose best and pooled figures are close is producing a
strong *region*, which is what the project is actually chasing.

**A single 100-episode figure is usable at ±10.** 51 checkpoints were measured twice on the same
day: mean spread **4.8** points, 47 of 51 within ±10, no systematic direction — comfortably inside
binomial expectation. An earlier warning here, built on `b4c` @869000 reading 51 / 42 / 32 across
three runs, should be read as "one checkpoint once behaved strangely" rather than a property of
the instrument. Pooling over many checkpoints is still what shrinks the interval (±1.3 at 6300
episodes).

## Policy quality changes materially within 1000 training steps

Evaluating each high-single-eval checkpoint **together with the checkpoints immediately
either side of it** — 100 episodes each — settles whether "this checkpoint is good" can be
distinguished from "this part of the run is good". It can:

| cluster | centre | neighbours at +/-1000 | centre advantage |
|---|---|---|---|
| 851000 (`b4c`) | **40.0%** | 28.5% | **+11.5 points** |
| 869000 (`b4c`) | **32.0%** | 23.0% | **+9.0 points** |
| 970000 (`b4c`) | **35.0%** | 7.5% | **+27.5 points** |
| 2806000 (`b8f`) | **80.0%** | 74.0% | **+6.0 points** |

Pooled over the first three, centres measure 35.7% (CI 30.5-41.2) against neighbours' 19.7%
(CI 16.7-23.0) — non-overlapping, and the effect is in the same direction in **4 of 4** clusters.

The `b8f` cluster is the weakest confirmation and the most informative one. Its graph values read
80% / **100%** / 70% and measured 74% / **80%** / 74%, so the centre still won — but by 6 points
with overlapping intervals, on an arm where *every* checkpoint in the region is strong. **The
advantage shrinks as the surrounding region improves**, which is what you would expect if the
spike reflects a genuinely better policy rather than a measurement artefact: there is less room
above a 74% neighbourhood than above a 7.5% one.

The 970000 cluster is the extreme case: **969000 measures 8%, 970000 measures 35%, 971000
measures 7%.** Those are 100-episode measurements, so **1000 training steps can gain or
lose 27 points of perfect-game rate.** Training is far more non-stationary at the
checkpoint level than this investigation previously assumed, and adjacent checkpoints are
not interchangeable samples of one policy.

## Checkpoint-to-checkpoint variance is large, and it is not sampling noise

Within `b6b`'s 1455-1464k cluster — 9000 train steps end to end, checkpoints that should
be nearly identical policies — measured rates at 100 episodes each:

| ckpt | 1455k | 1456k | 1461k | 1462k | 1463k | 1464k |
|---|---|---|---|---|---|---|
| perfect % | **36** | 25 | 24 | **16** | 24 | 31 |

**A 20-point spread across 9000 steps.** At 100 episodes each these are real differences,
not sampling error. Consequences:

- **One checkpoint does not characterise a policy region.** Evaluating a single checkpoint
  from this cluster would have yielded anywhere from 16% to 36% depending on the draw.
- **Pool across several checkpoints** for any number that gets compared across arms. This
  is why `top20` deliberately allows adjacent picks: spacing them out hides exactly this.
- The published **51% for `b4c` at 869000 is one checkpoint**, so it is the top of a
  distribution like this one, not the config's level. `b4c`'s pooled 31.8% is the fairer
  figure.
## The mechanism: `td_loss` doubles the effective priority exponent

`common.element_wise_huber_loss` uses delta 1.0, so for `|td_error| < 1` — most transitions once
a policy is decent — `td_loss = 0.5 * td_error^2`. Priorities are then raised to alpha, so
squaring inside and exponentiating outside **compounds**: `PRIORITY_SIGNAL=td_loss` with alpha
0.8 is really an exponent of ~1.6.

**So alpha 0.8 was never the config under test**, and every `td_loss` arm is incomparable to its
alpha label. The three "PER changes" recovered from `theSchlong` were also never independent:
`td_loss` and alpha 0.8 multiply into an extreme exponent, and dropping IS weights removes the
only thing correcting the resulting bias.

The current config is alpha 0.6 + `td_loss` = **effective ~1.2**.

**Sharpness is a variance dial, not a quality dial.** Among surviving arms the ceiling rises
monotonically with the effective exponent (~1.6 → 34%, ~1.2 → 21.7%, ~0.8 → 14.3%), and so does
the death risk (2 of 3 arms at ~1.6 died outright). The two best arms were also the two with the
most near-death excursions.

Two caveats that weakened the original version of this finding:

- **Lower sharpness may only delay death.** Seeding eff ~1.2 four times gave 2 deaths of 4, at
  573k and 1162k, against eff ~1.6's 2 of 3 at 246k and 272k. At these sample sizes 50% and 33%
  are not different, so "safer" has no support — but "later" does. **Any survival rate quoted
  here needs a fixed step horizon attached**, or it is partly an artefact of run length.
- **The "stability cliff" between eff 0.8 and 1.2 is retracted** — `b6b` crossed it and thrived.

Full per-arm tables and the batch-by-batch derivation:
[`archive/findings-superseded.md`](archive/findings-superseded.md).

## Falsified: epsilon reaching 0.0 does not cause the collapse

The hypothesis was that the epsilon ladder's last rung (`epsilon.assign(0.0)` once
`avg_reward > 100`) makes the collect policy fully greedy, turning the replay buffer into
a closed loop on the policy's own behaviour. The evidence looked strong: `b1a-base` was
the only arm that reached 0.0 and the only arm that collapsed, with a mechanism and a
timing that fit (0.0 at 92k, collapse at 265k, ~173k apart — about the time to flush a
100k buffer at ~800-step episodes).

Three arms settled it, all past the judgeable horizon:

| arm | epsilon regime | outcome |
|---|---|---|
| `b3b-epsfloor2` | floored at 0.001 from 147k | peaked 305k, declined 71 → 52 and 7.0% → 3.3% |
| `b3a-epsfloor` | floored at 0.001 from 267k | peaked ~300k, declined 74 → 61 and 8.6% → 1.3% |
| `b3c-buf500k` | **fully greedy at 0.0** from 282k | did not break in the predicted window; later died at 750k for unrelated reasons |
| `b4c-schlongper` | fully greedy from 121k | the best arm in the investigation |

The prediction failed in both directions. **The caveat recorded when the hypothesis was
proposed turned out to be the entire signal:** reaching 0.0 requires `avg_reward > 100`,
so only *strong* runs get there — "reached 0.0" and "was good enough to collapse from a
height" are entangled and the correlation cannot separate them.

Two things worth carrying forward:

- **That correlation was as strong as this domain produces** — one arm at 0.0, one arm
  collapsed, same arm, specific mechanism, timing that fit. It was still wrong. With n=1
  arms and a stated confound, a mechanism that "fits the timing" adds no evidence.
- **The test was still worth running.** It cost one knob and three arms that were going to
  run anyway, closed the question, and incidentally produced batch 3's best arm and the
  natural experiment that settled it.

`MIN_EPSILON` stays in the code, and knowing epsilon 0.0 is *safe* relative to 0.001 is a
useful result.

### Scope of that falsification, added 2026-08-04: it was never about the descent rate

**This result stands, and it is narrower than its own heading.** What batch 3 compared was a
floor of **0.001 against 0.0** — and 0.001 is ~1.2 forced non-greedy moves in a 1780-step game,
so both arms were playing essentially greedily. The test established that those two are
indistinguishable. It did not test exploration, because neither condition had any.

What went unexamined for four months is how fast the ladder got there. Measured across batches
10 and 11: **96.8% of all 31.1M training steps ran at epsilon exactly 0.0**, 99.6% at ≤0.001,
and the ladder bottomed out at median step 15000 of runs 3.2-4.7M long — while 7 of 8 arms were
still at 0% perfect games. So the correct reading of batch 3 is:

| tested | not tested |
|---|---|
| floor 0.001 vs 0.0, both effectively greedy | any floor large enough to change behaviour |
| whether the *last rung's value* matters | whether reaching the last rung at step 15k matters |

The schedule was rewritten 2026-08-04 for that reason — two phases, no ratchet, floor 0.002,
and exactly 0 rejected. See
[`hyperparamTuning.md`](hyperparamTuning.md#the-epsilon-schedule--rewritten-2026-08-04-and-it-breaks-curve-comparability).
**That rewrite is not evidence against anything on this page.** It is a change to an
untested part of the design.

#### Now measured, 2026-08-05: exploration was tested in both directions, and neither helped

Two batches closed the gap this section described, and the answer is that the untested part of the
design was untested because there was nothing there.

| batch | handover epsilon | result |
|---|---|---|
| 12 | 0.05 | **deadlock.** 4 arms, 0% perfect games to ~1M, greedy trailing 53-63 vs 84-88 |
| 12s | 0.05 + exploration shield | decay fixed, still plateaus at trailing ~83 with 0.3% perfect |
| 13 | 0.0125 + shield | works, and is **null on five metrics** vs batch 11, n=4 paired |

Batch 13's five, now that its checkpoints are measured: best ckpt -1.8 pp (p=0.875), top-3 -1.4 pp
(p=0.875), graph-100% tier **-0.1 pp** (p=1.000), `best_perfect30` **+0.0 pp** (p=1.000),
`strong_eval_fraction` +2.0 pp. Two independently-computed metrics landing within 0.1 pp is what a
genuinely zero effect looks like measured five ways.

So the honest closing position on epsilon: **too much exploration is actively harmful and the right
amount is indistinguishable from none.** 0.05 is fatal to sit at, because a collect policy at 3.3%
random actions never finishes a board and the buffer never holds the last ten food. 0.0125 is
harmless and buys nothing measurable.

What survives is only what was defensible on mechanism in the first place: the **ratchet** was a
real defect (`b11b` sat at 0.001 through a collapse from 64.6 to 8.8 with no way to recover), and
exactly 0.0 makes the buffer a closed loop. A 0.002 floor plus a stateless schedule fixes both
without needing elevated exploration. The original framing — "96.8% of steps at exactly 0 is a
defect" — was **cosmetic**, and it cost two batches to find that out.

**Do not re-open this at n=4.** Three batches have now failed to separate any epsilon regime from
batch 11's on a metric whose between-seed spread is ±12 pp.

The lesson worth keeping is about how the original hypothesis was framed. "Epsilon 0.0 causes
the collapse" named a *value*, so the experiment tested a value and closed the question at that
value — while the schedule that produced the value went unexamined. A hypothesis about a knob
should say which property of the knob is doing the work.

### Related: when each arm's epsilon treatment actually started

`MIN_EPSILON` only changes behaviour at the last rung, and crossing `avg_reward > 100` is
uncommon and late. **One crossing is all it takes**, because the ladder is a one-way
ratchet — a single eval over 100 pins epsilon permanently, and a later score drop never
raises it back.

| policy | first `avg_reward > 100` | epsilon after |
|---|---|---|
| `b4c-schlongper` | 121k | 0.0 |
| `b1a-base` | 92k (18 evals over) | 0.0 |
| `b3b-epsfloor2` | 147k | 0.001 (floored) |
| `b3a-epsfloor` | 267k | 0.001 (floored) |
| `b3c-buf500k` | 282k | 0.0 |
| `b4a-uniform` | 425k | 0.0 |
| `b4b-unifbuf500k` | 290k | 0.0 |
| `b2a-base2` | never (peaked 99.1) | 0.001 |

This is why a floored arm can be indistinguishable from an unfloored one for its first
few hundred thousand steps, and why `b3a-epsfloor` spent 267k steps as an accidental
baseline repeat.

## Not settled: whether a larger replay buffer helps

`REPLAY_BUFFER_MAX_LENGTH=500000` was tested twice with opposite results:

| arm | buffer | sampling | outcome |
|---|---|---|---|
| `b3c-buf500k` | 500k | PER alpha 0.6 | flattest curve in the investigation, then **died completely at 750k** |
| `b4b-unifbuf500k` | 500k | uniform | steadiest arm, healthy at 1.23M, but only 9.3% |

The difference between them is prioritization, not buffer size, which points the same way
as the `b4c` result. A 500k buffer with uniform sampling is stable-but-low; with PER it
died. Neither is evidence that buffer size is the lever, so **the diversity-squeeze
mechanism described in [`completedRuns.md`](completedRuns.md) is still an untested
hypothesis**, not a finding. `REPLAY_BUFFER_MAX_LENGTH=1000000` is in the backlog at low
priority.

## Re-opened 2026-08-02: n-step returns were never cleanly tested

| policy | steps | peak score (at) | best perfect-30 | 1st perfect |
|---|---|---|---|---|
| `b1c-nstep3` | 1.14M | 76.0 (255k) | 1.7% | 206k |
| `b2b-nstep2` | 580k | 74.6 (140k) | 0.7% | 121k |

Both peaked *below* every baseline, both then declined for hundreds of thousands of steps,
and both sat at zero perfect games in their trailing windows. Two arms ordered by n giving
the same shape looked like a trend rather than noise, and this overturned a still earlier read
that n=3 had "the best trajectory of the batch" — true through 200k, false afterwards.

**That conclusion is withdrawn, because the mechanism it tested was broken.** Terminal steps
carried a non-zero discount until 2026-08-02, and `to_n_step_transition` composes

```
r_t + g*d_t*r_{t+1} + g^2*d_t*d_{t+1}*r_{t+2} + ...
```

where those per-step `d` values are the **only** thing that truncates the sum at an episode
boundary. At `d = 0.9975` on a terminal step, an n-step return keeps accumulating past the end of
the episode into whatever sits next to it in the replay buffer. Both arms above were therefore
trained on returns that mix episodes together, which is a fair explanation for peaking below every
1-step baseline.

This is **not** evidence that n-step helps. It is a retraction of the evidence that it does not.

**Batch 15 tested `n=3` on 2026-08-06 and the predicted mechanism did not appear.** n-step's claim
here was faster credit propagation — the +100 perfect-game reward needs ~890 sequential backups to
cross a ~1780-step game at n=2 and ~593 at n=3 — so the pre-registered read was **steps to
pf30 ≥ 40%**. It came out **128k slower** than the batch-14 control, 3 of 4 seeds slower (p=0.250).
Level is a null: `strong_eval_fraction` +4.05 pp at p=0.625 equal-effort, peak trailing -0.10.

So the honest status is **falsified on speed, null on level**, at n=4. Contamination was never the
issue — at the measured epsilon of 0.0034-0.0039 an uncorrected n=3 return is exact for 99.5% of
targets — so the absence of an effect is not explained by the one cost the theory predicts. What
remains unexplained is *why* propagation did not speed up, and the most likely answer is that credit
propagation was never the binding constraint. Do not try `n=5`: the reason for preferring larger n
was that the effect scales with n, and the effect is absent at n=3 in the direction it should be
largest.

Design and full numbers in
[`runs.md`](runs.md#batch-15-is-stopped-at-55-58m-awaiting-evals--n-step-returns-at-n_step_update3);
checkpoint evals pending.

## The record across four environments: 51% → 92% → 93% → **96%**

Each environment change resets the comparison set, so the record is really four records. The
progression is still worth reading as one line, because every step came from a different cause:

| record | arm / checkpoint | environment | what moved it |
|---|---|---|---|
| 51% | `b7f-disc995seed3` @860k | pre-audit | `DISCOUNT=0.995` |
| 92% | `b8f-disc9975seed2` @2816k | pre-audit | **the horizon** — 2.8M steps instead of 1.06M |
| 93% | `b10d-disc995seed4` @1695k | 26-value (2026-08-02's seven fixes) | the environment fixes, not a config change |
| **96%** | `b11b-obs30seed2` @855k | **30-value (current)** | unattributable — see the caveat below |

**The 96% cannot be credited to the two new observations.** Batch 11 differs from batch 10 only by
those observations, and its close-out came out +4 to +5 pp on three metrics with p between 0.14 and
0.24 — consistent with a real effect and equally consistent with seed luck. A single record
checkpoint is the weakest possible evidence for a config change, being the max of 204 noisy
measurements in one arm; corrected for the winner's curse it is ~94%, against `b10d`'s ~87%.
Write-up: [`archive/batches1-11.md`](archive/batches1-11.md#batch-11--the-same-config-on-the-30-value-vector-no-significant-difference).

**Two of the four steps came from something other than hyperparameters** — the horizon, and the
environment audit. That is the most useful pattern in this table, and it is why the standing backlog
in [`runs.md`](runs.md) is ordered below the design fix rather than above it.

### The 92% of the batch-8 era, and why the horizon was the binding constraint

Final close-out measurement 2026-08-01, with the same arms' earlier measurements below for the
trajectory — which is the whole story of this section:

| arm | when | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | **close-out, 5.47M** | 52 | **92.0%** @2816k | **86.7%** | **66.3%** /5200 | 65.1-67.6 |
| `b8d-disc995clip` | close-out, 11.64M | 20 | 76.0% @5027k | 72.7% | 60.4% /2000 | 58.2-62.5 |
| `b8f` | mid-run, 2.65M | 63 | 88.0% @2581k | 82.7% | 59.2% /6300 | 57.9-60.4 |
| `b8d` | mid-run, 2.93M | 25 | **80.0%** @2538k | 74.7% | 58.4% /2500 | 56.5-60.3 |
| `b8f` | mid-run, 1.78M | 16 | 63.0% @1618k | 60.3% | 46.5% /1600 | 44.1-48.9 |
| `b8d` | mid-run, 2.08M | 10 | 62.0% @1688k | 58.7% | 48.3% /1000 | 45.2-51.4 |
| `b7f-disc995seed3` | final, 1.06M | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | final, 1.06M | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**Pooled figures are only comparable within one selector.** The close-out rows used the current rule
(all >=90%, fill to 20 from >=60%); the mid-run rows used the earlier >=80% rule. A more selective
set has a higher pooled rate by construction, so `b8f`'s 59.2% → 66.3% is partly the selector. The
**best-checkpoint column is comparable throughout**, and there the record went 51% → 88% → **92%**.

**The pooled column carries the claim.** 59.2% over 6300 episodes has a ±1.3 interval, so this is
not a best-of-N artefact: it is 20 points above the pooled figure that stood the same morning and
non-overlapping with it. `b8f` has 35 of 63 checkpoints at >=60%.

**The two configs stay tied on pooled** (overlapping intervals) with `b8f` ahead on best. The
champion checkpoint is preserved in [`../hallOfFame/`](../hallOfFame/README.md).

### The late-checkpoint hypothesis: confirmed for supply, mixed for quality

The previous version of this section flagged as speculative that "the horizon may have been
truncating the best checkpoints of good arms". Re-measurement supports it, but not uniformly, and
the distinction matters:

| | corr(step, measured) | 1.0-1.8M | 2.2-2.6M | 2.6-3.0M |
|---|---|---|---|---|
| `b8f` | **+0.61** | ~45% | **64.5%** | 63.6% |
| `b8d` | **-0.11** | 59.5% | 60.3% | 54.0% |

**What is solid is the supply of good checkpoints, not per-checkpoint quality.** In thirteen
hours `b8f` went from 16 checkpoints at >=80% to **63**, and `b8d` from 4 to 25. Both arms' best
checkpoints sit at ~2.55M, and every previous record-holder was stopped at 1.06M — before that
region existed.

**Per-checkpoint quality rises with steps for `b8f` (+0.61) and not for `b8d` (-0.11)**, whose
late band is slightly worse. So "train longer" is not a law. Note also that this correlation is
computed only over checkpoints that already cleared the 80% filter, which restricts the range and
understates any true relationship.

The counter-evidence from before still stands: `b7d` ran to 1.60M at 0.995 and peaked at 26%,
`b7a` reached 2.00M with a 19% ceiling. Long runs do not rescue a mediocre arm.

**Practical rule: do not stop a healthy arm at ~1M steps.** Both records came from territory the
old horizon forbade.

### The horizon has an upper bound too: peak ~2.5-3M, dead by ~7M

Followed to the end, both arms traced the same four-phase arc. `b8d` ran to **11.6M steps** — the
longest run in the project by more than 2x — and died:

| phase | steps | `b8f` perfect (per 1M) | `b8d` perfect (per 1M) |
|---|---|---|---|
| climb | 0-2M | 17.2% → 30.1% | 6.8% → 15.4% |
| **peak** | **~2.5-3M** | **40.9%** | **27.4%** |
| decline | 3-6M | 18.6% → 7.4% → 10.1% | 14.6% → 11.9% → 0.3% |
| death | 7M+ | — | **0.0%** for 4.5M steps |

Both arms' best measured checkpoints (2581k, 2538k) and best 30-eval windows (2828k, 2671k) fall in
the peak band. `b8d`'s last perfect game was at 5496k, 6.1M steps before it was still running.

**So the practical horizon is ~3-3.5M steps**, not 1M and not unlimited. The ~8.5M steps `b8d` spent
after its peak produced nothing measurable. That the decline ends in death rather than a plateau
also means a past-peak arm is not merely unproductive — it is on its way to zero.

#### Corollary: a sudden jump in step rate is a symptom of death

`b8d` advanced **7.3M steps in ~24 hours** while `b8f` managed 1.9M on the same machine. Almost all
of that gap is that **a dead policy plays very short episodes** — the snake dies immediately — so it
burns training steps several times faster than a competent one.

Never read step rate as progress. This is the same confound that once made eval cost look like a
config difference, and it now has a second use: an arm that suddenly starts advancing much faster
than its sibling is probably dying, not accelerating.

## Falsified: `GRADIENT_CLIPPING=10` does not buy stability

Clipping went in as a cheap independent stability aid on top of `DISCOUNT=0.995`, on the
reasoning that the 10.0 terminal reward produces occasional huge gradients and that clipping
them would prevent the catastrophic drops. After three seeds it is **1 of 3**, against **3 of
3** for plain 0.995:

| arm | peak trailing | best 30-eval pf | best measured | outcome |
|---|---|---|---|---|
| `b8d-disc995clip` | **86.9** | **50.0%** | **80.0%** (58.4% pooled) | peaked ~2.7M, declining at 3.48M |
| `b8e-clipseed2` | 85.9 | 21.3% | 32.0% (1 ckpt) | faded; stopped at 1.16M |
| `b8g-clipseed3` | 77.0 | 30.0% | **none >50%** | dead; stopped at 3.43M |

**It was briefly this file's headline, off `b8d` at 163k steps.** That reading — "the fastest
riser on record", 36.0% best-30 by 163k against `b7f`'s 699k — was wrong twice over. `b8d`'s
own early window was followed by a near-total collapse (0.4% mean perfect across 300-600k) and
everything durable came after 600k, so it was not a head start. And the two seeds that followed
did not reproduce it.

**The "raises the ceiling" escape hatch is now closed too.** `b8d` measured 62.0% best / 48.3%
pooled, which looked like a unique ceiling gain — until `b8f` measured **63.0% / 46.5% without
clipping**, with overlapping intervals. Re-measurement 13 hours later widened the gap the other
way: **`b8f` 88.0% / 59.2% against `b8d` 80.0% / 58.4%**, still tied on pooled but with the
non-clipped arm ahead on ceiling. Clipping shows **no measured benefit and a worse survival
record**. Do not adopt it.

Recording the process error, because it is the recurring one: that ceiling claim was written
while `b8d` was measured and `b8f` was not, off the arm that happened to finish first. A
two-arm comparison graded from one arm is not a comparison. Wait for both.

## An arm recovered from 1.2M steps at zero — and then died anyway

`b8g-clipseed3` sets both records at once, which is why it is worth its own section:

| block | mean trailing | mean perfect |
|---|---|---|
| 0-300k | 52.7 | 8.7% |
| 600-1800k | **1.7 - 14.7** | **0.0%** |
| **2100-2400k** | **63.7** | **4.3%** |
| 2700-3600k | **0.0** | 0.0% |

**The recovery.** 1.2M steps near zero, then back to 63.7 trailing and a 4.3% perfect rate. The
previous record was ~400k steps. Any stop rule that would have killed this arm at 1M steps —
including the one this project used for most of its life — was wrong on this case.

**The death.** It then collapsed and spent its final 900k pinned at 0.0. So the recovery bought
nothing in the end, and an arm that has completed a recovery arc can still be finished.

The rule that survives both halves: **read `zero_since` against the current step, and require
both a long pinned stretch and no recovery in progress.** `b8g` would satisfy that at 2625k
onward and would not at 1M. Two prior errors in this file — calling `b6b` permanently damaged
and calling `b7b` merely oscillating — were the two directions of getting this wrong, and
`b8g` is the case that contains both.

## Engineering facts worth not rediscovering

- **Importance-sampling weights must stay mean-normalized.** cpprb normalizes by the
  largest weight in the whole buffer, so raw batch weights average 0.087 at beta=0.4 and
  0.0027 at beta=1.0 — a silent 11x-370x cut to the learning rate that worsens as beta
  anneals. `normalize_is_weights()` fixes this; don't remove it. (Applies only when
  `IS_WEIGHTS=1`.)
- **`legacy.Adam` is not faster here** despite TF's M1/M2 warning: 0.809 ms/step vs 0.721
  ms for the modern optimizer. Ignore the warning.
- **Throughput is ~230-240 steps/s** for one run on an idle machine, and roughly holds up
  with 4 runs sharing 14 cores. That affects wall-clock only, not learning per step.
- **cpprb is ~2.4x faster than `PyUniformReplayBuffer`** with no measured learning cost.
- **The "upgrade to Gymnasium" warning is inert.** It costs a few log lines and the
  upgrade is unavailable; do not propose it.
