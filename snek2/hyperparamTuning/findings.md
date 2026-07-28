# Findings

What this investigation has actually established, organized by topic rather than by
the order it was discovered. Companion to [`runs.md`](runs.md) (what is next),
[`completedRuns.md`](completedRuns.md) (per-arm outcomes), and
[`failureModes.md`](failureModes.md) (the degradation patterns).

Don't re-litigate anything here without new evidence. Do re-read the falsified
section before proposing an epsilon or buffer experiment.

| finding | status |
|---|---|
| Best measured policy: 51% perfect games over 100 episodes | **measured** |
| Restoring `theSchlong`'s PER roughly triples the perfect rate | **retracted** — did not replicate, 0 of 2 |
| That config is a coin flip: 1 of 3 seeds survives, the rest die permanently | **established**, n=3 |
| `td_loss` + alpha 0.8 + no IS is effectively alpha 1.6 | **established** — it is arithmetic |
| Sharpness is a variance dial: higher ceiling *and* higher death risk | **supported**, 7 arms, n=1 per cell |
| There is a stability "cliff" between eff 0.8 and 1.2 | **retracted** — `b6b` crossed it and thrived |
| Reverting *either* factor alone survives the crisis | **established**, n=1 each |
| The committed config reaches ~1% at 1M steps | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Epsilon reaching 0.0 causes the collapse | **falsified** |
| A larger replay buffer prevents the collapse | **not settled** |
| n-step returns help | **falsified**, n=2 and n=3 |

---

## Retracted: the `theSchlong` PER config is not reliably better — it is a coin flip

`b4c-schlongper` (alpha 0.8, `td_loss`, no IS) produced the best result on record and
this document previously called that config established. **Batch 5 ran it twice more and
both repeats died permanently.**

| arm | config | outcome | died at | best 30-eval perfect |
|---|---|---|---|---|
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | **survived**, 51% measured | — (dipped to 10.1 @ 203k) | 34.0% |
| `b5a-schlong` | identical | **dead** | 272k | 10.0% |
| `b5b-schlong2` | identical | **dead** | 246k | 7.7% |

All three hit a crisis in the same 200-270k window. `b4c` bottomed at trailing 10.1 and
climbed back to 62+ by 320k; the other two flatlined at 0.0 and stayed there for 1.6M
and 1.7M further steps. So the config does not produce a better policy — it produces a
**~1-in-3 lottery ticket**, and the one time it paid out it paid out very well.

This is the third premature conclusion in this investigation, and the pattern is now
unmistakable: **every result here that rested on a single seed has failed to replicate.**

## The mechanism: `td_loss` doubles the effective priority exponent

`common.element_wise_huber_loss` uses delta 1.0, so for `|td_error| < 1` — which is
most transitions once a policy is decent — `td_loss = 0.5 * td_error^2`. Priorities are
then raised to alpha. Squaring inside and exponentiating outside **compounds**:

| priority signal | alpha | effective exponent on `\|td_error\|` | arms | outcome |
|---|---|---|---|---|
| `td_loss` | 0.8 | **~1.6** | `b4c`, `b5a`, `b5b` | 1 of 3 survived |
| `td_error` | 0.8 | ~0.8 | `b5d-schlongTDE` | alive, recovered from a 243k dip |
| `td_loss` | 0.8, IS on | ~1.6, corrected | `b5c-schlongIS` | alive, barely dipped |

The three "PER changes" recovered from `theSchlong` were never independent. `td_loss` and
alpha 0.8 multiply into an extreme exponent, and dropping IS weights removes the only
thing correcting the resulting bias. `b4c` was not running aggressive prioritization —
it was running prioritization roughly twice as sharp as any alpha value anyone intended
to test, uncorrected.

That reframes the whole line of investigation: **alpha 0.8 was never the config under
test.** The nominal value and the effective value differ by 2x whenever
`PRIORITY_SIGNAL=td_loss`, which makes every `td_loss` arm incomparable to its alpha
label. Treat the effective-exponent column above as the real independent variable.

### Sharpness is a variance dial, not a quality dial

The prediction made before batch 6 ran was that alpha 0.4 with `td_loss` (~0.8 effective)
survives and alpha 0.6 (~1.2) is marginal. The first half held. **The second half was
wrong, and the correction is more interesting than the prediction.**

Ranked by best 30-eval perfect rate, with the near-death-and-recovered count:

| arm | eff exp | best 30-eval pf | peak trailing | near-death recoveries | fate |
|---|---|---|---|---|---|
| `b4c-schlongper` | ~1.6 | **34.0%** @880k | 86.4 | 2 | survived, 51% measured |
| `b6b-alpha06` | ~1.2 | **21.7%** @1467k | 81.3 | 3 | running, still oscillating |
| `b5c-schlongIS` | ~1.6 corr | 17.0% @211k | 80.7 | 1 | running, declining 2M steps |
| `b6a-alpha04` | ~0.8 | 14.3% @372k | 82.8 | 1 | running, flat and stable |
| `b5d-schlongTDE` | ~0.8 | 10.7% @410k | 76.7 | 1 | running, stable |
| `b5a-schlong` | ~1.6 | 10.0% @84k | 76.5 | 1 | **died** 272k |
| `b5b-schlong2` | ~1.6 | 7.7% @129k | 74.7 | 1 | **died** 246k |

Among arms that survived, the **ceiling rises monotonically with effective exponent**:
~1.6 gives 34.0%, ~1.2 gives 21.7%, ~0.8 gives 14.3% and 10.7%. The **risk of permanent
death rises with it too** — 2 of the 3 arms at ~1.6 died outright. And the top two arms
are precisely the two with the most near-death excursions.

So prioritization sharpness buys **variance**, and variance buys both the high ceiling and
the absorbing failure. It is a risk/return dial:

| eff exponent | behaviour | ceiling | death risk |
|---|---|---|---|
| ~0.8 | tame, flat, boring | ~10-14% | none seen |
| ~1.2 | violent oscillation, always recovers | ~22% | none seen yet |
| ~1.6 | oscillation that can become absorbing | ~34% | 2 of 3 |
| ~1.6, IS-corrected | tame despite nominal sharpness | ~17% | none seen |

**Retracted from an earlier version of this section: the claim that there is a "cliff
between ~0.8 and ~1.2."** There is no cliff. `b6b` at ~1.2 crossed the supposed cliff and
became the second-best arm on record. What actually separates ~1.2 from ~1.6 is whether
the oscillation's low excursions are absorbing, and that looks like a dice roll rather
than a threshold.

### Retracted: `b6b`'s crash was not permanent capability loss

An earlier version of this section said `b6b-alpha06` suffered a "crash with permanent
capability loss" that "never regained a quarter of its 79.3 peak." **Both claims were
wrong.** It has gone to near-zero and fully recovered *twice*:

| block | mean trailing | min | mean perfect |
|---|---|---|---|
| 0-200k | 36.5 | **0.0** | 2.6% |
| 200-400k | 14.5 | 4.4 | 0.0% |
| 400-600k | 24.7 | 8.5 | 1.9% |
| 600-800k | 60.3 | 39.4 | 6.2% |
| 800-1000k | 71.3 | 61.6 | 10.4% |
| 1000-1200k | 66.4 | 47.0 | 3.9% |
| 1200-1400k | 23.7 | **0.9** | 2.8% |
| 1400-1600k | 61.8 | 39.8 | 10.5% |
| 1600-1800k | 61.3 | 32.1 | **13.3%** |

It is a **very long-period oscillator** whose perfect-game trend is *rising* across the
oscillations. Judging it required more than a million steps of patience, and every read
before ~600k would have been wrong.

Two rules come out of this. First, the death criterion is not "reached 0.0" — `b6b` hit
0.0 in its first block and went on to 21.7%. It is **stayed pinned at 0.0 for hundreds of
thousands of steps**, which is what `b5a`/`b5b` did for 1.7M+. Second, an oscillator's
period can exceed 1M steps, so a several-hundred-thousand-step flat stretch is not
evidence of a settled level. This is the fourth premature conclusion in this
investigation and the third to involve reading a trough as an ending.

## Reverting either factor alone survives the crisis

Both single-factor variants are alive past the step where both exact repeats died:

| arm | reverted | step | trailing now | worst dip in 180-300k | best 30-eval perfect |
|---|---|---|---|---|---|
| `b5c-schlongIS` | IS weights back on | 544k | 72.9 | 62.1 — barely noticed it | **17.0%** @211k |
| `b5d-schlongTDE` | `abs(td_error)` signal | 510k | 73.0 | 22.8 — dipped and recovered | 10.7% @410k |

`b5c`'s 17.0% is the second-best 30-eval window on record, behind only `b4c`'s 34.0%.
Neither is finished, so neither number is final — `b3c-buf500k` looked like the best arm
in the batch and then died at 750k.

Note the ordering: **IS weights, which the original port added and which `theSchlong`
lacked, look like the strongest stabilizer here.** `b5c` sailed through the window that
killed two arms without dropping below 62. That inverts the earlier reading that IS
weights were a mistake to have added.

---

## The headline: 51% perfect games, measured

The four best `b4c-schlongper` checkpoints were reloaded and evaluated over **100
greedy episodes each** with [`eval_checkpoints.py`](../eval_checkpoints.py). Full
results in `runs/b4c-schlongper_checkpoint_evals.json`.

| checkpoint | perfect % over 100 eps | 95% CI | mean score | median score |
|---|---|---|---|---|
| **869000** | **51.0%** | **41.3-60.6%** | 83.4 | **95.0** |
| 942000 | 29.0% | 21.0-38.5% | 74.3 | 84.0 |
| 775000 | 25.0% | 17.5-34.3% | 75.4 | 76.5 |
| 162000 | 22.0% | 15.0-31.1% | 51.9 | 67.5 |

**Checkpoint 869000 wins more than half the games it plays.** Its median score is 95 of
a possible 95 — over half its episodes are perfect wins — and the mean of 83.4 is
dragged down by a minority of early deaths (min 13).

These numbers were produced by an earlier version of `eval_checkpoints.py` that stopped
at the Nth finished episode, which **truncates in-flight episodes and so biases the rate
downward** (perfect games are the longest episodes). The harness now runs whole rounds
instead. The direction of the bias means 51% is a floor rather than an overstatement, but
these four are worth re-measuring when the machine is free.

This **vindicates the premise the investigation started from.** The brief said the
config should reach ~50% perfect games around 1M iterations. The committed config
reaches ~1%; with `theSchlong`'s PER restored, 51% arrives at 869k steps. The premise
was right, and three PER changes were the regression.

The methodological lesson from this measurement — that selecting checkpoints off the
graph overestimates, by up to 24 points here — is in
[`hyperparamTuning.md`](hyperparamTuning.md), since it is about how to measure rather
than about the policy.

## What `theSchlong` did differently: three PER changes, none validated

`theSchlong` (the 2022 code in the repo root, read-only) once reached a **76%
perfect-game rate**. Diffing it against `snek2` was the most productive single action in
this investigation — a working reference beats a guess.

**What the 76% was**, since it was got wrong twice: per the human, **a single checkpoint
that got very lucky**, not a mean over a run and not a sustained level. So it was never
a known-good target to reproduce; it was evidence that the config had a high ceiling and
a wide spread. That prediction is exactly what `b4c-schlongper` then showed.

**Rewards, grid size, network shape and the epsilon ladder are byte-identical**, so
scores compare directly across the two. `Snake.py` and `snake_environment.py` differ only
cosmetically (a policy-name overlay, a moved print, `set_display`/`get_score` helpers).
What changed is the replay buffer, and it changed **three ways at once**:

| aspect | `theSchlong` | `snek2` default | why it was changed |
|---|---|---|---|
| alpha | **0.8** | 0.6 | "0.6 against raw TD error is the usual choice" |
| priority signal | **`td_loss`** (element-wise Huber) | `abs(td_error)` | Huber shrinks small errors, widening spread → effective exponent ~1.6 |
| IS weights | **none at all** | mean-normalized, beta 0.4→1.0 | prioritizing without IS correction is biased |

Every one is defensible in isolation, and **every one was validated only at 30k steps** —
far below the ~250k line where anything becomes judgeable. The measurement that justified
them is therefore worthless for this question.

There is also a mechanical interaction: **IS weights partially cancel prioritization**,
because high-priority samples get downweighted gradients. So `alpha=0.6 + IS` is a much
gentler intervention than `alpha=0.8 + no IS`. All three changes push the same direction —
weaker effective prioritization.

`SNEK_PRIORITY_SIGNAL` and `SNEK_IS_WEIGHTS` now make the old behaviour reachable. Both
default to current behaviour.

## Restoring that PER config roughly triples the sustained perfect rate

`b4c-schlongper` reverted all three changes together and beat every other arm on every
measure:

| measure | `b4c-schlongper` | best of everything else |
|---|---|---|
| best 30-eval window | **34.0%** (851-880k) | 16.7% (`b1a-base`) |
| best single eval | **80%** | 40% |
| cumulative over run | **11.06%** | 5.89% (`b1a-base`) |
| peak avg score | **92.0** of 95 | 87.5 (`b1a-base`) |
| evals at >=50% perfect | **41** | 0 |

It held 50-60% repeatedly across 700k-1000k, where no other arm has produced a single
eval above 40%. Its block trajectory shows both the win and the cost:

| steps | score mean | perfect mean |
|---|---|---|
| 100-150k | 74.2 | 12.8 |
| 150-300k | 40.3 → 19.3 | **severe collapse**, score to ~19 |
| 350-400k | 70.4 | 11.0 |
| 700-750k | 75.5 | 13.2 |
| 850-900k | **79.0** | **32.2** |
| 950-1000k | 74.2 | 23.2 |
| 1000-1050k | 65.7 | 11.2 |

**Higher-performing and higher-variance.** It survived a near-total collapse around 250k,
recovered, then climbed for 600k steps to a level nothing else approached.

Three caveats that matter:

- **n=1.** This domain has produced 62.5-vs-18.0 from one config. Repeating this 2-3x is
  batch 5's first priority.
- **It would have been killed at 300k.** Its collapse was deep enough that the
  investigation's own stop criteria would have ended the run before its best 600k steps.
  That is a standing argument for longer horizons.
- **Three changes were reverted together**, so which one carries the gain is unknown. The
  knobs to separate them exist; that is batch 5 priorities 2 and 3.

## Uniform sampling helps a little; the relationship is not monotonic

Removing prioritization entirely (`PRIORITY_EXPONENT=0.0`) beat the committed
`alpha=0.6 + IS` default but landed at about a third of `b4c`'s rate:

| config | best perfect-30 |
|---|---|
| alpha 0.8 + `td_loss` + no IS (`b4c`) | **34.0%** |
| alpha 0 + 500k buffer (`b4b`) | 9.3% |
| alpha 0 (`b4a`) | 8.7% |
| alpha 0.6 + `abs(td_error)` + IS (default, `b1a`/`b2a`) | 16.7% / 7.0% |

So "how much prioritization" is not a dial that improves in one direction. Maximum
aggressive prioritization is best, none is middling, and the committed halfway setting is
not reliably better than none. The most likely explanation is the IS interaction above:
the default's IS weights cancel much of what its alpha asks for, giving the worst of both.

Prior evidence, now in context: PER at alpha=0.6 measured *worse* than uniform over 3
seeds at 30k steps (last-5 avg 46.7 sd 10.6, vs **60.1 sd 4.0** for alpha=0), and
alpha=0.8-with-Huber measured worse still. The 30k horizon makes all of that unreliable,
and the long-horizon result reverses the alpha=0.8 part of it.

## The committed config reaches ~1% at 1M steps, not ~50%

`b2a-base2` ran to 999k on committed defaults — the horizon where ~50% was expected:

| steps | score mean | perfect mean |
|---|---|---|
| 150-200k | 69.6 | **5.2** (its best) |
| 500-550k | 64.9 | 3.8 |
| 750-800k | 66.6 | 3.0 |
| 950-1000k | 64.3 | **1.1** |

Best window over the whole run: 7.0%. This is the measurement that motivated diffing
against `theSchlong`, and it is now explained — the PER changes.

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

`MIN_EPSILON` stays in the code — it defaults to 0.0 and changes nothing unless set, and
knowing epsilon 0.0 is *safe* is a useful result. Do not add a knob for the last-rung
threshold.

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

## Falsified: n-step returns

| policy | steps | peak score (at) | best perfect-30 | 1st perfect |
|---|---|---|---|---|
| `b1c-nstep3` | 1.14M | 76.0 (255k) | 1.7% | 206k |
| `b2b-nstep2` | 580k | 74.6 (140k) | 0.7% | 121k |

Both peak *below* every baseline, both then decline for hundreds of thousands of steps,
and both sit at zero perfect games in their trailing windows. Two arms ordered by n giving
the same shape is a trend, not noise.

This overturned an earlier read that n=3 had "the best trajectory of the batch" — true
through 200k, false afterwards. Do not plan an n=5 arm.

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
