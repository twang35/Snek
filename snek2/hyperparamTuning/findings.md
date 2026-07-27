# Findings

What this investigation has actually established, organized by topic rather than by
the order it was discovered. Companion to [`runs.md`](runs.md) (what is next),
[`completedRuns.md`](completedRuns.md) (per-arm outcomes), and
[`failureModes.md`](failureModes.md) (the degradation patterns).

Don't re-litigate anything here without new evidence. Do re-read the falsified
section before proposing an epsilon or buffer experiment.

| finding | status |
|---|---|
| Restoring `theSchlong`'s PER roughly triples the perfect rate | **established**, n=1 |
| Best measured policy: 51% perfect games over 100 episodes | **measured** |
| The committed config reaches ~1% at 1M steps | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Epsilon reaching 0.0 causes the collapse | **falsified** |
| A larger replay buffer prevents the collapse | **not settled** |
| n-step returns help | **falsified**, n=2 and n=3 |

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
