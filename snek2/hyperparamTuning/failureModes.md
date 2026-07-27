# Failure modes

Four distinct ways a policy in this project loses what it learned. They look similar in
a single trailing window and are easy to confuse, but they have different causes,
different timescales, and different implications for whether an arm is worth continuing.

Companion to [`findings.md`](findings.md) (what is established) and
[`charts.md`](charts.md) (the graphs each pattern is drawn from).

| mode | example | shape | recovers? | how to spot it |
|---|---|---|---|---|
| **1. Sharp collapse** | `b1a-base` at 265k | plateau, then a hard break | score yes, skill no | a visible step down in the 50k-block table |
| **2. Monotonic decline** | `b1c-nstep3` from 255k | peak then steady slide | no | every block lower than the last, for 500k+ steps |
| **3. Long oscillation** | `b2a-base2` | broad troughs ~100k steps wide | yes, repeatedly | opposite verdicts from snapshots 150k apart |
| **4. Total death** | `b3c-buf500k` at 750k | falls to score ~0 and stays | never | score near zero *and* step count accelerating |

**Max drawdown cannot tell these apart.** It collapses all four into one number, and it
scores mode 3 (harmless oscillation at a high level) the same as mode 1 (a genuine
collapse). It also barely registers mode 2 — `b1b-tgt200` scored *worse* on drawdown than
the baseline purely for oscillating at a high level. Use it as a diagnostic for "this arm
is erratic", never as a verdict.

**A better metric, not yet implemented:** largest *sustained* drop — a decline from a
running peak that persists over several consecutive evals rather than a single bad eval.
That would separate modes 1, 2 and 4 from mode 3, which is the distinction that actually
matters. Worth building before leaning on drawdown again.

---

## 1. Sharp collapse — recoverable in score, not in skill

`b1a-base` held a broad plateau from 30k to ~260k, peaked at **14% mean perfect rate with
40% spikes**, then broke hard:

| steps | score mean | score min | perfect mean |
|---|---|---|---|
| 200-250k | 71.9 | 55.9 | **14.0** |
| 250-300k | 45.8 | 10.7 | 8.2 |
| 300-350k | 37.1 | 20.4 | 5.0 |
| 350-400k | 63.6 | 31.5 | 6.2 |
| 400-450k | **65.1** | 54.8 | **2.9** |

The important part is the asymmetry of the recovery. Score climbed back from 37 to ~65
with no intervention; **the perfect rate kept falling** and ended at a fifth of its
pre-collapse level. The run relearned how to play competently and did *not* relearn how
to finish.

Two consequences:

- **Score stops proxying the objective after a collapse.** Score says "recovered" while
  the objective says "much worse than before". Judge post-collapse arms on the perfect
  rate directly.
- **Riding out a collapse is not a strategy.** The 14% rate never returned across 150k
  further steps. Preventing a collapse is worth more than waiting one out.

The collapse is also **stochastic, not a property of the config**: `b2a-base2` is the
identical config and never collapsed at all. Two runs of one config can diverge
completely after 250k, which means the noise in this domain extends to *whether a run
collapses*, not just to how well it scores.

## 2. Monotonic decline — the arm simply runs out

Both n-step arms peaked and then slid for the rest of their lives. `b1c-nstep3`:

| steps | score mean | perfect mean |
|---|---|---|
| 200-250k | 62.8 | 0.8 |
| 250-300k | 56.6 | 0.8 |
| 450-500k | 39.9 | 0.0 |
| 850-900k | 29.7 | 0.0 |

No break anywhere — every block is a little lower than the one before, for 850k steps.
This is the pattern most likely to be mistaken for "slow learner": through 200k this arm
looked like the batch's most promising, because it was still rising while others had
flattened.

**Distinguishing it from a slow riser needs a longer horizon than feels reasonable.** The
only reliable signal is the sign of the block-to-block change sustained over 300k+ steps.

## 3. Long oscillation — the trap for progress checks

`b2a-base2` never collapsed and never declined monotonically. It oscillated with a
wavelength longer than most runs are:

| steps | score mean | perfect mean |
|---|---|---|
| 150-200k | 69.6 | **5.2** |
| 450-500k | 68.7 | 3.6 |
| 550-600k | 62.8 | **1.0** ← trough |
| 650-700k | 64.8 | 1.4 |
| 750-800k | 66.6 | **2.9** ← recovering |
| 950-1000k | 64.3 | 1.1 |

A trough spans ~100k steps. **This produced two wrong calls in a row in this
investigation**: at 680k the arm was written up as slowly decaying, at 763k as
recovering, and by 999k as a shallow downward drift with big slow swings.

The rule that came out of it: **never call a trend from the most recent window.**
Trailing-window trends over 20-30 evals cannot see a 100k-step wavelength — only the
50k-block table can. Peak-to-trough on the perfect rate here is 5.2 → 1.0, so the swing
is large enough to look like a verdict while being nothing of the kind.

## 4. Total death — irrecoverable, and it looks fast

`b3c-buf500k` was praised as the flattest curve in the investigation at 500k. Then:

| steps | score mean | perfect mean |
|---|---|---|
| 250-300k | 70.7 | 4.2 |
| 650-700k | 58.9 | 0.4 |
| 700-750k | 34.3 | 0.0 |
| 750-800k | **0.6** | 0.0 |
| 800k-4.81M | ~0.0-3.2 | **0.0** |

Score 0 means the snake dies almost immediately. This is total policy destruction, and it
never recovered across **4 million** further steps — by far the most severe mode, and the
only one with no observed recovery.

**A dead policy is fast, and that is a trap.** Episodes end instantly, so evals become
nearly free and throughput explodes: this arm raced from 523k to 4.81M steps in a few
hours while its batch mates did ~1.2M. **Rapid step accumulation is a symptom of failure
here, not progress** — the eval-cost confound running in reverse (see
[`hyperparamTuning.md`](hyperparamTuning.md)). If an arm's step count is far ahead of its
batch mates, check its score before assuming it is doing well.

---

## Degradation is systemic, not config-specific

Every arm through batch 3 peaked between 236k and 312k and then degraded by one of the
modes above:

| policy | peak score (at) | best perfect-30 | where it ended |
|---|---|---|---|
| `b1a-base` | 87.5 (135k) | 16.7% | collapsed 265k, 2.3% at 503k |
| `b3a-epsfloor` | 83.5 (236k) | 11.0% | 61.4 / 1.3% at 545k |
| `b3b-epsfloor2` | 85.8 (305k) | 8.3% | 52.0 / 3.3% at 549k |
| `b3c-buf500k` | 85.7 (312k) | 5.7% | dead at 750k |
| `b2a-base2` | 83.8 (293k) | 7.0% | 64.3 / 1.1% at 999k |

Five configs, three epsilon regimes, two buffer sizes, same shape every time. That is
what made the sampling machinery — rather than any individual scalar knob — the thing
worth attacking, and it is why `b4c-schlongper` peaking at **875k** instead is the first
real break from the pattern.

`b4c` is not exempt from these modes: it suffered a mode-1 collapse at 150-300k and
recovered fully, which no other arm managed. Whether that resilience is a property of its
config or of its seed is unknown at n=1.
