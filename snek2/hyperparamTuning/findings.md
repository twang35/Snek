# Findings

What this investigation has actually established, organized by topic rather than by
the order it was discovered. Companion to [`runs.md`](runs.md) (what is next),
[`completedRuns.md`](completedRuns.md) (per-arm outcomes), and
[`failureModes.md`](failureModes.md) (the degradation patterns).

Don't re-litigate anything here without new evidence. Do re-read the falsified
section before proposing an epsilon or buffer experiment.

| finding | status |
|---|---|
| **`DISCOUNT=0.995` matches the best ceiling (51%) and survives 3 of 3 seeds** | **measured**, ~2.3x expected value |
| Prefer top-3 pooled over best-of-10; the max of 10 noisy measurements is upward-biased | **established** |
| `b6b-alpha06` (alpha 0.6, `td_loss`, no IS) is the best *bet* at 0.99: 24.5% over 1000 eps | **measured**, n=1 seed, and an **underestimate** |
| `b4c` has the best *ceiling*: ~31%, but survives only 1 of 3 | **measured**, 1400 eps |
| A high single 10-episode eval predicts a good checkpoint; smoothing is anti-predictive | **established**, +0.64 vs -0.40 |
| Policy quality changes materially within 1000 training steps | **established**, 3 of 3 clusters, up to 27 points |
| `b4c`'s best checkpoint is 851000 (~40%), not 869000 | **measured** |
| The published "51% at 869000" was the high draw of three measurements; pooled 41.7% | **corrected** |
| Restoring `theSchlong`'s PER roughly triples the perfect rate | **retracted** — did not replicate, 0 of 2 |
| That config is a coin flip: 1 of 3 seeds survives, the rest die permanently | **established**, n=3 |
| `td_loss` + alpha 0.8 + no IS is effectively alpha 1.6 | **established** — it is arithmetic |
| Sharpness is a variance dial: higher ceiling *and* higher death risk | **weakened** — eff 1.2 dies 2 of 4, eff 1.6 dies 2 of 3 |
| No prioritization setting tested so far survives reliably | **established**, 7 seeds across two sharpness levels |
| `DISCOUNT=0.995` is the current best lead | **measured**, 3 of 3 survived, 38.8% pooled |
| Higher discount is monotonically better | **falsified** — 0.999 died 2 of 2 |
| `GRADIENT_CLIPPING=10` on 0.995 reaches a good level much faster | **promising**, n=1, seeding now |
| The 5s perfect-game pause slowed good arms ~40% and biased wall-clock comparisons | **fixed**, now 500ms |
| Evals looked truncated but never were: 11 of 11 complete at 10 ckpts x 100 eps | **verified** |
| There is a stability "cliff" between eff 0.8 and 1.2 | **retracted** — `b6b` crossed it and thrived |
| Reverting *either* factor alone survives the crisis | **established**, n=1 each |
| The committed config reaches ~1% at 1M steps | **established** |
| Degradation after 236-312k is systemic across configs | **established**, 5 arms |
| Epsilon reaching 0.0 causes the collapse | **falsified** |
| A larger replay buffer prevents the collapse | **not settled** |
| n-step returns help | **falsified**, n=2 and n=3 |

---

## `DISCOUNT=0.995` matches the best ceiling and removes the death risk

The strongest result in the investigation, and the first one where the mechanism was
predicted before the run rather than reconstructed after it.

At `DISCOUNT=0.99` the effective horizon is ~100 steps, while a perfect game on the 9x9
grid runs several hundred. The `PERFECT_GAME_REWARD` was therefore discounted into
near-irrelevance — the value function could barely see the thing the whole project is
optimising for. Raising the discount to 0.995 was listed as the top untested candidate for
exactly that reason.

Three seeds, all with `PRIORITY_EXPONENT=0.6 PRIORITY_SIGNAL=td_loss IS_WEIGHTS=0`, all
measured with the outlier-top10 rule so every number below is comparable:

| arm | discount | best ckpt | top-3 pooled | all-10 pooled | survived |
|---|---|---|---|---|---|
| `b7f-disc995seed3` | **0.995** | **51%** @860k | **48.0%** (42.4-53.6) | 38.8% | yes |
| `b4c-schlongper` | 0.99 | 50% @869k | 46.7% (41.1-52.3) | 37.1% | **1 of 3 seeds** |
| `b7e-disc995seed2` | **0.995** | 39% @334k | 34.7% (29.5-40.2) | 29.5% | yes |
| `b7d-discount995` | **0.995** | 26% @1330k | 22.7% (18.3-27.7) | 16.4% | yes |
| `b7a-a06seed2` | 0.99 | 19% @1822k | 18.3% (14.4-23.1) | 12.0% | yes |

### The gain is reliability, and the ceiling claim would have been wrong

`b7f` (51%) and `b4c` (50%) are a **dead heat** — overlapping intervals on best checkpoint
*and* on top-3 pooled. So 0.995 did not raise the peak; it reproduced it. An earlier draft
of this section was going to claim a new record by comparing `b7f`'s 38.8% against `b4c`'s
previously published 31.4%. **That comparison was invalid**: the two used different
selection rules, and re-measuring `b4c` under the matched rule moved it to 37.1%. The
entire apparent 7-point gain was the selector, not the config. See the selection-rule
caveat below — it caught a false headline within a day of being written down.

What did change is the death rate:

| config | mean level across seeds | survival | expected value |
|---|---|---|---|
| `DISCOUNT=0.995` | 28.2% | **3 of 3** | **28.2%** |
| `b4c` config, eff ~1.6 | 37.1% | 1 of 3 | 12.4% |
| same config at 0.99 (`b7a`) | 12.0% | 2 of 4 | 6.0% |

**~2.3x the expected value of the best previous config**, purely from reaching the same
ceiling without discarding two runs in three. Every earlier lever traded ceiling against
reliability; this is the first to escape that tradeoff.

Secondary evidence that the peak has headroom: at its best checkpoint `b7f` averages **88.8
of a maximum 95**, against `b4c`'s 84.9. Even its failures end closer to a perfect game.

**Two caveats.** Survival is established only to **~1.1M steps** — `b7e` and `b7f` were
stopped at 1.28M and 1.06M while their 0.99 siblings died at 1162k and 573k, so they have
not outlived the danger window by much. And `avg_reward` is **not comparable** across a
discount change, since the discount rescales the reward; compare perfect rates only.

## Measured: `b6b-alpha06` is the best *bet*, `b4c` still has the best *ceiling*

Every batch 5/6 arm was measured with `eval_checkpoints.py <arm> top10` — its ten best
surviving checkpoints at 100 greedy episodes each, 1000 episodes per arm. Pooled:

| arm | eff exp | pooled perfect (1000 eps) | 95% CI | best ckpt | worst ckpt | graph said |
|---|---|---|---|---|---|---|
| `b6b-alpha06` | ~1.2 | **24.5%** (245/1000) | 21.9-27.3% | 36% @1455k | 12% | 21.7% |
| `b6a-alpha04` | ~0.8 | 8.1% (81/1000) | 6.6-10.0% | 13% @514k | 2% | 14.3% |
| `b5d-schlongTDE` | ~0.8 | 6.6% (66/1000) | 5.2-8.3% | 12% @1160k | 2% | 10.7% |
| `b5c-schlongIS` | ~1.6 corr | 2.1% (21/1000) | 1.4-3.2% | 6% @2239k | 0% | 17.0% |

`b6b` is **3x the next best arm with non-overlapping intervals** — the first
non-marginal separation since `b4c`.

### But it did not beat `b4c`, and the honest comparison matters

`b4c`'s earlier measurement used **4** checkpoints, so comparing its pooled number to
`b6b`'s 10-checkpoint pool is unfair to `b6b` — a deeper pool reaches into weaker
checkpoints. Taking each arm's **top 4 by the same a priori criterion** (smoothed graph
rate, before any 100-episode measurement):

| arm | top-4 pooled | 95% CI | best single checkpoint |
|---|---|---|---|
| `b4c-schlongper` | **31.8%** (127/400) | 27.4-36.5% | **51%** @869k |
| `b6b-alpha06` | 21.8% (87/400) | 18.0-26.1% | 36% @1455k |

So `b4c`'s config really does produce better policies **when it survives**. `b6b` did not
beat it. What `b6b` beats it on is *reliability*, and once the death rate is priced in the
ranking flips:

| config | eff exp | ceiling when it survives | survival | expected value |
|---|---|---|---|---|
| `b4c` (alpha 0.8 `td_loss` no IS) | ~1.6 | 31.8% | 1 of 3 | **~10.6%** |
| `b6b` (alpha 0.6 `td_loss` no IS) | ~1.2 | 24.5% | 1 of 1 | **~24.5%** |

**`b6b`'s config is the better bet by roughly 2x, despite the lower ceiling.** Its survival
is n=1, so that number is soft and the next thing to do is seed it 2-3 more times. But this
is the first config here that looks both good *and* repeatable, and repeatability has been
the binding constraint all along.

## An outlier eval is signal, not luck — and smoothing destroys it

**Terminology:** these checkpoints were briefly called "lucky" in this document. That was
wrong and the word is gone. They are **outliers**, and measurement shows they are genuinely
better than their neighbours rather than lucky draws from the same policy.


This is the most useful measurement result in the investigation, and it **falsified the
assumption `eval_checkpoints.py` was originally built on**. The script ranked checkpoints
by perfect rate smoothed over a centred 10-eval window, on the reasoning that a single
10-episode eval reaching 70-80% must be a lucky draw. Measuring both selections against
the truth says otherwise:

| selection rule | pooled measured | 95% CI | episodes |
|---|---|---|---|
| **raw single eval** (outliers) | **41.3%** | 35.9-47.0% | 300 |
| smoothed region rate | 27.1% | 24.0-30.6% | 700 |

Non-overlapping: the checkpoints picked as outliers are **14 points better**,
not worse. Correlation against the 100-episode measurement across the ten checkpoints:

| predictor | correlation with true rate |
|---|---|
| raw single 10-episode eval | **+0.64** |
| smoothed region rate | **-0.40** |

Smoothing is not merely weaker here, it is **anti-predictive**. The binomial says why: if a
policy's true rate were 27%, the chance a 10-episode eval shows 7+ perfect games is
**0.006**. A high single eval is strong evidence about *that checkpoint*. Averaging it with
its neighbours describes the *region* instead — and as the next section shows, the region
is a poor proxy for the checkpoint.

**Consequence: `b6b-alpha06` and `b6a-alpha04` were measured with the smoothed-heavy
selector and are therefore underestimates.** Their 24.5% and 8.1% are not comparable to
anything measured since.

They also **cannot be fixed by re-measuring**. `b6a`'s best graph point in 1415 evals is 50%
and `b6b` has exactly two above 50%, so under the current thresholds `b6a` yields nothing and
`b6b` yields two checkpoints. The alpha comparison needs new seeds, not new measurements.

### Refinement: a 50% floor and an 80% must-measure line

The ranking above says *which* checkpoints to prefer. It says nothing about how far down the
list to go, and the answer turns out to matter as much:

| single eval | rule | why |
|---|---|---|
| **>=80%** | always measure, even past the 10-slot target | 8+ perfect in 10 is the strongest available signal; a slot limit is no reason to drop one |
| **60-70%** | fill remaining slots, best first | the real candidate band for most arms |
| **<=50%** | never measure | 100 episodes buys precision about a checkpoint that was never going to be the arm's best |

Because a graph point is 10 episodes, `perfect_percent` only takes values 0, 10, … 100, so
these thresholds are coarser than they read: `>=80%` is {80, 90, 100} and the fill band is
exactly {60, 70}.

**The distribution is extremely skewed, which is what makes the floor worth having.** Across
all 30 arms run so far, 22 have **never produced a single eval above 50%** in thousands of
evals. Effort concentrates on very few arms:

| arm | evals | points at >=80% | points at 60-70% |
|---|---|---|---|
| `b8f-disc9975seed2` | 1757 | **16** (3 at 90%) | 101 |
| `b8d-disc995clip` | 2065 | 4 | 50 |
| `b7f-disc995seed3` | 1058 | 1 | 34 |
| `b4c-schlongper` | 1097 | 1 | 18 |
| 22 others | — | **0** | 0-3 |

So the same 10-checkpoint budget was previously spending 10 evals on arms with no candidate
at all, and capping `b8f` at 10 when it has 16 checkpoints that each cleared 80%.

## Policy quality changes materially within 1000 training steps

Evaluating each high-single-eval checkpoint **together with the checkpoints immediately
either side of it** — 100 episodes each — settles whether "this checkpoint is good" can be
distinguished from "this part of the run is good". It can:

| cluster | centre | neighbours at +/-1000 | centre advantage |
|---|---|---|---|
| 851000 | **40.0%** | 28.5% | **+11.5 points** |
| 869000 | **32.0%** | 23.0% | **+9.0 points** |
| 970000 | **35.0%** | 7.5% | **+27.5 points** |

Pooled, centres measure 35.7% (CI 30.5-41.2) against neighbours' 19.7% (CI 16.7-23.0) —
non-overlapping, and the effect is in the same direction in **3 of 3** clusters.

The 970000 cluster is the extreme case: **969000 measures 8%, 970000 measures 35%, 971000
measures 7%.** Those are 100-episode measurements, so **1000 training steps can gain or
lose 27 points of perfect-game rate.** Training is far more non-stationary at the
checkpoint level than this investigation previously assumed, and adjacent checkpoints are
not interchangeable samples of one policy.

## Two measurement caveats that change how numbers here should be read

#### Pooled rates only compare when the selection rule matches

`b4c-schlongper` measured three ways:

| selection | pooled | note |
|---|---|---|
| 4 hand-picked | 31.8% /400 | the original measurement |
| 3 outliers + 7 smoothed | 31.4% /1000 | agrees closely |
| 3 clusters of 3 + 1 | **26.2%** /1000 | *lower by construction* |

The cluster run is not a disagreement — 6 of its 10 picks are deliberately the weaker
neighbours, so it measures the spike-vs-neighbour gap rather than the config's level.
**`b4c`'s level is ~31%.** Never compare pooled numbers produced by different selection
rules.

The 2026-07-30 thresholds make this worse, not better, and deliberately so: the checkpoint
count itself now varies per arm (16 for `b8f`, 1 for `b8e`) and the population is truncated at
50%. Pooling over 16 checkpoints and over 1 are not the same statistic. **Use best checkpoint
for cross-arm comparison from here on**, and read pooled only as a within-arm consistency
check — a config whose best and pooled figures are close is producing a strong *region*, which
is the property the project is actually chasing.

#### 100 episodes is a weaker instrument than its interval implies

Checkpoint 869000 — frozen weights, greedy policy — has been measured three separate
times:

| run | rate | 95% CI |
|---|---|---|
| 4-ckpt hand-picked | **51%** | 41.3-60.6 |
| outliers+smoothed | 42% | 32.8-51.8 |
| clusters | **32%** | 23.7-41.7 |
| **pooled** | **41.7%** | **36.2-47.3** |

A 19-point spread on identical weights, roughly 2.8 sigma at the extremes — more than
binomial noise comfortably explains, so either the Wilson interval understates the real
variance or something differs between runs that has not been identified. Either way:

- **The published 51% was the high draw of three.** Use **41.7% over 300 episodes**.
- **The best checkpoint found is 851000, not 869000** (40-44% across two measurements).
- Prefer several hundred episodes, or repeat a measurement, before treating any single
  100-episode figure as settled.

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
  is why `top10` deliberately allows adjacent picks: spacing them out hides exactly this.
- The published **51% for `b4c` at 869000 is one checkpoint**, so it is the top of a
  distribution like this one, not the config's level. `b4c`'s pooled 31.8% is the fairer
  figure.

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

> **Weakened by batch 7.** Seeding eff ~1.2 four times gave **2 deaths of 4** (`b7b` at
> 1162k, `b7c` at 573k), against eff ~1.6's 2 of 3. At these sample sizes 50% and 33% are
> not meaningfully different, so the claim that lower sharpness is *safer* no longer has
> support. What survives is the narrower claim that eff ~1.6 has the higher **ceiling**.
>
> Both eff ~1.2 deaths arrived late — 573k and 1162k — well past where the eff ~1.6 arms
> died (246k, 272k). So lower sharpness may **delay** death rather than prevent it, which
> would make measured "survival" partly an artefact of how long an arm is run. Any future
> survival rate quoted here needs a fixed step horizon attached.

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

| eff exponent | behaviour | **measured** perfect rate | death risk |
|---|---|---|---|
| ~0.8 | tame, flat, boring | 6.6-8.1% | none seen |
| ~1.2 | violent oscillation, always recovers | **24.5%** | none seen yet |
| ~1.6 | oscillation that can become absorbing | 31.8% | 2 of 3 |
| ~1.6, IS-corrected | tame despite nominal sharpness | **2.1%** | none seen |

The measured column (1000 episodes per arm, 400 for `b4c`) replaces the graph-derived
figures an earlier version used, and it changes two things. The gap between ~0.8 and ~1.2
is **3x, much larger than the graphs suggested**, and the IS-corrected arm came **last**,
not mid-table — so "IS weights are the strongest stabilizer" is true about *stability* and
false about *quality*. Stability bought by IS correction appears to cost most of the
performance.

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

| arm | reverted | final step | worst dip in 180-300k | best 30-eval pf | **measured** |
|---|---|---|---|---|---|
| `b5c-schlongIS` | IS weights back on | 2.31M | 62.1 — barely noticed it | 17.0% @211k | **2.1%** |
| `b5d-schlongTDE` | `abs(td_error)` signal | 2.07M | 22.8 — dipped and recovered | 10.7% @410k | **6.6%** |

Both survived the window that killed the exact repeats, so reverting either factor does
buy stability. **But neither policy is any good**, and `b5c` — the arm that sailed through
the crisis most smoothly — measured **worst of all four arms at 2.1%**.

So the earlier reading here, that IS weights "look like the strongest stabilizer" and that
adding them during the port was not a mistake, needs splitting in two:

- **About stability, it holds.** `b5c` never dropped below 62 in the crisis window.
- **About quality, it is wrong.** IS correction cancels the prioritization it is correcting,
  and with it most of the benefit. 2.1% is barely above the ~1% committed baseline.

Caveat on `b5c`'s number specifically: it ran 2M steps past its peak, so its 17.0% peak
checkpoint had already been evicted and only weak survivors remained to measure. Its true
ceiling is somewhere above 2.1% and unrecoverable. That is a measurement failure caused by
letting the arm run, not purely a property of the config — see
[`runs.md`](runs.md) on checkpoint retention.

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

## The discount has an optimum near 0.995, not a monotone benefit

`DISCOUNT=0.995` was such a clear win over 0.99 that the obvious next move was more of it.
**More is worse.** Both `DISCOUNT=0.999` seeds died, and neither reached a decent level
first:

| discount | eff horizon | peak trailing | outcome |
|---|---|---|---|
| 0.99 | ~100 | 88.8 | 12.0% measured, dies 2 of 4 seeds |
| **0.995** | ~200 | 92.6 | **38.8% measured, 3 of 3 survived** |
| **0.9975** | ~400 | **89.4** | **best-ever 47.7% best-30, but 1 of 2 seeds** |
| 0.999 | ~1000 | 63.1 / 31.8 | **dead 2 of 2** (at 452k and 398k) |

`b8b-disc999seed2` never produced a single perfect game across 1.41M steps. The prediction
recorded before launch was that longer horizons grow bootstrapped targets and 0.999 might
destabilise rather than help — that is what happened, and for once it was unambiguous at
n=2 with nothing to wait out.

The shape makes sense given the mechanism. At 0.99 the ~100-step horizon is shorter than a
perfect game, so the terminal bonus is invisible; at 0.999 the ~1000-step horizon exceeds
episode length, so the value function is bootstrapping over a horizon longer than the task
and the targets stop being well conditioned. 0.995 sits close to actual episode length,
which is the point.

**Practical consequence: stop sweeping the discount upward.** Anything above 0.9975 is
answered. But the optimum's *location* is now genuinely open between 0.995 and 0.9975, which
was not the case when this section was written.

### Update 2026-07-30: 0.9975 produced the best arm on record

`b8f-disc9975seed2` beats every 0.995 arm on every graph statistic:

| arm | discount | peak trailing | best 30-eval pf | ckpts at >=80% | max single eval |
|---|---|---|---|---|---|
| **`b8f-disc9975seed2`** | **0.9975** | **89.4** | **47.7%** | **16** | **90%** |
| `b7f-disc995seed3` | 0.995 | 92.6 | 44.0% | 1 | 80% |
| `b8d-disc995clip` | 0.995 + clip | 86.9 | 38.3% | 4 | 80% |
| `b8c-disc9975` | 0.9975 | 79.8 | 14.7% | 0 | 40% |

The **16-vs-1 gap in checkpoints above 80%** is the more important column than the best-30
window. It says `b8f` sustains a strong region rather than spiking through one, which is the
property "consistent perfect rate" actually names.

Two things keep this from being a conclusion. `b8f` is **unmeasured** — no 100-episode
evaluation yet — so 47.7% is a graph window, and graph windows have misranked arms badly before
(`b5c` was 2nd of its batch on best-30 and last on measurement). And 0.9975 is **1 of 2**:
`b8c` ran the identical config and declined monotonically to a stop. So 0.9975 has the better
ceiling and the worse survival record, on one seed each way.

**Next step is seeds 3 and 4 at 0.9975**, not more 0.995. 0.995 is already at 3 of 3 and more
seeds would only re-confirm it, while 0.9975 could be either the new optimum or a coin flip and
two seeds decide which.

## The record is ~62%, and two different configs reach it

Measured 2026-07-30, both arms mid-run:

| arm | config | ckpts | best ckpt | top-3 | pooled | 95% CI |
|---|---|---|---|---|---|---|
| `b8f-disc9975seed2` | disc **0.9975** | 16 | **63.0%** @1618k | **60.3%** | 46.5% /1600 | 44.1-48.9 |
| `b8d-disc995clip` | disc 0.995 + clip | 10 | **62.0%** @1688k | 58.7% | **48.3%** /1000 | 45.2-51.4 |
| `b7f-disc995seed3` | disc 0.995 | 10 | 51% @860k | 48.0% | 38.8% /1000 | — |
| `b4c-schlongper` | disc 0.99 | 10 | 50% @869k | 46.7% | 37.1% /1000 | — |

**The pooled column carries the claim.** A best-of-N is the maximum of a noisy statistic with a
±9-point interval, and one frozen checkpoint here has read 51/42/32 across three measurements.
Pooled over 1000-1600 episodes the interval is ±3, and both new arms clear `b7f` by ~10 points
without overlapping it.

**The two configs are indistinguishable from each other** — 63.0 vs 62.0, overlapping pooled
intervals. This is a tie, not a ranking, and neither one beats the other on this evidence.

### Both records come from checkpoints past 1.6M steps

A pattern worth stating as a hypothesis, because it would change how arms are run:

| arm | best ckpt step | arm stopped at |
|---|---|---|
| `b8f` | 1618k | still running (2.12M) |
| `b8d` | 1688k | still running (2.34M) |
| `b7f` | 860k | **1.06M** |
| `b4c` | 869k | **1.06M** |

**Both previous record-holders were stopped before reaching the range where the new records
live.** No healthy `DISCOUNT>=0.995` arm had ever been allowed past ~1.3M steps.

The counter-evidence is real, though: `b7d` ran to 1.60M at 0.995 and its best checkpoint was
26% at 1330k, so late steps are not sufficient on their own. And `b7a` reached 2.00M with a 19%
ceiling. The defensible version is narrow — **the horizon may have been truncating the best
checkpoints of good arms**, which is cheap to test by simply not stopping healthy arms at ~1M.

## Filter, not ranker: the graph eval does not order checkpoints within the high band

The 26 checkpoints measured on 2026-07-30 are the first large sample of high-graph-eval
checkpoints measured under one rule, and they say the graph value **stops carrying information
once it is high**:

| graph point | n | mean measured |
|---|---|---|
| 90% | 3 | **34.7%** |
| 80% | 17 | 50.8% |
| 70% | 6 | 43.2% |

| correlation with measured rate | value |
|---|---|
| graph single eval, both arms pooled (n=26) | **-0.09** |
| graph single eval, `b8d` alone | +0.66 |
| graph single eval, `b8f` alone | **-0.57** |
| surrounding rate, both arms pooled | -0.03 |
| surrounding rate, `b8d` / `b8f` | -0.69 / +0.50 |

`b8f`'s three 90% points measured 39%, 21% and 44% — **the worst three of its sixteen**. The
sign of every correlation flips between the two arms, which is what no-signal looks like.

**This does not overturn the +0.64 finding above, because of range restriction.** That
correlation was measured across a wider spread of graph values; here every checkpoint is 70-90%,
and truncating a predictor's range attenuates its correlation mechanically. The two results are
compatible and the combined reading is:

- **As a filter the graph eval works well.** All 26 measured 21-63%, far above what a randomly
  chosen checkpoint of these arms would give. The >=80%/<=50% thresholds are doing their job.
- **As a ranker inside the high band it is useless.** Do not treat the top of the selected list
  as the best checkpoint, and do not skip the rest of the tier to save time — **measuring all of
  the >=80% checkpoints is exactly the right policy**, because there is no way to tell in advance
  which of them is the 63% and which is the 21%.

That is a stronger argument for the "measure every checkpoint at >=80%" rule than the one it was
introduced with.

## Falsified: `GRADIENT_CLIPPING=10` does not buy stability

Clipping went in as a cheap independent stability aid on top of `DISCOUNT=0.995`, on the
reasoning that the 10.0 terminal reward produces occasional huge gradients and that clipping
them would prevent the catastrophic drops. After three seeds it is **1 of 3**, against **3 of
3** for plain 0.995:

| arm | peak trailing | best 30-eval pf | best measured | outcome |
|---|---|---|---|---|
| `b8d-disc995clip` | **86.9** | **38.3%** | **62.0%** (48.3% pooled) | thriving at 2.34M |
| `b8e-clipseed2` | 85.9 | 21.3% | 32.0% (1 ckpt) | faded; stopped at 1.16M |
| `b8g-clipseed3` | 77.0 | 30.0% | **none >50%** | dead; stopped at 3.43M |

**It was briefly this file's headline, off `b8d` at 163k steps.** That reading — "the fastest
riser on record", 36.0% best-30 by 163k against `b7f`'s 699k — was wrong twice over. `b8d`'s
own early window was followed by a near-total collapse (0.4% mean perfect across 300-600k) and
everything durable came after 600k, so it was not a head start. And the two seeds that followed
did not reproduce it.

**The "raises the ceiling" escape hatch is now closed too.** `b8d` measured 62.0% best / 48.3%
pooled, which looked like a unique ceiling gain — until `b8f` measured **63.0% / 46.5% without
clipping**, with overlapping intervals. Clipping therefore shows **no measured ceiling benefit
and a worse survival record**. Do not adopt it.

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

## The perfect-game celebration was throttling the best arms

`Snake.render()` marks a perfect game with a **blocking** `pygame.time.wait()`, and
`PERFECT_GAME_WAIT_MS` defaulted to **5000**. Every training eval runs its first episode on
the *displayed* environment, so any eval whose first episode was a win stalled for 5
seconds — against roughly 5 seconds of actual training per 1000-step eval interval.

The cost scaled with how good the arm was:

| arm quality | share of evals stalling | wasted per eval | penalty |
|---|---|---|---|
| ~40% perfect (`b7f`) | ~40% | ~2.0s | **~40% slower** |
| ~10% perfect | ~10% | ~0.5s | ~10% slower |
| dead (0% perfect) | 0% | 0s | none |

So the mechanism **penalised exactly the arms worth running** and rewarded nothing. Now
`SNEK_PERFECT_WAIT_MS`, default 500ms, recorded in each run's `run_config`.

**This partly explains the step-count gap previously attributed entirely to episode
length.** Dead arms reaching 1.7-2M steps while good arms reached 1.0-1.3M in the same wall
clock was read as "a dead policy ends episodes instantly". That is still the main effect,
but a slice of it was the winner's 5-second pause. **Step-based comparisons are unaffected**
— every arm's eval series is indexed by step, not time — but any wall-clock or steps/second
comparison across arms of different quality was biased, and runs before this fix are not
comparable on wall clock to runs after it.

`eval_checkpoints.py` had the same problem, worse: it stalls the *whole round*, because
`parallel_env.step()` does not return until every worker has stepped, so one visible win
froze all 10 workers. Measured on a 45%-perfect checkpoint, 20 episodes took **92.0s at
5000ms against 79.7s at 400ms** — 13% on a small run, more at full scale. It now defaults to
`EVAL_PERFECT_WAIT_MS=400`.

### The visible eval window looks broken and is not

Two behaviours that look like a crashed eval, both cosmetic — **no eval has ever been
truncated.** All 11 completed eval files hold exactly 10 checkpoints x 100 episodes and
every log reaches its final `wrote` line, with the only exceptions being
`OSError: Bad file descriptor` from multiprocessing connection cleanup *after* the results
are written.

- **The window stops mid-game and vanishes.** A round ends when every worker has finished
  one episode. `ParallelPyEnvironment` steps all envs together, so a worker that finishes
  early keeps being stepped and auto-resets into fresh episodes that are **not counted**.
  The visible worker is therefore usually part-way through a throwaway game when the round
  ends, and the process exits after the last checkpoint, closing the window wherever it had
  got to.
- **It used to freeze for seconds at a time.** The 5000ms blocking wait above, during which
  no `pygame.event.pump()` runs, so macOS marks the window unresponsive.

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
