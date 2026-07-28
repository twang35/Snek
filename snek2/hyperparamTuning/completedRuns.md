# Completed runs

Every arm that has finished, with its config, final numbers and verdict. Companion to
[`runs.md`](runs.md) (what is running now), [`findings.md`](findings.md) (what was
learned), and [`charts.md`](charts.md) (the graphs).

Nothing here should be re-run without a reason. Configs that are closed are marked as
such; the resume commands are in [`runs.md`](runs.md).

## All arms, ranked by best sustained perfect rate

"Best perfect-30" is the highest 30-eval trailing average the arm reached. "Cumulative"
is the mean perfect rate over its whole run. Both come from
`runs/<policy>_evals.json`; see [`hyperparamTuning.md`](hyperparamTuning.md) for why a
single eval is not usable as a measurement.

**Sort by the measured column where it exists.** It is a pooled 100-episode-per-checkpoint
measurement; every other column is derived from 10-episode graph evals and systematically
misranks arms (`b5c` is 2nd by graph, last by measurement).

| policy | config change | final steps | **measured** | best perfect-30 | cumulative | verdict |
|---|---|---|---|---|---|---|
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | 1.06M | **31.8%** /400 | **34.0%** | **11.06%** | best ceiling, but **1 of 3 seeds survive** |
| `b6b-alpha06` | alpha 0.6, `td_loss`, no IS | 1.80M | **24.5%** /1000 | 21.7% | — | **best bet.** Survived; ~2x `b4c` expected value |
| `b6a-alpha04` | alpha 0.4, `td_loss`, no IS | 1.41M | 8.1% /1000 | 14.3% | — | stable, never near death, low ceiling |
| `b5d-schlongTDE` | alpha 0.8, `td_error`, no IS | 2.07M | 6.6% /1000 | 10.7% | — | stable, low ceiling |
| `b5c-schlongIS` | alpha 0.8, `td_loss`, **IS on** | 2.31M | 2.1% /1000 | 17.0% | — | IS correction cancels the benefit; peak ckpt evicted |
| `b1a-base` | none (control) | 503k | — | 16.7% | 5.89% | collapsed at 265k; score recovered, skill did not |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 545k | — | 11.0% | 3.91% | best of batch 3, degraded anyway |
| `b4b-unifbuf500k` | alpha 0 + 500k buffer | 1.23M | — | 9.3% | 4.03% | steadiest arm, but a low ceiling |
| `b4a-uniform` | alpha 0 | 1.25M | — | 8.7% | 3.50% | peaked ~575k, drifted down |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 549k | — | 8.3% | 4.41% | declined despite the floor — falsified hypothesis A |
| `b2a-base2` | none (repeat) | 999k | — | 7.0% | 2.64% | the 1M-step reference: no collapse, long oscillation, 1.1% at the end |
| `b3c-buf500k` | 500k buffer, alpha 0.6 | 4.81M | — | 5.7% | 0.27% | **died at ~750k**, score 0.0 for 4M steps |
| `b5a-schlong` | alpha 0.8, `td_loss`, no IS | 2.05M | 0% | 10.0% | — | **died at 272k**, `b4c` repeat |
| `b5b-schlong2` | alpha 0.8, `td_loss`, no IS | 1.92M | 0% | 7.7% | — | **died at 246k**, `b4c` repeat |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | — | 1.7% | 0.08% | dead end |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | — | 1.0% | 0.65% | stopped early, verdict weak |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | — | 0.7% | 0.10% | dead end |

`train` was a human-started run on committed defaults, stopped by the human. Never
touch `snek2/savedPolicies/train*`.

Four things this ranking makes visible that per-batch reading did not:

- **The top two arms are the same three PER changes at different sharpness.** `b4c`
  (eff ~1.6) and `b6b` (eff ~1.2) differ only in alpha. Nothing else tried comes close.
- **Ceiling and reliability trade off.** `b4c` is higher when it lives but dies 2 of 3
  times; `b6b` is lower and survived. Priced for the death rate, `b6b` wins ~24.5% to
  ~10.6%.
- **`b1a-base`, a plain baseline, outranks four deliberate interventions**
  (`MIN_EPSILON`, both n-step values, the 500k buffer with PER). Most changes tried in
  this investigation made things worse.
- **The graph misranks arms badly.** `b5c-schlongIS` is 2nd of the batch-5/6 arms by best
  perfect-30 (17.0%) and **last by measurement** (2.1%). Any ranking built on 10-episode
  graph evals is unreliable; see [`hyperparamTuning.md`](hyperparamTuning.md).

## Batch 5 — `b4c` repeat plus factor isolation

**Started 10:05 2026-07-27, restarted 17:09 for visible windows, all arms stopped
2026-07-28.** `b5a`/`b5b` stopped as dead; `b5c`/`b5d` stopped past peak.

All four arms shared `PRIORITY_EXPONENT=0.8` and differed only in the other two PER
factors, making the batch a replication attempt *and* a factor isolation at once:

| policy | differs from `b4c` by | role | outcome |
|---|---|---|---|
| `b5a-schlong` | nothing | `b4c` repeat, seed 1 | **died** 272k |
| `b5b-schlong2` | nothing | `b4c` repeat, seed 2 | **died** 246k |
| `b5c-schlongIS` | IS weights back on | isolates IS weights | survived, 17.0% peak then 2M-step decline |
| `b5d-schlongTDE` | `abs(td_error)` signal | isolates the priority signal | survived, stable, 10.7% ceiling |

`b5c` and `b5d` each differ from the repeats by **exactly one factor**, so the design was
meant to read three ways: all four high would give four seeds for the config plus the
factor isolation free; only the repeats high would name the factor carrying the gain; none
high would mean `b4c` was a lucky seed. That is the same information two sequential
batches would give, in half the wall clock.

**Outcome: the third reading, and the batch inverted its own premise.** Both exact repeats
died permanently in the 200-270k window and stayed flat at 0.0 for 1.7-1.9M steps. Both
single-factor arms survived. This retracted the "restoring `theSchlong`'s PER triples the
perfect rate" finding and led to the effective-exponent mechanism — see
[`findings.md`](findings.md).

### Restarted at ~36-47k to add visible windows

All four were killed and relaunched from their checkpoints so every arm renders a game
as it trains. Their graphs continue rather than restarting, with a resume marker at the
restart step. **Caveat: cpprb does not persist replay-buffer priorities across
save/restore**, so the restart reset all priorities to uniform and they rebuilt as
transitions were resampled. For a batch whose subject *is* prioritization that is a real
perturbation, but it landed at ~5% of the planned run and hit all four arms about
equally, so it should not bias the between-arm comparison. It is a reason not to restart
an arm deep into a run.

## Batch 4 — the sampling machinery

Three arms spanning the whole prioritization axis: none (`b4a`), none-plus-diversity
(`b4b`), and `theSchlong`'s original maximum (`b4c`). Spanning the axis paid off,
because the answer was at the end nobody expected — uniform sampling was the prior
favourite and came in at a third of `b4c`'s rate.

| arm | expected | actual |
|---|---|---|
| `b4a-uniform` | best of the three; PER already lost at 30k | 8.7% — better than the committed default, far below `b4c` |
| `b4b-unifbuf500k` | uniform plus diversity should stack | 9.3% — steadiest curve, marginal gain over `b4a` |
| `b4c-schlongper` | a long shot; "less correct" than the default | **34.0%** — the breakthrough |

Detail on why `b4c` won is in [`findings.md`](findings.md). The one caveat carried
forward: it is **n=1**, and its own history contains a collapse at 150-300k deep enough
to have ended the run if it had been judged at 300k.

## Batch 3 — the epsilon hypothesis

Led with interventions that plausibly prevented the ~265k collapse. **Result: A
falsified, B ambiguous and later retracted.**

| item | change | outcome |
|---|---|---|
| A | floor epsilon at 0.001 | **FALSIFIED** — `b3a-epsfloor`, `b3b-epsfloor2` both degraded anyway |
| B | `REPLAY_BUFFER_MAX_LENGTH=500000` | looked supported, then `b3c-buf500k` died at 750k. **Not settled** |
| C | `PRIORITY_EXPONENT=0.0` | promoted to batch 4, run as `b4a-uniform` |
| D | `GRADIENT_CLIPPING=10` | never run; still queued |
| E | third baseline repeat | effectively done — `b3a` spent its first 267k steps as one |
| F | LR schedule | dropped; no evidence of optimization instability |

The reasoning behind A and B at the time, kept because A was wrong and the reasoning
was still sound:

- **A — floor epsilon.** The one arm that reached epsilon 0.0 was the one arm that
  collapsed, and at 0.0 the buffer becomes a closed policy-data feedback loop. Expected
  the same 14%-class rate as `b1a-base` but held past 300k.
- **B — bigger buffer.** The buffer holds *transitions*, not episodes, and episode
  length grows with skill: at score 5 an episode is ~50 steps so 100k transitions span
  ~2000 episodes, but at score 80 it is ~800+ steps and spans only ~125. Experience
  diversity therefore *shrinks as the policy improves*, which fits the timing of a
  collapse that arrives well after the policy gets good.
- A and B were considered complementary rather than redundant: **A keeps exploratory
  data being generated, B keeps it from being evicted.**

## Batch 2 — baseline repeats and n=2

| arm | purpose | outcome |
|---|---|---|
| `b2a-base2` | repeat the baseline to learn its spread | ran to 999k as the 1M reference. No collapse, but a long oscillation ending at 1.1% |
| `b2b-nstep2` | test whether n-step's steadiness survives at n=2 | same shape as n=3, one step milder. Closed the n-step direction |

`b2a-base2` mattered more than expected: it is the same config as `b1a-base` and did
*not* collapse, which is what established that collapse is stochastic rather than a
property of a config.

## Batch 1 — the original plan, and how wrong it was

| policy | change | expected | actual |
|---|---|---|---|
| `b1a-base` | none (control) | a *reference*, not a winner | became the key run: collapsed at 265k, and still finished 2nd overall |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | smoother curve, smaller drawdown | faster early, *larger* drawdown |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | earlier learning onset, better late game | slowest to rise, then declined for 850k steps |

The reasoning at the time, kept because two of three predictions were wrong:

- **`b1a-base`** — a control under the same machine load as its batch mates, since
  throughput and contention vary between batches and make cross-batch comparison weaker
  than within-batch.
- **`b1b-tgt200`** — the highest-prior change of the batch. 8 gradient steps between
  target-network syncs is extremely frequent where standard DQN uses hundreds to
  thousands, and a target chasing the online network is a classic cause of oscillation
  and forgetting, which was exactly the reported symptom.
- **`b1c-nstep3`** — the food reward is immediate but the perfect-game bonus is terminal
  and extremely sparse, so credit has to crawl back one step per gradient update; n-step
  returns propagate it ~3x faster.

### The interim read at 83k, and why it was worthless

Batch 1 was first compared at matched step 83000:

| policy | trailing-20 perfect % | 1st perfect | last-5 score | curve mean | max drawdown | peak |
|---|---|---|---|---|---|---|
| `b1a-base` | **1.5** | 44000 | **66.0** | 52.4 | 19.2 | 78.6 |
| `b1b-tgt200` | 0.5 | **33000** | 59.2 | **53.4** | 27.4 | 76.9 |
| `b1c-nstep3` | 0.0 | never | 26.3 | 16.5 | 18.6 | 38.1 |

Three conclusions were drawn from this table and **all three were later overturned**:

1. That no arm showed catastrophic forgetting. `b1a-base` collapsed at 265k.
2. That `b1c-nstep3` had the batch's best trajectory. It peaked at 255k and declined
   for 850k steps.
3. That perfect games arrive ~20x earlier than the premise suggested, since `b1a-base`
   hit a 10% eval at 44k. A single 10-episode eval reading 10% is one perfect game in
   ten, which is noise, not an arrival.

This is the origin of the rule that **nothing can be judged below ~250k steps** — see
[`hyperparamTuning.md`](hyperparamTuning.md).

### `N_STEP_UPDATE=3`: the momentum that fooled everyone

Judged on trajectory rather than level, n=3 looked like batch 1's most interesting arm.
Score by 40k-step block:

| steps | mean score | max |
|---|---|---|
| 0-40k | 5.1 | 16.2 |
| 40-80k | 26.9 | 38.1 |
| 80-120k | 26.8 | 35.4 |
| 120-160k | 37.4 | 66.3 |
| 160-200k | 49.7 | 70.1 |
| 200-240k | 57.3 | 67.5 |

Near-monotonic, and at 201k it was still gaining +10.3 per 20 evals while `b1a-base` had
gone flat at ~69. That looked like exactly what the investigation wanted: slower but
consistently improving.

It peaked at 255k and declined for the next 850k steps. **The momentum was real and it
still meant nothing** — which is the strongest available argument for the ~350k minimum
horizon, since every signal available at 240k pointed the wrong way.

### `TARGET_UPDATE_PERIOD=200`: hypothesis not supported, but interesting

The prediction was smoother curves and smaller drawdowns. It got the **opposite** on
drawdown (27.4 vs 19.2) and a lower last-5. What it did do is learn **much faster
early** — roughly score 55 by 15k steps where the baseline needed ~25k — and reach its
first perfect game sooner.

So the frequent-target-update theory of forgetting looks wrong, while "less frequent
target updates learn faster early" is worth pursuing. `=50` and `=500` are in the
backlog. Note this arm was stopped at 104k, well short of the judgeable horizon, so
both readings are weak.
