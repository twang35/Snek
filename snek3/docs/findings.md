# Findings

**Empty on purpose.** snek3 inherits none of snek2's findings: they are results about snek2's
hyperparameters under TF-Agents, measured on a different collector with a different replay ratio and
a different RNG, and this project has already learned that framework details matter here — its single
largest result came from diffing snek2 against `theSchlong`.

What snek3 does carry is [`invariants.md`](invariants.md), which is a different kind of thing:
properties of the game and the instrumentation rather than conclusions about a knob.

snek2's findings remain readable at `../../snek2/hyperparamTuning/findings.md` and are a reasonable
source of **priors** — which hypotheses are worth testing first — but never of established fact about
snek3.

## Established

**Newest first.** A new finding goes directly under this heading, above the one before it, so the
top of the section is the most recent thing learned. Same rule in `Falsified` below.

### Epochs and network shape fix b4's drawdown; all four stability knobs only dent it

**Measured 2026-09-01** across five batches. Drawdown is the median over seeds of the share of an
arm's post-competence stage-A evals (onset = first eval at ≥80% perfect) below 50% perfect. Stage A
measures the argmax, so this is the deployed policy collapsing, not eval noise.

| batch | network | epochs | drawdown < 50% | < 80% |
|---|---|---:|---:|---:|
| **b4** | `fc (200,100)` | **8** | **2.0%** | **10.0%** |
| b5 | `fc (320,)` | 8 | 0.1% | 3.9% |
| b6 | `fc (200,100)` | 4 | 0.0% | 2.4% |
| b7 (32 arms, 8 layouts) | various | 4 | 0.1% | 2.6% |
| b8 (16 arms, 4 knobs) | `fc (200,100)` | 8 | 2.3% | 9.5% |

All rows truncated to a matched **50M** cap, b7's horizon. **Only the `fc (200,100)` + 8-epoch cell
collapses**; the other three cells of the 2x2 sit near zero, and b7 puts 32 arms across eight layouts
on the 4-epoch side of it at 0.1%. Every one of b7's layouts lands between 0.1% and 0.7%, including
`fc (400,200)`, its *worst* layout on record density — so the drawdown does not track layout at all.

**Against that, b8's four stability knobs are the weaker lever.** At b8's own 100M cap the control is
8.4% and the knobs run 2.2% (λ 0.95), 3.5% (anneal), 3.7% (entropy 0.003) and 5.9% (`target_KL`) —
real reductions, but dropping to 4 epochs takes the same config to 0.0-0.2%. b8 was the right
experiment on the wrong knob set.

**‡ What this does not settle.** b7 capped at 50M, where b4 has only reached 2.0% of its eventual
9.0%; the effect has an onset and develops monotonically (b4: 0.2% at 10M, 0.7% at 25M, 2.0% at 50M,
8.4% at 100M). b7's rate is *flat* across those same caps, which is why the reading is that it escapes
rather than lags — but nobody has run b7's config past 50M, so "never develops it" is untested.

### Steadying the training curve and banking record checkpoints trade off

**b8's four knobs, measured 2026-09-01**, ranked by drawdown against their ≥98%/500 density:

| knob | drawdown < 50% | ≥98%/500 | `hof5000` candidates | 5,000-ep mean |
|---|---:|---:|---:|---:|
| λ 0.95 | **2.2%** (best) | 2.3% (worst) | 14 | 97.44 |
| entropy 0.01 → 0.001 | 3.5% | 5.0% | 44 | 97.56 |
| entropy 0.003 | 3.7% | 4.3% | 23 | 97.77 |
| `target_KL` 0.02 | 5.9% (worst) | **6.0%** (best) | **54** | 97.63 |
| b4 control @100M | 8.4% | 5.7% | 274 (at 200M) | 97.71 |

**The two columns run opposite**: Spearman −0.8 between drawdown rank and candidate-count rank across
the four knobs. The knob that steadies the curve most produces the fewest record checkpoints, and the
knob that steadies it least produces the most. Mechanically this is plausible — the excursions that
make a curve ugly are also the excursions that occasionally land somewhere very good, and a
checkpoint is banked from a moment, not from an average.

**‡ Treat this as a hypothesis, not a result: n=4 knobs, so rs=−0.8 is p≈0.2.** It is stated here
because it reframes what a stability knob is *for* — b8 was queued on the assumption that fixing the
drawdown would raise the record rate, and across four knobs it did the reverse. Before another
stability batch, decide which of the two metrics is the target. **What is not in doubt is the
headline: no b8 group beat the control on density, and b8 produced one champion-level row in 135.**

### `fc (320,)` is the best network shape on this task, and b3's single-seed layout ranking inverted

**Measured by batch b7, 2026-09-01** — the fc-layout sweep, all four waves closed: 8 layouts x 4
seeds, 50M transitions each, everything but `fc_layers` at PPO's reference (4 epochs, lr 3e-4,
γ 0.99, λ 0.98, entropy 0.01, 128x128 rollout, minibatch 256, b2's reward), seeds 1-4 pinned to the
seed in each arm name. 32 arms, 28,006 stage-B rows at 500 episodes. Ranked by ≥98%/500 density:

| layout | ≥98%/500 | per-seed | vs `fc 320` |
|---|---:|---|---|
| **`fc (320,)`** | **17.3%** | 18.5 19.0 16.5 15.1 | — |
| `fc (200,100)` | 11.8% | 11.2 12.3 10.6 13.3 | p=0.029 |
| `fc (100,200,100)` | 11.6% | 9.9 15.5 6.1 14.8 | p=0.057 |
| `fc (100,100)` | 11.3% | 4.6 16.8 13.7 10.0 | p=0.114 |
| `fc (200,100,50)` | 10.8% | 10.3 8.9 12.8 11.2 | p=0.029 |
| `fc (160,160)` | 8.3% | 7.0 7.2 8.6 10.4 | p=0.029 |
| `fc (300,100)` | 6.8% | 6.6 4.9 6.8 9.0 | p=0.029 |
| `fc (400,200)` | 5.1% | 3.7 8.3 5.4 2.9 | p=0.029 |

p is an exact two-sided Mann-Whitney on the four per-seed shares; 0.029 is the floor at 4-vs-4 and
means complete separation — **every** `fc 320` seed above **every** seed of that layout. `fc 320`
clears five of the seven that way.

**This inverts the ranking b3 measured at one seed each**, which had `fc 300,100` 9.0% > `fc 200,100`
7.9% > `fc 320` 5.6% and was the reason b4 and b7 were queued at all. At four seeds the order of
those three is exactly reversed. Retraction is at
[Two hidden layers beat every single-layer width](#two-hidden-layers-beat-every-single-layer-width-and-width-past-320-actively-hurts).

**Two secondary readings, both about capacity going the wrong way.** Width past 320 hurts, now at
n=4: `fc (400,200)`, the largest network in the sweep, is last on density (5.1%) *and* last on
candidate count (683 screened checkpoints per arm against `fc 320`'s 1,001) — a double loss, not a
trade. And depth is not what mattered: the two three-layer shapes land mid-table, and `fc (100,100)`
at ~14k parameters ties `fc (200,100)`, so the second layer buys nothing that the first layer's width
does not already buy.

**What it does not settle.** 50M is a quarter of b4/b5/b6's horizon, so a density read against those
truncates. `fc 320` at 4 epochs and 50M scores 17.3% against b5's 9.6% at `fc 320` + 8 epochs and
255-271M, which is consistent with 4 epochs beating 8 but confounds budget with epochs. And b7's
`fc (200,100)` at 50M scores 11.8% against b6's 12.9% at the same shape and epochs with 4x the
budget — **so the ≥98%/500 density is nearly flat in budget past 50M**, which is what makes the
truncation tolerable and is worth knowing before buying another 4x.

### Screen checkpoint-rich configs on the stage-A ≥98% rate, not on `strong_eval_fraction`

**Measured across b7's 32 arms, 2026-09-01.** Both statistics come free from the same eval history,
and they disagree about which layout to want:

| predictor of a layout's ≥98%/500 density | across 32 arms | across the 8 layout means |
|---|---:|---:|
| stage-A share of evals ≥98% | **+0.80** | — |
| `best_perfect30` | +0.71 | +0.61 |
| `trailing_now` | +0.28 | — |
| `strong_eval_fraction` (share ≥80%) | **−0.19** | **−0.79** |

Pearson on the first row, Spearman on the rest. **`strong_eval_fraction` ranks the layouts backwards**
— its best layout, `fc (300,100)` at 95.0, is second-worst on density, and its worst, `fc 320` at
90.9, is the winner. The cause is the threshold: 80% perfect is far below the region a champion hunt
cares about, and the top of the distribution moves independently of it. `fc 320` spends **21.1%** of
its post-competence evals at ≥98% against `fc (400,200)`'s 12.4%, while spending *more* of them below
80%.

**This does not retire `strong_eval_fraction`** — it is still the lowest-variance summary of whether
an arm is *competent*, which is what it was chosen for, and it is what tells you a run is broken. It
is the wrong screen for **where the records are**, and those are different questions. Neither the
post-competence sd (+0.09) nor the post-competence mean (+0.08) predicts density at all, so this is
not "variance helps a max-hunt" — the ≥98 mass really is a distinct property of a config.

### A 500-episode ranking of two close batches did not survive 5,000 episodes

**Re-measured 2026-08-31 to 2026-09-01**, `hof5000` passes over every b4/b5/b6 checkpoint that
scored ≥98.5% on its 500-episode close-out (`above:98.5`), at 5,000 episodes each:

| batch | 500-ep headline | rows re-measured | 5,000-ep mean | rows ≥98 | ≥98.73 | best row |
|---|---:|---:|---:|---:|---:|---:|
| b6 | **12.9%** ≥98%/500 | 1,299 | 97.80 | 41.6% | 20 | 99.10 |
| b7 | 10.9% | **766** | 97.75 | 36.8% | 6 | **99.20** |
| b5 | 9.6% | 873 | 97.80 | 41.5% | **29** | **99.20** |
| b4 | 7.3% | 274 | 97.71 | 33.6% | 1 | 98.80 |
| b8 | 4.6% | 135 | 97.61 | 29.6% | 1 | 98.80 |

**Extended 2026-09-01 with b7 and b8**, which add 901 rows and make the regression the best-measured
quantity here: **−1.05 pp pooled over b7's 766 and −1.15 over b8's 135**, against b6's −0.94, and
−0.94 to −1.33 across all twelve groups of the two batches. A 500-episode row discounted by about a
point predicts its deep rate. The regression also scales with how thin the candidate pool is —
b7's `fc (400,200)` (31 candidates) regresses −1.27 and b8's λ 0.95 (14) regresses −1.33, against
−0.94 for `fc (320,)`'s 174 — which is the selection effect behaving exactly as it should.

**b6's lead over b5 dissolves**: identical means, ≥98 rates within 0.1 pp, and b5 ahead on the two
things a champion hunt is for — more champion-level rows and the top checkpoint. b4 stays clearly
last on all three, so that half of the earlier reading holds. The mechanism is the one
[`../CLAUDE.md`](../CLAUDE.md) and `invariants.md` already state — a maximum over N measurements is
upward-biased, and 500 episodes has a 0.72 pp sd against 5,000's 0.23 — but the size is the news:
**a 3.3 pp gap in the pre-registered headline was entirely selection.** Read a two-batch ranking off
the 500-episode density only when the gap is large; b4-vs-b6 (5.6 pp) survived and b5-vs-b6 (3.3 pp)
did not.

### Running b5 longer bought nothing; running b6 longer paid, modestly

**Measured 2026-08-31, re-checked 2026-09-01** by placing every champion-level row (≥98.73% over
5,000 episodes) in a decile of its own arm's run:

| batch | champion rows | decile of the run | first half / second half | arms better in their second half |
|---|---:|---|---:|---:|
| b5 (`fc 320`, 8 ep, 255-271M) | 29 | **21 of 29 in decile 1** | 24 / 5 | 2 of 8, mean **−0.09 pp** |
| b6 (`fc (200,100)`, 4 ep, 215-231M) | 20 | 12 of 20 in deciles 8-10 | 4 / **16** | 6 of 8, mean **+0.14 pp** |

**b5 is front-loaded and b6 is back-loaded**, so the same extension was wasted on one batch and paid
on the other. b5's top checkpoint — now snek3's record — is at 9.0M transitions, **3.5% of the way
in**; the arm then ran another 246M.

**Read the row-level split as indicative only.** Adjacent checkpoints are strongly correlated, so the
binomial on halves (p=0.0005 for b5, p=0.0118 for b6) overstates its own confidence. The honest test
is the per-arm sign test, and at 2-of-8 and 6-of-8 **neither batch is significant on its own**
(p=0.29 either way). What survives is the pairing: the two batches point in opposite directions from
the same protocol, and b6's +0.14 pp mean is a seventh of what its strongest single arm suggested
(b6a, +0.42 pp) — the usual maximum-over-8 bias.

**The practical rule stands on b5 alone: an arm's record region is not where its training curve
peaks.** A cap chosen to "let the curve settle" would have kept b5h running for 246M transitions
past the checkpoint that mattered.

### snek3 has a champion, and it beats snek2's at matched depth

**Admitted to [`../hallOfFame/HOF.md`](../hallOfFame/HOF.md) 2026-09-01**, on 30,000 fresh episodes
at seed 7 — a seed no selection pass used:

| entry | transitions | 30,000-episode rate |
|---|---:|---:|
| `b5h-ep8-seed8-ckpt9027584` | 9.0M | **98.96%** |
| `b6b-fc200x100-seed2-ckpt133120000` | 133.1M | 98.73% |
| snek2 champion `b44a-lowlr7-b29b-ckpt2739000`, re-measured | — | 98.48% |

**The comparison must be at matched depth or it says the opposite.** Against the snek2 champion's
*published* 98.73%/3,000 the new entry reads as a tie (p=0.26); re-measured on the same 30,000
episodes the champion reads 98.48% and b5h wins by +0.47 pp (z=5.16, p<1e-6). The published number
was itself a selected high. The admission rule and the basin tie-break are in the
[`hof-promote`](../skills/hof-promote/SKILL.md) skill.

### Network shape and PPO epochs interact negatively, so b3's per-knob ranking does not compose

**Measured 2026-08-31 across three 8-seed batches** on b2's reward function and one protocol, read
off the `results` branch 2026-09-01. Ranked by the ≥98%/500 density a champion hunt cares about:

| batch | network | epochs | transitions | best30 | pooled ≥98%/500 |
|---|---|---:|---:|---:|---:|
| b6 | `fc (200,100)` | **4** | 215-231M | 97.8-98.5 | **12.8%** (4,661 of 36,272) |
| b5 | `fc (320,)` | 8 | 255-271M | 97.8-98.5 | 9.6% (3,329 of 34,581) |
| b4 | `fc (200,100)` | 8 | 200M | 97.0-97.9 | 7.3% (1,079 of 14,733) |

Batch b3 swept one knob at a time at 20M and ranked epochs 8 first (9.7%) and `fc (200,100)` second
(7.9%), both above its `fc 320` reference (5.6%). **The arm carrying both is worse than either one
alone** — and worse than the reference config it was built from. Each pairwise reading is a
single-knob comparison: at `fc (200,100)`, 4 epochs beats 8 (12.8 vs 7.3); at 8 epochs, `fc 320`
beats `fc (200,100)` (9.6 vs 7.3).

**The general lesson is about the sweep, not the network.** A one-knob-at-a-time pass ranks knobs; it
does not license stacking the winners, because it measures no interaction at all. b4 was queued as
"the clean network-shape test" on exactly that reasoning — take b3's two best knobs and hold
everything else — and the composition is what failed, not the test.

Two caveats, both real. b4's 200M is the shortest of the three horizons, so every comparison truncates
there; and a count of rows above a threshold is not a stable statistic even at fixed depth (below).
The **sign** is worth acting on and the size is not, which is what [batch b7](runs.md) holds epochs at
4 for.

**The mechanism is visible in the charts, and it is instability rather than a lower ceiling.** Past its
competence onset b4 spends a median **9.1%** of its evals below 50% perfect, against 0.7% for b6 and
0.0% for b5 — and stage A measures the **argmax**, so that is the deployed policy collapsing and
recovering, not eval noise. An arm that spends a tenth of its life collapsed cannot bank record
checkpoints, which is how one number explains the other. The rate is 8.5% in b4's first 100M and 9.0%
in its second, so it is fully developed early and does not need the long tail — but it *is* invisible
at a 10M horizon, where all three batches look alike. [Batch b8](runs.md) tests four candidate fixes
against b4 as the control.

### 16 stage-B shards is right on the desktop, and snek2's "18 loses 6-10%" cliff does not reproduce

**Measured 2026-08-30 on the desktop**, 5 waves, identical work each (b5h's `screen:97` set, 3,039
checkpoints at 100 episodes), wall clock from the ledger:

| shards | minutes |
|---:|---:|
| 8 | 5.02 |
| 12 | 4.56 |
| **16** | **4.06** |
| 20 | 4.10 |
| 24 | 4.07 |

**The knee is at 16 and there is nothing above it** — 16, 20 and 24 sit inside 1% of each other, so
the inherited default was right, but *not* for the reason the docs gave. `HARD_MAX_EVAL_SHARDS` was 16
because snek2 measured 18 shards **losing 6-10%** on TensorFlow; on torch that cliff is absent and
over-subscribing simply stops helping. The ceiling is worth keeping as a "no gain past here" bound
rather than as protection from a regression.

**‡ One caveat on the 8-shard row.** It was the first eval wave and ran while the training phase's 12
stage-A workers were still idling out their 300 s, so it is the one row that could be penalised. The
12→16 step is unconfounded, and it is the larger part of the gain.

### The worker sweep did not resolve anything, because 3.2-minute waves are too short

**Measured 2026-08-30 on the desktop**, 6 waves × 8 warm-started arms, each advancing 1,196,032
transitions:

| workers | minutes |
|---:|---:|
| 4 | 3.20 |
| 6 | 3.24 |
| 8 | 3.23 |
| 10 | 3.73 |
| 12 | 3.77 |
| **12 (repeat)** | **3.23** |

**The repeat disagrees with its own twin by 0.54 min (17%), while the whole 4/6/8 range spans 0.04 min
(1.3%).** So the wave-to-wave repeatability is larger than every difference the sweep was built to
detect, and **none of these rows supports a conclusion** — including the tempting one that 4 workers
is as good as 8, which would contradict the +12% that 6→8 measured on this same box.

The cause is the window, not the design. A wave is ~3.2 min including ~40 s of startup, and the
stage-A queue has depth 16 — so a large fraction of each wave is the queue filling and draining rather
than steady state. The earlier hand-measurement used **30-35 min** windows for exactly this reason.

**What to keep from it:** the harness shape works — every wave does identical work, so ledger wall
clock is the whole measurement, with no instrumentation to misread. Re-run it with ~10x the
transitions per wave. **And the repeat pair is what caught this**; without it the 4/6/8 flatness would
have read as a finding.

### Six eval workers is the wrong number for eight PPO trainers, and for PPO the bound is free to raise

Measured 2026-08-30 with batches b5 (8 arms, desktop) and b6 (8 arms, laptop) both at ~25M
transitions and best30 ~97%:

| | eval workers | trainers | cores | box idle |
|---|---|---|---|---|
| desktop, 16 cores | **598 of 600%** — pinned | 492 of 800% | 16 | ~1/3 |
| laptop, 14 cores | **576 of 600%** — pinned | 371 of 800% | 14 | ~1/3 |

**Stage A is the binding constraint and the workers are saturated while the trainers idle at half
capacity.** That stage A dominates is already established above; what is new is that the *worker count*
is now the limit rather than lane drain. [`running.md`](running.md)'s 6-worker optimum was measured at
**four DQN trainers**, and PPO trains ~14x faster per transition, so eight PPO arms demand far more
eval throughput per unit time than that sweep ever saw. Per-arm rates at the time: 9,262 transitions/s
on the desktop, 7,850 on the laptop.

**And the reason the queue bound exists does not apply to PPO.** Queue depth is bounded because a
lagging row delays DQN's epsilon *refinement* schedule, which reads `perfect_percent` back out of eval
rows ([`invariants.md`](invariants.md) invariant 2). PPO's only schedule is the entropy coefficient and
it is a pure function of the step, so **queue lag cannot change a PPO arm's trajectory, only its
speed.** The second objection also does not apply: a queued worker's eval seed already mixes clock, pid
and round number rather than deriving from the arm's seed — `eval_queue.py` states this as the queue's
accepted cost — so extra workers introduce no new class of variation.

**Measured 2026-08-30: eight workers beat six by ~12%, and eight is the ceiling on a 16-core box.**
Two extra workers were attached to the *running* desktop batch at slots 6 and 7 — no restart needed,
because trainers only manage slots `0..target-1` and work is claimed by rename, so
`python -m tools.eval_worker --slot 6` joins the same queue untracked.

| desktop b5, 8 arms | workers | per-arm rate | window |
|---|---:|---:|---|
| baseline | 6 | 8,262 transitions/s | 35 min |
| after | **8** | **8,556 transitions/s** | 30 min |

**Raw that is only +3.6%, and the honest number is +12%.** The laptop's b6 batch stayed at six workers
and served as a control: over the same hour its rate decayed **7,850 → 7,250/s (−7.6%)** as its policies
matured and episodes lengthened. So a six-worker desktop would have been at ~7,630/s, and holding
8,556/s means the extra pair bought back the decay and more. **A before/after on a maturing arm
understates a speed change by roughly the decay rate** — which is why the control matters more than the
window length here.

**Ten workers is past the edge, and this is where `running.md`'s DQN sweep does transfer.** At eight
workers plus eight trainers the box reads **load 14.1 of 16, 91.4% user, 8.3% idle** — and that sweep's
finding was that past the optimum the box "reaches 4% idle to run *slower*", because the fastest
configuration leaves ~20% free. The PPO-specific argument (its trainers are the idle half) bought one
pair, not two. **Eight is the number for eight PPO trainers on 16 cores**; the laptop's 14 cores
should take fewer.

**‡ `ps -o %cpu` is the wrong instrument for this and cost a wrong reading.** On Linux it is CPU time
over the process's *whole lifetime*, so the trainers — two hours old — reported 492% → 503% across the
change and looked unaffected while the throughput moved 12%. Use `/proc/loadavg` or `top -bn1` for
anything instantaneous.

### Two hidden layers beat every single-layer width, and width past 320 actively hurts

**‡ Superseded 2026-09-01 by batch b7**, which swept the same axis at 4 seeds and 50M and found the opposite: `fc (320,)` first at 17.3% and `fc (300,100)` second-last at 6.8%, with the three shapes below in exactly reversed order. Everything in this section is n=1 at 20M. Kept because the reasoning is what queued b4 and b7, and because *width past 320 hurts* did replicate — `fc (400,200)` is last of eight. See the top of this section.

From batch b3's fc sweep, at 10M transitions each on b2's reward function, ranked by the statistic a
champion hunt actually cares about — how densely an arm produces ≥98%/500 checkpoints:

| network | parameters | best30 | ≥98%/500 | density | best checkpoint |
|---|---:|---:|---:|---:|---:|
| `fc 300,100` | 39,703 | 96.9 | **21** of 233 | **9.0%** | 99.0% |
| `fc 200,100` | 26,603 | **97.1** | 17 of 215 | 7.9% | **99.2%** |
| `fc 320` — the reference, and snek2's shape | 10,883 | 96.6 | 6 of 108 | 5.6% | 98.4% |
| `fc 500` | 17,003 | 94.7 | 3 of 93 | 3.2% | 98.4% |
| `fc 200` | 6,803 | 96.4 | 1 of 131 | 0.8% | 98.0% |

**It is not simply capacity.** `fc 500` has 2.5x the parameters of `fc 200` and a *lower* best30; `fc
200,100` has 1.6x the parameters of `fc 500` and is far better. So the two effects are separate: **more
width past 320 hurts, and a second layer helps.** The two-layer arms also hold the two highest single
checkpoints in the entire sweep.

**Why this matters beyond PPO.** `fc 320` is not a tuned snek3 choice — it is snek2's shape, carried
across so a champion's weights convert, and every batch in both eras has used it. This is the first
evidence in the project that it is the wrong shape, and it was found by a knob nobody had swept.
`dqn/net.py` takes the same `fc_layers` config, so **the same test is available to DQN for the price of
one arm** and has never been run.

The caveat is the usual one: n=1 per shape, and the best30 column spans 2.4 pp across the three best
shapes, which this project's noise cannot resolve. The **density** column is the load-bearing one
because it pools 100-233 independent 500-episode measurements per arm rather than resting on a single
peak.

### A config sweep read at a truncated cap did not merely mislead, it inverted

Batch b3's PPO sweep ran seven configs to 3M transitions and then continued the same seven arms to
10M. **The 3M ranking was not a noisy version of the 10M ranking — it was a different ranking.**

| arm | knob off the reference | best30 @3M | best30 @10M |
|---|---|---:|---:|
| `b3g-ent003` | entropy coefficient 0.003 | 76.8 — **6th** | **96.9 — 1st** |
| `b3e-lam95` | λ 0.95 | 75.7 — **7th** | **96.8 — 2nd** |
| `b3a-lr3e4-g99` | none: the reference | **88.2 — 1st** | 96.6 — 3rd |
| `b3j-lr5e4` | learning rate 5e-4 | 87.3 — 2nd | 96.5 — 4th |
| `b3i-lr1e4` | learning rate 1e-4 | 70.5 — 8th | 95.0 — 5th |
| `b3h-ent03` | entropy coefficient 0.03 | 70.2 — 9th | 90.6 |
| `b3f-lam100` | λ 1.0 | 85.0 | 90.8, **still rising at the cap** |

Two separate facts, and the second is the useful one.

**The order changed.** The two arms that finished 1st and 2nd were 6th and 7th at 3M, and the arm that
led at 3M finished 3rd. The mechanism is not mysterious — a smaller entropy bonus and a shorter GAE
horizon both explore less early and neither plateaus lower — but it is invisible at 3M, and a sweep
stopped there would have picked its two best configs *last*.

**The spread collapsed.** Those top four span **11.4 pp at 3M and 0.4 pp at 10M** (96.9 / 96.8 / 96.6
/ 96.5). In a domain where the same config has produced 62.5 and 18.0, four configs inside 0.4 pp at
n=1 each is **one number, not a ranking** — so the sweep's honest output is a *set* of equivalent
configs plus a list of losers (entropy 0.03, λ 1.0, lr 3e-3 at sd 18.4, γ 0.9975), not a winner.

This generalises the b1 finding above from "a cap hides how good a config gets" to "**a cap can order
configs wrongly**", which is worse, because a sweep's whole output is an order. The practical rule is
in [`protocol.md`](protocol.md): rank on peak only after checking no arm was still climbing, and treat
a sweep at a cap every arm was still rising through as having produced no ranking at all.

### PPO reaches DQN's perfect-game rate on this task at a matched transition budget, ~13x cheaper

Matched on transitions **and** on reward function (both on b2's `chase_safe c=0.1 gate=75`, no
food-distance term, fc 320):

| | best30 at 10M transitions | best stage-B checkpoint | ≥98%/500 | measurements |
|---|---|---|---:|---:|
| PPO b3, 15 configs, seed 1 | 89.7 – **97.2** | **99.2% / 500**, re-measured **97.9% / 3,000** [97.3, 98.3] | **95** | 1,862 |
| DQN b2, 4 seeds | 93.6 – **96.9** | 99.2% / 500 | 5 | 1,135 |

**The best single checkpoint is a tie at 99.2%, and PPO's got there on 5.05M transitions against b2's
18M.** Where the two genuinely differ is the *density* of record-region checkpoints — the metric
[`../plans/ppo.md`](../plans/ppo.md) §10 pre-registered for this comparison — where PPO is **11.6x**
ahead: 95 checkpoints at ≥98%/500 against 5.

That is not a claim that PPO is the better algorithm here, and the counter-evidence is in the same
table: **snek2's champion still measures 98.8%/3,000 against PPO's best at 97.9%**, on 2.74M
transitions against 5.05M. What this entry does rule out is the reading that phase 6b's gate arm
suggested, that PPO is *behind* on this task. That arm was on snek3's unshaped defaults at 508k
transitions, and the reward function was doing most of the work.

**The cost difference is not marginal.** PPO's seven arms reached 10M transitions in **~20 minutes**
sharing a 14-core laptop; b2's DQN arms took **~7-8 h each** for 18M on the 16-core desktop — about
13x per transition, because PPO takes one gradient step per 256 samples per epoch against DQN's one
per transition. The 3,000-episode re-measurement of a selected high fell **2.0 pp** from its 500-episode
value, consistent with the 1.4 pp mean fall across snek2's four best hall-of-fame entries.

### Potential-based shaping was not policy-invariant: b1 and b2 shaped at γ=1.0 while discounting at 0.99 and 0.9975

`train.py` built its collect env as `VecSnake(width, seed=...)` and never passed `shaping_discount`,
so it kept the constructor's default of 1.0. Shaping pays `c·(γ·Φ(s') − Φ(s))`, and the theorem that
makes `c` free — the discounted sum telescopes to exactly `−c·Φ(s₀)`, so `c` cannot change the optimal
policy — holds only when that γ is the agent's own.

| arm | shaped at | discounted at | un-cancelled potential per step |
|---|---:|---:|---:|
| b1 | 1.0 | 0.99 | 0 — shaping was off |
| b2 | 1.0 | **0.9975** | `c·(1−γ)·Φ` = **2.5e-4** at `c=0.1` |

That is the same order as `FOOD_DISTANCE_REWARD` (0.001), and unlike the shaping it is meant to be, it
does not cancel: it is a **standing bonus for being alive in a high-potential state**. Fixed
2026-08-29 by passing the agent's `discount`.

**It changed nothing for an arm with shaping off, and that is measured rather than argued.**
`VecSnake.step` only reaches `_shaping_reward` when a coefficient is non-zero. Fingerprinting three
short arms before and after — every eval row, the final weights, the step, the transition count and
epsilon — left the shaping-free arm **byte-identical** and moved both shaping arms. So b1 and every
b1-class comparison stand; only a shaping arm's dynamics change, and b2 was already running.

**The eval path still shapes at 1.0, deliberately.** `engine.measure` is handed a checkpoint and has
no agent to ask for a γ, and `SNEK_DISCOUNT` is not in `EVAL_RELEVANT_ENV` for the good reason that a
learning-rate-class knob must not re-shard a wave. It moves `avg_reward` only — never `avg_score` or
`perfect_percent`, which is what every gate and screen reads.

**‡ The first version of the check could not fail, and finding that out is the transferable part.**
The exactness harness ran its shaping arm at b2's own `CHASE_SAFE_GATE=75`. A 1,200-step arm plays a
snake of length 4-10, so Φ was identically 0, the shaping term contributed exactly 0.0 whatever γ
was, and a deliberately reintroduced `shaping_discount=1.0` came back **IDENTICAL**. Dropping the
gate to 5 — cleared by the first meal — is what made the check able to fail. This is
[`../CLAUDE.md`](../CLAUDE.md)'s "a fixture whose subject cannot violate it is not a fixture", found
in a verification harness rather than in a test.

### Queue depth, not worker count, is stage A's bottleneck — depth 16 makes an arm train-bound

**23 configurations, laptop (14 cores: 10P + 4E), arms warm-started from the b44a champion so stage A
costs what it costs at 58-87% perfect. Each config settled until its queue filled, then its true
training cursor sampled over a 120 s window; `iostat -c` idle over the same window.**

| arms | workers | depth | per-arm st/s | h to 3M | box st/s | evals/s | idle | bound by |
|---|---|---|---|---|---|---|---|---|
| 4 | — | queue off | 91.3 | 9.12 | 365 | 0.365 | 63.0% | stage A, in-loop |
| 4 | 2 | 8 | 172.2 | 4.84 | 689 | 0.680 | 60.2% | eval |
| 4 | 6 | 8 | 259.4 | 3.21 | 1038 | 1.062 | 27.0% | eval |
| 4 | 10 | 8 | 235.1 | 3.55 | 940 | 0.973 | 4.1% | eval |
| 4 | 4 | 16 | 321.7 | 2.59 | 1287 | 1.336 | 34.8% | **train** |
| **4** | **6** | **16** | **350.0** | **2.38** | 1400 | 1.516 | 19.7% | **train** |
| 4 | 6 | 24 | 342.1 | 2.44 | 1368 | 1.401 | 5.0% | **train** |
| **8** | **6** | **16** | 226.9 | 3.67 | **1815** | 1.857 | 10.2% | eval |
| 8 | 8 | 16 | 229.2 | 3.64 | 1833 | 2.032 | 0.2% | eval |
| 12 | 6 | 8 | 126.7 | 6.58 | 1520 | 1.578 | 14.9% | eval |

**A queued arm does nothing but wait for stage A, and the wait is exactly predictable:**

    per-arm st/s = evals/s x EVAL_INTERVAL / arms

within 3% across every queued config. So arm count is a pure divisor of an eval pool, and the only
question is what sizes that pool.

**It is not worker count.** Workers turn over: at 4 arms the peak is 6, and 10 workers are *slower*
than 6 while taking the box from 27% to 4.1% idle, because they starve the trainers — measured
unblocked rate falls 380 -> 306 st/s. Eight workers at 4 arms match six exactly.

**It is outstanding checkpoints — `arms x depth` — because a deeper `measure_stream` round packs lanes
better.** The same 4 workers deliver 0.513 evals/s at 2 arms and 1.071 at 8, a 2.09x spread from queue
population alone. Which is why depth is the lever: at 4 arms it is the *only* way to raise the
population without adding arms.

**Depth 16 is a phase change, not a trend.** Every one of the 19 depth-8 configs — 2 to 12 arms, 2 to
10 workers — sat pinned against its cap with the trainer idle. At depth 16 the queue drains to 11-13
and the arm runs at **94% of its unblocked rate**: the workers are finally ahead. Depth 24 regresses
(2.44 h) because there is nothing left to win. Depth 12 captures only 63% of the gain and is still
eval-bound.

**Idle CPU is not the objective, and targeting it selects badly.** The fastest config runs at 19.7%
idle; the two configs driven to ~0-4% idle (4a/10w, 8a/8w/d16) are slower or flat while their trainers
lose 20-25% of their unblocked rate. Depth is the opposite trade — 4a/4w goes 3.94 h -> 2.59 h *and*
47% -> 34.8% idle, because it makes the existing work cheaper rather than buying more of it.

**Recommended invocations** (both are env knobs; no default changed):

| goal | config | result |
|---|---|---|
| one arm to 3M fastest | `SNEK_EVAL_QUEUE=1 SNEK_EVAL_QUEUE_DEPTH=16 SNEK_EVAL_WORKERS=6`, 4 arms | 2.38 h/arm, **3.83x** |
| most arms per day | same, 8 arms | 8 arms in 3.67 h, **4.97x** box throughput |

**The cost is schedule lag, and it is the whole cost.** Depth is how many eval intervals the epsilon and
guided schedules read behind; 16 doubles the 8 chosen for that reason. 16,000 counted steps is 0.5% of a
3M arm but proportionally much more over the first tens of thousands of steps, where those schedules
move fastest. Nothing here measures whether that harms learning — only what it buys in wall time.

### A snek3 step is four game moves and a snek2 step was one, so "3M steps" is not one budget

**Measured 2026-08-29: exactly 4.00 transitions per `collector.step()`** under b2's config
(`collect_envs=1`, `fork_branches=4`). `VecSnake` advances every lane on every call, `_bank` banks one
transition per lane, and `_learn` spends its gradient budget **per transition** — so one counted step
is four game moves, four buffer rows and four `agent.update()` calls, while `self.step += 1` happens
once.

snek2 is one of each. `snek2/forking_collector.py:16` is explicit: branches advance **in round robin,
"one counted environment step per call"**, so they *share* the main line's budget and "the 1 collect
step : 1 gradient step ratio the training loop relies on is unchanged".

| per counted step | snek2 b29 | snek3 b2 |
|---|---:|---:|
| game moves | 1 | **4** |
| buffer rows | 1 | **4** |
| gradient steps | 1 | **4** |
| **gradient steps per transition** | 1.0 | **1.0** |

**The last row is the one that matters, and it is why `SNEK_REPLAY_RATIO` is the wrong lever.** Data
efficiency is *identical* — each new transition buys one batch of 128 in both eras — so `train.py`'s
per-transition budget does reproduce snek2, exactly as its docstring claims. What differs is only what
the counter means. **b2 at 3M has not learned harder than b29 at 3M; it has run four times longer.**
Dropping the ratio to 0.25 would make snek3 4x *less* data-efficient than snek2 ever was.

So the equivalence is **`b2 @ 750k counted steps ≈ b29 @ 3M`**, and it holds on every axis including
the awkward one: snek2's main line got roughly a quarter of its counted steps, so at 3M it played
~750k primary moves against b2's 750k at the matched point.

**This confounds the b2-vs-b47 interim reading** — see [`runs.md`](runs.md). b2d at 0.39M counted steps
had done 1.56M transitions and 1.56M gradient steps; b47c at 1.63M had done 1.63M and 1.63M. The two
arms were at matched values on **both** axes when b2d "passed b47c at 4x fewer steps". **b1-vs-b2 is
unaffected** — both ran the same collector at the same ratio — so "the five knobs are the whole
difference" survives; only comparisons that cross the eras at matched *step counts* do not.

**Two corollaries for anything reading a step count.** Never compare a snek3 step count to a snek2 one
without the 4x; and `SNEK_FORK_BRANCHES` silently rescales the x-axis of every snek3 chart, so an
ablation that changes it changes what its own step numbers mean.

### Raising `SNEK_COLLECT_ENVS` speeds up transitions and slows down the arm

Two units wear the name "steps/s" and they differ by the lane count. `train.py`'s
`steps_per_second` counts **counted steps**; the throughput table below is in **transitions/s**. At
b2's width of 4 they are 4x apart, which is enough to invert a conclusion.

**`self.step += 1` runs once per `collector.step()`, so the cap is in counted steps.** Raising
`collect_envs` from 1 to 8 therefore makes a 3M-*step* arm consume 8x the experience and do 8x the
backprop. Measured: transitions/s rises 1.46x (1,947 → 2,379 after the fixes below), so **time to the
same cap rises ~5.5x.** The earlier reading of this — "raising `SNEK_COLLECT_ENVS` buys 1.9x" — is
true per transition and backwards per arm. It is a **width** knob, not a speed knob; the way to spend
it is to lower the cap by the same factor.

### Two bit-exact changes are worth 20% of the training half

Neither changes a number an arm produces, which is why both could land mid-batch.

| change | before | after | evidence it is identical |
|---|---:|---:|---|
| `SumTree.set_one` — a scalar walk instead of `set([leaf],[v])` | 43.4 us | **2.6 us** | `nodes` equal element for element over 3,000 random updates |
| `torch.optim.Adam(fused=True)` | 240.7 us | **162.1 us** | max absolute parameter difference **0.0** after 2,000 seeded steps |

**The sum tree was 8% of an arm's entire wall clock.** `add` runs once per transition — four per
counted step — and the vectorised `set` spends a 17-level walk allocating size-1 index arrays to move
eight bytes per level. The batch path is genuinely faster at 128 leaves (50.4 us against ~330 us of
scalar walks), so both exist and the choice is purely a cost one.

End to end on the real loop: **406 → 487 counted steps/s, +20%.**

**Two things that still do not work, re-confirmed at every batch size.** More torch threads are slower
at 128 through 4,096 (1 thread beats 2 and 4 everywhere), and samples/s scales only **2.3x** from batch
128 to 1,024 before degrading — 310k, 482k, 616k, 724k, 599k, 514k. So there is no large win hiding in
the batch dimension, and the gradient half's floor is ~1,950 transitions/s per core.

### Stage A's cost is lane drain, so cutting episodes barely helps

The obvious economy is the wrong one. **Four times fewer episodes buys 1.6x, not 4x**, because a
single checkpoint's measurement is dominated by its *tail* — the last long episodes running alone at
width 1 — and the tail is set by episode *length*, not count.

| episodes per eval | champion (98.8%) | b1a @3M |
|---:|---:|---:|
| 25 | 2.06 s | 3.69 s |
| 100 | 3.87 s | 6.05 s |
| 200 | 6.87 s | 8.69 s |

**Width is the only lever, and only streaming supplies it.** The same 100 episodes measured inside a
sustained `measure_stream` cost **1.13 s** (champion) and **1.85 s** (b1a) per checkpoint at 0.85-0.86
utilisation — **3.3-3.4x**, not the 5.7x the 3,222-checkpoint pass suggested, because that figure
compared against a slower stage-A sample.

**A cheaper in-loop eval is nevertheless safe, if it is ever wanted for latency rather than cost.**
Simulated on b45a's 3,222 real rows by taking each row's first 25 episodes — a fair sample, since
rows are banked in start order: the trailing-30 perfect rate moves by **0.49 pp rms** (1.97 pp worst
case), and `screen:92` on 25 episodes recovers **98.1%** of what `screen:95` on 100 selects while
admitting 1.03x as many. So the 100 is not load-bearing for the schedule. It is load-bearing for
nothing else either — but it is also not where the time goes.

### The thing that knows what is running should own the window, and a pid is what makes that cheap

snek3's chart window was built twice, and the second version is a fifth of the size because of one
change in what the registry stores.

**Version 1: the desktop daemon opened it.** A fixed 2x2 of the wave it had just launched, closed and
reopened whenever an arm joined or finished, and **no window at all on the laptop**, where most one-off
arms actually run. Both failures are the same mistake — the daemon is not what knows a training is
happening.

**Version 2, briefly: each trainer opened its own.** That makes the laptop work and removes the
reopen, at the cost of four overlapping windows on a box running four arms. Rejected by the user on
sight, which was right: the window is a thing to look at, not a thing to manage.

**What works is one window per box that reads a registry of live arms.** Each trainer writes
`runs/.live/<policy>` holding **its own pid** before its first step, then calls `ensure()`; panels
appear as arms start, without anything being reopened. It closes itself five minutes after the last
arm goes.

**Version 3, 2026-08-30: the "one" moved out of the launcher and into the viewer, because version 2
did not deliver it.** Every arm now spawns a viewer and the *viewer* takes an exclusive `flock` on
`runs/.live/.window` as it starts; the losers exit in ~0.3 s having drawn nothing. Two things are
worth separating here, because conflating them is what produced the bug:

| question | answer | changed in v3? |
|---|---|---|
| does the viewer need restarting when the arm set changes? | no — it reads the registry | no, this was v2's real win |
| who launches it? | every arm, and the daemon could too. It stops mattering | no |
| how is "only one" enforced? | **the kernel, via `flock`** | **yes — v2 was wrong here** |

**v2's claim protocol did not hold, and the way it failed is the general lesson.** It was an `O_EXCL`
create with a "the slot's holder is dead, take it over" fallback, and the fallback had no exclusion of
its own: it wrote a pid and read it back, which every racer can win. Two ways in, neither rare — the
slot file exists but is not yet *written*, so it reads as unheld; or it holds a dead pid, which is the
state at the start of every batch, because nothing deleted it. Measured over 20 trials of 8 concurrent
arms: **a mean of 6.6 windows**, and the desktop opened **5** on 2026-08-29. Its own docstring
anticipated two as the worst case.

**It went unnoticed because the test named after the guarantee could not fail.**
`test_only_one_arm_of_a_wave_opens_a_window` called `ensure()` four times *sequentially in one
process*. A lock between processes has to be tested between processes — the fixture now starts eight
real interpreters on a starting gun, and `tests/mut_window.json` kills 11 mutants including a
`LOCK_EX` → `LOCK_SH` that reproduces the original bug exactly.

**Why it surfaced on that batch and not earlier:** two changes landed the same day. Window ownership
moved from the daemon to the trainers at 08:50, so this was the first desktop batch where arms raced
at all; and `HARD_MAX_TRAINERS` went 4 → 8 at 13:45, so eight raced instead of four. The laptop had
been hiding it all along, because arms there are launched by hand seconds apart and the second one
finds a live holder.

**The rule that comes out of it: prefer a primitive whose failure cases the kernel retires to a
protocol that handles them.** An `flock` cannot be held twice, is released however its holder dies —
`kill -9` included — and leaves nothing behind to be recognised as stale, so all three of v2's cases
stop existing rather than needing a fourth branch. The cost is 7 processes × 0.3 s per batch, paid
once, for a launcher with no protocol in it.

**Panels are sticky within a wave**, added 2026-08-29 at the user's request and for the right reason:
a batch is read as a batch, so with one arm left of four the other three are most of the answer. That
needs a rule for when a wave *ends*, or the set grows forever — here, the registry going empty and an
arm appearing again. snek2 had the sticky property and no such rule, and drew **eight panels for four
arms** when a batch was relaunched inside its 12 h TTL.

**Storing the pid is what shrinks it.** snek2's viewer had the same registry idea with a *name and a
timestamp* per arm, so nothing in the file could be asked whether it was still true, and it needed
`pgrep` liveness plus a 120 s grace window plus a 12 h TTL to compensate — which still opened **eight
panels for four arms** when a batch was relaunched inside the TTL, and separately showed **3 of 4** when
a wave's `exec` landed after the scan. A pid the process wrote about itself has neither hole: there is
no interval where the entry exists and the process does not, so there is no grace period to tune, and
`os.kill(pid, 0)` on a pid handed to you cannot match the wrong thing the way a `pgrep` pattern can
(including, famously, its own command line). Dead entries are dropped on the next read, which is also
why a trainer needs no `finally` and a `kill -9` cleans up after itself.

The window remains **disposable**: its own session, never read from, never waited on, never reopened by
a run. That is what makes it safe to kill and relaunch one while four arms are training — the property
snek2 could not offer, having lost all four arms of a batch to one XIO error in the trainer's own
canvas.

### The documented way to confirm an arm's config was blind to the shaping knobs

`grep 'hyperparameter override:'` is what both instruction files name as the check that an arm got
its config, and it reports only what `train.py` reads through `tuned()`. The reward and shaping knobs
are read by `env/constants.py` **at import** — before the trainer's config object exists — so they
print nothing. The blind set is precisely `CHASE_SAFE_SHAPING`, `CHASE_SAFE_GATE`, `FREE_SPACE_*`,
`FOOD_DISTANCE_REWARD`, `PERFECT_GAME_REWARD` and `ZERO_OBS`: the settings a shaping batch exists to
test.

Found while launching b2, whose whole purpose is `c=0.10` at gate 75. Its four logs listed seven
overrides and **not one of the three shaping values**, so the only way to confirm the batch was
running the config it was queued with was `sudo tr '\0' '\n' < /proc/<pid>/environ` on the box.

`train.py` now prints `vectorized/config.describe()` at startup, which already existed for eval
report headers and names every one of them:

    reward config: grid 10x10, max score 95, food 1.0, death -5.0, starve -0.5, perfect 100.0,
                   dist 0.0, chase_safe c=0.1 gate=75, free_space c=0.0 gate=85

The fixture is parametrised over the knobs `env/constants.py` reads and asserts each one **changes
the line**, rather than that its name appears in it — `describe()` prints `dist`, `perfect` and
`gate=`, so a name match would have failed while the code was right, which is this project's
own fixture trap.

### A test can pin the wrong contract, and 53 green fixtures did

The desktop daemon built its stage-B command as `evaluate.py <arg…> <policy> <policy> <policy>
--selector screen:95 --episodes 500 --shards 16`. `evaluate.py`'s real signature is
`evaluate.py <policy> [selector]` — **one** policy, and the selector **positional**. Every wave the
box could ever dispatch was going to exit 2 with `unrecognized arguments`, and the first one did.

`tests/test_desktop_runner.py` had four fixtures over `build_command`, all green. They asserted the
argv I believed in. Nothing compared it to the parser that has to accept it, so the suite pinned my
assumption and the mutation run confirmed the fixtures were sensitive to *changing* the assumption.
This is the sibling of "a fixture whose subject cannot violate it": here the subject could violate
it, but the fixture was pointed at the wrong subject.

**The check that would have caught it is one line** — hand the built argv to the real parser and
assert it parses:

    argv, *_ = launch.build_command(job, HOST, runtime())
    evaluate.build_parser().parse_args(argv[3:])      # raises SystemExit if the daemon is wrong

The daemon cannot import `evaluate` at runtime — it runs on base python before the conda env exists —
but a **test** runs in the env and can. The constraint that made the duplication necessary does not
extend to the fixture that guards it.

What kept this from costing the batch is the `attention` list, which is in the daemon for the snek2
incident where a failed eval was never retried and never surfaced: `** b1-stageb failed and will NOT
be retried automatically (rc=2)`. Built for one incident, caught a different one.

### Every arm of batch b1 was still improving at its step cap

All four seeds of the DDQN baseline ran 3M steps and **none plateaued** — b1a's trailing perfect rate
went ~20% at 500k to ~40% at 3M, b1d's 0% to ~80%, monotonically, with b1d's best band in its final
500k. So 3M steps measures how fast this config climbs, not what it converges to, and no verdict
about the learning code can be read off it.

Two things follow. A step cap is a *measurement choice* and has to be justified against the curve it
truncates; snek2's records came from 2.00M-step arms **with chase-safe shaping**, which is what made
that cap enough there. And the seed spread — 42.1% to 81.9% peak, **39.8 pp** — is far wider than the
~10 pp this project already says n=4 cannot resolve, so at this horizon the four arms cannot rank two
configs at all.

### The training loop's throughput ceiling is 1,600 steps/s, and two of the plan's claims about it were wrong

**Measured 2026-08-28 on the laptop, self-eval off, `fc_layer_params=(320,)`, batch 128, one torch
thread.** Agent steps/s:

| lanes (`SNEK_COLLECT_ENVS`) | ratio 1.0 | ratio 0.5 | ratio 0.25 |
|---:|---:|---:|---:|
| 1 | **809** | | |
| 16 | **1,512** | | |
| 64 | **1,587** | 2,280 | 3,703 |

**‡ The table's unit is transitions/s, not counted steps/s.** At 64 lanes a counted step banks 64
transitions, so 1,587 counted steps/s would be 101k gradient steps/s — 50x the measured ceiling. Read
every row as transitions/s, and see "Raising `SNEK_COLLECT_ENVS`..." above for why the same numbers say
the opposite about an arm's wall clock.

**Retraction 1: raising `SNEK_COLLECT_ENVS` does not "buy nothing".** It buys 1.9x. The plan reasoned
that the gradient work scales with the lane count so nothing is gained — true of the gradient half,
but the *env* half does not scale: `VecSnake.step` costs **536 us at one lane and 950 us at 64**,
because almost all of it is per-call numpy overhead rather than per-lane work. At one lane the env is
0.5 ms of every agent step; at 64 lanes it is 0.015 ms. The curve flattens at ~1,600 because the
gradient half then dominates, which is the half the plan's reasoning applied to.

**Retraction 2: the ratio-1.0 ceiling is ~1,250/s, not ~4,000/s.** A whole learn step is **802 us**,
against the 245 us an isolated `agent.update` benchmark predicted — `agent.update` 514 us,
`buffer.update_priorities` 147 us, `buffer.sample` 71 us. At ratio 1.0 one agent step *is* one learn
step, so the isolated gradient benchmark was measuring a third of the real cost.

**Two optimisations that do not work here, recorded so they are not retried.** `torch.compile` is
*slower* — 1,643/s against 2,001/s eager, because a 30 -> 320 -> 3 net has no kernel worth fusing and
the guard overhead dominates. So is more than one torch thread: **950 gradient steps/s at 10 threads
against 1,314 at one**, every op being far too small to amortise a fork-join. Hence
`SNEK_TORCH_THREADS=1` as the default, which matters more on the laptop where four arms run at once.

**And the conclusion that actually matters: none of this moves an arm's wall clock much.** Stage A is
5.0 h of a 3M-step arm; training is 62 min at 809 steps/s and 32 min at 1,587. So a 2x throughput win
takes an arm from ~6.0 h to ~5.5 h — **8%**. Training throughput is worth having because it makes
smoke tests and short experiments fast, not because it shortens an arm. The hours are in the eval.

### Deduplicating a sum tree's parents costs more than it saves

**0.167 ms per batch-128 priority update with `np.unique` per level, 0.067 ms without — 2.5x, and it
was 18% of a whole gradient step.** A batch of 128 leaves shares ancestors near the root, so
deduplicating each level looks like the obvious saving. It is not: every duplicate entry reads the
same two children and computes the same sum, so the repeated scatter writes are **idempotent** and
uniqueness buys only a shorter array — while costing 17 `np.unique` sorts. The arrays are small enough
that the sorts dominate.

`tests/test_replay.py` pins the repair against a one-leaf-at-a-time walk to the root, because this is
the rare case where a mutation test cannot help: deduplicating is *equivalent*, so no assertion can
distinguish it, and the mutation correctly survives.

### The flat one-stage protocol reproduces snek2's tiered close-out, row for row

**3,222 checkpoints of `b45a-lowlr8-b29b`, 100 episodes each, measured independently by both stacks.**
snek2's number is its own `_checkpoint_evals_vec.json`; snek3's is a four-shard stage-B wave over the
same explicit step list. A second snek3 seed was added to answer a question the first one raised.

| | ==100% | ≥99% | ≥98% | ≥95% | pooled |
|---|---:|---:|---:|---:|---:|
| snek3, seed 0 | 187 | 752 | 1,576 | 3,052 | 97.287% |
| snek3, seed 1 | 222 | 809 | 1,584 | 3,055 | 97.318% |
| snek2 | 239 | 797 | 1,568 | 3,026 | 97.291% |

**The agreement is as close as sampling allows.** Mean per-row difference −0.004 pp against a standard
error of 0.041 pp, so **0.09 SEs**; per-row spread 2.30 pp observed against 2.30 pp predicted by
sampling alone, a ratio of **1.00**. Nothing is left over for an implementation difference to live in.
Seed 1 gives +0.028 pp, 0.68 SEs.

Together with the exact-conversion finding below, phase 2's gate is met and **the flat protocol can be
trusted to replace the tiered one it was designed to delete.**

### A count of rows above a threshold is not a stable statistic, even at a fixed depth

Found while closing the A/B, and it nearly became a false alarm. Seed 0 produced **187** rows at
exactly 100/100 against snek2's 239 — a McNemar z of **−2.59**, p≈0.01, the only one of four
thresholds that was not flat. The mean rate could not explain it: making the 100/100 count fall by 24%
takes a uniform rate drop of **−0.24 pp**, which is 6 standard errors from the −0.004 pp measured.

Settling it took one more 16-minute wave. **Two seeds of the same code disagree almost as much**: seed
0 against seed 1 is z = −1.81 (187 vs 222), and seed 1 against snek2 is z = −0.82. So the −2.59 was a
food stream, not a stack.

Two rules follow, and the second is the one that matters.

- **Never conclude from a tail count what the mean contradicts.** `P(100/100) = q¹⁰⁰`, so
  `d ln P / d q = 100/q`: the count amplifies a rate difference ~100x, which makes it a *sensitive*
  statistic and an *unstable* one at the same time. When the two disagree, the mean is the one with
  the smaller variance.
- **The same hazard applies to the widest ≥98% run**, which `tools/stage_b_chart.py` reports and
  [`protocol.md`](protocol.md) asks a comparison to lead with. It is a run-length statistic over
  threshold crossings, so it is depth-sensitive exactly as [`invariants.md`](invariants.md) invariant
  8 describes. On `b45a` at 100 episodes the widest run is **9**, which says nothing about the arm; at
  the 500 episodes stage B actually runs, the per-row sd is 0.7 pp instead of 1.6 and the number
  begins to mean something. **Do not compare a region width across two different episode counts.**

### Stage A costs 5.3 h an arm, not 1.85 h, and the cause is lane drain rather than episode count

**Measured, same arm, same 322,200 episodes, three ways.** Stage A's shape — one checkpoint, 100
episodes, one process — runs at **16.9 episodes/s**. The identical work streamed through
`engine.measure_stream`, which refills lanes from the *next* checkpoint, runs at **96 episodes/s per
shard**.

| how the same 3,222 x 100 episodes are measured | wall clock |
|---|---:|
| one checkpoint at a time, one process — **this is stage A** | **5.30 h** |
| streamed, one process | 0.93 h |
| streamed, 4 shards — this is a stage-B wave | 0.23 h |

**A 5.7x tax, and it is structural.** A single checkpoint's measurement has nothing to refill lanes
with: 100 episodes start together and the batch drains toward width 1 as they finish, so the last few
episodes carry the full per-step numpy cost alone. `engine.measure` says so in a comment and is
correct to — the drain is inherent to measuring one checkpoint, not a defect.

**‡ The 5.3 h is confirmed and the "90% of wall clock" is not.** Measured on b2a from a single
source, 2026-08-29: 533,000 counted steps in 5,192 s wall = 102.7 st/s, against the log's own
training-only 299 st/s. That is 5.44 ms of stage A against 3.34 ms of training per step, so a 3M-step
arm is **8.1 h — stage A 5.33 h (66%), training 2.79 h (34%)**. The 5.33 h independently reproduces
the 5.30 h above; only the share was wrong, and it was wrong because the training half had never been
measured on the same arm at the same time.

**One trap in reading that.** The desktop's `status.json` `steps_per_sec` is a **wall-clock** rate — it
is a step delta over a real-time delta, so it includes stage A — while `runs/<arm>_evals.json`'s
`steps_per_second` excludes it. The two differ by 3x on a healthy arm and neither is labelled.

**This corrects the plan and this file's own arithmetic.** [`../plans/pytorch-port.md`](../plans/pytorch-port.md)
§6 estimated stage A at 1.85 h from snek2's ~45 episodes/s and concluded "an arm is ~2 h and stage A
is ~90% of it". The 90% share was right and the total was not: stage A alone is ~5.3 h.

**What follows is a design change, and the mechanism is not the one the backlog assumed.** The
backlog's "asynchronous self-eval" is filed as an 8x from overlapping eval with training. The
measurement says most of the win is from **keeping the lanes full**, which does not require asynchrony
at all: holding K pending checkpoints and measuring them in one `measure_stream` call with
`max_live=K` recovers ~5.7x on its own. Asynchrony then removes what remains. Either way the cost is
the same and it is a real one — the epsilon refinement schedule reads `perfect_percent`
([`invariants.md`](invariants.md) invariant 2), so its feedback would lag by up to K intervals. That
is a change to the training, and it should be pre-registered rather than slipped in as a speed-up.

### The port is faithful at the level of the policy, not just of the win rate

**A snek2 checkpoint converted to torch computes the same function, to float32.** Over 12,864 states
drawn from a seeded random rollout, the TF and torch networks' Q-values differ by at most **2.7e-5**
on values of magnitude ~30.6 — a relative error of ~1e-6, which is accumulation order — and the
**argmax is identical on all 12,864**. Measured 2026-08-28 on
`b44a-lowlr7-b29b-ckpt2739000`.

This matters more than the 98.8%/3,000 that followed it. A win rate is a noisy end-to-end number: at
n=3,000 and ~99% perfect it can only bound a systematic difference to a few tenths of a point, so
agreeing with snek2 there is consistent with a real divergence somewhere. Agreeing on every argmax
is not — it means the observation vector, the network and the weight layout are all right, and
therefore that **any future disagreement is in the environment or the RNG, not in the port.**

Kept as a finding rather than a test because it needs both conda envs at once: TensorFlow lives in
`snek` and torch in `snek3`, so nothing in `tests/` can assert it. What `tests/test_net.py` does
assert is the transpose itself, against an independent numpy forward pass.

## Falsified

### Two hidden layers beat every single-layer width — n=1, reversed at n=4

b3 ranked `fc 300,100` (9.0%) > `fc 200,100` (7.9%) > `fc 320` (5.6%) on ≥98%/500 density at one seed
each. b7 ran the same axis at four seeds and 50M: `fc 320` **17.3%**, `fc 200,100` 11.8%,
`fc 300,100` 6.8% — the order of all three reversed, and `fc 320` separates completely from five of
the seven other layouts. What replaced it is at the top of `Established`.

### b6 leads b5 on record density — a 500-episode reading that 5,000 episodes erased

Published as "12.8% against 9.6%" in `results.md`, this file and `plans/ppo.md` §6e. Re-measured at
5,000 episodes the two batches have identical means (97.80), ≥98 rates within 0.1 pp, and b5 ahead on
champion-level rows (29 vs 20) and on the top checkpoint (99.20 vs 99.10). The b4 half of the same
comparison survived. See `Established`.

---

**Format.** One `###` per finding, leading with what is now believed and the measurement that
supports it, then what it replaced. Mark a retraction rather than deleting the section it replaces —
snek2 overturned several of its own findings and each retraction was worth more than the section it
replaced.
