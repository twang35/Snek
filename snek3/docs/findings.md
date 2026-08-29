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
`runs/.live/<policy>` holding **its own pid** before its first step, then calls `ensure()`: the first
arm opens the window, the rest get `None`, and panels appear as arms start without anything being
reopened. It closes itself five minutes after the last arm goes.

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

## Falsified

*Nothing yet.*

---

**Format.** One `###` per finding, leading with what is now believed and the measurement that
supports it, then what it replaced. Mark a retraction rather than deleting the section it replaces —
snek2 overturned several of its own findings and each retraction was worth more than the section it
replaced.
