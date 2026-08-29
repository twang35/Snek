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

## Falsified

*Nothing yet.*

---

**Format.** One `###` per finding, leading with what is now believed and the measurement that
supports it, then what it replaced. Mark a retraction rather than deleting the section it replaces —
snek2 overturned several of its own findings and each retraction was worth more than the section it
replaced.
