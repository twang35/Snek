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

*Nothing yet.*

---

**Format.** One `###` per finding, leading with what is now believed and the measurement that
supports it, then what it replaced. Mark a retraction rather than deleting the section it replaces —
snek2 overturned several of its own findings and each retraction was worth more than the section it
replaced.
