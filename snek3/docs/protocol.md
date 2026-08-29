# Protocol — how a run is measured and judged

**Phase 0 note: the eval code does not exist yet.** This file is the decided design
([`../plans/pytorch-port.md`](../plans/pytorch-port.md) §6), written down now so the implementation
has a specification to match. It becomes a description rather than a plan at phase 2.

## Two stages, and only two

| stage | who | selection | episodes | writes |
|---|---|---|---:|---|
| **A** | the trainer, in-process, every 1,000 steps | every checkpoint | 100 | `runs/<policy>_evals.json`, `runs/<policy>.png` |
| **B** | a wave of shard processes, after the arm stops | every checkpoint at **≥95/100** in stage A | 500 | `runs/<policy>_checkpoint_evals.json`, `evals/<policy>_eval_progress.png` |

**Stage B is the hall-of-fame measurement.** There is no third stage. snek2 ran three — a graph eval,
a tiered close-out at 100 episodes, then a 500-episode re-measure of the close-out's ≥98% rows — and
the middle one re-measured the same checkpoints at the same depth as the first.

**Every row in a stage-B file is full length and directly comparable.** No tiered selection, no
screen/confirm split, no min-achievable gate, so no `abandoned` rows that are shorter than their
neighbours and no `min_achievable` to read out of the payload before pooling anything. That was not
true of snek2's files, which have four gate eras.

### What each stage costs

For a 3M-step champion-class arm, ~3,000 checkpoints, at the measured ~45 episodes/s per process:

| | episodes | 1 process | 4 shards | 16 shards |
|---|---:|---:|---:|---:|
| stage A (inside training) | 300,000 | 1.85 h | — | — |
| stage B, ~50% clear ≥95/100 | 750,000 | 4.6 h | 1.2 h | 18 min |
| stage B, ~70% clear | 1,050,000 | 6.5 h | 1.6 h | 25 min |

So **an arm is ~2 h of wall clock and stage A is ~90% of it.** That is the cost of the protocol
rather than waste: snek2 paid for those 300,000 episodes and then paid again to re-measure.

**Never read a stage-B wave still running after several hours as hung.** Check the two counts first —
when `grep -c "episodes in"` runs ahead of `grep -cE "^\[ *[0-9]+/"`, the controller is the
bottleneck and not the box. All shards idle at 0% while one process burns a core is the signature.

## Reading an arm

Read status from `runs/<policy>_evals.json`'s precomputed `summary` block, never from the log.

| field | meaning |
|---|---|
| `strong_eval_fraction` | **the primary metric.** Share of the arm's stage-A evals at ≥80% perfect. Lowest between-seed variance of the candidates |
| `best_perfect30` | best 30-eval trailing perfect rate. Kept for continuity |
| `peak_trailing` | best trailing average score |
| `zero_since` | the **current** unbroken near-zero stretch, or absent if the latest eval is above threshold |
| `dead_since` | the **earliest** sustained-zero stretch. History, not status |
| `max_single_eval`, `recent_perfect30`, `trailing_now`, `epsilon`, `step`, `evals` | |

Three cautions:

- **`strong_eval_fraction` is a fraction of each arm's own evals**, so compare only at a common step
  horizon.
- **Use `zero_since`, not `dead_since`, to ask whether an arm is dead now**, and neither is a verdict:
  a snek2 arm carried `dead_since=275000` and went on to a 36% best-30 window, and another recovered
  from **1.2M steps** near zero to 63.7 trailing before collapsing for good. Only call an arm dead
  after hundreds of thousands of steps pinned there *and* no recovery arc in progress.
- **`strong_eval_fraction` is a threshold-crossing statistic**, so it is not comparable across a
  change in episodes per eval. snek3 runs 100 throughout, so this only bites against snek2 numbers —
  see [`invariants.md`](invariants.md) invariant 8.

## Judging a batch

**A batch is four seed-matched arms.** One arm answers nothing: the same config has produced 62.5 and
18.0, and **n=4 cannot resolve an effect below ~10 pp**. Lead with the sign test across seeds, not
with a mean of four noisy numbers.

For a stage-B comparison, lead with **the ≥98%/500 count** — the width of the record region — rather
than the best row. The best row is a selected high; snek2's 99.0%/500 champion re-measured at
97.5%/1000 and its four best hall-of-fame entries fell a mean 1.4 pp.

**A record claim needs a fresh measurement of the single winner at 1,000+ episodes.** Stage B ranks
candidates; it does not certify one.

## Stopping an arm

**A trainer may not stop on SIGTERM — use `kill -9`, and verify.** Every durable file is written
`.partial` then `os.replace`d and checkpoints land every 1,000 steps, so `kill -9` is safe. **Do not
test liveness with `kill -0`** — it succeeds on a zombie; read `ps -o stat=`. On the desktop, **pause
the queue before killing** or the freed slots refill within one poll.

**When arms are stopped, update [`charts.md`](charts.md) and [`results.md`](results.md) in the same
pass**, and move the batch's rationale out of [`runs.md`](runs.md). Without the rationale a future
session cannot tell a surprising result from an arm that was never going to answer anything.
