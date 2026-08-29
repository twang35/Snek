# Protocol — how a run is measured and judged

**Stage B is built as of phase 2** ([`../plans/pytorch-port.md`](../plans/pytorch-port.md) §6).
Stage A arrives with the trainer in phase 3, so the columns below that name it are still a
specification.

```
PYTHONPATH=. python -u evaluate.py <policy>                     # screen:95, 500 episodes, 4 shards
PYTHONPATH=. python -u evaluate.py <policy> screen:98 --shards 8
PYTHONPATH=. python -u evaluate.py <policy> one --episodes 1000 # a record re-measure, one process
PYTHONPATH=. python -m tools.compare_results <file-a> <file-b>  # is the gap noise?
PYTHONPATH=. python -m tools.progress_chart <policy>            # redraw runs/<policy>.png
PYTHONPATH=. python -m tools.stage_b_chart <policy> --label L   # the pass, drawn and summarised
PYTHONPATH=. python -m tools.chart_viewer --glob 'runs/b48*.png'  # a live window
```

## Two stages, and only two

| stage | who | selection | episodes | writes |
|---|---|---|---:|---|
| **A** | the trainer, in-process, every 1,000 steps | every checkpoint | 100 | `runs/<policy>_evals.json`, `runs/<policy>.png` |
| **B** | a wave of shard processes, after the arm stops | every checkpoint at **≥95/100** in stage A | 500 | `runs/<policy>_checkpoint_evals[_<label>].json` |

**Stage B is the hall-of-fame measurement.** There is no third stage. snek2 ran three — a graph eval,
a tiered close-out at 100 episodes, then a 500-episode re-measure of the close-out's ≥98% rows — and
the middle one re-measured the same checkpoints at the same depth as the first.

**Every row in a stage-B file is full length and directly comparable.** No tiered selection, no
screen/confirm split, no min-achievable gate, so no `abandoned` rows that are shorter than their
neighbours and no `min_achievable` to read out of the payload before pooling anything. That was not
true of snek2's files, which have four gate eras.

### What each stage costs

Measured on the laptop over `b45a`'s 3,222 checkpoints, which is a champion-class arm's full history.
**The two stages run at very different episode rates, and the reason is not the episode count.**

| | rate | why |
|---|---:|---|
| stage A — one checkpoint, 100 episodes, in-process | **16.9 ep/s** | 100 episodes start together and the batch drains toward width 1 as they finish. Nothing refills the lanes, because there is only one policy |
| stage B — a shard streaming through a step list | **96 ep/s** | `engine.measure_stream` refills a finished lane from the *next* checkpoint, so the batch never drains |

For a 3M-step arm, ~3,000 checkpoints:

| | episodes | 1 process | 4 shards |
|---|---:|---:|---:|
| stage A (inside training) | 300,000 | **5.0 h** | — |
| stage B, ~50% clear ≥95/100 | 750,000 | 2.2 h | 33 min |
| stage B, ~70% clear | 1,050,000 | 3.0 h | 46 min |

So **stage A is ~90% of an arm's measurement cost and an arm is ~6 h, not ~2 h.** An earlier version
of this table read 1.85 h for stage A, from snek2's ~45 ep/s; the share was right and the total was
not. See [`findings.md`](findings.md) — the 5.7x gap between the two rows above is lane drain, it is
structural to measuring one checkpoint, and closing it means batching or detaching the self-eval,
which changes the epsilon schedule's feedback lag rather than being free.

**A wave's shards are independent processes with independent output files, and the wave itself does
no per-episode work** — so unlike snek2's, it cannot become the bottleneck. snek2's controller banked
every lane's episodes and re-serialised the result file 125 times per measurement, 58 s of
single-threaded bookkeeping against 46 s of lane work, and folded a 90-minute backlog with 16 workers
idle. Here progress is *read* off the shard files. If a wave looks stalled, check
`logs/<pass>-s<i>of<n>.log`: the per-row lines there are the ground truth, and the row counts the
wave prints come straight off disk.

**A killed wave loses nothing.** Each shard rewrites its own file after every completed measurement,
and rerunning the same command resumes every shard from where it stopped.

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

**Two independent measurements of the same policy disagree by more than people expect.** At 100
episodes and p≈0.975 the standard deviation of the *difference* is 2.2 pp, so per-row gaps of 3 or 4
points are what agreement looks like; what distinguishes agreement from a real effect is the mean
across rows, whose standard error shrinks as `sd / sqrt(rows)`. `tools/compare_results.py` reports
both and the ratio of observed to predicted spread. Over 3,222 rows that ratio came out **1.00** for
two stacks measuring the same checkpoints — the whole spread was sampling.

**Lead with the mean, and distrust a tail count that disagrees with it.** The count of rows at exactly
100/100 amplifies a rate difference about a hundredfold, which makes it sensitive *and* unstable: two
seeds of the same code gave 187 and 222 of 3,222. A gap there with no matching gap in the mean is a
food stream. See [`findings.md`](findings.md).

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
