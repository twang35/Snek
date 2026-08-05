# Archive — history only, do not read into context

Content moved out of the tuning docs on 2026-08-04, when they had grown to ~4,600 lines and
reading them was costing more than the information was worth.

**Nothing here should be loaded into context during normal work.** These files exist so that a
number cited in the live docs can be traced back to the arm and the batch that produced it, and so
that a closed question can be reopened with its original evidence intact. They are not a summary
and they are not current: several conclusions in them were later overturned, and the live docs are
the only place that reflects which.

| file | what is in it |
|---|---|
| [`batches1-8.md`](batches1-8.md) | per-batch write-ups and chart captions for batches 1-8 |
| [`findings-superseded.md`](findings-superseded.md) | findings about replaced observation vectors, and batch 1-8 config results later settled |

## What was kept live instead

Everything still load-bearing was **condensed** into the live docs rather than moved, so the
archive is not a superset of them:

- The **one-row-per-arm ranking for every arm ever run**, including batches 1-8, stays in
  [`../completedRuns.md`](../completedRuns.md). That table is canonical and dense; nothing was
  removed from it.
- Findings that still constrain decisions were rewritten shorter in
  [`../findings.md`](../findings.md) — the discount sweep, the `td_loss` exponent mechanism, how to
  read graph evals, and the measurement caveats. Read those, not these.
- Engineering facts that any session needs (rendering cost, worker counts, the audio trap) live in
  [`../../../CLAUDE.md`](../../../CLAUDE.md).

## Why these were chosen

Batches 1-8 ran on observation vectors this project has since replaced four times — 20, 21, 23 and
26 values against the current 30. Their *absolute numbers are not comparable to anything current*
(the same checkpoint that scored 92% before the 2026-08-01 audit reads 73% after), so the detail
behind them cannot inform a decision today. What survives from them is a handful of conclusions
about hyperparameters, and those were carried forward.

The exception worth knowing about: **the reasoning in an archived section is sometimes the reason a
current default exists.** `td_loss` + alpha 0.6 is the default because of the effective-exponent
derivation, and `DISCOUNT=0.995` because of a survival argument priced across seeds. Both have
condensed versions in the live `findings.md`; come here for the tables behind them.
