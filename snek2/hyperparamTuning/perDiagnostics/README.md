# Prioritized-replay and endgame diagnostics

| script | args | what it answers |
|---|---|---|
| `per_priorities.py` | `<out_dir> [policy ...]` | what the PER priority signal does to the sampling distribution, and each arm's value profile against snake length |
| `point_of_no_return.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | per lost episode, the last point at which the food was still reachable — and the outcome split, which is where the starvation finding came from |

`point_of_no_return.py` shards across seeds like the `diagnostics/` scripts do; six processes take
six cores and ~5 minutes for 360 episodes. It **checks its own simulator against the live game on
every step** and reports `mismatches`, which must be 0 — that is the guard that makes its search
trustworthy, and it is the reason to be suspicious if a future run reports anything else.

`per_priorities.py` is behind the finding in
[`../findings.md`](../findings.md#-measured-batches-19-20-compared-aggressive-per-against-uniform-replay)
that batches 19 and 20 compared aggressive PER against *uniform replay* rather than against
standard PER.

Kept separate from [`../diagnostics/`](../diagnostics/), which is frozen alongside
`claudeFeatureRecommendations.md` and is about the observation vector. This one is **not frozen** —
it takes policy names as arguments and is meant to be re-run on new arms.

```
cd snek2
PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -u \
    hyperparamTuning/perDiagnostics/per_priorities.py <out_dir> [policy ...]
```

With no policies it does batch 18 against batch 20 wave 1, the seed-matched pair the finding rests
on. ~4 minutes for 8 arms. Writes `<policy>_per.npz` per arm, a `summary.json`, and
`per-priorities.png`; the committed copy of that chart is
[`../charts/per-b18-vs-b20-priorities.png`](../charts/per-b18-vs-b20-priorities.png).

**Read-only with respect to `savedPolicies/` and `runs/`.** It restores checkpoints and never
writes one, and it starts no eval, so it cannot displace `evals/` charts and is safe beside a live
arm.

## Two things to know before trusting the output

**Priorities are not recoverable from a saved buffer.** `cpprb.save_transitions()` keeps the
transitions and resets every priority to the max, so the script recomputes them: it restores the
arm's own checkpoint and runs the arm's own buffer back through the same `DdqnAgent` loss that
`training.py` calls, reading `extra.td_error` and `extra.td_loss` — the exact two tensors
`SNEK_PRIORITY_SIGNAL` chooses between. The real in-buffer priorities were **staler** than these,
because a transition's priority only refreshes when it is sampled. So the concentration figures are
the sharpest the config could be, not exactly what it was.

**`SNEK_FC_LAYERS` must match the arms being measured.** The script defaults it to `50,100,50`,
which covers batches 1-19 and batch 20 wave 1 only. A wrong width restores **silently** under
`expect_partial()` and yields a network that is partly random — pass the right value for anything
from batch 20 wave 2 onward, and do not mix widths in one invocation.

## The checks that are load-bearing

`summarise()` reports a **top-1000 Jaccard between the two signals' rankings, which must be
1.0000**. Huber is strictly monotone in `|td_error|`, so the two orderings are identical by
construction; anything below 1.0 means the network loss is no longer element-wise Huber and every
conclusion in the findings section is stale.

`is_flattening_check()` measures **realised** exposure from actual cpprb draws instead of trusting
the `raw^(α(1−β))` algebra, because that algebra runs through cpprb's C++ weights and
`normalize_is_weights`, neither of which is in it. It reports a same-effort uniform noise floor
alongside, since a finite number of draws never reads as perfectly uniform — without the floor the
β=1.0 row looks like residual prioritization when most of it is sampling noise. It did find one real
residue: the batch-mean normalisation leaves more concentration behind the sharper the priorities
are.
