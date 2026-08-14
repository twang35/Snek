# Prioritized-replay and endgame diagnostics

| script | args | what it answers |
|---|---|---|
| `per_priorities.py` | `<out_dir> [policy ...]` | what the PER priority signal does to the sampling distribution, and each arm's value profile against snake length |
| `point_of_no_return.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | per lost episode, the last point at which the food was still reachable — and the outcome split, which is where the starvation finding came from |
| `eat_and_survive.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | whether eating that food was **survivable**, over every route to it — the retraction of the above script's "never a dead end" reading |
| `input_sensitivity_over_time.py` | `<out.json> <policy> <steps> [boards-policy]` | how one arm's reading of a given observation input, and its greedy action, change over training — the before/during/after of a drawdown |
| `drawdown_chart.py` | `<sens_dir> <out.png>` | draws the four-panel figure from the above; no measurement of its own |
| `behaviour_profile.py` | `<out.json> <ckpt-or-policy> <episodes> <seed>` | what a checkpoint *does*: steps per meal, starve headroom, packing and realised chase-safety, by snake length — the elite-vs-mediocre comparison |
| `champion_chart.py` | `<bp_dir> <measured.json> <out.png>` | draws that comparison plus the selection-noise panels |

`point_of_no_return.py` shards across seeds like the `diagnostics/` scripts do; six processes take
six cores and ~5 minutes for 360 episodes. It **checks its own simulator against the live game on
every step** and reports `mismatches`, which must be 0 — that is the guard that makes its search
trustworthy, and it is the reason to be suspicious if a future run reports anything else.

**Its `geom` column is not a survivability test, and reading it as one was wrong for four days.**
`eat_and_survive.py` imports that script's movement rule and guard rather than copying them, so the
two cannot drift into measuring different games, and it enumerates *every* eating route because the
shortest one is often the fatal one. Its shard seeds are recorded in the payload — the original run's
were not, which is why the retraction is on 70 fresh losses rather than the same 75.

`per_priorities.py` is behind the finding in
[`../findings.md`](../findings.md#-measured-batches-19-20-compared-aggressive-per-against-uniform-replay)
that batches 19 and 20 compared aggressive PER against *uniform replay* rather than against
standard PER.

`input_sensitivity_over_time.py` is behind the drawdown result in
[`../findings.md`](../findings.md#-falsified-a-drawdown-is-not-how-a-policy-escapes-a-local-minimum),
and it is what demoted the chase-flag mechanism from a cause to a marker. Two things it needs to be
used honestly:

- **The board set must be held fixed across the whole ladder**, which is why it defaults to a
  finished arm's buffer rather than the arm being measured. It also has to span the **whole game**:
  an endgame-only board set read 99% action agreement straight through a collapse from score 94 to 4,
  which says only that the collapse was not in the endgame.
- **Churn is only comparable at equal step gaps.** Agreement saturates, so dividing `1 - agree` by
  the gap flatters long gaps — and `SNEK_MIN_CHECKPOINT_SCORE` makes the gaps long exactly inside a
  trough. Filter to one gap width before comparing windows.

`behaviour_profile.py` runs **every checkpoint on the same seeds on purpose**. A greedy policy plus a
seed reproduces the same food sequence, so a set of checkpoints sharing seeds faces the same games and
the comparison between them is paired — game-set difficulty cancels exactly. The cost is that the
*absolute* rate carries that game set's difficulty: seeds 21,22 turned out to run ~1-2 pp hard against
seeds 31,32. **So compare columns freely, and cross-check on a second seed pair before quoting any
single number as a policy's rate.**

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
