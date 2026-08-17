# Prioritized-replay and endgame diagnostics

| script | args | what it answers |
|---|---|---|
| `per_priorities.py` | `<out_dir> [policy ...]` | what the PER priority signal does to the sampling distribution, and each arm's value profile against snake length |
| `point_of_no_return.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | per lost episode, the last point at which the food was still reachable — and the outcome split, which is where the starvation finding came from |
| `eat_and_survive.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | whether eating that food was **survivable**, over every route to it — the retraction of the above script's "never a dead end" reading |
| `endgame_packing.py` | `<policy-or-ckpt> <episodes> <seed> <out.json>` | how fragmented the free space is per meal from length 80 up, and whether the food that spawned could be eaten safely — the packing finding |
| `input_sensitivity_over_time.py` | `<out.json> <policy> <steps> [boards-policy]` | how one arm's reading of a given observation input, and its greedy action, change over training — the before/during/after of a drawdown |
| `drawdown_chart.py` | `<sens_dir> <out.png>` | draws the four-panel figure from the above; no measurement of its own |
| `behaviour_profile.py` | `<out.json> <ckpt-or-policy> <episodes> <seed>` | what a checkpoint *does*: steps per meal, starve headroom, packing and realised chase-safety, by snake length — the elite-vs-mediocre comparison |
| `champion_chart.py` | `<bp_dir> <measured.json> <out.png>` | draws that comparison plus the selection-noise panels |
| `chase_safe_potential.py` | `<out.json> <ckpt-or-policy> <episodes-per-seed> <seed[,seed...]>` | how often "head, food and tail in one region" **flips**, per length band — the Phase 0 calibration for `CHASE_SAFE_SHAPING`'s `c`, and the step/flip shares that decide whether to length-gate it |
| `gate_behavior.py` | `<out.json> <ckpt-or-policy> <episodes> <seed> <gate>` | board-state metametrics per snake length — φ, region count, one-piece packing, isolated pockets/cells, food reachability, starve headroom — plus per-episode outcome and full per-step arrays, so a seed-matched pair can be sliced before/after each arm's gate. The gate-75-vs-gate-85 (b29 vs b27) analysis: gate 75 keeps the board healthier at **every** length, and failures in both arrive at the gate already fragmented |
| `plasticity.py` | `<out.json> <policy> [stride] [extra] [boards]` | the three published loss-of-plasticity signatures against step — dormant units, feature rank, weight norm — each against a fresh net of the same shape |
| `plasticity_probe.py` | `<out.json> <policy> [stride] [extra] [boards]` | whether the checkpoint can still **fit a new target**, which is the question the signatures are only correlates of |
| `plasticity_analysis.py` | `<payload_dir> [out.png\|-] [probe_dir]` | the tables and figure from both: control→peak→end, drawdown events, flat stretches, early-vs-late, and the paired probe trend |
| `return_distribution.py` | `<out.json> <ckpt-or-policy> <episodes-per-seed> <seed[,seed...]>` | the distribution of the **discounted return from each visited state**, by outcome and by length band — the Phase 0 measurement that sizes C51's `[v_min, v_max]` grid and `num_atoms` |
| `c51_stability.py` | `--policy <name> [--policy ...] [--states N] [--points K] [--stride S] [--end STEP] [--states-from <policy>]` | whether a chaotic eval curve is a **policy that keeps changing its mind** or a stable policy seen through a noisy 10-episode sample — greedy-action churn on a fixed state set, the action gap, how far the value function moves per `stride` steps, and (c51 only) boundary-atom mass and effective atom count. **`--states-from` is mandatory for any cross-arm reading**; see below |
| `init_optimism.py` | `--policy <name> [--policy ...] [--states-from <policy>] [--states N] [--points K] [--tolerance R]` | **where a value head starts and how long it takes to come down** — a fresh net's expected Q with and without `SNEK_C51_ZERO_INIT`, then `V` on a fixed state set at log-spaced checkpoints with the excess over the arm's final value, alongside the arm's own `avg_score` at each rung. Reads `ddqn` arms too, and **that control is the point**: a scalar head starts at 0 and rises, so where it settles is an independent reading of the true value scale. Produced the finding that the grid midpoint costs nothing and that zero-init would be worse |
| `atom_resolution.py` | `<policy> [<policy> ...]` | whether the atom **spacing** is coarse relative to the decisions the policy makes — action-gap percentiles in reward units and in atoms, and the share of states under one atom, broken down by snake length |
| `value_by_length.py` | `<out.json> <ckpt-or-policy> <states> <max-episodes>` | what the network **believes** a state is worth by length band, in `return_distribution.py`'s bands. **`V − G` against realised return is the wrong comparison for a Q-learning agent** — it targets the optimal policy's value, not its own return; see the correction in the batch-33 finding. Use it for the *shape* of `V` against length |
| `endgame_gradient.py` | `<ckpt-or-policy> <states> <max-episodes> [out.json]` | whether the value function **pulls toward finishing**: `dV` per length band, the action gap the decision rests on, per-input saliency (**"is this observation ignored?"**) and per-input occupancy (**"are its weights trained at all?"**). Produced the `W > 1/(1 − γ^k)` threshold rule and found indices 18-20 dead |

`atom_resolution.py` is the range/resolution counterpart to the above: `c51_stability.py` shows the
support's *range* is right (boundary mass ~0), this asks whether its *spacing* is. Two rules for reading it,
both learned the hard way:

- **Never quote the mean.** The gap distribution is violently bimodal — median 0.28 reward units against a
  75th percentile of 42.7 — so the mean sits in neither mode. A mean-based reading is exactly what produced
  the retracted "the actions are not near-tied" claim in `findings.md`.
- **The length breakdown settles it, the pooled share does not.** 59-73% of states have a sub-atom gap, which
  sounds alarming until it is split by length: those states are early open-board play where several moves
  genuinely are interchangeable, while length 85-94 carries **25-atom** gaps. Resolution is not binding where
  games are decided, which is why shrinking `PERFECT_GAME_REWARD` to narrow the support is a bad trade — that
  endgame gap *is* the +100 being in or out of reach.

`c51_stability.py` is the script that separated C51's instability from C51 itself. It reads any arm of
either algorithm — `build_eval_agent` dispatches on `arch.json` — so a categorical arm and a scalar one
are measured by the identical code path, which is the whole point. Three things about reading it:

- **Churn is per `--stride` steps, so two arms are only comparable at the same stride**, and it falls as
  a policy converges, so `--end` exists to anchor both arms at the same *phase* rather than at the same
  distance from their cap. A tail-only reading answers a different question from the one a mid-training
  swing raises, and on this pilot the two readings disagreed.
- **Comparing arms needs `--states-from`, and the default per-arm set inflated a real result ~2×** (found
  2026-08-16). Churn is the share of a set where the argmax flips, so it depends on the margin between the
  top two actions — and that margin is **~0.2 reward units early-game against 20-24 in the endgame**. A weak
  arm dies early, so *its* set is dominated by near-tied states that flip for free. b32's `eps 1e-7` controls
  carried set mean lengths of **11.9 and 21.2** against the treated arms' 34.9-38.0, and churn, gap and `len`
  came out **rank-correlated across all six arms**. `--states-from` draws one set from a neutral third
  policy — use `hallOfFame/b29b-chase10g75seed2-ckpt1447000`, which at 1500 states has mean length **50.5**
  and spans whole games — and the effect halved from −47% to −26% but survived, 4 of 4. **The `len` column
  printed the problem all along and was read past**, which is the actual lesson.
- **`--end` does not filter the `--states-from` policy**, deliberately: it anchors the *phase of the arms
  being compared*, while the reference set is a yardstick and should come from the source's best checkpoint
  whatever phase is read. Passing it through made a champion pinned at 1447k unusable for a 600k comparison.
- **The per-arm set is still the right default for a single arm** — "how much does this arm change its mind
  on the states it actually visits" is a real question, and `len` is printed either way.
- **One arm is not a measurement.** The first run of this script compared a single C51 arm against a
  single ddqn arm and read 0.077 against 0.035; adding a second seed of each put both algorithms at
  0.033-0.058 at matched learning rate. The state collection is itself a sample.

`return_distribution.py` needs two things read before its numbers are used. It sets
**`SNEK_FOOD_DISTANCE_REWARD=0`, not the repo default of 0.001**, because every config since batch 17
does — left at the default the measured returns would carry a shaping term the arm will not have, so
every reward knob in effect is printed and stored in the payload. And it reports **several gammas from
one episode set**, which is exact rather than an approximation: the discount reaches the return only
through the recorded per-step `d`, and it changes neither the greedy action nor any reward. Its own
Phase 0 run found the returns are *not* concentrated near zero at γ=0.9975 — 60% of a champion's states
are above 25 — which falsified the premise the C51 plan's support section was built on.

## `SNEK_ALGO=c51` arms: which scripts read them

Everything that goes through `eval_agent.build_eval_agent` and only *plays* — `behaviour_profile.py`,
`chase_safe_potential.py`, `point_of_no_return.py`, `endgame_packing.py`, `eat_and_survive.py` — works
on a categorical policy with no change, because the builder reads the algorithm and the atom support
out of that policy's `arch.json` and returns the right greedy agent either way.

Three scripts read the *scalar head itself* and are refused, loudly, by
`policy_arch.refuse_categorical`:

| script | why it cannot run | port cost |
|---|---|---|
| `per_priorities.py` | every column is a scalar quantity — `abs_td_error`, Huber `td_loss`, `qmax` | rewrite the measurement as KL/CE priorities |
| `plasticity.py` | compares against a **fresh scalar net of the same shape**, and walks a plain `Sequential` | the layer walk plus a categorical fresh net |
| `plasticity_probe.py` | teacher and scratch student are both scalar nets | same |

`input_sensitivity_over_time.py` **is** ported: its `q_values`/`saliency`/`sensitivities` go through
`under_the_hood.expected_q`, which returns `sum_i z_i p_i(s, a)` for a categorical net — the same
reduction the policy's own argmax uses — and raises rather than guessing if the support is missing.
A raw logit is a float like any other, so nothing else would have surfaced the error.

`point_of_no_return.py` shards across seeds like the `diagnostics/` scripts do; six processes take
six cores and ~5 minutes for 360 episodes. It **checks its own simulator against the live game on
every step** and reports `mismatches`, which must be 0 — that is the guard that makes its search
trustworthy, and it is the reason to be suspicious if a future run reports anything else.

**A route enumeration without a node cap will take the laptop out.** The breadth-first search in
`eat_and_survive.eat_routes` keys its frontier on the **whole body**, so one key is several KB at
endgame lengths. Run per loss on a nearly-full board that is fine; `endgame_packing.py` calls it per
*step* from length 90, where ten free cells give enough branching that an uncapped search reached
**9.2 GB of RSS in 90 seconds**, and four shards at once left the machine struggling. `ROUTE_NODE_CAP`
and the tighter `SAFE_ROUTE_*` bounds exist for that, and both make a negative verdict inexact rather
than wrong — which is why `endgame_packing.py` records `spawn_exact` and reports safe-meal rates as
lower bounds. **Run these two scripts one process at a time**; sequentially, six shards of 30 episodes
take about four minutes at ~500 MB.

**Never filter to the exact rows to compute those rates.** A safe verdict returns the moment a
surviving route is found, so it is *always* exact; keeping only exact rows keeps every positive and
drops some negatives, which inflates the rate. Report over all rows with the `neg exact` share beside
it.

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

## The three plasticity scripts: four traps, all of them silent

Behind [the falsification](../findings.md#-falsified-2026-08-14-there-is-no-plasticity-loss--the-collapsed-networks-fit-a-new-target-better-than-their-own-peak).
Every one of these produced a full table of plausible numbers while being wrong:

- **The fresh-net control is the reference for everything, and it wobbled.** `tf.random.set_seed` does
  not determine an initialiser draw — the op-level seed comes off a per-process op counter — so two runs
  of the same arm read dormant 0.111 and 0.139. `plasticity.seeded_reinit` clones each layer's *own*
  initialiser config with an explicit seed, which also cannot drift from `under_the_hood.dense_layer`.
  The control is now 15 seeded draws and identical across processes; the trained rows always were.
- **Raw stable rank pins near 1.1 for a fresh net and a trained one alike**, because post-ReLU features
  carry a large DC offset that dominates σ₁. The centred ranks are the sensitive ones. Kumar et al.'s
  srank on Φ is kept beside them for comparability, not as the primary reading.
- **The weight-norm baseline has to be per-initialiser.** Hidden layers are `VarianceScaling(2, fan_in)`
  and the head is `RandomUniform(±0.03)`; one shared constant puts the head an order of magnitude below
  its own initialisation forever, so a real 30× head growth reads as 3×.
- **The probe measured output scale, not plasticity, on the first attempt** — `fit = -283`, because a
  trained net emits Q values of order 1 against a raw teacher target of variance 0.003. The target is
  standardised and the student's head zeroed so every network starts at `mse_start = 1.0` exactly.

Two things to hold onto when reading probe output. `relative` is **only comparable at equal budget**:
400 Adam steps puts a fresh net at 0.038 and the ratio at 8-9×, 2000 puts it at 0.53 and the ratio at
1.05-1.52×. And the per-checkpoint sd printed beside each row is the spread **across teachers**, which is
large and common to every checkpoint — a *change* between checkpoints is paired and must be read from
`plasticity_analysis.probe_table`, not against that sd.

`plasticity.py` accepts **any directory holding `arch.json` and `ckpt-*`**, which is how the
desktop-trained arms were measured: a 50k ladder is ~11 MB of a 527 MB policy directory. Stage those
**outside `savedPolicies/`** — a directory in there with holes reads as a real arm to every other tool.

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
