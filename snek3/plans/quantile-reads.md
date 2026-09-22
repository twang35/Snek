# Quantile reads: acting on the return distribution other than by its mean

**Status: reviewed 2026-09-22, decisions in §6. Nothing built, nothing queued.**

Group A's distributional rungs all act greedily on the mean of the quantiles (`algos/dist/net.py`,
`greedy_policy_fn`), and every stage-B number in `docs/results.md` is that read. The distribution is
there in the head and nothing but the CVaR read of b38 has ever acted on it. This pass takes the best
checkpoints of the four quantile cells and re-measures each one under a family of alternative reads, so
the question "does the tail carry information the mean throws away" gets a paired answer on the same
weights. It is `a-return-tail.md` §5's second-greedy-policy mechanism, broadened.

**Cells**: b37 `qrdqnlocal` (e-h), b38 `iqnlocal` (a-d), b39 `fqflocal` (a-d), b40 `mqrdqnlocal` (e-h).
Optionally b37 `c51local` (a-d), see §6.

## 1. The reads

Notation, per state and action: quantiles θ_1 ≤ … ≤ θ_N with masses w_i summing to 1. For QR-DQN and
IQN every w_i is 1/N; for FQF the w_i are the proposed fraction widths, which is what its own mean read
uses. `Neg` is the set with θ_i < 0, `Pos` the set with θ_i ≥ 0. Two building blocks:

| block | formula | what it is |
|---|---|---|
| partial expectation of a set S | Σ_{i∈S} w_i θ_i | the set's share of the mean: magnitude **times mass**. `Neg` and `Pos` partials sum to the mean |
| conditional mean of S | Σ_{i∈S} w_i θ_i / Σ_{i∈S} w_i, **0 when S is empty** | the average value *given* the outcome falls in S: magnitude, **mass discarded** |

The reads, each an argmax over actions of the score in the second column. The `variant` string is what
`--policy-variant` takes and what names the result file.

| # | variant | score per action | tie-break | asked for as |
|---|---|---|---|---|
| 0 | `mean` | Σ w_i θ_i | -- | **the control**, greedy. Must be in the same pass at the same depth (§3) |
| 1 | `leastneg` | conditional mean of `Neg` (0 is best) | conditional mean of `Pos` | variant 1: "least negative wins, ties by most positive" |
| 2 | `mix:0.7` | 0.7 · cond-mean(`Neg`) + 0.3 · cond-mean(`Pos`) | -- | variant 2 |
| 3 | `mix:0.5` | 0.5 · cond-mean(`Neg`) + 0.5 · cond-mean(`Pos`) | -- | variant 3; the worked example: two quantiles at −1 and eight at 10 score 4.5, not 7.8 |
| 4 | `abovemean:30` | conditional mean of {θ_i > 30} | `mean` when every action scores 0 | variant 4 |
| 5 | `abovemean:60` | conditional mean of {θ_i > 60} | `mean` | variant 5 |
| 6 | `abovemean:10` | conditional mean of {θ_i > 10} | `mean` | the midgame threshold added on 2026-09-22 |
| 11 | `leastneg:-2` | as `leastneg`, with `Neg` restricted to θ_i ≤ −2; quantiles in (−2, 0) count for neither side | conditional mean of `Pos` | added 2026-09-22: least *death-sized* loss first, then upside. −2 sits between the starve (−0.5) and death (−5) penalties |

Proposed additions, each one more label on the same pass and each cheap (§4):

| # | variant | score | why |
|---|---|---|---|
| 7 | `mix:0.3` | 0.3 · cond-mean(`Neg`) + 0.7 · cond-mean(`Pos`) | the risk-seeking mirror of 2, so the `mix` axis has three points either side of nothing |
| 8 | `mixmass:0.7` | 0.7 · partial(`Neg`) + 0.3 · partial(`Pos`) | the **mass-aware** version of 2 (equivalently argmax of Pos + 2.33·Neg, a loss-aversion utility). Isolates what §2's mass-blindness does |
| 9 | `above:30` | partial expectation of {θ_i > 30}, `mean` on all-zero | the **mass-aware** flavour of 4: nine quantiles at 20 and one at 35 score 3.5 here and 35 under `abovemean:30`; five at 32 score 16.0 here and 32 there. Same split as 2 against 8, at the top of the distribution |
| 10 | `cvar:0.25`, `cvar:0.5` | mean of the lowest α of the mass | already implemented; the literature's standard risk-averse read, and the one b38 trained under. Anchors the family to something published |

Reads 0 to 6 and 11 are the ask. 7 to 10 were recommended and accepted (§6). **Thresholds are in return
units**: with γ 0.97 and the +100 perfect-game reward, a quantile above 60 means the net expects the
board filled within about 17 moves, above 30 within about 40, above 10 within about 78 (or, earlier in
the game, several meals in quick succession). Reads 4 and 5 therefore act only in the endgame and fall
back to `mean` everywhere else; read 6 reaches into the midgame.

A worked `mix:0.3` on ten equal-mass quantiles −5, −1, −1, −1, −1, 10, 10, 10, 10, 10: the negative side
averages to −1.8, the positive to 10, and the score is 0.3 · (−1.8) + 0.7 · 10 = 6.46. The count on each
side never enters the mix; for FQF the within-side average is weighted by the proposed fraction widths.

## 2. What the conditional-mean reads do, stated before they run

Reads 1, 2, 3, 4, 5, 6, 7 and 11 discard mass. This is the design and the reason to run them, but it has a
consequence worth predicting rather than discovering:

| action | quantiles (N 32) | mean | `mix:0.5` | `leastneg` |
|---|---|---|---|---|
| A | one at −5, thirty-one at 20 | 19.2 | 0.5·(−5) + 0.5·20 = **7.5** | −5 |
| B | sixteen at −0.1, sixteen at 20 | 9.95 | 0.5·(−0.1) + 0.5·20 = **9.95** | −0.1 |

The mean prefers A (a 3% chance of dying for a much better game). `mix:0.5` and `leastneg` prefer B (a
50% chance of a trivial loss). So these reads are not "risk-averse" in the CVaR sense; they are
*severity*-averse and *probability*-blind. On Snek that predicts: fewer deaths, more of the starving
loops b26 found the reward made rational (a starve is −0.5, a death −5, so a loop is the least-negative
outcome available), and a lower perfect rate. Read 8 (`mixmass`) is the control for exactly this: same
weights, mass kept. If 2 and 8 separate, the mass-blindness is what moved the number.

The `abovemean` reads have the opposite blind spot: an action with *any* quantile above the threshold wins
even when it also carries death mass, and one quantile at 35 beats five at 32. Read 9 (`above:30`) is
the mass-aware control for that. Prediction: read 5 is indistinguishable from `mean` (it almost never
fires), read 4 is within noise or slightly down, read 6 is down by a measurable amount on the weaker
cells (b38, b39) where the tail is less reliable.

The overall prediction on record: **no read beats `mean` on the perfect rate by more than the paired
noise floor** (§3), and `leastneg` and `mix:0.7` are the worst by a clear margin. Where a read does
win, the expectation is `mixmass:0.7` or `cvar:0.5` on b37/b40, by under 1 pp, through fewer deaths.
Falsifying that is the point.

## 3. Measurement design

| | choice | why |
|---|---|---|
| checkpoints | **top 25 per arm by stage-A `perfect_percent`, ties by `trailing_avg_score` then later step**, 100 per cell. New selector `topa:<n>` (§5) | "top 100 per cell" taken as 25 per arm so one seed cannot supply all 100; the alternative is 100 per cell via per-arm `steps:` files, a spec per arm. **User's call**, §6 |
| depth | 1,000 episodes, every read, **including `mean`** | stage B is 500 and the hof passes 5,000 / 30,000, so no existing row is the control. Same episodes, same seed 0, same box, same pass |
| pairing | every read runs on the same 100 checkpoints of a cell; the comparison is each read's perfect rate minus `mean`'s on the same checkpoint, averaged over the 100, with a bootstrap CI over checkpoints | top-N selection guarantees regression to the mean; a paired control at the same depth absorbs it. Episodes are seeded through the game, so the first food sequence is shared but the trajectories diverge, hence pairing at the checkpoint level and not the episode level |
| noise floor | at p ≈ 0.95 one 1,000-episode row has SE 0.7 pp; the paired mean over 100 checkpoints has SE under 0.2 pp | a read has to move the cell by about 0.5 pp to be read as real. The `mix`/`leastneg` predictions are several pp |
| what is reported per read per cell | perfect %, death %, starve %, mean score, mean episode length, Δ perfect vs `mean` with CI | death vs starve is inferred from a row's `rewards − scores` (terminal −5 against −0.5; the 0.001 distance shaping is noise at that scale). Adding `died`/`starved` counts to the engine row is the cleaner fix and a bigger change; inference first |
| stop rule | none (`stageb` pass, `stop: None`): every read measures all 1,000 episodes on every checkpoint | the hof passes' early-retire would bias a read that starts badly |
| IQN / FQF fractions | **fixed** for this pass: IQN reads at the 32 evenly spaced τ_i = (2i−1)/64 instead of 32 draws, FQF at its proposed fractions as now | the sets `Neg`, `Pos`, `{θ > t}` are otherwise redrawn every call, which adds draw noise to every read and none to QR-DQN's. The `mean` control in this pass uses the same fixed taus, so the pass is internally consistent; stage B's sampled-tau mean stays the external reference and is reported beside it. **User's call**, §6 |

## 4. Cost

Measured: a 500-episode row on a QR-DQN head takes 18 to 24 s per shard on the laptop (b37e, b40e
stage B: 21 to 28 episodes/s per shard). Call it 20 episodes/s per shard and 16 shards on the desktop.

| design | checkpoints × reads × episodes | episodes | wall at 320 eps/s |
|---|---|---|---|
| reads 0 to 6 and 11, four cells | 400 × 8 × 1,000 | 3.2M | ~2.8 h |
| all thirteen reads (cvar as two), four cells | 400 × 13 × 1,000 | 5.2M | ~4.5 h |
| **the decided design**: thirteen reads, five cells with C51 | 500 × 13 × 1,000 | 6.5M | ~5.6 h |

FQF and IQN rows are slower per episode than QR-DQN (a cosine embedding per forward pass); the pilot
(§5 step 4) times them. Even at half the assumed rate the full design is an overnight pass. Cheap, as
the user said; the reads are the interesting part and pruning them saves little.

## 5. What has to be built, in order

| step | change | rule it falls under |
|---|---|---|
| 1 | **the reads**: `algos/dist/net.py` `parse_variant` learns `leastneg[:<t>]`, `mix:<a>`, `mixmass:<a>`, `above:<t>`, `abovemean:<t>` beside `cvar:<a>`; all three heads (`categorical`, `quantile`, `implicit`) gain one `read_values(observations, variant)` that computes the score table of §1 from `(quantiles, masses)`, and `greedy_policy_fn` calls it. The fixed-tau option for the implicit head. Tests pin every read on a hand-built (quantiles, masses) table including the §2 example, the empty-set zero, the all-zero-fallback to `mean`, and FQF's unequal masses; a `mut_dist_reads.json` mutation spec | **code**: built, described, waits for approval |
| 2 | **the selector**: `tools/step_selectors.py` gains `topa:<n>` (top n by stage-A `perfect_percent`, ties by `trailing_avg_score`, then the later step, intersected with the checkpoints present) | `tools/`, but it rides with step 1 so it waits with it |
| 3 | **the report**: `tools/read_compare.py` reads `runs/<arm>_checkpoint_evals_<label>.json` for a cell's arms and labels and prints the §3 table, Δ and CI included; the same table lands in `docs/results.md` | `tools/`, rides with step 1 |
| 4 | **pilot on the laptop**: two b40e checkpoints, every read, 200 episodes, `PYTHONPATH=. python -m tools.closeout b40e-mqrdqnlocal-seed5 --selector steps:… --episodes 200 --policy-variant X --label pilot-X --shards 4`. Checks: each read's actions differ from `mean`'s on some states (a read that never disagrees is a bug or a no-op and is dropped), the result files sit beside each other, and a per-episode timing for the QR head; one b39a checkpoint the same way for the FQF timing | the `laptop-run` skill; smoke output, deleted after |
| 5 | **deploy** to the desktop (`desktop-deploy`), confirm the new parser is live before any spec is pushed | the deploy rule; the memory `deploy-before-queueing-specs-that-need-new-knobs` |
| 6 | **rsync** b40e-h to the desktop: the selected 25 checkpoints per arm plus `arch.json`, and `runs/b40[e-h]-…_evals.json` into `desktop/runs/` so `topa` can resolve there. b37, b38 and b39 are already on the desktop (2,000 to 3,000 checkpoints per arm) | `hof-remeasure` step 2; the memory `desktop-eval-of-laptop-policy` |
| 7 | **the specs**: one `eval` spec per cell per read, `policies` the cell's four arms, `selector: "topa:25"`, `episodes: 1000`, `eval_shards: 16`, `eval_args: ["--policy-variant", "<variant>", "--label", "reads-<variant>"]`, `box: "desktop"`, id `b41r-<cell>-<variant>`, all under one priority so the desktop runs them back to back. 13 reads × 5 cells = 65 specs | **pushing to `ops` is the code rule**: queued only on the user's go for this job |
| 8 | **read-out**: `tools/read_compare.py` per cell, the table into `docs/results.md`, the finding (or its absence) into `docs/findings.md`, this plan's §2 predictions marked held or falsified | docs, standing authorization |

Two things this deliberately does not touch: stage A stays the `mean` read (it drives the epsilon
schedule, `a-return-tail.md` §5), and no training runs. The pass is measurement only, on frozen
weights.

## 6. Decisions (2026-09-22)

| question | decision |
|---|---|
| 25 per arm or 100 per cell | **25 per arm**, 100 per cell |
| thresholds: conditional mean or partial expectation | **conditional mean is the primary** (`abovemean:10/30/60`), matching the `mix` convention; `above:30` runs as the mass-aware comparison |
| fixed taus for IQN and FQF | **fixed** |
| the C51 cell | **included** (b37a-d); the reads use the atoms as θ and the probabilities as masses |
| reads 7 to 10 | **kept**; `leastneg:-2` added as read 11 |

## 7. What would change the plan

- If the pilot shows a read never disagreeing with `mean` on the pilot checkpoints (likely `above:60`),
  it is dropped before the specs are written rather than measured to 1,000 episodes.
- If FQF rows are more than three times slower than QR-DQN rows, the b39 cell runs reads 0 to 6 only.
- If any read beats `mean` by more than 1 pp on b37 or b40, the follow-up is the hof5000 protocol on
  those checkpoints under that read, and the HOF row carries the variant as §5 of `a-return-tail.md`
  already provides for.
