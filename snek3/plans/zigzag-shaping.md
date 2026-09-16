# Zigzag shaping: charging a left-right or right-left pair

**Status: proposal, 2026-09-16, for review.** Nothing built. The request: a potential-based reward
that pays for *not* zigzagging — every step that is not a reversal earns a little, and a reversal (a
`left` straight after a `right`, or a `right` straight after a `left`) takes it away.

This is the arm `plans/archive/obs-history.md` §5 deferred as "the optional fifth arm": *does the policy
need to **see** its zigzags, or merely be **charged** for them?* b27 answered the first half — making the
turn sequence visible (`hist8`) was the largest lever in the project, 49 → 94-95% density — and the
second half has never run. Two prior facts bound what to expect, and both are stated up front because
they change the design:

| fact | where | consequence here |
|---|---|---|
| a potential-based term **cannot change the optimal policy** (Ng, Harada & Russell 1999); it changes how fast the policy is learned, and b21 found the chase-safe potential a no-op for PPO | `env/constants.py` shaping notes, `docs/runs.md` b21 | if the goal is a converged policy that zigzags less, a pure potential is the wrong tool on paper. Run it as asked, and run the plain penalty beside it, so the batch says which |
| **zigzagging is not what kills the champions** (reversal rate 2-3 pp above matched perfect games before a starve; the deaths are food-sealed pockets and tail-following orbits) | `docs/findings.md` "Zigzagging is not the mechanism" | the headline metric stays the perfect rate and the stage-B density, and the batch also measures the reversal rate directly, so "it stopped zigzagging and nothing improved" is a legible finding rather than a null |

Recommendation: **build both terms behind two knobs, run a four-cell batch at four seeds on the b33 base,
and register the prediction that the potential term is level with the control while the plain penalty
moves the reversal rate.** Details below.

## 1. What a reversal is, and where it is read from

| term | definition |
|---|---|
| move *j* | the relative action taken *j* steps ago, `left` / `right` / `forward`, read off the body: cells *j*+1 → *j* → *j*−1 behind the head (`env.observations.move_history_obs`, `vectorized.vec_env.move_history_bits`). No buffer, nothing to reset or snapshot |
| **reversal pair** | moves *j* and *j*+1 are one `left` and one `right`, in either order. **Adjacent only**, as asked (`tools/death_analyze.reversal_flags` uses *k* = 2, "within two steps"; the diagnostic in §5 reports both, the reward uses *k* = 1) |
| **R_K(s)** | the number of reversal pairs among the last *K* moves the body shows, 0 ≤ R_K ≤ K−1. A body shorter than *K*+2 cells reads `forward` for the moves it cannot show, so the opening board reads 0 |

Two shapes worth checking against the definition before agreeing to it:

- A **boustrophedon fill** — the pattern every perfect game ends in — turns `left, left` then `right,
  right` at each lane end. Two same-direction turns in a row are a U-turn, not a reversal. **Not charged.**
- A **diagonal staircase** — `left, right, left, right` across open space — is the zigzag proper, R_K =
  K−1 at full window. **Charged on every step.** `left, forward, right` (a one-cell jog) is *not* adjacent
  and is not charged at *k* = 1; that is the one case where *k* = 2 would differ.

## 2. The two terms

### 2a. The potential term — what was asked for

Φ(s) = −R_K(s), and the shaping reward is the same form as the two existing potentials:

    F = c · (γ · Φ(s') − Φ(s)),   Φ(terminal) = 0

With `SNEK_ZIGZAG_SHAPING=c` and `SNEK_ZIGZAG_WINDOW=K`. What the agent sees per step:

| step | R changes by | F |
|---|---:|---|
| a reversal is made | +1 | ≈ −c |
| a reversal pair falls out of the window (K steps later) | −1 | ≈ +c |
| any other step, with r pairs in the window | 0 | c·(1−γ)·r — positive, and small (r ≤ 7, γ ≥ 0.99: at most 0.07c early, 0.007c at γ 0.999) |
| any other step, clean window | 0 | exactly 0 |

So it does match the request — a reversal is charged and every other step pays back — but the payback is
the *refund of the same charge K steps later*, and the sum over an episode is exactly −c·Φ(s₀) = 0. The
net discounted cost of one reversal to the agent is c·(1 − γ^K): **0.8% of c at γ 0.999, K 8.** That
is the policy invariance, stated as a number: the term is a credit-assignment hint, not an incentive.
This is why §2b runs beside it.

**Window.** Default `K = SNEK_OBS_HISTORY` (8 on every current base), with the knob to override. The
reason is Markov: Φ should be a function of what the policy observes, and with `hist8` the last eight
moves *are* indices 26-41 of the observation, so the critic can represent Φ exactly and the shaping is
learnable rather than noise. A window longer than the history charges the policy for state it cannot
see. (The alternative, Φ = −(reversals along the whole body), is a genuinely spatial quantity — how
S-bent the snake currently lies on the board — but it depends on up to 98 unobserved moves and grows
with length; not recommended for the first batch, kept as a follow-up.) A body shorter than the window
simply shows fewer moves; no gate is needed, and none is proposed — `Phi = 0` on the opening board
already, so the telescope is exactly 0 the way the other two gates arrange.

### 2b. The plain penalty — the comparison arm

`SNEK_REVERSAL_PENALTY=p`: subtract *p* on every step whose move is a reversal of the move before,
terminal steps included, exactly as `SNEK_STEP_PENALTY` is applied. Not invariant: it changes the
objective, which is the point — b26 found the step penalty "changes what the converged policy *does*,
not how fast it learns", and that is the effect the request is after.

**The "points for every other action" half is already in the base.** A bonus of *b* on every
non-reversal step is a per-step reward of +*b* plus a reversal charge of −*b*, and the base runs
`SNEK_STEP_PENALTY=0.01` — so a flat bonus would only shrink the existing step penalty, which b26 tuned.
The penalty knob carries the whole shape by itself and needs no bonus twin.

**Both knobs add**, as the two existing potentials add: an arm may run either or both.

## 3. Where it goes in the code

Mirrors the free-space term line for line; the vectorised env and the reference must agree to the bit
(`tests/test_vec_parity.py`).

| file | change |
|---|---|
| `env/constants.py` | `ZIGZAG_SHAPING` (0.0), `ZIGZAG_WINDOW` (`OBS_HISTORY`, or 8 when history is off), `REVERSAL_PENALTY` (0.0), each via `_num`. Read at import like every reward knob; **silent on `hyperparameter override:`**, so they go on the `reward config:` line |
| `env/observations.py` | `recent_moves(body_positions, depth)` → the last `depth` relative actions, most recent first, factored out of `move_history_obs` (which becomes two lines over it); `reversal_count(body_positions, window)` over it. The one definition of a reversal in the reference |
| `env/game.py` | `zigzag_potential` cache with the same three lifecycles as `free_space_potential` — set after `_rebuild_grid()` on reset, recomputed on snapshot restore (the forking note at lines 402-413 applies verbatim: the potential is a pure function of the body), and the PBRS block in `step()` with `0.0 if self.finished`. The penalty: `if REVERSAL_PENALTY and reversal_count(body, 2): reward -= REVERSAL_PENALTY`, beside `STEP_PENALTY` |
| `vectorized/vec_env.py` | `recent_rel_moves(body, hp, length, depth)` → `(n, depth)` relative codes, factored out of `move_history_bits` (which becomes a two-column expansion of it); `_zigzag_now()` counts adjacent `(0,1)` / `(1,0)` pairs over the first `K` codes; `zigzag_potential` joins `STATE_FIELDS` (the tuple check in `tests/test_vec_env.py` fails otherwise), `_refresh_potentials` and `_shaping_reward`; the penalty in `step()` next to the step penalty |
| `vectorized/config.py` | re-export the three names; `describe()` grows `zigzag c={} win={}, reversal p={}`; `tests/test_vec_config.py` pins them |
| `docs/running.md`, `docs/environment.md` | the knob line under "Rewards and shaping"; the reward table gains two rows |
| `tools/sweep_specs.py` / `plans/sweep-extra.json` | the b34 manifest, `requires_code` on the two knobs so the specs cannot be written before the code is deployed |

**Deploy before queueing.** An unknown `SNEK_*` is ignored silently, so the box must be on the new code
(`desktop-deploy`) and `reward config:` read on a smoke arm before the specs go to `ops`.

## 4. Tests, in the same pass

| test | pins |
|---|---|
| `tests/test_move_history.py` | `reversal_count` on hand bodies: staircase reads K−1; boustrophedon U-turns read 0; the straight opening body reads 0; a body two cells shorter than the window counts only the pairs it shows; `move_history_obs` unchanged after the refactor (byte-equal against the pre-refactor output on a fixed-seed run) |
| `tests/test_vec_parity.py` | `test_zigzag_shaping_parity` (growth + coiled endgame, as the free-space test), the three-terms-together test extended, `test_reversal_penalty_parity`; and the "would notice if silently off" guard extended to the new term |
| `tests/test_reward_shaping.py` | the telescope test at `c = 0.1`: discounted shaping over a whole episode is −c·Φ(s₀) = 0 to 1e-9, forked-restore included; the penalty fires on exactly the steps `reversal_flags(actions, k=1)` marks |
| `skills/mutation-test` | on `reversal_count` and the two `step()` blocks — the pairs `(0,1)`/`(1,0)` collapsing to "any two turns" is the mutant that matters, because a boustrophedon fill would then be charged and every perfect game would read worse |

Invariant 1 is untouched: nothing here compares a reward to a constant, and the perfect-game counter reads
score.

## 5. Batch b34

| | |
|---|---|
| base | b33's: `hist8`, pen01, preset `b2` (chase-safe 0.1 gate 75, food-distance 0), 50M, `SNEK_PPO_ANNEAL_FRACTION` 0.5 — every ramp final at 25M |
| cells | **`zz05`** `SNEK_ZIGZAG_SHAPING=0.05`, **`zz20`** `=0.2` (window 8); **`rp01`** `SNEK_REVERSAL_PENALTY=0.01` (one step penalty per reversal), **`rp05`** `=0.05` |
| seeds | 1-4 pinned to the letter, 16 arms, two desktop waves |
| control | b33's `win100` cell — the same base, cap and anneal, four seeds, closed 2026-09-14. Saves four arms; the cost is that it ran on a different day, which with pinned seeds and a deterministic env is no cost |
| judged on | stage-B density and the `hof5000` / `hof30k` gates as every batch; **plus the reversal rate**, from `tools/death_analyze.path_measures` over each cell's top checkpoint at 5,000 episodes (turn density, reversal rate at *k* = 1 and 2, mean straight run, split perfect / collision / starve and by fill bucket — the §3 measures of the obs-history plan, already built) |

**Doses.** The chase-safe dose was sized so the shaping budget per meal was ~25% of `FOOD_REWARD`.
A reversal rate of a few percent per step at ~10 steps per meal is a few tenths of a reversal per
meal, so c = 0.2 puts a reversal charge at 0.2 and the per-meal shaping traffic near 0.05-0.1 — the same
order as chase-safe's 0.1 — and c = 0.05 is the quarter-dose. For the penalty, 0.01 makes a reversal cost
one extra step and 0.05 five; above that the penalty is a tenth of a food reward per reversal, which
b26's step-penalty cliff (0.01 real, 1e-3 nothing) says is where to look next if 0.05 moves nothing.
**Before queueing, read the actual reversal rate of `b32g` @62423040 off `death_analyze`** and rescale
if it is far from a few percent; the numbers above are the plan's assumption, not a measurement.

**Prediction, to register with the user before queueing** (the agent's, 2026-09-16): the two potential
cells are level with the control on density and on the 30k top, and their reversal rate is within 1 pp
of the control's — the invariance holds and PPO gets nothing from the hint, as b21 found for chase-safe.
`rp05` cuts the *k* = 1 reversal rate by more than half; `rp01` by less. Neither penalty cell raises
density, because the failures run through sealed pockets and orbits, not zigzags; `rp05` may lose a few
points of density if straight-line preference makes the endgame fill harder. A density gain in a penalty
cell would falsify the "not the mechanism" finding and would be the interesting outcome.

## 6. Decisions for review

1. **Both terms, or the potential alone?** Recommended: both. The potential alone can, by theorem, only
   tie or lose against the control on the converged policy, and a batch whose best outcome is a tie is
   worth pairing with the arm that can win.
2. **Adjacent only (*k* = 1)** for the reward, as asked; the diagnostic reports *k* = 2 too. Or *k* = 2
   in the reward, charging `left, forward, right` as well.
3. **Window = the observation history (8)**, or the whole body.
4. **Doses** 0.05 / 0.2 and 0.01 / 0.05, pending the `b32g` reversal-rate read.
5. **Control**: reuse b33's `win100`, or a fifth cell of four.
6. **Base and cap**: b33's 50M with the half-length anneal, or b27's 100M `hist8` for direct comparison
   with the b27/b28 tables at the cost of double the wall time.
