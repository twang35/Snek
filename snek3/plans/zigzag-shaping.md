# Zigzag shaping: charging a left-right or right-left pair

**Status: reviewed 2026-09-16, decisions in §6; ready to build.** Nothing built yet. The request: a potential-based reward
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

Agreed in review: **both terms, behind two knobs; adjacent reversals only; window = the observation
history (8); b27's `hist8` config at 100M as the base; four arms of each term, one dose per term, sized
from the measurement in §5.**

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

## 5. The measurement: how much does the champion zigzag?

`b32g` @62423040 (the record, 29,967 /30,000) traced greedily for 5,000 episodes on seed 11
(`tools.death_trace`; 4,996 perfect, 2 died, 2 starved; `logs/zigzag/b32g-5k-s11.*`, throwaway):

| | per step | per meal (10.43 steps) | per episode |
|---|---:|---:|---:|
| turns | 0.298 | 3.11 | ~295 |
| **adjacent reversals (*k* = 1, the reward's definition)** | **0.0171** | **0.178** | **16.9** |
| reversals within two steps (*k* = 2, `death_analyze`'s) | 0.0688 | 0.718 | ~68 |
| U-turns (the same turn twice, the fill pattern) | 0.0822 | 0.86 | ~81 |
| steps with R₈ > 0 | 9.1% | | |

R₈ is 0 on 91% of steps, 1 on 6.7%, 2 on 1.9%, 3+ on 0.5%; mean 0.119.

**Where the reversals are.** Almost entirely in the open board, and gone before the endgame:

| board fill | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 | 80-99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reversal rate, *k* = 1 | 4.2% | 4.2% | 3.8% | 3.5% | 2.2% | 0.5% | 0.12% | 0.03% | 0.02% |
| turn density | 29% | 28% | 28% | 29% | 31% | 30% | 30% | 30% | 31% |

So the converged policy already lays the crowded half of the board without a single zigzag: **from
60% fill on, one adjacent reversal in every ~900 steps**, at unchanged turn density (every turn is a
U-turn or an isolated corner). What either term can act on is the first half of the game, where the
obs-history plan noted a zigzag costs nothing, and where 17 reversals per episode live. That sharpens
the prediction in §6 rather than changing the design: the batch now asks whether removing the *early*
zigzags changes anything downstream.

**Doses, from the numbers.** With 0.178 reversals per meal:

| term | traffic per meal at dose *x* | the reference scale | dose |
|---|---|---|---|
| potential, `SNEK_ZIGZAG_SHAPING` | 2 · 0.178 · *c* = 0.36 *c* (a charge and its refund) | chase-safe was sized at ~25% of `FOOD_REWARD` per meal, giving *c* 0.10 at 2.5-3.6 flips/meal | **c = 0.5** — 0.18 per meal, the same order as chase-safe's budget; 0.7 would match it exactly |
| penalty, `SNEK_REVERSAL_PENALTY` | 0.178 · *p* | the step penalty's flow is 0.104 per meal (10.43 × 0.01) and b26 found that real while a tenth of it did nothing | **p = 0.5** — 0.089 per meal, the step penalty's order; a reversal costs half a meal, ~8.5 per episode against ~49 of step penalty and ~95 of food. *p* 0.1 (0.018 per meal) would sit in b26's dead zone |

## 6. Batch b34

| | |
|---|---|
| base | **b27's `hist8` config, verbatim** (`docs/runs.md` b27 has it in full): 100M, `SNEK_PPO_ANNEAL_FRACTION` 0.5 — γ 0.99 → 0.999, λ 0.95 → 0.999, entropy 0.01 → 0.001, all final at 50M and held to 100M; preset `b2`, step penalty 0.01, win 100 |
| cells | **`zz`** `SNEK_ZIGZAG_SHAPING=0.5` (window 8, the history depth); **`rp`** `SNEK_REVERSAL_PENALTY=0.5` |
| arms | `b34a`-`b34d` `zz` seeds 1-4; `b34e`-`b34h` `rp` seeds 5-8. One desktop wave |
| control | b27's `hist8` cell — eight seeds of exactly this config at 100M, plus b28's first 100M (the same arms held on). No control arm of its own |
| judged on | stage-B density, the `hof5000` / `hof30k` gates and the 30k top against b27's `hist8` table (94-95% density, 99.81 top); **plus the §5 trace** re-run on each cell's top checkpoint at 5,000 episodes on seed 11, so the reversal-rate-by-fill table above is read against the control's |

**Prediction, registered with the user 2026-09-16 (the agent's).** `zz` is level with `hist8` on
density and on the 30k top, and its reversal rate by fill is within 1 pp of the table above at every
decile: the invariance holds, PPO takes nothing from the hint, as b21 found for chase-safe. `rp` cuts the
early-board (fill < 50) reversal rate by more than half and leaves the endgame's ~0 where it is; its
density is within noise of `hist8`, since the failures run through sealed pockets and orbits in the
crowded board where the champion already does not zigzag. The outcome that would matter: a density or
30k-top gain in `rp`, which would say the open-board path shape carries into the endgame — falsifying
"not the mechanism" — or a density *loss* in `rp`, which would say the early zigzags are load-bearing
(a way of buying time to line up the fill) and the penalty removes them at a cost.

**Order of work.** Build §3 and §4 on the laptop; run the suite and the mutation pass; smoke one arm
of each cell and read `reward config:` for the new knobs; `desktop-deploy`; then `queue-batch` the
eight specs (a b34 manifest in `plans/sweep-extra.json` with `requires_code` on both knobs), pinned to
the desktop. The specs wait for approval as every push to `ops` does.

## 7. Decisions taken in review (2026-09-16)

| question | decision |
|---|---|
| both terms, or the potential alone | both |
| reversal definition | adjacent only (*k* = 1); the diagnostic keeps reporting *k* = 2 |
| window | 8, the observation history |
| doses | one per term, sized from §5: 0.5 and 0.5 |
| control | none of its own; b27's `hist8` cell, the base itself, is the control |
| base and cap | b27's config at 100M, anneal fraction 0.5 (finals at 50M, held to 100M) |
