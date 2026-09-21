# Group D: data efficiency -- BBF-style resets, as the late-plasticity probe

**Status: re-planned 2026-09-20 as the reset probe; the knob is built** (`algos/dqn/resets.py`, `SNEK_RESET_*` on
`algos/dqn/algo.py`, `maybe_reset` on both agents; tests `tests/test_resets.py`, mutants `tests/mut_resets.json`
10 / 10 killed). **Where the row stands is the `status` column of §2 and the gate table of §3.** Group D of
[`algorithm-series.md`](algorithm-series.md); conventions in [`README.md`](README.md). Phase 5 of the running
order, after Group C (the value stack) and before Group E (memory). **Queued 2026-09-20 as b43** (`docs/runs.md`), the day b40 closed: 4 seeds every 600k gradient steps + 4 seeds every 2.4M, α 0.5, stop after 10.5M, on b40e-h's spec.

**What changed on 2026-09-20, and why.** This plan was phase 1, "how many steps does any of this need": BBF at
100k and 500k moves, its onset step against A1's, and a rule that halved every later value row's cap if it came in
fast. Three things undercut that. Group A answered the onset question itself -- every local cell reached 90%
between 0.3M and 1.8M steps and A2-A6 were queued at 2-3M, not 100M. BBF's lever is a replay ratio of 8 and a 4x
wider net, which buys data efficiency with compute, and on a game whose simulator is exact and in-process the
scarce resource is gradient steps, not moves. And the budget question Group A *left open* is the other end of
the curve: every value cell reached 88-94% and then drifted or never held; no A checkpoint has held 97%+. BBF's
shrink-and-perturb resets are a late-plasticity mechanism, and late drift is the shape of most of this project's
collapses, so the row is now that probe. The budget rule is replaced by what A produced: **value rows queue at
3M steps with hold-if-rising.**

## 1. The row

### D1 -- BBF-style resets on A6's best cell (Schwarzer et al. 2023, "Bigger, Better, Faster", §4)

**What this row is, and is not** (scoped 2026-09-20 after a review): b43 tests **shrink-and-perturb resets alone, on one cell**. It is not BBF
and not a test of BBF's data efficiency -- the replay ratio of 8, the ×4 IMPALA/C51 net, AdamW, the EMA target, SPR and the 100k regime are
all absent, by the decision below -- and **its result is never to be reported as a verdict on BBF.** The conclusion it can reach is "the reset
alone is / is not a hold mechanism on this cell", and even the negative half of that waits for the anneal wave (§5). This is also the one
group that departs from the series' rule of a paper cell as written (`algorithm-series.md` §0): BBF as written needs SPR built and a CNN
encoder that has no meaning on a 26-value observation. A reduced paper cell -- BBF without SPR and the CNN -- is buildable and is the user's
call, not a consequence of b43.

BBF resets the encoder toward a fresh initialisation every 40k gradient steps -- **shrink-and-perturb**, θ ← α θ
+ (1 − α) θ_fresh at α 0.5 -- and re-initialises the layers after it fully, with the optimiser state for those
parameters cleared, and keeps the last stretch of training reset-free so the final net has settled. Here that is
a knob on the shared DQN agent, so every value rung can take it, and nothing else of BBF comes with it in the
first wave.

| piece | the paper | here |
|---|---|---|
| **resets** | encoder shrink-and-perturbed 50 % toward a fresh init, the layers after it fully reset, every 40k gradient steps at replay ratio 8; no resets in the final stretch | `algos/dqn/resets.py`: the trunk (every `hidden.*` linear of `QNet`, which is `qnet.hidden.*` inside every distributional net) interpolated at `SNEK_RESET_ALPHA` (0.5); the head -- the Q or distributional head, IQN's cosine embedding, FQF's fraction proposal -- replaced by the fresh net's; the **target copied from the reset online net**; every optimiser's state cleared. Every `SNEK_RESET_INTERVAL` gradient steps, none after `SNEK_RESET_STOP_AFTER` (both in gradient steps; 0 is off / never stops). The fresh net comes from the same builder as the agent's, seeded from the arm's seed and the reset count, so two arms at one `SNEK_SEED` reset identically. `resets` is persisted, so a resume continues the count |
| the base cell | Impala ×4 C51 at replay ratio 8, 100k steps | **b40's `mqrdqnlocal`** (A6): QR-DQN N 32, Munchausen α 0.9 τ 0.03 l₀ −1, the local plumbing (fork 4, shield, PER, target period 8, replay ratio 1, batch 128, lr 1e-5), `hist8`, `fc 320`, **3M steps**. The densest, steadiest value cell of A2-A6 (`docs/runs.md` b40) |
| the cadence | every 40k gradient steps ≈ 20 cycles in a 100k-step run at ratio 8 | the cell runs ~4 gradient steps a counted step (fork 4 lanes, ratio 1), so 3M steps is **~12M gradient steps**. Two cells: **every 600k** (~20 cycles, the paper's count) and **every 2.4M** (~5 cycles); both stop after **10.5M** (the last eighth reset-free) |
| wider net, replay ratio 8, AdamW, EMA target, SPR, the 100k regime | the rest of the recipe | **dropped**: they are the data-efficiency levers, and the probe is the reset alone on a cell whose other knobs are already tuned. Each is a later cell if the reset earns one |
| n-step and γ annealed within each cycle (10 → 3, 0.97 → 0.997 over 10k gradient steps) | what lets a freshly reset net relearn fast -- BBF's own ablations put the schedules among the pieces that matter most *in the reset regime*, because a freshly reset head relearns from a shorter, more myopic target | **not in the first wave**, and that is the first wave's known limitation: the base cell is n-step 1 at γ 0.99 throughout. **Poor recovery after a reset in b43 therefore says nothing about resets as BBF uses them**, and the anneal wave (`SNEK_RESET_ANNEAL_*`, not built; §6) runs before the row can close on a negative -- whether b43 dips without holding or is worse everywhere. Built while b43 runs, so it can follow at once |

Tests (`tests/test_resets.py`): the partition puts every hidden linear in the trunk and the rest in the head, for
`QNet` and for FQF's net; α = 1 leaves the trunk and still re-initialises the head; α = 0 equals a fresh init at
the given seed everywhere; the trunk is interpolated exactly and the head is not; C51's support buffer is
untouched; through `update()` on the scalar, QR-DQN and FQF agents the reset fires on the interval, copies the
target, clears Adam and FQF's RMSProp; successive resets draw different fresh nets and two arms at one seed
reset the same; nothing fires past the stop, and interval 0 is off; the reset count survives a resume and an old
checkpoint reads as none; the knobs reach the agent and `describe()`, a bad α is refused by name, PPO refuses the
knobs. Mutants (`tests/mut_resets.json`, 10 / 10 killed 2026-09-20): head interpolated like the trunk, α and
1 − α swapped, trunk replaced outright, the stop guard dropped, the interval firing every step, the optimiser
state kept, the target not copied, the count not persisted, the fresh seed ignoring the count, FQF's optimiser
not cleared.

## 2. The batch

| batch | status | arms | base | read against | judged on |
|---|---|---|---|---|---|
| D1 | **queued 2026-09-20 as b43** (b40 closed the same day: 436 stage-B rows, best 98.0, no `hof5000` candidate) | 4 seeds `mqrdqnlocal` + `SNEK_RESET_INTERVAL=600000` + 4 seeds at `2400000`, both `SNEK_RESET_ALPHA=0.5`, `SNEK_RESET_STOP_AFTER=10500000`, 3M steps, one 8-arm wave | b40's `mqrdqnlocal` spec, verbatim, plus the three knobs | **b40e-h**, the same cell without resets | **hold**: the perfect rate after each reset's dip and in the reset-free tail, drawdown count, `zero_since`; then stage-B density, `hof5000`, `hof30k` against b40's |
| D1 anneal | not planned unless D1's dip is the problem | the 600k cell with n-step and γ annealed within each cycle | D1 | D1 | whether the anneal removes the dip without losing the hold |

The prediction to register when it is queued: the 600k cell shows a visible dip after each reset and a higher
late perfect rate than b40's; the 2.4M cell shows fewer, deeper dips. If neither holds better than b40 the
reset is not the lever here and the group closes.

## 3. Gates

**Where the row stands** (2026-09-20; a cell is filled in the pass that runs the gate, and an empty cell means
*not run*, not *passed*):

| gate | D1 |
|---|---|
| 1 smoke, checkpoint, restore; the log says `resets N` | **passed 2026-09-20**: `qrdqn` N 32 + Munchausen, `SNEK_RESET_INTERVAL=1000`, 5,000 steps (4 lanes, ~4 gradient steps a step): the log printed `resets 4 (every 1,000 gradient steps, at 4,000)`, `ckpt-1000` … `ckpt-5000` written (`SNEK_MIN_CHECKPOINT_SCORE=0` for the smoke), `ckpt-5000` restored through `evaluate.py smoke one` (500 episodes), and a relaunch resumed at 5,000 with the reset count in `resume.pt`. Run in a worktree so the live b40 wave never imported a half-edited module |
| 2 mutation spec kills every mutant | **passed 2026-09-20**: `mut_resets.json` 10 / 10 |
| 3 the default is unchanged | **passed 2026-09-20**: interval 0 adds one `due()` check that returns False; the whole suite (1,317 tests) passes with the knob wired into both agents |
| 4 the base cell closed | **passed 2026-09-20**: b40 closed on the laptop, both waves, all three passes (`docs/results.md`) |

1. `SNEK_ALGO=qrdqn SNEK_DIST_QUANTILES=32 SNEK_MUNCHAUSEN_ALPHA=0.9 SNEK_RESET_INTERVAL=1000 SNEK_MAX_STEPS=2500 ...
   train.py smoke` runs, resets several times, checkpoints, and the checkpoint restores through
   `evaluate.py smoke one`.
2. The mutation spec kills every mutant.
3. The default path is unchanged: the suite passes, and a fixed-seed arm at interval 0 is the same arm.
4. D1 is not queued until b40 has closed, because the wave is read against it.

## 4. What D1's number is used for

If the reset cells hold where b40 drifts, `SNEK_RESET_*` is offered to every later value row -- E2's R2D2 first,
and a re-run on Group C's best cell -- and the anneal wave decides whether the dip is worth removing. If they do
not, the late drift is not a plasticity problem on this game, which points the series at Group E (memory) and
Group G (planning) for what the value family cannot hold.

## 5. What would change the plan

- **The reset cells hold better than b40.** The knob goes to E2 and C's best; the anneal wave runs if the dip
  is large.
- **They dip and recover to the same plateau, or are worse everywhere.** Neither closes the row (changed 2026-09-20, review): without
  the within-cycle n-step / γ anneal a reset head relearns against the wrong target, so a poor recovery is expected from the *omission*, not
  evidence about the mechanism. The anneal wave runs; only if the annealed cells also fail to hold does the group close, as "the reset is not
  the lever on this game" -- not as a verdict on BBF.

## 6. The anneal wave: where it lands in the code

The knobs are `SNEK_RESET_ANNEAL_N_STEP` (`10,3`: n-step from the first value to the second over `SNEK_RESET_ANNEAL_STEPS`
gradient steps after every reset, BBF's 10 → 3 over 10k) and `SNEK_RESET_ANNEAL_GAMMA` (`0.97,0.997`), both off by default and
both inert unless `SNEK_RESET_INTERVAL` is set. **Nothing in the queue, the scheduler or the claims changes**: a spec carries `SNEK_*`
knobs already, and an unknown knob is ignored silently, which is why the knobs must be deployed before a spec names them
(`deploy-before-queueing-specs-that-need-new-knobs`). The change is three places in the training code, all on the shared DQN path
and all gated by the knob:

| place | what changes | why it is not free |
|---|---|---|
| `algos/dqn/collect.py` | `n_step` becomes settable between `step()` calls; a change flushes every lane's window (a window holds one n) | the window is per lane and its length is the invariant every n-step test pins |
| `algos/dqn/agent.py` (via `resets.py`) | the bootstrap's γ read from the schedule each update, `discount ** n` in the emitted transition following it | the emitted transition's `discount` is `γ^n` at *collection* time, so a change in γ is seen by new transitions only, as in BBF's own replay |
| `vectorized/vec_env.py` | `shaping_discount` follows γ | **the reward/discount coupling** (`docs/invariants.md`): the potential-based shaping terms are scaled by the agent's γ, so a γ that moves without the env's copy moving changes the reward, not only the target |

**Not a copy under `algos/bbf/`.** A directory holding copies of the collector and agent would contain the anneal, but it would also
hold ~1,000 duplicated lines whose bugs are then fixed in one place and not the other -- the root `CLAUDE.md`'s rule against two
implementations of one behaviour, and the reason `dist/` is subclasses of `DqnAlgo` rather than copies. The resets knob was put on the
shared agent so every value rung could take it, and the anneal follows it. What an `algos/bbf/` directory *can* be is what `sac2.py`
is to `sac`: a registration module naming a `bbf` algorithm whose defaults are the paper's values on the shared code, which is how a
reduced paper cell would be named if one is queued. The blast radius on the default path is the same as the reset knob's: a schedule
that returns the constant when off, pinned by the suite and by a fixed-seed arm at interval 0 being the same arm.
