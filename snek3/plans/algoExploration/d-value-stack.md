# Group D: the value stack -- Rainbow, Beyond the Rainbow

**Status: planned 2026-09-16, nothing built** (no `algos/rainbow/`, no `DuelingTrunk`, no `NoisyLinear`, no tests). **Where each piece and each row stands is the `status` column of §1 and §3 and the gate table of §4** (added 2026-09-20, on the same rule as `a-return-tail.md`: a cell is filled in the pass that does the work, and an empty cell means *not done*, not *passed*). Group D of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 4 of the running order; waits for Group A to close -- as of 2026-09-20 A1-A5 are closed (b39 closed today: FQF climbs where IQN flattened but never holds, stage B empty) and A6 (b40) is live on the laptop.

**Decided 2026-09-20, before implementation** (the questions raised when the plan was re-read against the code):

| decision | what it changes |
|---|---|
| **the noisy flag lifts the ε floor** | `build_config` refuses `SNEK_MIN_EPSILON` below `EPSILON_HARD_FLOOR` (1e-4, `algos/dqn/schedules.py`), so D1's ε 0 and D2's second-half ε 0 would be illegal. With `SNEK_RAINBOW_NOISY=1` the floor is lifted, since the noise supplies the exploration the floor protects; with noisy off it still holds |
| **D2 runs a paper cell and a local cell** | the paper's IQN 8 head plateaued at 55-65% in A4 with no stage-B checkpoint, so D2 as the paper alone might only repeat A4. The second cell of the wave is BTR's stack on the settings Group A's local cells found better, with A3's QR-DQN N 32 head (§2b, last paragraph); the layer-norm variant moves to the D2 ablation |
| **D1's paper cell is kept** | Group A's rule drops the paper cell after two rows show the same sign, and every A paper cell trailed. D1's is kept for 4 seeds because it is the only cell where noisy nets replace the shield and the fork, which is the row's question; revisited if it reads like A1 paper |
| **D builds `DuelingTrunk`** | it was B1's (BBF, phase 1, unbuilt). D owns it in `algos/rainbow/net.py`; B1 imports the scalar form |

The question: does the strongest single-box stack of value-learning tricks beat the PPO incumbent on
this game? D is read against A. If D2 beats D1 by about what IQN plus Munchausen beat C51 by in Group A,
the stack adds nothing beyond its parts. If it beats it by more, the trunk or the collection is doing
work worth isolating afterwards.

## 1. What the group shares

Both rows are compositions of pieces that exist by the time they run. The implementation work is the
composition and the one ingredient nothing earlier builds: **noisy nets**. Dueling arrives with B1 in phase 1.

| shared piece | status | decision |
|---|---|---|
| package | not built | `algos/rainbow/`, one `algo.py` with two `NAME`s, `rainbow` and `btr`, over a `net.py` that assembles the trunk from flags. Not two packages: BTR is Rainbow with different flags and a different head, and the flags are the experiment |
| the head | **exists** (A, built 2026-09-17) | Group A's `algos/dist/heads.py` -- `Categorical` for Rainbow, `Implicit` for BTR. Nothing distributional is written here |
| the loss | **exists** (A; Munchausen knobs built with A6, batch b40 queued 2026-09-19) | Group A's losses, with Munchausen's knobs from A6 available to both rows (on for BTR, off for Rainbow, as the papers have them). **BTR drops double-Q**: with Munchausen's soft target there is no separate argmax to decouple, and the paper's Table 1 lists it as removed; `SNEK_RAINBOW_DOUBLE` (1 for `rainbow`, 0 for `btr`) |
| replay, collection, schedules | **exists** | `algos/dqn/`'s, including n-step (already `SNEK_N_STEP_UPDATE`) and PER (already on). `algos/dqn/` does not change |
| the sidecar | `head` exists; `trunk` not built | `head` from Group A plus `trunk`: `{"dueling": true, "noisy": true, "residual": false}` for Rainbow or `{"dueling": true, "noisy": true, "residual": true, "blocks": 3, "spectral": true, "layer_norm": false}` for BTR. In the signature |
| restore | not built | two entries; both greedy over the head's mean, noise off at act time |
| the step | exists for D1; D2's `collect_envs`-per-step change not built | DQN's |
| the ε floor | not built | `SNEK_RAINBOW_NOISY=1` lifts `EPSILON_HARD_FLOOR` so `min_epsilon` 0 is legal; refused as before with noisy off (decided 2026-09-20, above) |

## 2. The rows

### D1 -- Rainbow (Hessel et al. 2018)

C51 plus double Q, dueling, prioritised replay, n-step (3), and noisy nets in place of ε-greedy. On this
codebase four of the six are on already, so the row is dueling and noisy nets composed with A2's head.

| piece | here |
|---|---|
| double Q | `algos/dqn/agent.py`'s, unchanged |
| PER | `algos/dqn/replay.py`, `SNEK_PRIORITY_EXPONENT` 0.5, β 0.4 → 1.0 linearly over the whole run (`SNEK_BETA_ANNEAL_STEPS` = the cap), the KL loss as the priority |
| n-step | `SNEK_N_STEP_UPDATE=3`, already implemented in `collect.py`'s windows, and already correct across episode boundaries (the bug `collect.py` records snek2 shipping) |
| C51 head | `algos/dist/heads.Categorical` at the support A2 settled on |
| **dueling** | `algos/rainbow/net.py::DuelingTrunk`: shared hidden stack from `QNet`, then a value stream `(atoms,)` and an advantage stream `(actions, atoms)`, combined as V + A − mean_a(A) per atom. **Built here** (decided 2026-09-20): D1 owns it and B1 (`b-data-efficiency.md`) imports the scalar form; D1 adds the atom axis |
| **noisy nets** | `algos/rainbow/noisy.py::NoisyLinear`, factorised Gaussian noise (Fortunato et al. 2018), σ_0 = 0.5, resampled per forward in training, **zeroed for the greedy `policy_fn`**. New. Replaces the epsilon schedule and the shield: `SNEK_RAINBOW_NOISY=1` sets `initial_epsilon = min_epsilon = 0` and `guided_fraction = 0` and lifts the ε hard floor (§1); the knobs are still accepted so the rung can be run with ε-greedy for the ablation in §4 |

`build_config` is DQN's plus C51's plus `SNEK_RAINBOW_NOISY`, `SNEK_RAINBOW_NOISY_SIGMA`,
`SNEK_RAINBOW_DUELING`, `SNEK_RAINBOW_DOUBLE`. The paper's optimiser and replay values are the row's
defaults under DQN's knob names (§2b), so a spec states only what departs.

Tests: the dueling combine has zero mean advantage over actions per atom; with noise zeroed
`NoisyLinear` equals `nn.Linear` with its μ weights; two forwards in training mode with different noise
give different outputs and the same in eval mode; the greedy policy is deterministic across calls.
Mutants: the mean subtracted over the atom axis instead of the action axis, σ not zeroed at act time,
the value stream broadcast to the wrong axis.

### D2 -- Beyond the Rainbow (Clark, Towers, Evers & Hare 2024)

Rainbow rebuilt for one desktop (Clark, Towers, Evers & Hare, arXiv 2411.03820, verified 2026-09-16
against the paper's Table D6 and the authors' code): IQN in place of C51, Munchausen on and double-Q
therefore out, dueling kept, **noisy nets kept** (factorised, σ₀ 0.5, *alongside* an ε-greedy anneal),
PER kept at a lower exponent, n-step 3, an IMPALA-style residual CNN at width scale 2 with **spectral
normalisation on the residual convolutions only** (the paper tried it on the linear layers and found it
"dramatically worse"), **no layer norm** in the published results, 64 vectorised environments with one
gradient step of batch 256 per vectorised step, and a target copy every 500 gradient steps. Its Atari
result is the practical ceiling of the value family on a single box, which is why it is in the series.

**The trunk does not transfer literally.** The IMPALA CNN and its adaptive max-pooling exist for
pixels; on a 26-value vector there is no image to convolve. The substitution, stated so the row is
honest about what it tests: a residual MLP trunk of the same depth budget -- three residual blocks of
width `SNEK_FC_LAYERS`, each two linears with a ReLU between, **spectral normalisation on the two
linears inside each block** (the analogue of the paper's placement: on the residual path, not the
stem or the head), and no layer norm. If the trunk is what carries BTR, this is the fair version of
it for this observation; the paper's post-submission layer-norm result (positive, App. H) is the
`SNEK_BTR_LAYER_NORM=1` ablation and not the paper cell.

| piece | here |
|---|---|
| head | `algos/dist/heads.Implicit` (A4), **N = N′ = K = 8** (the paper's "IQN taus 8"; the code uses one 8-sample pass for online, target and acting), 64 cosines, κ 1 |
| Munchausen | A6's knobs, α 0.9, τ 0.03, l₀ −1; `SNEK_RAINBOW_DOUBLE=0` |
| dueling, noisy | D1's `DuelingTrunk` and `NoisyLinear` over the residual trunk, both on |
| trunk | `algos/rainbow/net.py::ResidualTrunk`, `SNEK_BTR_BLOCKS` 3, `SNEK_BTR_SPECTRAL_NORM` 1 (inside the blocks), `SNEK_BTR_LAYER_NORM` 0. New |
| collection | `SNEK_COLLECT_ENVS=64`, `SNEK_BATCH_SIZE=256`, `SNEK_N_STEP_UPDATE=3`, **one gradient step per vectorised step** (`SNEK_REPLAY_RATIO` for 256 / 64 = 4 replayed samples per transition, half Rainbow's 8), replay 2²⁰ transitions, 200k transitions before the first update |
| optimiser | Adam lr 1e-4, ε 0.005 / 256 = 1.95e-5, gradient norm clipped at 10 |
| target | hard copy every 500 gradient steps (= 32k moves at 64 envs) |
| PER | α 0.2; β 0.45 held (the authors' code does not anneal it) |
| exploration | noisy nets **and** ε linear 1.0 → 0.01 over the first 2M agent steps (`SNEK_EPSILON_SCHEDULE=linear`, A's knob), then ε 0 for the second half of the run (`SNEK_EPSILON_ZERO_AT`, a fraction of the cap, 0.5); the fork off (`SNEK_FORK_BRANCHES=1`), since BTR's collection is wide rather than forked, and this is the one place D2's step changes: one counted step is `collect_envs` moves, and `advance()` reports it |
| discount | **0.997**, the paper's (`README.md`, "Translating") |

Tests: spectral norm bounds the largest singular value at 1 ± tolerance after the power iteration
converges; the residual block is the identity at zero-initialised final layers; with every flag at
Rainbow's values the `btr` net equals the `rainbow` net weight for weight (the ablation in §4 depends on
it); with `SNEK_RAINBOW_DOUBLE=0` the target's action is the target net's own argmax. Mutants: the
residual add dropped, spectral norm applied to the head, the wrong τ count on the target, ε not zeroed
at the fraction.

## 2b. The papers' settings, and how each lands here

| setting | Rainbow (Hessel et al. 2018, Table 1; Dopamine `rainbow.gin` for what the paper inherits) | BTR (Table D6; authors' code) | here |
|---|---|---|---|
| optimiser | Adam 6.25e-5, ε 1.5e-4 | Adam 1e-4, ε 1.95e-5 | the paper's, per row |
| batch, replay | 32; 1M; 20k steps before learning | 256; 2²⁰; 200k transitions before learning | the paper's, per row |
| update frequency | one update per 4 agent steps: 8 samples per transition | one update per 64-env step: 4 samples per transition | `SNEK_REPLAY_RATIO` 0.25 at batch 32 (Rainbow) and 1/64 ≈ 0.0156 at batch 256 (BTR); the knob is gradient steps per transition |
| target period | 8,000 agent steps = 2,000 gradient updates (32k frames) | 500 gradient steps | `SNEK_TARGET_UPDATE_PERIOD` 2,000 / 500 -- the knob counts gradient updates |
| PER | α 0.5, β 0.4 → 1 | α 0.2, β 0.45 held | the paper's |
| n-step | 3 | 3 | 3 |
| distribution | C51, 51 atoms, [−10, 10] | IQN, 8 taus | A2's stable support; IQN 8 / 8 / 8 |
| double-Q | on | off | per row |
| noisy nets | on, σ₀ 0.5, ε 0 | on, σ₀ 0.5, plus ε 1 → 0.01 over 2M steps, 0 after half the run | per row |
| dueling | on | on | on |
| Munchausen | -- | α 0.9, τ 0.03, l₀ −1 | per row |
| gradient clipping | none stated (the dueling paper: norm 10) | norm 10 | per row |
| discount | 0.99 | 0.997 | the paper's, per row |
| trunk | Nature CNN → 512 | IMPALA ×2, spectral norm on residual convs, adaptive maxpool 6×6, linear 512, no layer norm | `fc 320` for Rainbow; the residual MLP above for BTR |
| environments | 1 | 64 vectorised | 1 (`collect_envs` 1, no fork) for Rainbow; 64 for BTR |
| frames | 200M | 200M | 50M moves a cell, raised if still rising |
| reward | clipped [−1, 1] | clipped [−1, 1] | not clipped; supports from the return range |

Rainbow's **local** cell is D1 on `algos/dqn/`'s replay, target and collection defaults (the same split as
Group A). **BTR's local cell** (decided 2026-09-20) keeps the paper's stack -- residual trunk, dueling, noisy,
Munchausen, no double-Q -- on what Group A's local cells found better: `algos/dqn/`'s replay, target and
collection defaults (fork, shield, target period 8, replay ratio 1, batch 128, `collect_envs` 1) and **A3's
QR-DQN N 32 head** in place of IQN 8, the head that plateaued at 55-65% in A4. The paper cell's wide
collection is therefore one of the things the two cells differ in, and the D2 ablation is what separates it.

## 3. The batches

| batch | status | arms | base | read against | judged on |
|---|---|---|---|---|---|
| D1 | **waiting**: on `DuelingTrunk` and `NoisyLinear` (both D's, not built) and on A6 closing (gate 4); **the paper cell is kept** (decided 2026-09-20, above) | 4 seeds `rainbow` **paper** (§2b) + 4 seeds `rainbow` **local** (A1's local plumbing under the Rainbow head and flags) | the reference's reward, history and trunk | A2's two cells and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns |
| D1 ablation | **waiting** on D1 | 4 seeds paper with noisy off (ε 1 → 0.01 over the first **250k frames = 62.5k moves**, the paper's own non-noisy ablation) | D1 paper | D1 paper | whether noisy nets matter here |
| D2 | **waiting** on D1 closing; `ResidualTrunk` and the wide-collection step not built | 4 seeds `btr` **paper** (§2b, the paper's collection, IQN 8) + 4 seeds `btr` **local** (§2b's last paragraph: A's local plumbing and the QR-DQN N 32 head) | the reference's reward and history | paper: D1 paper and A4 + A6; local: D1 local and A3 + A6 | as D1 |
| D2 ablation | **waiting** on D2 | 4 seeds with the plain `QNet` trunk in place of the residual one + 4 seeds with spectral norm off; `SNEK_BTR_LAYER_NORM=1` (the paper's post-submission variant) as a third pair if the wave has room | D2 | D2 | is the trunk the difference, and is it the norm |

The two ablations are what make the group readable: without them D2 minus D1 is one number with four
changes behind it. They are cheap because they are flag flips on arms that have already been tuned.

## 4. Gates

**Where each row stands** (2026-09-20; a cell is filled in the pass that runs the gate, and an empty cell means
*not run*, not *passed*):

| gate | D1 Rainbow | D1 ablation | D2 BTR | D2 ablation |
|---|---|---|---|---|
| 1 smoke, checkpoint, restore (both names) | not run (nothing to run) | -- | not run | -- |
| 2 `btr` equals `rainbow` at Rainbow's flags | -- | -- | not run | -- |
| 3 mutation spec kills every mutant | no spec yet | -- | no spec yet | -- |
| 4 predecessor closed; tuning wave done | A2 closed 2026-09-18; **A6 open** (b40 live); tuning wave not run | D1 not closed | D1 not closed | D2 not closed |
| 5 `SNEK_MIN_EPSILON=0` refused with noisy off, accepted with noisy on | not run | -- | not run | -- |

1. Smoke as in `a-return-tail.md` §4, for both names.
2. The `btr`-equals-`rainbow` fixture at Rainbow's flags passes, so the ablations are exact.
3. The mutation spec kills every mutant.
4. D1 is not queued until A2's stable support and A6's Munchausen result are closed; D2 not until D1
   is. Tuning budget: one laptop wave per row for the learning rate and the replay ratio, then the row
   is queued as is or closed.
5. The ε floor test: `build_config` refuses `SNEK_MIN_EPSILON=0` with `SNEK_RAINBOW_NOISY=0` and
   accepts it with `SNEK_RAINBOW_NOISY=1`.

## 5. What would change the plan

- **D1 is level with A2.** Dueling and noisy nets add nothing here; D2 still runs with both, as the paper
  has them, and the D1 ablation's number is what the noisy result is read against.
- **D2 beats D1 by more than A4 + A6 beat A2.** The trunk ablation decides whether it is the trunk or
  the collection; whichever it is becomes a knob offered to every later value row, and to Group E's
  R2D2.
- **D2 beats the PPO reference on the 30k top.** The series has a new incumbent for the rows after it,
  and `hof-promote` runs on its checkpoint under the standing protocol.
