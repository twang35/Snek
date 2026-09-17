# Group B: the value stack -- Rainbow, Beyond the Rainbow

**Status: planned 2026-09-16, nothing built.** Group B of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 4 of the running order; waits for Group A to close.

The question: does the strongest single-box stack of value-learning tricks beat the PPO incumbent on
this game? B is read against A. If B2 beats B1 by about what IQN plus Munchausen beat C51 by in Group A,
the stack adds nothing beyond its parts. If it beats it by more, the trunk or the collection is doing
work worth isolating afterwards.

## 1. What the group shares

Both rows are compositions of pieces that exist by the time they run. The implementation work is the
composition and the one ingredient nothing earlier builds: **noisy nets**. Dueling arrives with F1 in phase 1.

| shared piece | decision |
|---|---|
| package | `algos/rainbow/`, one `algo.py` with two `NAME`s, `rainbow` and `btr`, over a `net.py` that assembles the trunk from flags. Not two packages: BTR is Rainbow with different flags and a different head, and the flags are the experiment |
| the head | Group A's `algos/dist/heads.py` -- `Categorical` for Rainbow, `Implicit` for BTR. Nothing distributional is written here |
| the loss | Group A's losses, with Munchausen's knobs from A6 available to both rows (on for BTR, off for Rainbow, as the papers have them). **BTR drops double-Q**: with Munchausen's soft target there is no separate argmax to decouple, and the paper's Table 1 lists it as removed; `SNEK_RAINBOW_DOUBLE` (1 for `rainbow`, 0 for `btr`) |
| replay, collection, schedules | `algos/dqn/`'s, including n-step (already `SNEK_N_STEP_UPDATE`) and PER (already on). `algos/dqn/` does not change |
| the sidecar | `head` from Group A plus `trunk`: `{"dueling": true, "noisy": true, "residual": false}` for Rainbow or `{"dueling": true, "noisy": true, "residual": true, "blocks": 3, "spectral": true, "layer_norm": false}` for BTR. In the signature |
| restore | two entries; both greedy over the head's mean, noise off at act time |
| the step | DQN's |

## 2. The rows

### B1 -- Rainbow (Hessel et al. 2018)

C51 plus double Q, dueling, prioritised replay, n-step (3), and noisy nets in place of ε-greedy. On this
codebase four of the six are on already, so the row is dueling and noisy nets composed with A2's head.

| piece | here |
|---|---|
| double Q | `algos/dqn/agent.py`'s, unchanged |
| PER | `algos/dqn/replay.py`, `SNEK_PRIORITY_EXPONENT` 0.5, β 0.4 → 1.0 linearly over the whole run (`SNEK_BETA_ANNEAL_STEPS` = the cap), the KL loss as the priority |
| n-step | `SNEK_N_STEP_UPDATE=3`, already implemented in `collect.py`'s windows, and already correct across episode boundaries (the bug `collect.py` records snek2 shipping) |
| C51 head | `algos/dist/heads.Categorical` at the support A2 settled on |
| **dueling** | `algos/rainbow/net.py::DuelingTrunk`: shared hidden stack from `QNet`, then a value stream `(atoms,)` and an advantage stream `(actions, atoms)`, combined as V + A − mean_a(A) per atom. **Built by F1** (`f-data-efficiency.md`, phase 1, scalar form); B1 adds the atom axis |
| **noisy nets** | `algos/rainbow/noisy.py::NoisyLinear`, factorised Gaussian noise (Fortunato et al. 2018), σ_0 = 0.5, resampled per forward in training, **zeroed for the greedy `policy_fn`**. New. Replaces the epsilon schedule and the shield: `SNEK_RAINBOW_NOISY=1` sets `initial_epsilon = min_epsilon = 0` and `guided_fraction = 0`; the knobs are still accepted so the rung can be run with ε-greedy for the ablation in §4 |

`build_config` is DQN's plus C51's plus `SNEK_RAINBOW_NOISY`, `SNEK_RAINBOW_NOISY_SIGMA`,
`SNEK_RAINBOW_DUELING`, `SNEK_RAINBOW_DOUBLE`. The paper's optimiser and replay values are the row's
defaults under DQN's knob names (§2b), so a spec states only what departs.

Tests: the dueling combine has zero mean advantage over actions per atom; with noise zeroed
`NoisyLinear` equals `nn.Linear` with its μ weights; two forwards in training mode with different noise
give different outputs and the same in eval mode; the greedy policy is deterministic across calls.
Mutants: the mean subtracted over the atom axis instead of the action axis, σ not zeroed at act time,
the value stream broadcast to the wrong axis.

### B2 -- Beyond the Rainbow (Clark, Towers, Evers & Hare 2024)

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
| dueling, noisy | B1's `DuelingTrunk` and `NoisyLinear` over the residual trunk, both on |
| trunk | `algos/rainbow/net.py::ResidualTrunk`, `SNEK_BTR_BLOCKS` 3, `SNEK_BTR_SPECTRAL_NORM` 1 (inside the blocks), `SNEK_BTR_LAYER_NORM` 0. New |
| collection | `SNEK_COLLECT_ENVS=64`, `SNEK_BATCH_SIZE=256`, `SNEK_N_STEP_UPDATE=3`, **one gradient step per vectorised step** (`SNEK_REPLAY_RATIO` for 256 / 64 = 4 replayed samples per transition, half Rainbow's 8), replay 2²⁰ transitions, 200k transitions before the first update |
| optimiser | Adam lr 1e-4, ε 0.005 / 256 = 1.95e-5, gradient norm clipped at 10 |
| target | hard copy every 500 gradient steps (= 32k moves at 64 envs) |
| PER | α 0.2; β 0.45 held (the authors' code does not anneal it) |
| exploration | noisy nets **and** ε linear 1.0 → 0.01 over the first 2M agent steps (`SNEK_EPSILON_SCHEDULE=linear`, A's knob), then ε 0 for the second half of the run (`SNEK_EPSILON_ZERO_AT`, a fraction of the cap, 0.5); the fork off (`SNEK_FORK_BRANCHES=1`), since BTR's collection is wide rather than forked, and this is the one place B2's step changes: one counted step is `collect_envs` moves, and `advance()` reports it |
| discount | 0.99, not the paper's 0.997 (`README.md`, "Translating") |

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
| discount | 0.99 | 0.997 | 0.99 |
| trunk | Nature CNN → 512 | IMPALA ×2, spectral norm on residual convs, adaptive maxpool 6×6, linear 512, no layer norm | `fc 320` for Rainbow; the residual MLP above for BTR |
| environments | 1 | 64 vectorised | 1 (`collect_envs` 1, no fork) for Rainbow; 64 for BTR |
| frames | 200M | 200M | 50M moves a cell, raised if still rising |
| reward | clipped [−1, 1] | clipped [−1, 1] | not clipped; supports from the return range |

Rainbow's **local** cell is B1 on `algos/dqn/`'s replay, target and collection defaults (the same split as
Group A); BTR has no local cell, because its collection *is* one of the things it changes.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| B1 | 4 seeds `rainbow` **paper** (§2b) + 4 seeds `rainbow` **local** (A1's local plumbing under the Rainbow head and flags) | the reference's reward, history and trunk | A2's two cells and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns |
| B1 ablation | 4 seeds paper with noisy off (ε 1 → 0.01 over 250k steps, the paper's own non-noisy ablation) | B1 paper | B1 paper | whether noisy nets matter here |
| B2 | 4 seeds `btr` (§2b, the paper's collection) + 4 seeds `btr` with `SNEK_BTR_LAYER_NORM=1` (the paper's post-submission variant) | the reference's reward and history | B1 paper and A4 + A6 | as B1 |
| B2 ablation | 4 seeds with the plain `QNet` trunk in place of the residual one + 4 seeds with spectral norm off | B2 | B2 | is the trunk the difference, and is it the norm |

The two ablations are what make the group readable: without them B2 minus B1 is one number with four
changes behind it. They are cheap because they are flag flips on arms that have already been tuned.

## 4. Gates

1. Smoke as in `a-return-tail.md` §4, for both names.
2. The `btr`-equals-`rainbow` fixture at Rainbow's flags passes, so the ablations are exact.
3. The mutation spec kills every mutant.
4. B1 is not queued until A2's stable support and A6's Munchausen result are closed; B2 not until B1
   is. Tuning budget: one laptop wave per row for the learning rate and the replay ratio, then the row
   is queued as is or closed.

## 5. What would change the plan

- **B1 is level with A2.** Dueling and noisy nets add nothing here; B2 still runs with both, as the paper
  has them, and the B1 ablation's number is what the noisy result is read against.
- **B2 beats B1 by more than A4 + A6 beat A2.** The trunk ablation decides whether it is the trunk or
  the collection; whichever it is becomes a knob offered to every later value row, and to Group D's
  R2D2.
- **B2 beats the PPO reference on the 30k top.** The series has a new incumbent for the rows after it,
  and `hof-promote` runs on its checkpoint under the standing protocol.
