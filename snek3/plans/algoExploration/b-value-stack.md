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
| the loss | Group A's losses, with Munchausen's two knobs from A6 available to both rows (on for BTR, off for Rainbow, as the papers have them) |
| replay, collection, schedules | `algos/dqn/`'s, including n-step (already `SNEK_N_STEP_UPDATE`) and PER (already on). `algos/dqn/` does not change |
| the sidecar | `head` from Group A plus `trunk`: `{"dueling": true, "noisy": true, "norm": "none"}` or `{"dueling": true, "noisy": false, "norm": "layer", "spectral": true}`. In the signature |
| restore | two entries; both greedy over the head's mean, noise off at act time |
| the step | DQN's |

## 2. The rows

### B1 -- Rainbow (Hessel et al. 2018)

C51 plus double Q, dueling, prioritised replay, n-step (3), and noisy nets in place of ε-greedy. On this
codebase four of the six are on already, so the row is dueling and noisy nets composed with A2's head.

| piece | here |
|---|---|
| double Q | `algos/dqn/agent.py`'s, unchanged |
| PER | `algos/dqn/replay.py`, `SNEK_PRIORITY_EXPONENT` 0.5 and the paper's β schedule |
| n-step | `SNEK_N_STEP_UPDATE=3`, already implemented in `collect.py`'s windows, and already correct across episode boundaries (the bug `collect.py` records snek2 shipping) |
| C51 head | `algos/dist/heads.Categorical` at the support A2 settled on |
| **dueling** | `algos/rainbow/net.py::DuelingTrunk`: shared hidden stack from `QNet`, then a value stream `(atoms,)` and an advantage stream `(actions, atoms)`, combined as V + A − mean_a(A) per atom. **Built by F1** (`f-data-efficiency.md`, phase 1, scalar form); B1 adds the atom axis |
| **noisy nets** | `algos/rainbow/noisy.py::NoisyLinear`, factorised Gaussian noise (Fortunato et al. 2018), σ_0 = 0.5, resampled per forward in training, **zeroed for the greedy `policy_fn`**. New. Replaces the epsilon schedule and the shield: `SNEK_RAINBOW_NOISY=1` sets `initial_epsilon = min_epsilon = 0` and `guided_fraction = 0`; the knobs are still accepted so the rung can be run with ε-greedy for the ablation in §4 |

`build_config` is DQN's plus C51's plus `SNEK_RAINBOW_NOISY`, `SNEK_RAINBOW_NOISY_SIGMA`,
`SNEK_RAINBOW_DUELING`. Learning rate 6.25e-5 and Adam ε 1.5e-4 are the paper's and are set as the
row's defaults under DQN's knob names, so the spec states them.

Tests: the dueling combine has zero mean advantage over actions per atom; with noise zeroed
`NoisyLinear` equals `nn.Linear` with its μ weights; two forwards in training mode with different noise
give different outputs and the same in eval mode; the greedy policy is deterministic across calls.
Mutants: the mean subtracted over the atom axis instead of the action axis, σ not zeroed at act time,
the value stream broadcast to the wrong axis.

### B2 -- Beyond the Rainbow (Clark, Towers, Evers & Farquhar 2024)

Rainbow rebuilt for one desktop: IQN in place of C51, Munchausen on, dueling kept, noisy nets *off*,
PER and n-step kept, an IMPALA-style residual trunk with spectral normalisation, vectorised
collection with a larger batch, and a longer n-step. Its Atari result is the practical ceiling of the
value family on a single box, which is why it is in the series.

**The trunk does not transfer literally.** The IMPALA CNN and its adaptive max-pooling exist for
pixels; on a 26-value vector there is no image to convolve. The substitution, stated so the row is
honest about what it tests: a residual MLP trunk of the same depth budget (three residual blocks of
width `SNEK_FC_LAYERS`), with **spectral normalisation on each linear** and layer norm between blocks --
the two regularisers the paper credits for stability at a high replay ratio. If the trunk is what
carries BTR, this is the fair version of it for this observation.

| piece | here |
|---|---|
| head | `algos/dist/heads.Implicit` (A4), N_τ = 8 online / 8 target as in the paper |
| Munchausen | A6's knobs, α 0.9, τ 0.03 |
| dueling | B1's `DuelingTrunk`, over the residual trunk |
| trunk | `algos/rainbow/net.py::ResidualTrunk`, `SNEK_BTR_BLOCKS` 3, `SNEK_BTR_SPECTRAL_NORM` 1, `SNEK_BTR_LAYER_NORM` 1. New |
| collection | `SNEK_COLLECT_ENVS=64`, `SNEK_BATCH_SIZE=256`, `SNEK_N_STEP_UPDATE=3`, `SNEK_REPLAY_RATIO` set so gradient steps per transition match the paper's 1 per 4 (the knob is exact whether the fork is on or off, so this is one number) |
| exploration | ε-greedy from `algos/dqn/schedules.py`; the fork off (`SNEK_FORK_BRANCHES=1`), since BTR's collection is wide rather than forked, and this is the one place B2's step changes: one counted step is `collect_envs` moves, and `advance()` reports it |

Tests: spectral norm bounds the largest singular value at 1 ± tolerance after the power iteration
converges; the residual block is the identity at zero-initialised final layers; with every flag at
Rainbow's values the `btr` net equals the `rainbow` net weight for weight (the ablation in §4 depends on
it). Mutants: the residual add dropped, the norm applied after the activation, the wrong τ count on
the target.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| B1 | 4 seeds of `rainbow` | A1's config for everything not Rainbow's; the paper's optimiser values | A2 (C51) and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns |
| B1 ablation | 4 seeds noisy off (ε-greedy and the shield back on) | B1 | B1 | whether noisy nets matter on a game with a shield |
| B2 | 4 seeds of `btr` | A1's reward and history; the paper's collection values | B1 and A4 + A6 | as B1 |
| B2 ablation | 4 seeds with the plain `QNet` trunk in place of the residual one | B2 | B2 | is the trunk the difference |

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

- **B1 is level with A2.** Dueling and noisy nets add nothing here; B2 then runs with dueling only and
  the noisy code is left in place but off.
- **B2 beats B1 by more than A4 + A6 beat A2.** The trunk ablation decides whether it is the trunk or
  the collection; whichever it is becomes a knob offered to every later value row, and to Group D's
  R2D2.
- **B2 beats the PPO reference on the 30k top.** The series has a new incumbent for the rows after it,
  and `hof-promote` runs on its checkpoint under the standing protocol.
