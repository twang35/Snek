# Group C: the value stack -- Rainbow, Beyond the Rainbow

**Status: built 2026-09-20, reworked 2026-09-23 after an external review (the table below), nothing queued** (`algos/rainbow/`: `noisy.py`, `net.py`, `agent.py`, `algo.py`, the names `rainbow` and `btr` in
`train.ALGOS` and `tools/restore.py`; `tests/test_rainbow.py`, mutants `tests/mut_rainbow.json`, 28 of 28 killed after the 2026-09-23 rework). Gates 1, 2, 3 and 5 passed the same day
(§4); **C1 is not queued until B2 has been queued** (decided 2026-09-20) and its tuning wave has run -- **B2 was queued as b42 later the same
day** (`docs/runs.md`), so under the queue-what-does-not-depend rule C1 is now queueable and is the next batch to write. Originally: planned 2026-09-16, nothing built. **Where each piece and each row stands is the `status` column of §1 and §3 and the gate table of §4** (added 2026-09-20, on the same rule as `a-return-tail.md`: a cell is filled in the pass that does the work, and an empty cell means *not done*, not *passed*). Group C of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 4 of the running order, after Group B (SAC); waits for Group A to close -- as of 2026-09-20 A1-A5 are closed (b39 closed today: FQF climbs where IQN flattened but never holds, stage B empty) and A6 (b40) is live on the laptop.

**Decided 2026-09-20, before implementation** (the questions raised when the plan was re-read against the code):

| decision | what it changes |
|---|---|
| **the noisy flag lifts the ε floor** | `build_config` refuses `SNEK_MIN_EPSILON` below `EPSILON_HARD_FLOOR` (1e-4, `algos/dqn/schedules.py`), so C1's ε 0 and C2's second-half ε 0 would be illegal. With `SNEK_RAINBOW_NOISY=1` the floor is lifted, since the noise supplies the exploration the floor protects; with noisy off it still holds |
| **C2 runs a paper cell and a local cell** | the paper's IQN 8 head plateaued at 55-65% in A4 with no stage-B checkpoint, so C2 as the paper alone might only repeat A4. The second cell of the wave is BTR's stack on the settings Group A's local cells found better, with A3's QR-DQN N 32 head (§2b, last paragraph); the layer-norm variant moves to the C2 ablation |
| **C1's paper cell is kept** | Group A's rule drops the paper cell after two rows show the same sign, and every A paper cell trailed. C1's is kept for 4 seeds because it is the only cell where noisy nets replace the shield and the fork, which is the row's question; revisited if it reads like A1 paper |
| **C builds `DuelingTrunk`** | it was D1's (BBF, then phase 1, unbuilt; now phase 5, after this group). C owns it in `algos/rainbow/net.py`; D1 imports the scalar form if its second wave wants dueling |

**Decided 2026-09-20, at implementation** (the questions the code raised):

| decision | what it changes |
|---|---|
| **the noise is on while collecting** | the plan said "zeroed for the greedy `policy_fn`", but the collector acts through the agent's greedy read, so with epsilon 0 that would have left the paper cell with no exploration at all. `RainbowAgent.greedy_actions` is a noisy forward, a fresh draw per act, as Rainbow and Dopamine act; stage A, restore, `watch.py` and every pass read the `mu` weights (`net.greedy_policy_fn`, noise off inside every call) |
| **a separate package, and `algos/dqn/`, `algos/dist/` are imported, not edited** | Group A's heads build their own `QNet` and read features by walking it, so dueling and a residual trunk could not be composed onto them from outside. `algos/rainbow/net.py` is its own network offering the reads `DistAgent` calls, so the losses are still A's to the line; `build_config` is this module's because the defaults are the papers' and the epsilon floor is conditional on the noisy flag |
| **the paper cell is as close to the paper as the game allows** | every `rainbow` and `btr` default is the paper's value from §2b; a spec states only what departs. Two names, one module, so `btr` at Rainbow's flags is `rainbow` weight for weight (gate 2) |
| **the target net stays in eval mode** | a spectral-normed linear runs a power iteration on every training-mode forward, so a target in train mode drifted between copies; its `_u`/`_v` arrive with each hard copy and its normalised weight is the online net's at the copy. The power-iteration vectors are drawn from the arm's seeded generator, so a seed pins the whole state dict |
| **`SNEK_RAINBOW_EPSILON_ZERO_AT` is a fraction of the run's moves** | `max_steps` x lanes; 0.5 for `btr`, 0 (never) for `rainbow`. With the eval-driven schedule it does nothing |
| **the knob-naming test covers every registered algorithm** | it read `train.py` and `algos/dqn/algo.py` only, so the dist and SAC knobs were never checked; now parametrised over `train.ALGOS` and reading every module under `algos/` |

**Decided 2026-09-23, after an external review of the build against both papers and the BTR code** (`github.com/VIPTankz/BTR`, read the same day). Nothing had been queued on either name, so every change below is to the cells before they run; `mut_rainbow.json` is 28 / 28:

| finding | what it changes |
|---|---|
| **spectral norm started unconverged** | `ResidualBlock` overwrote PyTorch's power-iteration vectors with fresh random ones and never iterated, so norms were 5 to 60 at build; the eval-mode target kept them until its first hard copy (500 updates in the paper cell), and the early TD errors raised `max_priority` for the whole run. Now PyTorch's own 15 iterations run on a seed forked from the arm's generator. The old test hid it by running 30 training forwards first |
| **the streams had no hidden layer** | Rainbow Table 4 and BTR (`fc1V`/`fc1A`, noisy, 512) give each dueling stream a hidden layer, both noisy. `SNEK_RAINBOW_STREAM_HIDDEN=1` (the default for both names): the plain trunk's last width moves into each stream, so a single-stream net is still `QNet` weight for weight; the residual trunk keeps its width and each stream adds one of it. BBF keeps the one-linear layout |
| **Munchausen's quantile target was Group A's mixture** | `algos/dist/` builds all `A x M` shifted samples weighted `pi / M`; M-IQN and BTR average actions inside each sample, `M` targets. `RainbowAgent` now has **its own update** with the paper's form, `pi` read off the same target samples as BTR's code does. Group A is untouched, so A6 stays reproducible -- and A6 ran the mixture (`a-return-tail.md` §2, A6) |
| **the quantile loss averaged the online quantiles** | the papers and BTR sum them (mean over the targets), `N` times Group A's; it matters through the gradient clip at 10 and Adam's epsilon. The rainbow update sums |
| **priorities were the loss** | Rainbow's is the KL, `CE - H(target)` (a matched distribution scored `ln 5` as CE); BTR's is the pairwise \|TD\| summed over online, averaged over target. Both now |
| **BTR's paper and code disagree on three settings** | **rule: `btr` follows the released code where the two disagree**, since the code produced the published numbers; `rainbow` follows the paper. So PER's importance exponent is **0.2** (`PER.py` uses `alpha` for it, "an accident but actually performed better"), not the declared 0.45; epsilon decays **geometrically**, `eps -= (eps - 0.01) / 2M` per move (about 0.37 at 2M, not 0.01; `SNEK_RAINBOW_EPSILON_DECAY`); Munchausen's `tau log pi(a|s)` is read off the **online** net (`SNEK_RAINBOW_MUNCHAUSEN_LOGPI`; Vieillard et al. use the target) |
| **Rainbow's beta was tied to a 50M-move run** | at the default 10M cap it ended at 0.52. `SNEK_BETA_ANNEAL_STEPS=0` (Rainbow's default) now means the run's cap, `max_steps` x lanes x replay ratio updates |
| **kept, and stated** | importance weights **mean-normalised** (`algos/dqn/replay.py`, measured in snek2) rather than max; both papers normalise by the max (BTR per batch). The warmup is fully random and off the epsilon clock, where BTR acts under the schedule from move 0 (about 0.9 by 200k); a 10% shift of a 2M time constant. An n-step row, not a move, advances the gradient clock, which matches BTR's one update per vector step on average |

The question: does the strongest single-box stack of value-learning tricks beat the PPO incumbent on
this game? C is read against A. If C2 beats C1 by about what IQN plus Munchausen beat C51 by in Group A,
the stack adds nothing beyond its parts. If it beats it by more, the trunk or the collection is doing
work worth isolating afterwards.

## 1. What the group shares

Both rows are compositions of pieces that exist by the time they run. The implementation work is the
composition and the one ingredient nothing earlier builds: **noisy nets**. Dueling arrives with D1 in phase 1.

| shared piece | status | decision |
|---|---|---|
| package | **built 2026-09-20** | `algos/rainbow/`, one `algo.py` with two names, `rainbow` and `btr` (each a thin module, as the registry maps `module.NAME`), over a `net.py` that assembles the trunk from flags. Not two packages: BTR is Rainbow with different flags and a different head, and the flags are the experiment |
| the head | **built** (in `algos/rainbow/net.py`, in Group A's formats) | c51 for Rainbow, IQN for BTR, read in the shapes `DistAgent` calls |
| the loss | **built 2026-09-23** (`RainbowAgent.update`) | Group A's projection and quantile Huber, with the papers' Munchausen target, loss reduction and priorities rather than Group A's (§ decided 2026-09-23, above); Munchausen's knobs from A6 (on for BTR, off for Rainbow, as the papers have them). **BTR drops double-Q**: with Munchausen's soft target there is no separate argmax to decouple, and the paper's Table 1 lists it as removed; `SNEK_RAINBOW_DOUBLE` (1 for `rainbow`, 0 for `btr`) |
| replay, collection, schedules | **exists** | `algos/dqn/`'s, including n-step (already `SNEK_N_STEP_UPDATE`) and PER (already on). `algos/dqn/` does not change |
| the sidecar | **built** (`tools/arch.py` `OPTIONAL_FIELDS`, in the signature) | `head` from Group A plus `trunk`: `{"dueling": true, "noisy": true, "residual": false}` for Rainbow or `{"dueling": true, "noisy": true, "residual": true, "blocks": 3, "spectral": true, "layer_norm": false}` for BTR. In the signature |
| restore | **built** | two entries; both greedy over the head's mean, noise off |
| the step | **exists for both** | DQN's: one counted step is one `collector.step()`, which at `collect_envs` 64 is 64 moves, and `advance()` already reports both |
| the ε floor | **built** | `SNEK_RAINBOW_NOISY=1` lifts `EPSILON_HARD_FLOOR` so `min_epsilon` 0 is legal; refused as before with noisy off (decided 2026-09-20, above). In `algos/rainbow/algo.py`'s own `build_config`, since DQN's validation cannot see the flag |

## 2. The rows

### C1 -- Rainbow (Hessel et al. 2018)

C51 plus double Q, dueling, prioritised replay, n-step (3), and noisy nets in place of ε-greedy. On this
codebase four of the six are on already, so the row is dueling and noisy nets composed with A2's head.

| piece | here |
|---|---|
| double Q | `algos/dqn/agent.py`'s, unchanged |
| PER | `algos/dqn/replay.py`, `SNEK_PRIORITY_EXPONENT` 0.5, β 0.4 → 1.0 linearly over the whole run (`SNEK_BETA_ANNEAL_STEPS=0`, resolved to the cap's updates), the KL as the priority |
| n-step | `SNEK_N_STEP_UPDATE=3`, already implemented in `collect.py`'s windows, and already correct across episode boundaries (the bug `collect.py` records snek2 shipping) |
| C51 head | `algos/dist/heads.Categorical` at the support A2 settled on |
| **dueling** | `algos/rainbow/net.py::RainbowNet`: shared hidden stack from `QNet` less its last layer, then a value stream `(atoms,)` and an advantage stream `(actions, atoms)`, each a hidden layer of that last width and an output, combined as V + A − mean_a(A) per atom. **Built here** (decided 2026-09-20): C1 owns it and D1 (`d-data-efficiency.md`) imports the scalar form; C1 adds the atom axis |
| **noisy nets** | `algos/rainbow/noisy.py::NoisyLinear`, factorised Gaussian noise (Fortunato et al. 2018), σ_0 = 0.5, resampled per forward in training, **zeroed for the greedy `policy_fn`**. New. Replaces the epsilon schedule and the shield: `SNEK_RAINBOW_NOISY=1` sets `initial_epsilon = min_epsilon = 0` and `guided_fraction = 0` and lifts the ε hard floor (§1); the knobs are still accepted so the rung can be run with ε-greedy for the ablation in §4 |

`build_config` is DQN's plus C51's plus `SNEK_RAINBOW_NOISY`, `SNEK_RAINBOW_NOISY_SIGMA`,
`SNEK_RAINBOW_DUELING`, `SNEK_RAINBOW_DOUBLE`. The paper's optimiser and replay values are the row's
defaults under DQN's knob names (§2b), so a spec states only what departs.

Tests: the dueling combine has zero mean advantage over actions per atom; with noise zeroed
`NoisyLinear` equals `nn.Linear` with its μ weights; two forwards in training mode with different noise
give different outputs and the same in eval mode; the greedy policy is deterministic across calls.
Mutants: the mean subtracted over the atom axis instead of the action axis, σ not zeroed at act time,
the value stream broadcast to the wrong axis.

### C2 -- Beyond the Rainbow (Clark, Towers, Evers & Hare 2024)

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
| dueling, noisy | C1's streams (a hidden layer and an output each, both `NoisyLinear`) over the residual trunk, both on |
| trunk | `algos/rainbow/net.py::Trunk` with `residual`, `SNEK_BTR_BLOCKS` 3, `SNEK_BTR_SPECTRAL_NORM` 1 (inside the blocks), `SNEK_BTR_LAYER_NORM` 0. New |
| collection | `SNEK_COLLECT_ENVS=64`, `SNEK_BATCH_SIZE=256`, `SNEK_N_STEP_UPDATE=3`, **one gradient step per vectorised step** (`SNEK_REPLAY_RATIO` for 256 / 64 = 4 replayed samples per transition, half Rainbow's 8), replay 2²⁰ transitions, 200k transitions before the first update |
| optimiser | Adam lr 1e-4, ε 0.005 / 256 = 1.95e-5, gradient norm clipped at 10 |
| target | hard copy every 500 gradient steps (= 32k moves at 64 envs) |
| PER | α 0.2; importance exponent **0.2** held -- the code's (`PER.py` raises to `-alpha`); the paper declares 0.45 (decided 2026-09-23) |
| exploration | noisy nets **and** ε 1.0 → 0.01 **geometric** with time constant 2M agent steps, as the code (`SNEK_RAINBOW_EPSILON_DECAY=geometric`; about 0.37 at 2M) (`SNEK_EPSILON_SCHEDULE=linear`, A's knob), then ε 0 for the second half of the run (`SNEK_EPSILON_ZERO_AT`, a fraction of the cap, 0.5); the fork off (`SNEK_FORK_BRANCHES=1`), since BTR's collection is wide rather than forked, and this is the one place C2's step changes: one counted step is `collect_envs` moves, and `advance()` reports it |
| discount | **0.997**, the paper's (`README.md`, "Translating") |

Tests: spectral norm bounds the largest singular value at 1 ± tolerance **at build and in a fresh target**,
before any training forward (the first version of this test ran 30 forwards first and hid an unconverged
start); the residual block is the identity at zero-initialised final layers; with every flag at
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
| PER | α 0.5, β 0.4 → 1 | α 0.2; β declared 0.45, the code uses 0.2 (α) with batch-max weights | Rainbow the paper's; BTR the code's 0.2; mean-normalised weights for both |
| n-step | 3 | 3 | 3 |
| distribution | C51, 51 atoms, [−10, 10] | IQN, 8 taus | A2's stable support; IQN 8 / 8 / 8 |
| double-Q | on | off | per row |
| noisy nets | on, σ₀ 0.5, ε 0 | on, σ₀ 0.5, plus ε 1 → 0.01 geometric over 2M steps (the code), 0 after half the run | per row |
| dueling | on | on | on |
| Munchausen | -- | α 0.9, τ 0.03, l₀ −1; actions averaged inside each target quantile; log π(a\|s) from the online net (the code) | per row |
| gradient clipping | none stated (the dueling paper: norm 10) | norm 10 | per row |
| discount | 0.99 | 0.997 | the paper's, per row |
| trunk | Nature CNN, then a 512 hidden layer in each dueling stream | IMPALA ×2, spectral norm on residual convs, adaptive maxpool 6×6, then a noisy 512 hidden layer in each stream, no layer norm | `fc 320` for Rainbow, moved into each stream; the residual MLP above for BTR, with a 320 hidden layer per stream |
| environments | 1 | 64 vectorised | 1 (`collect_envs` 1, no fork) for Rainbow; 64 for BTR |
| frames | 200M | 200M | 50M moves a cell, raised if still rising |
| reward | clipped [−1, 1] | clipped [−1, 1] | not clipped; supports from the return range |

Rainbow's **local** cell is C1 on `algos/dqn/`'s replay, target and collection defaults (the same split as
Group A). **BTR's local cell** (decided 2026-09-20) keeps the paper's stack -- residual trunk, dueling, noisy,
Munchausen, no double-Q -- on what Group A's local cells found better: `algos/dqn/`'s replay, target and
collection defaults (fork, shield, target period 8, replay ratio 1, batch 128, `collect_envs` 1) and **A3's
QR-DQN N 32 head** in place of IQN 8, the head that plateaued at 55-65% in A4. The paper cell's wide
collection is therefore one of the things the two cells differ in, and the C2 ablation is what separates it.

## 3. The batches

| batch | status | arms | base | read against | judged on |
|---|---|---|---|---|---|
| C1 | **built, waiting to queue**: A6 (b40) closed 2026-09-20; **C1 waits for B2 to be queued** (decided 2026-09-20), then its tuning wave; **the paper cell is kept** (decided 2026-09-20, above) | 4 seeds `rainbow` **paper** (§2b) + 4 seeds `rainbow` **local** (A1's local plumbing under the Rainbow head and flags) | the reference's reward, history and trunk | A2's two cells and PPO's `hist8` table | stage-B density, `hof5000`, `hof30k`, drawdowns |
| C1 ablation | **waiting** on C1 | 4 seeds paper with noisy off (ε 1 → 0.01 over the first **250k frames = 62.5k moves**, the paper's own non-noisy ablation) | C1 paper | C1 paper | whether noisy nets matter here |
| C2 | **built** (`btr`, residual trunk, 64-lane collection); **waiting** on C1 closing | 4 seeds `btr` **paper** (§2b, the paper's collection, IQN 8) + 4 seeds `btr` **local** (§2b's last paragraph: A's local plumbing and the QR-DQN N 32 head) | the reference's reward and history | paper: C1 paper and A4 + A6; local: C1 local and A3 + A6 | as C1 |
| C2 ablation | **waiting** on C2 | 4 seeds with the plain `QNet` trunk in place of the residual one + 4 seeds with spectral norm off; `SNEK_BTR_LAYER_NORM=1` (the paper's post-submission variant) as a third pair if the wave has room | C2 | C2 | is the trunk the difference, and is it the norm |

The two ablations are what make the group readable: without them C2 minus C1 is one number with four
changes behind it. They are cheap because they are flag flips on arms that have already been tuned.

## 4. Gates

**Where each row stands** (2026-09-20; a cell is filled in the pass that runs the gate, and an empty cell means
*not run*, not *passed*):

| gate | C1 Rainbow | C1 ablation | C2 BTR | C2 ablation |
|---|---|---|---|---|
| 1 smoke, checkpoint, restore (both names) | **passed 2026-09-20** (`rainbow-smoke`, 3,000 steps, 6 checkpoints, `tools.restore` loads the last); **passed again 2026-09-23 after the rework** (5,000 steps, 1,250 updates = 5,000 x 0.25, β's resolved length; resumed to 6,000; KL priorities, finite weights) | -- | **passed 2026-09-20** (`btr-smoke`, 64 lanes, 41 st/s on the laptop, 6 checkpoints, restored); **passed again 2026-09-23** with IQN (37 st/s) and with the local cell's QR-DQN N 32 head (84 st/s): 3,000 steps, 2,999 updates, ε 0.98428 at step 500 = the geometric formula, 0 from move 96,000, block norms 1.00-1.01 in online and target | -- |
| 2 `btr` equals `rainbow` at Rainbow's flags | -- | -- | **passed** (`test_btr_at_rainbows_flags_is_rainbow_weight_for_weight`) | -- |
| 3 mutation spec kills every mutant | **passed**, 14/14 (`tests/mut_rainbow.json`); **28/28** after the 2026-09-23 rework | -- | same spec | -- |
| 4 predecessor closed; tuning wave done | A2 closed 2026-09-18; A6 closed 2026-09-20 (b40); **B2 not yet queued**; tuning wave not run | C1 not closed | C1 not closed | C2 not closed |
| 5 `SNEK_MIN_EPSILON=0` refused with noisy off, accepted with noisy on | **passed** (`test_min_epsilon_zero_is_refused_with_noisy_off_and_accepted_with_it_on`) | -- | same test | -- |

1. Smoke as in `a-return-tail.md` §4, for both names.
2. The `btr`-equals-`rainbow` fixture at Rainbow's flags passes, so the ablations are exact.
3. The mutation spec kills every mutant.
4. C1 is not queued until A2's stable support and A6's Munchausen result are closed; C2 not until C1
   is. Tuning budget: one laptop wave per row for the learning rate and the replay ratio, then the row
   is queued as is or closed.
5. The ε floor test: `build_config` refuses `SNEK_MIN_EPSILON=0` with `SNEK_RAINBOW_NOISY=0` and
   accepts it with `SNEK_RAINBOW_NOISY=1`.

## 5. What would change the plan

- **C1 is level with A2.** Dueling and noisy nets add nothing here; C2 still runs with both, as the paper
  has them, and the C1 ablation's number is what the noisy result is read against.
- **C2 beats C1 by more than A4 + A6 beat A2.** The trunk ablation decides whether it is the trunk or
  the collection; whichever it is becomes a knob offered to every later value row, and to Group E's
  R2D2.
- **C2 beats the PPO reference on the 30k top.** The series has a new incumbent for the rows after it,
  and `hof-promote` runs on its checkpoint under the standing protocol.
