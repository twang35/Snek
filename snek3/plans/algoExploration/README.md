# The algorithm series: plans

[`algorithm-series.md`](algorithm-series.md) is the ordering and the reasoning. Each group has one
implementation plan here, and every plan leans on the conventions below rather than restating them.

| plan | group | rows |
|---|---|---|
| [`a-return-tail.md`](a-return-tail.md) | A | DQN, C51, QR-DQN, IQN, FQF, Munchausen |
| [`b-entropy.md`](b-entropy.md) | B | Discrete SAC, Revisiting Discrete SAC |
| [`c-value-stack.md`](c-value-stack.md) | C | Rainbow, Beyond the Rainbow |
| [`d-data-efficiency.md`](d-data-efficiency.md) | D | BBF's resets, as the late-drift probe |
| [`e-memory.md`](e-memory.md) | E | recurrent PPO, R2D2 |
| [`f-exploration.md`](f-exploration.md) | F | NGU, Agent57 |
| [`g-planning.md`](g-planning.md) | G | AlphaZero-style MCTS, MuZero, EfficientZero V2, Muesli |
| [`h-gdi.md`](h-gdi.md) | H | GDI / LBC |

## How an algorithm lands in snek3

These are the rules the codebase already enforces; a plan says only where its algorithm departs.

| convention | rule | where it is enforced |
|---|---|---|
| one package | `algos/<name>/` with an `algo.py` exposing `NAME`, `build_config(tuned)`, `reportable(config)`, `build(config, arch, device)`; one entry in `train.ALGOS`; **never a second trainer** | `tests/test_train.py` asserts the seam over every registry entry |
| the seam | the fourteen members `algos/dqn/algo.py` documents: `step_granularity`, `prefill`, `advance`, `fields`, `on_eval`, `net`, `policy_fn`, `state_dict`/`load_state_dict`, `init_from`, `save_side_state`/`load_side_state`, `describe`, `log_note`, `log_extra` | same |
| knobs | every config key is its `SNEK_` variable lowercased, prefixed with the algorithm's name where the meaning is not shared (`SNEK_PPO_LEARNING_RATE`); `COLLECT_ENVS` and `DISCOUNT` are shared names. **Another algorithm's knobs are refused by name, never ignored** (`algos/ppo/algo.py` §2) | `test_every_config_key_is_its_knob_lowercased`; a `REJECTED` tuple per module |
| the sidecar | `arch.json` carries `algo`, `fc_layer_params`, `num_actions`, `obs_len`, `obs_era`. An algorithm whose network has more shape than that (atoms, quantiles, a recurrent width) **adds a field**, and `tools/arch.py`'s signature includes it, so a checkpoint cannot load into the wrong head silently | `tools/arch.py` `write_arch` refuses a different signature |
| restore | one entry in `tools/restore.ALGORITHMS` returning a module with `build(arch, device)` and a greedy policy. Every shard, `watch.py`, `record_gif.py` and the HOF tooling load through it and nothing else | `tools/restore.py` |
| the policy seam | `policy_fn` is `(m, obs_len) float32 -> (m,) int64`, stateless, over the rows the engine hands it. `vectorized/` imports no torch. **Two groups need more than this** (D and E need per-row state, G needs the board) and the extension is designed once, in `e-memory.md` §1 and `g-planning.md` §1, and reused | `tests/test_module_layering.py` |
| a step | `step_granularity` says what a counted step is; `advance()` returns `(steps, transitions)` and every row carries `transitions`. A plan states what its step is, because `SNEK_MAX_STEPS`, `SNEK_EVAL_INTERVAL` and every chart's x-axis read it | `algos/ppo/algo.py` §1 |
| tests | a `tests/test_<name>_*.py` per module with the arithmetic pinned (a loss on a hand-worked batch, a target on a known transition), and a `tests/mut_<name>.json` mutation spec whose mutants all die before the row is queued | `skills/mutation-test` |
| the batch | 4 seeds a cell, seed pinned to the arm letter, the current PPO reference config for everything the algorithm does not own (reward preset, `SNEK_OBS_HISTORY`, `SNEK_FC_LAYERS`), stage A at 100 episodes on every checkpoint, stage B, `hof5000`, `hof30k`. Judged on stage-B density, the depth passes and the drawdown count, never on a single eval | `docs/protocol.md` |
| **paper fidelity** | **every row's first cell is the paper's configuration**, translated to Snake by the rules in the next section and nothing else -- the paper's optimiser, learning rate, batch, replay size, target period, exploration schedule and network width, with this codebase's own additions (the fork, the shield, the eval-driven epsilon, the horizon anneal) **off**. A second cell, the **local** cell, runs the same algorithm on the codebase's tuned defaults where the two differ materially; it is what says whether a gap is the algorithm or the plumbing. The paper cell is the headline, the row's plan carries a "paper settings" table naming each knob's paper value, its translation and the reason where one does not transfer, and a spec that departs from the table says so in its manifest note | each plan's "paper settings" section; `docs/protocol.md` |
| the gate | a row starts when the row it is read against has a closed stage-B number; a row that does not stabilise within the tuning budget its plan names is closed as a finding | `algorithm-series.md` §3 |

## Translating a paper's setting to Snake

The rows are meant to emulate their papers, on this game. Most settings carry over as written; the
ones below do not, and every plan translates them the same way so the rows stay comparable.

| the paper says | here | why |
|---|---|---|
| **frames** (Atari, frame-skip 4) | one Snake move is one agent step, so **1 agent step = 4 frames** and a 200M-frame run is 50M moves; a 100k-step (400k-frame) data-efficiency budget is 100k moves. Budgets are stated in moves, and a plan converts once | the papers' budgets are in frames and their periods (target update, anneal, ε) are mostly in agent steps or gradient updates; mixing the units has cost this project before |
| **reward clipping to [−1, 1]** | **not applied.** Clipping would turn the +100 perfect game into +1, and b33 showed the win reward is a monotone onset lever that saturates at 100. Instead every scale that the papers set against a unit reward -- a categorical support, a Huber threshold, Munchausen's temperature, a value transform's range -- is set from **the discounted return the reward preset implies** (about [−6, 110] at γ 0.99, win 100, food 1, death −5), and the smoke asserts it brackets a prefill batch | the papers' scales are for clipped rewards; the equivalent operation here is scaling the head, not the game |
| **discount 0.99 / 0.995 / 0.997 / 0.9999, or none** | **the paper's, as written** (`SNEK_DISCOUNT` per row: 0.99 for the DQN family, Rainbow and SAC, 0.997 for BTR, R2D2, NGU, MuZero, EfficientZero and GDI, 0.995 for Muesli, BBF's 0.97 → 0.997 anneal, Agent57's ladder to 0.9999, AlphaZero's undiscounted outcome). The local cell takes the reference's value and anneal | decided 2026-09-16: the series compares algorithms on Snake, not algorithms on PPO's settings, and each gets the horizon its authors tuned for. A perfect game is 1,000-2,300 moves, so the papers' longer horizons are not unsuited to it. The one consequence to carry: every scale set from the discounted return range (a C51 support, MuZero's value support, R2D2's rescaling) is computed at **the row's own γ**, and the smoke asserts it brackets a prefill batch |
| **ε-greedy, 1.0 → 0.1 (or 0.01) over the first 1M frames, 0.001 at eval** | the paper's linear schedule over `SNEK_EPSILON_ANNEAL_STEPS` moves, then held; **the eval-driven `refine_epsilon` schedule, the shield (`SNEK_GUIDED_FRACTION`) and the fork (`SNEK_FORK_BRANCHES`) are off** in the paper cell and on in the local cell. Eval is greedy (ε 0) as every snek3 eval already is | none of the three exist in any paper; they are this codebase's answers to the endgame, and the local cell measures them |
| **a 1M-transition replay with a 50k (80k-frame) prefill** | **1M transitions** (`SNEK_REPLAY_BUFFER_MAX_LENGTH`) and the paper's prefill, at 26+16 float32 values a row about 170 MB -- affordable on both boxes | the codebase's 100k default was sized for the fork; the paper cell takes the paper's |
| **target network period 8k-10k updates (32k-40k frames)** | the paper's, in gradient updates | the codebase's `SNEK_TARGET_UPDATE_PERIOD` default of 8 is a hard copy every 8 updates and is the single largest departure from every value paper; it stays in the local cell only |
| **batch 32, one update per 4 agent steps** (8 replayed samples per transition) | batch 32, `SNEK_REPLAY_RATIO` set for 8 samples per transition | the paper's replay ratio is a load-bearing setting (BBF is *about* it), so the paper cell matches samples per transition rather than gradient steps |
| **a CNN trunk (Nature DQN, IMPALA, ResNet)** | the reference's `fc 320` MLP over the 26+16-value observation for the paper cell; a plan that needs a trunk with more shape (a residual stack, a wider net) states the MLP analogue and its budget | there is no image; the trunk substitution is stated per row so it is not mistaken for the paper's |
| **an LSTM of 512 (R2D2) or a 256-wide recurrent core** | the paper's width where the trunk is comparable, otherwise the width the plan states with the paper's as the reference | the observation is 42 values, not 3136 CNN features; a plan says which |
| **200M frames, 5 seeds; 100k steps, 10-50 seeds** | 4 seeds a cell as everywhere here; a plan budgets the moves per arm from the paper's frames when the paper's budget is the question (D1, G3) and from the reference's 100M transitions otherwise | the protocol's seed count is fixed by the boxes |

## What every plan decides

Each plan answers, per algorithm: what it is in one paragraph and which paper; the modules and what is
reused from `algos/dqn/` or `algos/ppo/`; the seam answers that differ from the base; **the paper's
settings and how each lands here (the paper cell), and where the local cell departs**; the knobs and their
defaults; the sidecar fields; the tests and the mutants; the batch and what it is read against; the
tuning budget before the row is closed; and the one result that would change the next row's plan.

Implementation detail beyond that -- the code -- is the implementer's, and lands with its tests under the
code rule in the root `CLAUDE.md`.
