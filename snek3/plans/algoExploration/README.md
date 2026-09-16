# The algorithm series: plans

[`algorithm-series.md`](algorithm-series.md) is the ordering and the reasoning. Each group has one
implementation plan here, and every plan leans on the conventions below rather than restating them.

| plan | group | rows |
|---|---|---|
| [`a-return-tail.md`](a-return-tail.md) | A | DQN, C51, QR-DQN, IQN, FQF, Munchausen |
| [`b-value-stack.md`](b-value-stack.md) | B | Rainbow, Beyond the Rainbow |
| [`c-entropy.md`](c-entropy.md) | C | Discrete SAC, Revisiting Discrete SAC |
| [`d-memory.md`](d-memory.md) | D | recurrent PPO, R2D2 |
| [`e-exploration.md`](e-exploration.md) | E | NGU, Agent57 |
| [`f-data-efficiency.md`](f-data-efficiency.md) | F | BBF |
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
| the policy seam | `policy_fn` is `(m, obs_len) float32 -> (m,) int64`, stateless, over the rows the engine hands it. `vectorized/` imports no torch. **Two groups need more than this** (D and E need per-row state, G needs the board) and the extension is designed once, in `d-memory.md` §1 and `g-planning.md` §1, and reused | `tests/test_module_layering.py` |
| a step | `step_granularity` says what a counted step is; `advance()` returns `(steps, transitions)` and every row carries `transitions`. A plan states what its step is, because `SNEK_MAX_STEPS`, `SNEK_EVAL_INTERVAL` and every chart's x-axis read it | `algos/ppo/algo.py` §1 |
| tests | a `tests/test_<name>_*.py` per module with the arithmetic pinned (a loss on a hand-worked batch, a target on a known transition), and a `tests/mut_<name>.json` mutation spec whose mutants all die before the row is queued | `skills/mutation-test` |
| the batch | 4 seeds a cell, seed pinned to the arm letter, the current PPO reference config for everything the algorithm does not own (reward preset, `SNEK_OBS_HISTORY`, `SNEK_FC_LAYERS`), stage A at 100 episodes on every checkpoint, stage B, `hof5000`, `hof30k`. Judged on stage-B density, the depth passes and the drawdown count, never on a single eval | `docs/protocol.md` |
| the gate | a row starts when the row it is read against has a closed stage-B number; a row that does not stabilise within the tuning budget its plan names is closed as a finding | `algorithm-series.md` §3 |

## What every plan decides

Each plan answers, per algorithm: what it is in one paragraph and which paper; the modules and what is
reused from `algos/dqn/` or `algos/ppo/`; the seam answers that differ from the base; the knobs and their
defaults; the sidecar fields; the tests and the mutants; the batch and what it is read against; the
tuning budget before the row is closed; and the one result that would change the next row's plan.

Implementation detail beyond that -- the code -- is the implementer's, and lands with its tests under the
code rule in the root `CLAUDE.md`.
