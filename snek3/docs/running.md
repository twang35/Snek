# Running things

**`train.py` does not exist yet** — it arrives with `dqn/` in phase 3, so the training knobs below
are the decided design rather than a description. Everything else here runs today. Once `train.py`
exists, `tuned()` in it is the authoritative knob list and this file is a summary.

```
conda activate snek3       # or /opt/miniconda3/envs/snek3/bin/python directly
cd snek3

PYTHONPATH=. python -u train.py <policy>                # train
PYTHONPATH=. python -u evaluate.py <policy> [selector]  # a stage-B wave
PYTHONPATH=. python -u evaluate.py <policy> one         # one checkpoint, in this process
PYTHONPATH=. python -u watch.py <policy> [step]         # a live window
PYTHONPATH=. python -u record_gif.py <policy|hof>       # -> gifs/, throwaway
python -m pytest -q                                     # the suite; conftest.py handles the path
```

**`conda run` buffers stdout**, even with `python -u` — a backgrounded run's log can stay empty for
90+ seconds while the process is fine, and `kill -9` then discards the buffer permanently. Call the
env's python directly for anything backgrounded.

**Always pass `smoke` for verification runs.** The policy name doubles as the checkpoint directory
under `savedPolicies/`, so `smoke` keeps throwaway output isolated and safe to delete.

## Knobs

Every hyperparameter comes from a `SNEK_*` environment variable, so variants run side by side without
editing files. Each override prints a `hyperparameter override:` line at startup — **that grep is how
a misconfigured control arm gets caught**, and it has been.

### Run control

| knob | default | notes |
|---|---|---|
| `SNEK_SEED` | 1 | seeds the network initialisation, the exploration coins, the replay sampler, the env's food and every eval. Recorded in `runs/<policy>.md`, so two arms of the same config are the same arm |
| `SNEK_MAX_STEPS` | 10,000,000 | **absolute**, not "run this many more" — `global_step` is restored on resume. An arm at its cap prints so and exits after its opening eval |
| `SNEK_MIN_CHECKPOINT_SCORE` | 40 | below this no checkpoint is written, so a short smoke run writes none and cannot resume. Set 0 to test resume |
| `SNEK_DEBUG` | 0 | verbose logging. For debugging, not status |
| `SNEK_TORCH_THREADS` | 1 | **measured 1.4x faster than one-per-core**: a 30 -> 320 -> 3 net has no op large enough to amortise a fork-join. Compounds when four arms share the laptop |

### Network and optimiser

| knob | default |
|---|---|
| `SNEK_FC_LAYERS` | `320` — one hidden layer, which is what every record-holding snek2 arm used. snek2's *code* default was `50,100,50`; no champion ran it |
| `SNEK_LEARNING_RATE` | 1e-5 |
| `SNEK_ADAM_EPSILON` | 1e-7 |
| `SNEK_BATCH_SIZE` | 128 |
| `SNEK_DISCOUNT` | 0.99 |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 |
| `SNEK_TARGET_UPDATE_TAU` | 1.0 (a hard copy) |
| `SNEK_GRADIENT_CLIPPING` | 0 (off) |
| `SNEK_N_STEP_UPDATE` | 1 |

### Exploration

| knob | default | notes |
|---|---|---|
| `SNEK_INITIAL_EPSILON` | 0.4 | |
| `SNEK_MIN_EPSILON` | 0.002 | **exactly 0 is rejected, not clamped.** A fully greedy collect policy makes the buffer a closed loop on its own behaviour |
| `SNEK_GUIDED_FRACTION` | 0.8 | share of refinement-phase episodes where the epsilon coin's random move is drawn from non-fatal actions. **Never the greedy action** |

### Replay

| knob | default |
|---|---|
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100,000 |
| `SNEK_PRIORITY_EXPONENT` | 0.6 |
| `SNEK_IS_BETA`, `SNEK_IS_BETA_FINAL`, `SNEK_BETA_ANNEAL_STEPS` | 0.4, 1.0, 300,000 |
| `SNEK_IS_WEIGHTS` | 1 |
| `SNEK_INITIAL_COLLECT_STEPS` | 2,000 — collected before the first gradient step, at the initial epsilon rather than uniformly |
| `SNEK_REPLAY_RATIO` | 1.0 — gradient steps per **transition banked**, so the ratio is exact whether forking is on or off. **The only lever past ~1,600 agent steps/s, and a learning-dynamics change** |
| `SNEK_COLLECT_ENVS` | 1 — collect lanes per iteration. **Raising it to 16 is 1.9x** (809 -> 1,512 steps/s): the gradient work scales with it but `VecSnake.step` costs the same for 1 lane as for 64. It is still a dynamics change — 16 concurrent episodes feeding one buffer — so it is not the default |

### Collection

| knob | default | notes |
|---|---|---|
| `SNEK_FORK_BRANCHES` | 4 | 1 is off. Advances one of several branches of the same game, so the buffer holds the consequence of the *untaken* endgame action too |
| `SNEK_FORK_PROB` | 0.5 | |
| `SNEK_FORK_MIN_LENGTH` | 85 | |
| `SNEK_FORK_MAX_STEPS` | 60 | |

### Rewards and shaping

`SNEK_PERFECT_GAME_REWARD` (100), `SNEK_FOOD_DISTANCE_REWARD` (0.001),
`SNEK_CHASE_SAFE_SHAPING` + `SNEK_CHASE_SAFE_GATE` (0, 85),
`SNEK_FREE_SPACE_SHAPING` + `SNEK_FREE_SPACE_GATE` (0, 85).

**Changing any of these changes the objective**, so they are the loudest thing an arm can carry.
`SNEK_PERFECT_GAME_REWARD` cannot be moved without re-deriving `SNEK_DISCOUNT` — see
[`invariants.md`](invariants.md) invariant 6.

### Stage A, inside training

| knob | default | notes |
|---|---|---|
| `SNEK_GRAPH_EVAL_EPISODES` | 100 | **pinned.** The stage-A gate is literally "95 of 100"; a different denominator is a different gate |
| `SNEK_EVAL_INTERVAL` | 1000 | **sets the checkpoint interval too, from the same value.** They must be equal — a checkpoint at a step no eval screens can never be measured — so there is one knob rather than two that can disagree. Lower it for a smoke test and nothing else |

**A run report's config rows are the knobs.** Every key in `runs/<arm>.md`'s Config table is its
`SNEK_` variable lowercased, so `| priority_exponent | 0.6 |` means `SNEK_PRIORITY_EXPONENT` with no
lookup. `tests/test_train.py` pins both the correspondence and the defaults in this file.

### Stage B — flags, not environment variables

**A deliberate split.** A hyperparameter is an env var so an arm's config travels with its launch and
is recorded; a *measurement* parameter is a flag, so a re-measure is a command you can read rather
than an environment you have to reconstruct. snek2 had `SNEK_EVAL_EPISODES`,
`SNEK_SCREEN_THRESHOLD`, `EVAL_MIN_ACHIEVABLE`, `EVAL_SCREEN_EPISODES`, `EVAL_CONFIRM_COUNT` and
`VEC_WAVE_PROCS`, and a result file recorded none of them.

| flag | default | notes |
|---|---|---|
| `<selector>` | `screen` = `screen:95` | which checkpoints. See [`../tools/step_selectors.py`](../tools/step_selectors.py) |
| `--episodes` | 500 | stage B's depth. 100 when reproducing a snek2 close-out |
| `--shards` | 4 | how parallel, and nothing else. Measured **96 episodes/s per shard** on the laptop with 4 — against 16.9 for a single checkpoint measured on its own, because a shard's lanes are refilled from the next checkpoint and never drain |
| `--label` | none | names the pass, so an A/B does not overwrite the file it is compared with |
| `--width` | derived | games in lockstep per shard. The engine picks; there has been no reason to override it |
| `--seed` | 0 | the food stream. Two seeds are two independent samples of the same policy |
| `--no-resume` | off | re-measure rows already on disk instead of skipping them |

### Diagnostics

| knob | notes |
|---|---|
| `SNEK_ZERO_OBS` | comma-separated observation indices to zero, for ablations. Zeroes rather than deletes, so the checkpoint still loads |
| `SNEK_CHART_SCALE` | 2.0 — the chart PNG's dpi is `100 x` this. A viewer magnifies the PNG ~1.5-2x, so a lower value looks blurry blown up |
| `SNEK_TILE_PIXELS` | window size, cosmetic only — every pixel constant derives from it. **Must be set before `env.render` is imported** |
| `WATCH_FPS` | default 60; drop to 20-30 to follow the moves. 0 is uncapped, which tops out near 180 because of the display flip |
| `SNEK2_PYTHON` | `/opt/miniconda3/envs/snek/bin/python` — the interpreter `tools/import_tf_checkpoint.py` runs its TensorFlow half under |
