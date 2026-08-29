# Running things

**Phase 0 note:** the entry points below are the decided design
([`../plans/pytorch-port.md`](../plans/pytorch-port.md)); only `env/` and `vectorized/` exist so far.
Once `train.py` exists, `tuned()` in it is the authoritative knob list and this file is a summary.

```
conda activate snek3       # or /opt/miniconda3/envs/snek3/bin/python directly
cd snek3

PYTHONPATH=. python -u train.py <policy>                # train
PYTHONPATH=. python -u evaluate.py <policy|batch> [sel]  # a stage-B wave
PYTHONPATH=. python -u watch.py <policy> [step]          # a live window
PYTHONPATH=. python -u record_gif.py <policy|hof>        # -> gifs/, throwaway
PYTHONPATH=. python -m pytest -q                         # the suite
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
| `SNEK_SEED` | unset | unset means unseeded and not reproducible. Set it and it is recorded in `runs/<policy>.md` |
| `SNEK_MAX_STEPS` | 10,000,000 | **absolute**, not "run this many more" — `global_step` is restored on resume. An arm at its cap prints so and exits after its opening eval |
| `SNEK_MIN_CHECKPOINT_SCORE` | 40 | below this no checkpoint is written, so a short smoke run writes none and cannot resume. Set 0 to test resume |
| `SNEK_DEBUG` | 0 | verbose logging. For debugging, not status |

### Network and optimiser

| knob | default |
|---|---|
| `SNEK_FC_LAYERS` | `50,100,50` |
| `SNEK_LEARNING_RATE` | 1e-5 |
| `SNEK_ADAM_EPSILON` | 1e-7 |
| `SNEK_BATCH_SIZE` | 128 |
| `SNEK_DISCOUNT` | 0.99 |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 |
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
| `SNEK_REPLAY_RATIO` | 1.0 — gradient steps per env step. **The only lever past ~4,000 steps/s, and a learning-dynamics change** |
| `SNEK_COLLECT_ENVS` | 1 — collect lanes per iteration. Raising it alone buys nothing; the gradient work scales with it |

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

### Eval

| knob | default | notes |
|---|---|---|
| `SNEK_GRAPH_EVAL_EPISODES` | 100 | **pinned.** The stage-A gate is literally "95 of 100"; a different denominator is a different gate |
| `SNEK_EVAL_INTERVAL` | 1000 | **pinned** to the checkpoint interval, or a checkpoint exists that no screen can select |
| `SNEK_EVAL_EPISODES` | 500 | stage B |
| `SNEK_SCREEN_THRESHOLD` | 95 | stage-A perfect count that admits a checkpoint to stage B. The cost knob |
| `VEC_WAVE_PROCS` | `cpu_count − 2` | **each host's optimum is its own** — 12 on the laptop, 16 on the desktop (whose 16 are SMT threads over 8 cores). Neither transfers; sweep a new host |

### Diagnostics

| knob | notes |
|---|---|
| `SNEK_ZERO_OBS` | comma-separated observation indices to zero, for ablations. Zeroes rather than deletes, so the checkpoint still loads |
| `SNEK_TILE_PIXELS` | window size, cosmetic only — every pixel constant derives from it. **Must be set before `env.constants` is imported** |
| `WATCH_FPS` | default 90; drop to 20-30 to follow the moves |
