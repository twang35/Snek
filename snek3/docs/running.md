# Running things

`tuned()` is the authoritative knob list and it is read in **two** files — `train.py` for what is not
algorithm-specific, and the algorithm's own `algo.py` (today only [`algos/dqn/algo.py`](../dqn/algo.py)) for
the rest. This file is a summary of both.

```
conda activate snek3       # or /opt/miniconda3/envs/snek3/bin/python directly
cd snek3

PYTHONPATH=. python -u train.py <policy>                # train
PYTHONPATH=. python -u evaluate.py <policy> [selector]  # a stage-B wave
PYTHONPATH=. python -u evaluate.py <policy> one         # one checkpoint, in this process
PYTHONPATH=. python -u watch.py <policy> [step]         # a live window
PYTHONPATH=. python -u record_gif.py <policy|hof>       # -> gifs/, throwaway
PYTHONPATH=. python -m tools.scheduler --shared --queue logs/laptop-queue/   # the shared queue: claim waves, run them, passes, the window
PYTHONPATH=. python -m tools.scheduler --reopen-window   # a fresh chart window, from the running scheduler
python -m pytest -q                                     # the suite; conftest.py handles the path
```

**The scheduler opens the chart window** — one per box, every arm of the wave in it while training and
every arm's stage-B chart during a pass, nothing to launch (`../CLAUDE.md`). A training opens none.
Killing or closing the window is free, because no training reads it, waits on it or reopens it; the
scheduler brings it back at its next launch, or now with `--reopen-window`.

**`conda run` buffers stdout**, even with `python -u` — a backgrounded run's log can stay empty for
90+ seconds while the process is fine, and `kill -9` then discards the buffer permanently. Call the
env's python directly for anything backgrounded.

**Always pass `smoke` for verification runs.** The policy name doubles as the checkpoint directory
under `savedPolicies/`, so `smoke` keeps throwaway output isolated and safe to delete.

## Knobs

Every hyperparameter comes from a `SNEK_*` environment variable, so variants run side by side without
editing files. Each override prints a `hyperparameter override:` line at startup — **that grep is how
a misconfigured control arm gets caught**, and it has been.

**It does not cover every knob, and the gap is the shaping set.** `hyperparameter override:` covers
everything read through `tuned()`, in `train.py` and in the algorithm's module alike — they share the
one function, which is why the split cost the grep nothing. The reward and shaping knobs are read by
`env/constants.py` **at import**, before the trainer's config exists, so they print no override line
— `SNEK_CHASE_SAFE_SHAPING`, `SNEK_CHASE_SAFE_GATE`, `SNEK_FREE_SPACE_*`,
`SNEK_FOOD_DISTANCE_REWARD`, `SNEK_STEP_PENALTY`, `SNEK_PERFECT_GAME_REWARD`, `SNEK_ZERO_OBS`,
`SNEK_OBS_HISTORY`, `SNEK_ZIGZAG_*`, `SNEK_REVERSAL_PENALTY`. For those, grep
**`reward config:`**, one line printed at startup from `vectorized/config.describe()`. Before it
existed, b2's shaping dose had to be confirmed by reading `/proc/<pid>/environ` on the desktop.

### Run control

| knob | default | notes |
|---|---|---|
| `SNEK_RUNS_DIR` | `runs/` | where every run artifact goes -- graph, report, `_evals.json`, stage-B files, `.live/`, `.evalq/`. **The desktop daemon sets it to the gitignored `desktop/runs/` for every job**, so the box's checkout never holds an untracked file under a path master tracks and its deploy cannot collide with a committed chart; the laptop leaves it unset, and its `runs/` is the archive master tracks. A tool spawned by a job inherits it; a tool run by hand on the box needs it exported |
| `SNEK_ALGO` | `dqn` | which algorithm to train. An unknown value is **refused by name** rather than defaulting, so an arm launched as something this build has no code for cannot quietly train DQN and be reported as the other thing |
| `SNEK_SEED` | 1 | seeds the network initialisation, the exploration coins, the replay sampler, the env's food and every eval. Recorded in `runs/<policy>.md`, so two arms of the same config are the same arm |
| `SNEK_MAX_STEPS` | 10,000,000 | **absolute**, not "run this many more" — `global_step` is restored on resume. An arm at its cap prints so and exits after its opening eval. **Counted steps, not game moves**: at the default `fork_branches=4` a DQN step is four moves, and every eval row carries `transitions` for that reason |
| `SNEK_INIT_FROM` | unset | `<policy>@<step>`: a fresh arm's first weights are that arm's `ckpt-<step>.pt` instead of the initialiser's, **at step 0**, so its ramps span its own `SNEK_MAX_STEPS` from their initial values -- "anneal on from the converged values over the next 50M" is then `SNEK_MAX_STEPS` 100M and `SNEK_PPO_ANNEAL_FRACTION` 0.5, not arithmetic on the source's step. `<policy>` is a name under `savedPolicies/` or a path (a `hallOfFame/` entry); the source's `arch.json` must match this arm's, and its checkpoints must be on the box the arm runs on (pin the spec). Ignored once the arm has a `resume.pt` of its own, so a relaunch continues rather than restarting from the source. A PPO arm also takes the critic and the optimiser from the source's `resume.pt` when one is beside the checkpoint (the checkpoint holds the actor alone, and a fresh critic would push a converged actor around with noise advantages); a DQN arm takes the net and copies it to the target. Recorded in `runs/<policy>.md` as `init_from`. Batch b32 |
| `SNEK_MIN_CHECKPOINT_SCORE` | 40 | below this no checkpoint is written, so a short smoke run writes none and cannot resume. Set 0 to test resume |
| `SNEK_DEBUG` | 0 | verbose logging. For debugging, not status |
| `SNEK_TORCH_THREADS` | 1 | **measured 1.4x faster than one-per-core**: a 26 -> 320 -> 3 net has no op large enough to amortise a fork-join. Compounds when four arms share the laptop |
| `SNEK_CHART_WINDOW` | 1 | 0 opens no window. Read by the **scheduler** (`tools/window.py`), which is the only thing that opens one; the test suite sets it, and `runtime.json`'s `viewer: false` sets it for the desktop's scheduler |
| `SNEK_CHART_WINDOW_SCALE` | 1.0 | fraction of the screen the window fills. There is one window per box, so the default is the whole screen, subject to the caps in the next row |
| `SNEK_CHART_WINDOW_MAX_PX` | unset | hard ceiling on window width in logical pixels, on either box. Unset leaves the window bounded by the screen and by the charts — **a panel is never drawn wider than its source PNG** (730 px training, 1000 px eval), so a window with little in it opens small rather than upscaling to fill the display |

### Network and optimiser

| knob | default |
|---|---|
| `SNEK_FC_LAYERS` | `320` — one hidden layer, which is what every record-holding snek2 arm used. snek2's *code* default was `50,100,50`; no champion ran it |
| `SNEK_LEARNING_RATE` | 1e-5 |
| `SNEK_ADAM_EPSILON` | 1e-7 |
| `SNEK_BATCH_SIZE` | 128 |
| `SNEK_DISCOUNT` | 0.99 |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 |
| `SNEK_TARGET_UPDATE_TAU` | 1.0 (a hard copy). `agent.update()` applies it; nothing else may call `maybe_update_target()` or a soft tau applies twice per step |
| `SNEK_GRADIENT_CLIPPING` | 0 (off) |
| `SNEK_N_STEP_UPDATE` | 1 |

### Exploration

| knob | default | notes |
|---|---|---|
| `SNEK_INITIAL_EPSILON` | 0.4 | |
| `SNEK_MIN_EPSILON` | 0.002 | **exactly 0 is rejected, not clamped.** A fully greedy collect policy makes the buffer a closed loop on its own behaviour |
| `SNEK_GUIDED_FRACTION` | 0.8 | share of refinement-phase episodes where the epsilon coin's random move is drawn from non-fatal actions. **Never the greedy action** |
| `SNEK_EPSILON_SCHEDULE` | `eval` | `eval` is the two-phase schedule off the eval history (`algos/dqn/schedules.py`, every batch to date). **`linear` is the papers'**: a straight line from `SNEK_INITIAL_EPSILON` to `SNEK_MIN_EPSILON` over `SNEK_EPSILON_ANNEAL_STEPS` **moves**, then held, with the shield at its configured fraction from move 0. The algorithm series' paper cells (`plans/algoExploration/`) run it at 1.0 -> 0.01 over 250,000 moves; the local cells keep `eval`. 2026-09-17 |
| `SNEK_EPSILON_ANNEAL_STEPS` | 250000 | moves, for `linear` only. 1M Atari frames = 250k agent steps |
| `SNEK_MUNCHAUSEN_ALPHA` | 0 (off) | Munchausen RL (Vieillard, Pietquin & Geist 2020) on any DQN-family algorithm: the clipped, `tau`-scaled log-policy of the taken action is added to the reward and the bootstrap becomes the soft value under the target net's policy at temperature `tau`. Paper 0.9. **0 is plain double DQN to the bit**; the fixture says so. Group A, row A6 |
| `SNEK_MUNCHAUSEN_TAU` | 0.03 | the temperature, in reward units (the paper's value against a unit reward; food here is 1) |
| `SNEK_MUNCHAUSEN_L0` | -1 | the log-policy clip's floor |
| `SNEK_RESET_INTERVAL` | 0 (off) | BBF's shrink-and-perturb resets (Schwarzer et al. 2023) on any DQN-family algorithm, every this many **gradient steps**: the trunk (every hidden linear) is pulled toward a fresh initialisation, the head (and IQN's embedding, FQF's fraction net) is re-initialised, the target becomes a copy of the reset net and every optimiser's state is cleared. `algos/dqn/resets.py`; `plans/algoExploration/d-data-efficiency.md` |
| `SNEK_RESET_ALPHA` | 0.5 | how much of the old trunk survives a reset: θ ← α θ + (1 − α) θ_fresh. 1 keeps the trunk and still re-initialises the head; 0 is a full re-initialisation |
| `SNEK_RESET_STOP_AFTER` | 0 (never) | the gradient step after which no reset fires, so the final stretch of training settles |
| `SNEK_RESET_ANNEAL_N_STEP`, `SNEK_RESET_ANNEAL_GAMMA` | off | BBF's within-cycle anneal, both or neither: `10,3` and `0.97,0.997` restart n-step and γ at the first value after every reset and take them to the second over `SNEK_RESET_ANNEAL_STEPS` gradient steps (exponential in n and in log(1 − γ), Dopamine's schedule; the first cycle starts at gradient step 0). On the DQN-family path the collector's window, its discount and the env's `shaping_discount` move **together** (`docs/invariants.md` §6), so a banked transition keeps the `γ^n` it was collected under and new ones carry the new; each eval row carries a `cycle` block. Inert without resets beyond the first cycle. 2026-09-20; `algos/dqn/resets.py` `CycleSchedule` |
| `SNEK_RESET_ANNEAL_STEPS` | 10,000 | the anneal's length in gradient steps |

### Replay

| knob | default |
|---|---|
| `SNEK_REPLAY_BUFFER_MAX_LENGTH` | 100,000 |
| `SNEK_PRIORITY_EXPONENT` | 0.6 |
| `SNEK_IS_BETA`, `SNEK_IS_BETA_FINAL`, `SNEK_BETA_ANNEAL_STEPS` | 0.4, 1.0, 300,000 |
| `SNEK_IS_WEIGHTS` | 1 |
| `SNEK_INITIAL_COLLECT_STEPS` | 2,000 — collected before the first gradient step, at the initial epsilon rather than uniformly |
| `SNEK_REPLAY_RATIO` | 1.0 — gradient steps per **transition banked**, so the ratio is exact whether forking is on or off, and it already matches snek2's 1 gradient step per transition. **Do not lower it to reproduce a snek2 arm** — that makes snek3 *less* data-efficient than snek2 ever was; `SNEK_MAX_STEPS` is the comparability knob. As a dynamics change it is worth ~2x at batch 512 |
| `SNEK_COLLECT_ENVS` | 1 — collect lanes per iteration. **It is a width knob, not a speed knob, and it makes an arm slower.** Transitions/s rises ~1.46x at 8 lanes, but `SNEK_MAX_STEPS` counts *counted steps* and each one now banks 8x the transitions — so time to the same cap rises ~5.5x. Spend it only alongside a cap lowered by the same factor. It is also a dynamics change: 8 concurrent episodes feeding one buffer |

### Collection

| knob | default | notes |
|---|---|---|
| `SNEK_FORK_BRANCHES` | 4 | 1 is off. Advances one of several branches of the same game, so the buffer holds the consequence of the *untaken* endgame action too |
| `SNEK_FORK_PROB` | 0.5 | |
| `SNEK_FORK_MIN_LENGTH` | 85 | |
| `SNEK_FORK_MAX_STEPS` | 60 | |

### Distributional heads — only under `SNEK_ALGO=c51`, `qrdqn`, `iqn` or `fqf`

Group A of the algorithm series (`plans/algoExploration/a-return-tail.md`; `algos/dist/`). Each is DQN
with a distributional head and the loss that fits it, on the same replay, collector and schedules, so
**every DQN knob above applies unchanged**, and the sidecar gains a `head` field that the restore path
reads (`tools/arch.py`, `OPTIONAL_FIELDS`). The greedy policy every eval measures is the argmax of the
distribution's **mean**; `--policy-variant cvar:<alpha>` on `evaluate.py`, `tools.closeout` and
`tools.shard` measures the argmax of its lower tail instead, under an explicit `--label` so the rows sit
beside the mean's file (`a-return-tail.md` §5).

| knob | default | rungs | notes |
|---|---|---|---|
| `SNEK_DIST_ATOMS` | 51 | c51 | atoms of the fixed support |
| `SNEK_DIST_V_MIN`, `SNEK_DIST_V_MAX` | -10, 110 | c51 | the support. Set from **this game's discounted return range** (win 100, food 1, death -5), not the papers' clipped [-10, 10]; the smoke asserts a prefill batch's targets do not pile on the end atoms |
| `SNEK_DIST_QUANTILES` | 200 (qrdqn), 32 (fqf) | qrdqn, fqf | N |
| `SNEK_DIST_TAU_SAMPLES`, `SNEK_DIST_TAU_PRIME_SAMPLES`, `SNEK_DIST_POLICY_SAMPLES` | 64, 64, 32 | iqn | N, N', K: online and target loss samples and the acting read's draws |
| `SNEK_DIST_EMBEDDING` | 64 | iqn, fqf | the cosine embedding's width |
| `SNEK_DIST_KAPPA` | 1.0 | qrdqn, iqn, fqf | the quantile Huber's threshold, in reward units |
| `SNEK_DIST_FRACTION_LR`, `SNEK_DIST_FRACTION_ENTROPY` | 2.5e-9, 0.001 | fqf | the fraction proposal net's RMSProp rate (the paper's) and entropy bonus (the released code's) |
| `SNEK_DIST_RISK_ALPHA` | 1.0 (neutral) | all | the CVaR level of the risk-sensitive **training** policy, used only with the next knob |
| `SNEK_DIST_RISK_TRAIN` | 0 | all | 1 trains the paper's risk-sensitive agent (Dabney et al. 2018 §4): acting and the target's argmax under the CVaR read at `SNEK_DIST_RISK_ALPHA`, the target's values undistorted. Stage A still measures the mean read; the CVaR read of a checkpoint is a pass's `--policy-variant` |

Throughput on the laptop, stage A off, paper-cell plumbing (1 lane, batch 32, 0.25 gradient steps a
move), 2026-09-17: dqn ~1,870 st/s, c51 ~1,700, fqf ~760, iqn ~640, qrdqn ~520. The local cell's four
gradient steps a counted step at batch 128 cost IQN 15 st/s, so the heavy heads have no local cell.

### Discrete SAC — only under `SNEK_ALGO=sac` or `sac2`

Group B of the algorithm series (`plans/algoExploration/b-entropy.md`; `algos/sac/`). A categorical actor
(DQN's `QNet` read as logits, exactly PPO's, so the checkpoint is the actor alone and `arch.json` gains no
field), two Q critics with target copies, and an automatically tuned or fixed temperature; off-policy from
DQN's replay and collector with the fork and the shield **off** -- the agent samples from its own policy.
`sac` is Christodoulou 2019 at its paper values; `sac2` is the same class with Zhou et al. 2022's fixes as
its defaults (fixed α 0.05, entropy-penalty, double average Q with a Q-clip, Polyak 0.005). Every DQN, PPO
and distributional knob is **refused by name**, as under PPO; `SNEK_COLLECT_ENVS` (default **16** here) and
`SNEK_DISCOUNT` are shared.

| knob | `sac` | `sac2` | notes |
|---|---|---|---|
| `SNEK_SAC_LEARNING_RATE`, `SNEK_SAC_CRITIC_LEARNING_RATE` | 3e-4 | 1e-5 | Adam, actor and critics |
| `SNEK_SAC_ADAM_EPSILON` | 1e-8 | 1e-8 | torch's default, the papers' |
| `SNEK_SAC_BATCH_SIZE` | 64 | 64 | 128 in the local cell |
| `SNEK_SAC_TARGET_UPDATE_PERIOD`, `SNEK_SAC_TAU` | 2000, 1.0 | 1, 0.005 | gradient updates between target moves; τ 1.0 is a hard copy, below it Polyak |
| `SNEK_SAC_ALPHA` | `auto` | 0.05 | the temperature: `auto` tunes log α by gradient, a number fixes it and builds no α optimiser |
| `SNEK_SAC_TARGET_ENTROPY_RATIO`, `SNEK_SAC_INIT_ALPHA`, `SNEK_SAC_ALPHA_LEARNING_RATE` | 0.98, 1.0, 3e-4 | same | `auto` only: the target is `ratio · ln 3 = 1.0766` |
| `SNEK_SAC_REPLAY_RATIO` | 0.25 | 0.1 | gradient updates per game move, carried as a fraction |
| `SNEK_SAC_N_STEP` | 1 | 3 | |
| `SNEK_SAC_ENTROPY_PENALTY` | 0 (off) | 0.5 | β · ½ E_s[(H_old(s) − H(s))²], H_old(s) the collecting policy's entropy at that state, stored with the transition (the replay's `aux` column) and read back per row. Before 2026-09-20 it compared consecutive minibatches' *mean* entropies, which is a different term; no arm ran it |
| `SNEK_SAC_CRITIC_COMBINE` | `min` | `avg` | how the two critics' Q(s, ·) are read, in the target and the actor loss |
| `SNEK_SAC_Q_CLIP` | 0 (off) | 0.5 | the critic loss becomes max((Q − y)², (Q′ + clip(Q − Q′, ±c) − y)²) on the squared error; off, it is `SNEK_SAC_CRITIC_LOSS` |
| `SNEK_SAC_CRITIC_LOSS` | `mse` | `mse` | both papers' critic loss. `huber` (δ 1) is the local departure for the unclipped +100 terminal; b41's paper cell ran Huber before this knob existed (2026-09-20), the local cell keeps it |
| `SNEK_SAC_REPLAY_BUFFER_MAX_LENGTH` | 1,000,000 | 100,000 | 100,000 in the local cell |
| `SNEK_SAC_PRIORITY_EXPONENT` | 0 (uniform) | 0 | 0.6 is the local plumbing's PER, with DQN's β anneal 0.4 → 1 over 300k |
| `SNEK_SAC_PREFILL` | 20,000 | 20,000 | transitions from the untrained actor before the first update |

Each eval row carries `alpha` and a `sac` block -- `entropy`, `target_entropy`, `critic_loss`, `actor_loss`,
`alpha_loss`, `entropy_penalty`, `clip_fraction` (the share of critic samples the clipped branch won) and
`episodes` -- so the gates in `b-entropy.md` §4 read off the history file.

### BBF — only under `SNEK_ALGO=bbf`

Bigger, Better, Faster (Schwarzer et al. 2023), the paper's recipe whole (`algos/bbf/`; `plans/algoExploration/d-data-efficiency.md`
§7), built 2026-09-20 to read how it does on Snake and expected to rest after a few batches. Its own package because two pieces do not
fit the shared path: the n-step return is summed **when a batch is drawn**, from a replay that keeps each lane's rows in order, so the
anneal of n and γ reaches every drawn transition; and SPR needs the K observations after each state. It shares the collector (at n-step
1, fork and shield off), the Rainbow network (dueling C51, noisy off; **a `bbf` checkpoint restores as a Rainbow one**, the SPR heads
live in `resume.pt`), Group A's C51 losses, the reset and cycle schedules, and every eval and pass. Knobs DQN owns keep DQN's names at
the paper's defaults; BBF's own carry `BBF_`; every other algorithm's knob is refused by name.

| knob | paper default | notes |
|---|---|---|
| `SNEK_LEARNING_RATE`, `SNEK_ADAM_EPSILON`, `SNEK_BBF_WEIGHT_DECAY` | 1e-4, 1.5e-4, 0.1 | one AdamW over the Q net and the SPR heads |
| `SNEK_BATCH_SIZE`, `SNEK_REPLAY_RATIO` | 32, 8 | gradient steps per game move |
| `SNEK_DISCOUNT`, `SNEK_N_STEP_UPDATE` | 0.997, 3 | the **end** of each cycle; with the anneal off, the constants throughout |
| `SNEK_RESET_INTERVAL`, `_ALPHA`, `_STOP_AFTER` | 40,000, 0.5, 0 | the encoder (`trunk.hidden.*`) and the transition model shrunk, the streams, projection and predictor replaced, both targets copied, AdamW cleared |
| `SNEK_RESET_ANNEAL_N_STEP`, `_GAMMA`, `_STEPS` | `10,3`, `0.97,0.997`, 10,000 | as above, applied at sample time here |
| `SNEK_TARGET_UPDATE_TAU` | 0.005 | EMA every gradient step, Q net and SPR heads |
| `SNEK_GRADIENT_CLIPPING` | 10 | |
| `SNEK_INITIAL_EPSILON`, `SNEK_MIN_EPSILON`, `SNEK_EPSILON_ANNEAL_STEPS` | 1, 0, 2001 | linear in moves after the prefill; **no hard floor**, the paper trains at ε 0 |
| `SNEK_INITIAL_COLLECT_STEPS`, `SNEK_REPLAY_BUFFER_MAX_LENGTH`, `SNEK_PRIORITY_EXPONENT` | 2,000, 1,000,000, 0.5 | prefill at ε 1; PER with Dopamine's weights (`p^-0.5` over the batch max) |
| `SNEK_COLLECT_ENVS` | 1 | lanes; the replay keeps each lane's sequence and refuses a row that does not continue its lane |
| `SNEK_DIST_ATOMS`, `SNEK_DIST_V_MIN`, `SNEK_DIST_V_MAX` | 51, -10, 110 | this game's return range |
| `SNEK_BBF_SPR_WEIGHT`, `SNEK_BBF_SPR_STEPS`, `SNEK_BBF_PROJECTION` | 5, 5, 512 | the SPR weight, K, the projection width; 0 weight or 0 steps is no SPR |
| `SNEK_BBF_TRANSITION_WIDTH` | 256 | the transition model's hidden width (0: the latent's). BBF's is two 64-channel convolutions beside a far larger encoder; at the latent's 2048 it was 8x the whole step's cost |
| `SNEK_BBF_DUELING`, `SNEK_BBF_DOUBLE` | 1, 1 | |
| `SNEK_FC_LAYERS` | `1280,2048` in the paper cell | the plan's analogue of the ×4 IMPALA encoder plus the 2048 dense layer |

Each eval row carries `cycle` (`n_step`, `gamma`, `since_reset`) and a `bbf` block (`resets`, `train_step`, `td_loss`, `spr_loss`,
`grad_norm`). **The reward/discount coupling** holds exactly only with the potential shaping off (`SNEK_CHASE_SAFE_SHAPING=0`, the paper
cell): under a sample-time anneal a transition collected at γ 0.97 can be drawn into a 0.997 target.

Throughput on the laptop **under a live 4-arm wave**, 1 thread, replay ratio 8 (2026-09-20, lower bounds): `fc 1280,2048` transition
256: **25 gradient steps/s = 3 moves/s**; transition 2048: 8/s; transition 64: 31/s; `fc 1280`: 61/s; `fc 320`: 210/s; 4 threads was
slower than 1 under that load. A 100k-move paper arm is ~800k gradient steps, about nine hours of learning before stage A.

### PPO — only under `SNEK_ALGO=ppo`

Every knob above whose meaning is DQN-specific — `SNEK_FORK_*`, `SNEK_INITIAL_EPSILON`,
`SNEK_MIN_EPSILON`, `SNEK_GUIDED_FRACTION`, `SNEK_REPLAY_*`, `SNEK_PRIORITY_EXPONENT`, `SNEK_IS_*`,
`SNEK_TARGET_UPDATE_PERIOD`, `SNEK_N_STEP_UPDATE`, `SNEK_INITIAL_COLLECT_STEPS`, and the four
optimiser knobs — is **refused by name** under `SNEK_ALGO=ppo`, with every offender listed in one
error. The optimiser four are the ones that matter: `SNEK_LEARNING_RATE=1e-5` is DQN's tuned value,
PPO takes ~64x fewer gradient steps per transition, and an arm that silently took it would report a
flat curve and a conclusion about PPO. Hence `SNEK_PPO_LEARNING_RATE`, a separate name.

`SNEK_COLLECT_ENVS`, `SNEK_DISCOUNT`, `SNEK_SEED`, `SNEK_MAX_STEPS`, `SNEK_FC_LAYERS` and every
reward knob mean the same thing in both and are shared — which is what makes a seed-matched A/B
launchable at all.

| knob | default | notes |
|---|---|---|
| `SNEK_COLLECT_ENVS` | **128** for PPO | lanes. `envs × rollout` is the transitions per iteration, and the arm's step increment |
| `SNEK_PPO_ROLLOUT` | 128 | T, the steps each lane takes before an update. 128 x 128 = **16,384 transitions** |
| `SNEK_PPO_EPOCHS` | 4 | passes over each rollout. Every sample is seen exactly once per pass |
| `SNEK_PPO_MINIBATCH` | 256 | **refused if larger than a whole rollout**, or "4 epochs" would silently mean 4 gradient steps |
| `SNEK_PPO_CLIP` | 0.2 | must be in (0, 1) |
| `SNEK_PPO_CLIP_FINAL` | unset | set it and the clip ramps linearly to it over `SNEK_MAX_STEPS`, clamped at both ends, like the entropy coefficient. **Must stay in (0, 1)** — a clip of 0 admits no update, so anneal to a floor such as 0.02. Batch b17 |
| `SNEK_PPO_GAE_LAMBDA` | 0.98 | higher than the conventional 0.95. The advantage horizon is `1/(1 − γλ)` — **33.6** steps at γ=0.99, 44.5 at γ=0.9975 — and a perfect game is ~950 moves from the opening, so the +100 reaches the policy through the critic, not through GAE |
| `SNEK_PPO_GAE_LAMBDA_FINAL` | unset | set it and lambda ramps linearly to it over the anneal fraction of `SNEK_MAX_STEPS`, like the three ramps below. Read at the step a rollout *begins*, so one rollout's advantages use one value. Must be in [0, 1]. Batch b24 |
| `SNEK_PPO_DISCOUNT_FINAL` | unset | the same ramp for gamma, and the env's shaping discount follows it (it is the agent's gamma). `PPO_` in the name although `SNEK_DISCOUNT` is shared: DQN has no ramp, and a knob DQN would silently ignore must not share a name DQN reads. Must be in (0, 1]. `runs/<policy>.md` adds `ppo_horizon_final` beside `ppo_horizon` when either anneal is set. Batch b24 |
| `SNEK_PPO_ENTROPY_COEF` | 0.01 | max entropy on 3 actions is ln 3 = 1.0986 |
| `SNEK_PPO_ENTROPY_COEF_FINAL` | unset | set it and the coefficient ramps linearly to it over `SNEK_MAX_STEPS`, clamped at both ends. Unset is a constant |
| `SNEK_PPO_VF_COEF` | 0.5 | near-inert: the towers are separate, so it only rescales the critic's own learning rate |
| `SNEK_PPO_LEARNING_RATE` | 3e-4 | **not** DQN's 1e-5, and that is the point of the separate name |
| `SNEK_PPO_LEARNING_RATE_FINAL` | unset | the same ramp for Adam's step size; 0 is allowed and means the tail of the run takes no gradient steps. Both ramps re-stretch if an arm is resumed to a higher cap, so an annealed arm is never resumed for comparison. Batch b17 |
| `SNEK_PPO_ANNEAL_FRACTION` | 1.0 | the share of `SNEK_MAX_STEPS` every ramp (clip, entropy, learning rate, lambda, gamma) takes to reach its `_FINAL` value; 0.8 lands at 80% of the cap and holds the floor for the last 20%. Must be in (0, 1]. Batch b17 |
| `SNEK_PPO_ADAM_EPSILON` | 1e-7 | |
| `SNEK_PPO_GRADIENT_CLIPPING` | 0.5 | global norm over both towers. 0 disables |
| `SNEK_PPO_TARGET_KL` | 0 (off) | stops the epoch loop early when `approx_kl` exceeds it — **between epochs, never mid-epoch**, or some samples are used more often than others. `approx_kl` is reported either way |
| `SNEK_PPO_NORMALIZE_ADV` | 1 | zero-mean, unit-sd per minibatch. What makes the update invariant to the reward scale |
| `SNEK_PPO_VALUE_LOSS` | `huber` | or `mse`. Huber for the reason `algos/dqn/agent.py` gives: one +100 terminal would dominate a squared error over 256 samples |

**A PPO step is one transition is one game move**, so `SNEK_MAX_STEPS` means game moves for a PPO arm
and four-moves-per-step for a DQN arm at `fork_branches=4`. Read `transitions`, which both write.

**`SNEK_EVAL_INTERVAL` is rounded up to a whole rollout** — 1,000 becomes 16,384 at the defaults — and
the value used is what `runs/<policy>.md` reports. There is no PPO-specific interval knob: an eval at a
step no checkpoint exists at is the one thing the protocol cannot tolerate.

### Rewards and shaping

`SNEK_PERFECT_GAME_REWARD` (100), `SNEK_FOOD_DISTANCE_REWARD` (0.001), `SNEK_STEP_PENALTY` (0, subtracted on every step -- b26's knob), `SNEK_OBS_HISTORY` (0; N appends 2N values of move history to the observation and `-histN` to the era -- b27's knob; **one depth per wave**, since a shared eval worker serves one width),
`SNEK_CHASE_SAFE_SHAPING` + `SNEK_CHASE_SAFE_GATE` (0, 85),
`SNEK_FREE_SPACE_SHAPING` + `SNEK_FREE_SPACE_GATE` (0, 85),
`SNEK_ZIGZAG_SHAPING` + `SNEK_ZIGZAG_WINDOW` (0; the window defaults to `SNEK_OBS_HISTORY`, or 8 when history is off -- a potential-based term, Φ = −(reversal pairs among the last `window` moves), a reversal being a left straight after a right or the converse; b34's knob, `plans/zigzag-shaping.md`),
`SNEK_REVERSAL_PENALTY` (0; subtracted on every step whose move reverses the one before, terminal steps included, as the step penalty is -- b34's other knob).

**Changing any of these changes the objective**, so they are the loudest thing an arm can carry.
`SNEK_PERFECT_GAME_REWARD` cannot be moved without re-deriving `SNEK_DISCOUNT` — see
[`invariants.md`](invariants.md) invariant 6.

### Stage A, inside training

| knob | default | notes |
|---|---|---|
| `SNEK_GRAPH_EVAL_EPISODES` | 100 | **pinned.** The stage-A gate is literally "95 of 100"; a different denominator is a different gate |
| `SNEK_EVAL_INTERVAL` | 1000 | **sets the checkpoint interval too, from the same value.** They must be equal — a checkpoint at a step no eval screens can never be measured — so there is one knob rather than two that can disagree. Lower it for a smoke test and nothing else. **Rounded up to a whole algorithm step**, which changes nothing for DQN (granularity 1) and is what lets an algorithm whose step is a whole rollout keep the equality; `runs/<policy>.md` records the interval the arm actually ran |
| `SNEK_EVAL_QUEUE` | **1** | hands stage A to shared worker processes instead of measuring it in the training loop. **Measured 3.83x** end to end at the tuned defaults (9.12 h -> 2.38 h per arm to 3M, four arms, laptop). It **changes the training** — bounded schedule lag, and stage-A rows are no longer bit-reproducible from the arm's seed — so `0` is what an arm being diffed against b1 or b2 must use. See below |
| `SNEK_EVAL_QUEUE_DEPTH` | **16** | how many checkpoints may be **unmeasured**. **This is both the schedule's blind spot and the throughput lever**: at 16 the epsilon and shield schedules read a measurement up to 16,000 counted steps old. 16 is where an arm stops being eval-bound — all 19 swept configurations at depth 8 sat pinned against their cap with the trainer waiting, while at 16 the queue drains and the arm reaches 94% of its unblocked rate. 24 regresses, 12 gets 63% of the gain. **0 is the verification mode** — the trainer measures each checkpoint before resuming, which reproduces an unqueued arm bit for bit and recovers nothing |
| `SNEK_EVAL_WORKERS` | **6** | worker processes **per box**, shared by every arm on it. **‡ 6 was tuned at four DQN trainers and is too low for eight PPO ones** — measured with workers pinned at 100% with the trainers at half and a third of the box idle. For PPO the bound is also free to raise, because its only schedule is a function of the step rather than of the eval history; see [`findings.md`](findings.md). Unlike depth this turns over: at four arms the sweep measured 4.84 h (2w), 3.94 h (4w), **3.21 h (6w)**, 3.27 h (8w), 3.55 h (10w) — past six, workers starve the trainers, whose unblocked rate falls 380 -> 306 st/s, and the box reaches 4% idle to run *slower*. **Idle CPU is not the target**; the fastest configuration leaves ~20% of the box free. Starting none is safe — the arm measures its own and runs at today's speed |

**‡ Sweeping this knob only works in ascending order, and a descending step is silently wrong.**
`eval_queue.ensure_workers(target)` iterates `range(target)` and starts the slots that are not live —
it **never reaps a worker in a slot ≥ target.** A worker self-exits only after
`IDLE_EXIT_SECONDS = 300` of claiming nothing, and the desktop's gap between waves is one 30 s poll,
so a wave that follows a *higher*-worker wave inherits the extras — and they are not idle, because the
new wave's arms are producing checkpoints, so they claim its queue work and contribute. The wave then
measures more workers than its label says. A first draft of the desktop sweep ordered
8, 4, 6, 10, 12 and would have read "4 is surprisingly fine". The same leftovers during a *stage-B*
phase are harmless: there is no stage-A work then, so they poll with sleeps at ~200 MB and near-zero
CPU, and exit within five minutes.

**Where the 1.67x comes from, and why it is not more.** The saving is the *streaming efficiency* —
a checkpoint measured inside a sustained round costs ~1.1-1.9 s against ~4.2 s drained — plus whatever
of the remainder overlaps with training. It is **not** free parallelism: total CPU work is roughly
conserved, and the laptop test was core-bound, with the arms' own training rate falling from 426 to
270 counted steps/s while two workers ran. A box with spare cores overlaps more; a saturated one gets
only the efficiency.

#### What the queue changes about the training, which is two things and not one

Both are pre-registered rather than slipped in as a speed-up, because `perfect_percent` steers
exploration ([`invariants.md`](invariants.md) invariant 2) and is a feedback loop rather than a readout.

- **The schedule's feedback lags by up to `DEPTH` evals**, bounded — the trainer takes the work back
  at the bound, so the worst case is today's behaviour and never an unbounded drift.
- **Stage-A rows stop being bit-reproducible from the arm's seed.** A queued measurement shares one
  wide `VecSnake` with the other checkpoints in its round and lanes migrate between them, which is
  exactly why streaming is faster — so a job cannot own its RNG and does not see the boards
  `eval_seed(seed, step)` would draw. The two instruments agree to 0.09 standard errors over 3,222
  rows (phase 2), so nothing is lost statistically, but **a queued arm cannot be diffed byte for byte
  against an unqueued one.**

**A queued row's trailing window is "the last N rows that have landed", not "the last N steps".** A
streamed round completes in lane order, so rows arrive out of step order; holding them back to fix that
serialised the arm behind the slowest member of a round and was worth nothing (measured: stalled at
step 9,000 for 55 s with 2 of 9 rows unmerged). The file itself stays in step order, and every row's
own content is independent of when its neighbours arrive.

**One column shifts, and nothing selects on it.** A queued row's `epsilon` and `guided_fraction` are
what the arm *ran under* — governing the interval **ending** at that step — where an unqueued row's are
what its own eval just set, governing the interval **starting** there. Exactly one row apart, measured.
The measurement half of every row — score, reward, perfect rate, trailing mean — is bit-identical, and
`screen:95` reads only that half.

**Nothing can deadlock.** A trainer at its depth bound measures the oldest checkpoint itself. In the
steady state it never asks whether a worker exists or is keeping up; reaching the bound *is* the
signal. So killing every worker mid-batch slows the arms down and cannot stop them. (The tail drain is
the one place that checks, and only to decide how long to wait for a backlog that is landing.)

**The desktop needs no changes to run it** — the workers are started by the first arm that wants one,
exactly as the chart window is, so a job spec turns it on with nothing else:

```json
{"project": "snek3", "id": "...", "type": "train", "policy": "...",
 "env": {"SNEK_SEED": "1", "SNEK_EVAL_QUEUE": "1"}}
```

Wave-barrier scheduling means trainings and evals never overlap on the box, so four trainers plus two
workers is six processes on sixteen cores and the 16-shard stage-B wave still comes afterwards.

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
