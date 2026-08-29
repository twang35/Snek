# snek3 — the PyTorch port

**Status: approved 2026-08-28. Phase 0 closed 2026-08-28; phase 1 next.** This is the plan for
standing up `snek3/` as a clean-slate PyTorch project that keeps everything snek2 can do — train, evaluate,
chart, watch, record GIFs, queue batches to the desktop — with the learning framework replaced and
the accumulated shape of the thing thrown away.

`snek2/` is frozen from here. Nothing moves out of it, nothing in it is edited, and it stays
runnable in the `snek` conda env for A/B. Code duplication between the two directories is expected
and is the point.

## Decisions taken 2026-08-28

| | decision |
|---|---|
| **Eval** | **One stage, not two.** The training self-eval is the screen; every checkpoint at **≥95/100 perfect** gets a **500-episode** measurement. That single pass replaces both snek2's close-out and its HOF re-measure. §6 |
| **Desktop** | snek3's daemon takes over the box on the same three branches. Every spec carries a `project` field so a stale snek2 spec cannot be dispatched. §9 |
| **Gate** | Full DDQN reproduction before PPO — phases 3 and 5 both stand. §10 |
| **Day one** | Reward shaping (free, already inside `vec_env`), the exploration shield, and the forking collector. **c51 deferred.** |
| **Observation** | **Identical to snek2** — 9x9 playable, 30 values, era `b09c616`. Phases 1-2 depend on it |
| **Champion transfer** | **Yes.** Convert `b44a-lowlr7-b29b-ckpt2739000` and make its 98.73%/3000 the phase-1 gate. §10 |
| **Instructions** | **Three `CLAUDE.md` files.** snek2's 1,080 lines of mechanics move verbatim into `snek2/CLAUDE.md`. §14 |
| **Self-eval** | **Synchronous, 100 episodes every 1,000 steps.** ~2 h an arm. Async is a measured follow-up, not a day-one default. §5 |
| **Stage-A gate** | **95/100.** Tunable later; it is the cost knob |
| **Tools dir** | **`tools/`**, not `scripts/` — it holds libraries as well as executables |
| **Conda env** | New `snek3` env: python 3.12, torch, numpy, pygame, matplotlib, pillow, imageio, **pytest** |

The eval decision has a consequence worth stating up front, because it inverts an earlier draft of
this plan: **the graph self-eval is now load-bearing measurement, not a progress readout.** It has to
run at 100 episodes on every checkpoint, so it cannot be made cheaper, and it stays ~90% of a
training arm's wall clock. What it stops being is *waste* — snek2 paid for those 300,000 episodes and
then paid again to re-measure the same checkpoints.

---

## 1. Why, in four numbers

Everything here was measured in snek2, mostly by
[`snek2/plans/snek3-pytorch-rewrite.md`](../../snek2/plans/snek3-pytorch-rewrite.md) on
2026-08-23. That document is the feasibility study behind this one; read it for the arithmetic, not
this section.

| fact | number | consequence for snek3 |
|---|---|---|
| TF-Agents `policy.action`, batch 1 | **217 us**, of which the network is ~1% | the framework is the bill, not the arithmetic |
| a torch batch-128 gradient step | **245 us** (4,074/s) | the training half caps at ~4,000 agent steps/s without changing the replay ratio |
| the vectorised numpy env, 1024 lanes | **196k env-steps/s, 221 episodes/s** | already built in snek2 and parity-exact — snek3 inherits it |
| snek2's training self-eval | **73% of a training arm's wall clock** at 100 episodes | it becomes ~90% in snek3, and under the single-stage protocol that is the *measurement*, not overhead |

Two of those have already been banked in snek2: the vectorised eval engine shipped 2026-08-24 and
the self-eval moved onto it 2026-08-27. So the remaining prize is the training loop, and the honest
size of it is **5-30x**, not the 42x the feasibility study projected for a whole arm — because most
of the arm-level 42x is eval, and eval is already collected.

**The real reason to do this is not throughput.** It is that PPO, and anything else worth trying
next, needs a framework where a custom loss is fifty lines instead of a subclass of a library agent;
and that snek2's 43,036 lines of Python and 12,099 lines of Markdown have reached the point where
the cost of a change is dominated by reading. snek3 targets **~7,000 lines of Python and ~1,200
lines of docs** for strictly more capability, and the reduction comes almost entirely from deleting
paths that no longer run.

---

## 2. Layout

```
snek3/
  README.md              one page: what runs, how to run it
  CLAUDE.md              snek3's own instructions — see §14
  train.py               train.py <policy> [--algo dqn]
  evaluate.py            evaluate.py <policy|batch> [selector]
  watch.py               watch.py <policy> [step]
  record_gif.py          record_gif.py <policy|hof>

  env/                   the scalar game. pygame lives here and nowhere else
    constants.py           geometry, rewards, paths — no pygame import
    render.py              colours, fonts, tile pixels, drawing
    game.py                Snake.Game
    observations.py        the 30-value vector, scalar reference implementation
    scalar_env.py          reset/step/obs, no framework wrapper

  vectorized/            N games in lockstep, pure numpy, no pygame, no torch
    config.py              grid geometry shared with env/constants.py
    vec_env.py             VecSnake
    engine.py              measure_stream / measure — the policy_fn seam
    shard.py               one process measuring one slice of one arm
    wave.py                the multiprocess controller

  dqn/                   DDQN
    net.py  replay.py  agent.py  collect.py  schedules.py
  ppo/                   later. README only until then

  tools/                 the tools and the libraries behind them
    eval_plan.py  run_report.py  arch.py  checkpoints.py
    selectors.py  progress_chart.py  chart_viewer.py  eval_progress.py  seeding.py

  desktop/               the git-bus job queue, unchanged in design
    runner/  systemd/  config/  README.md

  docs/                  the investigation. starts nearly empty
    README.md  running.md  environment.md  protocol.md  invariants.md
    runs.md  results.md  findings.md  charts.md
  plans/                 designs. this file
  charts/                diagnostic figures docs link to — not per-arm charts
  tests/

  runs/  evals/  savedPolicies/  hallOfFame/  gifs/     outputs
```

Four choices in there are deliberate and worth stating.

**`env/` owns pygame and nothing else does.** In snek2 `snake_constants.py` builds a `pygame.Rect`
at import, so `vectorized/config.py` — a file with no drawing in it — pulls in pygame and its audio
trap. Splitting the pure geometry into `env/constants.py` and the drawing into `env/render.py` means
the entire vectorised and eval path imports no pygame at all. That kills the `SDL_AUDIODRIVER`
class of bug by construction rather than by discipline.

**`vectorized/` holds the batched env as well as the batched driver.** It is one unit: a pure-numpy
reimplementation plus the scheduler that feeds it. `env/` is its parity reference and the only thing
that can draw.

**`charts/` holds diagnostic figures, not per-arm charts.** snek2 copies `runs/<policy>.png` into
`hyperparamTuning/charts/` so `charts.md` can link a stable path, and that duplication is the
entire reason for `refresh_charts.sh`, the completeness-check snippet, and the ~40 lines of CLAUDE.md
that surround it. `docs/charts.md` will link `../runs/<policy>.png` directly. `charts/` then holds
only the handful of one-off figures a finding refers to.

**`tools/`, not `scripts/`.** It holds libraries as well as executables — `eval_plan.py` and
`run_report.py` are imported by `train.py` and `evaluate.py` — and "scripts" would promise only the
second kind.

---

## 3. What comes from snek2

Roughly 5,600 lines of snek2's eval stack collapses to ~2,000, and about 1,200 of those are
near-mechanical copies. The table is the whole port.

### Copy near-verbatim — 2,400 lines, the load-bearing half

| snek2 | lines | snek3 | note |
|---|---|---|---|
| `vectorized/vec_env.py` | 738 | `vectorized/vec_env.py` | pure numpy, parity-exact, imports only numpy + config. **Stays numpy — see §4.** |
| `vectorized/vec_engine.py` | 337 | `vectorized/engine.py` | the `policy_fn(obs) -> actions` seam and the lane scheduler |
| `state_helpers.py` | 739 | `env/observations.py` | the scalar reference `vec_env` is asserted against |
| `Snake.py` | 738 | `env/game.py` + `env/render.py` | trim the retired display paths |
| `record_gif.py` | 597 | `record_gif.py` | four seams change (§8); everything else is framework-free and hard-won |
| `run_report.py` | 226 | `tools/run_report.py` | fix one wart: `max_score` is a string `"95/95"` while `min_score` is an int |
| `under_the_hood.display_progress` + `trailing_average` | ~150 | `tools/progress_chart.py` | keep the `Figure`/`FigureCanvasAgg` rule; `plt.subplots` leaked ~1 MB an eval and OOM'd the desktop |
| `under_the_hood.fold_episode_sample` | ~35 | `tools/run_report.py` | the shared fold, and the single perfect-game counter |
| `desktop/runner/{config,job,gitbus,trigger}.py` | 596 | same | zero snek2 imports; coupling is string-level only |
| `vectorized/README.md` | 656 | `docs/environment.md` (excerpted) | the engine's design record. Excerpt, don't copy whole |

### Copy simplified

| snek2 | lines | snek3 | what goes |
|---|---|---|---|
| `eval_plan.py` | 1012 | `tools/eval_plan.py` (~400) | keep the payload/row schema, `wilson_interval`, `RowCache`, `WriteGate`, `merge_checkpoint_evals`, both selectors. Drop the three-stage protocol, the abandon gate, `equal_effort_pooled`, and every pre-2026-08-08 branch — see §6 |
| `eval_progress.py` | 1387 | `tools/eval_progress.py` (~450) | keep the analysis and text layers, minus the stage/screen/abandon/`num_workers` paths. Rewrite the 370-line matplotlib block. Drop `pyformulas` window mode |
| `vectorized/vec_wave.py` + `vec_eval.py` | 945 | `vectorized/wave.py` + `shard.py` (~450) | drop the 35-line `sys.path` bootstrap duplicated in both, the chart-viewer lock negotiation, the dead protocol fields |
| `eval_wave.py` | 1328 | `tools/selectors.py` (~120) | keep only the argv layer: `parse_selector`, `resolve_policies`, `arms_for_prefix`, `batch_of`, `describe_selector`. The lane/thread/worker-pool model is what `vectorized/` replaced |
| `policy_arch.py` | 328 | `tools/arch.py` (~120) | keep `obs_len`, `obs_era`, `num_actions`, the atomic sidecar, `assert_same_network`. Torch's `load_state_dict(strict=True)` covers the shape half; **`obs_era` is the half no shape check can catch** |
| `self_eval.py` | 177 | `dqn/collect.py` (inline) | one `tf.function` becomes a `torch.no_grad()` call. The fresh-seed-per-eval rule stays |
| `forking_collector.py` | 330 | `dqn/collect.py` (~250) | day one. It exists because the buffer holds the consequence of the action *taken* at an endgame decision point and never the alternative, so `Q(s, a_good)` for the untaken safe action trains on nothing. Default in snek2 since batch 17; the record holder used 4 branches |
| `chart_viewer.py` | 1010 | `tools/chart_viewer.py` (~300) | see §7 |
| `training.py` epsilon schedule | ~180 | `dqn/schedules.py` | the two-phase bootstrap/refine schedule ports as pure functions |
| `shielded_policy.py` | 137 | `dqn/agent.py` (inline, ~30) | a mask over the exploration draw only, never the greedy action |
| `prioritized_replay_buffer.py` | 176 | `dqn/replay.py` (~150) | **replace cpprb with a numpy sum tree** — cpprb silently ignores `seed=`, which is incompatible with §5 |
| `desktop/runner/{runner,launch}.py` | 1295 | same (~1,050) | drop `_ensure_viewer` (~150 lines of sticky-PNG-set logic); rewrite `build_command` |
| `desktop/{README,SETUP}.md` | 978 | `desktop/README.md` (~350) | keep the failure catalogue, drop the TF and eval-engine specifics |

### Drop

| snek2 | lines | why |
|---|---|---|
| `eval_checkpoints.py` | 887 | the scalar engine's CLI. Dead on both hosts since 2026-08-24 |
| `eval_workers.py` | 432 | spawned TF worker processes. Keep the doctrine, not the code: truncating in-flight episodes reads **high**, because perfect games average ~1,780 steps against ~2,200 for failures |
| `eval_agent.py`, `snake_environment.py`, `under_the_hood` net builders | ~400 | TF-Agents constructors |
| `categorical_agent.py` | 391 | c51. **Deferred, not rejected** — it runs on the vec engine fine, but it is a second agent class to maintain before PPO exists |
| `backfill_arch.py`, `pick_c51_lr.py`, `hyperparamTuning/{diagnostics,perDiagnostics}/` | ~5,000 | snek2's measurement tools, tied to snek2's internals |
| `chart_viewer` arm registry, claim lock, `--arms` mode | ~500 | §7 |
| tiered selection, the min-achievable gate, `EVAL_SCREEN_EPISODES`, `EVAL_CONFIRM_COUNT`, the `closeout → HOF` chain | ~600 | §6 |
| `pyformulas` / `cv2` in-process windows | — | already off by default; a fatal XIO error killed four desktop arms at once on 2026-08-09 |

**None of snek2's findings, run tables or batch narratives come across.** `docs/findings.md` starts
empty. What does come across is a short list of *invariants* — facts about the game, the measurement
and the file formats that are cheap to state and expensive to rediscover. That list is §11, and it
is nine bullets, not 3,300 lines.

---

## 4. The env stays numpy, and the policy is the only torch

**Recommendation, against the obvious one.** Do not translate `vec_env.py` into torch tensors.

The temptation is real — a `(n, 1, 12, 12)` boolean board through a masked `max_pool2d` is a much
prettier flood fill than a packed-uint64 dilation, and it would run on the GPU. Three measurements
say no:

- **The policy is 8.2% of an env step** at width 1024; the numpy observation build is 4,296 us of
  5,050 us. So **1.09x is the ceiling for any accelerator, however fast.**
- **`tensorflow-metal` measured 2.4x slower** on the policy call at that width, and — the
  disqualifying part — four hall-of-fame champions measuring 97-98% measured **0.0%** on MPS with no
  error raised. That specific bug is TF's, but the lesson is that a silent zero is the failure mode
  of an accelerator path here.
- `vec_env.py` is 738 lines whose *only* claim to correctness is elementwise parity against
  `env/observations.py` over 18,053 states. Rewriting it and re-earning that parity is the largest
  single risk in this port, spent to buy at most 9%.

So: `vectorized/` is numpy end to end, `torch` appears only inside `policy_fn`, and the seam between
them is one function of shape `(m, 30) float32 -> (m,) int64`. That seam is what already lets
`engine.py` be benchmarked against a hand-written heuristic policy with no framework imported at
all, and it is what let c51 run on the engine without the driver knowing what an atom is.

**The one ablation worth running before paying for the current cost:** region enumeration is 33% of
the connectivity block and exists *solely* for observation indices 10, 12 and 14. A two-flood
shortcut is proven exactly equal to the reference for indices 9/11/13 and 15-17 over all 18,053
states. Dropping 10/12/14 buys ~1.5x on the observation. Batch 45 reached 99% with them in, so this
is a cost question, not a correctness one — and it is cheap and well-posed.

---

## 5. Training loop

### The seam every algorithm sits on

```python
agent.act(obs, epsilon)      -> actions          # (n, 30) -> (n,)
agent.observe(transitions)                        # whatever the algo needs banked
agent.update()               -> {loss: ..., ...}  # one optimisation step, or none
agent.policy_fn              -> callable          # greedy, for engine.measure
agent.state_dict() / load_state_dict()
```

`train.py` owns the parts that are not algorithm-specific: seeding, the arch sidecar, the checkpoint
cadence, the self-eval, the progress chart, the run report, the step cap. `dqn/collect.py` and later
`ppo/collect.py` own their own loops, because an on-policy rollout and a replay-driven step do not
share one.

### Collect width, and the knob that actually sets throughput

snek2 does one env step and one batch-128 gradient step per iteration. That fixes the **replay
ratio** at 1.0 and caps the loop at ~4,000 agent steps/s whatever the env costs. snek3 exposes the
two halves separately:

| knob | default | meaning |
|---|---|---|
| `SNEK_COLLECT_ENVS` | 1 | lanes of `VecSnake` the collector advances per iteration |
| `SNEK_REPLAY_RATIO` | 1.0 | gradient steps per env step |
| `SNEK_BATCH_SIZE` | 128 | as snek2 |

**The default reproduces snek2's dynamics**, and that is deliberate: a rewrite that changes
framework, collector, replay ratio and RNG at once will not reproduce 99% by construction, and this
project's largest single result came from diffing snek2 against `theSchlong`. Framework details
demonstrably matter here.

Raising `SNEK_COLLECT_ENVS` above 1 while holding the ratio at 1.0 buys nothing — the gradient work
scales with it. **Lowering the replay ratio is the only way past ~4,000 steps/s, and it is a
learning-dynamics change, not a free win.** Pre-register it as an experiment.

### The self-eval, which is now stage A of the measurement

Under the single-stage design (§6) the graph eval selects which checkpoints get measured properly,
so three things are pinned rather than tunable:

| | value | why it cannot move |
|---|---|---|
| `SNEK_GRAPH_EVAL_EPISODES` | **100** | the threshold is literally "95 of 100"; a different denominator is a different gate |
| `SNEK_EVAL_INTERVAL` | **1000** | it must equal the checkpoint interval, or a checkpoint exists that no screen can select |
| checkpoint interval | **1000** | as above |

**So the self-eval stays ~90% of a training arm's wall clock, and that is now correct rather than
wasteful.** The arithmetic, for a 3M-step arm:

| | episodes | at 45 ep/s |
|---|---:|---:|
| training (collect + gradient) | — | 12.5 min at 4,000 steps/s |
| stage A, the graph eval | 3,000 × 100 = **300,000** | **1.85 h** |
| **arm wall clock, synchronous** | | **~2.0 h** |

Against snek2's ~20 h for the same arm plus an 8-hour close-out, ~2 h is a 10-14x win and it is
enough. **Start synchronous.** It is the simple thing, it needs no weight-snapshot IPC, and it keeps
the epsilon feedback path exact.

**The next 8x is the asynchronous eval, and it should be pre-registered rather than assumed.** Two or
four worker processes measuring off a weight snapshot would take an arm to ~20-30 minutes. The cost
is that `perfect_percent` feeds `epsilon_for`'s refinement phase, so the exploration schedule goes
onto a lag of one eval interval — this is a feedback loop, not a readout, the same property that made
the `PERFECT_GAME_REWARD` counter bug change the *training* and not only the numbers. One interval of
staleness is almost certainly fine; "almost certainly" is why it is a measured follow-up and not a
day-one default.

Two contracts to keep from `self_eval.py`, both cheap and both deliberate: the eval seed is **fresh
per eval**, derived from `(seed, step)` — a fixed seed would replay the same 100 boards every eval,
which is a different statistic and one that rewards memorising those boards; and each arm of a wave
draws from its own stream, because sharing boards across four seeds correlates their graph noise.

### Checkpoints

Replace the TF `Checkpointer` with two files, which is a straight improvement:

| file | contents | cadence |
|---|---|---|
| `savedPolicies/<policy>/ckpt-<step>.pt` | policy weights only, ~45 KB | every 1,000 steps, gated on `MIN_CHECKPOINT_SCORE` |
| `savedPolicies/<policy>/resume.pt` | weights + target + optimizer + step + RNG state | every 10,000 steps, rolling |
| `savedPolicies/<policy>/arch.json` | as snek2 | once, at start |
| `savedPolicies/<policy>/replay.npz` | the buffer | every 10,000 steps |

A snek2 arm costs ~570 MB of checkpoints (188 KB × 3,000, because every checkpoint carries the
optimizer). A snek3 arm costs **~135 MB**, and an eval loads 45 KB instead of 188. `arch.json`
travels with the weights, as it must — a checkpoint without it will not load.

**Keep `MIN_CHECKPOINT_SCORE` and drop `max_to_keep`.** The rolling window is what destroyed
`b5c-schlongIS`'s 17.0% peak and truncated `b8d`'s first 1.64M steps. At 45 KB a checkpoint there is
no reason for a rotation at all.

---

## 6. Evaluation is one stage

**Decided 2026-08-28.** snek2 measures a policy three times: the graph eval during training (100
episodes), then a close-out (100 episodes, tiered selection), then a HOF re-measure of the close-out's
≥98% rows (500 episodes, flat). The middle stage is redundant now that the graph eval runs at 100
episodes on the vec engine — it re-measures the same checkpoints at the same depth. So it goes:

| stage | who | selection | episodes |
|---|---|---|---:|
| **A** | the trainer, in-process | every checkpoint | 100 |
| **B** | a wave of shard processes, after the arm stops | every checkpoint at **≥95/100** in stage A | **500** |

That is the whole protocol. There is no third stage: **stage B *is* the HOF measurement**, so a
promotion reads its file directly instead of chaining another job behind it.

### What this deletes

`top50`, `ALWAYS_EVAL_SINGLE`, `ALWAYS_FULL_SINGLE`, `EVAL_SCREEN_EPISODES`, `EVAL_CONFIRM_COUNT`,
`EVAL_MIN_ACHIEVABLE`, `equal_effort_pooled`, `abandoned` rows, `pooled_equal_effort`, the
`training → closeout → HOF` chain in the desktop daemon, and the four gate eras a reader has to check
per file before pooling anything. About **600 lines of code and one whole pipeline stage.**

The gate is the load-bearing removal. `EVAL_MIN_ACHIEVABLE=97` abandons a measurement once it cannot
reach 97%, which makes abandoned rows shorter than their neighbours and therefore not comparable
with them — and it left exactly one point of headroom under the HOF selection gate of 98, an
invariant that only a test held together and that had already drifted across two files. With one flat
500-episode pass there is nothing to abandon and nothing to keep consistent.

### The cost, measured off snek2's own files

| snek2 file, vec engine | checkpoints | episodes | wall clock |
|---|---:|---:|---:|
| `b45a-lowlr8-b29b_checkpoint_evals_vec.json` (close-out) | 3,222 | 322,200 | 1 h 51 m, one process |
| `b45a..._vechof500` (4 shards) | 1,568 | 784,000 | 1 h 14 m |
| `b45c-lowlr8-b40b..._vec.json` | 3,031 | 303,100 | 1 h 48 m |

Projected for a snek3 champion-class arm, 3M steps, ~3,000 checkpoints:

| | episodes | 1 process | 4 shards | 16 shards |
|---|---:|---:|---:|---:|
| stage A (inside training) | 300,000 | 1.85 h | — | — |
| stage B, if ~50% clear ≥95/100 | 750,000 | 4.6 h | 1.2 h | 18 min |
| stage B, if ~70% clear | 1,050,000 | 6.5 h | 1.6 h | 25 min |

**Total measurement compute is roughly what snek2 already spends** — the win is that snek2's 322,200
close-out episodes were duplicated work, and that a batch now has one eval wave to queue and wait for
instead of two chained ones. The honest saving is ~25% of the compute and 50% of the pipeline.

**`≥95/100` is the cost knob, and it is where the design is worth revisiting first.** It is a
threshold on a noisy estimator: a checkpoint whose true rate is 0.90 reads ≥95/100 about 4% of the
time, so false positives are rare — but on a champion-class arm most checkpoints genuinely clear it,
which is why stage B is 2-3x stage A. Raising it to 98/100 would cut stage B by roughly half; the
reason not to, yet, is that snek2 found 6 of 8 arms' best checkpoints *below* the tier that got
full-length measurement, and this project has repeatedly had to lower such a threshold rather than
raise it.

### One protocol consequence, and it is not optional

**Selecting on stage A and reporting stage B is sound — the samples are independent. Reporting the
*maximum* over stage B is a selected high.** That is exactly why `b29b`'s 99.0%/500 re-measured at
97.5% over 1,000 fresh episodes, and why the four best hall-of-fame entries fell a mean 1.4 pp when
re-measured. So a **record claim needs a fresh measurement of the single winner**, at 1,000+
episodes, run after the fact. Stage B ranks candidates; it does not certify one.

### What is kept from `eval_plan.py`, and why

`RowCache` and `WriteGate` look like micro-optimisations and are not. Without them the controller
rebuilt every banked row and re-serialised the whole file 125 times per 500-episode measurement — 58
s of single-threaded bookkeeping against the 46 s four lanes needed to produce one measurement, so
the controller overtook its own lanes and `b43`'s HOF pass folded a backlog for 90 minutes with 16
workers idle. Keep both, and keep the diagnostic: when `grep -c "episodes in"` runs ahead of
`grep -cE "^\[ *[0-9]+/"`, the controller is the bottleneck, not the box.

Keep the row schema whole, including `episode_scores` / `episode_perfect` / `episode_rewards`. Those
three arrays at ~1.6 KB a row are what make a pass resumable and a median poolable; storing only
summaries lost 192 rows and 7,534 episodes once. At 500 episodes a row is ~8 KB, so a 2,000-row file
is ~16 MB — large enough to be worth knowing about, small enough to keep committing.

Keep three contracts from the engine that each have an incident behind them: episodes are banked at
the slot they **started** in, not appended on completion (episode length correlates with outcome, so
completion order put a 20-of-100 prefix at 0.25% failures against a true 2.23%); `max_live` is
**derived** as `max(8, 4·ceil(width/episodes))` and raises if it is too small (getting it wrong
measured 4% utilisation and a 10x slowdown); and the stall cap raises rather than spinning.

**Do not carry `num_workers` semantics.** In that schema it means "episodes advance in indivisible
rounds of this size", not "how parallel is this". Reporting the batch width there inflated an ETA by
10.24x.

The two selectors that remain are `screen:<threshold>` — read `runs/<policy>_evals.json`, take every
checkpoint at or above the threshold, default 95 — and `above:<threshold>` reading a prior stage-B
file, for the record re-measure. Nothing else.

### What is kept from `eval_plan.py`, and why

`RowCache` and `WriteGate` look like micro-optimisations and are not. Without them the controller
rebuilt every banked row and re-serialised the whole file 125 times per 500-episode measurement — 58
s of single-threaded bookkeeping against the 46 s four lanes needed to produce one measurement, so
the controller overtook its own lanes and `b43`'s HOF pass folded a backlog for 90 minutes with 16
workers idle. Keep both, and keep the diagnostic: when `grep -c "episodes in"` runs ahead of
`grep -cE "^\[ *[0-9]+/"`, the controller is the bottleneck, not the box.

Keep the row schema whole, including `episode_scores` / `episode_perfect` / `episode_rewards`. Those
three arrays at ~1.6 KB a row are what make a pass resumable and a median poolable; storing only
summaries lost 192 rows and 7,534 episodes once.

Keep three contracts from the engine that each have an incident behind them: episodes are banked at
the slot they **started** in, not appended on completion (episode length correlates with outcome, so
completion order put a 20-of-100 prefix at 0.25% failures against a true 2.23%); `max_live` is
**derived** as `max(8, 4·ceil(width/episodes))` and raises if it is too small (getting it wrong
measured 4% utilisation and a 10x slowdown); and the stall cap raises rather than spinning.

**Do not carry `num_workers` semantics.** In that schema it means "episodes advance in indivisible
rounds of this size", not "how parallel is this". Reporting the batch width there inflated an ETA by
10.24x.

---

## 7. Charts and the viewer

`under_the_hood.display_progress` ports almost unchanged into `tools/progress_chart.py` — the
twin-axis layout, the guides, the title burned into the image, `.partial.png` + `os.replace`, and
above all the `Figure` + `FigureCanvasAgg` construction. `plt.subplots` kept its artists alive
through pyplot's global figure manager, leaked ~0.45 MB an eval, and OOM'd the desktop. Do not
revert it.

`chart_viewer.py` is 1,010 lines of which **~500 are auto-spawn machinery**: a `pgrep`/`ps` liveness
scan, an arm registry with a grace period, an `O_EXCL` claim lock, zombie detection, and a dedupe.
Every one of those lines is a dated scar, and they all trace to one requirement: *four peer training
processes each independently try to open one shared window, and none of them knows about the
others.*

**Remove the requirement.** In snek3 the **launcher** opens the window — one explicit process per
wave, given an explicit file list, which is exactly what the desktop daemon already does and why the
desktop never had any of these bugs. A `tools/launch_wave.sh` (or the desktop's dispatcher) starts
four trainers and one viewer. That deletes the registry, the claim lock, the zombie handling, the
`pgrep` corroboration and the `--arms` mode: **~300 lines of viewer and ~150 of test** instead of
1,010 and 1,825.

Five things from the scars are kept regardless, because they are cheap and each was expensive to
learn:

- `exit_now()` closes the figures *first*, then `os._exit()`. Interpreter shutdown with a live Tk
  window aborts the process and pops a macOS crash dialog.
- The SIGTERM handler is installed **after** `subplots()`, because Tk installs its own inside the
  first one. An install before it is dead code — 5 of 5 kills still aborted.
- `flush_events` and a short sleep, never `plt.pause`.
- A **hard panel cap**, selected by mtime. Nothing sweeps `evals/`, so a glob ages badly.
- A negative liveness answer needs corroboration. `pgrep` exits 0 on a match, 1 on no match and ≥2
  on an error, and all three produce empty stdout; reading only stdout turned a *failed* check into
  the strongest possible answer and closed a live window with five hours left to run.

---

## 8. Watch and record_gif

Both port with a single seam change and are the highest value per line in the set.

`record_gif.py`'s 597 lines are almost entirely framework-free and encode real, hard-won domain
knowledge: the GIF container stores delays in hundredths of a second and **truncates**, so asking
for 60 fps writes a 10 ms delay which every browser clamps to 100 ms — the request lands 2.5x
*slower* than the default, in a file whose stored delay is exactly what was asked for. 50 fps is the
ceiling. One frame is one game step. The game cannot draw either of its own ending frames, so the
filled board and the death position have to be redrawn and the message composited over them. The
HUD is blitted under the sprites, so it must be suppressed on `Snake`'s module globals — not
`snake_constants`', because the star import bound copies. Size is driven by frame count, not
resolution or palette.

The TF coupling is four places, all lazily imported inside `main`: the import block, the checkpoint
restore, the `ckpt-N.index` scan, and three tf-agents calls in `play_episode`. In snek3 they become
`torch.load` + `load_state_dict` + `net.eval()`, a `ckpt-*.pt` glob, and a plain
`argmax(net(obs))` loop under `torch.no_grad()`.

`watch.py` is the same shape: ~40 of 159 lines are plumbing. Keep the checkpoint-following loop (it
re-reads the newest checkpoint between episodes, so an open window shows a live arm improving), the
fps cap inside `render()` where the 5.2 ms flip actually is, and the `SNEK_TILE_PIXELS`-before-import
rule. Dedupe `available_steps` / `checkpoint_steps` — snek2 has them in both files.

---

## 9. Desktop

The design is good and stays: a stdlib-only systemd daemon that imports nothing from the project and
talks to the laptop only through three single-writer git branches. `ops` (laptop writes job specs and
`runtime.json`), `ops-status` (desktop writes `status.json`), `results` (desktop writes artifacts).
The coupling to snek2 is entirely string-level — script names, the `SNEK_*` vocabulary, artifact
paths, a `^(b\d+)` batch regex — so `runner/{config,job,gitbus,trigger}.py` port untouched and
`runner.py` needs only `build_command` rewritten and `_ensure_viewer` deleted.

Keep, because each is a documented incident: the boot-id reattach that marks a job **`interrupted`**
rather than `done` after a reboot (before 2026-08-13 this published truncated arms as finished and
silently consumed their close-outs); the wave barrier, so trainings and evals never overlap; the
`--force-with-lease` single-writer discipline and the stale-lock sweep; the 10-minute `git_seconds`
against the 30-second `poll_seconds`, so work the box generates for itself starts in seconds while
github traffic stays at ~144 fetches a day; and `trigger`, which forces a cycle and answers "did it
start?" in one round trip.

**Fix two known bugs while porting rather than carrying them.** `publish_results` has no retry, so a
failed push leaves the commit local while the ledger says `done` — indistinguishable from a pass that
legitimately found nothing, and it once hid four HOF-500 files including a 98.2%/500 checkpoint for
hours. And a `failed` close-out is silently never retried, which cost `b46` wave 1 its measurement;
surface it in `at_a_glance` at minimum.

**Simplify the eval chain to match §6.** `_auto_closeout_jobs` synthesises one eval job per
`(batch, closeout_group_env)` and `auto_hof` then chains a second job behind each. There is only one
stage now, so the `auto_hof` path, the `-hof` id handling and the `phase_of` legacy branch all go.
`EVAL_RELEVANT_ENV` stays, and so does the reason for it: keying a wave on the *whole* inherited env
split `b45` — four arms differing only in `SNEK_SEED` — into three waves of 2/1/1, measuring a batch
at a quarter of the intended lanes. A seed, a learning rate or a target-update period cannot reach a
measurement of an already-trained checkpoint; shaping and reward knobs can.

**snek3's daemon takes over the box** (decided 2026-08-28), on the same three branches. Every spec
gains a required `project` field, validated in `job.py`, and the daemon refuses anything that is not
`snek3` — so the ~150 stale snek2 specs sitting in `queue/pending/` on `ops` cannot be dispatched by
accident, and a future snek4 inherits the guard. snek2 desktop jobs are retired; snek2 stays runnable
on the laptop.

---

## 10. Phases and gates

Each phase has a pre-registered pass condition. Phase 1 is the one that makes the rest cheap.

| # | phase | gate |
|---:|---|---|
| 0 | The instruction split and the `docs/` skeleton (§14), then `env/` + `vectorized/` + the parity harness. No learning code. | three `CLAUDE.md` files, no stale `snek2/` reference outside `snek2/`; parity harness green — 0 mismatches on all 30 indices over ≥18,000 states, ≥12 hand-made mutants killed. **Met 2026-08-28**: 36,000 states × 30 indices, 0 mismatches, 17 of 17 mutants killed, 167 tests green |
| 1 | **Import a snek2 champion.** Convert its TF weights to a torch `state_dict`. | `engine.measure` scores `b44a-lowlr7-b29b-ckpt2739000` at **98.7% ± 0.6 pp over 3,000 episodes** (snek2: 98.73%). `watch.py` plays it; `record_gif.py` records it |
| 2 | The eval wave, `run_report`, `arch`, charts. | convert **all 3,222** checkpoints of `b45a-lowlr8-b29b` and reproduce snek2's own `_checkpoint_evals_vec.json` row for row within noise |
| 3 | `dqn/` — DDQN + PER + the epsilon schedule + the forking collector + the shield. | one arm reaches **≥90% perfect**, and **≥1,500 agent steps/s** on the laptop with the self-eval *off*, ~2 h for a 3M-step arm with it on |
| 4 | `desktop/` | a 4-arm batch dispatches, runs its one eval wave, and publishes without a hand touching the box |
| 5 | A seed-matched b47-class comparison, 4 arms. | a ≥98%/500 region of comparable width on comparable seeds. **Not** a matched point estimate |
| 6 | `ppo/` | the actual research |

Phase 2 runs on an **explicit step list**, not the `screen:95` selector — the converted checkpoints
have no snek3 graph evals to screen on. That is worth keeping as a third selector (`steps:<file>`)
precisely because it is the only way to reproduce a measurement exactly.

**Phase 1 is the whole trick, and it is cheap.** Every snek2 policy is a plain MLP —
`b44a`'s is `30 → 320 (relu) → 3`, and the TF checkpoint holds exactly
`_sequential_layers/{0,1}/{kernel,bias}`. Converting is a transpose. That single conversion validates
the env, the observation vector, the policy path, the eval engine, the watcher and the recorder
**before any training code exists**, against a number snek2 measured over 3,000 episodes. A ~40-line
script buys the strongest test in the plan.

**Phase 2 extends it into a 3,222-row A/B between two independent stacks.** It costs ~2 hours of
compute and it is the only way to know the flat eval agrees with the tiered one it replaces.

**Phase 5 is the honest gate on the port, and it should not be set at 99%.** `b29b`'s 99.0%/500 is
partly a selected high — it re-measured 97.5% over 1,000 fresh episodes, and the four best HOF
entries fell a mean 1.4 pp on re-measurement. A snek3 arm that misses 99.0% is not a regression.

---

## 11. Invariants carried from snek2

Nine bullets, for `docs/invariants.md`. Everything else in snek2's 3,300-line `findings.md` is a
result about a hyperparameter and is deliberately left behind.

1. **A perfect game is identified by its score, never by its reward.** A reward is a sum of terms,
   so anything derived from it breaks silently when a term is added. When `CHASE_SAFE_SHAPING`
   shipped, a perfect game paid 99.9 instead of 100 and all three counters read 0% — and because
   `perfect_percent` feeds the epsilon schedule, eight arms trained *handicapped*, not just
   mismeasured, for 300k+ steps. Keep one definition and one AST tripwire.
2. **`arch.json` travels with the weights, and `obs_era` is the field that matters.** A checkpoint
   restores whenever the vector is the same *length*; nothing checks the values still mean what they
   meant. On 2026-08-02 two indices were repurposed at constant length and every champion restored
   silently and played like a beginner — 90.3% to scoring 0, 0, 1. Torch's `strict=True` catches
   shape; only the era marker catches meaning.
3. **1 means good or safe throughout the observation, and new blocks append at the end.**
4. **An input that fires in 0.01% of states cannot be trained, however informative it looks.**
   `perfect_game_move` (indices 18-20) is nonzero in 0.000-0.025% of states; no policy here has ever
   learned to win from it. The ones that win do it through board-fill, which is rank 1 of 30 by
   saliency in every arm measured. Measure occupancy before adding a block.
5. **`PERFECT_GAME_REWARD` and `DISCOUNT` cannot be tuned independently.** With `k` steps per meal,
   progress only raises value when `W > 1/(1 - γ^k)` — 34-58 at this project's numbers. Batch 33 cut
   the win to 10, missed by 3-6x, and the agents correctly learned to avoid finishing.
6. **Never call bare `pygame.init()`.** It opens a real CoreAudio stream per process; 10 idle
   workers drove `coreaudiod` to 15% CPU. In snek3 the vectorised path imports no pygame at all.
7. **Rendering costs ~5.2 ms a frame** — a round trip to the window server, not our drawing. Training
   never draws; `watch.py` and `record_gif.py` are the only ways to see a game.
8. **A rate compares across an episode-count change; a threshold crossing does not.** Fewer episodes
   means more noise, which *raises* a maximum and *raises* a threshold-crossing fraction, so the arm
   with fewer episodes always looks better than it is. `strong_eval_fraction` at a true rate of 0.50
   reads 9.3x higher on 10 episodes than on 20.
9. **This domain is very noisy** — the same config has produced 62.5 and 18.0. Never conclude from a
   single run; n=4 cannot resolve an effect below ~10 pp.

Plus one process rule, which is not about the game: **never delete `runs/`, `hallOfFame/`, or a
user's own `savedPolicies/train/`.** A wrongly kept file costs a few KB; a wrongly deleted one costs
a training run.

---

## 12. Still open

Everything in this section was answered on 2026-08-28 and is recorded in the decisions table at the
top of the file. Nothing is outstanding.

The one item deliberately parked rather than settled: **the asynchronous self-eval** (§5). It is the
next 8x on training wall clock and the only thing standing between a 2-hour arm and a 20-minute one,
but it puts `epsilon_for`'s refinement phase on a lag of one eval interval. Revisit once a phase-3
arm has a measured curve to compare a lagged one against.

---

## 13. Risks

| risk | mitigation |
|---|---|
| **The port does not reproduce snek2's results, and nobody can tell whether that is the framework or a bug.** This is the whole risk. | phases 1-2 validate env, obs, policy and eval against snek2's own numbers *before* any training code exists. A phase-3 failure is then localised to the learning code by construction |
| **A session working in snek3 follows snek2's instructions**, because the root `CLAUDE.md` is 1,283 lines of them and loads every time. | §14, and it is a phase-0 deliverable, not a tidy-up at the end |
| Translating `vec_env.py` loses the compaction tricks and reads as "vectorisation does not help". | do not translate it (§4). Copy it. The two compaction fixes are worth 5.1x at n=1024 and 8.5x at n=16384, and a naive version costs the same as the scalar path |
| The self-eval silently becomes 97% of wall clock. | §5. Measure the share on the first real arm and report it, the way snek2 did by sampling `/proc` |
| Determinism is claimed and not delivered. | cpprb ignores `seed=`; that is why `dqn/replay.py` is a numpy sum tree. Add a fixture: two runs at one seed produce identical eval curves for 50k steps |
| The desktop's two known bugs come across with the port. | §9 |
| snek2 gets edited by accident. | it is frozen in `snek3/CLAUDE.md` and in the root file's header |

---

## 14. Instructions and docs — a phase-0 deliverable

**The root `CLAUDE.md` is 1,283 lines and essentially all of it is snek2 mechanics.** 69 of those
lines name `snek2` directly, and of the sixteen top-level sections only four —
*Work as a collaborator*, *Environment*, *Git workflow* and *Markdown traps* — are about the
repository rather than about snek2's specific files. The rest is `snek2/hallOfFame/`,
`snek2/hyperparamTuning/`, the `evals/` sweep that no longer exists, `SNEK_*` knob semantics,
`chart_viewer`'s claim lock, the scalar eval protocol's gate eras, and so on.

That file loads into every session. Left alone it is an active hazard: a session working in snek3
would be told to run `refresh_charts.sh`, to check `pgrep -fl "python -u snek2.py"`, to read a
`min_achievable` out of a payload that no longer has one, and to keep `charts.md` in sync with a
directory snek3 does not use that way.

**Recommendation: three files, split by scope.**

| file | length | contents | when it loads |
|---|---:|---|---|
| `CLAUDE.md` (root) | ~200 | what the repo is and its three eras (`theSchlong` → `snek2` → `snek3`); collaboration rules; git workflow; the two compute hosts; markdown traps; never-delete; **"snek2 is frozen, snek3 is active"** | always |
| `snek2/CLAUDE.md` | ~1,080 | every snek2-specific section, moved verbatim | only when a session touches `snek2/` |
| `snek3/CLAUDE.md` | ~250 | snek3's rules, written fresh | only when a session touches `snek3/` |

Moving snek2's mechanics *into* `snek2/CLAUDE.md` rather than deleting them is what makes this safe:
nothing is lost, snek2 becomes self-documenting, and a nested `CLAUDE.md` is loaded only when files in
that directory are in play — so a snek3 session stops paying for it and stops being misled by it. It
adds a file to `snek2/` and edits nothing that is already there, which is inside the freeze.

### Checklist

1. **Root `CLAUDE.md`**: cut to the four project-wide sections plus an era header. Re-point or delete
   every surviving `snek2/...` path. The *Environment* section names the `snek` conda env — it needs
   both, and it needs to say which project each belongs to.
2. **`snek2/CLAUDE.md`**: the twelve snek2-specific sections, verbatim, with a one-line frozen notice.
   Paths become relative to `snek2/` where the move makes them wrong.
3. **`snek3/CLAUDE.md`**: written, not ported. The two-host topology, the single-stage eval protocol,
   the invariants of §11 by reference, the test command, and the freeze on snek2.
4. **Root `README.md`**: 13 lines mention snek2. Add snek3 and mark the eras.
5. **`.gitignore`**: add `/snek3/savedPolicies/` and `/snek3/gifs/`. Do **not** touch the anchored
   `/results/` rule or move anything into `.git/info/exclude` — that file lives in the common git dir,
   is shared by every linked worktree, and the same pattern there silently stops the desktop
   publishing results with no error anywhere.
6. **Cross-references**: `grep -rn 'snek2/' --include='*.md' snek3/ CLAUDE.md README.md` after each
   step. A broken `](file.md#anchor)` renders as normal text rather than erroring, so nothing surfaces
   it — this project has already shipped sixteen of them at once.

**Do it in phase 0, before any snek3 code exists.** Both directions of this go wrong if it is left to
the end: a snek3 file written under snek2's instructions, or a snek2 instruction quietly deleted
because it looked stale from inside snek3.
