# `vectorized/` — a batched numpy env and a batched eval driver

A drop-in replacement for the **measurement** half of this project. Training is untouched.

The point is arithmetic: **99.65% of an arm's ~1.45B env steps are measurement**, not learning. So the
cheapest large win available is not a faster optimiser — it is making an env step cost less and making
a policy call decide more than one move. Both come from the same change: stop running one pygame
`Game` per process and run N boards as numpy arrays.

| file | what it is |
|---|---|
| `config.py` | every constant, **imported** from `snake_constants` rather than copied |
| `vec_env.py` | `VecSnake` — N games in a packed-bitboard env, byte-identical observations |
| `vec_engine.py` | the batched measurement loop: one wide env, many checkpoints, migrating lanes |
| `vec_eval.py` | the CLI driver; writes the existing result schema through `eval_plan` |

Tests live in `../tests/`: `test_vectorized_config.py`, `test_vectorized_env.py`,
`test_vectorized_parity.py`, `test_vec_engine.py`.

## Using it

**`vec_wave.py` is the default engine on both hosts** (2026-08-24). It takes `eval_wave.py`'s CLI, so a
close-out is one command and the HOF re-measure chains off it:

```
cd snek2
PYTHONPATH=. python -u vectorized/vec_wave.py --chain top50 b45      # a batch's whole measurement
PYTHONPATH=. python -u vectorized/vec_wave.py top50 b45a-x b45b-y    # arms named explicitly
```

`vec_eval.py` is the single-arm tool underneath it, and stays the way to run one arm by hand:

```
PYTHONPATH=. python -u vectorized/vec_eval.py <policy_name> [selector]
```

`selector` is `top50` (default), `above:98` (the HOF pass), `all`, or explicit comma-separated steps —
the same selectors `eval_checkpoints.py` takes, because they *are* `eval_plan`'s selectors. `vec_wave`
parses argv with `eval_wave`'s own functions rather than its own copies, so `top50`, `above:98`,
`--chain` and a bare batch id cannot come to mean two different things depending on which was typed.

| knob | default | notes |
|---|---|---|
| `VEC_EVAL_EPISODES` | 100 | 500 for a HOF re-measure |
| `VEC_EVAL_WIDTH` | 1024 | total env lanes |
| `VEC_EVAL_MAX_LIVE` | derived | **leave it derived** — see the capacity trap below |
| `VEC_EVAL_SEED` | 0 | the only stochastic input to a measurement |
| `VEC_EVAL_SHARD` | unset | `i/n`, 1-based — this process's strided slice of the selection |
| `VEC_EVAL_RESUME` | `1` | reuse depth-matching rows already in the output file |
| `VEC_EVAL_CHART_DIR` | `evals/vec` | `vec_wave` overrides it to `evals/` |
| `EVAL_OUT_SUFFIX` | `_vec` | so a **hand-run** can never overwrite a TF result |

`vec_wave.py` adds two of its own, and passes everything above through:

| knob | default | notes |
|---|---|---|
| `VEC_WAVE_PROCS` | `os.cpu_count()` − 2 | shards to spread across the wave's arms. **Hardware threads, not cores** — see the sweep below |
| `TF_NUM_INTRAOP_THREADS` | TensorFlow's | not ours, but **1 is +3.6% on the desktop**; a shard is single-threaded numpy |
| `EVAL_EPISODES` | 100 | stage A's depth; stage B is `eval_plan.HOF_EPISODES` |
| `EVAL_OUT_SUFFIX` | none | the **canonical** path, because a wave *is* the close-out |

**‡ `VEC_WAVE_PROCS` is bounded by cores after all, and one shard is 553-601 MB rather than 690**
(swept on `the-claw-den` 2026-08-24 with
[`perDiagnostics/vec_wave_sweep.py`](../hyperparamTuning/perDiagnostics/vec_wave_sweep.py): 240
checkpoints of `b45a-lowlr8-b29b`, 100 episodes each = 24,000 episodes per config, the same explicit
step list every time, top rows repeated). **Run-to-run noise on this box is under 1%**, against ~5% on
the laptop, so 1% differences are real here:

| procs | width | intra-op | episodes/s | CPU idle | peak RSS | min `MemAvailable` |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1024 | default | 238.0 | 74.8% | 2.4 GB | 9.7 GB |
| 8 | 1024 | default | 326.3 | 50.1% | 4.7 GB | 8.4 GB |
| 12 | 1024 | default | 344.1 | 28.9% | 7.0 GB | 7.1 GB |
| 14 | 1024 | default | 358.5 | 17.3% | 8.3 GB | 6.7 GB |
| 16 | 1024 | default | 361.2 | 5.7% | 9.4 GB | 6.1 GB |
| 18 | 1024 | default | 328.9 | 5.8% | 10.7 GB | 5.1 GB |
| 20 | 1024 | default | 313.0 | 5.8% | 11.9 GB | 4.6 GB |
| 24 | 1024 | default | 317.7 | 4.1% | 14.0 GB | 3.7 GB |
| 12 | 1024 | **1** | 360.7 | 29.9% | 6.7 GB | 7.8 GB |
| 14 | 1024 | **1** | 371.4 | 18.3% | 7.8 GB | 7.2 GB |
| **16** | **1024** | **1** | **373.9** | **5.8%** | **8.9 GB** | **6.7 GB** |
| 18 | 1024 | **1** | 337.5 | 6.9% | 10.0 GB | 6.1 GB |
| 16 | 2048 | default | 320.8 | 6.4% | 9.5 GB | 6.0 GB |

**The box is a Ryzen 7 9700X: 8 physical cores, 16 SMT threads.** `os.cpu_count()` reports the
threads, so `DEFAULT_PROCS = cores − 2` has always meant *threads* − 2 here — and this file and
CLAUDE.md both called it a "16-core box", which is where the 14 came from. The natural prediction from
the topology, that 14 shards on 8 cores is the oversubscribed regime the laptop measured as 20% slower,
is **wrong**: throughput climbs past the physical core count all the way to 16, so SMT is worth about
+10% here (326 → 361) even though one shard saturates about one core. What is right is that the
default was never measured on this host.

**Three readings, in order of what they are worth.**

- **Pinning the thread pools is the largest single win, +3.6%, and it is free.** The sweep sets
  **three** knobs together — `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1` and
  `OMP_NUM_THREADS=1` — so the gain belongs to the bundle; which one carries it is not yet isolated.
  Threads live in one process after 50 matmuls, measured on the box:

  | environment | threads |
  |---|---:|
  | default | **50** |
  | TF intra + inter = 1 | 20 |
  | `OMP_NUM_THREADS=1` alone | 35 |
  | all three | **5** |

  **There are two pools, not one, and they are independent**: TF's Eigen pools are ~30 of those
  threads and oneDNN's OpenMP pool is ~15 (`test_util.IsMklEnabled()` is True here), which is why
  `OMP_NUM_THREADS` is in the bundle at all. So **a 16-shard wave at the defaults runs ~800 threads on
  16 hardware threads; pinned it runs ~80.** For a workload that is 95% single-threaded numpy in the
  observation build, with TF ops small enough that the fork-join barrier costs more than the tiling
  saves, those 720 threads are pure scheduler contention — and Eigen's workers spin before sleeping, so
  they burn cycles the other fifteen shards need. That is why the loss shows up as throughput at 5.8%
  idle: part of "busy" is spin-wait. Pinning also takes ~35 MB off each shard.

  **`tf.config.threading.get_intra_op_parallelism_threads()` cannot verify any of this** — it reads the
  `ConfigProto` field, which stays 0 (= auto) whatever the environment says, because TF reads the env
  var when it *creates* the pool (`process_util.cc`) rather than into that field. It returns 0 both
  with and without, so it reads as "the knob did nothing". Count threads instead.

### ‡ It is the inter-op pool, and only that (2026-08-24)

Isolated at 14 shards on the desktop, each row the mean of two runs of 24,000 episodes:

| pinned | episodes/s | vs default | RSS/shard |
|---|---:|---:|---:|
| nothing (TF's default) | 358.1 | — | 589 MB |
| `TF_NUM_INTRAOP_THREADS` only | 359.2 | +0.3% | 589 MB |
| `OMP_NUM_THREADS` only | 356.9 | −0.3% | 591 MB |
| **`TF_NUM_INTEROP_THREADS` only** | **369.3** | **+3.1%** | **557 MB** |
| inter + intra | 370.2 | +3.4% | 558 MB |
| all three | 371.9 | +3.9% | 556 MB |

**Inter-op alone is the whole effect** — intra-op and oneDNN each do nothing, and adding them to
inter-op adds nothing beyond noise. That is the pool TensorFlow's executor uses to dispatch
independent ops, so at TF's default sizing each of 14 shards dispatches its graph across a 16-thread
pool: ~800 threads on 16 hardware threads, for ops small enough that dispatch costs more than it
parallelises. The ~33 MB a shard comes from the same pool.

**‡ On the laptop the same change is null, and the first attempt to measure it tested the wrong
config.** Two things make arm64 different, both measured:

- **`OMP_NUM_THREADS=1` *undoes* the reduction there.** Threads in one process: 30 at the default, 17
  with inter-op pinned, 17 with intra-op pinned, and **30 again with all three pinned** — reproducible
  across samples. On the desktop the same bundle goes 50 → 5. So the first laptop A/B, run at `1`
  (all three), was comparing 30 threads against 30 threads; its null result measured nothing at all.
- **Pinning inter-op alone shrinks the laptop's pool (30 → 17) and buys CPU rather than speed.**
  Throughput is flat — paired A/B at 12 shards gives **+0.55%** across four pairs (+2.83, −0.89,
  −3.85, +4.11) and RSS is unchanged at 537-539 MB — but **CPU idle goes 9.4% → 15.0%**, i.e. the same
  work for **~0.8 fewer of the 14 cores**. The two hosts differ in where the saving lands, not in
  whether there is one: the desktop runs 14 shards on 8 physical cores and is CPU-bound, so removing
  dispatch overhead becomes +3.6% throughput; the laptop runs 12 shards on 14 real cores, so each
  shard already owns one and the saving has nowhere to go but idle. **On the laptop that is the more
  useful half anyway**, since it is the interactive machine.

  **Idle resolves this effect where throughput cannot.** The pinned runs read 15.0% and 15.0% against
  9.4% mean for auto, while `episodes/s` scatters ±4% — so on a noisy host, measure the utilisation
  and not just the rate.

**‡ Do not read the `load` column for a short run.** Load average is a 1-minute EMA and a config here
takes 65-100 s, so each row carries the tail of the row before it — the CPU-utilisation pair above read
`load` 13.3 for auto and 16.2 for pinned, the *opposite* order to their idle, and two back-to-back rows
both read exactly 17.0. `idle` is a delta taken over precisely the run and is the trustworthy column;
`load` is only worth reading across runs long enough for the EMA to converge.

**The laptop needs paired configs and the desktop does not**, which is worth knowing before running
either sweep again. The laptop's run-to-run spread is **±4%** against the desktop's **under 1%**, and a
laptop sweep in launch order also shows a **warm-up drift** — the same `default` config read 270.8 in
round 1 and 305.8 in round 2, and every round-2 config beat its round-1 twin, which inverted the
ranking between rounds. Alternating A/B/A/B cancels a monotone drift; reading configs in launch order
does not, and that is how `omp1` came out "best" on one laptop round while doing nothing on the
desktop.
- **The cliff is at `cpu_count`, not at the cores.** 16 → 18 loses 6-10% and it never recovers; 20 and
  24 are 12-13% down. "Run it harder" past 16 is strictly worse, and **the peak already sits at ~6%
  idle** — there is no configuration on this box that is both busier and faster, so the last few
  percent of idle is not slack that can be converted.
- **Width 2048 is 11% worse**, the same direction the laptop measured.

**The memory table this section used to carry was pessimistic by 3-4 GB**, because it subtracted
`procs × 690 MB` from `MemAvailable` — and `MemAvailable` counts reclaimable page cache, which the
kernel duly hands back. Measured: 16 shards peak at **8.9 GB resident and leave 6.7 GB available**, not
the ~1.6 GB the arithmetic predicted, and even 24 shards (14.0 GB) leave 3.7 GB without swapping
anything to death. Per-shard RSS is **553 MB at `intraop=1`, 585-601 MB at TensorFlow's default** — the
590-600 MB figure being roughly flat in the shard count is what says the cost is TF's arena plus the
1024-lane env rather than anything shared.

So the OOM risk this section was built around is smaller than it looked, but it has not disappeared:
four trainers add ~4.2 GB, which against a 16-shard wave's 8.9 GB is 13.1 GB of 15,030 MB. **14 shards
at `intraop=1` cost 0.7% of throughput and 1.1 GB of memory**, which is the trade to make if a wave has
to overlap live training. Both RSS figures are still short-run (64-101 s per config), so a multi-hour
wave's memory remains unmeasured — watch `free -m` through the first long close-out.

The laptop's measured point of 12 × width 1024 is a throughput optimum on *its* topology (14 physical
cores, no SMT); neither host's number transfers to the other, and the sweep script is how a third host
would get its own.


Two traps in that knob. **`EVAL_WORKERS` does nothing here** — it sizes TF worker processes and this
engine has none — but `launch.py` reads `job.eval_workers or runtime['vec_wave_procs']`, so an old job
spec still carrying `eval_workers: 4` silently *caps* the wave at 4 shards. And both RSS figures are
short-run (54-68 s); no multi-hour wave's memory has been measured on the box yet, so watch the first
long close-out rather than assuming the peak is the plateau.

### ‡ MPS (Metal) is disqualified, and being slower is the lesser reason (2026-08-24)

Tested with `tensorflow-metal` 1.2.0 in a **clone of the `snek` env** — identical numpy 1.26.4, TF
2.15.1 and tf-agents 0.18.0, differing only by the plugin — with the CPU arm produced by hiding the GPU
inside that same env (`set_visible_devices([], 'GPU')` from a `sitecustomize`), so numpy could not drift
between the arms. numpy is 92% of a step, so a cross-env comparison would have confounded the result
with a numpy version.

**It computes the wrong policy.** Four hall-of-fame champions, 300 episodes each:

| checkpoint | CPU | MPS |
|---|---|---|
| 1447000 (`b29b`) | **97.3%** perfect, avg score 94.69 | **0.0%**, avg score 0.44 |
| 1342000 (`b24d`) | **98.0%**, 94.36 | **0.0%**, 0.25 |
| 1513000 (`b40b`) | **96.3%**, 94.55 | **0.0%**, 0.10 |
| 2860000 (`b24b`) | **97.3%**, 94.58 | **0.0%**, 1.99 |

**And it is not float tie-breaking**, which is the explanation this codebase would reach for first,
since near-ties flipping an argmax between batch widths is already documented here. The Q-values agree
to **5.7e-06**; bare `argmax`, graph-mode `argmax` and `tfp.Categorical.mode` are each individually
**correct** on Metal; `act()` is deterministic across repeats on both devices; the restored weights are
bit-identical. But the composed `GreedyPolicy(QPolicy)` graph disagrees with `argmax` over its own
Q-values on **23 of 64 states**, and the action it discards is worse by a **median 0.64** in reward
units — 22 of the 23 gaps exceed 0.1, and none is under 1e-4. A 5.7e-06 perturbation cannot cross a
0.64 gap. Every component is right and the composition is wrong.

**The failure mode is the dangerous part: a silent 0.0% and a *faster* wall clock.** The MPS run
finished in 8.3 s against 27.4 s, because the episodes died immediately — 12x fewer env steps for the
same episode count. Nothing raised, nothing warned. The tell was `env-steps/s` and `utilisation` in the
run's own footer, not the wall clock. This is the same shape as the observation-era trap in the
hall-of-fame README: a checkpoint that restores and plays like a beginner.

**It would also have been slower.** Two measurements, before correctness was even in question:

| batch | CPU | MPS | |
|---:|---:|---:|---|
| 256 | 250 us | 1044 us | 4.2x slower |
| **1024** (the operating width) | **390 us** | **919 us** | **2.4x slower** |
| 4096 | 830 us | 960 us | 1.2x slower |
| 16384 | 2402 us | 1471 us | 1.63x *faster* |

MPS carries a fixed ~900 us a call — 256 and 1024 cost nearly the same, which is dispatch and transfer,
not arithmetic — and only overtakes the CPU above ~8000 rows. Our net is 11,650 MACs a row, so a
1024-row batch is **~24 MFLOP**: far too little to amortise a round trip. At the operating width that is
**~10% slower end to end**, and the crossover does not rescue it, because the numpy env cost scales with
width — at 16384 lanes a step is ~74 ms and the policy share falls to ~2%, so the 722 us saved is worth
under 1%. A wave also runs `VEC_WAVE_PROCS` shards that would all contend for one GPU.

**‡ The decision is closed, and the test env is deliberately gone** (user's call, 2026-08-25, taken with
everything above in hand): MPS is too slow to be useful for this project's purposes, so the
`snek-mps` clone and the throwaway comparison harness were deleted rather than kept. **The findings are
kept and this section is why** — do not rebuild the env to re-derive them. `tensorflow-metal` is not
installed in `snek` and nothing in the repo reaches for a GPU, so an eval here is CPU-only by
construction rather than by configuration; there is no flag to get this wrong.

`eval_device_split.py` **stays**, because it is not MPS-specific: its main output is the numpy/TF split
above, it runs CPU-only in `snek` (printing one CPU row, which is itself the answer to "is an
accelerator even visible"), and its `--verify` guards *any* new device, accelerator build or TF version
— the guard is worth more than the plugin it caught.

**The general point, and the reason not to revisit this without new evidence: the bottleneck is not on
the GPU's side of the fence.** The policy is **8.2%** of a step at width 1024 (413 us of 5050 us) and the
observation build alone is 4296 us — so **1.09x is the ceiling for any accelerator, however fast**. That
observation build is a bitboard flood fill in numpy, which is not a tensor program. Re-take both numbers
with [`perDiagnostics/eval_device_split.py`](../hyperparamTuning/perDiagnostics/eval_device_split.py),
and **run its `--verify` before trusting any new device, accelerator build or TF version** — the failure
mode is a silent zero, which looks exactly like a bad arm.

### Where the output goes, and why the two tools differ

`vec_eval.py` alone writes `_vec` and `evals/vec/`; a wave writes the canonical
`runs/<policy>_checkpoint_evals.json` and `evals/<policy>_eval_progress.png`.

That is not an inconsistency, it is the point. The probe defaults exist so a hand-run can never
overwrite a TF result and so a vec eval and a TF eval can run side by side during a validation
comparison — which is the whole reason to run one. A **wave** is the close-out, so it has to land where
`eval_progress.best_of`, `select_checkpoints_above`, `refresh_charts.sh`, the desktop's publish globs
and every tuning doc already look. `vec_wave` passes both to its children explicitly, so neither tool's
behaviour depends on the other's default.

**Nothing is moved out of `evals/` to make room, by anything, since 2026-08-24.** Every eval used to
sweep that folder into `evals/archive/<timestamp>/` on startup; it is gone, and an arm rewriting its
own chart by name is all the correctness that was ever needed. The viewer's lock namespace still
follows the *directory* rather than the tool: at `evals/` a vec eval contends for the same `-eval`
slot a TF eval would, because two viewers over one directory would each show the other's panels.

## What it is worth — measured on this laptop, 2026-08-23

Reference: `eval_checkpoints.py` as **4 processes x 4 workers** (the standing throughput point, the
whole machine). Vec: **one process**. Same 144 checkpoints of `b43c-lowlr-b40b`, 100 episodes each,
flat, no abandon gate on either side.

| | wall clock | per checkpoint | processes |
|---|---|---|---|
| `eval_checkpoints.py`, 4x4 | **1684 s** | 11.7 s | 4 (+16 workers) |
| `vec_eval.py`, width 1024 | **267 s** | 1.86 s | 1 |
| `vec_eval.py`, width 2048 | 255 s | 1.77 s | 1 |

**6.3x end-to-end on a quarter of the processes — about 25x per lane.** Utilisation was 89% at width
1024. A single 500-episode measurement of one champion checkpoint is 16 s against ~183 s, so the
per-measurement figure is ~11x and the rest of the gain is keeping the batch full across checkpoints.

Three measurements explain the shape, and they are the ones to re-take if any of this stops holding:

| what | number |
|---|---|
| policy inference, batch 512 | 1.5M rows/s |
| env step **without** the observation, n=512 | 211 us |
| env step **with** it, n=512 | 4323 us |
| env step+obs throughput at n = 128 / 256 / 512 / 1024 / 2048 | 55k / 88k / 118k / 151k / 174k steps/s |

So **the observation is 95% of a step and the policy is nearly free** — the inverse of the per-worker
shape, where `tf.function` dispatch dominates. Two things follow: width is the lever, and one wide env
can serve several checkpoints at once with a policy call each.

## Filling the box: 12 processes, not 4

One process saturates about **one core** — the observation build is single-threaded numpy and is 95%
of a step — so a four-process run leaves most of a 14-core laptop idle. Measured on 240 checkpoints
of `b45a`, 100 episodes each, machine-wide (10 P-cores + 4 E-cores, 36 GB):

| processes | width | episodes/s | CPU idle | wall |
|---|---|---|---|---|
| 4 | 1024 | 168 | 59% | 151 s |
| 8 | 1024 | 269 | 35% | 95 s |
| **12** | **1024** | **347-350** | **2-6%** | **76 s** |
| 13 | 1024 | 337-340 | 0.5% | 80 s |
| 14 | 1024 | 329 | 0.0% | 82 s |
| 16 | 1024 | 280 | 0.0% | 99 s |
| 12 | 512 | 305 | 1.6% | 87 s |
| 12 | 2048 | 329 | 5.7% | 88 s |

**12 x width 1024 is the operating point.** Past it the box is oversubscribed and throughput *falls*
while idle sits at zero — 16 processes are 20% slower than 12 — so "0% idle" is not the target;
0-5% with the highest throughput is. Run-to-run noise is ~5%, so 12/13/14 need repeat measurements to
separate: 12 won both repeats on both throughput and wall clock.

Against `eval_checkpoints.py` at its own standing point of 4 processes x 4 workers (**8.55 eps/s**
machine-wide), 348 eps/s is **40.7x**. An earlier note in this file put the figure at 22x and called a
pre-registered 40x gate missed; that measurement was taken at 4 vec processes with 59% of the machine
idle, and it was the process count that was wrong rather than the engine.

`VEC_EVAL_SHARD` is how the processes are fed, and **`vec_wave.plan_shards` is what fills it in** —
one `vec_eval.py` process per shard, all started together, waited on, then merged. The budget is
**machine-wide**, so four arms measured together get three shards each, not twelve: allocation is
proportional to each arm's selection size (largest-remainder), because a wave's arms differ by an order
of magnitude — b45's HOF selections were 1568 / 1264 / 1173 / 298 — and equal shares would leave the
small arm's processes finished and its cores idle through the tail. Every arm with work gets at least
one shard even if that overshoots the budget: dropping an arm to hold a process count would silently
not measure it. Shards are **strided**
(`steps[i-1::n]`), because per-checkpoint cost tracks policy quality and quality drifts along a run —
contiguous blocks would hand one shard every slow checkpoint. Each shard writes its own
`<suffix>-sNofM` file and `eval_plan.merge_checkpoint_evals` puts them back together; the merge is
lossless for disjoint shards, which is what a stride guarantees.

**The merge is a row combiner, not a payload builder**, and that needed one more step.
`merge_checkpoint_evals` writes seven top-level keys and drops everything else `build_payload` emits —
`requested_steps`, the protocol fields, the progress fields. Left alone that would make an arm's
close-out file depend on *how many processes measured it*: a one-shard arm and a twelve-shard arm in
the same wave publishing different-shaped files. So `vec_wave.stitch_payload` rebuilds the merged file
through `eval_plan.build_payload` — the one definition — with the merged rows and the protocol read
back out of the shards rather than restated. A field added to `build_payload` therefore reaches a
merged file for free.

**A shard's result file is also its resume state.** `VEC_EVAL_RESUME` (on by default) reuses any row
already in the output file whose episode count matches the request, so a wave killed partway through
re-measures only what it had not finished. The shard files are kept for that reason rather than cleaned
up after the merge.

## Episode order in the output, and the ETA

Two things in the result file were wrong in ways that only showed up under analysis.

**`episode_scores` is in start order, not completion order.** `eval_plan.equal_effort_pooled` truncates
a row to a common prefix, and its correctness rests on the episodes being exchangeable — "the first 20
of a 100-episode measurement are as good a 20-episode sample". Appending on completion breaks exactly
that, because episode length correlates with outcome: a starving lane burns its whole 500-step budget
after its last meal while a perfect game ends sooner, so failures finish last. Measured on 40
checkpoints of `b45a`, failures sat at mean position **0.92** of a completion-ordered array, and a
20-of-100 prefix read **0.25%** failures against the row's true **2.23%**. The row's totals were never
wrong — only its order — so `_Job.record` now banks each episode at the slot it was *started* in.

### ‡ A *prefix* of one row is fair; prefixes pooled *across an arm* are not (2026-08-24)

Start-order fixed the within-row property `equal_effort_pooled` needs. It did **not** make start-slot
`k` independent across rows, and on a 100-episode close-out it is strongly not. Measured on `b45a`'s
3222 rows:

| | slot 0 | slots 1-2 | slots 3-19 | whole row |
|---|---|---|---|---|
| failure rate | **11.58%** | 1.96 / 1.43 | 0.25-0.68 | 2.71% |

So a 20-episode prefix of that file reads **1.13%** failures against the rows' true **2.71%** — 1.6 pp
optimistic — and pooling prefixes across the arm gave per-arm offsets of **+1.4, −1.4, +1.4, +1.0 pp**
against the TF close-out, with **random sign**. A bias has one sign; this is variance with a tiny
effective *n*.

**The mechanism is the shared food RNG plus resident count.** One `VecSnake` owns one RNG for all
lanes, so an episode's food stream depends on *when* its reset happened in the global sequence. At 100
episodes `default_max_live(1024, 100)` is **44** checkpoints resident, so slot `k` of 44 different
checkpoints is drawn from nearly the same stretch of that stream — slot `k` becomes a fixed scenario of
fixed difficulty, and one slot in `b45a` fails **54%** of the time. At 500 episodes `max_live` is
**12** and each checkpoint spans a much longer stretch: the same file shows slot 0 at 2.17% against a
2.67% baseline, i.e. no structure at all.

**Nothing published is affected, and that is not luck — it is the flat protocol.** The row totals
average over every slot: dropping slot 0 from all 3222 rows moves the arm's pooled rate by **+0.09
pp**. And `pooled_equal_effort` — the one field that *is* a prefix statistic — is `null` in every vec
file, because it is only computed when `screen_episodes` is set and this engine never screens. So the
statistic this affects is one no vec file reports.

Two rules follow. **Never compare engines on prefixes** — compare full rows at matched depth, which is
what the 500-episode head-to-head did. And **`tests/test_vec_engine.py::test_a_prefix_of_a_row_is_a_fair_sample_of_it`
is a within-row claim only**; it splits one row in half, which averages slot 0 into 250 slots, so it
neither tests nor contradicts any of the above.

Worth knowing when comparing engines: **`eval_checkpoints`' arrays are mildly completion-ordered too.**
On the same rows, a 24-episode prefix of a desktop b45 close-out reads **+0.16 pp** (b45a) and +0.11 pp
(b45c) above the full row. Small, but systematic and in the optimistic direction, so a prefix-based
comparison between the two engines is biased toward whichever is more sorted. Only full-row-vs-full-row
at matched depth is clean.

**`num_workers` must be null in this file.** In this schema the field does not mean "how parallel is
it" — it means "episodes advance in indivisible rounds of this size", and
`eval_progress.remaining_episodes` multiplies every checkpoint still ahead by
`whole_rounds(episodes, num_workers)`. That is right for the batched TF path, which really does run one
episode per worker per round. This engine runs an exact quota, so reporting the 1024-lane width there
rounded each 100-episode checkpoint up to a whole 1024-episode round and inflated the chart's ETA by
**10.24x** — b45's four arms read 6-8 h against a true ~50 min. The fix is null, not a change to
`whole_rounds`, which would under-price a real batched run.

The driver now also publishes `arm_eta_seconds` from its own **completion rate** over a trailing window
of 60, which `eval_progress.summarize` prefers over its own arithmetic. Completions are the right
observable because up to `max_live` checkpoints are in flight at once, so a row's `seconds` field is
concurrent wall clock and summing those over-counts by roughly the residency factor — 18x on b45a.

## The capacity trap — the one number that must not be set by hand

A checkpoint's quota is consumed the moment its episodes are assigned to lanes, but it holds its slot
until its *last* episode ends. So if `max_live * episodes` merely **equals** `width`, then after the
opening assignment no resident checkpoint has quota left, no new one may load, and every lane that
finishes an episode idles until some checkpoint completes outright.

Measured: width 1200, episodes 100, `max_live` 12 ran at **4% utilisation** — 568 s for work that
takes 54 s at `max_live` 24. It looked like the design was simply slow, which is worse than a crash.
`vec_engine.default_max_live` now derives the value with 4x slack and `measure_stream` **raises** on a
configuration that would collapse, with `tests/test_vec_engine.py` pinning both halves.

## Why the parity argument is deductive, not statistical

If the observation is bit-identical then a greedy policy's argmax is identical, so the action sequence
is identical, so the episode is identical, so any measurement over it is identical *by construction*.
That is why parity is asserted elementwise rather than by comparing win rates: a statistical check at
n=500 and ~99% perfect can only bound a bias to about a point, and would never see a divergence that
fires on one board topology in ten thousand.

Food is **forced** in every parity test rather than seeded. The reference rejection-samples the module
global `random` an unpredictable number of times per placement — near a full board that is 20-50 draws
— so no seeding discipline can align the two streams. Distribution equality is a separate question and
is tested separately.

### The evidence, as it stands

| layer | what it does | result |
|---|---|---|
| L1/L2 heuristic | elementwise all 30 indices, growth + coiled endgame + starve + win regimes | **0 mismatches** in 36,000+ states |
| L1/L2 champion | the same lockstep driven by a real 98%-perfect checkpoint, forced food | **0** observation, action, reward and done mismatches in **124,672 states**; 77/80 perfect games |
| L3 mutation | hand-broken variants must be detected | **17 of 17 killed** |
| L4 end-to-end | 144 checkpoints x 100 episodes, vec vs `eval_checkpoints.py` | see below |

The champion lockstep is the load-bearing one. The heuristic covers board *topology* well, but a
champion visits a narrow and different part of the state space — coiled, near-full, long episodes — and
it is the only policy whose decisions ever get measured. Zero **action** mismatches also rules out a
float32 near-tie flipping an argmax between batch widths.

### L4, and the trap in reading it

| run | pooled perfect % | mean avg_score |
|---|---|---|
| `eval_checkpoints.py` (4x4) | 93.97 | 92.978 |
| vec, seed 0 | 93.56 | 92.635 |
| vec, seed 1 | 93.84 | 92.711 |
| vec, seed 2 | 94.53 | 93.087 |
| vec, seed 3 | 94.19 | 92.920 |
| **vec, mean of 4** | **94.03** | 92.838 |

Difference of the vec mean from the reference: **+0.06 pp, z = +0.25**. The reference sits mid-spread.

**Read this table, not the first row of it.** Seed 0 alone came in 0.42 pp low, and on `avg_score` that
reached t = -2.73 — which, taken by itself, looks like a real systematic deficit. It is not: two runs
of the *same* engine differed by a comparable amount (+0.20 +- 0.16), and the across-seed sd is
0.42 pp, twice the naive binomial SE because the pooled rate inherits the spread of 144 different
checkpoints. **One run cannot resolve an effect of this size**, which is the same lesson this project
already carries as "never conclude from a single run" — it applies to validating an instrument, not
only to comparing arms.

Two smaller notes on comparability. `avg_reward` is not expected to match a file measured under
different `SNEK_*` shaping, since the reward is a sum of configured terms — `config.describe()` is
printed in the run header for exactly that reason. And **do not validate against a set of checkpoints
selected by the measurement you are comparing to**: rows chosen because the TF pass scored them >= 98%
have an upward-biased TF value, so an unbiased re-measure reads low by construction. The 144 here were
chosen by *step*, evenly across the arm, independent of both engines.

## Deliberate differences from `eval_checkpoints.py`

| | `eval_checkpoints` | here |
|---|---|---|
| staging | screen / confirm tiers | **flat** — every checkpoint gets the same episode count |
| abandon gate | `EVAL_MIN_ACHIEVABLE=97` | **none** |
| algorithm | ddqn or c51 | the same — both read off `arch.json` |
| `in_flight` payload block | one checkpoint's progress | omitted |

Flat is not laziness. Staging exists to avoid paying full length for a checkpoint that will not place;
at this throughput that saving is small, and it costs a lot of interpretive load — `pooled_equal_effort`,
the `screen_episodes` field, the rule that rows of different depths must not be pooled, and the gate
recorded in every payload all exist to cope with rows of unequal effort. Flat rows make all of that
vacuously true. Same for the gate: an abandoned row is shorter than a full one, so a file holding both
cannot be pooled directly and every reader has to know which gate produced it.

**‡ c51 works here, and it needed no atom arithmetic** (2026-08-24). This engine shipped refusing
categorical policies via `policy_arch.refuse_categorical`, on the stated reasoning that supporting them
meant reading the support out of `arch.json` and reducing over atoms in `vec_eval.py`. That was wrong
about *where* the reduction lives. `AgentPool` never touches a Q head — it builds through
`eval_agent.build_eval_agent`, which picks the agent class off the sidecar, and then calls
`policy.action(...)`. A `CategoricalDqnAgent`'s policy is a `GreedyPolicy` over a `CategoricalQPolicy`,
which computes `argmax_a sum_i z_i p_i(s, a)` internally from the support the sidecar named. Deleting
the refusal was the whole change; `split_arms` and the `eval_wave.py` fallback went with it, so a wave
is one engine again.

Validated the same way the ddqn switch was: six `b38a-c51fc320eps3125seed1` checkpoints spanning
35-96%, 200 episodes per checkpoint per engine, flat and ungated on both sides.

| step | scalar | vec | diff |
|---:|---:|---:|---:|
| 480000 | 75.5% | 76.5% | +1.00 pp |
| 1214000 | 40.0% | 35.0% | −5.00 pp |
| 2355000 | 93.0% | 93.0% | 0.00 pp |
| 2408000 | 93.0% | 90.5% | −2.50 pp |
| 2687000 | 88.0% | 94.5% | +6.50 pp |
| 2818000 | 81.5% | 80.5% | −1.00 pp |
| **pooled (1200 ep each)** | **78.50%** | **78.33%** | **−0.17 pp, z = −0.10** |

Per-checkpoint scatter is binomial: at n=200 a difference has SE ≈ 4.7 pp, so the +6.50 is z ≈ 1.4 and
the −5.00 is z ≈ −0.9. The pooled figure is the comparison, and it matches the ddqn result (−0.058 pp,
z = −0.28).

**‡ And the range of the support turns out not to matter to an evaluation at all.** Found while
mutation-testing the above: `sum_i p_i = 1`, so replacing `z` with `a·z + b` replaces every action's `Q`
with `a·Q + b`, which for `a > 0` leaves the argmax alone. Measured over 256 states, supports `[-5, 120]`,
`[-10, 10]`, `[0, 1]` and `[-1000, 3]` chose the **same action every time**; only a *reversed* support
(`v_min > v_max`) differed, on all 256, because it is then an argmin. So `v_min`/`v_max` are **not** the
field a c51 eval can be silently wrong about — `num_atoms` is, since it sets the logits width and a
mismatch fails the restore on shape. The invariance covers greedy actions only: anything reading a `Q`
*value* still needs the trained support, which is why `tests/test_c51_eval_path.py` pins the support at
construction rather than through the chosen actions.

`in_flight` is omitted because the payload's block describes **one** checkpoint and this driver has up
to `max_live` in flight at once; naming one of twelve would misreport the other eleven. Nothing is
lost, since that block exists to make a ~5-minute measurement visible and here they land every two
seconds.

## Assumptions that would break it

- **Square board, and `PERFECT_SCORE == PLAY * PLAY`.** `config.py` raises at import otherwise; the
  packed bitboard's row stride and the wall ring both depend on it.
- **The wall ring must stay unplayable.** A dilation shifts by +-1 across a row boundary, and only the
  ring stops an open cell wrapping into the next row.
- **`obs_era`.** The observation's *meaning* is pinned by `snake_environment.OBS_ERA`, checked at
  restore by `policy_arch.assert_restorable`. `config.py` deliberately does not import it, so the env
  stays importable without TensorFlow; `vec_eval.py` imports it where it acts on it.
- **Only `groups_mode='full'` is parity-correct.** `'fast'` and `'none'` exist to price the
  connectivity block and must never be used for a measurement.

## Both hosts run this by default now (2026-08-24)

| | how the engine is chosen | opt-out |
|---|---|---|
| laptop | `chain_closeout_after_training.sh` → `vec_wave.py` | `SNEK_EVAL_ENGINE=scalar` |
| desktop | `launch.py` `build_command`, `runtime.json`'s `eval_engine` (default `vec`) | `eval_engine: "scalar"`, or `SNEK_EVAL_ENGINE` in a job spec's `env` |

The knob is kept for two reasons: it is the only way to reproduce a pre-switch measurement, and a
regression in this engine has to be answerable without a deploy. `runtime.json` validates it as an enum
and rejects the whole file on a typo — the daemon then keeps its last-known-good config, which is much
better than a bad value reaching `build_command` and failing every eval dispatch one job at a time.

Three things the switch touched that are easy to miss:

- **`chart_viewer --watch` had to learn both new names.** The pattern is an ERE and a miss reads as
  "the jobs stopped" — six of those in a row close the window on a live wave. `vec_eval.py` matters as
  much as `vec_wave.py`, because the supervisor is one short-lived process per stage while its shards
  are what run for hours.
- **`EVAL_WORKERS` / `EVAL_LANES` are not set for a vec wave**, and `vec_wave` strips them from what it
  passes its shards. They size TF worker processes and this engine has none; a value silently ignored is
  how someone concludes a wave ran with four workers when it ran twelve shards.
- **`chain_closeout_after_training.sh` went 176 lines → 105.** Everything it lost was a second copy of
  something `--chain` and `eval_plan.hof_settings` already own: the per-arm pid bookkeeping, the inline
  `complete` check, the hand-copied HOF recipe, and its own `closeout gate < HOF gate` assertion. Its
  header used to say "copied from `desktop/runner/runner.py`; if that changes, change this too", which
  is the failure mode rather than the mitigation.

## Not done

Training still runs on TF-Agents and pygame — deliberately, so this can be validated against the
existing instrument rather than against a moving target. `snek3/` does not exist yet; when it does,
these files are the ones to copy, and this directory stays as the frozen record that the parity
evidence above refers to.
