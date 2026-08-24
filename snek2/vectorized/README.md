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
| `VEC_WAVE_PROCS` | cores − 2 | shards to spread across the wave's arms |
| `EVAL_EPISODES` | 100 | stage A's depth; stage B is `eval_plan.HOF_EPISODES` |
| `EVAL_OUT_SUFFIX` | none | the **canonical** path, because a wave *is* the close-out |

**‡ `VEC_WAVE_PROCS` is bounded by memory before it is bounded by cores** (measured on `the-claw-den`
2026-08-24). One `vec_eval.py` process peaks at **644 MB at 100 episodes and
690 MB at 500** — flat in the episode count, because the cost is TensorFlow's arena plus the 1024-lane
env, not the resident agent pool (44 agents against 12). So the shard count multiplies ~690 MB:

| procs | memory |
|---:|---:|
| 6 | ~4.1 GB |
| 8 | ~5.5 GB |
| 12 | ~8.3 GB |
| **14** (the box's default: 16 cores − 2) | **~9.7 GB** |

The desktop has **15,030 MB**, ~11.3 GB available idle, and its four trainers add ~4.2 GB. So 14 fits
an idle box with under 2 GB spare and does *not* fit one that is training.

**It is left at the default regardless** (user's decision, 2026-08-24, taken with this table in front
of them). `runtime.json` carries no `vec_wave_procs` key and should be left without one — the full 14
shards are right for the common case, an idle box running a close-out chained off a finished training,
and the exposure is narrow: `max_evals` is 1, so there is only ever one wave, and an OOM'd wave is
recoverable rather than lost (`interrupted` → relaunched, and `vec_eval` resumes from banked rows of
matching depth). **The one configuration that can OOM is a wave overlapping four live trainers; if that
happens, this is the cause and `vec_wave_procs` is the fix** — do not pre-emptively pin it.

The laptop's measured point of 12 is a throughput optimum, not a memory one; check `free -m` on any
host before raising it.

Two traps in that knob. **`EVAL_WORKERS` does nothing here** — it sizes TF worker processes and this
engine has none — but `launch.py` reads `job.eval_workers or runtime['vec_wave_procs']`, so an old job
spec still carrying `eval_workers: 4` silently *caps* the wave at 4 shards. And both RSS figures are
short-run (54-68 s); no multi-hour wave's memory has been measured on the box yet, so watch the first
long close-out rather than assuming the peak is the plateau.

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
