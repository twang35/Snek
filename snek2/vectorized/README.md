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

```
cd snek2
PYTHONPATH=. python -u vectorized/vec_eval.py <policy_name> [selector]
```

`selector` is `top50` (default), `above:98` (the HOF pass), `all`, or explicit comma-separated steps —
the same selectors `eval_checkpoints.py` takes, because they *are* `eval_plan`'s selectors.

| knob | default | notes |
|---|---|---|
| `VEC_EVAL_EPISODES` | 100 | 500 for a HOF re-measure |
| `VEC_EVAL_WIDTH` | 1024 | total env lanes |
| `VEC_EVAL_MAX_LIVE` | derived | **leave it derived** — see the capacity trap below |
| `VEC_EVAL_SEED` | 0 | the only stochastic input to a measurement |
| `EVAL_OUT_SUFFIX` | `_vec` | so a run can never overwrite a TF result |

Charts go to **`evals/vec/`**, not `evals/`, and this driver never calls
`archive_existing_eval_pngs` — so it cannot displace a real close-out's panels, which that function
has already done to two batches. The live window works the same way (`chart_viewer.spawn_for_eval`
with its own `b43-veceval` lock namespace, so it and a TF eval window coexist).

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
| algorithm | ddqn or c51 | **ddqn only**, hard-fail on c51 |
| `in_flight` payload block | one checkpoint's progress | omitted |

Flat is not laziness. Staging exists to avoid paying full length for a checkpoint that will not place;
at this throughput that saving is small, and it costs a lot of interpretive load — `pooled_equal_effort`,
the `screen_episodes` field, the rule that rows of different depths must not be pooled, and the gate
recorded in every payload all exist to cope with rows of unequal effort. Flat rows make all of that
vacuously true. Same for the gate: an abandoned row is shorter than a full one, so a file holding both
cannot be pooled directly and every reader has to know which gate produced it.

**c51 is refused rather than attempted.** The greedy action for a categorical agent is
`argmax_a sum_i z_i p_i(s, a)`, so a c51 checkpoint restored against the wrong support loads perfectly
and evaluates a *different policy*. Until the support is read from `arch.json` and parity-tested,
refusing is the only safe answer — `policy_arch.refuse_categorical` does it.

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

## Not done

Training still runs on TF-Agents and pygame — deliberately, so this can be validated against the
existing instrument rather than against a moving target. `snek3/` does not exist yet; when it does,
these files are the ones to copy, and this directory stays as the frozen record that the parity
evidence above refers to.
