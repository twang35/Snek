# Vectorised env and batched eval

**Status: approved 2026-08-23, implementation started.** Scope is an eval engine only — no change
to snek2's training loop, no PyTorch, no new algorithms. Rests on the measurements in
[`snek3-pytorch-rewrite.md`](snek3-pytorch-rewrite.md), which found that **99.65% of an arm's 1.45
billion env steps are measurement** and that a parity-exact vectorised prototype reaches 176,830
env-steps/s against the ~3,800 the current eval delivers per lane.

Lives in **`snek2/vectorized/`**, with tests in `snek2/tests/`. It stays there when `snek3/` is
created: snek3 gets a *copy*, and this one remains as a working record that does not break as snek3
diverges.

## The decision that shapes everything: TensorFlow stays

The env needs ~176k inferences/s. Batched TF delivers **3.3M rows/s** at batch 1024 (0.30 us/row,
measured), so **inference is not on the critical path and the network is not worth porting yet.**
Phase 1 vectorises only the environment, which is where all of the win is. Keeping TF buys:

- **No weight conversion**, so no silently-wrong policy. That failure mode has bitten this repo
  twice (`expect_partial()`, the `game_over` index).
- **The same network code path b45 used** — `eval_agent.build_eval_agent`, so `arch.json`, the
  observation-era check and `build_q_net` are shared rather than re-implemented. CLAUDE.md is
  explicit that a second copy of that construction is the bug.
- **One process, one conda env.** The prototype runs unmodified under the `snek` env's numpy
  1.26.4, verified before this plan was written, so the vectorised env and TF inference are
  co-resident and the parity tests need no cross-env staging.

## Parity is deductive, not statistical

The argument the test suite is built around:

> If the vectorised observation is bit-identical to the reference under the same food, the greedy
> argmax is identical, so the action sequence is identical, so the episode is identical, so the
> measurement is identical **by construction.**

That reduces measurement parity to trajectory parity and leaves only food *distribution* to check
statistically.

| layer | what it does | pre-registered bar |
|---|---|---|
| **L1 observation** | forced-food states from the real `Snake.Game`, compare all 30 indices | 100% elementwise, >= 20,000 states, lengths 5-99, perfect games included |
| **L2 trajectory** | drive both with identical actions and forced food, whole episodes | 0 mismatches over >= 50 champion episodes, including reward, score, termination and starve |
| **L3 mutation** | hand-broken variants must fail L1/L2 | >= 12 mutants killed, each named in the test |
| **L4 measurement** | the same b45 checkpoints through both engines | `perfect_percent` inside b45's own 95% CI, >= 8 checkpoints spanning the quality range |

**L4 compares only b45 rows with `abandoned: false`** — those are the full-500-episode rows, so they
are directly comparable and the abandon gate need not be reimplemented to do the A/B.

**L4's power is limited and L1/L2 carry the weight.** At n=500 and ~99% perfect, L4 bounds a
*systematic* bias to about +/-1pp; a rare divergence on one board topology in ten thousand would not
show up. That is why L2 runs at champion skill specifically — the coiled endgame boards that stress
the connectivity code only occur there.

## What moves, what is imported, what is left behind

| | |
|---|---|
| reimplement vectorised | grid mechanics, the 30-value observation, food sampling, starve rule, reward terms, perfect-game check |
| import from snek2 | `eval_agent.build_eval_agent`, `policy_arch`, `state_helpers.is_perfect_score` |
| duplicate and pin with a test | board geometry, reward constants, `OBS_ERA` — a test asserts each equals snek2's value, so drift fails loudly. Same pattern as `runner.EVAL_RELEVANT_ENV` |
| do not bring | pygame, sprites, `chart_viewer`, `eval_wave`, the replay buffer, training |

## Three safety rules, built in from the start

1. **Never write into `evals/`, and never call `archive_existing_eval_pngs`.** The tool draws no
   charts, which sidesteps the "any eval displaces every chart at the top level" trap outright —
   load-bearing while b45b/b45d close-outs are live on the desktop.
2. **Output to a distinct `_vec` suffix**, so a real close-out result can never be overwritten.
3. **Hard-fail on c51 by name.** `arch.json` records the algo, and a categorical checkpoint
   restored against the wrong support loads perfectly and evaluates a *different* policy. Phase 1
   supports ddqn and refuses anything else loudly.

## Two implementation details that decide whether this works

**Compaction in the flood fill, from day one.** Vectorised, every board iterates until the *batch
maximum* dilation count — 125 rounds when the median board needs 15. Measured on the prototype: a
diverse batch cost 24.98 us against 3.43 us for identical boards, and dropping converged rows
recovers 5.1x at 1024 envs. **Without it the vectorised observation costs the same as the scalar
one** and the whole exercise is worth ~3x rather than ~50x. This is the highest-risk detail here.

**Episode-quota accounting.** With `N` envs and `M` episodes wanted, autoreset keeps lanes full but
discards whatever is in flight when the `M`th episode completes, so waste is about `N` x half an
episode. At N=256, M=500 that is ~25%; at N=128, ~13%. Default N=256, exposed as a knob. The later
optimisation is `K` checkpoints concurrently (8 x 128 envs), which gets full throughput at low
waste — not needed to hit the target.

## Phasing and gates

| phase | deliverable | gate |
|---|---|---|
| 1 | vectorised env, constants pinned, L1/L2/L3 tests | every parity bar above green |
| 2 | batched eval driver writing the existing JSON schema under `_vec` | L4 against b45a and b45c |
| 3 | throughput measurement and write-up | **>= 40x** per lane on a 500-episode measurement |

The bar is pre-registered at 40x rather than the 80x the prototype suggests, because TF inference
and episode accounting both add overhead the prototype never paid.

## Decisions taken

- **No abandon gate in the vec eval.** It exists only to save episodes, and at 40x the truncated
  rows and the incomparable-pooling they cause cost more than they save. Every selected checkpoint
  is measured at full length, so every row is comparable and `pooled_equal_effort` becomes exact
  rather than a correction.
- **Validate on the laptop.** b45b/b45d close-outs have ~9.3 h invested on the desktop and must not
  be contended with.
- **Food sampling matches distributionally, not by sequence.** The reference rejection-samples
  `random.randint` over the whole board until it misses the snake, which is uniform over free
  cells; the vectorised sampler draws from free cells directly. Same distribution, different RNG
  consumption — which is why L1 and L2 force food rather than trying to align seeds. It also means
  the reference slows near a full board (~20-50 retries per draw at length 95) where this will not:
  a cost difference, not a behaviour difference.

---

## Results, 2026-08-23

All three phases are built. **Parity is established; the throughput gate is missed** — 33x per lane
against a pre-registered 40x. Code is in [`snek2/vectorized/`](../vectorized/README.md), which carries
the full detail; this section records the outcome against the bars set above.

### Parity: every bar green, plus one layer the plan did not ask for

| layer | result |
|---|---|
| L1/L2, heuristic policy | **0 mismatches**, elementwise on all 30 indices, 36,000+ states across growth, coiled endgame, starve and win regimes |
| **L1/L2, champion policy** | **0** observation, **0 action**, 0 reward and 0 done mismatches over **124,672 states**; 77/80 perfect games; max length 100 |
| L3 mutation | **17 of 17** mutants killed, each named in `tests/test_vectorized_parity.py` |
| L4 end-to-end | vec mean of 4 seeds **94.03%** vs `eval_checkpoints.py` **93.97%** — **+0.06 pp, z = +0.25** |

**The champion lockstep is the layer that matters and it was nearly skipped.** The plan said "L2 runs
at champion skill specifically", and the first implementation used a hand-written heuristic instead,
which tops out at length 79 and never visits the coiled, long-episode states a 98%-perfect policy
spends its time in. Driving the real checkpoint found nothing wrong — but it is the only test that
could have, and zero *action* mismatches additionally rules out a float32 near-tie flipping an argmax
between batch widths, which no observation comparison addresses.

**L4 needed four samples, not one, and one would have been read as a failure.** Seed 0 alone came in
0.42 pp low, and on `avg_score` that reached **t = -2.73** — a result that looks like a real systematic
deficit. It is not: two runs of the *same* engine differed by a comparable amount, and the across-seed
sd is 0.42 pp, twice the naive binomial SE because a pooled rate over 144 different checkpoints
inherits their spread. **"Never conclude from a single run" applies to validating an instrument, not
only to comparing arms** — an obvious point in hindsight that was one plausible table away from
retracting a correct implementation.

Two methodological traps worth carrying forward. **Do not validate against checkpoints selected by the
measurement you are comparing to**: rows chosen because the TF pass scored them >= 98% carry an
upward-biased TF value, so an unbiased re-measure reads low by construction. The 144 here were picked
by *step*, evenly across the arm, independent of both engines. And `avg_reward` is not comparable
across different `SNEK_*` shaping at all, since a reward is a sum of configured terms — the driver
prints `config.describe()` in its header for that reason.

### Throughput: 22x like-for-like, 33x per lane, gate missed

Same machine, same work, `b43c-lowlr-b40b` (a continuation arm at champion skill — the expensive case).

| configuration | episodes/s, machine-wide | s/checkpoint (500 eps) |
|---|---|---|
| `eval_checkpoints.py`, 4 processes x 4 workers | **8.55** | ~58.5 |
| `vec_eval.py`, 1 process | 70.2 | **7.13** |
| `vec_eval.py`, 4 processes | **190.5** | 2.62 |

- **22x machine-wide** at the same 4-process allocation, and **33x per lane** on a 500-episode
  measurement at the best measured utilisation (92%).
- End-to-end on the 144-checkpoint/100-episode L4 set: **1684 s -> 267 s (6.3x) using one process
  instead of four.**
- The 4-process figure is pessimistic: each process got only 6 checkpoints, so utilisation fell to
  67% against 92% single-process. A real close-out's hundreds of checkpoints per arm would sit near
  the higher figure.

**The gate was 40x per lane and the answer is 33x, so it is missed.** The plan already discounted the
prototype's 80x to 40x for "TF inference and episode accounting"; the discount was not deep enough.
What the prototype never paid for is that **the observation is 95% of an env step** (4323 us against
211 us at n=512) and its cost falls only slowly with width — 55k steps/s at n=128 rising to 174k at
n=2048 — so a measurement capped at `episodes` lanes cannot reach the prototype's width. Serving
several checkpoints from one wide env recovers most of that and is why the figure is 33x rather than
11x, which is what one checkpoint at a time gives.

### The one bug that mattered, and it was a default

`max_live * episodes` must **exceed** `width`, not equal it. A checkpoint's quota is consumed the
moment its episodes are assigned but it holds its slot until its last episode ends, so at exact
capacity no resident checkpoint ever has quota left and every finishing lane idles. Measured at
**4% utilisation** — 568 s for work that takes 54 s. It presented as "the design is just slow", which
is a worse failure than a crash, and the original defaults (width 4000, `max_live` 12) were inside
that region for any episode count below ~333. `max_live` is now derived, `measure_stream` raises on a
collapsing configuration, and both halves have fixtures.

### Changed outside `vectorized/`

- `chart_viewer.spawn_for_eval` takes `chart_dir` and `slot_suffix` (defaults unchanged), so the vec
  eval's window watches `evals/vec/` under its own lock namespace and can run beside a TF eval.
- `tests/test_perfect_game_counting.py`'s AST tripwire scanned **four hardcoded filenames**, so a new
  counter was outside the rule the day it was written — a mutation test confirmed the reward version
  of the vec engine's perfect flag survived the entire suite. It is now a glob over every top-level
  module plus `vectorized/`, and it matches any name *containing* `PERFECT_GAME_REWARD`, which closes
  a second hole: `final_reward == DEFAULT_PERFECT_GAME_REWARD` was previously invisible to it.

### Not done

The training loop is untouched, deliberately. c51 is refused rather than approximated. `snek3/` does
not exist; when it does, these files are what gets copied, and `snek2/vectorized/` stays as the frozen
record the parity evidence above refers to.
