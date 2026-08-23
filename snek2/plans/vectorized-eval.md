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
