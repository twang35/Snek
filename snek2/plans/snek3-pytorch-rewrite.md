# snek3: a PyTorch rewrite, and where the time actually goes

**Status: investigated 2026-08-23, measured, not built.** This is a feasibility measurement plus a
recommended order of work. Nothing here is approved to implement yet.

Everything below was measured on the laptop (14 cores, arm64) with no trainer running. Two IT
daemons held ~2 cores throughout, so absolute throughputs are conservative; the ratios are
single-threaded and unaffected. A vectorised prototype was built in a scratch directory and
checked for exact parity against the live game before any of its numbers were believed.

## The headline

| | now | snek3 projected | factor |
|---|---|---|---|
| training, 5M steps | **10.3 h** | 21 min + 13 min eval | **~18x** |
| close-out (b45a, measured) | 31.9 h | ~40 min | ~48x |
| HOF-500 (b45a, measured) | 66.6 h | ~1.4 h | ~48x |
| **one arm, end to end** | **~109 h** | **~2.6 h** | **~42x** |

The gain is overwhelmingly in **evaluation**, not training, and the reason is Finding 2 below.

## Finding 1 — the network's arithmetic is ~1% of the framework's cost

| call | cost |
|---|---|
| TF-Agents `policy.action`, batch 1 | 217 us |
| `q_net(x)` `tf.function`, batch 1 | 93 us |
| `q_net(x)` `tf.function`, batch 1024 | 309 us (**0.30 us/row**) |
| `agent.train`, batch 128 | 649 us |
| `agent.train`, batch 1024 | 1585 us (**1.55 us/row**) |

Fitting the two `agent.train` points gives **515 us fixed per call plus 1.05 us per row**: a batch of
1024 costs 2.4x a batch of 128 while carrying 8x the data. In a cProfile of the real loop, TF's own
op dispatch (`TFE_Py_Execute`) is 2.2 s of a 62.6 s run; the remainder is `tf.function` signature
machinery — `convert_to_eager_tensor`, `tensor_conversion_registry.convert`,
`trace_type_builder.from_value`, `inspect._bind`, and 4.2M `isinstance` calls.

**So batching is nearly free, and per-call overhead is the whole bill.** The same profile puts
`ParallelPyEnvironment` startup at 45 s for 20 workers, and a checkpoint restore at 1.9 ms — 1,584
restores in b45a's HOF pass total 3 s against a 66.6 h pass, so restore is not worth optimising.

## Finding 2 — 99.65% of an arm's env steps are measurement

| phase | env steps (b45a, 5M-step arm) |
|---|---|
| collect (training) | 5,000,000 |
| in-training self-eval | 140,300,000 |
| close-out | ~422,000,000 |
| HOF-500 re-measure | ~879,000,000 |
| **total** | **~1.45 billion**, of which training is **0.35%** |

The in-training eval runs **28 env steps of measurement per env step of training** — 20 episodes of
1,403 steps every 1,000 collect steps. Measured wall clock from the four b45 result files, all
written *after* the 2026-08-20 controller fix and so representing the current best case:

| pass | wall clock | episodes | eps/s per 4-worker lane |
|---|---|---|---|
| b45a close-out | 31.9 h | 301,022 | 2.62 |
| b45a HOF-500 | 66.6 h | 626,466 | 2.61 |
| b45c close-out | 28.7 h | 272,928 | 2.64 |
| b45c HOF-500 | 43.9 h | 419,091 | 2.65 |

**171 lane-hours for two arms**, and the rate is flat to within 2% across all four — which is the
signature of a fixed per-step overhead, not of anything about the policies being measured.

## Finding 3 — the blocking eval is ~80% of training wall clock

At champion skill, 1,000 collect steps cost ~1.5 s and the eval that follows them costs 6.1 s
(20 episodes, 20 workers, 4,700 env-steps/s, 1,403 steps per episode). So the effective training
rate on a good arm is **~135 steps/s, not the 690 the console reports.**

Note this got worse when `num_eval_episodes` went 10 to 20 on 2026-08-19. That trade was priced in
close-out episodes (-25% against +15,500 self-eval episodes per arm); the *wall-clock* half of it is
the 80% share above.

## Finding 4 — the flood fill is not the bottleneck

This contradicts the obvious hypothesis, so it is worth stating flatly. cProfile of 6,000
champion-skill env steps (0.552 s, 92 us/step):

| component | share of env step |
|---|---|
| `get_observations` total | 69% |
| — of which `count_groups` | **19%** |
| — of which `get_adjacent_groups` | 12% |
| `Snake.step` (incl. `_rebuild_grid`, pygame sprite groups) | 22% |

`count_groups` runs **1.92 times per step at champion skill, not 3** — `group_obs` short-circuits
fatal moves, and a coiled champion has about one per step. Measured upper bounds: if the flood fill
were *free* the observation would be 1.47x faster, and folding three fills into one gives 1.27x.
Against an env that is 31% of a single-process game loop, that is at most ~5% end to end.

What actually costs is Python call volume: `get_grid_value` is called **132,415 times in 6,000
steps** (22 per step) and `get_relative_pos` 93,722 times. Roughly 300 scalar helper calls per step,
spread across the observation builder. Vectorising deletes all of it; memoising the flood fill
deletes the wrong 19%.

**Corollary: an incremental / memoised flood fill is not the thing to build.** If it is ever wanted,
the right shape is union-find over open cells rather than caching — freeing the tail only ever
*merges* regions, and blocking the new head only ever *splits* one, which is a local articulation
test on at most four neighbours. But it is inherently sequential per env, so it fights the
vectorisation that Finding 5 shows is worth far more.

## Finding 5 — a vectorised env, measured

A prototype (`VecSnake`) steps N games in lockstep with branchless numpy on a padded 12x12 flat
grid, body as a circular buffer of flat cell indices, and the connectivity block as a bitboard
dilation. **Parity is exact**: 100% elementwise match on all 30 observation indices over 18,053
reference states from the live game, snake lengths 5 to 99, including three perfect games, plus 11
step-mechanics fields and a 48-episode sequential rollout with zero mismatches. Twelve hand-made
mutants all fail the harness, so the comparison has teeth.

| num_envs | mechanics only | step + full obs |
|---|---|---|
| 1 | 31,054 | **2,047** |
| 256 | 4,216,715 | 91,435 |
| 1024 | 8,177,431 | 195,591 |
| 4096 | 11,447,620 | **296,488** |

For reference, the live `Game.step + get_observation` measures **12,834 env-steps/s** in one process
at the same length distribution. An independent re-run of the prototype reproduced 163,554 (1024)
and 203,582 (4096), so treat the table as ±30%.

Eval at 1024 envs: **176,830 env-steps/s, 221 episodes/s, 2.3 s for 500 episodes** — against the
183 s a 500-episode measurement takes today. That single ratio, ~80x, is where the 42x per arm comes
from.

MLP 30-320-3 inference: 5.8M rows/s on CPU at batch 1024; **35.5M rows/s on mps at 4096**, so the
GPU is only worth it for pure inference sweeps, and it *loses* inside the training loop.
`torch.compile` made the gradient step **1.6x slower** and is not worth using here.

### Four things that make it worse than the headline

1. **A naive vectorisation is 7-8x worse than it looks.** Every board in the batch runs the flood
   fill for the batch's *maximum* dilation count — 125 rounds at n=1024 when a typical board needs
   ~15 — so a diverse batch cost 24.98 us against 3.43 us for identical boards. Compacting the
   working set (drop a row once it stops growing, in both the flood and the region enumeration)
   recovers 5.1x at 1024 and 8.5x at 16384. **Without that fix the vectorised observation costs the
   same as the scalar one** and the whole exercise is worth ~3x rather than ~23x.
2. **n=1 regresses 6x**, 2,047 against 12,834 env-steps/s. Anything single-game — `watch.py`, a
   debugging session, a one-off probe — gets slower, so the scalar path has to be kept.
3. **Scaling past ~1024 envs adds little** (196k to 296k) and past 4096 almost nothing.
4. **A random-action benchmark overstates all of this by 3-4x**, because snakes sit at length 5 and
   the flood converges in two dilations. Cost peaks mid-game at length 40-60 and *falls* near a full
   board. Every number above is driven by a greedy heuristic with the mean length recorded.

### The training half is capped by the optimizer, not the env

At the current ratio of one batch-128 gradient step per agent step, the gradient step alone runs at
4,074/s, so the loop caps at **~4,018 agent steps/s — 5.8x the present 690, not 400x.** Reaching
42x on the training half (29,362 steps/s) needs the replay ratio cut to one gradient step per eight
agent steps, which is **a learning-dynamics change, not a free win**. Against the *effective* 135
steps/s of Finding 3, though, the unchanged-dynamics number is still ~30x.

**And most of the present loss is not the env at all.** The existing env already delivers 12,834
steps/s in one process while the system delivers 690, so ~95% is TF-Agents / TF / IPC overhead. A
plain PyTorch port with *no* vectorisation would capture most of the training gain on its own.
Vectorising is what unlocks the eval, which is the 99.65%.

## Two design points worth settling before any code

**A perfect game cannot be rendered from the observation.** The 30-value vector is purely relative
features — per-action food direction and reciprocal distance, safety flags, region counts, budgets —
with no absolute position anywhere, so it cannot reconstruct a board. Replaying **actions plus food
positions** does reproduce a game exactly, and `Snake.GameSnapshot` (body, food, head dir, score,
counters) is already the right interface for a thin pygame renderer. So snek3's env should emit that
snapshot for one env index and a renderer should consume it; feeding it observations will not work.

**Only `num_groups` needs region enumeration.** The prototype proved a two-flood shortcut — seed at
the head's neighbours, seed at the food — is *exactly* equal to the reference for `head_with_tail`
and `safe_to_chase_food`, 100% over all 18,053 states. Region enumeration is 33% of the connectivity
cost and exists solely for observation indices 10, 12 and 14. Dropping those three buys ~1.5x on the
observation; dropping the whole connectivity block (9-17) buys ~12x, confirmed independently
at 1.87M env-steps/s with `groups_mode='none'`. That makes indices 10/12/14 a cheap, well-posed
ablation to run *before* committing to their cost — and note batch 45's arms reach 99% with them in,
so this is a cost question, not a correctness one.

## Recommended order

| step | why | risk |
|---|---|---|
| 1. Make the in-training eval non-blocking, in snek2 | ~5x on training wall clock today, no rewrite | low; see the epsilon caveat |
| 2. Vectorised env + batched eval, still measuring snek2 checkpoints | the 99.65%; ~48x on close-out and HOF | medium — needs the parity harness |
| 3. PyTorch DDQN+PER, reproduce a b45-class arm | validates the rewrite | this is the gate |
| 4. Only then: PPO, replay-ratio changes, new algorithms | the actual research | free once 1-3 hold |

**Step 1's caveat is real and not just a report:** `perfect_percent` feeds `training.epsilon_for`'s
refinement phase, so an asynchronous eval puts the exploration schedule on a lag. One eval interval
of staleness is fine, but this is a feedback loop, not a readout — the same property that made the
`PERFECT_GAME_REWARD` counter bug change the *training* and not only the numbers.

**Step 3 is the gate, and it should be treated as one.** A rewrite that changes framework, collector,
replay ratio and RNG will not reproduce 99% by construction, and this project's single largest result
came from diffing `snek2` against `theSchlong` — framework details demonstrably matter here. snek2
must stay runnable for A/B until a snek3 arm has matched a b45-class close-out on the same gate.

**TD3 does not apply.** It is a continuous-action algorithm and this task has three discrete actions.
PPO does fit and pairs naturally with a fast vectorised env; the discrete off-policy actor-critic
equivalents would be SAC-discrete or Munchausen-DQN.

## Scratch artifacts

The prototype and its parity harness are throwaway and live outside the repo, under this session's
scratchpad `vecenv/`: `vec_snake.py` (env), `dump_ref.py` (stage-1 reference dump, snek env),
`test_parity.py` (stage-2 comparison, torch env), `ref_cost.py` (isolated scalar costs), `bench.py`.
Parity has to run in two stages because the live game needs the `snek` env and torch is only in the
`tictactoe` env.
