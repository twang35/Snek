# The environment

Identical to snek2's, deliberately — same board, same actions, same rewards, same 30-value
observation, same era marker `b09c616`. That is what lets a snek2 champion's weights convert straight
across and gives the port a real correctness gate before any training code exists
([`../plans/archive/pytorch-port.md`](../plans/archive/pytorch-port.md) §10).

**Changing anything here changes the MDP.** Bump `env.constants.OBS_ERA` whenever the observation's
*meaning* changes, even at constant length — see [`invariants.md`](invariants.md) invariant 3 for what
it costs not to.

## The board

| | |
|---|---|
| playable grid | **10 x 10** (`GRID_LENGTH = 9`, inclusive, so indices 0-9), with a one-cell wall ring |
| starting snake | 4 segments at `(5, 3)` |
| perfect score | **100** cells filled |
| max score reported | **95** = `PERFECT_SCORE − START_SEGMENTS − 1`, the number of meals in a perfect game |
| actions | **3**, relative turns: `left`, `right`, `forward` — never compass directions |
| headings | `left`, `right`, `up`, `down` |
| starve budget | `10 × snake_len`, clamped to **[100, 500]** steps since the last meal |

A score of 95 *is* a filled board. That has been misread as a near-miss.

## Rewards

| term | value | when |
|---|---|---|
| `FOOD_REWARD` | **+1.0** | a meal |
| `DEATH_REWARD` | **−5.0** | wall or body |
| `STARVE_REWARD` | **−0.5** | the starve budget runs out |
| `PERFECT_GAME_REWARD` | **+100** | the board is filled |
| `FOOD_DISTANCE_REWARD` | 0.001, off in every recent arm | subtracted on an ordinary move that *increases* Manhattan distance to the food |
| `CHASE_SAFE_SHAPING` | `c`, potential-based, gated on snake length | head, food and tail in one region |
| `FREE_SPACE_SHAPING` | `c`, potential-based, gated on snake length | `1 / open-region-count` |

**A reward is a sum of terms**, which is why nothing may identify a perfect game by comparing the
final reward with `PERFECT_GAME_REWARD`. Both shaping terms are potential-based, so they pay `−c·Φ(s)`
at a terminal step — which is exactly how snek2 silenced every perfect-game counter for 300k steps.

**`PERFECT_GAME_REWARD` and `DISCOUNT` are coupled** and cannot be tuned independently:
`W > 1/(1 − γ^k)`. See [`invariants.md`](invariants.md) invariant 6.

## The observation — 30 values

Anything "per action" is ordered by `ACTIONS` — **left, right, forward, as relative turns**, not
compass directions.

| idx | n | what |
|---:|---:|---|
| 0-5 | 6 | food: `[is closer, 1/(distance+1)]` per action |
| 6-8 | 3 | is the move safe (not body or wall) |
| 9-14 | 6 | `[can still reach tail, lg(open regions) scaled to [0,1]]` per action |
| 15-17 | 3 | is it safe to chase the food (head, food and tail in one region) |
| 18-20 | 3 | does the move win the game |
| 21 | 1 | starve budget left, lg-compressed to [0,1] |
| 22 | 1 | fraction of the board the snake fills |
| 23-25 | 3 | is the post-move head hugging a wall or body on its left or right |
| 26-28 | 3 | is the move **NOT** a tail-chase (0 = it lands on the cell the tail is vacating) |
| 29 | 1 | room around the food: 1 roomy or no food, 0.5 a two-cell pocket, 0 sealed in |

**1 means good or safe throughout.** New blocks go on the end, never in the middle — the order is
chronological rather than logical and that is deliberate.

## What the record checkpoints actually read — measured 2026-09-07

Three HOF checkpoints (`b10ck` @30523392, `b17cl` @11386880, `b9ch` @47251456), each played greedy
for 2,000 episodes (seed 1) with one feature block replaced by its mean over real play — so the
number is *information lost*, not distribution shock. Zeroing was measured too and is worse for
every near-constant input, which is the shock. Noise at this depth is about ±0.35 pp. Script:
`tools/feature_audit.py` (`analyse`, then `ablate` and `ablate_phase`).

| block | perfect % with the block blanked (baseline 99.75 / 99.4 / 98.95) | verdict |
|---|---|---|
| 0,2,4 food is-closer | 0 / 0 / 0 | essential |
| 9,11,13 tail reachable | 0 / 0.2 / 0 | essential |
| 15-17 chase-safe | 0 / 2 / 13 | essential |
| 6-8 move is safe | 98.8 / 83.5 / 57 | large |
| 22 board fill | **98.4** / 1.6 / 41.5 | large for two arms, **near-irrelevant to the record holder** |
| 21 starve budget | **0** / 98.6 / 98.8 | **the record holder's clock**; minor for the others |
| 23-25 hugging | 95 / 71 / 59 | real, and larger than hypothesised |
| 26-28 not a tail-chase | 92.5 / 75.5 / 86 | real, though nonzero in only 0.6% of states |
| 1,3,5 food 1/(dist+1) | 99.45 / 95.9 / 95.7 | minor; `b10ck` does not need it |
| 10,12,14 lg(open regions) | 99.2 / 97.9 / 94.3 | minor, and the only expensive block |
| **18-20 this move wins** | **99.7 / 99.4 / 99.35** | **nothing**, all three arms |
| **29 room around food** | **99.5 / 99.4 / 99.3** | **nothing**, all three arms |

Blanking a block only at fill ≥ 0.8 shows where the value sits: late, the blocks that still cost are
tail-reach (62-74), chase-safe (71-87), and for the two fill-readers, fill (93 / 78); the region
counts cost 0.5 pp late for `b10ck` and 1.75 for `b9ch`; the win flags and index 29 cost nothing in
either phase. Endgame decisions are also the most robust: at fill ≥ 0.95, shuffling any single input
flips under 1% of `argmax` choices, against 5-19% early.

What follows for the individual indices:

- **Indices 6-8 are the only place legality is stated.** A *fatal* move reads 1 at indices 26-28,
  because that flag only asks "is this the tail's cell".
- **Index 22 (board fill) is rank 1 of 30 by gradient saliency in every arm measured, snek2 and
  snek3 — and saliency is the wrong instrument here.** `b17cl` collapses without it (1.6%) and `b9ch`
  halves, but the record holder `b10ck` (γ 1.00) loses 1.35 pp and reads the **starve budget**
  instead: blank index 21 and it scores 0.0. Its episodes are 3x longer than the other two (3,200
  steps against ~1,150) — it stalls until the budget forces a meal. Undiscounted training found a
  different clock.
- **Indices 18-20 are nonzero in 0.00-0.06% of states and carry nothing.** Blanking them changes no
  arm's rate; forcing the flag to 1 at length 99 moves that action's logit by +1 to +10 against a
  decision margin of 18-37, so it could not change a decision even where it fires. They cost 6 µs of
  a ~10 ms observation build, so the waste is three untrained inputs, not compute. Never credit an
  endgame result to them.
- **Index 29 sits at 1 in 99.75% of states and carries nothing either.** Mean-substitution is free
  for all three; *zeroing* it drops `b10ck` to 53.7% — the net uses the constant as a bias, which is
  the hazard a near-constant input is.
- **Indices 10, 12 and 14 are the only consumers of region *enumeration***, which is 33% of the
  connectivity cost, and connectivity (9-17) is ~93% of the observation build; every other block is
  under 3% (hugging ~3%, index 29 <1%, the rest negligible). They are worth 0.5 / 1.5 / 4.6 pp to
  nets trained with them, so dropping them is a retrain-and-measure question with a known upper
  bound on the loss and ~1.5x on the observation as the prize.

Indices 23-25 and 26-28 were carried across as unvalidated hypotheses; the table validates both.
Index 29 was carried across on the same footing and no measurement has validated it.

## Two implementations, and one is the reference

| | `env/` | `vectorized/` |
|---|---|---|
| what | one game, pygame, drawable | N games in lockstep, pure numpy, no pygame |
| speed | ~12,800 env-steps/s in one process | **~196,000 env-steps/s at 1024 lanes** |
| role | the parity **reference**, plus `watch.py` and `record_gif.py` | every eval, and training collection |

**If the two disagree, `vectorized/` is wrong.** `tests/test_vectorized_parity.py` asserts both
elementwise — all 30 observation indices plus the step mechanics — over ≥18,000 states drawn from
real play, and a set of hand-made mutants must all fail the harness so the comparison has teeth.

`VecSnake` keeps the body as a circular buffer of flat padded cell indices on a 12x12 grid, so the
tail is an O(1) lookup rather than a walk, and runs the connectivity block as a **bitboard dilation**
on packed uint64 words — one round is ~17 numpy ops on three words per board rather than a pass over
144 cells. **Two compaction tricks dominate its measured speed** and a naive reimplementation loses
them: the flood drops a row from the working set the moment it stops growing, and the region
enumeration compacts the same way. Without those, every board in the batch pays the batch's *maximum*
dilation count — 125 rounds at n=1024 where a typical board needs ~15 — and the vectorised
observation costs the same as the scalar one.

**`n=1` regresses ~6x** against the scalar env, so the scalar path is not vestigial: anything
single-game wants it.

## Why the env stays numpy

The policy is **8.2%** of an env step at width 1024; the numpy observation build is 4,296 us of
5,050. So **1.09x is the ceiling for any accelerator, however fast**, and the bottleneck is a
bitboard flood fill rather than a tensor program.

Separately, snek2 measured `tensorflow-metal` as 2.4x *slower* on the policy call at that width — but
the disqualifying part was correctness: four champions measuring 97-98% measured **0.0%** on MPS with
no error raised and a *faster* wall clock. Run a device-parity check before trusting any new device,
accelerator build or framework version. **The failure mode is a silent zero, which reads as a bad
arm.**
