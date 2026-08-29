# The environment

Identical to snek2's, deliberately — same board, same actions, same rewards, same 30-value
observation, same era marker `b09c616`. That is what lets a snek2 champion's weights convert straight
across and gives the port a real correctness gate before any training code exists
([`../plans/pytorch-port.md`](../plans/pytorch-port.md) §10).

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

Five things about specific indices are worth knowing before touching any of them:

- **Indices 6-8 are the only place legality is stated.** A *fatal* move reads 1 at indices 26-28,
  because that flag only asks "is this the tail's cell".
- **Index 22 (board fill) is rank 1 of 30 by saliency in every snek2 arm measured**, and it is how
  the winning policies actually learn to finish.
- **Indices 18-20 are nonzero in 0.000-0.025% of states** and are not meaningfully trained. Never
  credit an endgame result to them.
- **Index 29 sits at 1 in ~99.95% of states**, so it is nearly a constant — the same hazard.
- **Indices 10, 12 and 14 are the only consumers of region *enumeration***, which is 33% of the
  connectivity cost. A two-flood shortcut is exactly equal to the reference for the other
  connectivity indices over all 18,053 parity states, so dropping 10/12/14 is a cheap, well-posed
  ablation worth ~1.5x on the observation. Batch 45 reached 99% with them in, so it is a cost
  question, not a correctness one.

Indices 23-25, 26-28 and 29 are **unvalidated** — hypotheses about what a feature enables, carried
across because the record-holding policies were trained with them, not because any of them has a
measured effect.

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
