# Group E: exploration -- NGU, Agent57

**Status: planned 2026-09-16, nothing built.** Group E of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 6 of the running order; waits for D2, which both rows
are built on.

The question this group asks is one the game does not: whether exploration machinery matters when
there is no hard-exploration problem. The food is always reachable, the tour proves it, and a random
policy finds reward on its first step. The rows are in the series to *confirm* the null and to
measure one thing the null does not settle -- whether an intrinsic term **hurts** a game where the
shortest route is already the right one, and whether Agent57's meta-controller learns to switch it
off. The expected result is written down first so a null is a finding and not a disappointment.

## 1. What the group shares

Both rows are R2D2 (D2) with an intrinsic reward and a policy conditioned on how much of it to want.

| shared piece | decision |
|---|---|
| package | `algos/ngu/`, one `algo.py` with `NAME`s `ngu` and `agent57`; the second adds the split heads and the meta-controller (§2) |
| base agent | `algos/r2d2/`, imported: the LSTM net, the sequence replay, the burn-in agent. The intrinsic modules feed it an augmented reward and an augmented observation |
| the intrinsic reward | NGU's: an **episodic** term from a k-nearest-neighbour count in a learned embedding, reset every episode, times a **lifelong** term from random network distillation, clipped to [1, L]. Both new, in `algos/ngu/intrinsic.py` |
| the embedding | NGU's inverse-dynamics embedding: f(obs) trained to predict the action between consecutive observations. On a 26-value vector the embedding is an MLP to 32 dims |
| the conditioning | a family of N policies indexed by β_i (intrinsic weight) and γ_i (discount), the index one-hot appended to the observation (UVFA). `SNEK_NGU_MIXTURES` (8 in the plan; 32 in the paper, cut for the actor count), β_max `SNEK_NGU_BETA_MAX` (0.3), γ from `SNEK_DISCOUNT` down to `SNEK_NGU_GAMMA_MIN` (0.99 → 0.97; the game's horizon is ~2,300 moves so the paper's 0.997 range does not transfer) |
| the sidecar | `recurrent` from D and `conditioning`: `{"mixtures": 8, "obs_extra": 8}`; in the signature |
| restore | greedy under the **exploitative** index (β = 0, the highest γ), which is what the eval measures; a `--policy-variant mixture:<i>` in the style of `a-return-tail.md` §5 measures any other |
| the step | D2's |

**The episodic memory is per lane.** The k-NN store is one array per lane, reset with the lane, sized
by the episode bound (the starve budget × the maximum score is the most steps an episode can take,
and that is the store's capacity, so nothing reallocates mid-episode). This is the same per-lane state
ownership D1's seam decision gives the recurrent policy, extended by one array.

## 2. The rows

### E1 -- NGU (Badia et al. 2020, "Never Give Up")

R2D2 plus the intrinsic reward above, with the extrinsic and intrinsic rewards summed as
r = r_e + β_i r_i and one Q-network over the sum.

| module | contents |
|---|---|
| `algos/ngu/intrinsic.py` | the RND predictor and target (`SNEK_NGU_RND_HIDDEN` 128) with a running normalisation of the error; the inverse-dynamics embedding and its classifier; the episodic k-NN (k = 10, kernel ε = 1e-3, cluster distance 8e-3, pseudo-count constant 1e-3, all as the paper, exposed as knobs prefixed `SNEK_NGU_`) |
| `algos/ngu/algo.py` (`ngu`) | wraps `R2d2Algo`: `advance()` computes r_i per lane per step from the embedding of the new observation, augments the reward before it enters the sequence buffer, appends the mixture one-hot to the observation, trains RND and the embedding on the same batch; `fields()` reports the mean r_i and the mixture in use per lane |
| the actors | each lane is assigned a mixture index at episode start, uniformly (NGU) -- and stays on it for the episode |

Tests: the episodic reward on a repeated observation falls with each repeat and resets on a new
episode; RND error is ~1 in normalised units on novel inputs and falls on trained ones; the inverse
dynamics classifier reaches a non-chance accuracy on a scripted trajectory; with β = 0 the augmented
reward equals the extrinsic one to the bit. Mutants: the episodic store not cleared on reset, the
lifelong term unclipped, the one-hot appended for the wrong index.

### E2 -- Agent57 (Badia et al. 2020)

NGU with two changes: the Q-function is **split** into an extrinsic and an intrinsic head trained on
their own rewards and combined as Q_e + β Q_i (so the intrinsic scale cannot destabilise the extrinsic
estimate), and a **meta-controller** -- a sliding-window UCB bandit per actor -- chooses the mixture
index at each episode start from the extrinsic return the index earned, instead of drawing it
uniformly.

| module | contents |
|---|---|
| `algos/ngu/net.py::SplitHead` | two heads on the LSTM output; each trained with its own target on its own reward stream (both through the rescaling); combined only for acting |
| `algos/ngu/meta.py` | the bandit: per lane, a window of `SNEK_A57_WINDOW` (90) episodes of (index, extrinsic return), UCB with `SNEK_A57_UCB_BETA` (1.0) and ε `SNEK_A57_EPSILON` (0.5); the chosen index is what the bandit selected, logged per episode |
| `algos/ngu/algo.py` (`agent57`) | `ngu` with the split head on and the meta-controller replacing the uniform draw; `fields()` adds the fraction of episodes the bandit spent at β = 0 -- **the number this row is for** |

Tests: the combined Q at β = 0 equals the extrinsic head; each head's loss sees only its own reward
(a fixture zeroes one stream and checks the other head's gradient is unchanged); the bandit converges
to the arm with the higher mean return on a two-arm synthetic problem within the window. Mutants: the
intrinsic head trained on the summed reward, the bandit's exploration bonus dropped, the window not
sliding.

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| E1 | 4 seeds of `ngu`, exploitative index measured | D2's config | D2 | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step; **the mean r_i trace**, which should fall to its clip floor early if the game has nothing to explore |
| E2 | 4 seeds of `agent57` | E1's | E1, D2 | as E1; **the bandit's β = 0 fraction over training** |

**Registered prediction (the agent's, 2026-09-16).** E1 trails D2 on onset and is level or below on
density: the intrinsic term pays for visiting board states the shortest route avoids, and a lane on a
high-β mixture wastes its episode. E2 recovers D2's numbers and its bandit spends >80% of late episodes
at β = 0, which is the meta-controller learning that the game has no exploration problem. A result that
would matter: E1 *above* D2, which would say the endgame's rare fatal states are undersampled by the
greedy policy and novelty reaches them.

## 4. Gates

1. D2 closed. Both rows import it and there is nothing to smoke before it exists.
2. Smoke for both names; the exploitative checkpoint restores and watches.
3. The mutation specs kill every mutant.
4. Tuning budget: one laptop wave on β_max and the mixture count. The rows are not tuned further; a
   null here is the expected finding.

## 5. What would change the plan

- **E1 beats D2.** The one surprising outcome; it would send the series back to Group A's diagnosis,
  because it would mean the rare fatal states are a *coverage* problem the replay never sees, and
  that is a different lever (targeted replay, or G's search) from a better value estimate.
- **E2's bandit does not settle at β = 0.** Either the window is too short for a 2,000-move game's
  return signal or intrinsic reward is genuinely earning extrinsic return; the β trace and the r_i
  trace together say which, and only the second reopens the plan.
- **Both null, as predicted.** The finding is written in one line and the group closes.
