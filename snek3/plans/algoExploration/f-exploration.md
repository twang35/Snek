# Group F: exploration -- NGU, Agent57

**Status: planned 2026-09-16, nothing built.** Group F of [`algorithm-series.md`](algorithm-series.md);
conventions in [`README.md`](README.md). Phase 7 of the running order; waits for E2, which both rows
are built on.

The question this group asks is one the game does not: whether exploration machinery matters when
there is no hard-exploration problem. The food is always reachable, the tour proves it, and a random
policy finds reward on its first step. The rows are in the series to *confirm* the null and to
measure one thing the null does not settle -- whether an intrinsic term **hurts** a game where the
shortest route is already the right one, and whether Agent57's meta-controller learns to switch it
off. The expected result is written down first so a null is a finding and not a disappointment.

## 1. What the group shares

Both rows are R2D2 (E2) with an intrinsic reward and a policy conditioned on how much of it to want.

| shared piece | decision |
|---|---|
| package | `algos/ngu/`, one `algo.py` with `NAME`s `ngu` and `agent57`; the second adds the split heads and the meta-controller (§2) |
| base agent | `algos/r2d2/`, imported: the LSTM net, the sequence replay, the burn-in agent. The intrinsic modules feed it an augmented reward and an augmented observation |
| the intrinsic reward | NGU's: an **episodic** term from a k-nearest-neighbour count in a learned embedding, reset every episode, times a **lifelong** term from random network distillation, clipped to [1, L]. Both new, in `algos/ngu/intrinsic.py` |
| the embedding | NGU's inverse-dynamics embedding: f(obs) trained to predict the action between consecutive observations. On a 26-value vector the embedding is an MLP to 32 dims |
| the conditioning | a family of N policies indexed by β_i (intrinsic weight) and γ_i (discount), the index one-hot appended to the observation (UVFA). `SNEK_NGU_MIXTURES` **32, the paper's** -- the lanes are vectorised, so 32 lanes is one lane per mixture and costs nothing the box lacks; β_max `SNEK_NGU_BETA_MAX` (0.3) with the paper's spacing β_0 = 0, β_{N−1} = β_max, β_i = β_max · σ(10 (2i − (N − 2)) / (N − 2)); γ from `SNEK_DISCOUNT` (γ_0, the exploitative end, **0.997** for NGU and **0.9999** for Agent57, the papers') down to `SNEK_NGU_GAMMA_MIN` (0.99), spaced as each paper's equation (`README.md`, "Translating") |
| the sidecar | `recurrent` from E and `conditioning`: `{"mixtures": 32, "obs_extra": 32}`; in the signature |
| restore | greedy under the **exploitative** index (β = 0, the highest γ), which is what the eval measures; a `--policy-variant mixture:<i>` in the style of `a-return-tail.md` §5 measures any other |
| the step | E2's |

**The episodic memory is per lane.** The k-NN store is one array per lane, reset with the lane, sized
by the episode bound (the starve budget × the maximum score is the most steps an episode can take,
and that is the store's capacity, so nothing reallocates mid-episode). This is the same per-lane state
ownership E1's seam decision gives the recurrent policy, extended by one array.

## 2. The rows

### F1 -- NGU (Badia et al. 2020, "Never Give Up")

R2D2 plus the intrinsic reward above, with the extrinsic and intrinsic rewards summed as
r = r_e + β_i r_i and one Q-network over the sum.

| module | contents |
|---|---|
| `algos/ngu/intrinsic.py` | the RND predictor and target (`SNEK_NGU_RND_HIDDEN` 128, output 128 as the paper) with a running normalisation of the error, clipped to [1, L] with L = 5; the inverse-dynamics embedding (32 dims, classifier hidden 128, L2 1e-5) at lr 5e-4; the episodic k-NN (k = 10, kernel ε 1e-4 per Table 6 -- the paper's text says 1e-3, and the knob carries whichever the smoke shows is not degenerate on a 32-dim embedding -- cluster distance 8e-3, pseudo-count constant 1e-3, maximum similarity 8, memory capacity 30,000, all exposed as knobs prefixed `SNEK_NGU_`) |
| `algos/ngu/algo.py` (`ngu`) | wraps `R2d2Algo`: `advance()` computes r_i per lane per step from the embedding of the new observation, augments the reward before it enters the sequence buffer, appends the mixture one-hot to the observation, trains RND and the embedding on the same batch; `fields()` reports the mean r_i and the mixture in use per lane |
| the actors | each lane is assigned a mixture index at episode start, uniformly (NGU) -- and stays on it for the episode |

Tests: the episodic reward on a repeated observation falls with each repeat and resets on a new
episode; RND error is ~1 in normalised units on novel inputs and falls on trained ones; the inverse
dynamics classifier reaches a non-chance accuracy on a scripted trajectory; with β = 0 the augmented
reward equals the extrinsic one to the bit. Mutants: the episodic store not cleared on reset, the
lifelong term unclipped, the one-hot appended for the wrong index.

### F2 -- Agent57 (Badia et al. 2020)

NGU with two changes: the Q-function is **split** into an extrinsic and an intrinsic head trained on
their own rewards and combined as Q_e + β Q_i (so the intrinsic scale cannot destabilise the extrinsic
estimate), and a **meta-controller** -- a sliding-window UCB bandit per actor -- chooses the mixture
index at each episode start from the extrinsic return the index earned, instead of drawing it
uniformly.

| module | contents |
|---|---|
| `algos/ngu/net.py::SplitHead` | **two networks of identical architecture**, as the paper (§3.1), not two heads on one LSTM: Q_e(θ_e) and Q_i(θ_i), each trained with its own transformed Retrace target on its own reward stream under the shared target policy argmax(Q_e + β_j Q_i); combined only for acting. `SNEK_A57_SPLIT=heads` shares the trunk and LSTM for the cheaper local variant |
| `algos/ngu/meta.py` | the bandit: per lane, a sliding window of `SNEK_A57_WINDOW` episodes of (index, extrinsic return) -- **160**, the actors' value in the paper's §4 (its Table 3 prints 90; the text is followed) -- UCB with `SNEK_A57_UCB_BETA` (1.0) and ε `SNEK_A57_EPSILON` (0.5), every arm pulled once first; the chosen index is what the bandit selected, logged per episode |
| `algos/ngu/algo.py` (`agent57`) | `ngu` with the split head on and the meta-controller replacing the uniform draw; `fields()` adds the fraction of episodes the bandit spent at β = 0 -- **the number this row is for** |

Tests: the combined Q at β = 0 equals the extrinsic head; each head's loss sees only its own reward
(a fixture zeroes one stream and checks the other head's gradient is unchanged); the bandit converges
to the arm with the higher mean return on a two-arm synthetic problem within the window. Mutants: the
intrinsic head trained on the summed reward, the bandit's exploration bonus dropped, the window not
sliding.

## 2b. The papers' settings, and how each lands here

| setting | NGU (Table 6) | Agent57 (Table 3, App. E) | here |
|---|---|---|---|
| base | R2D2: LSTM 512, dueling, Adam 1e-4 ε 1e-4, batch 64, trace 80 / replay period 40, target 1,500 | the same, trace **160** / period 80 | E2's paper cell; F2 at 160 / 80 |
| return estimator | transformed Retrace, λ 0.95 | the same | Retrace λ 0.95 replaces E2's n-step target in this group (`SNEK_NGU_RETRACE_LAMBDA`); E2's n-step is the ablation |
| mixtures | 32; β_max 0.3; γ 0.997 → 0.99 | 32; γ 0.9999 → 0.99 with the three-piece spacing of App. G.1 | **the paper's, per row**: F1 γ 0.997 → 0.99 with NGU's spacing; F2 γ 0.9999 → 0.99 with the three-piece spacing. The exploitative index acts at the top of the ladder, so F1 is measured at 0.997 and F2 at 0.9999; a 2,300-move game at 0.9999 is effectively undiscounted, which is what the paper's exploitative arm is too |
| priority | exponent 0.9, η 1.0 (max only), IS 0 | exponent 0.9, η 0.9, IS 0 | per row |
| replay | 5e6 observations, 6,250 sequences before learning | the same | 1e5 sequences as E2; 6,250 before learning |
| intrinsic module | RND 128, embedding 32, classifier 128, lr 5e-4, L 5, k 10, memory 30,000 | the same | the same, over an MLP in place of the conv stack |
| bandit | -- | window 160, β 1, ε 0.5, per actor | per lane |
| exploration | ε 0.4^(1 + 8 i / 255) over 256 actors; eval ε 0.01 | the same | the Ape-X ladder over 32 lanes with α 8 (E2's `apex` schedule); eval greedy |
| frames | 35B | ~90B | 50M moves a cell -- the paper's budgets are 3 orders of magnitude beyond the box, and the rows are here to confirm a null |

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| F1 | 4 seeds of `ngu` on §2b, exploitative index measured | E2's paper config | E2 paper | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step; **the mean r_i trace**, which should fall to its clip floor early if the game has nothing to explore |
| F2 | 4 seeds of `agent57` on §2b (two networks, trace 160) + 4 seeds `SNEK_A57_SPLIT=heads` | F1's | F1, E2 paper | as F1; **the bandit's β = 0 fraction over training** |

**Registered prediction (the agent's, 2026-09-16).** F1 trails E2 on onset and is level or below on
density: the intrinsic term pays for visiting board states the shortest route avoids, and a lane on a
high-β mixture wastes its episode. F2 recovers E2's numbers and its bandit spends >80% of late episodes
at β = 0, which is the meta-controller learning that the game has no exploration problem. A result that
would matter: F1 *above* E2, which would say the endgame's rare fatal states are undersampled by the
greedy policy and novelty reaches them.

## 4. Gates

1. E2 closed. Both rows import it and there is nothing to smoke before it exists.
2. Smoke for both names; the exploitative checkpoint restores and watches.
3. The mutation specs kill every mutant.
4. Tuning budget: one laptop wave on β_max and the kernel ε. The rows are not tuned further; a
   null here is the expected finding.

## 5. What would change the plan

- **F1 beats E2.** The one surprising outcome; it would send the series back to Group A's diagnosis,
  because it would mean the rare fatal states are a *coverage* problem the replay never sees, and
  that is a different lever (targeted replay, or G's search) from a better value estimate.
- **F2's bandit does not settle at β = 0.** Either the window is too short for a 2,000-move game's
  return signal or intrinsic reward is genuinely earning extrinsic return; the β trace and the r_i
  trace together say which, and only the second reopens the plan.
- **Both null, as predicted.** The finding is written in one line and the group closes.
