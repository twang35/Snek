# Group F: exploration -- NGU, Agent57

**Status: planned 2026-09-16, reviewed against the papers 2026-10-12 (two passes: the agent's, then a second reviewer's, whose accepted points are marked **review 2**), nothing built.** Group F of
[`algorithm-series.md`](algorithm-series.md); conventions in [`README.md`](README.md). Phase 7 of the running
order; E2 closed 2026-10-09 (b52, `docs/runs.md`), so the gate is open.

The question this group asks is one the game does not: whether exploration machinery matters when
there is no hard-exploration problem. The food is always reachable, the tour proves it, and a random
policy finds reward on its first step. The rows are in the series to *confirm* the null and to
measure one thing the null does not settle -- whether an intrinsic term **hurts** a game where the
shortest route is already the right one, and whether Agent57's meta-controller learns to switch it
off. The expected result is written down first so a null is a finding and not a disappointment.

## 0. Paper-cell fidelity review (2026-09-23, completed 2026-10-12)

The paper cell follows the paper **wherever the game allows** (`README.md`, the paper-fidelity row, widened
2026-09-23); the local cell carries every codebase choice. The 2026-10-12 pass read both papers' text and
hyperparameter tables (NGU Table 6 and Appendices A, B, F; Agent57 Table 3, §3-4, Appendices D, E, G) and
settled every *to confirm*; the rows it added are marked **new**. Each item is fixed **before the row's paper
cell is queued**. Where the papers contradict themselves the plan says which value it takes and why.

| item | paper | here | fix | status |
|---|---|---|---|---|
| NGU optimiser and target | **confirmed**: Adam 1e-4, ε 1e-4, β (0.9, 0.999), clip 40; target copy every 1,500 updates. Agent57 the same | E2's 1e-3 and 2,500 | `SNEK_ADAM_EPSILON=1e-4`, `SNEK_TARGET_UPDATE_PERIOD=1500` in both rows' specs; no new knob | **decided** |
| replay size | 5e6, **units unstated in both papers** (the Agent57 table reads "replay capacity 5e6"); read as observations, ~1.25e5 sequences at NGU's geometry, ~6e4 at Agent57's | E2 ran 30,000 windows, the most the 15 GB box holds four of (`e-r2d2.md` §2b) | **the box's**: 30,000 windows, a stated departure in the manifest note, the same in every F cell as it was in every E2 cell | **decided** |
| prefill | 6,250 sequences before learning (both tables), the actors on their own ε ladder | `SNEK_INITIAL_COLLECT_STEPS` counts rows; E2's prefill already runs the ladder | **6,250 windows**: a new `SNEK_NGU_MIN_WINDOWS` the prefill waits for beside the row count (E2's rule, "rows banked **and** a batch of windows", gains the paper's number). The ladder prefill is inherited from E2 | **decided** |
| episodic kernel ε | **confirmed contradiction**: Table 6 says 1e-4, §2's text says "fixed to 1e-3 in all our experiments"; Agent57's table repeats 1e-4 | tuning inside the paper cell | **1e-4** in the paper cell (two tables against one sentence); the tuning wave (§4) tries 1e-3 and the local cell carries the better | **decided** |
| NGU inputs | one-hot of the mixture index, previous action, previous **extrinsic** reward and previous **intrinsic** reward are concatenated to the encoder's output and enter the LSTM (NGU §2 and Fig. 1; Agent57 App. E lists the same five inputs plus the state) | "appended to the observation" | **fed to the core as the paper does**: `R2d2Net._cell_input` already concatenates the previous action one-hot and reward to the trunk's features, so the mixture one-hot and the previous r_i are two more columns there. The observation stays 26 values and `obs_len` is untouched; the sidecar's `conditioning` block carries `mixtures` | **decided** |
| importance weights | IS exponent **0** (Agent57 Table 3; NGU's table does not list it and Agent57 reproduces NGU's agent) | E2's 0.6, max-normalised | `SNEK_R2D2_IS_BETA=0` in the spec: every weight is 1 and the normalisation is moot | **decided** |
| **new** mixture assignment (F1) | NGU App. A: "actor j is assigned β_h with **h = j mod (N−1)**" -- fixed for the run, not drawn per episode; the top mixture β_{N−1} is never acted on | §2 drew an index uniformly at each episode start | **the paper's**: lane j holds mixture j mod (N−1) for the run (so lanes 0 and 31 both hold the exploitative index 0). The uniform draw becomes the local-cell variant `SNEK_NGU_ASSIGN=episode` | **decided** |
| **new** Retrace needs the behaviour probability | Retrace's trace c_s = λ min(1, π(a_s\|x_s)/μ(a_s\|x_s)) needs μ, the actor's ε-greedy probability of the action it took, stored per step; the target policy π is greedy on Q (Q_e + β_j Q_i for Agent57), **ε-soft at the table's "target ε 0.01"** (NGU Table 6) | `SequenceReplay` stores no behaviour probability | one more per-row column, `mu`, written by the collector from the lane's ε and whether the action was the greedy one; `SNEK_NGU_TARGET_EPSILON` 0.01 for π. The paper's h-transformed form: h(Σ ... h⁻¹(Q)) as E2's n-step target already does | **decided** |
| **new** the embedding the actors use | NGU Table 6: "embeddings target update: once per episode" -- an actor's copy of f is fixed within an episode, so the keys in its episodic memory are all under one embedding | one process, the collector would read the live f every step and the store's keys would drift under it | **a snapshot of f per lane, refreshed at the lane's episode start**: 32 copies of a 26→128→32 MLP is nothing, and the stored keys stay consistent as the paper's are | **decided** |
| **new** the auxiliary losses' frames | NGU App. A: "use last 5 frames of the sampled sequences to train the action prediction network" and "also use last 5 frames of the sampled sequences to train the predictor of RND" | "on the same batch", unqualified; the first pass put RND on every loss step, which trains it on 16x (F1) to 32x (F2) the observations and changes how fast lifelong novelty vanishes | **the last 5 valid steps of each window for both losses**, `SNEK_NGU_EMBED_STEPS` 5 (a window shorter than 5 steps uses what it has; a terminal row's successor is not read) | **decided** (review 2) |
| **new** Agent57's evaluator | App. E: the evaluator runs **its own bandit** (window 3,600, ε_UCB 0.01), alternating five training episodes with five evaluation episodes at the bandit's greedy arm (γ_j, β_j); it does **not** evaluate β = 0 | restore measured the exploitative index | **two numbers, both reported**: the protocol's eval stays the exploitative index (what every row in the series is measured on, and what NGU's paper evaluates), and the trainer writes the bandit's late greedy arm into the row (`on_eval`, the per-lane mode over the window); `--policy-variant mixture:<i>` measures that arm in a pass. A stated translation: the series has no stateful evaluator | **decided** |
| **new** the bandit's reward | "the **undiscounted** extrinsic episode returns" (§3.2) | "the extrinsic return" | undiscounted, the sum of r_e over the episode | **decided** |
| **new** Agent57's discount ladder is misprinted | App. G.1's printed equation (§2b) gives γ_31 = 0.99997 rather than the γ_2 = 0.99 its text and Fig. 11 intend, and its sigmoid piece *rises* from γ_1 at j = 1 to γ_0 at j = 6 before dropping to γ_1 at j = 7, where the figure falls monotonically | the first pass copied the equation | **read as an erratum**: the intended ladder is γ_0 at j = 0, falling through γ_0 − (γ_0 − γ_1) σ(10 (2j − 6) / 6) over j = 1..6 to γ_1 at j = 7, then 1 − exp(((N−9−(j−8)) log(1−γ_1) + (j−8) log(1−γ_2)) / (N−9)) for j ≥ 8, which is γ_1 at j = 8 and γ_2 at j = N−1 -- the NGU interpolation on the lower piece. A fixture pins the endpoints (0.9999, 0.997 at 7 and 8, 0.99 at 31) and strict monotonicity | **decided** (review 2) |
| **new** the checkpoint and the restore contract | the exploitative policy still reads r^i_{t−1}, so even at β = 0 acting needs the embedding, the RND pair and the error-normalisation statistics, and a per-lane episodic store | `tools/checkpoints.py` saves `algo.net.state_dict()` alone; R2D2's net would leave the intrinsic modules behind and the restored policy would see a zero where the trainer saw r_i | **the net owns the intrinsic modules**: one `NguNet` module holding Q (or Q_e and Q_i), the embedding, the RND predictor and target and the normalisation buffers, so the checkpoint is complete and `tools/restore.py` builds the stateful policy from it; that policy computes r_i per lane with its own episodic store, zeroed with the lane (the `begin` seam `algos/stateful.py` already has). The sidecar carries `conditioning: {mixtures, beta_max, gamma_top, gamma_min, split}` so the β/γ ladder and the two-network layout are rebuilt from the file. `resume.pt` carries the auxiliary optimisers, the normalisation statistics and the bandit windows | **decided** (review 2) |
| **new** matched controls | NGU's own baseline is "R2D2 (Retrace)"; Agent57's ablations hold the architecture and vary the bandit | §3 read F1 against b52 and F2 against F1, pairs that differ in the estimator, target period, Adam ε, IS exponent, prefill, ladder and (F2) the networks, the γ family and the trace, so a gap could not be attributed | **a control cell in each wave** (§3): F1 runs beside `ngu` at β_max = 0 (every β_i 0, everything else identical -- the paper's R2D2 (Retrace)); F2 beside `agent57` with the bandit replaced by F1's fixed assignment. b52 stays the series' cross-row reading; the controls carry the attribution | **decided** (review 2) |
| **new** budget | -- | §2b carried "50M moves a cell" | **a benchmark gate** (§3, §4): F2's paper cell is two LSTM networks over a 200-slot window, ~3-4x b52's 517 ms an update, so ~24 h of a 4-arm wave buys ~35-45k updates against b52's 130k. The number is set from the benchmark and the shortfall stated, as E2's was | **open until the benchmark** |

Three paper-internal contradictions the plan takes a side on: kernel ε (above); NGU's priority mixing η,
1.0 in App. A against 0.9 in Table 6 -- **App. A's 1.0** (it is the text's description of the method, and
Agent57's table gives 0.9 for its own agent, so the two rows differ as the papers do); and NGU's value-rescaling ε,
1e-3 in Table 6 against 1e-2 in App. E -- **1e-3** (R2D2's, and Agent57's table). Agent57's bandit window is 160
in the text (§4) and 90 in Table 3; **160**, the text, as before.

## 1. What the group shares

Both rows are R2D2 (E2) with an intrinsic reward and a policy conditioned on how much of it to want.

| shared piece | decision |
|---|---|
| package | `algos/ngu/`, one `algo.py` with `NAME`s `ngu` and `agent57`; the second adds the split networks and the meta-controller (§2) |
| base agent | `algos/r2d2/`, imported: the LSTM net, the sequence replay, the burn-in agent. The intrinsic modules feed it an augmented reward and two more cell inputs |
| the intrinsic reward | NGU's: an **episodic** term from a k-nearest-neighbour count in a learned embedding, reset every episode, times a **lifelong** term from random network distillation, clipped to [1, L]. Both new, in `algos/ngu/intrinsic.py` |
| the embedding | NGU's inverse-dynamics embedding: f(obs) trained to predict the action between consecutive observations, on 5 steps per sampled window (§0). On a 26-value vector the embedding is an MLP to 32 dims. **Each lane acts on a snapshot of f taken at its episode start** (§0) |
| the conditioning | a family of N policies indexed by β_i (intrinsic weight) and γ_i (discount). **The index one-hot, the previous extrinsic reward and the previous intrinsic reward enter the LSTM beside the previous action**, as the paper's Fig. 1 (§0); the observation is not widened. `SNEK_NGU_MIXTURES` **32, the paper's** -- the lanes are vectorised, so 32 lanes is one lane per mixture and costs nothing the box lacks; β_max `SNEK_NGU_BETA_MAX` (0.3) with the paper's spacing β_0 = 0, β_{N−1} = β_max, β_i = β_max · σ(10 (2i − (N − 2)) / (N − 2)); γ from `SNEK_DISCOUNT` (γ_0, the exploitative end, **0.997** for NGU and **0.9999** for Agent57, the papers') down to `SNEK_NGU_GAMMA_MIN` (0.99), spaced as each paper's equation (§2b) |
| the assignment | `SNEK_NGU_ASSIGN`: **`fixed`** -- lane j holds mixture j mod (N − 1) for the run (NGU App. A; §0), F1's and F2's control cell's; **`bandit`** -- the meta-controller's choice per episode, F2's; `episode` -- a uniform draw per episode, the local variant |
| the target | **transformed Retrace, λ 0.95** (`SNEK_NGU_RETRACE_LAMBDA`), replacing E2's n-step target in this group; the replay gains a `mu` column (the behaviour probability of the action taken) and the target policy is ε-soft greedy at `SNEK_NGU_TARGET_EPSILON` 0.01 (§0). E2's n-step is the ablation knob (`SNEK_NGU_RETRACE_LAMBDA=0` falls back to it) |
| the sidecar | `recurrent` from E and `conditioning`: `{"mixtures": 32, "beta_max": 0.3, "gamma_top": 0.997, "gamma_min": 0.99, "split": "none" \| "networks" \| "heads"}`; in the signature, so the ladder and the layout are rebuilt from the file |
| restore | greedy under the **exploitative** index (β = 0, the highest γ), which is what the eval measures and what NGU's paper evaluates; a `--policy-variant mixture:<i>` in the style of `a-dqn-distributional.md` §5 measures any other, and is how Agent57's bandit arm is measured (§0). **The checkpoint is the whole `NguNet`** -- Q network(s), embedding, RND pair, normalisation buffers -- and the restored policy computes r^i_{t−1} per lane with its own episodic store (§0) |
| the step | E2's |

**The episodic memory is per lane.** The k-NN store is one ring per lane of the paper's 30,000 keys
(`SNEK_NGU_MEMORY`), reset with the lane; no Snake episode approaches that, so nothing wraps in practice. This
is the same per-lane state ownership E1's seam decision gives the recurrent policy, extended by one array.

## 2. The rows

### F1 -- NGU (Badia et al. 2020, "Never Give Up")

R2D2 plus the intrinsic reward above, with the extrinsic and intrinsic rewards summed as
r = r_e + β_i r_i and one Q-network over the sum.

| module | contents |
|---|---|
| `algos/ngu/intrinsic.py` | the RND predictor and target (`SNEK_NGU_RND_HIDDEN` 128, output 128 -- the papers' architecture figures, not their text) with a running normalisation of the error, α_t = 1 + (err − μ)/σ clipped to [1, L] with L = 5; the inverse-dynamics embedding (32 dims, classifier hidden 128, L2 1e-5) at lr 5e-4; both the predictor and the classifier trained on the last 5 valid steps of each sampled window (§0); the episodic k-NN (k = 10, kernel ε 1e-4 (§0), cluster distance 8e-3, pseudo-count constant 1e-3, maximum similarity 8, memory 30,000, all exposed as knobs prefixed `SNEK_NGU_`) |
| `algos/ngu/algo.py` (`ngu`) | wraps `R2d2Algo`: `advance()` computes r_i per lane per step from the lane's snapshot embedding of the new observation, augments the reward before it enters the sequence buffer, hands the cell the index one-hot and both previous rewards, writes `mu`, trains RND and the embedding on the same batch; `fields()` reports the mean r_i and the mixture per lane |
| the actors | lane j on mixture j mod (N − 1) for the run (§0), on its own ε from the ladder |

Tests: the episodic reward on a repeated observation falls with each repeat and resets on a new
episode; RND error is ~1 in normalised units on novel inputs and falls on trained ones; the inverse
dynamics classifier reaches a non-chance accuracy on a scripted trajectory; with β = 0 the augmented
reward equals the extrinsic one to the bit; the Retrace target on a hand-worked three-step sequence with
μ = π reduces to the λ-return and with μ ≠ π cuts the trace at min(1, π/μ); the lane's embedding snapshot
does not move mid-episode; each γ ladder is strictly decreasing with the papers' endpoints (§0's erratum); a restored checkpoint reproduces the trainer's r_i on a scripted episode to the bit. Mutants: the episodic store not cleared on reset, the lifelong term unclipped,
the one-hot for the wrong index, the trace coefficient uncapped at 1, the snapshot replaced mid-episode.

### F2 -- Agent57 (Badia et al. 2020)

NGU with two changes: the Q-function is **split** into an extrinsic and an intrinsic network trained on
their own rewards and combined as Q_e + β Q_i (so the intrinsic scale cannot destabilise the extrinsic
estimate), and a **meta-controller** -- a sliding-window UCB bandit per actor -- chooses the mixture
index at each episode start from the undiscounted extrinsic return the index earned, instead of a
fixed assignment.

| module | contents |
|---|---|
| `algos/ngu/net.py::SplitHead` | **two networks of identical architecture**, as the paper (§3.1, App. E: "the associated optimizer state to be separated"), not two heads on one LSTM: Q_e(θ_e) and Q_i(θ_i), each with its own Adam, each trained with its own transformed Retrace target on its own reward stream under the shared target policy argmax(Q_e + β_j Q_i), both at the mixture's γ_j (the table's "discount for r_i 0.99" is read as γ_min, the ladder's bottom, not a second discount per head); combined only for acting. `SNEK_A57_SPLIT=heads` shares the trunk and LSTM for the cheaper local variant |
| `algos/ngu/meta.py` | the bandit (App. D): per lane, every arm pulled once first (A_k = k), then with probability 1 − ε_UCB the arm maximising μ̂(a, τ) + β √(1 / N(a, τ)) over the last τ episodes (no log term), else a uniform arm; τ `SNEK_A57_WINDOW` **160** (§4's actors' value; Table 3 prints 90), β `SNEK_A57_UCB_BETA` 1.0, ε_UCB `SNEK_A57_EPSILON` 0.5; the reward is the episode's undiscounted extrinsic return; the chosen index is logged per episode |
| `algos/ngu/algo.py` (`agent57`) | `ngu` with the split networks and the meta-controller replacing the fixed assignment; `fields()` adds the fraction of episodes the bandit spent at β = 0 and the lanes' modal greedy arm over the window -- **the numbers this row is for**; `on_eval` writes the modal arm into the row (§0) |

Tests: the combined Q at β = 0 equals the extrinsic network; each network's loss sees only its own reward
(a fixture zeroes one stream and checks the other network's gradient is unchanged); the bandit converges
to the arm with the higher mean return on a two-arm synthetic problem within the window, and pulls every arm
once before any UCB choice. Mutants: the intrinsic network trained on the summed reward, the bandit's
exploration bonus dropped, the window not sliding, the return discounted.

## 2b. The papers' settings, and how each lands here

| setting | NGU (Table 6, App. A, F) | Agent57 (Table 3, App. D, E, G) | here |
|---|---|---|---|
| base | R2D2: LSTM 512, dueling, Adam 1e-4 ε 1e-4, clip 40, batch 64, trace 80 / replay period 40, target 1,500 | the same, trace **160** / period 80 (App. E also mentions 80 / 40 as an alternative) | E2's paper cell with `SNEK_ADAM_EPSILON=1e-4`, `SNEK_TARGET_UPDATE_PERIOD=1500`; F1 at `SNEK_R2D2_SEQ_LENGTH=120 BURN_IN=40 STRIDE=40` (block 80), F2 at `200 / 40 / 80` (block 160). Burn-in is not stated in either paper; R2D2's 40 is carried |
| return estimator | transformed Retrace, λ 0.95; target ε 0.01 | the same | Retrace λ 0.95 with `mu` stored and the ε-soft target policy at 0.01 (§0); E2's n-step is the ablation |
| mixtures | 32; β_max 0.3; γ 0.997 → 0.99, γ_i = 1 − exp(((N−1−i) log(1−γ_max) + i log(1−γ_min)) / (N−1)) | 32; γ_0 0.9999, γ_1 0.997, γ_2 0.99 in three pieces (App. G.1). **As printed** the middle piece is γ_1 + (γ_0 − γ_1) σ(10 (2j − 6) / 6) for j = 1..6 and the lower is 1 − exp(((N−9) log(1−γ_1) + (j−8) log(1−γ_2)) / (N−9)) for j ≥ 8, which rises then falls and ends at 0.99997; **read as the erratum §0 states**: monotone from γ_0 to γ_1 at j = 7 and the NGU interpolation from γ_1 at j = 8 to γ_2 at j = 31 | **the paper's, per row**. The exploitative index acts at the top of the ladder, so F1 is measured at 0.997 and F2 at 0.9999; a 2,300-move game at 0.9999 is effectively undiscounted, which is what the paper's exploitative arm is too |
| assignment | actor j → β_{j mod (N−1)}, fixed | the bandit, per actor, per episode | per lane, as each paper (§0) |
| inputs to the core | encoder features ⊕ one-hot β_i ⊕ a_{t−1} ⊕ r^e_{t−1} ⊕ r^i_{t−1} | the same, j the bandit's index | the same, in `_cell_input` (§0) |
| priority | exponent 0.9, η 1.0 (App. A; the table's 0.9 is not taken), IS 0 | exponent 0.9, η 0.9, IS 0 | `SNEK_PRIORITY_EXPONENT=0.9 SNEK_R2D2_IS_BETA=0`; `SNEK_R2D2_PRIORITY_ETA` 1.0 for F1, 0.9 for F2 |
| replay | 5e6 (units unstated), 6,250 sequences before learning | the same | 30,000 windows, the box's (§0); 6,250 windows before learning |
| intrinsic module | RND 128, embedding 32 (App. B.1: 32 "consistently better" than 64), classifier 128, lr 5e-4, L2 1e-5, L 5, k 10, memory 30,000, embedding snapshot once per episode, 5 steps a sequence for the inverse-dynamics loss | the same | the same, over an MLP in place of the conv stack |
| bandit | -- | window 160, β 1, ε_UCB 0.5, per actor; undiscounted extrinsic return; the evaluator's own bandit at 3,600 / 0.01 | per lane; the evaluator's bandit becomes the logged modal arm plus a `mixture:<i>` pass (§0) |
| exploration | ε_l = 0.4^(1 + 8 l / 255) over 256 actors (Agent57 App. E; NGU's text does not print the formula); eval ε 0.01 | the same | the Ape-X ladder over 32 lanes with `SNEK_APEX_ALPHA=8` (E2 ran 7, R2D2's); eval greedy |
| value rescaling | h with ε 1e-3 (Table 6; App. E says 1e-2) | 1e-3 | 1e-3, E2's |
| frames | 35B | the text gives milestones (51 games passed at 5B, Skiing at 78B), the Atlas carries 90B | set by the update count the benchmark gate affords (§3), the paper's budgets being three orders beyond the box, and the rows here to confirm a null |

## 3. The batches

| batch | arms | base | read against | judged on |
|---|---|---|---|---|
| F1 | 4 seeds of `ngu` on §2b + **4 seeds the same at `SNEK_NGU_BETA_MAX=0`** (every β_i 0; the same Retrace, inputs, ladder, prefill, replay and updates -- NGU's own R2D2 (Retrace) baseline, and the control that says whether the intrinsic term hurts), exploitative index measured | E2's paper config at §0's changes | **the β = 0 cell** for the intrinsic term; E2 paper (b52a-d) for the series' cross-row reading, with the estimator, target period, Adam ε, IS exponent, prefill and ladder named as what else moved | stage-B density, `hof5000`, `hof30k`, drawdowns; the onset step; **the mean r_i trace**, which should fall to its clip floor early if the game has nothing to explore |
| F2 | 4 seeds of `agent57` on §2b (two networks, trace 160) + **4 seeds the same at `SNEK_NGU_ASSIGN=fixed`** (the split networks, the 0.9999 ladder and the 160 trace with the bandit replaced by F1's fixed assignment: the control that isolates the meta-controller). `SNEK_A57_SPLIT=heads`, a cost saver and not a question, moves to the tuning wave | F1's | **the fixed-assignment cell** for the bandit; F1 and E2 paper for the cross-row reading | as F1; **the bandit's β = 0 fraction over training** and its modal arm |

**The budget is set by a benchmark, not assumed (2026-10-12).** b52's update cost was 517 ms at a 125-slot window
(`e-r2d2.md` §2b), and 130k updates was the ~24 h wave. F1 adds the intrinsic modules to that (~1.2x). F2's paper cell
unrolls **two** LSTM networks over a **200-slot** window: ~3-4x b52 an update, so 24 h of a 4-arm wave is ~35-45k
updates, far under b52's 130k and the 200k working floor. Before either spec is written: benchmark `ngu` and `agent57`
updates and collectors solo (the 2,000-step rule), then set `SNEK_REPLAY_RATIO`, `SNEK_MAX_STEPS` and the cap so each cell
reaches the most updates ~24 h buys, state the shortfall against b52's 130k in the manifest note, and read F2 against F1
at a common update count. If F2 cannot reach half of b52's updates in 24 h, the batch is the exception
`algorithm-series.md` §0 allows and runs longer, said so in the spec. Memory: a 200-slot window at 30,000 windows is
~2.6 GB an arm plus two networks; four arms fit the box.

**Registered prediction (the agent's, 2026-09-16).** F1 trails E2 on onset and is level or below on
density: the intrinsic term pays for visiting board states the shortest route avoids, and a lane on a
high-β mixture wastes its episode. F2 recovers E2's numbers and its bandit spends >80% of late episodes
at β = 0, which is the meta-controller learning that the game has no exploration problem. A result that
would matter: F1 *above* E2, which would say the endgame's rare fatal states are undersampled by the
greedy policy and novelty reaches them.

## 4. Gates

1. E2 closed -- **done 2026-10-09** (b52).
2. Smoke for both names; the exploitative checkpoint restores and watches; a `mixture:<i>` variant restores.
3. The mutation specs kill every mutant.
4. **The benchmark** (§3): update and collector cost for both names, the update count written into the specs.
5. Tuning budget: one laptop wave on β_max and the kernel ε (1e-4 against the text's 1e-3). The rows are not
   tuned further; a null here is the expected finding.

## 5. What would change the plan

- **F1 beats E2.** The one surprising outcome; it would send the series back to Group A's diagnosis,
  because it would mean the rare fatal states are a *coverage* problem the replay never sees, and
  that is a different lever (targeted replay, or G's search) from a better value estimate.
- **F2's bandit does not settle at β = 0.** Either the window is too short for a 2,000-move game's
  return signal or intrinsic reward is genuinely earning extrinsic return; the β trace and the r_i
  trace together say which, and only the second reopens the plan.
- **Both null, as predicted.** The finding is written in one line and the group closes.
