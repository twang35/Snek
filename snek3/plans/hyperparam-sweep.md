# The PPO hyperparameter sweep — one knob per batch, four seeds per value

**Written 2026-09-01, while b8 wave 2 was training.** This is the design for batches **b9 onward**:
one knob per batch, every value at four seeds, all on one frozen base, all at one horizon, all
judged on the same pre-registered numbers. The machine-readable half is
[`hyperparam-sweep.json`](hyperparam-sweep.json); `tools/sweep_specs.py` expands one batch of it
into desktop specs.

**The goal is not a record.** It is to know what each knob *does* on this task — the shape of its
curve, where it is flat, where it breaks — so that a default can be chosen on evidence, and so that
the combination batch at the end can be designed rather than guessed. The record attempt comes after.

## 1. What this project has learned about sweeps, and what it forces here

Three findings shape every rule below. Each is in [`../docs/findings.md`](../docs/findings.md).

| finding | consequence |
|---|---|
| **b3's 3M ranking was not a noisy 10M ranking, it was a different ranking.** The two arms that finished 1st and 2nd were 6th and 7th at the cap | a cap every arm is still climbing through produces no ranking. The horizon must be past where the statistic settles, and "still rising at the cap" is a result to record, not a number to rank |
| **b3's two best single knobs stacked into the worst 8-seed batch** (b4) | a one-at-a-time sweep ranks knobs; it does not license stacking winners. The combination phase is a small factorial, not one arm |
| **b5-vs-b6's 3.3 pp gap at 500 episodes was entirely selection**; b4-vs-b6's 5.6 pp survived. And **n=4 cannot resolve an effect below ~10 pp** on the peak statistics | fine grids (λ 0.91, 0.92, 0.93 ...) cannot be told apart at n=4. Spend arms where the curve bends, and read density and stability, which pool hundreds of rows, over `best30`, which is one |

And one from b7 that decides the metric: **`strong_eval_fraction` ranks configs backwards on record
density** (Spearman −0.79). The primary readouts are the ≥98%/500 density and the stage-A ≥98% share.

## 2. The base config, frozen for the whole sweep

**The base is b7's reference at `fc (320,)`, which is PPO's default everywhere else** — the best
configuration this project has measured at n≥4, and the one b7 has already run four seeds of at
exactly the horizon below. Those four arms, `b7aa`-`b7ad`, **are the control for every batch in this
sweep**, so no batch queues control arms.

| knob | value | why this and not something else |
|---|---|---|
| algorithm, network | PPO, `fc (320,)` | b7: 17.3% record density, every seed above every seed of five other layouts |
| reward | preset `b2` — chase-safe 0.1 at gate 75, no food-distance term | every PPO batch has used it; changing it re-opens every comparison |
| epochs, minibatch, rollout | 4, 256, 128 lanes x 128 steps | 4 not 8: at both shapes 8 epochs lost (b4 vs b6, b5 vs b7) |
| lr, γ, λ, entropy, clip | 3e-4, 0.99, 0.98, 0.01, 0.2 | b3's reference; every one of them is a knob this sweep tests |
| `target_KL`, anneals | off | never exercised on this shape; b8 is testing two of them on `fc (200,100)` |
| seeds | 1, 2, 3, 4, pinned to the arm letter | seed-matched to b7's control and to every batch since b1 |
| stage A | queue on, depth 16, 8 workers | the desktop's measured optimum for eight PPO trainers |

**The base does not move mid-sweep**, even when a batch finds a clear winner. Folding a winner into
the base halfway would give the later batches a different control from the earlier ones, and the
whole point of the design is one control shared by all of them. Winners meet in the combination
batch, section 6.

**b8 is not folded in either.** It runs on `fc (200,100)` at 8 epochs because b4 is its control; its
two entropy treatments read "marginally ahead and unresolved" at 71M, and whatever it finds needs
re-running on this base before it counts. Its entropy and `target_KL` cells appear again below on
the right shape, which is that re-run.

## 3. The horizon: 50M transitions, not 30M

**Recommendation: 50,003,968 transitions an arm (3,052 rollouts), b7's cap.** The question was
whether 30M would do, and the data says no, for three reasons in decreasing order of weight.

**The record-density statistic is still climbing steeply at 30M.** Read off `b7aa-fc320-seed1`, the
first control arm — competence (trailing-30 ≥ 90%) arrives at 9.45M, and the share of post-competence
stage-A evals at ≥98% is:

| horizon | post-competence evals | share ≥98% |
|---:|---:|---:|
| 20M | 280 | 8.2% |
| 30M | 503 | 11.3% |
| 50M | 1,138 | **19.0%** |

The number nearly doubles between 30M and 50M. A 30M cap would rank configs on how fast they climb
into the record region, which is the b3 failure exactly: the 3M→10M inversion happened because the
cap sat on the climb. 50M is the shortest horizon at which this project has *measured* the statistic
and found it usable — b7's finding that density is "nearly flat in budget past 50M" (b7's 11.8% at 50M
against b6's 12.9% at 215M on the same shape) says 50M is roughly where it settles, and says nothing
about 30M.

**The drawdown statistic needs post-competence length.** The collapse rate this sweep is chasing —
the share of post-competence evals below 50% perfect — was 9.1% in b4 and *invisible at 10M*. 50M
gives the control ~40M of post-competence history, or ~2,400 evals per arm; 30M gives 20M.

**The control is free at 50M.** b7's four `fc 320` arms ran to exactly this cap with exactly this
base. At any other horizon the control would have to be re-run (4 arms, half a wave) or truncated,
and truncation loses the last 20M where b7's density concentrates.

**The cost.** b7 measured 8 arms at 50M as ~2.8 h a wave including its stage B, so a 32-arm batch is
~11 h and the whole core sweep below is ~3.5 days of desktop. 30M would save ~1 h a wave and lose the
ranking. A slow config (λ 0, γ 0.8, lr 4e-5, 1 epoch) may not reach competence by 50M; that is a
result, recorded as "onset not reached at 50M", and because `SNEK_MAX_STEPS` is absolute the arm can
be resumed to 100M with a second spec if the curve says it is worth it.

## 4. The readouts, pre-registered

Read every batch on these, in this order, per seed and pooled. Nothing else is a headline.

| # | readout | what it answers | where it comes from |
|---:|---|---|---|
| 1 | **≥98%/500 density** — share of stage-B rows at ≥98% | how often the config produces a record-region checkpoint | the batch's stage-B pass, automatic |
| 2 | **collapse share** — post-competence stage-A evals below 50% perfect; also below 80% | the drawdowns. Stage A measures the argmax, so this is the deployed policy failing, not noise | `runs/<arm>_evals.json`, free |
| 3 | **stage-A ≥98% share**, post-competence | the record region again, from the other instrument. Correlates +0.80 with readout 1 | same file |
| 4 | **competence onset** — first step at which trailing-30 ≥ 90% | learning speed; also whether the arm got there at all | same file |
| 5 | `best_perfect30` | the peak. Reported, not ranked on — it is one number per arm | summary block |
| 6 | **5,000-episode rows ≥98.73% and ≥99.0%**, over `above:98.5` candidates | "truly over 99%". The only readout that measures what the user actually wants | a `hof-remeasure` pass, ~25 min a batch on the laptop |

**Readout 6 is the north star and it is expensive, so run it on every batch's winners rather than on
every arm** — at minimum the best two values plus the control. No snek3 checkpoint has yet held 99% at
depth: the record is 98.96% on 30,000 episodes, and the best 5,000-episode rows in b4/b5/b6 were
98.80/99.20/99.10. A config that moves readout 6 is the config; one that moves only readout 1 has
moved the 500-episode selection.

**The test is an exact Mann-Whitney on the four per-seed values, and 0.029 is the floor** — it means
every seed of one value beat every seed of the other. Anything short of that at n=4 is a lean, and a
lean is recorded as a lean. Post-competence statistics are computed from the arm's own onset (readout
4), so a slow config is not penalised for its climb, only for what it did after arriving.

## 5. The knobs — what each controls, and what to expect

Each batch is one knob. `ctrl` marks the base value, which is never re-run. **Core** is the grid to
queue; **fine** is the extra resolution to add only if the core shows a gradient in that region,
because n=4 cannot resolve neighbours 0.01 apart. Predictions are written down so they can be wrong.

### b9 — GAE λ (`SNEK_PPO_GAE_LAMBDA`)

**What it controls.** How the advantage is estimated. λ=0 is one-step TD: `A = r + γV(s') − V(s)`,
so the advantage is entirely the critic's opinion — low variance, biased by every critic error. λ=1
is Monte Carlo: `A = Σγᵏr − V(s)`, unbiased and as noisy as the rest of the episode. In between, the
advantage sees `1/(1 − γλ)` steps — 33.6 at the base, 100 at λ=1, 1 at λ=0. **The +100 for a
perfect game reaches the policy through the critic at every λ below 1**, because a game is ~950
moves and no λ<1 horizon comes close; λ only decides how much of the *local* signal is read from the
rollout versus from the critic.

| end | prediction |
|---|---|
| **λ = 0.0** | slowest onset in the sweep — the terminal reward has to propagate one bootstrap step per update, DQN-style, but with PPO's ~256 updates per rollout rather than one per transition. Smooth once competent, if the critic is good (explained variance was 0.90 on the gate arm). The interesting question is whether it plateaus *below* the base: a critic-only advantage cannot express anything the critic has not learned. Expect onset >30M or not at all |
| **λ = 1.0** | b3f: 90.8 best30 at 10M and still rising, worst of the λ arms, sd 5.2 — noisy advantages, slow. Expect more drawdowns than the base and a later onset; at γ=0.99 the effective horizon is the discount's own 100 steps |
| **0.8 – 0.9** | horizon 4.8–10 steps. The local signal is short and the critic carries the rest. Probably fine on onset, possibly *better* on collapse (b3e at 0.95 had 0.0% of evals below 80%, n=1), possibly a lower ceiling |
| **0.95 – 0.995** | the plateau. Expect these to be within noise of each other and of the base; the sweep's job is to show the plateau is real and where its edges are |

**Expected optimum: 0.95–0.99, with a broad flat top.** If the collapse share is lower at 0.9–0.95
than at 0.98 with complete separation, that is the sweep's first real result.

Core: `0.0, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 1.0` (8 values, 32 arms, 4 waves). Fine:
`0.91, 0.92, 0.93, 0.94, 0.96` — added only if core shows a slope between 0.9 and 0.97.

### b10 — discount γ (`SNEK_DISCOUNT`)

**What it controls.** How far ahead the value function looks: horizon `1/(1−γ)` — 5 steps at 0.8,
100 at 0.99, 400 at 0.9975, unbounded at 1.0. It also sets the shaping's discount (fixed 2026-08-29,
so the potential-based term stays policy-invariant), so a γ change slightly changes the dense reward
too — correctly, but worth knowing. And it interacts with λ through `1/(1−γλ)`: at γ=0.8 the GAE
horizon is 4.6 steps whatever λ is.

**Invariant 6 bounds it from below and it does not bite here.** A win of 100 raises value only when
`100 > 1/(1−γᵏ)` for k steps a meal; at γ=0.8, k=10 that is 1.12, at 0.99 it is 10.4, at 0.9975 it
is 34–58. So no value in the grid makes finishing unattractive. What low γ does instead is
**myopia in the endgame**: a trap that kills 30 moves later is invisible to a 5-step value function.

| end | prediction |
|---|---|
| **γ = 0.8, 0.9** | fast early learning — food is 7–12 steps away and fully visible — then a ceiling below perfect, because the last 10–20 squares need planning beyond the horizon. The chase-safe shaping is a dense safety signal, so the ceiling may be higher than the theory says; this is the cell most likely to surprise |
| **γ = 0.9975, 0.999, 1.0** | b3d at 0.9975 read 81.6 at 3M and was capped there; b3o at 0.995 read 96.7 with the *lowest* sd30 in b3 (1.8). Long horizons make the value target the sum of a whole game — ~95 food + 100 + shaping — so the critic's targets are larger and noisier and explained variance should drop. At exactly 1.0 the bootstrap at the rollout boundary carries full weight, and a truncated 128-step buffer is bootstrapping most of an episode. Expect slower onset, and *either* the smoothest arms in the sweep (b3o's hint) or the noisiest (b3d's) |
| **0.95 – 0.995** | the plateau; 0.995 is the one value with an n=1 hint of being better than the base on stability |

**Expected optimum: 0.99–0.995.** snek2's record DQN config ran 0.9975; PPO has run it once for 3M.

Core: `0.8, 0.9, 0.95, 0.98, 0.995, 0.9975, 0.999, 1.0` (8 values, 32 arms, 4 waves). Fine:
`0.91–0.97 by 0.01`, only if core shows a slope in that region.

### b11 — Adam step size (`SNEK_PPO_LEARNING_RATE`)

**What it controls.** The size of every parameter update. It is the crudest of the three "update
size" knobs — clip bounds how far the *policy ratio* may move, `target_KL` bounds how far the
*policy* may move per rollout, lr bounds how far the *parameters* move per gradient step — and it is
the only one of the three that also scales the critic.

**b3's n=1 readings, which this batch tests at n=4:** peaked at 3e-4. 1e-3 read 85.2 (sd 7.2) and
3e-3 read 69.9 (sd 18.4) at 3M; 5e-4 matched the base; 1e-4 read 95.0 with *more* post-competence
evals below 80% (14.5% vs 12.2%), a later onset and zero ≥98%/500 rows. So the low end did not buy
stability, which is the counter-intuitive reading worth confirming.

| end | prediction |
|---|---|
| **4e-5, 1e-4** | late onset (possibly >50M at 4e-5), no fewer collapses, lower density. If the collapse mechanism is a rare over-sized update, a smaller lr should shrink it in proportion; b3 says it did not, so the mechanism may be many moderate updates rather than a tail |
| **8e-4, 1e-3** | more collapses, higher sd, earlier onset. 1e-3 should still be competent at 50M (b3b was 85 at 3M) but with visible drawdowns — this is the cell that shows what a drawdown *is* on this shape |
| **2.5e-4, 5e-4** | within noise of the base |

**Expected optimum: 2.5e-4 – 5e-4, i.e. the base.** The information is in the two ends.

Core: `4e-5, 1e-4, 2.5e-4, 5e-4, 8e-4, 1e-3` (6 values, 24 arms, 3 waves).

### b12 — epochs (`SNEK_PPO_EPOCHS`)

**What it controls.** Passes over each rollout, so gradient steps per sample. Each pass moves the
policy further from the one that collected the data: the ratio drifts, the clip binds more, `approx_kl`
grows. 1 epoch never sees a sample twice and the clip is nearly inert; 16 epochs re-fits the same
16,384 samples sixteen times.

**This is the axis that has moved most in this project.** b3: minibatch 1024 (0.25x steps) 89.7,
base (1x) 96.6, epochs 8 (2x) 97.2 — monotone at n=1. Then at n=8, 8 epochs lost to 4 at both shapes,
and b4's 8-epoch arms spent 9.1% of their post-competence evals collapsed against 0.7% for b6's
4-epoch arms. So the current reading is: **onset speed rises with epochs and stability falls, and 4 is
where the trade sits.**

| end | prediction |
|---|---|
| **1, 2** | very stable, slow. 1 epoch may not reach the record region by 50M; 2 probably does. Collapse share near zero — this is the cell that says how much stability costs in speed |
| **12, 16** | b4's collapse pattern, worse: `approx_kl` tails, clip saturating, `clip_fraction` high. Expect the highest collapse shares in the sweep short of entropy 0.03 |
| **3, 5, 6** | resolve the shape between 2 and 8 |

**Expected optimum: 3–4.** The user's suggestion of pairing high epochs with a lower lr or clip is
the right follow-up and belongs in the combination batch (section 6) as two interaction cells:
16 epochs x lr 1e-4 asks whether it is the *number* of steps or their *total size* that breaks things.

Core: `1, 2, 3, 5, 6, 8, 12, 16` (8 values, 32 arms, 4 waves). 8 is re-run here on purpose: it has
never been measured against 4 on `fc 320` at a matched budget (b5 was 255M+ vs b7's 50M).

### b13 — minibatch (`SNEK_PPO_MINIBATCH`)

**What it controls.** Two things at once, which is why b3's one reading was ambiguous. Smaller
minibatches mean *noisier gradients* (32 samples) **and** *more gradient steps per epoch* (512 at 32
against 64 at 256) — so a minibatch of 32 at 4 epochs takes as many steps as a minibatch of 256 at 32
epochs. Advantage normalisation is also per minibatch, so at 32 the advantage scale itself is noisy.

| end | prediction |
|---|---|
| **32, 64** | fast onset, high collapse share — the epochs-16 pattern by another route, plus gradient noise. If 32 is *not* worse than 16 epochs, the step count is not the mechanism and the reuse (seeing each sample many times) is |
| **1024, 2048** | b3r at 1024: 89.7, the worst non-diverged arm. Few, smooth steps; slow. 2048 is added to see whether the slope continues or the arm simply never arrives |
| **128, 512** | 128 is the value nearest the base on the fast side and the most likely alternative default |

**Expected optimum: 128–256.** Read this batch beside b12: the two together separate "steps" from
"reuse".

Core: `32, 64, 128, 512, 1024, 2048` (6 values, 24 arms, 3 waves).

### b14 — rollout horizon T (`SNEK_PPO_ROLLOUT`)

**What it controls.** Steps each of the 128 lanes takes before an update, so transitions per update
(`128 x T`) and, at a fixed cap, the number of update rounds (3,052 at T=128; 381 at 1,024; 6,104 at
64). It also truncates GAE: an advantage near the end of the buffer is bootstrapped after fewer than T
steps, which matters at λ→1 and not at the base's 33.6-step horizon.

**Two side effects to hold in mind.** Stage A's interval is one rollout, so T=1,024 writes a
checkpoint every 131k transitions — 8x fewer checkpoints and 8x fewer stage-B rows; the density
*share* stays comparable, the *counts* do not, and the record region is sampled more coarsely. And a
cap of 50,003,968 is not a multiple of 131,072, so the T=1,024 arms run 382 rollouts to 50,069,504.

| end | prediction |
|---|---|
| **64** | b3p: ~2.5 pp below the base at n=1. More, smaller updates from less data each; expect slightly worse everywhere |
| **256, 512** | more data per update, fewer rounds — expect equal or slightly better stability, similar density, marginally later onset. 256 is the most likely alternative default |
| **1,024** | 381 rounds by 50M is probably too few; expect late onset and a strong stability reading if it arrives |

**Expected optimum: 128–256.**

Core: `64, 256, 512, 1024` (4 values, 16 arms, 2 waves).

### b15 — entropy coefficient (`SNEK_PPO_ENTROPY_COEF`, `_FINAL`)

**What it controls.** A bonus for a stochastic policy — max entropy on 3 actions is ln 3 = 1.099,
and the gate arm went from 1.086 to 0.27 in 500k transitions. High values keep the policy random; in
a game where one random move in the endgame is fatal, a policy that cannot commit cannot be perfect.
Low values let it commit early and risk a premature local optimum — but with three actions and a dense
safety signal, premature determinism may cost nothing here.

**b3's reading is the most monotone in the sweep**: post-competence evals below 80% ran 2.9% at 0.003,
12.2% at 0.01, **45.6% at 0.03**. b8 is testing 0.003 and the 0.01→0.001 anneal on `fc (200,100)`;
both were "marginally ahead" at 71M. This batch is their re-run on the base.

| cell | prediction |
|---|---|
| **0.001, 0.003, 0.005** | fewer collapses than the base, similar onset. If monotone down to 0.001 with no loss of density, the default moves |
| **0.02, 0.03** | 0.03 confirms the catastrophe at n=4; 0.02 says where the cliff is |
| **anneal 0.1 → 0.001** | the user's proposal: explore hard, commit late. 0.1 early is 3x b3h's catastrophic constant, so the first ~15M will look broken; the question is whether the endgame that follows is *better* for it. The ramp is linear over `SNEK_MAX_STEPS`, so it crosses 0.03 at 35M and 0.01 at 45M — most of the run is spent above the known-bad value. Prediction: late onset, possibly the best late stability. A 0.03→0.001 anneal is included as the milder version |
| **anneal 0.01 → 0.001** | b8's cell on this shape |

**Expected optimum: 0.001–0.005, or a short anneal.** Note the ramp is a function of the cap: an arm
resumed to a higher cap re-stretches its schedule.

Core: `0.001, 0.003, 0.005, 0.02, 0.03, anneal 0.1→0.001, anneal 0.03→0.001, anneal 0.01→0.001`
(8 cells, 32 arms, 4 waves).

### b16 — `target_KL` (`SNEK_PPO_TARGET_KL`)

**What it controls.** An early stop on the epoch loop: after each epoch, if the epoch's mean
`approx_kl` exceeds the threshold, the remaining epochs are skipped. **It removes only the tail of
large updates and leaves every other update identical to the control**, and at 4 epochs it has
exactly three places to stop, so its effect is bounded — it can at most turn a 4-epoch update into a
1-epoch one.

**Where it binds is known.** b4's `approx_kl` per update: median 0.0035, p95 0.0079, p99 0.023, worst
0.514. So 0.02 fires on ~1% of updates, 0.01 on ~4%, 0.005 on ~25%, and **0.03 and above almost
never fire and would be arms identical to the control.** Values 0.013 and 0.015 differ by which
handful of updates they catch and cannot be told apart at n=4. b8 is testing 0.02 on `fc (200,100)`.

| cell | prediction |
|---|---|
| **0.02** | near-null: cuts the top 1%. If collapses are caused by the rare huge update, this is enough and the collapse share drops; if by many moderate ones, nothing happens |
| **0.01** | cuts the top ~4% — the first value with a plausible effect |
| **0.005** | fires on a quarter of updates; effectively ~3 epochs on average with the tail removed. Expect slightly later onset and the best stability in the batch if the mechanism is the tail |
| **0.03** | the control by another name; included as the null check |

**Expected outcome: a small effect, monotone in the threshold, with 0.01 the useful value if any is.**
This is the knob most likely to produce a clean null, which is also worth knowing.

Core: `0.005, 0.01, 0.02, 0.03` (4 values, 16 arms, 2 waves). Fine: `0.008, 0.013, 0.015` only if
0.005 and 0.01 separate.

### b17 — clip, and the two anneals (`SNEK_PPO_CLIP`; **needs code**)

**What clip controls.** The trust region on the probability ratio per sample; 0.2 lets an action's
probability move 20% per update. Never swept in this project. Smaller is a tighter region — slower,
steadier; larger lets one update move the policy further.

**What annealing does.** The PPO paper's Atari recipe anneals clip 0.1→0 *and* lr 2.5e-4→0 over
training: the policy is allowed big moves while it is bad and ever smaller ones as it gets good, so
late training cannot undo what it has learned. **This is the textbook lever for late-training
drawdowns and it is the cell in this whole sweep with the strongest prior of helping.** The cost is
that a mistake learned late cannot be unlearned late either, and the schedule is tied to the cap.

| cell | prediction |
|---|---|
| **clip 0.1, 0.3 (static)** | 0.1: slower onset, fewer collapses; 0.3: the reverse. Sets the reference for the anneals |
| **clip 0.2 → 0.02** | steady endgame, fewer late collapses; the record region should be *denser* late in the run if b6's back-loading is general. Not to exactly 0 — `ppo/algo.py` refuses a clip outside (0, 1), and a clip of 0 admits no update at all |
| **lr 3e-4 → 0** | same shape of effect through the parameters instead of the ratio; also cools the critic. The second half of the run is at <1.5e-4, so expect the b3i penalty on onset — competence later, but not much |
| **both** | the Atari recipe. Predicted best late stability of anything in the sweep; the risk is a lower ceiling |
| **clip 0.1 → 0.02** | the paper's start value, for completeness |

**Expected: the anneals win on collapse share, and the question is what they cost on density.**

**Code needed first**: `SNEK_PPO_CLIP_FINAL` and `SNEK_PPO_LEARNING_RATE_FINAL`, both the same linear
ramp over `SNEK_MAX_STEPS` that `ppo/schedules.entropy_coef_for` already implements, applied in
`ppo/algo.advance()` beside the entropy coefficient — the clip as `agent.clip`, the lr by setting
`param_group['lr']` on the one optimiser. Two ramps, a test each, a mutation check, and a smoke run
that greps the value at 10% of a short cap, as b8's knobs were smoked. Until it lands, b17 is queued
without the three anneal cells or not at all.

Core: `clip 0.1, clip 0.3, clip 0.2→0.02, lr→0, both, clip 0.1→0.02` (6 cells, 24 arms, 3 waves).

## 6. The combination batch, and why it is not one arm

**b18, designed after b9–b17 are read.** Take every knob whose winning value separated from the base
completely (p=0.029 on readout 1 or 2) and whose readout-6 pass confirmed it. Call them A, B, C —
expect two or three, not eight. Then run the **full factorial** of those, 4 seeds each, at 50M:
base+A, base+B, base+AB, base+C, base+AC, base+BC, base+ABC — 7 cells, 28 arms, padded to 32 with
the two epochs-interaction cells (16 epochs x lr 1e-4, 16 epochs x clip 0.1) if b12 made them
interesting. Every cell is compared to the base *and* to its own sub-cells, so an interaction like
b4's shows up as a cell below its parts rather than as a surprise.

**Only then the champion attempt**: the best factorial cell at 200M+ with 8 seeds, the `hof5000` pass
on everything above 98.5, and a 30,000-episode measurement of the single winner at a fresh seed.

If b8 has closed by then with a winner on `fc (200,100)`, its knob enters the factorial only through
its b15/b16 re-run on the base, never directly.

## 7. Budget and order

| batch | knob | cells | arms | waves | ~hours | needs code |
|---|---|---:|---:|---:|---:|---|
| b9 | λ | 8 | 32 | 4 | 11 | |
| b10 | γ | 8 | 32 | 4 | 11 | |
| b11 | lr | 6 | 24 | 3 | 8.5 | |
| b12 | epochs | 8 | 32 | 4 | 11 | |
| b13 | minibatch | 6 | 24 | 3 | 8.5 | |
| b14 | rollout T | 4 | 16 | 2 | 5.5 | |
| b15 | entropy | 8 | 32 | 4 | 11 | |
| b16 | target_KL | 4 | 16 | 2 | 5.5 | |
| b17 | clip + anneals | 6 | 24 | 3 | 8.5 | **yes** |
| **core total** | | **58** | **232** | **29** | **~81 h** | |
| b18 | factorial | ~8 | 32 | 4 | 11 | |

The user's full grid as pasted is ~285 arms; the difference is the fine tiers, which are in the
manifest and are queued only where the core shows a slope. **~3.5 days of desktop for the core**, at
b7's measured 2.8 h per 8-arm wave including stage B, with the box otherwise idle. Every batch is a
multiple of 8 arms so no wave straddles two knobs and every auto-queued stage B measures one batch.

**Order.** b9 and b10 first, as the user asked and because the advantage pair is the least explored;
then the update-size quartet b11–b14, which is where the drawdown mechanism most likely lives; then
the stability pair b15–b16, which b8 will have partly previewed by then; b17 as soon as its code
lands, which can be during b9. Queue two batches at a time with ascending priorities so the box never
idles between them and there is always a decision point before the third.

**Cost of readout 6.** One `hof-remeasure` pass on a 32-arm batch's `above:98.5` candidates: b4's 274
rows took 26 min on the laptop; a batch with denser winners will have more. Budget an hour a batch
on the laptop, run while the desktop trains the next one.

## 8. Rules for running it

- **Smoke every never-exercised value before its batch is queued** — λ 0.0, γ 1.0, T 1024, minibatch
  32 and 2048, every anneal — with a short cap on the laptop and `grep 'hyperparameter override:'`
  plus a read of the value in the log at 10% of the cap. b8 did this and it is why its two knobs are
  known to be live. A silently ignored knob costs four arms.
- **Specs come from the manifest, not by hand.** `tools/sweep_specs.py <batch> --tier core` writes
  them into the `ops` worktree and validates each against `parse_job`; the `desktop-batch` skill
  does the push. A manifest edit is a design change and is made in the JSON so the specs follow.
- **Every arm's `notes` field carries the prediction for its cell**, copied from the manifest, so the
  reader of the spec knows what the arm was expected to do without opening this file.
- **A batch closes when its stage B has landed and readouts 1–5 are in `results.md`**, per seed and
  pooled, with the Mann-Whitney against the b7 control. Readout 6 follows for the winners. A batch
  whose slow cells did not reach competence records that and may resume them; it does not rank them.
- **The base does not change until b18.** A batch that shows a clear winner writes it in
  `findings.md` as a candidate for the factorial, not as a new default.
