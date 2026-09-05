# The PPO hyperparameter sweep — one knob per batch, four seeds per value

**Written 2026-09-01, revised the same day after review.** This is the design for batches **b9–b21**:
one knob per batch, every value at four seeds, all on one frozen base, all at one horizon, all judged
on the same pre-registered numbers. The machine-readable half is
[`hyperparam-sweep.json`](hyperparam-sweep.json); `tools/sweep_specs.py` expands one batch of it into
desktop specs and writes the laptop smoke script for the batch's never-exercised values.

**The goal is not a record.** It is to know what each knob *does* on this task — the shape of its
curve, where it is flat, where it breaks — so that a default can be chosen on evidence and the
combination batch at the end can be designed rather than guessed. The record attempt comes after.

**Revision notes.** The first draft split every grid into a core tier and a fine tier; the review
asked for the full grids, because the granularity is the point — a curve read across 16 values shows
its shape even where no single neighbouring pair separates at n=4. So: **every value runs, four arms
each, no tiers.** `target_KL` 0.1 is dropped as unable to fire; 0.04 and 0.05 stay as the null check.
Five batches were added for knobs the paste did not name (section 5, b17–b21), and the smoke
procedure is now generated rather than described.

## 0. Batch order, at a glance

Run order (section 7), each batch one knob, four seeds per value.

| order | batch | knob | what it is | waves | ~time | state (2026-09-04 19:30) |
|---:|---|---|---|---:|---:|---|
| 1 | b9 | GAE λ | how far the advantage reads from the rollout vs. the critic | 8 | 20 h | closed 09-02 16:48 |
| 2 | b10 | discount γ | how far ahead the value function looks | 8 | 23 h | closed 09-03 16:08 (the γ 1.00 wave ran 6 h) |
| 3 | b11 | learning rate | Adam step size | 4 | 10.3 h + 1.2 h hof | closed 09-04 02:24; hof passes done 19:03 |
| 4 | b12 | epochs | passes over each rollout | 5 | 12.9 h + ~1.3 h hof | closed 09-04 15:15; `hof5000` (652 rows) running |
| 5 | b13 | minibatch | gradient batch size, and steps per epoch | 4 | laptop, ~6 h/wave | closed 09-04; hof passes queued on the desktop |
| 6 | b14 | rollout T | steps per lane before an update | 3 | laptop, ~6 h/wave | wave 2 training; closes ~midday 09-05 |
| 7 | b15 | entropy coef | bonus for a stochastic policy, plus two anneals | 5 | 13 h + hof | wave 1 closed 17:50; waves 2-5 behind three hof passes, **~08:00-10:00 09-05** |
| 8 | b16 | target_KL | early-stops the epoch loop after a big update | 5 | ~14.5 h | **~evening 09-05** |
| 9 | b17 | clip + anneals | trust region on the policy ratio; anneal clip and lr to ~0 (anneal code landed) | 8 | ~23 h | **~evening 09-06** |
| 10 | b18 | grad-norm clip | ceiling on the global gradient norm before each step | 3 | ~9 h | ~morning 09-07 |
| 11 | b19 | switches | adv-norm, value loss (mse/huber), Adam ε, vf_coef | 3 | ~9 h | ~afternoon 09-07 |
| 12 | b20 | collect lanes | parallel games vs. per-episode depth, same batch size as b14 | 2 | ~6 h | ~evening 09-07 |
| 13 | b21 | shaping coef + gate | the reward's dense safety signal (runs last — it's a reward knob) | 3 | ~9 h | **~morning 09-08** |
| — | b22 | factorial | combine whichever knobs above won cleanly | 4 | ~12 h | after b21, once designed |

**What the times are.** The closed batches are measured, from the `results` branch: the first wave's
arms land ~2.3 h after the previous batch's last stage-B pass, each wave after that lands 2.3-3.0 h
behind the one before (stage B runs beside the next wave's training, so it costs the batch only its
last pass, ~1 h), which comes to **~2.6 h per 8-arm wave all in** — b7's 2.8 h estimate held. Cells
with more gradient steps run at the slow end (b12's epochs 12-16 waves took 2.8-3.0 h). The two
hall-of-fame passes a batch gets after stage B — `hof5000` over its ≥99/500 rows, `hof30k` over the
≥99/5,000 survivors — took b11 47 min and 25 min (661 rows, then 22), and **they hold training back
while they run**: b15's waves 2-5 are waiting on b12's and b13's passes now. b16-b21 are queued with a
pass per wave rather than per batch, so their estimates are 2.6 h plus ~0.3 h per wave. Laptop waves
(b13, b14) train in ~5.7 h, not 2.3, with everything else it runs.

**Remaining desktop work from here, b15's waves 2-5 through b21: ~80 h, so b21 closes around
2026-09-08 morning** if nothing is pulled from the queue and nothing else is put in front of it. The
original budget stands: 58 waves at ~2.6 h is ~150 h of waves plus ~16 h of hof passes, against the
~162 h planned in section 7.

## 1. What this project has learned about sweeps, and what it forces here

Three findings shape every rule below. Each is in [`../docs/findings.md`](../docs/findings.md).

| finding | consequence |
|---|---|
| **b3's 3M ranking was not a noisy 10M ranking, it was a different ranking.** The two arms that finished 1st and 2nd were 6th and 7th at the cap | a cap every arm is still climbing through produces no ranking. The horizon must be past where the statistic settles, and "still rising at the cap" is a result to record, not a number to rank |
| **b3's two best single knobs stacked into the worst 8-seed batch** (b4) | a one-at-a-time sweep ranks knobs; it does not license stacking winners. The combination phase is a small factorial, not one arm |
| **b5-vs-b6's 3.3 pp gap at 500 episodes was entirely selection**; b4-vs-b6's 5.6 pp survived. And **n=4 cannot resolve an effect below ~10 pp** on the peak statistics | a single pair of neighbouring values will rarely separate. What the dense grids buy is the **curve**: a monotone trend across nine values is evidence that no one 4-vs-4 comparison is. Read the grid as a curve, and read density and collapse share, which pool hundreds of rows, over `best30`, which is one |

And one from b7 that decides the metric: **`strong_eval_fraction` ranks configs backwards on record
density** (Spearman −0.79). The primary readouts are the ≥98%/500 density and the collapse share.

## 2. The base config, frozen for the whole sweep

**The base is b7's reference at `fc (320,)`, which is PPO's default everywhere else** — the best
configuration this project has measured at n≥4, and the one b7 has already run four seeds of at
exactly the horizon below. Those four arms, `b7aa`–`b7ad`, **are the control for every batch in this
sweep**, so no batch queues control arms.

| knob | value | why this and not something else |
|---|---|---|
| algorithm, network | PPO, `fc (320,)` | b7: 17.3% record density, every seed above every seed of five other layouts |
| reward | preset `b2` — chase-safe 0.1 at gate 75, no food-distance term | every PPO batch has used it; b21 is the one batch that varies it, and it runs last |
| epochs, minibatch, rollout, lanes | 4, 256, 128 steps, 128 lanes | 4 not 8: at both shapes 8 epochs lost (b4 vs b6, b5 vs b7) |
| lr, γ, λ, entropy, clip | 3e-4, 0.99, **0.99** (0.98 until b9 closed 2026-09-02), 0.01, 0.2 | b3's reference; every one of them is a batch below. b11-b14 were re-based to λ 0.99 before they started; b10 runs at 0.98 as queued |
| grad-norm clip, vf_coef, Adam ε, adv-norm, value loss | 0.5, 0.5, 1e-7, on, huber | PPO's defaults; b18 and b19 check them |
| `target_KL`, anneals | off | never exercised on this shape; b8 is testing two of them on `fc (200,100)` |
| seeds | 1, 2, 3, 4, pinned to the arm letter | seed-matched to b7's control and to every batch since b1 |
| stage A | queue on, depth 16, 8 workers | the desktop's measured optimum for eight PPO trainers |

Every one of these is written explicitly into every spec, so a later change to a code default cannot
move the base under a running sweep.

**The base does not move mid-sweep**, even when a batch finds a clear winner. Folding a winner into
the base halfway would give the later batches a different control from the earlier ones, and the
whole point of the design is one control shared by all of them. Winners meet in the factorial,
section 6.

**b8 is not folded in either.** It runs on `fc (200,100)` at 8 epochs because b4 is its control; its
two entropy treatments read "marginally ahead and unresolved" at 71M, and whatever it finds needs
re-running on this base before it counts. Its entropy and `target_KL` cells appear again in b15 and
b16 on the right shape, which is that re-run.

## 3. The horizon: 50M transitions, not 30M

**50,003,968 transitions an arm (3,052 rollouts), b7's cap.** The question was whether 30M would do,
and the data says no, for three reasons in decreasing order of weight.

**The record-density statistic is still climbing steeply at 30M.** Read off `b7aa-fc320-seed1`, the
first control arm — competence (trailing-30 ≥ 90%) arrives at 9.45M, and the share of post-competence
stage-A evals at ≥98% is:

| horizon | post-competence evals | share ≥98% |
|---:|---:|---:|
| 20M | 280 | 8.2% |
| 30M | 503 | 11.3% |
| 50M | 1,138 | **19.0%** |

The number nearly doubles between 30M and 50M. A 30M cap would rank configs on how fast they climb
into the record region, which is the b3 failure exactly. 50M is the shortest horizon at which this
project has *measured* the statistic and found it usable — b7's finding that density is "nearly flat
in budget past 50M" (b7's 11.8% at 50M against b6's 12.9% at 215M on the same shape) says 50M is
roughly where it settles, and says nothing about 30M.

**The drawdown statistic needs post-competence length.** The collapse rate this sweep is chasing —
the share of post-competence evals below 50% perfect — was 9.1% in b4 and *invisible at 10M*. 50M
gives the control ~40M of post-competence history, or ~2,400 evals per arm.

**The control is free at 50M.** b7's four `fc 320` arms ran to exactly this cap with exactly this
base. At any other horizon the control would have to be re-run or truncated.

**The cost.** b7 measured 8 arms at 50M as ~2.8 h a wave including its stage B. A slow config (λ 0,
γ 0.7, lr 4e-5, 1 epoch, shaping off) may not reach competence by 50M; that is a result, recorded as
"onset not reached at 50M", and because `SNEK_MAX_STEPS` is absolute the arm can be resumed to 100M
with a second spec if the curve says it is worth it. Anneals re-stretch on resume — see b15.

## 4. The readouts, pre-registered

Read every batch on these, in this order, per seed and pooled. Nothing else is a headline.

| # | readout | what it answers | where it comes from |
|---:|---|---|---|
| 1 | **≥98%/500 density** — share of stage-B rows at ≥98% | how often the config produces a record-region checkpoint | the batch's stage-B pass, automatic |
| 2 | **collapse share** — post-competence stage-A evals below 50% perfect; also below 80% | the drawdowns. Stage A measures the argmax, so this is the deployed policy failing, not noise | `runs/<arm>_evals.json`, free |
| 3 | **stage-A ≥98% share**, post-competence | the record region from the other instrument. Correlates +0.80 with readout 1 | same file |
| 4 | **competence onset** — first step at which trailing-30 ≥ 90% | learning speed; also whether the arm got there at all | same file |
| 5 | `best_perfect30` | the peak. Reported, not ranked on — it is one number per arm | summary block |
| 6 | **5,000-episode rows ≥98.73% and ≥99.0%**, over `above:98.5` candidates | "truly over 99%". The only readout that measures what the sweep is for | a `hof-remeasure` pass, ~25–60 min a batch on the laptop |
| 7 | **the curve** — readouts 1–4 plotted against the knob across the whole grid | the shape: flat top, cliff, monotone slope. This is what the dense grids are for | computed from the above |

**Readout 6 is the north star and it is expensive, so run it on every batch's winners rather than on
every arm** — at minimum the best two values plus the control. No snek3 checkpoint has yet held 99% at
depth: the record is 98.96% on 30,000 episodes, and the best 5,000-episode rows in b4/b5/b6 were
98.80/99.20/99.10. A config that moves readout 6 is the config; one that moves only readout 1 has
moved the 500-episode selection.

**The pairwise test is an exact Mann-Whitney on the four per-seed values, and 0.029 is the floor** —
it means every seed of one value beat every seed of the other. Anything short of that at n=4 is a
lean, and a lean is recorded as a lean; the curve (readout 7) is what turns a run of leans into a
finding. Post-competence statistics are computed from the arm's own onset, so a slow config is not
penalised for its climb, only for what it did after arriving.

## 5. The knobs — what each controls, and what to expect

Each batch is one knob; `ctrl` marks the base value, which is never re-run. Predictions are written
down so they can be wrong, and each arm's spec carries its cell's prediction in `notes`. Grids are
padded to a multiple of 8 arms with informative values so no wave straddles two knobs.

### Values from the paste that were left out, and why

| value | why not |
|---|---|
| lr 3e-3 | b3c diverged there (69.9, sd 18.4). 2e-3 is the top of the grid and locates the cliff |
| `target_KL` 0.1 | cannot fire: b4's worst update in 97,656 was 0.514 and its p99 was 0.023, so 0.1 is the control with extra steps. 0.04 and 0.05 stay as the null check |
| minibatch 4096 and rollout 2048 | not asked for and not added: 4 gradient steps an epoch, or 191 update rounds in the whole run, are outside what the base could recover from. 2048 and 1024 respectively are the far ends |

Everything else in the paste is in a grid, including the deliberately far-off ends (λ 0, γ 0.7,
entropy 0, shaping off). Those are cheap and each answers a question nothing nearer would.

### b9 — GAE λ (`SNEK_PPO_GAE_LAMBDA`) — 16 values, 64 arms

**What it controls.** How the advantage is estimated. λ=0 is one-step TD: `A = r + γV(s') − V(s)`,
so the advantage is entirely the critic's opinion — low variance, biased by every critic error. λ=1
is Monte Carlo: `A = Σγᵏr − V(s)`, unbiased and as noisy as the rest of the episode. In between, the
advantage sees `1/(1 − γλ)` steps — 33.6 at the base, 100 at λ=1, 1 at λ=0. **The +100 for a
perfect game reaches the policy through the critic at every λ below 1**, because a game is ~950
moves; λ only decides how much of the *local* signal is read from the rollout versus from the critic.

| end | prediction |
|---|---|
| **0.0, 0.5** | slowest onset in the sweep — the terminal reward propagates one bootstrap step per update, DQN-style. Smooth once competent if the critic is good (explained variance 0.90 on the gate arm). May plateau *below* the base: a critic-only advantage cannot express what the critic has not learned. λ 0 possibly not competent by 50M |
| **1.0, 0.999** | b3f at 1.0: 90.8 at 10M and still rising, worst λ arm, sd 5.2. Noisy advantages, later onset, more drawdowns; at γ=0.99 the horizon is the discount's own 100 steps |
| **0.8 – 0.9** | horizon 4.8–10 steps. Fine on onset, possibly *better* on collapse (b3e at 0.95 had 0.0% below 80%, n=1), possibly a lower ceiling |
| **0.91 – 0.995** | the plateau. The grid's job is to show it is flat and where its edges are |

**Expected optimum: 0.95–0.99, broad flat top.** A lower collapse share at 0.9–0.95 than at 0.98 that
holds across the neighbouring values is the sweep's first real result.

Grid: `0.0, 0.5, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.99, 0.995, 0.999, 1.0`.
Smoked: 0.0 and 1.0 — both reach the trainer (GAE horizon reports 1 and 100 steps, 2026-09-01).

### b10 — discount γ (`SNEK_DISCOUNT`) — 16 values, 64 arms

**What it controls.** How far ahead the value function looks: horizon `1/(1−γ)` — 3 steps at 0.7,
100 at 0.99, 400 at 0.9975, unbounded at 1.0. It also sets the shaping's discount (fixed 2026-08-29,
so the potential-based term stays policy-invariant), so a γ change slightly changes the dense reward
too — correctly, but worth knowing. And it interacts with λ through `1/(1−γλ)`: at γ=0.8 the GAE
horizon is 4.6 steps whatever λ is.

**Invariant 6 bounds it from below and it does not bite here.** A win of 100 raises value only when
`100 > 1/(1−γᵏ)` for k steps a meal; at γ=0.7, k=10 that is 1.03, at 0.99 it is 10.4, at 0.9975 it
is 34–58. No value in the grid makes finishing unattractive. What low γ does instead is **myopia in
the endgame**: a trap that kills 30 moves later is invisible to a 5-step value function.

| end | prediction |
|---|---|
| **0.7, 0.8, 0.85** | fast early learning — food is 7–12 steps away and fully visible — then a ceiling below perfect, because the last 10–20 squares need planning beyond the horizon. The chase-safe shaping is a dense safety signal, so the ceiling may be higher than the theory says; the cells most likely to surprise |
| **0.9975, 0.999, 1.0** | b3d at 0.9975 read 81.6 at a 3M cap; b3o at 0.995 read 96.7 with the *lowest* sd30 in b3 (1.8). Long horizons make the value target a whole-game sum — ~95 food + 100 + shaping — so critic targets are larger and explained variance should drop. At 1.0 the rollout-boundary bootstrap carries full weight. Slower onset; *either* the smoothest arms in the sweep or the noisiest |
| **0.9 – 0.98** | the ramp up to the base, at 0.01 resolution: where does the endgame ceiling lift? |
| **0.995** | the one value with an n=1 hint of beating the base on stability |

**Expected optimum: 0.99–0.995.** snek2's record DQN config ran 0.9975; PPO has run it once for 3M.

Grid: `0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.995, 0.9975, 0.999, 1.0`.
Smoke: 1.0.

### b11 — Adam step size (`SNEK_PPO_LEARNING_RATE`) — 8 values, 32 arms

**What it controls.** The size of every parameter update. It is the crudest of the three "update
size" knobs — clip bounds how far the *policy ratio* may move, `target_KL` bounds how far the
*policy* may move per rollout, lr bounds how far the *parameters* move per gradient step — and the
only one that also scales the critic.

**b3's n=1 readings, which this batch tests at n=4:** peaked at 3e-4. 1e-3 read 85.2 (sd 7.2) and
3e-3 read 69.9 (sd 18.4) at 3M; 5e-4 matched the base; 1e-4 read 95.0 with *more* post-competence
evals below 80% (14.5% vs 12.2%), a later onset and zero ≥98%/500 rows. **The low end did not buy
stability**, which is the counter-intuitive reading worth confirming.

| end | prediction |
|---|---|
| **4e-5, 1e-4, 1.5e-4** | late onset (possibly >50M at 4e-5), no fewer collapses, lower density. If the collapse mechanism were a rare over-sized update, a smaller lr should shrink it in proportion; b3 says it did not, so the mechanism may be many moderate updates |
| **8e-4, 1e-3, 2e-3** | more collapses, higher sd, earlier onset. 2e-3 sits between 1e-3 and the divergence and should show where the cliff is |
| **2.5e-4, 5e-4** | within noise of the base |

**Expected optimum: 2.5e-4 – 5e-4, i.e. the base.** The information is in the two ends.

Grid: `4e-5, 1e-4, 1.5e-4, 2.5e-4, 5e-4, 8e-4, 1e-3, 2e-3`.

### b12 — epochs (`SNEK_PPO_EPOCHS`) — 10 values, 40 arms

**What it controls.** Passes over each rollout, so gradient steps per sample. Each pass moves the
policy further from the one that collected the data: the ratio drifts, the clip binds more,
`approx_kl` grows. 1 epoch never sees a sample twice and the clip is nearly inert; 16 re-fits the same
16,384 samples sixteen times.

**This is the axis that has moved most in this project.** b3: minibatch 1024 (0.25x steps) 89.7,
base (1x) 96.6, epochs 8 (2x) 97.2 — monotone at n=1. Then at n=8, 8 epochs lost to 4 at both shapes,
and b4's 8-epoch arms spent 9.1% of their post-competence evals collapsed against 0.7% for b6's
4-epoch arms. Current reading: **onset speed rises with epochs and stability falls, and 4 is where the
trade sits.**

| end | prediction |
|---|---|
| **1, 2** | very stable, slow. 1 epoch may not reach the record region by 50M; 2 probably does. Collapse share near zero — says what stability costs in speed |
| **10, 12, 16** | b4's collapse pattern, worse: `approx_kl` tails, clip saturating. The highest collapse shares in the sweep short of entropy 0.03 |
| **3, 5, 6, 7, 8** | the shape between 2 and 10 at unit resolution. 8 has never been measured against 4 on `fc 320` at a matched budget |

**Expected optimum: 3–4.** The suggestion of pairing high epochs with a lower lr or clip is the right
follow-up and belongs in the factorial (section 6) as two interaction cells: 16 epochs x lr 1e-4 asks
whether it is the *number* of steps or their *total size* that breaks things.

Grid: `1, 2, 3, 5, 6, 7, 8, 10, 12, 16`. Smoke: 16.

### b13 — minibatch (`SNEK_PPO_MINIBATCH`) — 8 values, 32 arms

**What it controls.** Two things at once, which is why b3's one reading was ambiguous. Smaller
minibatches mean *noisier gradients* (32 samples) **and** *more gradient steps per epoch* (512 at 32
against 64 at 256) — so minibatch 32 at 4 epochs takes as many steps as minibatch 256 at 32 epochs.
Advantage normalisation is per minibatch, so at 32 the advantage scale itself is noisy.

| end | prediction |
|---|---|
| **32, 64** | fast onset, high collapse share — the epochs-16 pattern by another route, plus gradient noise. If 32 is *not* worse than 16 epochs, step count is not the mechanism and reuse is |
| **1024, 2048** | b3r at 1024: 89.7, the worst non-diverged arm. Few, smooth steps; slow. 2048 (8 steps an epoch) says whether the slope continues or the arm never arrives |
| **128, 192, 384, 512** | the neighbourhood of the base. 128 is the likeliest alternative default. 192 and 384 leave a short last minibatch each epoch, which is harmless |

**Expected optimum: 128–256.** Read beside b12: together they separate "steps" from "reuse".

Grid: `32, 64, 128, 192, 384, 512, 1024, 2048`. Smoke: 32, 2048.

### b14 — rollout horizon T (`SNEK_PPO_ROLLOUT`) — 6 values, 24 arms

**What it controls.** Steps each of the 128 lanes takes before an update, so transitions per update
(`128 x T`) and, at a fixed cap, the number of update rounds (3,052 at 128; 12,208 at 32; 381 at
1,024). It also truncates GAE: an advantage near the end of the buffer is bootstrapped after fewer than
T steps, which matters at T=32 against the base's 33.6-step horizon and at λ→1.

**Two side effects to hold in mind.** Stage A's interval is one rollout, so T=1,024 writes a
checkpoint every 131k transitions — 8x fewer checkpoints and 8x fewer stage-B rows; the density
*share* stays comparable, the *counts* do not. And caps that are not a multiple of the rollout run one
rollout past 50,003,968.

| end | prediction |
|---|---|
| **32, 64** | b3p at 64: ~2.5 pp below the base at n=1. More, smaller updates from less data each; 32 also truncates GAE below its horizon. Worse everywhere |
| **1,024** | 381 rounds by 50M is probably too few; late onset, strong stability if it arrives |
| **192, 256, 512** | more data per update, fewer rounds — equal or slightly better stability, similar density, marginally later onset. 256 is the likeliest alternative default |

**Expected optimum: 128–256.** Read beside b20, which changes the same batch size the other way.

Grid: `32, 64, 192, 256, 512, 1024`. Smoke: 32, 1024.

### b15 — entropy coefficient (`SNEK_PPO_ENTROPY_COEF`, `_FINAL`) — 10 cells, 40 arms

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
| **0.0** | no bonus at all. Says whether the term matters on this task; the risk is an early-deterministic policy that stalls |
| **0.001, 0.003, 0.005** | fewer collapses than the base, similar onset. If monotone down to 0.001 with no density loss, the default moves |
| **0.02, 0.03** | 0.03 confirms the catastrophe at n=4; 0.02 says where the cliff is |
| **anneal 0.1 → 0.001** | explore hard, commit late. 0.1 early is 3x the catastrophic constant, so the first ~15M will look broken; the question is whether the endgame that follows is *better* for it. The ramp is linear over `SNEK_MAX_STEPS`, crossing 0.03 at 35M and 0.01 at 45M. Late onset, possibly the best late stability. 0.03→0.001 is the milder version |
| **anneal 0.01 → 0.001, 0.01 → 0** | b8's cell on this shape, and the same ramp taken all the way to zero |

**Expected optimum: 0.001–0.005, or a short anneal.** The ramp is a function of the cap: an arm
resumed to a higher cap re-stretches its schedule, so an annealed arm is never resumed for comparison.

Grid: `0.0, 0.001, 0.003, 0.005, 0.02, 0.03, 0.1→0.001, 0.03→0.001, 0.01→0.001, 0.01→0`.
Smoke: 0.0, the 0.1 anneal.

### b16 — `target_KL` (`SNEK_PPO_TARGET_KL`) — 10 values, 40 arms

**What it controls.** An early stop on the epoch loop: after each epoch, if the epoch's mean
`approx_kl` exceeds the threshold, the remaining epochs are skipped. **It removes only the tail of
large updates and leaves every other update identical to the control**, and at 4 epochs it has
exactly three places to stop, so its effect is bounded.

**Where it binds is known.** b4's `approx_kl` per update: median 0.0035, p95 0.0079, p99 0.023, worst
0.514. So 0.003 fires on most updates, 0.005 on ~25%, 0.01 on ~4%, 0.02 on ~1%, and **0.03 and above
almost never** — those cells are predicted identical to the control and are in the grid as the null
check the prediction deserves. Neighbouring thresholds differ by which handful of updates they catch;
the curve across them is what will show whether anything happens at all.

| cell | prediction |
|---|---|
| **0.003, 0.005** | the aggressive end: ~1–3 epochs on average. Best stability in the batch if the tail is the mechanism; slightly later onset |
| **0.008 – 0.015** | cuts the top 1.5–5%. The first values with a plausible effect |
| **0.02** | b8's value on this shape. Near-null unless collapses come from rare huge updates |
| **0.03, 0.04, 0.05** | the control by another name |

**Expected outcome: a small effect, monotone in the threshold, with ~0.01 the useful value if any is.**
The knob most likely to produce a clean null, which is also worth knowing.

Grid: `0.003, 0.005, 0.008, 0.01, 0.013, 0.015, 0.02, 0.03, 0.04, 0.05`.

### b17 — clip, and the clip and lr anneals (`SNEK_PPO_CLIP`; **anneals need code**) — 16 cells, 64 arms

**What clip controls.** The trust region on the probability ratio per sample; 0.2 lets an action's
probability move 20% per update. Never swept in this project. Smaller is a tighter region — slower,
steadier; larger lets one update move the policy further, and at 0.4 with 4 epochs it barely binds.

**What annealing does.** The PPO paper's Atari recipe anneals clip 0.1→0 *and* lr 2.5e-4→0 over
training: the policy is allowed big moves while it is bad and ever smaller ones as it gets good, so
late training cannot undo what it has learned. **This is the textbook lever for late-training
drawdowns and the cell in this whole sweep with the strongest prior of helping.** The cost is that a
mistake learned late cannot be unlearned late either, and the schedule is tied to the cap.

| cell | prediction |
|---|---|
| **clip 0.05, 0.1, 0.15** | tighter: slower onset, fewer collapses; 0.05 possibly never reaches the record region |
| **clip 0.3, 0.4** | looser: earlier onset, more collapses; 0.4 approaches unclipped policy gradient |
| **clip 0.2 → 0.02** | steady endgame, fewer late collapses; the record region should be *denser* late in the run. Not to exactly 0 — `ppo/algo.py` refuses a clip outside (0, 1), and a clip of 0 admits no update |
| **clip 0.1 → 0.02** | the paper's start value |
| **lr 3e-4 → 0**, **→ 3e-5** | the same shape of effect through the parameters; also cools the critic. Half the run is below 1.5e-4, so expect b3i's onset penalty, mildly. The 3e-5 floor tests whether the last stretch near zero was doing anything |
| **both** | the Atari recipe. Predicted best late stability of anything in the sweep; the risk is a lower ceiling |
| **clip 0.2 → 0.1**, **0.4 → 0.02** (*added 2026-09-03*) | 0.2→0.1 is "annealing at all" without ending tight; 0.4→0.02 is the widest ramp, loose start and tight end |
| **clip 0.2 → 0.005**, **→ 0.001** (*added 2026-09-03*) | floors below 0.02, which the code allows (any value in (0, 1)). At 0.001 the tail can barely move the policy: is 0.02 still too loose late, or does a floor this low cost the endgame? |
| **clip 0.2 → 0.02** and **0.2 → 0.001, held from 80%** (*added 2026-09-03*) | `SNEK_PPO_ANNEAL_FRACTION=0.8`: reaches the floor at 40M and trains the last 10M there, so the endgame is measured under the tight region rather than still descending; the 0.001 hold asks whether a near-frozen endgame keeps what it had or stalls. Need the fraction knob deployed on the box; the generator drops both cells until then |

**Expected: the anneals win on collapse share, and the question is what they cost on density.**

**Code needed first**: `SNEK_PPO_CLIP_FINAL` and `SNEK_PPO_LEARNING_RATE_FINAL`, both the same linear
ramp over `SNEK_MAX_STEPS` that `ppo/schedules.entropy_coef_for` already implements, applied in
`ppo/algo.advance()` beside the entropy coefficient — the clip as `agent.clip`, the lr by setting
`param_group['lr']` on the one optimiser. Two ramps, a test each, a mutation check, and the smoke
runs. Until it lands the generator drops the five anneal cells, and **b17 should wait for them rather
than run its five static cells as a short batch** — the code is a morning's work and can land during
b9.

Grid: `clip 0.05, 0.1, 0.15, 0.3, 0.4; 0.2→0.02; 0.1→0.02; lr→0; lr→3e-5; both; 0.2→0.1; 0.4→0.02; 0.2→0.001; 0.2→0.005; 0.2→0.02 held from 80%; 0.2→0.001 held from 80%`.
Smoke: clip anneal, lr anneal, the 80% hold (smoked 2026-09-03: clip read 0.03125 at 75% of the cap and 0.02 at the cap, lr untouched).

### b18 — gradient-norm clipping (`SNEK_PPO_GRADIENT_CLIPPING`) — 6 values, 24 arms — *added*

**What it controls.** A ceiling on the global gradient norm over both towers before each Adam step;
0.5 by default, 0 turns it off. Never swept. It is the one knob that acts on the *size of a single
gradient* rather than on the policy or the ratio, which makes it **a direct test of the tail-update
hypothesis** — if b4's worst updates (`approx_kl` 0.514) are carried by a few huge gradients, turning
the clip off should make collapses worse and tightening it should make them rarer, without touching
the epoch count.

| cell | prediction |
|---|---|
| **0 (off)** | if collapses come from rare huge gradients, the worst arm in the batch; if they are policy-level drift, nothing changes. Either answer is useful |
| **0.1, 0.25** | clips most updates: slower, steadier — like a lower lr applied only to the big steps |
| **1.0, 2.0, 5.0** | binds less and less; 5.0 is effectively off |

**Expected optimum: 0.5–1.0; the information is in 0.**

Grid: `0, 0.1, 0.25, 1.0, 2.0, 5.0`. Smoke: 0.

### b19 — the switches: advantage normalisation, value loss, Adam ε, vf_coef — 6 cells, 24 arms — *added*

Four knobs with two or three sensible values each, gathered into one batch because none has a curve
to map; each cell checks that a default is not silently costing something.

| cell | what it changes | prediction |
|---|---|---|
| **adv-norm off** | the update now scales with the reward; a +100 terminal in a minibatch dominates it | more collapses, a policy that over-weights the win. The base's normalisation is what makes the update reward-scale invariant |
| **value loss `mse`** | squared error instead of Huber | one +100 terminal dominates 256 samples — the reason Huber is the default. Noisier critic, lower explained variance, possibly more collapses |
| **Adam ε 1e-5** | the PPO paper's value, 100x the base | damps the step for parameters with tiny second moments: slightly steadier, possibly slightly slower |
| **Adam ε 1e-8** | torch's default | within noise of 1e-7 |
| **vf_coef 0.1, 1.0** | the critic loss weight | **predicted inert**: the towers share no parameters and Adam is scale-invariant per parameter, so it can only shift the critic's effective step through the ε term. Two arms buy the null the prediction deserves |

Grid: `adv-norm 0; mse; ε 1e-5; ε 1e-8; vf 0.1; vf 1.0`. Smoke: adv-norm off, mse.

### b20 — collect lanes (`SNEK_COLLECT_ENVS`) — 4 values, 16 arms — *added*

**What it controls.** How many games run in lockstep — the other half of the rollout. Transitions
per update are `lanes x T`, so 256 lanes at T=128 is the same batch as 128 lanes at T=256, **but from
twice as many independent episodes with half the per-episode depth.** Read beside b14: if `lanes256`
beats `roll256`, episode diversity matters more than depth. Rollout size changes the stage-A interval
exactly as T does.

| cell | prediction |
|---|---|
| **32, 64** | small, correlated batches from few games; worse than the matching T cells because diversity is lower too |
| **256, 512** | same batch as `roll256` and `roll512`; 763 update rounds by 50M at 512 |

Grid: `32, 64, 256, 512`. Smoke: 32, 512.

### b21 — chase-safe shaping coefficient and gate (`SNEK_CHASE_SAFE_SHAPING`, `_GATE`) — 6 cells, 24 arms — *added*

**A reward knob, not a PPO knob, and the one batch that changes what every other batch held fixed** —
so it runs last. Stage B measures the perfect rate, which no reward term can move, so the comparison
stands; the desktop keys eval waves on these two variables, so each value gets its own stage-B wave
automatically.

**What it controls.** The dense safety signal: a potential-based bonus for keeping the free space
reachable, paid at coefficient `c` once the score passes the gate. snek2's batches 28–29 found the
gate to be the lever and `c=0.10` at gate 75 set the records; on this stack, b1 (shaping off) never
reached 95/100 in 3M DQN steps while b2 (b29's config) did. PPO has only ever run b2's setting.

| cell | prediction |
|---|---|
| **c = 0** | shaping off. The 508k gate arm on the bare reward was 1% perfect; the question is whether PPO gets there at all by 50M without it |
| **c = 0.05, 0.2** | potential shaping is policy-invariant, so `c` should change learning speed and not the optimum; expect both within noise once competent, 0.2 faster to onset |
| **gate 60, 85** | 60 shapes 15 more squares of the game (snek2's direction of improvement); 85 is snek3's own default and shapes only the last 10 — expect worse than 75 |
| **gate 0** | shaped from the first move. Says whether the gate does anything for PPO or was a DQN-era artefact |

Grid: `c 0.0, 0.05, 0.2; gate 0, 60, 85`. Smoke: gate 0.

## 6. The factorial, and why it is not one arm

**b22, designed after b9–b21 are read.** Take every knob whose winning value separated from the base
completely (p=0.029 on readout 1 or 2), whose curve (readout 7) is coherent around it, and whose
readout-6 pass confirmed it. Call them A, B, C — expect two or three, not thirteen. Run the **full
factorial**, 4 seeds each, at 50M: base+A, +B, +AB, +C, +AC, +BC, +ABC — 7 cells, 28 arms, padded to
32 with the two epochs-interaction cells (16 epochs x lr 1e-4, 16 epochs x clip 0.1) if b12 made them
interesting. Every cell is compared to the base *and* to its own sub-cells, so an interaction like
b4's shows up as a cell below its parts rather than as a surprise.

**Only then the champion attempt**: the best factorial cell at 200M+ with 8 seeds, the `hof5000` pass
on everything above 98.5, and a 30,000-episode measurement of the single winner at a fresh seed.

If b8 has closed with a winner on `fc (200,100)`, its knob enters the factorial only through its
b15/b16 re-run on the base, never directly.

## 7. Budget and order

| batch | knob | cells | arms | waves | ~hours | code first |
|---|---|---:|---:|---:|---:|---|
| b9 | λ | 16 | 64 | 8 | 22 | |
| b10 | γ | 16 | 64 | 8 | 22 | |
| b11 | lr | 8 | 32 | 4 | 11 | |
| b12 | epochs | 10 | 40 | 5 | 14 | |
| b13 | minibatch | 8 | 32 | 4 | 11 | |
| b14 | rollout T | 6 | 24 | 3 | 8.5 | |
| b15 | entropy | 10 | 40 | 5 | 14 | |
| b16 | target_KL | 10 | 40 | 5 | 14 | |
| b17 | clip + anneals | 16 | 64 | 8 | 22 | **yes** |
| b18 | grad-norm clip | 6 | 24 | 3 | 8.5 | |
| b19 | switches | 6 | 24 | 3 | 8.5 | |
| b20 | lanes | 4 | 16 | 2 | 5.5 | |
| b21 | shaping | 6 | 24 | 3 | 8.5 | |
| **total** | | **116** | **464** | **58** | **~162 h** | |
| b22 | factorial | ~8 | 32 | 4 | 11 | |

**About a week of desktop** at b7's measured 2.8 h per 8-arm wave including stage B, with the box
otherwise idle. Every batch is a multiple of 8 arms so no wave straddles two knobs and every
auto-queued stage B measures one batch. The wave time is dominated by stage A and stage B, not by the
gradient steps, so cells with many more updates (epochs 16, minibatch 32) will run somewhat longer and
cells with far fewer checkpoints (T 1024) shorter.

**Order.** b9 and b10 first, as asked and because the advantage pair is the least explored; then the
update-size quartet b11–b14, which is where the drawdown mechanism most likely lives; b15 and b16,
which b8 will have partly previewed; b17 as soon as its code lands (it can land during b9); then the
three added checks b18–b20; b21 last because it moves the reward. **Six batches are queued at a time**
(the user's call, 2026-09-01), with ascending priorities so the box runs them in order and never idles;
b9–b14 went up first, and b15–b20 follow as b9–b14 drain, with b17 now unblocked by the anneal code.
A batch that produces an unexpected cliff can still be pulled from the queue before it starts.

**Cost of readout 6.** One `hof-remeasure` pass on a batch's `above:98.5` candidates: b4's 274 rows
took 26 min on the laptop; a 64-arm batch with denser winners will have more. Budget an hour a batch
on the laptop, run while the desktop trains the next one.

## 8. Rules for running it

- **Smoke every never-exercised value before its batch is pushed.** `tools/sweep_specs.py <batch>
  --smoke <file.sh>` writes a zsh script that runs each cell marked `smoke` in the manifest for
  65,536 transitions on the laptop under the policy name `smoke`, printing the
  `hyperparameter override:` lines, the config line and the reward line, and cleaning up after
  itself. Read the value back — the GAE horizon for λ, the rollout size for T and lanes, the ramp
  value at a known fraction for an anneal. b8 did this by hand and it is why its two knobs are known
  to be live; a silently ignored knob costs four arms. b9's two smokes ran 2026-09-01 and both
  reached the trainer.
- **Specs come from the manifest, not by hand.** `tools/sweep_specs.py <batch> --out <dir>` writes
  them into the `ops` worktree and validates each against `parse_job`; the `desktop-batch` skill does
  the push. A design change is made in the JSON so the specs follow.
- **Every arm's `notes` field carries the prediction for its cell**, copied from the manifest, so the
  reader of the spec knows what the arm was expected to do without opening this file.
- **A batch closes when its stage B has landed and readouts 1–5 and 7 are in `results.md`**, per seed
  and pooled, with the Mann-Whitney against the b7 control and the curve drawn. Readout 6 follows for
  the winners. A batch whose slow cells did not reach competence records that and may resume them
  (never an annealed one); it does not rank them.
- **Log of what went up.** 2026-09-01 20:30: b9, b10 (128 arms). 2026-09-01 ~21:00: b11–b14 (128 arms),
  after the anneal knobs were committed (`fd2584011`) and the box fast-forwarded to `13def02e8`. 2026-09-03 18:10: b15-b21 (208 arms, priorities 170-232), the whole rest of the sweep behind b11-b14, so the queue holds eleven batches at once; b22 is designed once all of them are read.
- **The base does not change until b22.** A batch that shows a clear winner writes it in
  `findings.md` as a candidate for the factorial, not as a new default.
