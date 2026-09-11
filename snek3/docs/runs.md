# Runs — every batch's config, why it ran, and what it taught

**One entry per batch, newest at the top.** Each entry is the config in human terms (what varies, off
which base, how many seeds, how long), the prediction registered before it ran and whether it **held**
or was **falsified**, why the batch was worth running, and what was learned — a hundred words each,
at most. The prediction row is there because a third of these batches overturned their own spec, and
that is the fact worth seeing at a glance. What is *running right now*, on which box, with what ETA, is
`status.json` (`git fetch origin ops-status && git show origin/ops-status:status.json`) and the
[live page](https://twang35.github.io/Snek/), not this file. Per-arm numbers are
[`results.md`](results.md), conclusions with their evidence are [`findings.md`](findings.md), the
records are [`../hallOfFame/HOF.md`](../hallOfFame/HOF.md). The status logs this file used to carry
are in git history before 2026-09-10.

## Open

- **b28's `hof30k` pass** decides whether the 100M hold moved the top: a row at ≥99.83 /30,000 is an
  `hof-promote` candidate, otherwise b28 reads "wider plateau, same top" and the 100M cap stands.
- **Should `SNEK_OBS_HISTORY=8` become the default.** b27 says yes; nothing has run against it yet.
- **Next sweeps on the `hist8` base**: a `hist16` cell, and step penalties above 0.01 (0.02, 0.05) —
  b26's curve never turned.
- **b24/b25's unmeasured `hof30k` arms**, and a fresh 30,000 on `b25a` @106168320, need the old
  30-value observation era to load.

## At a glance

| batch | varies | base | cells × seeds | cap | prediction | result in one line |
|---|---|---|---:|---:|---|---|
| [b28](#b28--the-hist8-config-held-for-100m-more) | 100M more hold | pen01 + hist8, anneal over 25% | 1 × 8 | 200M | — | 97.5% density and 4x the near-record rows; first 30k rows level with the HOF, not above. **Passes still running** |
| [b27](#b27--move-history-depth) | `SNEK_OBS_HISTORY` 0 / 4 / 8 | pen01 | 3 × 8 | 100M | falsified | **the largest lever found**: 49 → 94-95% density; `b27t`/`b27k` 99.81 /30k, the record |
| [b26](#b26--step-penalty) | `SNEK_STEP_PENALTY` 0 / 1e-4 / 1e-3 / 0.01 | horizon anneal, obs26 | 4 × 4 | 50M | held | 0.01 nearly doubles density (25.5 → 46.3%); smaller values do nothing |
| [b25](#b25--the-ladder-top-at-200m) | cap and seeds | ladder top | 1 × 8 | 200M | held | longer cap paid everywhere: 77.5% density, `b25a` 99.60 /30k |
| [b24](#b24--the-horizon-anneal-at-200m) | a different road to γ 0.999 / λ 0.999 | horizon anneal | 1 × 8 | 200M | falsified in part | same peak as b25 (99.5 /30k), lower density (60.5%) and stability |
| [b23](#b23--corner-grid-ladder-rungs-3-4) | + `mse` value loss; + clip hold | b22 rung 2 | 2 × 4 | 50M | falsified in part | `mse` is the largest single step measured: 32.7 → 61.6%, collapses gone |
| [b22](#b22--corner-grid-ladder-rungs-1-2) | + γ 0.999; + rollout 512 | λ 0.99 | 2 × 4 | 50M | falsified | γ 0.999 adds density and drawdown; rollout 512 on top trades one for the other |
| [b21](#b21--chase-safe-shaping) | shaping dose 0-0.2, gate 0 / 60 / 85 | PPO reference | 6 × 4 | 50M | falsified | no-op for PPO; shaping off is the most stable cell |
| [b20](#b20--collect-lanes) | `SNEK_COLLECT_ENVS` 32-512 | PPO reference | 4 × 4 | 50M | falsified | a throughput knob; 512 lanes the smoothest endgame |
| [b19](#b19--the-switches) | adv norm off, `mse`, Adam ε, vf coef | PPO reference | 6 × 4 | 50M | falsified in part | `mse` most stable and +5 pp; the rest within noise |
| [b18](#b18--gradient-norm-clip) | grad clip 0-5.0 | PPO reference | 6 × 4 | 50M | held | no-op; collapses are policy-level, and the stability column's noise is ~4 pp |
| [b17](#b17--clip-and-the-anneals) | clip 0.05-0.4, clip/lr anneals, hold | PPO reference | 16 × 4 | 50M | falsified in part | static clip flat; holding the annealed floor for the last 10M is +6-7 pp; `b17cl` 99.50 /30k |
| [b16](#b16--target-kl) | `target_kl` 0.003-0.05 | PPO reference | 10 × 4 | 50M | held | no-op at 4 epochs |
| [b15](#b15--entropy-coefficient) | entropy 0-0.03, four anneals | PPO reference | 10 × 4 | 50M | held | density and stability trade monotonically; anneals average their endpoints |
| [b14](#b14--rollout-horizon) | rollout 32-1024 | λ 0.99 | 6 × 4 | 50M | falsified | density peaks at 512 (+11 pp), stability improves through 1024 |
| [b13](#b13--minibatch) | minibatch 32-2048 | λ 0.99 | 8 × 4 | 50M | falsified | plateau at 256-512; 128 is 8 pp short |
| [b12](#b12--epochs) | epochs 1-16 | λ 0.99 | 10 × 4 | 50M | falsified in part | 3-4 is the top; the collapse cliff is at 12-16, not 8 |
| [b11](#b11--learning-rate) | lr 4e-5 to 2e-3 | λ 0.99 | 8 × 4 | 50M | falsified in part | plateau 1e-4 to 5e-4, cliffs at both ends; stability rises with lr |
| [b10](#b10--discount-γ) | γ 0.70-1.00 | PPO reference | 16 × 4 | 50M | falsified in part | monotone to 0.999; γ 1.00 collapses half the time and held the record (`b10ck` 99.65) |
| [b9](#b9--gae-λ) | λ 0-1.00 | PPO reference | 16 × 4 | 50M | falsified | 0.99 doubles density over the 0.98 default, plateau above; λ 0.99 became the default |
| [b8](#b8--the-stability-knobs-on-b4s-config) | entropy 0.003, entropy anneal, `target_kl` 0.02, λ 0.95 | b4's config | 4 × 4 | 100M | falsified | every knob cut the drawdown, none beat the control on density |
| [b7](#b7--network-shape) | 8 `fc` layouts | PPO reference | 8 × 4 | 50M | falsified | `fc (320,)` wins; b3's ranking inverted; `sef` ranks backwards |
| [b4](#b4--fc-200100--8-epochs) | b3's two best knobs stacked | PPO reference | 1 × 8 | 200M | falsified | the weakest 8-seed batch: shape and epochs interact negatively |
| [b5, b6](#b5-b6--fc-320--8-epochs-and-fc-200100--4-epochs) | `fc 320` + 8 epochs; `fc (200,100)` + 4 epochs | PPO reference | 1 × 8 each | ~220-270M | — | a 500-episode lead that 5,000 erased; `b5h` and `b6b` the first HOF entries |
| [b3](#b3--the-ppo-tuning-sweep) | 15 one-knob arms | PPO reference | 15 × 1 | 10M | falsified | no winner at n=1; PPO's record density is 11.6x DQN's |
| [b2](#b2--snek2s-record-config-on-the-torch-stack) | snek2 b29's five knobs | DQN defaults | 1 × 4 | 3M steps | held | the phase-3 gate met; a snek3 step is four game moves |
| [b1](#b1--the-ddqn-baseline) | nothing | DQN defaults | 1 × 4 | 3M steps | falsified | no checkpoint at 95/100; the wrong config to gate on |

## The bases

Every batch is a few knobs off one of these. The names below are what each entry's "base" means.

| base | what it is | used by |
|---|---|---|
| **DQN defaults** | double DQN, snek3's own defaults: `fc 320`, lr 1e-5, γ 0.99, IS weights on, target update every 8 steps, food-distance reward 0.001, chase-safe shaping off. One collect lane, 3M counted steps | b1 |
| **b29 repro** | DQN defaults with snek2 batch 29's five record knobs: IS weights **off**, target update **1000**, γ **0.9975**, food-distance reward **0**, chase-safe shaping **0.1 at gate 75** | b2 |
| **PPO reference** | PPO on the b2 reward (shaping 0.1, gate 75, food-distance 0): `fc (320,)`, 4 epochs, minibatch 256, rollout 128 × 128 lanes, lr 3e-4, γ 0.99, **λ 0.98**, entropy 0.01, clip 0.2, target KL off, gradient-norm clip 0.5, vf coef 0.5, Adam ε 1e-7, advantage normalisation on, huber value loss. 50M transitions. The control arms are `b7aa`-`b7ad` | b3, b7, b9, b10, b15-b21; b4-b6 and b8 with the shape or epochs changed |
| **λ 0.99** | the PPO reference at λ 0.99, the default since 2026-09-02. Control `b9bw`-`b9bz` | b11-b14, b22 |
| **ladder top** | λ 0.99 + γ 0.999 + rollout 512 + `mse` value loss + clip annealed 0.2 → 0.001 over the first 80% of the cap and held | b23 (built rung by rung), b25 |
| **horizon anneal** | the user's config: γ 0.99 → 0.999 and λ 0.95 → 0.999 annealed over the first half of the cap, entropy 0.01 → 0.001, lr 2.5e-4, minibatch 512, rollout 256, huber, clip 0.2 static | b24; b26 under the 26-value observation |
| **pen01** | the horizon anneal + step penalty 0.01, 26-value observation | b27; b28 with the anneal over 25% of a 200M cap |

**Two observation eras.** Until 2026-09-07 the observation was snek2's 30 values (`obs30`); from b26
on it is 26 (`obs26-20260907`, [`environment.md`](environment.md)), and `SNEK_OBS_HISTORY=N` appends
2N more. Checkpoints load only in the era they trained in, which is why b24/b25's deep passes stopped
part-way.

**How a batch is read.** Stage B measures every screened checkpoint at 500 episodes; the headline is
the share of rows at ≥98% perfect (record density), then the collapse share (post-competence stage-A
evals below 50% and 80%), then best30. `hof5000` re-measures the top rows at 5,000 episodes and
`hof30k` at 30,000 on seed 7, and only a 30,000-episode number is a record. n=4 separates cells only
at complete separation (Mann-Whitney p=0.029). Details in [`protocol.md`](protocol.md).

---

## b28 — the `hist8` config held for 100M more

| | |
|---|---|
| base | pen01 + `SNEK_OBS_HISTORY=8` (b27's `hist8` cell) |
| varies | the cap: 200M, with `SNEK_PPO_ANNEAL_FRACTION` 0.25 so every anneal spans the same first 50M as b27 and the last 150M run at the final values |
| cells × seeds | 1 × 8 (seeds 9-16, `b28i`-`b28p`) |
| control | b27's `hist8` arms, seeds 17-24 |
| predicted | none registered — the question was open — — |

**Why.** b27's `hist8` rows at 30,000 episodes sat at 99.8 with the anneal still finishing at 100M.
The question is whether the ceiling is the config or the cap: does holding the converged values for
another 100M raise the top, or only widen the plateau under it? The same anneal length keeps the first
100M window-for-window comparable with b27.

**Learned so far** (stage B and `hof5000` closed, `hof30k` running 2026-09-10). The first 100M
matches b27's `hist8` (94.9% against 95.4), so the extra hold is what reads 97.5% density and 7,718
rows through the 99.6 /5,000 gate — 4.2x b27's — with 38 at 99.9 and every arm holding a 99.9 best.
The first two 30k arms read 99.79, level with the record pair's 99.81 and inside one standard error.
Provisional verdict: the hold multiplies near-record checkpoints without raising the top.

## b27 — move history depth

| | |
|---|---|
| base | pen01 (horizon anneal + step penalty 0.01, 26-value observation) |
| varies | `SNEK_OBS_HISTORY` 0 / 4 / 8 — two bits per past move, `[turned left, turned right]`, read off the body ([`../plans/obs-history.md`](../plans/obs-history.md)) |
| cells × seeds | 3 × 8, 100M |
| control | `hist0`, the batch's own |
| predicted | no effect on the perfect rate (the best checkpoints starve in loops history cannot break) — **falsified** |

**Why.** The user's question: does the policy need to see its recent path? The plan's phase-1
investigation found the best checkpoints die by *starving* in a closed loop while the food sits
reachable (87-96% of failures), a failure history should not touch — so the registered prediction
was **no effect on the perfect rate**. Built and run anyway to measure it, at 100M and 8 seeds so a
negative would be a clean one.

**Learned.** The prediction was wrong, by the widest margin in the project: `hist4` 93.6% density and
`hist8` 95.4 against `hist0`'s 49.2, every history seed above every control seed, stage-A ≥98 share
84-85% against 47, onset unchanged. At 30,000 episodes `hist4` and `hist8` each put rows at 99.8 where
the HOF's top was 99.65; `b27t` @85065728 and `b27k` @77889536 (99.81 each, indistinguishable) are
the record pair. `hist8` is a small, consistent step past `hist4` on density and rows through every
gate, level on the peaks. Mechanism — what the bits do to the starvation orbit — is not settled.

## b26 — step penalty

| | |
|---|---|
| base | horizon anneal, under the new 26-value observation (the first `obs26` batch) |
| varies | `SNEK_STEP_PENALTY` 0 / 0.0001 / 0.001 / 0.01, a per-step reward cost |
| cells × seeds | 4 × 4, 50M |
| control | `pen0`, the batch's own; b24's 200M arms are a loose reference only |
| predicted | 0.01 large enough to be felt, the two smaller values within noise — held |

**Why.** The death-trace finding: the best checkpoints fail by orbiting reachable food, and a per-step
cost is the only reward term that charges for a lap that does not eat. Three decades of penalty to
find where it starts to bite against the b2 preset's per-step terms.

**Learned.** 0.01 is a real lever at n=4 and the other two are not: 46.3% density (42.3-52.6) against
the control's 25.5 (16.3-33.5), every seed separated from all twelve other arms; 0.001 and 0.0001 sit
inside the control's spread. The scale explains the shape — a perfect game is ~1,800 steps, so 0.01 is
~18 of a ~195 return, a tenth; 0.001 is one percent. Onset, late entropy and KL are identical across
cells: the penalty changes what the converged policy *does*, not how fast it learns. The curve had
not turned, so 0.02 and 0.05 are open.

## b25 — the ladder top at 200M

| | |
|---|---|
| base | ladder top (γ 0.999, λ 0.99, rollout 512, `mse`, clip 0.2 → 0.001 over 80% then held) |
| varies | the cap and seed count: 200M × 8 against b23's 50M × 4 |
| cells × seeds | 1 × 8 |
| control | b23's `g999roll512msehold` cell |
| predicted | fewer evals below 80 than b24, and a ≥99.3 /30k checkpoint — held |

**Why.** The first champion attempt on the corner grid's best rung. Every one-knob sweep had read at
50M and 4 seeds; the sweep design's last step was always "the best cell at 200M+, 8 seeds, deep
passes on the winner".

**Learned.** The longer cap paid on every column: 77.5% density (64.3-83.0, seven of eight seeds above
the reference's best) against 64.4, best30 99.34, stage-A ≥98 share 72.7% against 58.2, and the
collapses stayed gone (0.45% of evals below 80). `b25a` @106168320 reads 99.60 /30,000 — level with
the HOF's second place and the first checkpoint outside the γ 1.00 cell to get there — but the 30k
pass covers only three of eight arms, because the observation changed to 26 values under it and the
rest no longer load. Not promoted; awaits a fresh 30,000 under the old era.

## b24 — the horizon anneal at 200M

| | |
|---|---|
| base | horizon anneal (the user's config, first run here) |
| varies | nothing within the batch; it is a second 200M × 8 config beside b25 |
| cells × seeds | 1 × 8 |
| control | b23's `g999roll512msehold` cell, and b25 as the same-cap comparison |
| predicted | earlier onset than b25, then a denser but less stable second half — **falsified in part** — less stable held, denser did not |

**Why.** A different road to the same horizon: instead of starting at γ 0.999 / λ 0.99 (the ladder
top), anneal γ 0.99 → 0.999 and λ 0.95 → 0.999 over the first half so the critic learns short before
it learns long, with the entropy anneal, huber, minibatch 512 and rollout 256 as the user's other
choices. Prediction: earlier onset than b25, then a denser but less stable second half.

**Learned.** It matches the ladder top on the peak and trails it on the rest: best30 99.30 against
99.34, `b24a` @196706304 at 99.5 /30k against 99.6, but 60.5% density against 77.5 and twice the evals
below 80 (0.83% against 0.45). "Less stable" held, "denser" did not. Only two arms were measured at
30k before the observation change broke the pass. The ladder top was the one to carry forward — yet
the anneal is the config b26-b28 built on, because the user's step-penalty question was asked on it.

## b23 — corner-grid ladder, rungs 3-4

| | |
|---|---|
| base | b22's rung 2 (λ 0.99 + γ 0.999 + rollout 512) |
| varies | rung 3 adds the `mse` value loss; rung 4 adds the clip anneal 0.2 → 0.001 over 80% of the cap, held for the last 10M |
| cells × seeds | 2 × 4, 50M |
| control | b22's `g999roll512` cell |
| predicted | `mse` adds density and the hold adds b17's late density lift on top — **falsified in part** — `mse` held, the hold's lift did not repeat |

**Why.** The sweep's levers stacked one at a time: `mse` was b19's most stable cell and the hold was
b17's density lift, both measured on b7's λ 0.98 base. The ladder says whether they still help on the
horizon rungs, and in what order.

**Learned.** `mse` on this base is the largest single step this project has measured: 61.6% density
against 32.7 (every seed separated), best30 98.92, and the drawdowns γ 0.999 and λ 0.99 had added
gone (0.48% below 50 against 3.54). The hold adds nothing on density (64.4%, inside `mse`'s spread)
and takes out the last collapses. Neither beats the HOF's 99.65 at 30,000 — both top out at 99.2 —
but they produce ≥99 /30k checkpoints in bulk, 64 in one wave against a dozen from all of b9-b21.
b17's hold lift does not repeat on a base that already has few late collapses.

## b22 — corner-grid ladder, rungs 1-2

| | |
|---|---|
| base | λ 0.99 (`b9bw`-`b9bz`) |
| varies | rung 1 adds γ 0.999; rung 2 adds rollout 512 on top |
| cells × seeds | 2 × 4, 50M |
| control | b9's λ 0.99 cell |
| predicted | γ 0.999 + rollout 512 the densest cell measured so far — **falsified** |

**Why.** b9 found the λ plateau at γ 0.99 and b10 found γ 0.999 at λ 0.98 — each one knob off the
same cell, never run together. b14's rollout 512 was the other lever on the λ 0.99 side. The ladder
tests whether the horizon gains add and what they cost.

**Learned.** γ 0.999 on λ 0.99 adds density and adds drawdown: 36.3% against 27.3 (a lean, not a
separation), above both parents, with 3.6% of evals below 50 against 0.8 — a fifth of the way to
γ 1.0's regime. Rollout 512 on top does *not* add: 32.7%, below rung 1 on everything but stability,
where it halves the share below 80. The prediction "densest cell yet" is falsified — b14's +11 pp was
measured on γ 0.99, and on γ 0.999 the longer rollout trades density for stability instead. Best at
depth 99.1 /30k.

## b21 — chase-safe shaping

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | `SNEK_CHASE_SAFE_SHAPING` 0 / 0.05 / 0.2 at gate 75; gate 0 / 60 / 85 at dose 0.1 |
| cells × seeds | 6 × 4, 50M |
| control | `b7aa`-`b7ad` |
| predicted | late onset with shaping off; gate 85 worse than 75 — **falsified** |

**Why.** The one reward knob every other batch held fixed, and snek2's record lever (batches 28-29:
the gate mattered more than the dose). Shaping off had never reached 95/100 in DQN's 3M steps; for
PPO the question was whether it did anything at all. Run last for that reason.

**Learned.** A no-op for PPO: dose 0 to 0.2 and gate 0 to 85 all inside the base's noise on density
(16.3-18.2% against 17.3) and best30, every cell a little more stable than the base, and shaping
*off* the most stable of them (3.6% of evals below 80). Both spec predictions — a late onset with
shaping off, worse at gate 85 — falsified. Shaping off is a free simplification; it has not yet been
adopted.

## b20 — collect lanes

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | `SNEK_COLLECT_ENVS` 32 / 64 / 256 / 512 at rollout 128 |
| cells × seeds | 4 × 4, 50M |
| control | `b7aa`-`b7ad` |
| predicted | 32 lanes worse than rollout 32, because episode diversity is lower — **falsified** |

**Why.** The other half of the update batch: lanes × rollout is transitions per update, so 256 lanes
at T 128 is the same batch as 128 lanes at T 256 from twice the episodes at half the depth. Read
beside b14 to separate depth from batch size and episode diversity.

**Learned.** A throughput knob, not a learning knob: density within noise at every value (17.3-20.7%),
best30 97.75-98.35 with the base at the bottom, row counts scaling with the update cadence rather than
the policy. The prediction that 32 lanes would be worse than rollout 32 is falsified. What moves is
stability — 512 lanes has 0.0% of evals below 50 and 2.2% below 80 against 6.2 — and against b14 the
direction is clear: rollout 512 read +11 pp on its base, 512 lanes +0.5 on this one, so **the
rollout's gain was depth**. 128 stays for speed.

## b19 — the switches

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | advantage normalisation off; `mse` value loss instead of huber; Adam ε 1e-5 and 1e-8; vf coef 0.1 and 1.0 |
| cells × seeds | 6 × 4, 50M |
| control | `b7aa`-`b7ad` |
| predicted | both switches add collapses; Adam ε and vf coef inert — **falsified in part** — the switches removed collapses, the rest held |

**Why.** Four knobs with two or three sensible values each and no curve to map, gathered into one
batch as a check that no default was silently costing something. Both switches were predicted to
*add* collapses: an unnormalised advantage lets a +100 terminal dominate a minibatch, and squared
error lets it dominate the critic.

**Learned.** Both switches removed collapses instead — 6.2% of evals below 80 to ~1% — which locates
the base's collapses in the critic and the advantage scale rather than the policy step. `mse` is the
most stable cell at this base (0.97%) *and* 5 pp denser (22.2%) with the highest stage-A ≥98 share;
advantage normalisation off is as stable and 4 pp short. Adam ε and the vf coefficient are within
noise, so the *form* of the value loss is the lever. `mse` entered the corner grid and is in every
base since b23.

## b18 — gradient-norm clip

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | `SNEK_PPO_GRADIENT_CLIPPING` 0 (off) / 0.1 / 0.25 / 1.0 / 2.0 / 5.0 |
| cells × seeds | 6 × 4, 50M |
| control | `b7aa`-`b7ad` at 0.5 |
| predicted | off is the worst arm if collapses come from huge gradients, nothing changes if they are policy-level — held — the nothing-changes branch |

**Why.** Never swept, and a direct test of the tail-update hypothesis: b4's worst updates had
approx-KL 146x the median. If collapses come from rare huge gradients, clipping off should make them
worse and tightening should make them rarer, without touching the epoch count.

**Learned.** A no-op from off to 5.0: every cell inside the base's noise on density (16.7-19.5%
against 17.3) and best30; off reads the base's density to the decimal and is more stable. The
collapses are policy-level, not rare huge gradients. The more useful result is a calibration: 0 and
5.0 are both effectively no clip and differ by 3.8 pp on the share of evals below 80, so **that
column's noise at n=4 is ~4 pp** — which put b21's "a little more stable" and most of b20's
stability reading inside the noise.

## b17 — clip, and the anneals

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | static clip 0.05 / 0.1 / 0.15 / 0.3 / 0.4; clip annealed to 0.02, 0.005, 0.001, 0.1, from 0.1 and from 0.4; lr annealed to 0 and to 3e-5; both anneals; clip anneals to 0.02 and 0.001 over 80% of the cap **then held** for the last 10M (`SNEK_PPO_ANNEAL_FRACTION`, written for this batch) |
| cells × seeds | 16 × 4, 50M |
| control | `b7aa`-`b7ad` at clip 0.2 |
| predicted | the anneals win on collapse share; a tighter static clip means fewer collapses — **falsified in part** — anneals held, a tighter static clip had *more* collapses |

**Why.** The trust region has been PPO's headline knob since the paper, and the Atari recipe anneals
both clip and lr to zero. Prediction: the anneals win on collapse share; the question is what they
cost on density. The hold cells ask whether the endgame wants to be *at* the small clip rather than
still descending toward it.

**Learned.** The static clip is flat from 0.05 to 0.2 and worse above, and loosening it *reduces*
collapses — the reverse of the prediction. Every anneal beats the base on best30, but the plain anneals
only match it on density whatever the floor; the two hold cells read +6-7 pp (23.5-24.3%) with the
batch's most `hof5000` candidates, and lr → 0 does the same (+6). **The endgame wants a small trust
region held for a long time.** `b17cl-clipanneal001hold80-seed4` @11386880 — a checkpoint at 11.4M —
read 99.50 /30,000 and is in the HOF.

## b16 — target KL

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | `SNEK_PPO_TARGET_KL` 0.003 / 0.005 / 0.008 / 0.01 / 0.013 / 0.015 / 0.02 / 0.03 / 0.04 / 0.05 |
| cells × seeds | 10 × 4, 50M |
| control | `b7aa`-`b7ad` at 0 (off) |
| predicted | a small effect at most, 0.01 if any — held |

**Why.** Early-stops the epoch loop when the policy has moved too far. From b4's approx-KL
distribution, 0.02 fires on ~1% of updates, 0.005 on ~25%. If collapses come from the tail of large
updates, cutting the tail should show. 0.04 and 0.05 were kept as null checks predicted identical to
the control.

**Learned.** A no-op at 4 epochs: no column trends along the sweep and no cell leaves the reference's
noise. The stop does fire — same-seed arms at 0.03-0.05 diverge from one another during onset, where
approx-KL is large — and changes nothing stage B can see; with 4 epochs there are three stopping points
and the clip already bounds the step. Off stays. The null cells earned their place: they were what
caught b15-b21's base mix-up (below).

## b15 — entropy coefficient

| | |
|---|---|
| base | PPO reference (λ 0.98) |
| varies | `SNEK_PPO_ENTROPY_COEF` 0 / 0.001 / 0.003 / 0.005 / 0.02 / 0.03; anneals 0.1 → 0.001, 0.03 → 0.001, 0.01 → 0.001, 0.01 → 0 |
| cells × seeds | 10 × 4, 50M |
| control | `b7aa`-`b7ad` at 0.01 |
| predicted | fewest collapses at 0.001; the default moves if that costs no density — held — and the condition for moving the default failed |

**Why.** b3 at n=1 had the share of evals below 80 running 2.9% at 0.003, 12.2% at 0.01, 45.6% at
0.03 — the one stability signal monotone in both directions — and b8 had run 0.003 and the anneal on
the wrong shape. If lower entropy was monotone down to 0.001 with no density loss, the default would
move.

**Learned.** Density and stability trade monotonically: 0 and 0.001 are the most stable arms trained
to that point (1.7-1.9% below 80) at half the density (8.7-11.7%); 0.03 is the densest cell (24.1%)
with 14% of evals below 80. The anneals average their endpoints, and the hard-explore 0.1 → 0.001 is
the worst cell in the batch (23.8% below 80). The condition for moving the default was falsified;
0.01 stays. `b15ay` read 99.4 /30k, a candidate not a record.

**‡ b15-b21 ran at λ 0.98, not the re-based 0.99.** They were generated from b7's frozen base after
b11-b14 had been re-based, so their reference is `b7aa`-`b7ad` (17.3%), not b9's λ 0.99 cell (27.3%).
Read against the wrong base every cell looked 8-14 pp short; the null-check cells caught it. The user
kept b18-b21 at 0.98 so b15-b21 stay one comparable set.

## b14 — rollout horizon

| | |
|---|---|
| base | λ 0.99 |
| varies | `SNEK_PPO_ROLLOUT` 32 / 64 / 192 / 256 / 512 / 1024 at 128 lanes |
| cells × seeds | 6 × 4, 50M |
| control | `b9bw`-`b9bz` at 128 |
| predicted | optimum 128-256; 1024 too few update rounds by 50M — **falsified** — 512 is the peak and 1024 arrives |

**Why.** T sets transitions per update (128 × T) and how far GAE can see before it bootstraps; 32
truncates a 50-step horizon. b3's rollout 64 had lost 2.5 pp at n=1. Note T 1024 writes 8x fewer
checkpoints, so shares are compared, not counts.

**Learned.** Density rises with T to 512 (38.3% against 27.3, +11 pp) and stability keeps improving
through 1024 (2.9% below 80 against 6.4); 32 is the short side of the plateau (13.0%, 17% below 80)
and 64 matches the base on everything but density. 512 joined the corner grid, where b22 found its
gain does not survive γ 0.999, and b20 showed the gain was depth rather than batch size.

## b13 — minibatch

| | |
|---|---|
| base | λ 0.99 |
| varies | `SNEK_PPO_MINIBATCH` 32 / 64 / 128 / 192 / 384 / 512 / 1024 / 2048 |
| cells × seeds | 8 × 4, 50M |
| control | `b9bw`-`b9bz` at 256 |
| predicted | 128 the likeliest alternative default — **falsified** — 128 is 8 pp short |

**Why.** Moves gradient noise and gradient steps per epoch at once (512 steps per epoch at 32, 8 at
2048), read beside b12 to separate step count from data reuse. 128 was the predicted alternative
default; b3's 1024 had been the worst non-diverged arm.

**Learned.** A plateau at 256-512 (26.9-28.2%, 512 holding the batch's one 100/500 row) with density
lost on both sides: 128 — the predicted alternative — is 8 pp below on every seed, 32 is 6.8% with 15%
of evals below 80, and 1024-2048 arrive but noisily. Stability is best at 64-192. Nothing moved the
base; 512 was the only cell that might ride along, and it did in the horizon anneal.

## b12 — epochs

| | |
|---|---|
| base | λ 0.99 |
| varies | `SNEK_PPO_EPOCHS` 1 / 2 / 3 / 5 / 6 / 7 / 8 / 10 / 12 / 16 |
| cells × seeds | 10 × 4, 50M |
| control | `b9bw`-`b9bz` at 4 |
| predicted | 3-4 the top; 8 shows b4's collapse pattern — **falsified in part** — 8 did not collapse |

**Why.** The axis that had moved most: gradient steps per transition was b3's one monotone knob, and
at n=8 8 epochs lost to 4 at both network shapes (b4 vs b6, b5 vs b7). Prediction: 6 is where b4's
collapse pattern starts and 8 collapses.

**Learned.** 3-4 is the top (26.5-27.3%); every epoch past 4 costs 2-5 pp of density; 1 epoch is the
*least* stable cell (4.2% below 50, 19% below 80), the reverse of its prediction. Stability is best at
6-8 — 8 epochs did not collapse, so b4's collapses were its `fc (200,100)` net's, not the epoch
count's — and breaks at 12-16 (42% of evals below 80 at 16). Base stays at 4.

## b11 — learning rate

| | |
|---|---|
| base | λ 0.99 |
| varies | `SNEK_PPO_LEARNING_RATE` 4e-5 / 1e-4 / 1.5e-4 / 2.5e-4 / 5e-4 / 8e-4 / 1e-3 / 2e-3 |
| cells × seeds | 8 × 4, 50M |
| control | `b9bw`-`b9bz` at 3e-4 |
| predicted | a plateau around 3e-4; 8e-4 and 1e-3 collapse visibly — **falsified in part** — they were the most stable cells |

**Why.** b3 had said a low lr did not buy stability (1e-4 had *more* evals below 80) and that 3e-3
diverged; this locates both ends at four seeds. Prediction: 8e-4 and 1e-3 collapse visibly.

**Learned.** A plateau, not a lever: 1e-4 through 5e-4 sit at 21-31% within seed noise of the
reference's 27.3; 4e-5 is the slow end (2.2%) and 2e-3 the cliff (6.7%). The surprise is the sign of
the stability effect — 8e-4 and 1e-3 were predicted to collapse and are the two most stable cells
(4.6-4.9% below 80), with a lower ceiling. 2.5e-4 is the best cell (30.9%) and went into the horizon
anneal; 3e-4 stays the default. `b11ag` @33243136 read 99.4 /30k.

## b10 — discount γ

| | |
|---|---|
| base | PPO reference (λ 0.98; queued before the λ re-base, kept so it stays a clean one-knob sweep) |
| varies | `SNEK_DISCOUNT` 0.70 / 0.80 / 0.85 / 0.90-0.98 by 0.01 / 0.995 / 0.9975 / 0.999 / 1.00 |
| cells × seeds | 16 × 4, 50M |
| control | `b7aa`-`b7ad` at 0.99 |
| predicted | optimum 0.99-0.995; γ 1.00 the latest onset and unstable — **falsified in part** — monotone to 0.999; the γ 1.00 half held |

**Why.** b3 at n=1 had called γ 0.995 the stability candidate and 0.9975 was snek2's record DQN γ.
γ also sets the shaping discount, so the dense reward moves with it, correctly. The low end asks how
short a value horizon can still reach the endgame; γ 1.00 asks what an undiscounted critic does.

**Learned.** Below γ 0.94 the endgame is unreachable at 50M — 28 arms produced no screened checkpoint
— and it is γ itself, not the GAE horizon: b9's λ 0.90 at the same advantage horizon reached 96.75.
From 0.96 to 0.999 density is monotone (0.9 → 30.7%), the default again mid-ramp; drawdown rises with
it. **γ 1.00 is different in kind**: 44% of post-competence evals below 50, the policy oscillating for
its whole run — and the richest checkpoints in the batch. `b10ck` @30523392 read 99.65 /30k, the
record until b27.

## b9 — GAE λ

| | |
|---|---|
| base | PPO reference |
| varies | `SNEK_PPO_GAE_LAMBDA` 0 / 0.5 / 0.8 / 0.85 / 0.90-0.97 by 0.01 / 0.99 / 0.995 / 0.999 / 1.00 |
| cells × seeds | 16 × 4, 50M |
| control | `b7aa`-`b7ad` at 0.98 |
| predicted | a broad flat top at 0.95-0.99; λ 1.0 the worst arm — **falsified** |

**Why.** The first batch of the one-knob sweep ([`../plans/hyperparam-sweep.md`](../plans/hyperparam-sweep.md)):
every PPO knob at four seeds off b7's winning cell, with `b7aa`-`b7ad` as a free control at the same
cap. Prediction: a broad flat top at 0.95-0.99; λ 1.0 the worst arm (b3 said so at n=1).

**Learned.** The default was not the top. Density is monotone to 0.99 — 17.3% at 0.98 to 27.3%, every
0.99 seed above every 0.98 seed — and flat from 0.99 to 1.00 (25.6-29.5%). Stability moves the other
way: λ 1.00 has 2.1% of evals below 50, b4's collapse. `sef` ranked the sweep backwards a third time.
**λ 0.99 became the default** (a third of λ 1.0's drawdown, the tightest seed spread) and b11-b14 were
re-based before they ran. `b9ch-lam999-seed4` @47251456 read 99.30 /30k, the first entry above 99 at
depth. b3's "λ 1.0 loses" is inverted.

## b8 — the stability knobs, on b4's config

| | |
|---|---|
| base | b4's config: PPO reference with `fc (200,100)` and 8 epochs |
| varies | entropy 0.003; entropy 0.01 → 0.001 annealed over the cap; `target_kl` 0.02; λ 0.95 |
| cells × seeds | 4 × 4, 100M |
| control | b4 itself, truncated to 100M |
| predicted | at least one stability knob fixes b4's collapse and lifts it — **falsified** — every knob cut the drawdown, none lifted density |

**Why.** "What fixes b4's collapse." Four candidates that each had a stability signal: entropy 0.003
from b3, the anneal and `target_kl` because neither had ever been exercised (b4 ran 8 epochs in all
97,656 updates), λ 0.95 from b3's two λ arms. Held on b4's shape so the control matches, knowing b7
was about to show that shape to be the wrong one.

**Learned.** Every knob cut the drawdown (8.4% below 50 to 2.2-5.9%) and none beat the control on
record density; λ 0.95, the best on stability, was worst on density. Both new knobs were confirmed
live by the smoke tests. No HOF candidate: one row of 135 above 98.73 at 5,000. The lesson that
shaped everything after: **steadying the curve and banking record checkpoints trade off**, and
epochs and shape were the lever, not these.

## b7 — network shape

| | |
|---|---|
| base | PPO reference |
| varies | `SNEK_FC_LAYERS` (320,) / (200,100) / (100,200,100) / (100,100) / (200,100,50) / (160,160) / (300,100) / (400,200) |
| cells × seeds | 8 × 4, 50M |
| control | `fc (320,)` is the reference cell |
| predicted | b3's order: `fc (300,100)` first, `fc 320` last of the three — **falsified** — inverted |

**Why.** The network-shape test the docs had called for since b3: one knob, matched epochs, matched
budget. b3 at one seed had put `fc (300,100)` first and `fc 320` last of those three, and b4-b6 had
confounded shape with epochs.

**Learned.** `fc (320,)` wins at 17.3% density, every seed beating every seed of five of the seven
other layouts (p=0.029); `fc (400,200)` is last at 5.1%, so width past 320 hurts. b3's ranking
inverted. At 5,000 episodes the win is volume, not quality — the layout means span 0.39 pp while
candidate counts span 31 to 174. And `strong_eval_fraction` ranks the layouts *backwards* (Spearman
−0.79): the stage-A ≥98 rate is the screen instead. `fc 320` is also snek2's shape, so a champion's
weights still convert. Became the base for b9-b21.

## b4 — `fc (200,100)` + 8 epochs

| | |
|---|---|
| base | PPO reference |
| varies | b3's two best single knobs stacked: `fc (200,100)` and 8 epochs |
| cells × seeds | 1 × 8, 200M |
| control | b5 and b6, which each carry one of the two knobs |
| predicted | b3's two best knobs stack to the best 8-seed arm — **falsified** — the weakest |

**Why.** b3 had ranked epochs 8 and `fc (200,100)` first and second at n=1; the natural next arm was
both together at eight seeds and length. Also completes the 2×2 with b5 and b6.

**Learned.** The weakest of the three 8-seed batches: 7.3% density against b6's 12.9 and b5's 9.6,
best30 below both on every seed, and the collapses that b8 was built to fix (8.4% of evals below 50
at 100M). Holding shape, 4 epochs beats 8; holding epochs, `fc 320` beats `fc (200,100)` — **the two
knobs interact negatively, so a one-knob-at-a-time ranking licenses no stacking**. One row above 98.73
at 5,000, none at 99. b3's epochs ranking retired.

## b5, b6 — `fc 320` + 8 epochs, and `fc (200,100)` + 4 epochs

| | |
|---|---|
| base | PPO reference |
| varies | b5: 8 epochs at `fc (320,)`; b6: `fc (200,100)` at 4 epochs |
| cells × seeds | 1 × 8 each; cap 400M, stopped at 255-271M (b5) and 215-231M (b6) |
| control | each other, imperfectly |
| predicted | none registered; b6's 500-episode lead was read as real — — (the lead vanished at 5,000) |

**Why.** b3's two leads — the epoch count and the two-layer net — each run at eight seeds and length
to see whether either held. Meant as a network-shape comparison, which it was not: the two differ in
two knobs.

**Learned.** b6 led at 500 episodes (12.8% against 9.6) and the lead vanished at 5,000: identical means
(97.80 both), b5 ahead on champion-level rows (29 to 20) and on the top checkpoint. **A 500-episode
ranking of two close batches did not survive 5,000** — the finding behind the `hof5000` pass. Running
b5 longer bought nothing; b6 longer paid modestly. `b5h` @9027584 (98.96 /30k) and `b6b` @133120000
(98.73) were the first HOF entries, and `b5h` beat snek2's champion at matched depth.

## b3 — the PPO tuning sweep

| | |
|---|---|
| base | PPO reference, freshly defined |
| varies | one knob per arm: lr 1e-4 / 5e-4 / 1e-3 / 3e-3; γ 0.995 / 0.9975; λ 0.95 / 1.0; entropy 0.003 / 0.03; `fc` 200 / 500 / (200,100) / (300,100); rollout 64; minibatch 1024; epochs 8 |
| cells × seeds | 15 × 1 (seed 1), 10M; `b3b`-`b3d` stopped at 3M |
| control | `b3a`, the reference |
| predicted | from the gate arm: PPO behind DQN at matched budget, and the lr too low — **falsified**, both |

**Why.** The first PPO batch after the gate arm: find out which knobs move anything before spending
seeds. A tuning pass, not a gate — nothing is seed-matched, so no row supports a between-config claim
on its own.

**Learned.** No winner: nine arms within 0.8 pp on best30 and three metrics giving three orderings.
One axis moved monotonically — gradient steps per transition (minibatch 1024 89.7, reference 96.6,
epochs 8 97.2). The lr is peaked at 3e-4; 1e-3 and 3e-3 are worse. PPO's record density is 11.6x
DQN's at the same protocol, though every 500-episode high fell 1.3-2.0 pp on re-measure. The
two-layer lead and the epochs-8 lead were both later inverted at four seeds (b7, b12) — the cost of
n=1.

## b2 — snek2's record config on the torch stack

| | |
|---|---|
| base | DQN defaults |
| varies | the five knobs of snek2's batch 29: IS weights off, target update 1000, γ 0.9975, food-distance reward 0, chase-safe shaping 0.1 at gate 75 |
| cells × seeds | 1 × 4, 3M counted steps |
| control | b1, and snek2's b29 / b41 / b47 seed for seed |
| predicted | the phase-3 gate met on snek2's record config — held |

**Why.** The phase-3 gate of the port, re-run on the configuration snek2 actually set records with,
after b1 gated on the wrong one. Five knobs differ, not the two b1's write-up suggested — found by
reading snek2's b47 spec rather than its results summary.

**Learned.** The gate is met (`b2d` best30 96.8, 52 evals at ≥95/100; b1 had none), so the five knobs
are the whole b1-b2 difference — snek2's own batch-28/29 finding reproduced. One seed carries the
batch, as in every snek2 run of this config; the carrier is a coin. And **a snek3 counted step is four
game moves where snek2's was one**, so "b2 leads b47 at a quarter of the budget" was a units artefact:
at matched work b2d *matches* b47c. Every cross-era step comparison since reads `transitions`.

## b1 — the DDQN baseline

| | |
|---|---|
| base | DQN defaults |
| varies | nothing |
| cells × seeds | 1 × 4, 3M counted steps |
| control | snek2's baseline-class runs |
| predicted | the phase-3 gate met at snek3's defaults — **falsified** |

**Why.** The port's phase-3 gate: does the torch stack learn at all, at snek3's own defaults, before
anything is tuned.

**Learned.** No checkpoint in any arm reached 95/100, so stage B measured nothing — the honest result,
not a failure. Every arm was still climbing at the cap (b1d 0 → ~80% perfect, monotonically). It was
the wrong batch to gate on: snek3's defaults are shaping off and IS weights on, the class snek2 also
found far from records, and the gate's wording did not say whether it meant a trailing rate or a
single checkpoint. b2 fixed both.

## Not batches

The PPO gate arm `ppo-smoke` (508k transitions, untuned defaults, 1% perfect), the converted snek2
policies `b44a-import` and `b45a-import` (the port's phase-1 and phase-2 fidelity checks), and the
2026-08-30 parallelism sweep (eval workers and shards; the eval side answered, the training side did
not) are in [`results.md`](results.md) and [`findings.md`](findings.md). None is seed-matched against
anything here.
