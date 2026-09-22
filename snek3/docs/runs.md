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

- **Group A runs on the local plumbing** (decided 2026-09-17): b35 and b36 showed the papers' recipe does not reach
  competence in 10M moves on this game for DQN or C51, so rows A2-A4 are queued on b35's local cell (b37: C51 and QR-DQN;
  b38: IQN neutral and CVaR-trained). b37 closed 2026-09-18: the heads buy the hold, not the ceiling. b38 closed
  2026-09-19: IQN at N = N′ 8 plateaus at 55-65%, no checkpoint reached stage B, the CVaR-trained cell lower and one seed
  dead; the CVaR hand pass is moot. b39 closed 2026-09-20: FQF at the same N 8 climbs where IQN flattened (three seeds
  cross 90) but never holds and never reaches 97 -- the sampling was b38's ceiling, N 8 is still short of N 32. Still open:
  b40 closed 2026-09-20: Munchausen's target is QR-DQN's hold made slightly tighter and DQN's drawdowns made slightly
  shallower, the same 98.0 / 97.8 best rows, no `hof5000` candidate -- the target is not where the ceiling is either, so
  Group A closes with seven rungs at 96.6-98.0 against PPO's 99.8 best30. Still open, and both now second-order: FQF at N 16
  or 32 in waves of 4 (a steadier plateau at best, not the ceiling), and whether the paper cell arrives at its full 50M-move
  budget (~18 h an arm).
- **Group B's temperature target and hold** (b41 queued 2026-09-20, its paper cell stopped dead the same day; b42 queued 2026-09-20): the
  2019 paper's 0.98 · ln|A| forces a near-uniform policy on three actions and α runs away, confirmed four times over in b41a-d. b41's local
  cell at 0.1 gave the fastest onset in the project (best30 89-93 by 60k-78k steps) and then drifted to 67-72% by 1M -- the hold, not the
  onset, is Group B's question now, and b42 puts Zhou et al.'s two hold mechanisms on exactly that cell. Whether the target should track
  PPO's 0.001-0.009 nats instead, and whether the Q-clip binds at all at lr 1e-5 (0.8% of samples at most in the gate arm), stay open.
- **Group D's reset probe** (b43 queued 2026-09-20 on b40's close): whether BBF's shrink-and-perturb resets hold where b40's M-QR-DQN
  drifts. If they do, `SNEK_RESET_*` goes to every later value row; if not, late drift is not a plasticity problem here -- and the anneal wave
  (`SNEK_RESET_ANNEAL_*`, built 2026-09-20) runs before the row closes on a negative. **b44** (written 2026-09-20, awaiting the go-ahead) is a
  different question beside it: BBF's whole recipe as written (`algos/bbf/`), four seeds at the paper's 100k-move budget, to read how the
  algorithm itself does on Snake.
- **Queue what does not depend** (2026-09-20, the user's rule): a batch waits for another only when it reads that batch's numbers to be
  specified or judged. b42 and b43 were queued together with b41 still live for that reason; Groups C, E, F, G and H have nothing built yet
  and are the next implementation work, in the series' order.
- **`rp` annealed to zero after onset.** b34's per-reversal penalty gave the fastest onset and the best stability on the
  `hist8` base and cost 5 pp of plateau density; a penalty that decays to 0 by ~10M would say whether the two can be
  separated. The reversal-rate-by-fill measurement the b34 prediction named is still owed.
- **What the perfect-game reward's onset effect is made of.** b33 says W below 100 slows the arrival and W above it
  buys nothing; whether the slow onset at 1000 is the huber critic (δ 1) learning the terminal jump — testable
  with `mse` at 1000 — or the normalised advantages, is open. The starve/death split stage B does not record.
- **A clean test of the PPO paper's optimiser schedule**: b29 confounded lr/clip → 0 with removing the
  horizon anneal. The unconfounded arm keeps γ/λ → 0.999 and adds lr/clip → 0 on top.
- **Warm starts.** b32 is the first batch to start from a checkpoint (`SNEK_INIT_FROM`, step 0). If its
  arms hold the 99.8 plateau from their first eval, any knob can be tried *on the converged policy* at
  100M instead of 200M from scratch, and a collapse there is a verdict on the knob rather than on the seed.
- **b32's record.** `b32g` @62423040 stood at 29,967 /30,000 when the pass closed (2026-09-13) and read 29,957 on a second
  30,000 at seed 13; it is in the Hall of Fame as the record. The open question is whether the gain is the warm start's
  extra converged training or the horizon reaching 1.0 — b30 says the horizon alone does nothing from scratch, and seven of
  the eight warm starts moved nothing, so the next batch should separate the two (a warm start that keeps 0.999, and the
  same eight sources re-run on other seeds).
- **The 99.82 /30k ceiling** was reached by b28 (200M hold), b30 (horizon 1.0) and matched by nothing at
  100M from scratch; b29 and b31 both fell short. Whatever moves it is not steps, horizon, optimiser decay or
  the value loss.
- **Should `SNEK_OBS_HISTORY=8` become the default.** b27 says yes; nothing has run against it yet.
- **Next sweeps on the `hist8` base**: a `hist16` cell, and step penalties above 0.01 (0.02, 0.05) —
  b26's curve never turned.
- **b24/b25's unmeasured `hof30k` arms**, and a fresh 30,000 on `b25a` @106168320, need the old
  30-value observation era to load.

## At a glance

| batch | varies | base | cells × seeds | cap | prediction | result in one line |
|---|---|---|---:|---:|---|---|
| [b44](#b44--bbf-the-papers-recipe-on-snake) | the algorithm: BBF as written (`SNEK_ALGO=bbf`: ×4 dueling C51 `fc 1280,2048`, replay ratio 8, batch 32, AdamW 1e-4 wd 0.1, EMA τ 0.005, SPR K 5 weight 5, resets every 40k gradient steps with n 10 → 3 and γ 0.97 → 0.997 over 10k, ε 1 → 0 over 2,001 moves, PER 0.5, 1M replay) | none: the paper cell alone; hist8, b2 reward, step penalty 0.01, shaping **off**, 1 lane | 1 × 4 | **100k moves** (= 100k steps at 1 lane, ~800k gradient steps) | registered | — |
| [b43](#b43--bbf-style-resets-on-b40s-m-qr-dqn-cell) | BBF-style shrink-and-perturb resets, the reset alone (no replay ratio 8, wider net, AdamW, EMA target, SPR or within-cycle anneal) (`SNEK_RESET_INTERVAL` 600k / 2.4M gradient steps, `_ALPHA` 0.5, `_STOP_AFTER` 10.5M) | b40's M-QR-DQN cell (`b40e`-`h`) | 2 × 4 | 3M steps | registered | — |
| [b42](#b42--revisiting-discrete-sac-paper-cell-beside-b41s-local-cell-with-the-fixes) | Zhou et al. 2022's fixes: per-state entropy-penalty 0.5, double average Q with Q-clip 0.5. Paper cell as written (lr 1e-5, α 0.05 fixed, batch 64, 1e5 uniform, 0.1 updates a move, Polyak 0.005, 3-step, 2 × 512, MSE) / b41's local cell plus the two fixes | PPO's reward, hist8, 16 lanes / `b41e`-`h` | 2 × 4 | 3.125M steps = 50M moves | registered | — |
| [b41](#b41--discrete-sac-paper-cell-beside-local-cell) | the algorithm: discrete SAC (`SNEK_ALGO=sac`). Paper cell as written (target entropy 0.98 ln 3, batch 64, 1M uniform, 0.25 updates a move) / local cell (target 0.1 ln 3, batch 128, PER 0.6, 100k, 0.5 updates a move, target every 8) | PPO's reward, hist8, `fc 320`; 16 lanes | 2 × 4 | 3.125M steps = 50M moves | paper half held; local cell live | paper cell stopped at 0.57M counted steps: zero perfect games on all four seeds, α 2.5 × 10⁸, entropy pinned at the 0.98 ln 3 target (1.077 nats); the local cell (0.1 ln 3) started 12:49 on the desktop |
| [b40](#b40--munchausen-on-the-local-plumbing-m-dqn-beside-m-qr-dqn) | the value target: Munchausen's log-policy reward term and soft target (`SNEK_MUNCHAUSEN_ALPHA` 0.9, `_TAU` 0.03, `_L0` -1) on DQN / on QR-DQN N 32 | b35's local cell / b37's QR-DQN cell | 2 × 4 | 3M steps | held (the M-DQN best row 97.8, 0.8 over the line) | the target is not where the ceiling is: M-QR-DQN is QR-DQN with a slightly tighter hold (onset 1.13-1.54M, 89-93 after onset with 0-3% below 80, 436 rows, best 98.0 against 346 / 98.0), M-DQN is DQN with shallower drawdowns (73-87 after onset, 13-59% below 80 against 12-76%; 100 rows, 93 of them one seed's, best 97.8 against 96.6). No arm reached 99.2; both passes empty. Group A closes with every rung at 96.6-98.0 |
| [b39](#b39--fqf-at-n-8-on-the-local-plumbing) | the head: FQF, 8 learned fractions (`SNEK_DIST_QUANTILES` 8) | b35's local cell, `SNEK_ALGO=fqf` | 1 × 4 | 2M steps | falsified on onset, held on the rest | not IQN's band: three of four seeds cross 90 at 1.25-1.65M (IQN at N 8 never did), best30 85 against 65, but none holds -- 71-82 after onset, 27-84% of evals below 80 -- and no checkpoint reached 97, so stage B is empty. Eight learned fractions beat eight sampled ones and still trail N 32 fixed |
| [b38](#b38--iqn-risk-neutral-beside-trained-under-cvar-025-on-the-local-plumbing) | IQN (N = N′ 8) trained risk-neutral / under CVaR 0.25 (`SNEK_DIST_RISK_ALPHA` 0.25, `SNEK_DIST_RISK_TRAIN` 1) | b35's local cell, `SNEK_ALGO=iqn` | 2 × 4 | 2M steps | falsified | no checkpoint reached 97: the neutral cell climbs to 50% by 0.1-0.2M and sits at 55-65% to the cap (max eval 82); the CVaR-trained cell 40-54%, one seed dead from 0.9M. Zero stage-B rows, the first value batch with none. FQF at the same N 8 (b39) is drawing the same band |
| [b37](#b37--c51-and-qr-dqn-on-the-local-plumbing) | the head: C51 (51 atoms) / QR-DQN (N 32) | b35's local cell | 2 × 4 | 3M steps | held on the plateau, split on onset | both heads hold 88-93 after onset where DQN oscillates at 72-87 (evals below 80: C51 1-6%, QR-DQN 0.3-7%, DQN 12-76%); C51 reaches 90% at 0.27-0.34M, QR-DQN not until 1.1-1.8M; best rows 96.8 / 98.0 against 96.6, no `hof5000` candidate. The head buys the hold, not the ceiling |
| [b36](#b36--c51-stability-two-supports-on-the-paper-cells-plumbing) | C51's support: 51 atoms / 101 atoms on [-10, 110] | b35's paper cell, C51 head, Adam 2.5e-4 | 2 × 2 | 10M moves | registered | — |
| [b35](#b35--dqn-the-value-familys-control-paper-cell-beside-local-cell) | the plumbing: the DQN-Adam paper cell (batch 32, 1M uniform replay, target 2,000 updates, ε linear 1 → 0.01, no shield, no fork) beside snek3's DQN defaults (PER, fork 4, shield, eval-driven ε) | hist8 observation and b27's reward, `fc 320` | 2 × 4 | 10M moves / 3M steps | falsified on local, held on paper | paper: 7-16% perfect at 10M moves, still rising, no stage B; local: 90% by 0.6-1.6M then a 71-88% oscillation, 45 stage-B rows, best 96.6, none at 98. The plumbing is the whole gap; neither is near PPO's 95% |
| [b34](#b34--zigzag-shaping-a-reversal-potential-beside-a-reversal-penalty) | `SNEK_ZIGZAG_SHAPING` 0.5 (potential, window 8) / `SNEK_REVERSAL_PENALTY` 0.5 (plain) | b27's `hist8`, verbatim | 2 × 4 | 100M | held for `zz`, falsified for `rp` | `zz` level with the base everywhere; `rp` the fastest onset and most stable cell on this base, 30k top level (`b34e` @3.6M, 29,954 /30k), but 90.1% density against 95.4 -- a standing penalty caps the plateau |
| [b33](#b33--the-perfect-game-reward) | `SNEK_PERFECT_GAME_REWARD` 0 / 10 / 30 / 50 / 100 / 200 / 300 / 1000 | pen01 + hist8, anneal final at 25M | 8 × 4 | 50M | held on 0-300, falsified on 1000 | a monotone onset lever saturating at 100 (density 87.3 → 94.9, then 94.9-96.5); no collapse after 15M in any cell — 1000's drawdowns are a 15M onset; `win0` reaches 99.78 best30 and 9 rows at ≥99.8 /5k, so invariant 6 falls; nothing at 99.8 /30k from any cell at 50M, where b27's 100M had 9 |
| [b32](#b32--b28s-best-checkpoints-annealed-on-to-a-horizon-of-10) | warm start from b28's eight best /30k checkpoints; γ and λ 0.999 → 1.0 over 50M, held 50M | b28's converged values | 1 × 8 | 100M | falsified (the top moved) | stage B 99.8% in every window from the first eval; `hof30k` 92 full rows, 78 at ≥99.8; **`b32g` @62423040 at 29,967 /30k, the new record** (99.86 on a second seed), from a 61-63M plateau averaging the old ceiling; the other seven arms stayed on it |
| [b31](#b31--mse-value-loss-on-the-hist8-base) | `SNEK_PPO_VALUE_LOSS` mse | pen01 + hist8 | 1 × 8 | 100M | falsified | worse on this base: onset 51 vs 83% at 0-25M, 88.0% density, nothing reaches 30k at 99.8; only the stability columns keep `mse`'s old gain |
| [b30](#b30--the-horizon-annealed-to-10) | γ and λ finals 1.0, not 0.999 | pen01 + hist8 | 1 × 8 | 100M | falsified | level with `hist8` at every depth, no collapse; top `b30a` 29,946 /30k, the same 99.82 ceiling as `b28k` |
| [b29](#b29--lr-and-clip-annealed-to-zero) | lr 2.5e-4 → 0 and clip 0.2 → 0.001 over the whole cap; horizon fixed | pen01 + hist8 | 1 × 8 | 100M | falsified | worse in every window: 83.1% density, nothing through the 99.6 /5k gate; confounded with the missing horizon anneal |
| [b28](#b28--the-hist8-config-held-for-100m-more) | 100M more hold | pen01 + hist8, anneal over 25% | 1 × 8 | 200M | — | 97.5% density; 152 rows at 99.8 /30k against 9, top 99.82 — the plateau widens, the top does not move |
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

## b44 — BBF: the paper's recipe, on Snake

| | |
|---|---|
| base | none. `SNEK_ALGO=bbf` at its defaults, which are the paper's (Schwarzer et al. 2023, Table 1 and the released code): dueling C51 (51 atoms on [−10, 110]) on `fc 1280,2048` (the plan's MLP analogue of the ×4 IMPALA encoder plus the 2048 dense layer), replay ratio 8 at batch 32, AdamW lr 1e-4 ε 1.5e-4 wd 0.1, grad clip 10, EMA target τ 0.005, SPR K 5 weight 5 (transition width 256, projection 512), shrink-and-perturb resets every 40k gradient steps at α 0.5 with n-step 10 → 3 and γ 0.97 → 0.997 over the 10k gradient steps after each, ε 1 → 0 over 2,001 moves after a 2,000-move prefill, PER 0.5, 1M replay, 1 lane; hist8, the b2 reward with step penalty 0.01 and the potential shaping **off** (the sample-time anneal keeps the reward/discount coupling only with it off) |
| varies | nothing: one cell, `bbfpaper`, 4 seeds |
| cells × seeds | 1 × 4, seeds 1-4 pinned to the letter |
| cap | **100k moves**, the paper's regime (at 1 lane, 100k counted steps and ~800k gradient steps; about nine hours of learning an arm at the laptop's 25 gradient steps/s, more on the desktop) |
| control | none in-family. b35's paper cells (DQN-Adam, C51 at 10M moves: 7-16%) and b43 give the shape of a value paper cell here; BBF's claim is 100× fewer moves |
| predicted | registered 2026-09-20 by the agent: the stage-A trace saw-tooths at every reset (every 5k moves, 20 in the run) and the perfect rate at 100k moves stays under 20% on every seed, with no checkpoint at 97 and stage B empty; the SPR loss falls within the first cycle and stays low. If a seed reaches 50% the recipe has done in 100k moves what b35's paper cells could not in 10M, which would be the finding |

**Why.** The user's ask (2026-09-20): a read on how BBF itself does on Snake, separate from Group D's reset probe (b43, the reset alone
on a tuned cell) -- the plan had dropped the paper cell because SPR and the CNN had no meaning here, and the package `algos/bbf/` now
carries the recipe whole with the MLP analogue stated (`plans/algoExploration/d-data-efficiency.md` §7). Read for the shape first:
whether anything is learned in 100k moves, what each reset costs and recovers, whether the anneal's early myopic cycles show. Gates: 25
tests and 18 / 18 mutants, a smoke with resets every 2,000 gradient steps that checkpointed, resumed with its buffer and restored through
`evaluate.py`; the whole suite passes. Waits for the go-ahead to commit, deploy and push.

## b43 — BBF-style resets on b40's M-QR-DQN cell

| | |
|---|---|
| base | `b40e`-`h`'s spec verbatim: QR-DQN N 32, κ 1, Munchausen α 0.9 τ 0.03 l₀ −1, the local DQN plumbing (fork 4, shield, PER 0.6, target every 8, replay ratio 1, batch 128, lr 1e-5), hist8, `fc 320`, 3M counted steps (~12M gradient steps at ~4 a step) |
| varies | the reset cadence. **reset600k** (`b43a`-`d`): `SNEK_RESET_INTERVAL=600000`, ~20 cycles, the paper's count. **reset2400k** (`b43e`-`h`): `2400000`, ~5 cycles. Both `SNEK_RESET_ALPHA=0.5` (the trunk pulled halfway to a fresh init, the quantile head re-initialised, the target copied, Adam cleared) and `SNEK_RESET_STOP_AFTER=10500000` (the last eighth reset-free) |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | 3M counted steps, b40's, so the reset cells read on b40's x-axis |
| control | `b40e`-`h`, the same cell without resets (89-93 after onset, 0-3% of evals below 80, 436 stage-B rows, best 98.0, no `hof5000` candidate). Judged on the **hold**: the perfect rate after each reset's dip and in the reset-free tail, drawdowns, `zero_since`; then the three passes |
| predicted | registered 2026-09-20 by the agent, from the plan: the 600k cell shows a visible dip after each reset and a higher late perfect rate than `b40e`-`h`'s; the 2.4M cell fewer, deeper dips. **What a negative can and cannot say** (scoped the same day after a review): b43 is the reset alone, without BBF's within-cycle n-step / γ anneal that lets a reset head relearn, so if neither cell holds the anneal wave runs before the row closes, and no b43 result is a verdict on BBF |

**Why.** Row D1, re-planned 2026-09-20 from "how many steps does BBF need" to the late-plasticity probe: every
Group A cell reached 88-94% and then drifted or never held, late drift is the shape of most of this project's
collapses, and BBF's shrink-and-perturb reset is the one mechanism in the series aimed at exactly that. **It is
not BBF**: the
rest of BBF (the ×4 net, replay ratio 8, AdamW, EMA target, SPR) is data-efficiency machinery and stays out
until the reset earns a second wave; the within-cycle n-step / γ anneal is that wave, if the cells dip but do
not hold. Queued on b40's close, the same day, since the base cell's numbers are in. Gates: smoke with resets
every 1,000 gradient steps checkpoints, restores and resumes with the count; `mut_resets.json` 10 / 10.

## b42 — Revisiting Discrete SAC, paper cell beside b41's local cell with the fixes

| | |
|---|---|
| base | `sac2` (`algos/sac/`): discrete SAC with Zhou et al. 2022's two fixes -- the **entropy-penalty**, β · ½ E_s[(H_old(s) − H(s))²] with H_old the collecting policy's entropy at that state, stored in the replay with the transition (rewritten to this per-state form 2026-09-20 after a review); and **double average Q with a Q-clip**, the target on avg(Q′₁, Q′₂) and the critic loss max((Q − y)², (Q′ + clip(Q − Q′, ±0.5) − y)²). PPO's reward preset, hist8, 16 lanes, γ 0.99, prefill 20k |
| varies | **sac2paper** (`b42a`-`d`): the paper's Table 3 as written -- Adam 1e-5 actor and critics, batch 64, 1e5 uniform replay, 0.1 updates a move, Polyak 0.005 every update, 3-step, α fixed 0.05, MSE, network **2 × 512**. Not the paper's: snek3's unclipped reward (the paper clips to ±1), the 26-value observation, 50M moves against 10M Atari steps. **sac2local** (`b42e`-`h`): `b41e`-`h`'s cell -- α auto toward 0.1 · ln 3, batch 128, PER 0.6, 100k, 0.5 updates a move, hard target every 8, lr 3e-4, 1-step, `fc 320` -- plus the two fixes and nothing else; the clip puts the critics on squared error where b41 used Huber |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter; the local cell shares seeds 5-8 with `b41e`-`h` |
| cap | 3,125,000 counted steps = 50M moves, b41's |
| control | `b41e`-`h` for the local cell (best30 89-93 by 59k-78k steps, then 67-72% perfect at 1.03M with a strong-eval fraction of 15-20%: the fastest onset in the project and no hold); PPO's hist8 strip; b41's paper cell for the paper cell (dead at 0.57M, α 2.5 × 10⁸) |
| predicted | registered 2026-09-20 by the agent: the paper cell reads a non-zero perfect rate -- the temperature is fixed, so b41's α runaway cannot recur -- but at lr 1e-5 its onset is late, after 1.5M if within the budget at all, its plateau below `b41e`-`h`'s and its Q-clip fraction near 0. The local cell has `b41e`-`h`'s onset (best30 above 85 by 0.1M) and the test is the hold: entropy within 0.02 nats of the 0.11 target, perfect rate after 1M above 80 against b41's 67-72, a higher strong-eval fraction at the same horizon, the clip binding on 1-5% of samples at 3e-4; still short of PPO's 95% stage-B density |

**Why.** Row B2 asks whether B1's result was the idea or the implementation, and b41's local cell has just
given it a sharp question: the fastest onset the project has seen followed by a drift the two 2022 fixes are
built to stop. The paper cell runs as written by the series' rule, expected weak (lr 1e-5; the gate arm's
clip bound on under 1% of samples). The plan's earlier second cell, α auto at the paper's 0.98 target, was
dropped after b41 showed that target degenerate on three actions; the local cell's α auto at 0.1 stands in,
and differs from `b41e`-`h` by the fixes alone, so the pair attributes. Gates 2026-09-20: `mut_sac.json` 16 / 16
after the penalty rewrite and the post-critic-step actor evaluation; 500k-move laptop arms of both cells log a
non-zero per-state penalty on every eval. Queued before b41 closes: nothing in it reads a b41 number to run.

## b41 — Discrete SAC, paper cell beside local cell

| | |
|---|---|
| base | discrete SAC (Christodoulou 2019, `algos/sac/`): a categorical actor (PPO's `QNet`-as-logits, the checkpoint), two Q critics with target copies, the temperature tuned by gradient from 1.0 toward a target entropy; off-policy from DQN's replay and collector with the fork and shield off, the agent sampling from π. PPO's reward preset, hist8, `fc 320`, 16 lanes, γ 0.99, Adam 3e-4, prefill 20k moves. **Critic loss Huber in both cells** -- the paper's is MSE; the departure was for the unclipped +100 terminal, and `SNEK_SAC_CRITIC_LOSS` (default `mse`) exists since 2026-09-20, after this batch was queued. Not what killed the paper cell (a temperature failure) |
| varies | **sacpaper** (`b41a`-`b41d`): the paper as written -- target entropy 0.98 · ln 3, batch 64, 1M uniform replay, 0.25 updates a move, hard target copy every 2,000 updates. **saclocal** (`b41e`-`b41h`): two local tweaks -- target entropy **0.1 · ln 3** = 0.11 nats, and snek3's replay settings (batch 128, PER 0.6 with DQN's β anneal, 100k replay, 0.5 updates a move, target every 8) |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | 3,125,000 counted steps = 50M moves at 16 lanes, the plan's budget; ~5 h for the paper cell's four arms, ~10 h for the local cell's |
| control | PPO's hist8 cell (`b27q`-`x`, the viewer's reference strip), b35's local DQN, b40's M-DQN (the entropy-regularised value target). Judged on stage-B density, the passes, drawdowns, and the **policy entropy trace** beside PPO's |
| predicted | registered 2026-09-20 by the agent: the paper cell's α runs away within 100k moves and the cell never reads a non-zero perfect rate -- the paper's own failure on three actions; the local cell reaches 90% before 0.3M counted steps, the fastest onset of any value agent yet, holds steadier than DQN, does not reach PPO's 95% stage-B density, and its entropy settles at 0.05-0.12 nats, an order above PPO's 0.001-0.009 -- **paper half held** (stopped 2026-09-20 at 0.57M: zero perfect games on every seed, α 2.5 × 10⁸, entropy pinned at 1.0766 nats); the local half is live |

**Why.** Group B asks whether PPO's advantage on this game is the entropy bonus rather than the policy
gradient; SAC learns a maximum-entropy policy off-policy, so it separates the two. The series' rule
(`plans/algoExploration/algorithm-series.md` §0) is four arms as the paper wrote them, even where that is
known to be worse, beside four with local tweaks. The gate found the paper's target degenerate on three
actions: 0.98 · ln|A| is 1.077 nats against a maximum of 1.099, a near-uniform policy, so α rose at Adam's
full rate to 66,000 while the score fell from 46 to 3. At 0.1 the same 500k-move arm's α fell to 0.03, its
entropy sat at 0.1, and it read 50-64% perfect -- faster than any DQN cell. That target is the local
cell's first tweak; the replay plumbing is its second, halved to 0.5 updates a move so its four arms land
near the 8-hour budget.

**Learned (paper cell, stopped 2026-09-20 at 0.57M counted steps, 2.3M moves).** The paper cell was stopped
early, by the user's call, because it was dead by every column and was going to stay dead: at 0.57M all four
seeds had read zero perfect games in every eval, the average score had peaked at 47-59 in the first 100k moves
and collapsed to 0.02-0.07 a game, α had run from 1.0 to 2.4-2.5 × 10⁸ and was still climbing at Adam's full
rate, and the policy entropy sat pinned at the target, 1.0766 nats on every seed -- a near-uniform policy over
three actions, the gate's 500k-move failure reproduced four times at 2.3M moves. The prediction's first half
held exactly ("α runs away within 100k moves, never a non-zero perfect rate"). Two and a half hours of training
and 2.3 h of passes that would have measured nothing were returned to the local cell, which the desktop began
at 12:49. What it cost: the paper cell never reached its 50M-move budget, so "does the paper cell arrive by
50M" (`## Open`) is answered by extrapolation, not measurement -- with α at 10⁸ and rising there is no
mechanism by which it could. The arms' charts and evals at 0.57M are archived in `runs/`.

## b40 — Munchausen on the local plumbing: M-DQN beside M-QR-DQN

| | |
|---|---|
| base | b35's local cell (`b35e`-`b35h`'s knobs, verbatim: hist8, b27's reward, `fc 320`, γ 0.99, lr 1e-5, batch 128, 100k PER 0.6, target every 8 updates, the eval-driven ε with shield 0.8 and fork 4), with Munchausen's two changes to the value target: the clipped, τ-scaled log-policy of the taken action added to the reward, and a soft (log-sum-exp) bootstrap in place of the double-Q argmax. `SNEK_MUNCHAUSEN_ALPHA=0.9`, `SNEK_MUNCHAUSEN_TAU=0.03`, `SNEK_MUNCHAUSEN_L0=-1`, the paper's values; ε-greedy acting, 1-step |
| varies | the head under the term. **mdqnlocal** (`b40a`-`b40d`): `SNEK_ALGO=dqn`, M-DQN. **mqrdqnlocal** (`b40e`-`b40h`): `SNEK_ALGO=qrdqn`, N 32, κ 1 -- M-QR-DQN, the "M-best" arm on the rung with the densest stage B of A2-A5 (b37's QR-DQN, 346 rows, best 98.0; C51 239 / 96.8; IQN and FQF at N 8 none) |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | 3M counted steps = 12M moves, b37's cap, so M-QR-DQN and QR-DQN read on one x-axis (QR-DQN's onset was 1.1-1.8M) |
| control | b35's local DQN cell and b37's QR-DQN cell, each against its Munchausen twin; the two twins against each other. The term is α·clip(τ log π, l₀, 0) ∈ [−0.9, 0] a step -- up to 90% of a food reward, under 1% of the +100 win |
| predicted | registered 2026-09-19 by the agent: M-DQN holds a steadier plateau than DQN -- the entropy-regularised target is a hold mechanism, so fewer evals below 80 and best30 above 88 -- but does not raise the best row past 97; M-QR-DQN arrives no earlier than QR-DQN and its plateau and best row are within noise of `b37e`-`h`, because QR-DQN already holds and the ceiling on this game is not in the value target -- **held** (closed 2026-09-20): M-QR-DQN's onset 1.13-1.54M against QR-DQN's 1.12-1.76M, 89-93 after onset with 0-3% below 80 against 87-93 and 0-7%, the same 98.0 best row, no `hof5000` candidate on either; M-DQN steadier than DQN by a few points (13-59% below 80 against 12-76%, best30 92.1), best row 97.8 -- 0.8 over the "not past 97", one seed's 1.7-2.1M stretch, still under 98 |

**Why.** Row A6 closes Group A: every rung so far changed what the critic *represents*, and Munchausen
changes what it is *trained toward* -- an entropy-regularised, KL-damped target that the paper shows lifting
DQN to Rainbow's level and IQN past it. b37 said the distribution buys the hold and not the ceiling; b38
and b39 said eight sampled fractions buy neither. If the ceiling is in the target rather than the head,
this is the row that moves it. Two cells so the term is read on a scalar critic and on the best quantile
one. Unpinned, and the desktop now runs waves of 4 (`max_trainers` 4 with this batch) because eight arms on
the box ran each arm 3.4x slower than four; the laptop takes arms one at a time while its scheduler is up.

**Learned.** The target is not where the ceiling is. Munchausen's entropy-regularised target does on this game
exactly what it did to the hold in b37's reading of the heads: a few points fewer evals below 80 on both critics,
a slightly denser stage B (436 rows against 346 on QR-DQN, 100 against 45 on DQN), and the same best row -- 98.0
on the quantile critic, 97.8 on the scalar one, no checkpoint at 99.2. With A6 closed, seven rungs of Group A
share one result: stage-B best 96.6-98.0, zero `hof5000` candidates, against PPO's 12,479 on the same observation.
What surprised: how little the term moved *anything* on QR-DQN -- onset, plateau, best row and drawdowns all within
a seed's spread. What it changed: FQF at N 32 is not worth a Group-A wave for the ceiling, and the question moves to
the actor (Group B, b41).

## b39 — FQF at N 8 on the local plumbing

| | |
|---|---|
| base | b35's local cell (`b35e`-`b35h`'s knobs, verbatim: hist8, b27's reward, `fc 320`, γ 0.99, lr 1e-5, batch 128, 100k PER 0.6, target every 8 updates, the eval-driven ε with shield 0.8 and fork 4), `SNEK_ALGO=fqf`: **N 8** fractions, K 32, cosine embedding 64, κ 1; the fraction proposal's own RMSProp at 2.5e-9 with entropy bonus 0.001 (the released code's defaults) |
| varies | nothing within the batch: one cell, `fqflocal` (`b39a`-`b39d`), risk-neutral. The head is what varies against b38's neutral IQN cell at the same N |
| cells × seeds | 1 × 4, seeds 1-4 pinned to the letter |
| cap | **2M** counted steps = 8M moves, b38's cap, so the two N 8 rungs read on one x-axis |
| control | b38's neutral IQN cell (N = N′ 8, the reference row in the viewer); b37's C51 and QR-DQN; b35's local DQN. The CVaR *read* of every checkpoint (`tools.closeout --policy-variant cvar:0.25`) is a hand pass after the batch closes |
| predicted | registered 2026-09-18 by the agent: onset with QR-DQN's (1-2M), not C51's; the plateau at or above QR-DQN's 88-93 with the fewest evals below 80 of the quantile rungs; the best row within 2 points of 98.0 and no `hof5000` candidate -- learned fractions sharpen a tail the loss already fits, and this game's ceiling is not in the loss -- **falsified on onset, held on the rest** (closed 2026-09-20): onset 1.25-1.65M is QR-DQN's, not C51's, as predicted, but the plateau is *below* QR-DQN's (71-82 after 90, 27-84% below 80 against 0.3-7%), and the best row was never measured: no checkpoint reached 97, so there is no stage B and no `hof5000` candidate |

**Why.** Row A5, the last head of the ladder: FQF learns *where* the quantiles sit, which should matter
most when the return is bimodal, and a perfect game against a fatal move is that shape. N is 8, not the
paper's 32, for b38's reason: measured on the laptop 2026-09-18 (2,000 steps, one arm, stage A off) FQF runs
16 / 31 / 68 counted steps/s at N 32 / 16 / 8, and the desktop's 8-arm wave runs at about a seventh of the
solo rate, so N 32 is 250 h a wave and N 8 with four arms about 40 h. N 8 also makes b38's IQN the exact
control. One cell: the paper cell is dropped by the plan's §6 rule after b35 and b37, and a CVaR-trained cell
is b38's question. Gates: smoke and restore passed, 28 / 28 mutants killed; the 500k laptop arm was not run,
so a cell still at zero perfect at 500k is stopped.

**Learned.** The head matters after all at N 8: where IQN's eight sampled fractions flattened at 55-65% for
1.8M steps, FQF's eight learned ones keep climbing and three seeds cross 90 by 1.65M -- so b38's ceiling
was the sampling, not the quantile loss. But learned fractions do not buy the hold: after onset the seeds
oscillate 60-95, no eval reaches 97, stage B is empty. N 8 is still short of N 32 fixed quantiles or 51
atoms. The surprise was how wrong the 0.8M reading was -- "drawing b38's band" -- the band moved at 1.2M,
which is the usual onset for the quantile heads on this plumbing.

## b38 — IQN, risk-neutral beside trained under CVaR 0.25, on the local plumbing

| | |
|---|---|
| base | b35's local cell (`b35e`-`b35h`'s knobs, verbatim: hist8, b27's reward, `fc 320`, γ 0.99, lr 1e-5, batch 128, 100k PER 0.6, target every 8 updates, the eval-driven ε with shield 0.8 and fork 4), `SNEK_ALGO=iqn`: **N = N′ 8**, K 32, cosine embedding 64, κ 1. Not Dopamine's 64: IQN's cosine embedding is a `Linear` per (sample, τ), and 64 runs at 14 counted steps/s on this CPU against 92 at 8 -- the paper's own remark that 8 "appears to be sufficient" is what the plan quoted |
| varies | the acting rule during training. **neutral** (`b38a`-`b38d`): τ uniform on [0, 1]. **cvar25** (`b38e`-`b38h`): `SNEK_DIST_RISK_ALPHA=0.25`, `SNEK_DIST_RISK_TRAIN=1` -- Dabney et al. 2018 §4's risk-sensitive agent, the distortion τ ← 0.25τ on the acting policy and on the target's argmax; every eval risk-neutral, as the paper scores it |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | **2M** counted steps = 8M moves: b35's local cell reached 90% by 0.6-1.6M and its plateau is what is read, and 3M at IQN's rate is a day an arm |
| control | b35's local DQN cell; b37's two heads; the two cells against each other. The CVaR *read* of the neutral checkpoints (`tools.closeout --policy-variant cvar:0.25`) is a hand pass after the batch closes |
| predicted | registered 2026-09-17 by the agent: the neutral cell sits with b37's heads; the CVaR-trained cell arrives later (it under-explores the food-seeking moves a neutral policy takes) but holds a steadier plateau -- fewer evals below 80, a higher stage-A share at ≥98 -- which is the tail-risk diagnosis showing through -- **falsified** (closed 2026-09-19): the neutral cell never sat with b37's heads (55-65% at the cap against C51's 88-91), and the CVaR-trained cell was lower throughout, 40-54% for three seeds with `b38f` collapsing to zero at 0.9M; no checkpoint reached stage A's 97 in either cell |

**Why.** Row A4 is the row Group A exists for: acting on the low quantiles is the one thing a scalar
critic cannot do, and it is the most direct test of the diagnosis that the failures are rare fatal
moves rather than noisy returns. It runs on the local plumbing for the reason b37 does.

**Learned.** IQN at N = N′ 8 does not learn this game past 65%: every arm climbs to 50% within 0.2M --
faster than any rung yet -- and then holds a flat 55-65% band to 2M with the score line at DQN's 93-94, so
the snake plays but does not finish. Training under CVaR 0.25 made it worse, not steadier, and killed one
seed. No checkpoint reached 97, so stage B is empty and the CVaR hand pass is moot. The open question is
whether the ceiling is N 8 -- eight samples against a +100 point mass -- or the implicit head itself; b39
(FQF, N 8) is the first half of that answer and is drawing the same band.

## b37 — C51 and QR-DQN on the local plumbing

| | |
|---|---|
| base | b35's local cell (`b35e`-`b35h`'s knobs, verbatim: hist8, b27's reward, `fc 320`, γ 0.99, lr 1e-5, batch 128, 100k PER 0.6, target every 8 updates, the eval-driven ε with shield 0.8 and fork 4) |
| varies | the head. **c51local** (`b37a`-`b37d`): `SNEK_ALGO=c51`, 51 atoms on [−10, 110]. **qrdqnlocal** (`b37e`-`b37h`): `SNEK_ALGO=qrdqn`, **N 32**, κ 1. Not the paper's 200: the quantile Huber is N × N′ pairs, and 200 runs at 10 counted steps/s on this CPU against 225 at 32; 32 is also FQF's N, so A3 and A5 share the count |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | 3M counted steps = 12M moves, b35's local cap |
| control | b35's local DQN cell (90% by 0.6-1.6M, then 71-88%; 45 stage-B rows, best 96.6); the two heads against each other. C51's stability, which b36 could not read, reads off this cell's drawdown columns |
| predicted | registered 2026-09-17 by the agent: both heads reach 90% in DQN's 0.6-1.6M and plateau higher and steadier -- recent perfect above 71-88%, fewer evals below 80 -- with QR-DQN at or above C51; neither near PPO's 95% density, stage B in the tens of rows -- **held on the plateau, split on onset** (closed 2026-09-18): both heads hold 88-93 after onset with 0.3-7% of evals below 80 against DQN's 12-76%, best30 94.2 / 94.5 against 88.3, QR-DQN's best row (98.0) above C51's (96.8); but C51 arrives at 0.27-0.34M and QR-DQN at 1.1-1.8M, and neither head has a `hof5000` candidate |

**Why.** Rows A2 and A3, moved from the papers' plumbing to snek3's own because b35 and b36 showed the
papers' recipe does not reach competence in 10M moves on this game for either DQN or C51
(`../plans/algoExploration/a-return-tail.md` §6, first bullet; decided with the user 2026-09-17). The
local cell arrives in a million steps at 4.3 h an arm, so the heads can be compared where a head has
something to act on; with the base fixed the two rows share one wave.
**The rungs' CPU cost, measured 2026-09-17** (laptop, one arm at a time, 2,000 steps, counted steps/s): DQN 484, C51 285,
QR-DQN N 32 / 64 / 200 at 225 / 74 / 10, IQN N = N′ 64 / 32 / 16 / 8 at 14 / 23 / 52 / 92. The papers' counts were set for a
GPU; here they are days per arm, which is why b37 runs N 32 and b38 N 8 at a 2M cap.

**Learned.** The distribution buys the hold, not the ceiling. Both heads keep the 90% plateau the local DQN
loses -- 88-93 to the cap against DQN's 72-87 oscillation, five to eight times the stage-B rows -- and C51
gets there two to four times sooner than DQN, but the best row barely moves (96.8 / 98.0 against 96.6) and
no checkpoint reached `hof5000`. The surprise was QR-DQN's late onset at N 32, a million steps behind C51,
and its being the steadiest cell once there. C51 is stable on this plumbing, which b36 could not read. The
ceiling now rests on b38's CVaR-trained cell.

## b36 — C51 stability: two supports on the paper cell's plumbing

| | |
|---|---|
| base | b35's paper cell (its entry below), with C51's optimiser: Adam 2.5e-4, ε 3.125e-4 |
| varies | `SNEK_DIST_ATOMS` 51 (`b36a`-`b36b`) / 101 (`b36c`-`b36d`) on `SNEK_DIST_V_MIN=-10`, `SNEK_DIST_V_MAX=110` |
| cells × seeds | 2 × 2, seeds 1-4 pinned to the letter |
| control | none: a stability batch, judged on its own curve (`zero_since` never above 200 evals once an arm has read 80%; no target mass piling on the end atoms) |
| predicted | registered 2026-09-17 by the agent: both supports stable at this reward -- snek2's C51 instability was the 30-value observation and the b2-era reward scale, not the head -- and 101 atoms level with 51 on onset and density, so 51 is the comparison's setting |

**Why.** Row A2 of the algorithm series (`plans/algoExploration/a-return-tail.md` §3) runs C51 against
b35's DQN, but snek2's C51 was unstable enough that a win-reward shrink was tried and falsified, so the
plan puts a stability batch first. The support is this game's discounted return range, not the paper's
clipped [−10, 10]: at 51 atoms that is a 2.4-wide bin against the paper's 0.4, so the second cell halves
it, the same step the paper's 21 → 51 ablation took. Only a stable support proceeds to the four-seed
comparison.

## b35 — DQN, the value family's control: paper cell beside local cell

| | |
|---|---|
| base | the hist8 observation (`SNEK_OBS_HISTORY=8`), b27's reward (preset `b2`, chase-safe 0.1 at gate 75, food-distance 0, step penalty 0.01, win 100), `fc 320`, γ 0.99; `SNEK_ALGO=dqn` |
| varies | the plumbing. **paper** (`b35a`-`b35d`): the Munchausen paper's DQN-Adam -- Adam 5e-5, ε 3.125e-4, batch 32, `SNEK_REPLAY_RATIO` 0.25 (one update per 4 moves), 1M uniform replay (`SNEK_PRIORITY_EXPONENT=0`), 20k prefill, hard target copy every 2,000 updates, 1-step, `SNEK_EPSILON_SCHEDULE=linear` 1.0 → 0.01 over 250k moves, shield 0, fork 1, one lane; eval every 2,500 steps. **local** (`b35e`-`b35h`): snek3's DQN defaults -- lr 1e-5, batch 128, replay ratio 1, 100k PER 0.6, target every 8 updates, the eval-driven ε 0.4 → 0.002, shield 0.8, fork 4; eval every 1,000 |
| cells × seeds | 2 × 4, seeds 1-8 pinned to the letter |
| cap | paper 10M moves (= 10M counted steps, one lane and no fork); local 3M counted steps = 12M moves, b2's cap. Both sized under 8 h with stage B and the hof passes, from the laptop's 1,865 and 357 st/s |
| control | PPO's `hist8` table (`b27q`-`b27x`, 94-95% density, 99.81 /30k); the two cells against each other |
| predicted | registered 2026-09-17 by the agent: the local cell reaches 90% perfect by 1M counted steps as b2 did and ends at 40-60% stage-B density, well under PPO; the paper cell learns more slowly (ε is 0.5 for its first 125k moves), reads a non-zero perfect rate by 3M moves and ends below the local cell, because uniform replay and an unshielded ε 0.01 keep feeding the endgame deaths the fork exists to avoid -- **falsified on the local cell** (90% by 1M for one seed of four; stage-B density 0%, best row 96.6), **held on the paper cell**, which at 10M moves reads 7-16% perfect and gives stage B nothing |

**Why.** Row A1 of the algorithm series (`plans/algoExploration/a-return-tail.md`): every value-based
row -- C51 through FQF, Rainbow, R2D2 -- is read against DQN, and no DQN has run on the 26+16
observation or b27's reward. Two cells because the series compares algorithms on their papers'
settings, and this codebase's own plumbing (the fork, the shield, PER, the fast target copy) exists in
no paper: the gap between the cells is what that plumbing is worth here, measured once, so every later
row's paper cell can be read with it in mind.

**Learned.** Neither DQN is near PPO on this observation and reward, and the cells are far apart. The paper recipe does not
arrive in 10M moves: 7-16% perfect, trailing score 80-85, still rising, no stage B. The local plumbing reaches 90% in
0.6-1.6M counted steps and then oscillates at 71-88% with 74.5% of its evals below 80; 45 stage-B rows, best 96.6, none at
98. So the fork, the shield, PER and the fast target are worth the whole gap, and the value ladder cannot be built on the
paper cell at this budget (a fifth of the plan's 50M). `plans/algoExploration/a-return-tail.md` §6, first bullet, applies.

## b34 — zigzag shaping: a reversal potential beside a reversal penalty

| | |
|---|---|
| base | b27's `hist8` config, verbatim (its entry below has every knob): 100M, `SNEK_PPO_ANNEAL_FRACTION` 0.5, so γ, λ and entropy reach their finals at 50M and hold to 100M |
| varies | the reward, two ways. **`zz`**: `SNEK_ZIGZAG_SHAPING=0.5`, potential-based, Φ = −(reversal pairs among the last 8 moves the body shows -- the observation history depth), F = c·(γΦ(s′) − Φ(s)). **`rp`**: `SNEK_REVERSAL_PENALTY=0.5`, subtracted on every step whose move is a `left` straight after a `right` or the converse. A reversal is adjacent moves only; a U-turn (the same turn twice, the fill pattern) is not one. [`../plans/zigzag-shaping.md`](../plans/zigzag-shaping.md) |
| cells × seeds | 2 × 4 (`b34a`-`b34d` `zz` seeds 1-4, `b34e`-`b34h` `rp` seeds 5-8) |
| control | b27's `hist8` cell (`b27q`-`b27x`), the base itself: 94-95% density, 99.81 /30k |
| predicted | registered 2026-09-16 by the agent with the user: `zz` level with `hist8` on density and the 30k top, and its reversal rate by board fill within 1 pp of the champion's -- the invariance holds and PPO takes nothing from the hint, as b21 found for chase-safe. `rp` cuts the early-board (fill < 50%) reversal rate by more than half, leaves the endgame's ~0 where it is, and sits within noise of `hist8` on density. A density or 30k gain in `rp` falsifies "zigzagging is not the mechanism"; a density loss says the early zigzags are load-bearing -- **held for `zz`, falsified for `rp`**: `zz` is the base to the decimal; `rp` loses 5 pp of density (89.4-90.8 against 94.3-95.7, complete separation) while its onset is the fastest on this base (98% by 1.9-2.5M) and its stability the best measured, and the 30k top is level |

**Why.** The move-history plan deferred this as its "fifth arm": b27 showed that letting the policy
*see* its turn sequence was the largest lever found, and whether *charging* for zigzags does anything
has never run. Measured before queueing, the record checkpoint reverses on 1.7% of steps (0.178 per
meal, 17 per episode), almost all below 50% board fill and one in ~900 steps from 60% on -- so both
terms act on the open early board. The doses are sized from that: 0.5 puts the potential's traffic at
0.18 per meal (chase-safe's order) and the penalty at 0.089 per meal (the step penalty's flow, which
b26 found real where a tenth of it did nothing).

**Learned.** The potential is a no-op, as b21 found for chase-safe; the penalty is not. `rp50` arrives fastest on this
base (98% by 1.9-2.5M in every seed, before the reference's quickest at 4.1M), is the most stable cell measured (0.04% of
evals below 80), and matches the 30k top -- `b34e` @3637248 ran 29,954 /30,000, the 99.82 ceiling from a 3.6M checkpoint --
but its 500-episode density is 90.1% against 95.4, complete separation: the standing charge caps the plateau at 97-98.
Neither term enters the base. Worth one wave: `rp` annealed to zero after onset.

## b33 — the perfect-game reward

| | |
|---|---|
| base | b27's `hist8` config (pen01 + `SNEK_OBS_HISTORY=8`) at 50M, `SNEK_PPO_ANNEAL_FRACTION` 0.5: every ramp final at 25M, the last 25M at the finals |
| varies | `SNEK_PERFECT_GAME_REWARD` 0 / 10 / 30 / 50 / 100 / 200 / 300 / 1000 — the bonus paid on the last meal in place of its food reward of 1 |
| cells × seeds | 8 × 4 (`b33aa`-`b33bf`, seeds 1-4) |
| control | the batch's own `win100` cell — also the first run of b27's config at 50M; b27's `hist8` cell at 100M is the reference |
| predicted | registered 2026-09-13 by the agent with the user: 0 and 10 lose density but do not stall as snek2's b33 did, since the step penalty and starvation already charge for dawdling — watch the starve/death split; 30-300 level with 100 at the top, no monotone trend readable at n=4; 1000 the lowest density of the eight through slower onset (the huber critic at δ 1 learns a 1000-point terminal jump slowly, and the value error swamps normalised advantages), few or no collapses; the 30k top of every cell from 30 up within noise of 99.8. The user expects 10 to do poorly and 1000 to be unstable — **held on the mechanism, falsified on 1000's density and on instability**: 0 and 10 at 87% with no stall; 50-300 level with 100; 1000 the *highest* density (96.5%) after a 15M onset, and no eval below 80 after 25M in any cell |

**Why.** At γ 0.999 a meal of delay costs ~1% of the win bonus, so the bonus sets how urgent finishing
is relative to eating — 0.1 of a food reward at 10, one at 100, ten at 1000. Invariant 6 says progress
toward the win only raises value when W > 1/(1−γ^k), 84-143 at γ 0.999, so 100 is marginal and
anything below should fail as snek2's win-10 batch did; yet b30 at γ 1.0 reached the 99.82 ceiling. The
rule was derived for DQN without a step penalty and is under test for PPO. Two decades either side of
the default, dense enough to see whether the response is monotone or a plateau.

**Learned.** The reward is a monotone onset lever that saturates at 100: 0-25M density 67.5% at 0, 88-90% from 50 up;
25-50M 96.0% at 0, 99.1-99.5% from 100 up. Every sub-50 eval in the batch is before 15M and no sub-80 eval after
25M, so 1000 is slow (15M to arrive against 4M), not unstable. `win0` and `win10` do not decline to finish —
99.78 / 99.58 best30, `hof5000` rows at ≥99.8 — so invariant 6 was a property of a reward without a step penalty.
At 30k the 50M cap is the ceiling: no cell ran a full 30,000 at 99.8 (b27's 100M had 9), and the `win100` control
reaches b27's 50-75M density in its 25-50M window but loses the top. 100 stays.

## b32 — b28's best checkpoints annealed on to a horizon of 1.0

| | |
|---|---|
| base | b28's `hist8` arms at their converged values: γ 0.999, λ 0.999, entropy 0.001, lr 2.5e-4, clip 0.2 |
| varies | the start. Each arm is warm-started (`SNEK_INIT_FROM`) from one of b28's eight best /30k checkpoints — its actor, plus the source arm's critic and optimiser — and anneals γ and λ 0.999 → 1.0 over its first 50M, then holds 1.0 for 50M (`SNEK_PPO_ANNEAL_FRACTION` 0.5 of 100M; the step starts at 0) |
| cells × seeds | 1 × 8 (`b32a`-`b32h`, seeds 1-8; sources `b28k` @162.86M / 162.69M / 162.96M, `b28m` @131.50M / 134.58M / 138.31M, `b28n` @185.93M, `b28o` @136.48M — 29,940-29,946 /30,000 each) |
| control | b28's own arms, the plateau these start on (99.79-99.82 /30k); b30, the same 1.0 finals reached from scratch |
| predicted | registered at queue time 2026-09-11 by the agent, not the user: every arm's first stage-A evals read ≥ 98 (the warm start holds), no arm collapses, stage-B density at or above b28's 97.5% — and the 30k top stays within noise of 99.82, nothing at 99.83 or above — **held on the arms and falsified on the top** (2026-09-13): every arm ≥99.5% stage-B density from its first window, no collapse — and `b32g` @62423040 read 29,967 /30,000, the first row above 99.82 |

**Why.** b28 said holding converged values widens the plateau and does not raise it, and b30 asks
whether the last 0.001 of horizon raises it from scratch. This asks the same of the best policies
already found, at a tenth of the cost per answer: 100M from a 99.8 start rather than 200M from zero,
with the critic and optimiser carried over so the one change is the horizon. Eight starts in the same
plateau also say how much of a 30k rank is the checkpoint and how much is the seed.

**Learned.** The warm start holds (stage B 99.8% in every 25M window, the first included, where b28 read 89.3% from
scratch), and one arm of eight left the plateau: `b32g` @62423040 at **29,967 /30,000**, confirmed at 29,957 on seed 13
and promoted as the record (`hallOfFame/HOF.md`). Its 61-63M plateau averages 29,950 — the old ceiling as a region's
mean — while the other seven arms' tops (29,943-29,958) sit in basins on the plateau and fall back to it on a fresh seed.
So a 30k rank is mostly the checkpoint's basin, and the recipe found a higher one once in eight; whether the horizon or
the extra 100M did it is the open item above. `hof30k` 92 full rows of 6,859, 5,000 → 30,000 drop +0.02 pp.

## b31 — `mse` value loss on the `hist8` base

| | |
|---|---|
| base | pen01 + `SNEK_OBS_HISTORY=8` (b27's `hist8` cell) at 100M, anneals final at 50M |
| varies | `SNEK_PPO_VALUE_LOSS` `mse` instead of `huber`; nothing else |
| cells × seeds | 1 × 8 (seeds 1-8, `b31a`-`b31h`) |
| control | b27's `hist8` arms, seeds 17-24 |
| predicted | registered at queue time 2026-09-10 by the agent, not the user: density above b27 `hist8`'s 95.4% and fewer evals below 80, on b19 and b23's `mse` result; the 30k top unchanged — **falsified** on both counts: 88.0% against 95.4 with complete separation, and no row reached 30,000 at 99.8 (best 99.77, stopped) |

**Why.** `mse` was the largest single step on the corner-grid ladder — b19's most stable cell at
+5 pp, and b23's 32.7 → 61.6% with the collapses gone — but the horizon-anneal base has run `huber`
since b24 and the two have never been combined. One knob off the current best config, at the cap
the reference used, says whether that step still exists on top of move history.


**Learned.** It does not. `mse` reads 88.0% density (85.4-91.3) against 95.4 with the seeds cleanly
separated, and the loss is onset — 51.0% in the first 25M against 82.9 — that the endgame (99.2 against
99.8) never recovers. The stability gain b19 found survives (0.17% of evals below 80 against 0.68) but the
density gain that made `mse` the ladder's largest step does not transfer: it belonged to the λ 0.99 base.
`hof5000` 833 rows through the gate against 1,836; `hof30k` retired every row (best 99.77). `huber` stays.

## b30 — the horizon annealed to 1.0

| | |
|---|---|
| base | pen01 + `SNEK_OBS_HISTORY=8` (b27's `hist8` cell) at 100M, anneals final at 50M |
| varies | `SNEK_PPO_DISCOUNT_FINAL` and `SNEK_PPO_GAE_LAMBDA_FINAL` 1.0 instead of 0.999; entropy 0.01 → 0.001, lr and clip fixed as before |
| cells × seeds | 1 × 8 (seeds 1-8, `b30a`-`b30h`) |
| control | b27's `hist8` arms, seeds 17-24 |
| predicted | registered at queue time 2026-09-10 by the agent, not the user: some seeds collapse after 50M as b10's fixed γ 1.0 cell did (44% of evals below 50), the survivors level with b27 `hist8` at the top — **falsified so far**: no seed has an eval below 50, stage B 95.3% against 95.4; passes pending — **falsified**: no seed had an eval below 50, and the survivors' top (29,946 /30k) equals b28's rather than b27's |

**Why.** The horizon anneal ends at 0.999 because b10 ran γ 1.0 from step 0 and it collapsed half
the time while holding the record (`b10ck` 99.65). Reaching 1.0 only after 50M of training under a
finite horizon is a different regime, and the undiscounted objective is the one the game actually
scores. This asks whether the last 0.001 of horizon is worth anything once the policy is already
competent.


**Learned.** Nothing visible, and nothing lost. Level with `hist8` at every depth (stage B 95.3 against
95.4, `hof5000` 1,844 gate rows against 1,836, `hof30k` 43 rows at ≥99.8 against 9) and no collapse in any
seed — the b10 regime does not return when γ reaches 1.0 after 50M under a finite horizon. The top,
`b30a` @94371840 at 29,946 /30,000, is exactly `b28k`'s count: the 99.82 ceiling reached a third time, by a
third route. The ceiling is not the horizon and not the step count.

## b29 — lr and clip annealed to zero

| | |
|---|---|
| base | pen01 + `SNEK_OBS_HISTORY=8` (b27's `hist8` cell) at 100M |
| varies | the anneal moves from the horizon to the optimiser: `SNEK_PPO_LEARNING_RATE_FINAL` 0 and `SNEK_PPO_CLIP_FINAL` 0.001 ramp from step 0 to the cap (`SNEK_PPO_ANNEAL_FRACTION` 1.0); γ 0.99, λ 0.95 and entropy 0.01 held fixed, no `_FINAL` |
| cells × seeds | 1 × 8 (seeds 1-8, `b29a`-`b29h`) |
| control | b27's `hist8` arms, seeds 17-24 |
| predicted | registered at queue time 2026-09-10 by the agent, not the user: a frozen endgame — density at or above b27 `hist8` over the last 20M with fewer drawdowns, a lower top because γ 0.99 never sees the long horizon that put b27's rows at 99.8 — **falsified** on density (83.1% against 95.4, behind in every 25M window, nothing through the 99.6 /5k gate); the lower top held |

**Why.** The PPO paper's Atari schedule anneals lr and clip to zero together, and b17 found that
holding the annealed clip floor for the last 10M was worth +6-7 pp. b24-b28 anneal the horizon
instead and leave the optimiser static. This is the other schedule on the same base, and the clip
floor is 0.001 rather than 0 because the trainer refuses a clip of exactly 0.

**Learned.** Worse everywhere: 25M-window density 68.6 / 73.7 / 89.2 / 95.0% against 82.9 / 95.4 / 99.5 /
99.8, all eight seeds below the reference's eight, best30 99.39 against 99.79, and `hof5000` passed none of
3,749 rows through the 99.6 gate so `hof30k` ran nothing. The deficit is already there at 0-25M, where lr
has fallen only a quarter, which points at the fixed γ 0.99 / λ 0.95 rather than the optimiser decay: the
arm removed the horizon anneal as well as adding its own, so the paper's schedule is not cleanly tested.
A clean test keeps γ/λ → 0.999 and adds lr/clip → 0 on top.

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

**Learned.** The first 100M matches b27's `hist8` (94.9% against 95.4), so the extra hold is what reads
97.5% density, 7,718 rows through the 99.6 /5,000 gate (4.2x) and 152 rows at 99.8 /30,000 against 9. The
top did not move: `b28k` @162856960 at 29,946 /30,000 (99.82) is one game above the record pair, z ≈ 0.3,
and nothing reached the 99.83 that would separate. Verdict: holding converged values widens the plateau
by an order of magnitude and does not raise it; the 100M cap is enough for this config, and the next
lever is a knob. Nothing promoted.

## b27 — move history depth

| | |
|---|---|
| base | pen01 (horizon anneal + step penalty 0.01, 26-value observation) |
| the config in full, since `hist8` is the base every batch since has run on | PPO, `fc 320`, 128 lanes, rollout 256, minibatch 512, 4 epochs, lr 2.5e-4, clip 0.2, huber value loss, vf 0.5, grad clip 0.5, Adam ε 1e-7, adv norm on, `target_kl` 0. Reward: preset `b2` (chase-safe 0.1, gate 75, food-distance 0), step penalty 0.01, win 100 (default), death −5, starve −0.5. **Anneals: `SNEK_PPO_ANNEAL_FRACTION` 0.5 of the 100M cap** — γ 0.99 → 0.999, λ 0.95 → 0.999 and entropy 0.01 → 0.001 all ramp linearly to their finals at 50M and hold them for the last 50M; lr and clip are not annealed. b33 ran the same config at a 50M cap, so its ramps land at 25M |
| varies | `SNEK_OBS_HISTORY` 0 / 4 / 8 — two bits per past move, `[turned left, turned right]`, read off the body ([`../plans/archive/obs-history.md`](../plans/archive/obs-history.md)) |
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

**Why.** The first batch of the one-knob sweep ([`../plans/archive/hyperparam-sweep.md`](../plans/archive/hyperparam-sweep.md)):
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
