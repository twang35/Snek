# Runs — current state and forward plan

**Newest at the top, in every doc in this directory.** Current state first, then what is next,
then how we got here. A batch that closes is written above the batch before it, and a new finding
goes directly under `## Established` in [`findings.md`](findings.md).

## Now

**b10 — the γ sweep — is on the desktop, wave 4 of 8, and the laptop is idle.** At 2026-09-03 01:20
the eight γ 0.93 and γ 0.94 arms (`b10ay`-`b10bf`) were 39-44% through 50M, 59 min in, so wave 4 lands
around 02:40 and its stage B follows automatically; waves 5-8 (γ 0.95, 0.96, 0.97, 0.98, 0.995, 0.9975,
0.999, 1.00) are queued behind it, then b11-b14 at their re-based λ 0.99. b10 should close mid-afternoon
2026-09-03. Ledger: 252 done, 164 queued, 8 running, `attention` empty, 153 GB free, load 14. The
laptop's only process is the stage-B chart window left from b9's `hof5000` pass, which is free to close.

**Queued on the desktop 2026-09-03 09:44: `b9-hof30k`** — b9's **33 checkpoints at ≥99 /5,000** from 9
arms, at **30,000 episodes on seed 7**, the seed no selecting pass used, 16 shards, label `hof30k`,
priority 5 so it takes the next wave boundary ahead of b10's last training wave. ~1M episodes, ~10 min.
(A first version of the spec said ≥90, which is all 727 rows and ~4 h; corrected before it started.) A
row here is a confirmed rate, not a selected high — the hall-of-fame measurement for the whole top of
the plateau at once. The record to beat is `b5h` at 98.96.

**b9's `hof5000` pass is closed and documented** (below, and in [`results.md`](results.md)): 727 rows,
exit 0, **77 at ≥98.73 and 33 at ≥99**, and `b9ch-lam999-seed4 @47251456` at **99.40 /5,000** with a
98.70 basin is the promotion candidate — [`hof-promote`](../skills/hof-promote/SKILL.md) is the next
step and has not been run.

**b10's first three waves say the discount is not a knob with a broad top — below ~0.95 the endgame
is out of reach at 50M.** Six γ values, 24 arms, stage B closed on all of them, and **not one
checkpoint passed the stage-A screen**: zero stage-B rows across the whole set. b7aa-b7ad (γ 0.99) are
the reference:

| γ | value horizon 1/(1−γ) | best30 | range | sef | drawdown < 50% | stage-B rows |
|---:|---:|---:|---|---:|---:|---:|
| 0.70 | 3.3 | 3.95 | 2.7-5.2 | 0.0 | never competent | 0 |
| 0.80 | 5 | 40.27 | 24.4-49.7 | 0.0 | never competent | 0 |
| 0.85 | 6.7 | 66.47 | 64.1-68.0 | 0.0 | 54.4% (2 of 4 reached 80%) | 0 |
| 0.90 | 10 | 79.55 | 75.8-81.6 | 3.6 | 38.7% | 0 |
| 0.91 | 11.1 | 81.90 | 77.5-85.5 | 5.8 | 26.5% | 0 |
| 0.92 | 12.5 | 84.75 | 83.1-86.6 | 12.1 | 18.4% | 0 |
| 0.93 / 0.94 (live, 21M) | 14.3 / 16.7 | 81.2-87.7 / 85.5-88.7 | | 4-27 | | — |
| **0.99** (b7) | 100 | **97.75** | 97.7-97.8 | 90.9 | 0.29% | 4,003 |

**‡ γ and λ are not interchangeable, even at the same GAE horizon.** b9's λ 0.90 at γ 0.99 has an
advantage horizon 1/(1−γλ) of 9.2 steps and reached best30 96.75 with 3.9% record density; b10's
γ 0.92 at λ 0.98 has a horizon of 10.2 and reaches 84.75 with none. The advantage estimator's horizon
is not what a short γ costs — the *value target* is discounted too, and a critic that cannot see the
+100 ten steps ahead cannot value the endgame. The spec's prediction for γ 0.80 ("fast early, then an
endgame ceiling below perfect") is right; "same shape as 0.8, milder" for γ 0.90 understated it — the
ceiling is still 20 points below perfect at 10 steps. Where the curve reaches the base is what waves
5-8 measure; the queued γ 0.995-1.00 cells are the ones that could beat it.

## Just closed: b9 — λ 0.99+ doubles the record density, and the top is a plateau

**All eight waves and all eight stage-B passes closed 2026-09-02 16:48**, 64 arms at 50M on the
desktop, 43,969 stage-B rows. b9 changes only `ppo_gae_lambda` off b7's winning cell — `fc (320,)`,
4 epochs, entropy 0.01, lr 3e-4, γ 0.99, clip 0.2 — so **b7aa-b7ad are its λ 0.98 arms** and the
sweep is one curve. Drawdown is the median share of post-competence stage-A evals below 50% (and 80%)
perfect, as in b8:

| λ | GAE horizon | rows | ≥98%/500 | per-seed share | ≥98.5 (`hof5000` cands) | best row | best30 (4 seeds) | sef | drawdown < 50% | < 80% |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| 0.00 | 1.0 | 8 | 0.0% | 0 0 0 0 | 0 | 94.0 | 88.7 85.1 90.8 88.5 | 24.1 | 11.29% | 68.6% |
| 0.50 | 2.0 | 392 | 0.0% | 0 0 0 0 | 0 | 96.6 | 94.2 94.2 94.9 93.7 | 81.8 | 0.05% | 11.1% |
| 0.80 | 4.8 | 1,133 | 0.8% | 0.4 1.7 0.4 0.7 | 3 | 98.6 | 96.6 95.6 96.1 96.2 | 85.9 | 0.02% | 9.3% |
| 0.85 | 6.3 | 1,279 | 0.8% | 0.3 1.1 1.7 0.0 | 1 | 98.8 | 95.2 96.3 96.1 96.0 | 87.3 | 0.00% | 8.8% |
| 0.90 | 9.2 | 2,130 | 3.9% | 3.7 5.3 5.1 1.1 | 16 | 99.2 | 96.5 97.1 96.9 96.5 | 89.4 | 0.07% | 6.5% |
| 0.91 | 10.1 | 2,168 | 4.3% | 2.3 7.8 3.6 3.7 | 14 | 99.2 | 96.5 97.4 96.9 96.7 | 89.4 | 0.02% | 6.7% |
| 0.92 | 11.2 | 2,363 | 4.8% | 2.0 6.9 3.8 5.7 | 18 | 99.2 | 96.5 97.6 96.8 96.8 | 89.6 | 0.02% | 6.1% |
| 0.93 | 12.6 | 2,545 | 5.4% | 4.1 9.9 3.3 4.2 | 22 | 99.2 | 96.9 97.5 97.1 97.2 | 90.6 | 0.02% | 4.9% |
| 0.94 | 14.4 | 2,359 | 4.6% | 4.5 5.6 2.8 5.2 | 21 | 99.2 | 97.0 96.6 96.7 96.9 | 90.8 | 0.00% | 5.6% |
| 0.95 | 16.8 | 2,665 | 5.3% | 3.7 9.0 4.6 3.4 | 23 | 99.6 | 97.1 97.0 96.9 96.9 | 92.3 | 0.02% | 3.9% |
| 0.96 | 20.2 | 2,882 | 8.3% | 4.6 12.7 8.1 7.0 | 67 | 99.6 | 96.8 97.8 97.2 97.4 | 92.0 | 0.03% | 5.0% |
| 0.97 | 25.2 | 3,613 | 11.7% | 9.0 17.4 9.2 10.8 | 130 | 99.6 | 98.1 98.0 97.6 97.8 | **93.0** | 0.03% | **4.1%** |
| 0.98 (b7) | 33.6 | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 174 | 99.6 | 97.8 97.8 97.7 97.7 | 90.9 | 0.29% | 6.2% |
| **0.99** | 50.3 | 5,173 | **27.3%** | 27.2 24.1 31.4 26.0 | 451 | **100.0** | 98.3 98.3 98.4 98.3 | 90.6 | 0.77% | 6.4% |
| 0.995 | 66.9 | 4,998 | 27.3% | 24.8 27.4 34.3 22.3 | 496 | 99.8 | 98.2 98.5 98.8 98.1 | 89.1 | 1.03% | 7.7% |
| 0.999 | 91.0 | 4,897 | 25.6% | 27.8 17.4 24.2 31.3 | 453 | **100.0** | 98.5 98.2 98.1 **99.0** | 87.9 | 1.26% | 8.1% |
| **1.00** | 100 | 5,364 | **29.5%** | 25.2 32.7 29.6 30.8 | **574** | 99.8 | 98.4 98.5 98.4 98.3 | 88.4 | 2.10% | 8.7% |

**Three readings, in order of weight:**

- **Density is monotone in λ up to 0.99 and flat above it.** 0.00 → 0.50 → 0.80-0.85 → 0.90-0.95 →
  0.96 → 0.97 → 0.98 → 0.99 reads 0, 0, 0.8, 4-5, 8.3, 11.7, 17.3, 27.3%; then 27.3, 25.6, 29.5.
  The per-seed ranges of the four plateau groups all overlap (λ 0.99 24-31, λ 1.00 25-33), so n=4
  cannot order them. In record-region checkpoints per four arms that is **693 at λ 0.98 against
  1,412 at 0.99 and 1,582 at 1.00** — the row count rises too (4,003 → 5,173 → 5,364), so this is more
  strong checkpoints, not a thinner gate. best30 moves with it: 97.75 → 98.33-98.45, and every
  plateau seed is ≥98.1.
- **Stability moves the other way, exactly as b8's trade-off said it would.** Drawdown below 50% is
  ≤0.07% for every λ from 0.50 to 0.97, 0.29% at 0.98, then 0.77 → 1.03 → 1.26 → **2.10%** at λ 1.00;
  the sub-80% share bottoms at 3.9-4.1% (λ 0.95-0.97) and rises to 8.7%. b8 found that the knobs which
  steady the curve bank the fewest records; b9 finds the converse from the same knob — the λ that banks
  the most records has the noisiest deployed policy. λ 1.00 at 50M matches b4's 2.0% at 50M, the
  collapse b8 was built to fix. **This is why 0.99 rather than 1.00 is the recommendation above.**
- **`sef` ranks the sweep backwards for the third time.** It peaks at **93.0 at λ 0.97** and falls to
  88.4 at λ 1.00, so ranking on it picks the 11.7% end of the curve over the 29.5% end, a 2.5x miss.
  The stage-A ≥98% share (20.4% at 0.98, 28.9% at 0.99, 30.5% at 1.00) tracks density, as
  [`findings.md`](findings.md) said it would.

**What it does not settle.** All of this is at 50M and 500 episodes: the plateau's ordering, whether
any 100/500 row is a real record, and whether λ 1.00's drawdown keeps growing past 50M the way b4's
did. The `hof5000` pass answers the second; the third would need a longer arm. **b3's single-seed
call that λ 1.0 was a loser is inverted** — same pattern as the fc-layout ranking b7 overturned.

**λ 0 is the floor**: 8 stage-B rows from four arms, `b9ab-lam0-seed2` screened none, onset of
competence at 5-10M against 1.3-2.8M for every λ ≥ 0.80. Per-arm numbers in
[`results.md`](results.md), 127 panels in [`charts.md`](charts.md).

## Just closed: b8 — every knob cut the drawdown, none produced a record

**Both waves and both stage-B passes closed 2026-09-01**, 16 arms at 100M on the desktop, plus a
135-row `hof5000` pass on the laptop. b4 is the control, truncated to b8's 100M cap:

| group | arms | drawdown < 50% | ≥98%/500 | ≥98.73 /5,000 |
|---|---|---:|---:|---:|
| `target_KL` 0.02 | `b8i`-`b8l` | 5.9% | **6.0%** | 1 |
| entropy 0.01 → 0.001 | `b8e`-`b8h` | 3.5% | 5.0% | 0 |
| entropy 0.003 | `b8a`-`b8d` | 3.7% | 4.3% | 0 |
| λ 0.95 | `b8m`-`b8p` | **2.2%** | 2.3% | 0 |
| **b4 control @100M** | `b4a`-`b4h` | 8.4% | 5.7% | 1 (at 200M) |

**One champion-level row in 135 deep measurements, and no hall-of-fame candidate.** Both
never-exercised knobs did fire — `target_KL` stopped the epoch loop on 1.9-3.3% of updates, the anneal
completed 0.0100 → 0.0010 at the cap — so this is a real measurement of both, not a silent no-op. Per-arm
numbers in [`results.md`](results.md), 32 charts in [`charts.md`](charts.md).

## Just closed: b7, the fc-layout sweep — `fc (320,)` wins

**All four waves and all four stage-B passes closed 2026-09-01**, 32 arms in ~11 h on the desktop.
Pooled 10.9% of 28,006 stage-B rows in the ≥98%/500 record region, and the spread across layouts is
3.4x:

| layout | ≥98%/500 | | layout | ≥98%/500 |
|---|---:|---|---|---:|
| **`fc (320,)`** | **17.3%** | | `fc (200,100,50)` | 10.8% |
| `fc (200,100)` | 11.8% | | `fc (160,160)` | 8.3% |
| `fc (100,200,100)` | 11.6% | | `fc (300,100)` | 6.8% |
| `fc (100,100)` | 11.3% | | `fc (400,200)` | 5.1% |

**Every `fc 320` seed beats every seed of five of the seven other layouts** (exact Mann-Whitney
p=0.029, the floor at 4-vs-4). This **inverts b3's single-seed ranking**, which put `fc 300,100`
first and `fc 320` last of those three and is what queued b4 and b7 in the first place. Per-arm
numbers in [`results.md`](results.md), the reading and the two retractions in
[`findings.md`](findings.md), 64 charts in [`charts.md`](charts.md).

**It also changes the primary metric for this kind of question.** `strong_eval_fraction` ranks b7's
layouts *backwards* (Spearman −0.79 across the eight layout means); the stage-A ≥98% rate ranks them
right (+0.80). Both are free from the same eval history — see [`findings.md`](findings.md).

## b9 just closed, as it read at 2026-09-02 17:53 (superseded)

**b9 closed on the desktop at 2026-09-02 16:48 — all 64 arms trained and all eight stage-B waves
done — and b10, the γ sweep, is on wave 1 of 8.** At 17:53 its eight arms (γ 0.70 and γ 0.80 x seeds
1-4, `b10aa`-`b10ah`) were 47-58% through 50M, 65 min in, so wave 1 lands about 18:45 and its stage B
follows automatically. b9's eight waves took ~19 h end to end, so b10 closes around midday
2026-09-03. Behind it: b11 (lr 4e-5, 32+8), b12 (1 epoch, 40+8), b13 (minibatch 32, 32+8), b14
(rollout 32, 24+8). Ledger: 225 done, 189 queued, 8 running, `attention` empty, 157 GB free, load 14.
**Nothing is running on the laptop.**

**b9's answer: PPO's λ 0.98 default was not the top of the curve.** Record density (≥98%/500 stage-B
rows) climbs monotonically through the whole sweep, **17.3% at λ 0.98 → 27.3% at λ 0.99**, then sits
on a plateau — 27.3, 27.3, 25.6, 29.5% across 0.99, 0.995, 0.999 and 1.00, indistinguishable at four
seeds. Every seed of λ 0.99 beats every seed of λ 0.98 (24.1-31.4 against 15.1-19.0, Mann-Whitney
p=0.029, the floor). Three stage-B rows scored **100/500** — two adjacent `b9ch-lam999-seed4`
checkpoints at 47.2-47.3M and `b9bw-lam99-seed1` at 48.4M — the first perfect 500s this project has
measured. Full table and reading under *Just closed: b9* below.

**Both decisions were taken the same evening (user, 2026-09-02 ~19:00):**

1. **The `hof5000` pass on b9 ran on the laptop, 19:12-20:29** — 36 arms, the 727 checkpoints at
   ≥99/500 (the cut was raised from 98.5 to 99 in the same decision), 8 shards, exit 0. **77 rows at
   ≥98.73 and 33 at ≥99, against 6 and 0 for all of b7.** `b9ch-lam999-seed4 @47251456` measures
   **99.40 [99.1, 99.5]** with a 27-neighbour basin at 98.70 — a stronger candidate on both numbers
   than `b5h` was when it was promoted (99.20 own, 98.54 basin, confirmed 98.96 /30,000). **Next:
   `hof-promote` on it**, a 30,000-episode confirm at seed 7. Table in [`results.md`](results.md).
2. **λ 0.99 is the new default** (`ppo/algo.py`, commit `6b5f8390e`, deployed to the box) **and b11-b14's
   128 specs were re-based to it on `ops`** before any of them started. b10 runs at 0.98 as queued: γ and
   λ set the GAE horizon together (1/(1−γλ)), so it stays a clean one-knob sweep off b7's cell, and its
   winner is read against b7aa-b7ad. Rationale for 0.99 over 1.00: the plateau's density within noise,
   the tightest seed spread of any group (98.3-98.4), a third of λ 1.0's drawdown.

**‡ Two things to know about the daemon's read.** The glance line labels the running b10 wave
"g85, wave 2 of 8" while the ledger's eight running arms are the g70/g80 cell, wave 1 — a labelling
quirk, not a dispatch problem. And `p0q-ep8-long` still reads `failed` from the retired `p`-prefix
era; `attention` is empty.

## b9 at wave 5, as it read at 2026-09-02 08:38 (superseded)

**Batch b9 — the λ sweep — is on the desktop and nothing is on the laptop.** 40 of 64 arms trained at
2026-09-02 08:38; `b9-stageb-w5` is the only job running (41 m in, no trainers), stage B is closed on
waves 1-4, and waves 6-8 are queued as 24 arms covering **λ 0.96, 0.97, 0.99, 0.995, 0.999 and 1.00**.
Behind b9 sit b10 (γ 0.70), b11 (lr 4e-5), b12 (1 epoch), b13 (minibatch 32) and b14 (rollout 32) —
197 jobs done, 221 queued, load 16 on 8 trainers' worth of box, 161 GB free.

**b9 changes only `ppo_gae_lambda` off b7's winning cell, so b7aa-b7ad *are* its λ 0.98 arms** and the
sweep reads as one curve:

| λ | GAE horizon | best30 | range | sef | ≥98%/500 |
|---:|---:|---:|---|---:|---:|
| 0.00 | 1.0 | 88.28 | 85.1-90.8 | 24.1 | 0.0% |
| 0.50 | 2.0 | 94.25 | 93.7-94.9 | 81.8 | 0.0% |
| 0.80 | 4.8 | 96.12 | 95.6-96.6 | 85.9 | 0.8% |
| 0.85 | 6.3 | 95.90 | 95.2-96.3 | 87.3 | 0.8% |
| 0.90 | 9.2 | 96.75 | 96.5-97.1 | 89.4 | 3.9% |
| 0.91 | 10.1 | 96.88 | 96.5-97.4 | 89.4 | 4.3% |
| 0.92 | 11.2 | 96.92 | 96.5-97.6 | 89.6 | 4.8% |
| 0.93 | 12.6 | 97.17 | 96.9-97.5 | 90.6 | 5.4% |
| 0.94 | 14.4 | 96.80 | 96.6-97.0 | 90.8 | pending |
| 0.95 | 16.8 | 96.97 | 96.9-97.1 | **92.3** | pending |
| **0.98** (b7) | **33.6** | **97.75** | 97.7-97.8 | 90.9 | **17.3%** |

**Density is monotone in λ across every measured value, and the biggest step is the last one.** 0.93
to 0.98 is 5.4% to 17.3%, a 3.2x jump larger than every increment below it combined — so **the sweep
has not found a peak and the interesting arms are the six still queued**, which bracket 0.98 on both
sides. λ 0.98 also has the tightest seed spread of any group (97.7-97.8 against 96.5-97.6 at λ 0.92).

**λ 0 is a floor worth having measured**: best30 88.28, `sef` 24.1, and 8 stage-B rows from four arms
against λ 0.93's 2,545. `b9ab-lam0-seed2` screened **zero** checkpoints. A one-step advantage cannot
learn this task.

**‡ `sef` picks the wrong λ, exactly as predicted when b8 closed.** It rises to 92.3 at λ 0.95 and
falls to 90.9 at λ 0.98, so ranking on it would take the 5.4%-and-below end of the curve over the
17.3% end. This is now reproduced on two batches and two knobs — see [`findings.md`](findings.md).

**‡ One stale ledger entry, not a live problem:** `p0q-ep8-long` reads `failed`, from the retired
`p`-prefix era. `attention` is `None` and the box is otherwise clean.

**What this says about the forward plan.** b9's own result so far is "the default was right, and
possibly not high enough" — which makes b10-b14 sweeps *off a λ that may still move*. If λ 0.99 or
0.995 beats 0.98, every batch queued behind b9 is holding a superseded λ. Worth deciding whether to
let b10-b14 run as queued or re-base them once b9's top end lands.

## b9's first wave, as it read at 2026-09-01 22:50 (superseded)

**The b9-b14 sweep is running on the desktop; the laptop is idle.** b9 is on wave 6 of 8 —
λ 0.94 and λ 0.95 x 4 seeds (`b9bg`-`b9bn`, 50M each) — 64% done at 2026-09-02 07:22, 4,600-8,300
steps/s per arm, 1 h 03 m in. Waves 1-4 (λ 0, 0.50, 0.80, 0.85, 0.90, 0.91, 0.92, 0.93 x 4 seeds, 32
arms) have finished both stages; two more waves and b9's stage B are queued behind the current one,
then b10 (γ, 64+8 arms), b11 (lr, 32+8), b12 (epochs, 40+8), b13 (minibatch, 32+8) and b14 (rollout
length, 24+8), each with `auto_stage_b`. **Nothing is running on the laptop.**

**b9's finished waves already answer part of the question the ‡ note below raised, and it's the
opposite of the worry.** Pulling the 32 finished arms' stage-B rows off the `results` branch, the
≥98%/500 density climbs monotonically with λ at this 50M cap: 0% at λ 0 and λ 0.50, 0.8% at λ
0.80-0.85, then 3.9/4.3/4.8/5.4% across λ 0.90-0.93 — closing in on the 5-6% region b4 and b8's
controls showed at their own caps, not falling away from it. So far, higher λ is not costing record
density the way b8's λ 0.95 did at 100M; whether that holds, turns over, or is just b9 not yet having
reached b8's regime is what the currently-training λ 0.94/0.95 pair and the waves above 0.95 will
say. Not yet checked against drawdown — that reading waits for stage A's post-competence share, same
as b8's.

**Batch b8 closed 2026-09-01 and it did not find what it was looking for.** Details below; the short
version is that all four stability knobs cut b4's drawdown and none of them beat the control on record
density, and [`findings.md`](findings.md) now has the 2x2 showing that **epochs and network shape are
the lever** — b7's 32 four-epoch arms sit at 0.1% drawdown against b8's 2.3% and b4's 2.0% at a
matched cap.

**‡ This is why the b9-b14 sweep needs care in how it's read.** b9 sweeps λ, and λ 0.95 was b8's
*best* knob on drawdown and its *worst* on density at b8's 100M cap — the two metrics ran opposite
across all four knobs ([`findings.md`](findings.md)). Decide before b9's stage B closes which of the
two its arms are being ranked on; ranking on `sef` or on drawdown would have picked b8's weakest
group there. b9's own density trend above is a first read, not the final one — it is at 50M, not
b8's 100M, and the drawdown side is still unread.

## b8 mid-flight, as it read at wave 1's 71M (superseded)

**Batch b8 — "what fixes b4's collapse" — is training on the desktop, wave 1 of 2, ~70% done at
2026-09-01 14:03.** Eight arms at 100M transitions each: entropy **0.003** x seeds 1-4
(`b8a`-`b8d`) and the **0.01 -> 0.001 anneal** x seeds 1-4 (`b8e`-`b8h`). 2 h 49 m in, ~7,000
steps/s per arm, 69-75M done, so wave 1 lands around 15:15 and its stage B follows automatically.
Wave 2 (`target_KL` 0.02, λ 0.95) and b8's stage B are queued behind it. **Nothing is running on the
laptop.**

Read at b8's own horizon against b4's control truncated to the same 71M — **both treatments are
marginally ahead and neither is resolved.** n=4 cannot resolve an effect below ~10 pp, and best30 is
a peak statistic that is still rising in every arm:

| group | n | best30 | range | sef |
|---|---:|---:|---|---:|
| b4 control, entropy 0.01, truncated to 71M | 8 | 96.69 | 95.8-97.3 | 85.8 |
| b8 entropy 0.003 | 4 | 96.98 | 96.5-97.4 | 88.2 |
| b8 entropy anneal 0.01 -> 0.001 | 4 | 96.93 | 95.9-98.0 | 87.1 |

`b8g-entanneal-seed3` at best30 **98.0** is already above every b4 arm's final best30 at twice the
budget, which is the single most encouraging number in the batch and also exactly the kind of
maximum-over-8 that this project keeps having to retract. Wait for stage B.

**‡ One caveat on b8's design, now that b7 has closed: it holds `fc (200,100)`, which b7 has just
shown to be the *wrong* shape.** b8 is still the right experiment — it is asking what fixes b4's
drawdowns, and b4 is `fc (200,100)`, so the control has to match — but a stability knob that helps
here has to be re-confirmed on `fc (320,)` before it goes into a champion attempt.

## The b8 design, as queued

**"What fixes b4's collapse".** 4 stability knobs x 4 seeds at 100M, holding b4's config fixed
(`fc (200,100)`, 8 epochs, b2's reward, seeds 1-4) so exactly one knob moves per group, with **b4
itself as the control** at 8 seeds — no control arms queued. Priorities 60/70 put both waves behind
every b7 wave, which is why wave 1 only started once b7's fourth close-out was done.

| wave | knob | why it, and not something else |
|---:|---|---|
| **1** | entropy **0.003** | the only knob whose stability signal in b3 is monotone in both directions: the share of post-competence evals below 80% perfect ran 2.9% at 0.003, 12.2% at the 0.01 reference, **45.6%** at 0.03 |
| **1** | entropy **0.01 → 0.001** | the anneal in `ppo/schedules.py`, **never used by an arm**. Explore early, commit late; it completes exactly at this cap |
| 2 | **`target_KL` 0.02** | **never exercised** — `epochs_run` is 8 in all 97,656 of b4's recorded updates. 0.02 binds on the tail and not the body: b4's `approx_kl` is 0.0035 median and 0.0079 at p95, but **0.023 at p99 and 0.514 at worst**, 146x the median |
| 2 | **λ 0.95** | b3's two λ arms had the best drawdown profile in the sweep — 0.0% below 50% *and* below 80%, mean 93.9 against the reference's 88.2 — at one seed each |

**Why this and not a second fc sweep at a smaller learning rate.** `lr 1e-4` is the one knob b3
measured as peaked, and it came out worse on the very axis the consistency argument is about: 14.5% of
its post-competence evals below 80% against the reference's 12.2%, best30 95.0 against 96.6, **zero**
≥98%/500 rows against 6, and the latest competence onset in the sweep. Smaller lr bought slower
learning, not steadier learning. It is n=1, so it is not settled — but b7 is already resolving the
network axis, and the collapse is the larger unexplained effect.

**Both new knobs were smoke-tested before queueing**, because a silently-ignored knob costs four arms:
the anneal reads 0.0091 at 10% of a 0.01→0.001 run, which is the linear value, and `target_KL` at a
deliberate 0.001 stops the epoch loop after 1 of 8 epochs with `stopped_early=True`.

**Batch b4 closed on the desktop 2026-08-31**: pooled **7.3%** of stage-B rows at ≥98%/500 against
b6's 12.9% and b5's 9.6%, and best30 **97.0-97.9** against 97.8-98.5 for both. **The arm built from
b3's two best single knobs is the weakest of the three 8-seed batches** — shape and epochs interact
negatively, and a one-knob-at-a-time sweep licenses no stacking. That is the finding b7 was built on.
Its charts were imported and redrawn 2026-09-01, and a laptop `hof5000` pass re-measured its 274
≥98.5% candidates at 5,000 episodes: **one row clears the snek2 champion and none reaches 99%**
([`results.md`](results.md)). **‡ The b5-vs-b6 half of that comparison did not survive the same
re-measure** — the two are identical at 5,000 episodes and b5 is ahead on champion-level rows; see
[`findings.md`](findings.md).

**Before that — 2026-08-30 17:34 — three things closed and both boxes went idle.**

| what | where | outcome |
|---|---|---|
| **batch b6** stage B, 8 arms | laptop | done in 226.1 min, status 0. Pooled **12.8%** of rows ≥98%/500 |
| **batch b5** stage B, 8 arms | desktop | done in 222.6 min, status 0. Pooled **9.6%**; best row **100.0%/500** |
| **the parallelism sweep**, 11 waves | desktop | all 53 jobs done. **The eval side answered; the training side did not** |

Full per-arm numbers and the b5/b6 comparison are in [`results.md`](results.md); the charts are in
[`charts.md`](charts.md); the sweep is in [`findings.md`](findings.md).

**b6 leads b5 on the pooled headline but the sign test is a coin (5 of 8, p≈0.29→0.73), and rank 1 is
a tie at 98.5 best30.** More importantly the two batches differ in **two** knobs — `fc (200,100)` + 4
epochs against `fc (320,)` + 8 epochs — so this is not the network-shape test
[`results.md`](results.md) and this file have both been calling for. **b4 ran that test on 2026-08-31
and b7 is sweeping the axis outright** — see the top of this section.

### Next, in the order the evidence argues for

**The sweep after b8 is designed: [`../plans/hyperparam-sweep.md`](../plans/hyperparam-sweep.md)**
(2026-09-01, revised) -- batches b9-b21, one knob each at four seeds on b7's `fc (320,)` base at 50M, with
b7aa-b7ad as the shared control; the machine-readable grid is `plans/hyperparam-sweep.json` and
`tools/sweep_specs.py` expands a batch into specs. **b9 (λ, 16 values) and b10 (γ, 16 values) were
queued on the desktop 2026-09-01 20:30, 128 arms behind b8's stage B**, every never-exercised value of
b9-b21 smoke-tested on the laptop first. The clip and learning-rate anneal knobs for b17 are
committed and deployed (box at `13def02e8`, 2026-09-01 ~21:00), and **b11-b14 were queued behind
b9-b10 the same evening** -- 256 arms in the queue, six batches at a time from here on.

1. ~~**One batch varying only the network**, matched epochs and matched budget.~~ **Done: b7, closed
   2026-09-01.** `fc (320,)` won and b3's ranking inverted. What this opens, in order:
   **a champion attempt at `fc (320,)` + 4 epochs**, which no batch has yet run at length — b5 was
   `fc 320` at *8* epochs and b7 was 4 epochs at only 50M, and the two best cells of the grid have
   never been in the same arm; and the **DQN** half of the same sweep, since `dqn/net.py` takes the
   same `fc_layers` config, which would say whether the shape effect is PPO's or the task's.
2. **Re-confirm whatever b8 finds on `fc (320,)`.** b8 holds b4's `fc (200,100)` because b4 is its
   control, and b7 has since shown that shape to be the weaker one, so a stability knob that helps b8
   is a candidate rather than a result until it is re-run on the winning shape.
3. **Re-run the worker sweep with long waves.** 3.2-minute waves cannot resolve it — see
   [`findings.md`](findings.md). ~30 min per wave is what the earlier hand-measurement used.
4. **Sweep arm count**, which the queue cannot express: `_dispatch` takes pending jobs in priority
   order up to `max_trainers` regardless of priority value, so any group of 8+ trainer jobs launches
   as exactly 8. It needs a `runtime.json` commit between waves, or a real sweep job type.

### Next for PPO

**‡ Rewritten 2026-09-01, because b7 refuted the paragraph that was here.** It read "two hidden
layers beat every single-layer width tried", from b3's one-seed-each densities — `fc 300,100` 9.0%,
`fc 200,100` 7.9%, `fc 320` 5.6%, `fc 500` 3.2%, `fc 200` 0.8%. At four seeds and 50M those three
reverse: `fc 320` **17.3%**, `fc 200,100` 11.8%, `fc 300,100` 6.8%. What survives from the old
paragraph is the capacity half — width past 320 hurts, and `fc (400,200)` is now last of eight — and
the convenient fact that **`fc 320` is snek2's shape, carried across so a champion's weights convert,
and every batch in both eras has used it.** `dqn/net.py` takes the same `fc_layers` config, so the
same sweep is one batch away for DQN and has never been run — see [`findings.md`](findings.md).

**The follow-up wave, designed and not yet launched** — push the axis that moved rather than resample
the flat ones: epochs 12 and 16, minibatch 128, rollout 256, and `fc 200,100` + epochs 8 as the one
interaction worth a slot. Depth belongs to a "better agent" batch rather than to a seed-matched
comparison, which has to hold the network at 320 to stay matched against b2.

## The b2-era plan — superseded, kept for the reasoning

**Written 2026-08-29, before b3, b5, b6 and b4 ran.** Item 3 below describes a 4-arm b4 holding
`fc 320` fixed, which was never run: the `b4` name was reused on 2026-08-30 for the network-shape
test, and *that* is the b4 that closed on 2026-08-31. What survives here is why each item was
queued, which is worth more than the schedule was.

1. **Read b2 against b1 and against b29/b41/b47.** A b2-vs-b29 difference smaller than the
   b29-vs-b41 process-noise gap is noise, not a port regression — snek2 ran that config three times
   precisely to have the yardstick.
2. **Phase 6 — `ppo/`.** The reason snek3 exists, and the design is
   [`../plans/ppo.md`](../plans/ppo.md). **Phases 6a, 6b and 6c are all closed** — the algorithm seam
   is in `train.py` with three fixed-seed DQN arms byte-identical across it, `ppo/` is written and
   tested (122 fixtures, 14 of 14 mutants killed), and batch b3 has run 15 arms. Deployed to the
   desktop 2026-08-29 once b2's stage-B wave published. **6d — batch b4 — is next**, at **18M**
   transitions to match b2 (3M counted steps x 6 transitions per step; the plan's 12M was wrong).

   **‡ Two claims made from the 6b gate arm are withdrawn, and both were withdrawn by b3.** The gate
   arm was 508k transitions on snek3's *unshaped* defaults, and neither conclusion survived a shaped
   arm at 20x the budget:

   - **"PPO is behind DQN rather than beside it"** — withdrawn. Matched on transitions *and* on
     reward function, the two ranges top out at the same number (96.9 best30), and on the ≥98%/500
     count PPO is ~10x denser. The gate arm's gap was the reward function, not the algorithm.
   - **"`clip_fraction` 0.03 says the learning rate is low"** — falsified outright. Raising it to
     1e-3 and 3e-3 both made things *worse* (85.2 and 69.9 best30 at 3M, the latter at sd 18.4), and
     1e-4 was worse than 1e-3. The learning rate is peaked at the default and a low clip fraction did
     not mean what I read into it.

   The gate arm's chart stays at [`../runs/ppo-smoke.png`](../runs/ppo-smoke.png) as a record of
   what an unshaped PPO arm does. It is a gate arm, not a p-series arm.

3. **Batch b4 — the seed-matched gate batch.** 4 arms, seeds 1-4, **18M transitions**, b2's env
   config, **fc 320 held fixed** so the comparison is seed-matched against b1, b2, b29, b41 and b47.
   **b3 hands it the reference config unchanged**, because b3 found no winner — which makes b4 a
   cleaner comparison than the plan expected rather than a blocked one. Phase 3's ≥90% bar is already
   cleared by b3, so b4's job is the comparison, not the gate.

**The stage-A queue is next after b2 and the numbers are now measured rather than projected.** Stage A
is **66%** of an arm's 8.1 h (not 90%), and streaming recovers **3.3-3.4x** of it (not 5.7x) — see
[`findings.md`](findings.md). Cutting episodes does not work: 4x fewer buys 1.6x, because the cost is
lane drain. Every way of recovering it makes the epsilon schedule's feedback lag
([`invariants.md`](invariants.md) invariant 2), so the lag must be **bounded** rather than left to
float. b1 is the baseline any such change is measured against.

| an arm at 3M counted steps | training | stage A | total |
|---|---:|---:|---:|
| as b2 runs today | 2.79 h | 5.33 h | **8.1 h** |
| + the two bit-exact fixes (landed) | 2.33 h | 5.33 h | 7.7 h |
| + a bounded eval queue, 2 workers per 4 trainers | 2.33 h | ~0 | **2.3 h** |

The queue's arithmetic closes: 4 trainers at 299 st/s demand 1.20 checkpoints/s, and one streamed
worker supplies 0.54-0.89, so **two workers serve four trainers** — six processes on the desktop's 16
cores.

## Backlog

One line per idea, with a prior. A design that is settled enough to implement gets a file in
[`../plans/`](../plans/) and a row here.

| idea | prior |
|---|---|
| **PPO** | [`../plans/ppo.md`](../plans/ppo.md) — **phases 6a and 6b closed 2026-08-29; batch b3 is next.** No longer a backlog item. The reason snek3 exists. On-policy and wide, so it is the algorithm that actually exploits a 196k env-steps/s vectorised env, where DQN's replay ratio caps the loop at ~4,000 steps/s |
| **Batched or asynchronous self-eval** | **the next change. 8.1 h an arm becomes ~2.3 h, measured.** The win is keeping the lanes full, so a queue drained by streaming workers gets it; the drained shape is the whole cost and cutting episodes does not touch it. Cost is a lag on the epsilon schedule — **bound it**, do not let queue depth set it |
| **Replay ratio < 1** | ~~the only way past ~4,000 agent steps/s~~ **do not use this to reproduce snek2.** Ratio 1.0 already matches snek2's 1 gradient step per transition; lowering it makes snek3 *less* data-efficient than snek2 ever was. It remains a real dynamics knob, worth 2x at batch 512, but it is not a comparability fix — `SNEK_MAX_STEPS` is |
| **Drop observation indices 10/12/14** | ~1.5x on the observation build. Region enumeration is 33% of the connectivity cost and those three indices are its only consumers. Batch 45 reached 99% with them in, so this is a cost question |
| **Munchausen-DQN, SAC-discrete** | the discrete off-policy actor-critic options, if PPO underperforms. **TD3 does not apply** — it is continuous-action and this task has three discrete actions |

## How we got here — the closed phases

History, kept because each phase gate is a claim someone may want to re-check. Nothing here is
current state.

**Batch b2 — b29's record config on the torch stack, seeds 1-4, 3M steps. Closed 2026-08-29**, all
four arms and the stage-B wave `done`; results on the `results` branch, unread into
[`results.md`](results.md). Launched on the desktop 2026-08-29 08:09. This is the phase-3 gate re-run on the configuration snek2 actually set
records with; b1 ran snek3's bare defaults and that was the wrong batch to gate on.

| knob | snek3 default | b2 = b29 |
|---|---|---|
| `SNEK_IS_WEIGHTS` | 1 | **0** |
| `SNEK_TARGET_UPDATE_PERIOD` | 8 | **1000** |
| `SNEK_DISCOUNT` | 0.99 | **0.9975** |
| `SNEK_FOOD_DISTANCE_REWARD` | 0.001 | **0** |
| `SNEK_CHASE_SAFE_SHAPING` | 0.0 | **0.1** |
| `SNEK_CHASE_SAFE_GATE` | 85 | **75** |
| `SNEK_FC_LAYERS` | 320 | 320 |

**Five knobs differ, not the two b1's write-up first suggested.** The target-update period and the
discount are substantial algorithmic differences, and they were nearly missed by reading the results
summary instead of snek2's own b47 spec. Read the spec.

Seed N is pinned to arm letter N, so every arm is seed-matched against b29a-d, b41a-d, b47a-d **and**
snek3's own b1a-d. Budget ~7 h an arm; the desktop auto-queues one stage-B wave for the batch.

### ‡ Interim reading at 0.36-0.41M of 3M — the phase-3 gate is met; the lead over b47 was a units artefact

Read 2026-08-29 09:13, ~1 h in, at **106 counted steps/s an arm** (the 290 st/s in the log excludes the
self-eval; stage A is ~2/3 of the wall clock). ETA ~16:00. Both tables are 100-episode graph evals, so
b2 and b47 are on the same instrument and the counts are directly comparable.

**‡ Corrected 2026-08-29: a snek3 counted step is four game moves and a snek2 step was one.** See
[`findings.md`](findings.md). Every b2 step number below is 4x a b47 step number in game moves,
buffer rows and gradient steps alike, so **the "b2 leads b47 on every seed" headline compares b2 at
1.4-1.6M transitions against b47 at 0.34-0.39M.** The b47 column is truncated to the same *counter*
value, which is not the same work. Data efficiency is identical in both eras (1 gradient step per
transition), so nothing here is a learning-rate difference — it is a budget difference.

| seed | step | b2 ≥95 | b2 ≥98 | b2 best30 | b47 ≥95 | b47 ≥98 | b47 best30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.36M | 0 | 0 | 85.9 | 0 | 0 | 34.1 |
| 2 | 0.37M | 7 | 1 | 92.7 | 0 | 0 | 67.0 |
| 3 | 0.34M | 0 | 0 | 81.0 | 0 | 0 | 55.0 |
| 4 | 0.39M | **52** | **17** | **96.8** | 0 | 0 | 68.3 |

b47 at its own furthest point — 1.38-1.63M, where snek2 froze mid-batch — had best30 82.2 / 90.5 /
**96.0** / 83.6 and ≥95 counts 0 / 8 / 173 / 0.

**‡ "b2d at 0.39M has already passed b47c's best30 at 1.63M" is withdrawn.** b2d at 0.39M counted
steps had done **1.56M** transitions and 1.56M gradient steps; b47c at 1.63M had done 1.63M and 1.63M.
The two are at matched values on both axes, so the correct statement is **b2d matches b47c at matched
work**, not that it beats it at a quarter of the budget. The matched-work comparison is the one to
make from here: read b2 at 4x b47's step number.

**The phase-3 gate (≥90% perfect) is met on both readings and by two arms.** b2d's trailing-30 perfect
rate peaks at 96.8% and it has 52 single evals at ≥95/100 including a 100; b2b is at 92.7% with 7. b1
never reached one such eval in 3M steps, so the five knobs are the whole difference — which is snek2's
own batch-28/29 finding, reproduced.

**‡ Same shape as b47, different carrier.** One arm carries the batch and two produce nothing ≥95, as
in b47 (carrier seed 3) and b41 (also seed 3) and b29 (seed 2). b2's carrier is seed 4, so it is now 1
of 4 for every seed and the carrier is a coin, not a property of the seed. **Do not read b2a or b2c as
a regression**: snek2 saw the same on the same config.

**What is not yet answered is phase 5**, which needs a ≥98%/500 *region* from the stage-B wave rather
than these 100-episode rows. b2d's 17 rows ≥98/100 and 7 ≥99/100 make a region plausible and not
assured — snek2's winner's-curse drops on this instrument were −2.6, −3.2 and −6.2 pp. And the snek2
side of that comparison is **b29's own close-out, not b47's**: b47 was frozen at 69-81% of its 2M cap
and never closed out, so no b47 ≥98%/500 data exists.

**b1-vs-b2 is unaffected by the correction** — both arms ran the same collector at the same ratio, so
they share the 4x and their step axes are directly comparable. "The five knobs are the whole
difference" stands.

**b1 is closed** ([`results.md`](results.md)): four arms at 3M, peak perfect 42.1 / 58.3 / 56.7 /
81.9%, **no checkpoint anywhere at 95/100**, every arm still climbing at its cap. Its stage-B wave
has run and published 0 rows an arm, which is the honest measurement rather than a failure.

**Phases 0-4 are closed and batch b1 has run.** `env/`, `vectorized/`, the measurement engine,
checkpoint I/O, the eval wave, the charts, the viewers, `dqn/`, `train.py` and the desktop daemon are
all in, and the box runs snek3 rather than snek2. **The phase-3 gate is still open** — see Now.

**Phase 0 — the two env implementations agree.** 36,000 states × 30 observation indices, **0
mismatches**, across a growth regime (24,000 states, 49 episodes, lengths to 60) and a coiled endgame
regime (12,000 states, 280 episodes, 26 perfect games), with rewards, terminations, both shaping
terms and the win path in parity too. **17 of 17 hand-made mutants killed.**

**Phase 1 — the snek2 champion plays in torch.** `b44a-lowlr7-b29b-ckpt2739000` converted and
measured **98.8% perfect over 3,000 episodes** against snek2's 98.73%, inside the ±0.6 pp gate. The
conversion itself is exact rather than close: on 12,864 states the two networks' Q-values differ by
at most 2.7e-5 on values of magnitude ~30.6, and the **argmax is identical on every state**, so the
policies are the same function. `watch.py` plays it and `record_gif.py` records it.

**Phase 2 — the flat protocol reproduces the tiered one.** All **3,222** checkpoints of
`b45a-lowlr8-b29b` converted and measured, against snek2's own close-out: mean per-row difference
**−0.004 pp** on a 0.041 pp standard error, and observed spread / predicted spread **1.00**. 14
minutes on four shards. Three findings came out of it, including a **5.7x correction to the cost of
stage A** — see [`findings.md`](findings.md).
