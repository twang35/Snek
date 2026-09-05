# Runs — current state and forward plan

**Newest at the top, in every doc in this directory.** Current state first, then what is next,
then how we got here. A batch that closes is written above the batch before it, and a new finding
goes directly under `## Established` in [`findings.md`](findings.md).

## Now

**b14 (rollout), b15 (entropy) and b16 (target KL) closed with their hof passes; b17 (clip) is on wave 4 of 8 on the
desktop and b19 (switches) on wave 2 of 3 on the laptop.** As of 2026-09-05 15:35:

| box | batch | state | ETA |
|---|---|---|---|
| desktop | b17 (clip + anneals, 64 arms, 8 waves) | waves 1-3 closed with hof5000/hof30k; wave 4 (clip 0.1 anneal, lr anneal) training from 15:30 | training done ~03:40 2026-09-06; b18 (grad-norm clip, 24 arms) behind it; box clears ~09:20 Sun |
| laptop | b19 (switches, 24 arms, 3 waves) | wave 1 (noadvnorm, mse) closed with hof5000; wave 2 (Adam ε 1e-5, 1e-8) training from ~15:25 | ~19:00 today; then b20 (lanes, 16 arms) and b21 (shaping, 24 arms); clears ~04:10 Sun |

**‡ b15-b21 were generated from b7's base at λ 0.98, not the re-based λ 0.99.** Every arm's config table says
`ppo_gae_lambda 0.98` (b13's and b14's say 0.99), and the spec notes name `b7aa-b7ad` as the control. Until this update
the tables read them against b9's λ 0.99 cell (27.3% density), which made every b15-b17 cell look 8-14 pp short —
b16's null-check cells included, which were predicted identical to the control. `viewer/references.json` now points
b15-b21 at `b7aa-b7ad` (17.3%, best30 97.75, 6.2% of evals below 80%), and every reading below is against that.
**Decision (user, 2026-09-05): b18, b20 and b21 stay queued at 0.98**, so b15-b21 are one comparable set against b7's
cell. What 0.98 loses is the direct comparison with b11-b14; the corner grid is where the two sets meet.

**What closed.** b14: density rises with rollout to 512 (38.3%) and stability with it to 1024; 512 joins the corner
grid. b15: the entropy coefficient trades density for stability monotonically, the anneals average their endpoints,
0.01 stays. b16: target KL is a no-op at 4 epochs. Verdicts in [`results.md`](results.md), tables in
[`charts.md`](charts.md), findings under `## Established` in [`findings.md`](findings.md).

**HOF candidates, not records.** `hof30k` rows at 99.4 /30,000: `b15ay-entanneal10-seed1` @24182784 and (2026-09-04)
`b11ag-lr1e4-seed3` @33243136; the HOF's third place, `b9ch`, is 99.30 [99.2, 99.4]. Four b16 rows and three b15 rows
read 99.3. Promotion is the `hof-promote` skill and the user's call; nothing approaches `b10ck`'s 99.65.

**Live so far.** b17: clip is flat around the base from 0.05 to 0.2 and worse past it; loosening the clip *reduces*
collapses; the 0.2→0.02 anneal matches the base on density with better stability. b19: **the mse value loss** reads
22.2% density at 0.97% of evals below 80% (base 17.3% at 6.2%) — the most stable cell at this base, and denser;
noadvnorm is as stable and 4 pp short.

**Infrastructure today.** The desktop's `runner` is `daemon` ([`../plans/rename-runner-to-daemon.md`](../plans/rename-runner-to-daemon.md));
the desktop scheduler restarted on the new code at 13:59 with b17 wave 3 adopted mid-training. `tools/progress_update.py`
now tables every batch in `references.json` (b16 had closed without ever being tabled), separates an anneal cell from the
fixed value it starts at, names the cells of a switches batch, and reads a laptop batch's state from the laptop's own
status lines — uncommitted, awaiting review.

## b14 and b15 mid-flight, as it read at 2026-09-04 19:10 (superseded)

**b14 (rollout) wave 1 closed and wave 2 is training on the laptop; b15 (entropy) wave 1 closed on the
desktop, which is now running the hall-of-fame passes queued ahead of b15's remaining waves; b11's two
passes are done.** As of 2026-09-04 19:10:

| box | batch | state | ETA |
|---|---|---|---|
| laptop | b14 (rollout, 24 arms) | wave 1 (32, 64) closed 18:30; wave 2 (192, 256) at 34-43M of 50M, 45 min in; wave 3 (512, 1024) to train | wave 2 close-out tonight; wave 3 through the morning of 2026-09-05 |
| desktop | b15 (entropy coef, 40 arms) | wave 1 (0, 0.001) closed ~18:00; 4 waves queued **behind `b12-hof5000` (running since 19:03), `b12-hof30k`, `b13-hof5000`, `b15-hof5000`** | the tool's ~05:13 is training time alone; with three passes ahead at ~40-80 min each, closer to 08:00-10:00 2026-09-05 |
| desktop | b11 hof passes | **done** — `hof5000` over 661 checkpoints, `hof30k` over 22 rows | — |

Behind b15 on the desktop: b16-b21 (target-KL, clip, gradient clip, advantage normalisation, lanes,
shaping), every one at λ 0.99 against **b9's λ 0.99 arms `b9bw`-`b9bz`** (best30 98.3-98.4, density 27.3%).
`attention` is empty. **2026-09-05 12:50: b21 moved from the desktop to the laptop** (`move-batch`): `status.json`'s
new `remaining` read ~31 h on the desktop against ~14 h on the laptop, so the laptop now runs b19, b20, b21
in that order and the desktop b17 then b18.

**b11's 30,000-episode result: a third-place candidate, not a record.** `b11ag-lr1e4-seed3` @33243136 read
**99.4 /30,000** [99.3, 99.4], with its two neighbours at 99.3 and 99.1; `b11ba-lr1e3-seed3` @47382528 and
`b11am-lr2.5e4-seed1` @47644672 read 99.2. That sits below `b10ck`'s 99.65 and 99.55 and above `b9ch`'s
99.30, so it would enter [`hallOfFame/HOF.md`](../hallOfFame/HOF.md) third if promoted — the `hof-promote`
skill's basin check is the step, and it is the user's call. The 22 rows at 30k were the ≥99 /5,000 rows
from 661 measured at 5,000; the counts match the ledger.

**What the two closed waves say.** b14 rollout 64 matches the reference on best30, stability and best row,
and is a little below on density (22.9% vs 27.3%); rollout 32 is below on everything. b15 entropy 0 and
0.001 are the most stable arms trained so far (1.7-1.9% of evals below 80%) at less than half the density
(8.7-11.7%): the spec's condition for moving the default is falsified, and only the annealing schedules in
waves 4-5 can still move the base. Readings in [`charts.md`](charts.md).

**Infrastructure, 2026-09-04 evening.** The laptop joined the router's 6 GHz radio at 18:05 (the stuck
5 GHz association is the suspect; verdict needs a day of load — [`../plans/laptop-wifi.md`](../plans/laptop-wifi.md)).
`ssh the-claw-den` now works from outside the home LAN through a port forward
([`../desktop/README.md`](../desktop/README.md), "Reach the box from outside the home LAN"); this update ran
over a phone hotspot, live charts rsync'd from the box included. And `progress_update` now imports a finished
hall-of-fame pass's files from `results` — b11's had been sitting there unread.

**Standing decision**, unchanged: the γ × λ corner grid — γ {0.99, 0.9975, 0.999} × λ {0.98, 0.99, 0.999},
with lr {3e-4, 2.5e-4} and minibatch {256, 512} as candidate extra axes — runs **after b14 closes**. Every
one-knob sweep since b9 has come back "plateau, base stays": the corner grid is where the next gain is, if
there is one.

## b12 and b13 closed, b14 and b15 in flight, as it read at 2026-09-04 16:30 (superseded)

**b12 (epochs) and b13 (minibatch) closed; b14 (rollout) is on the laptop with wave 1 in stage B; b15
(entropy) is on the desktop at wave 1 of 5.** As of 2026-09-04 16:30:

| box | batch | state | ETA |
|---|---|---|---|
| desktop | b12 (epochs, 40 arms) | **closed** — reading in [`results.md`](results.md), finding in [`findings.md`](findings.md) | — |
| laptop | b13 (minibatch, 32 arms) | **closed** — same | — |
| laptop | b14 (rollout, 24 arms, dequeued from the desktop) | wave 1 (32, 64) trained 10:32-16:14, `b14-stageb` 17 min in; waves 2-3 (192/256, 512/1024) to train | ~midday 2026-09-05 at wave 1's cadence; faster with the box to itself |
| desktop | b15 (entropy coef, 40 arms) | wave 1 (0, 0.001) at ~80% of 50M, 70 min in; 4 waves queued (0.003, 0.005, 0.02, 0.03, four anneal schedules) | ~05:00 2026-09-05 |

Behind b15 on the desktop: b16-b21 (target-KL, clip, gradient clip, advantage normalisation, lanes,
shaping), every one at λ 0.99 against **b9's λ 0.99 arms `b9bw`-`b9bz`** (best30 98.3-98.4, density 27.3%).
`attention` is empty. The laptop's Wi‑Fi was stable through the afternoon after the morning's toggle;
the plan for it is [`../plans/laptop-wifi.md`](../plans/laptop-wifi.md).

**Two more knobs are plateaus, and both killed a prior.** b12: 3-4 epochs is the top, every epoch past 4
costs 2-5 pp of density, stability is *best* at 6-8 (8 was predicted to collapse — b4's 8-epoch collapses
were its `fc (200,100)` net's) and breaks at 12-16 (42% of evals below 80% at 16). b13: minibatch 256-512 is
the plateau (512 holds the batch's 100/500 row), 128 — predicted as the alternative default — is 8 pp below
on every seed, 1024-2048 arrive but noisily. Neither moves the base; 512 and 3 epochs are the only cells
that might ride along.

**Early stage A on the two live batches.** b14 rollout 64 matches the reference on best30 and stability but
reaches ≥98 on 18.6% of evals against 28.9%; rollout 32 is below on everything, as predicted. b15 entropy 0
and 0.001 are the most stable arms trained so far (1.9-2.0% of evals below 80%, a third of the reference)
with a lower top so far (best30 97.6-97.9 against 98.3).

**Hall-of-fame passes.** b10's `hof5000` and `hof30k` ran on the laptop on 2026-09-03 (21 arms, 71 rows at
30,000) and produced the current record, `b10ck` @30523392 at 99.65 /30,000 — see
[`hallOfFame/HOF.md`](../hallOfFame/HOF.md); the "on hold" carried in earlier Now blocks was stale. **b11's
are queued on the desktop** as of 17:21 (`b11-hof5000`, 661 checkpoints ≥99/500 from 25 arms, priority 5;
`b11-hof30k` over its ≥99 rows at seed 7, priority 6), dispatching after b15 wave 1's stage B and before b15
wave 2. **Standing decision**, unchanged: the γ × λ corner grid — γ {0.99, 0.9975, 0.999} × λ {0.98, 0.99,
0.999}, with lr {3e-4, 2.5e-4} and minibatch {256, 512} as candidate extra axes — runs **after b14 closes**. Every one-knob sweep since b9 has come back "plateau, base stays": the
corner grid is where the next gain is, if there is one.

## b11 closed, b12 and b13 in flight, as it read at 2026-09-04 09:45 (superseded)

**b11 closed on the desktop; b12 is on its stage B for wave 3 with two waves to train; b13 finished
training on the laptop and its last close-out is measuring.** As of 2026-09-04 09:45:

| box | batch | state | ETA |
|---|---|---|---|
| desktop | b11 (lr) | **closed**, every wave measured — reading in [`results.md`](results.md) | — |
| desktop | b12 (epochs) | 24 of 40 trained (epochs 1-7 in, `b12-stageb-w3` 32 min in); epochs 8, 10, 12, 16 queued in two waves | ~15:25 today |
| laptop | b13 (minibatch) | 32 of 32 trained; `b13-stageb-w4` (mb 1024, 2048) 20 min in, 26 of 32 arms measured | ~10:30 today |

Behind b12 on the desktop: b14 (rollout) through b21, every one at λ 0.99 against **b9's λ 0.99 arms
`b9bw`-`b9bz`** (best30 98.3-98.4, density 27.3%). `attention` is empty.

**b11's answer: the learning rate is a plateau, not a lever.** 1e-4 through 5e-4 sit at 21-31% density
within seed noise of the reference's 27.3%; 4e-5 is the slow end (2.2%) and 2e-3 the cliff (6.7%). The
surprise is the sign of the stability effect — 8e-4 and 1e-3 were predicted to collapse and are the two most
stable cells (4.6-4.9% of evals below 80%), with a lower ceiling. Base stays 3e-4; 2.5e-4 (30.9%, the best
cell) goes into the corner grid as a second value. Full reading in [`results.md`](results.md), finding in
[`findings.md`](findings.md).

**b12 so far (epochs 1-7):** density climbs 10.8 → 15.7 → 26.5% from 1 to 3 epochs and plateaus at 23-27%
through 6; 1 epoch is the *least* stable cell (4.2% drawdown below 50%), the reverse of its prediction, and
stability keeps improving through 7. **b13 so far (minibatch 32-1024):** monotone rise to a plateau at
256-512 (26.9-28.2%, 512 holds the batch's 100/500 row), 1024 falling to 20.6% on two seeds; stability
U-shaped with the small end noisiest. Both readings in [`charts.md`](charts.md).

**Standing decisions**, unchanged: `hof5000` on b10 is **on hold**; the γ × λ corner grid — γ {0.99, 0.9975,
0.999} × λ {0.98, 0.99, 0.999}, now with lr {3e-4, 2.5e-4} as a candidate third axis — runs **after b14
closes**.

**Tooling notes from this update.** `tools/progress_update.py` now (a) survives an rsync to the box that
hangs rather than aborting the whole update, (b) reads a laptop-run batch's specs from `logs/<batch>specs/`
when `ops` no longer has them, and (c) states such a batch's state from `runs/` and the live pid files —
without which b13 was invisible to the digest. Code change, awaiting review. Separately, the laptop's Wi‑Fi
was found in a stuck association during this update (25-55% loss to the router at ‑55 dBm, cleared by a
Wi‑Fi toggle, the desktop on the same router clean throughout); a hung `git fetch` or `rsync` is that
before it is "off-LAN".

## b11 on wave 1, as it read at 2026-09-03 18:01 (superseded)

**b11 — the learning-rate sweep — has wave 1 trained and its stage B running; nothing is on the laptop.**
As of 2026-09-03 18:01 the box's ledger reads 8 b11 arms done (lr 4e-5 and 1e-4, seeds 1-4), the
`b11-stageb` wave 19 min in, and 24 b11 arms queued behind it in three waves (lr 1.5e-4, 2.5e-4, 5e-4,
8e-4, 1e-3, 2e-3). At b10's cadence of ~2.5 h a wave, b11 closes around **01:40 on 2026-09-04**. The queue
behind it now runs **b12 through b21** — epochs, minibatch, rollout, entropy, target-KL, clip, gradient
clip, advantage normalisation, lanes, shaping — every one at the re-based λ 0.99, every one read against
**b9's λ 0.99 arms `b9bw`-`b9bz`** (best30 98.3-98.4, density 27.3%). `attention` is empty.

**b11 wave 1 at its cap, from the pulled reports:** lr 4e-5 best30 97.0-97.6, `sef` 65-79, still rising —
the slow end the spec predicted; lr 1e-4 best30 97.8-98.6, `sef` 85, at the reference's ceiling on two
seeds. No stage-B row yet; the table in [`charts.md`](charts.md) fills in when the wave's results land.

**The update itself is now a tool.** `tools/progress_update.py` fetches, imports closed waves, pulls live
charts, publishes the site, regenerates the batch tables and `charts.md` sections, and prints a digest;
the skill runs it and writes only the readings, on Sonnet. This is the first update written that way —
compare its readings against the tables before trusting the split.

**Standing decisions** (from the b10 close, unchanged): `hof5000` on b10 is **on hold**; the γ × λ corner
grid — γ {0.99, 0.9975, 0.999} × λ {0.98, 0.99, 0.999}, 4 seeds — runs **after b14 closes**.

## b10 closed and b11 on wave 1, as it read at 2026-09-03 17:10 (superseded)


**b10 — the γ sweep — closed on the desktop 2026-09-03 16:08, and b11 — the learning-rate sweep — is
on wave 1 of 4.** At 17:10 b11's eight arms (lr 4e-5 and 1e-4 x seeds 1-4, `b11aa`-`b11ah`) were ~65%
through 50M, 52 min in; wave 1 lands about 17:40 and its stage B follows. Waves 2-4 (lr 1.5e-4, 2.5e-4,
5e-4, 8e-4, 1e-3, 2e-3) and then b12 (epochs), b13 (minibatch), b14 (rollout) are queued, **all at the
re-based λ 0.99** — so b11-b14's reference cell is **b9's λ 0.99 arms `b9bw`-`b9bz`** (best30 98.3-98.4,
density 27.3%), not b7. Ledger: 298 done, 124 queued, 8 running, `attention` empty, 146 GB free, load 13.
**Nothing is running on the laptop.** The site republished at 17:11 with b10 complete and b11 live.

**b11's first wave at 33M, read against b9's λ 0.99 seeds at their cap:** lr 4e-5 is the slow end as
predicted — best30 93.4-96.4, `sef` 47-68, still climbing; lr 1e-4 is already at best30 97.8-98.6 with
`sef` 72-81, on course for the reference. Neither has a stage-B row yet; wave 1's stage B says more.

**b10's answer: the discount behaves like λ did — monotone to the top of the grid, with the same
stability cost, and the undiscounted end is a cliff.** Record density is zero for every γ ≤ 0.93 (not one
checkpoint of 28 arms passed the stage-A screen), appears at 0.96, then climbs through every value:
0.9% → 3.6% → 10.4% → 17.3% (b7's 0.99) → 19.6% → 25.6% → **30.7% at γ 0.999**, where the batch's best30
(98.55-98.60 at 0.9975-0.999) and its 277 `hof5000` candidates also sit. Drawdown climbs with it, 0.29% at
0.99 → 1.55% at 0.999, as λ's did across b9's plateau. **γ 1.00 is different in kind**: its surviving
checkpoints are the richest in the batch (38.6% of 1,535 rows, two 100/500 rows) but the deployed policy
spends **44% of its post-competence evals below 50% perfect** — the undiscounted critic's targets are
whole-game sums, and the arm collapses and recovers for its entire run. Full table under *Just closed*.

**Decisions this puts in front of us:**

1. **Run `hof5000` on b10 on the idle laptop** — its ≥99/500 rows number 726 (277 at γ 0.999, 178 at 1.00,
   164 at 0.9975, 85 at 0.995), the same size as b9's pass (~76 min), and γ 0.999 has 4,003 → 6,614 stage-B
   rows over b7's cell. Whether a γ 0.999 checkpoint matches `b9ch`'s 99.30 at depth is the question.
   **Recommendation: run it**, ≥99 cut, then `hof30k` on the desktop for whatever clears 99 /5,000.
2. **A γ × λ corner is now the obvious next batch.** b9 found λ 0.99-1.00 at γ 0.99 (27-30%); b10 finds
   γ 0.999 at λ 0.98 (30.7%). Both are one knob off the same cell and neither was run with the other's
   winner. A 3x3 grid — γ {0.99, 0.9975, 0.999} x λ {0.98, 0.99, 0.999} at 4 seeds, 36 arms, ~9 h —
   would say whether the two add, and where the drawdown becomes b4's. It would go behind b14, or ahead of
   b12-b14 if we would rather know this than the epochs/minibatch/rollout answers first.

## Just closed: b10 — density is monotone in γ to 0.999, and γ 1.00 is a cliff

**All eight waves and all eight stage-B passes closed 2026-09-03 16:08**, 64 arms at 50M on the desktop,
22,741 stage-B rows, ~21 h end to end (the `hof30k` wave took ~35 min of it). b10 changes only `discount`
off b7's cell — `fc (320,)`, 4 epochs, **λ 0.98**, lr 3e-4, entropy 0.01 — so **b7aa-b7ad are its γ 0.99
arms**. Drawdown as in b8/b9: median share of post-competence stage-A evals below 50% (and 80%).

| γ | value horizon 1/(1−γ) | rows | ≥98%/500 | per-seed share | ≥99 (`hof5000` cands) | best row | best30 (4 seeds) | sef | drawdown < 50% | < 80% |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|
| 0.70 | 3.3 | 0 | – | never screened | 0 | – | 2.7 5.2 5.1 2.8 | 0.0 | never competent | |
| 0.80 | 5 | 0 | – | never screened | 0 | – | 49.7 24.4 41.8 45.2 | 0.0 | never competent | |
| 0.85 | 6.7 | 0 | – | never screened | 0 | – | 67.2 64.1 66.6 68.0 | 0.0 | 54.4% (2 of 4 reached 80%) | 99.7% |
| 0.90 | 10 | 0 | – | never screened | 0 | – | 81.6 75.8 79.4 81.4 | 3.6 | 38.7% | 95.5% |
| 0.91 | 11.1 | 0 | – | never screened | 0 | – | 85.5 79.9 77.5 84.7 | 5.8 | 26.5% | 92.6% |
| 0.92 | 12.5 | 0 | – | never screened | 0 | – | 85.1 86.6 84.2 83.1 | 12.1 | 18.4% | 85.3% |
| 0.93 | 14.3 | 0 | – | never screened | 0 | – | 86.9 88.5 87.1 87.7 | 21.0 | 13.9% | 74.0% |
| 0.94 | 16.7 | 11 | 0.0% | 0 0 0 0 | 0 | 93.2 | 90.4 89.0 90.2 89.8 | 34.5 | 4.85% | 56.8% |
| 0.95 | 20 | 95 | 0.0% | 0 0 0 0 | 0 | 97.8 | 93.6 91.6 93.2 93.0 | 51.5 | 3.35% | 44.7% |
| 0.96 | 25 | 319 | 0.9% | 4.3 0 0 0 | 0 | 98.4 | 95.6 93.4 94.9 94.7 | 63.9 | 2.60% | 30.0% |
| 0.97 | 33 | 756 | 3.6% | 0.8 5.1 2.0 7.1 | 2 | 99.4 | 96.4 94.7 95.4 95.6 | 75.6 | 0.28% | 19.8% |
| 0.98 | 50 | 2,540 | 10.4% | 6.0 14.1 8.1 13.1 | 20 | 99.6 | 97.5 97.1 97.3 97.3 | 86.5 | 0.14% | 8.3% |
| 0.99 (b7) | 100 | 4,003 | 17.3% | 18.5 19.0 16.5 15.1 | 50 | 99.6 | 97.8 97.8 97.7 97.7 | 90.9 | 0.29% | 6.2% |
| 0.995 | 200 | 5,027 | 19.6% | 19.0 17.8 20.5 20.7 | 85 | **100.0** | 98.1 97.7 98.5 98.1 | 92.5 | 0.78% | **4.5%** |
| **0.9975** | 400 | 5,844 | 25.6% | 24.3 29.0 20.3 28.5 | 164 | 99.8 | **98.6 98.4 98.7 98.7** | **92.6** | 1.30% | 4.7% |
| **0.999** | 1,000 | **6,614** | **30.7%** | 21.7 22.9 39.2 35.4 | **277** | 99.8 | 98.5 98.2 98.7 98.6 | 91.7 | 1.55% | 4.9% |
| 1.00 | ∞ | 1,535 | 38.6% | 36.4 45.8 39.4 33.9 | 178 | **100.0** | 98.4 97.3 98.7 98.3 | **39.2** | **44.4%** | 63.6% |

**Four readings:**

- **Below γ 0.94 the endgame is unreachable at 50M, and it is γ itself, not the GAE horizon.** 28 arms
  across seven values produced no screened checkpoint at all; best30 climbs 4 → 40 → 66 → 80 → 82 → 85 →
  88. b9's λ 0.90 at γ 0.99 sits at the same *advantage* horizon as b10's γ 0.92 at λ 0.98 (9-10 steps)
  and reached 96.75 with 3.9% density against 84.75 and none: a discounted value target that cannot see
  the +100 ten steps out cannot value the endgame, whatever λ does to the advantage.
- **From 0.96 to 0.999 density is monotone**, 0.9% → 30.7%, with the base at 0.99 in the middle of the
  ramp rather than at a peak — the same shape b9 found for λ, and the second time the default has been
  left with a factor of 1.8 on the table. best30 peaks at 0.9975-0.999 (98.60, 98.55) with every seed
  ≥98.2. γ 0.999 also produces the most stage-B rows of any cell yet (6,614) and the most `hof5000`
  candidates (277 at ≥99/500).
- **Stability moves the other way, again.** Drawdown below 50% is ≤0.3% from 0.97 to 0.99, then 0.78 →
  1.30 → 1.55% up the plateau; the sub-80% share bottoms at 4.5-4.9% for 0.995-0.999. Same trade-off as
  b8 and b9, now on a third knob.
- **γ 1.00 collapses half the time and holds records the other half.** Median drawdown **44.4%**, `sef`
  39.2, only 1,535 rows screened — and 38.6% of those are ≥98, two are 100/500, and 178 are `hof5000`
  candidates. Undiscounted, the critic's target is the whole-game return and the rollout-boundary
  bootstrap carries full weight (the spec predicted both); the policy oscillates between a near-perfect
  regime and a broken one for its whole run. Not a base to build on; possibly a source of checkpoints.

**Two more things.** `sef` ranks the top of this sweep backwards too (92.6 at 0.9975 vs 91.7 at 0.999,
and 39.2 at the densest cell) — fourth reproduction. And b3's single-seed call on γ 0.9975 ("the
stability candidate") was the right cell for the wrong reason: it is the batch's best30 and *not* its
most stable. Per-arm numbers in [`results.md`](results.md), 121 panels in [`charts.md`](charts.md).

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

## b10 mid-flight, as it read at 2026-09-03 01:20 (superseded)

**b10 — the γ sweep — is on the desktop, wave 4 of 8, and the laptop is idle.** At 2026-09-03 01:20
the eight γ 0.93 and γ 0.94 arms (`b10ay`-`b10bf`) were 39-44% through 50M, 59 min in, so wave 4 lands
around 02:40 and its stage B follows automatically; waves 5-8 (γ 0.95, 0.96, 0.97, 0.98, 0.995, 0.9975,
0.999, 1.00) are queued behind it, then b11-b14 at their re-based λ 0.99. b10 should close mid-afternoon
2026-09-03. Ledger: 252 done, 164 queued, 8 running, `attention` empty, 153 GB free, load 14. The
laptop's only process is the stage-B chart window left from b9's `hof5000` pass, which is free to close.

**New record, 2026-09-03: `b9ch-lam999-seed4` @47251456 at 99.30% /30,000 on seed 7**, promoted to
[`hallOfFame/HOF.md`](../hallOfFame/HOF.md). The desktop `hof30k` pass re-measured b9's 33 checkpoints at
≥99 /5,000 at 30,000 episodes, 10:21-10:55, exit 0: **18 beat `b5h`'s 98.96**, the mean 5,000 → 30,000
drop was −0.14 pp, and the winner's four neighbours read 99.20. Six plateau arms hold a checkpoint above
the old record; one is promoted. `record_gif.py`'s `HOF_RECORD` now names it.

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
