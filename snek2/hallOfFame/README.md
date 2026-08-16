# Hall of Fame

The best Snake policies this project has produced, preserved as standalone checkpoints so
they survive whatever happens to `savedPolicies/`.

**Why this folder exists.** Training writes a checkpoint every 1000 steps and keeps the most
recent `max_to_keep` (currently 10000), so a long run eventually **deletes its own best
checkpoint**. That has already cost real evidence — `b5c-schlongIS`'s 17.0% peak became
permanently unmeasurable once the arm passed 1.28M steps. Copies here are outside that
rotation and are not deleted by anything.

## The current record: **99.0% over 500 episodes**, `b29b-chase10g75seed2-ckpt1447000`

Batches 11-29 train on the **30-value observation vector** — the current one (era `b09c616`), after the
following-tail block (26-28) and food-space (29) landed 2026-08-03 — so **these twelve entries all load on
`master` as it stands.** Each came from a batch close-out or, for batches 24 and 29, its 500-episode HOF
re-measurement. Most are the default `FC_LAYERS=50,100,50`; **the batch-24 and batch-29 entries are the only
non-default architecture here (`fc 320`)** and load because `arch.json` (copied in with each) rebuilds the
recorded net — see [`../CLAUDE.md`](../CLAUDE.md) on the `arch.json` sidecar.

**`b29b` @1447k edges `b24d` (98.0%/500) on the point estimate and their CIs overlap** — a narrow lead, taken
as the record under this folder's 500-episode standard exactly as `b24d` was taken over `b18b`. What sets it
apart from every prior record is that it is a **region**, not an isolated point: 18 checkpoints of the same
arm hold ≥98%/500 across 1446k-1529k (its sibling `b29a` holds 3 more), where `b24d`/`b18b` were lone peaks.
It is also the first record from **reward shaping** — chase-safe potential-based shaping at `c=0.10`, gate 75.

**Every entry must clear a fresh ≥95-episode re-measurement before it is added** — the 500-episode HOF eval
for batch 24, a close-out re-measure for the rest — because a selected /100 high is inflated ~5-6 pp by
selection and mostly does not survive (see the batch-24 note below: 199 ≥97%/100 candidates, only 9 held
≥97%/500). A checkpoint enters this folder on its *re-measured* number, never its close-out /100.

| checkpoint | measured | config |
|---|---|---|
| **`b29b-chase10g75seed2-ckpt1447000`** | **495/500 = 99.0%** (CI **97.7-99.6**) — **the record**; it *rose* from 98.0/100 on re-measurement, the genuine-region signature, and is the head of an **18-checkpoint ≥98%/500 band** (1446k-1529k) — a region, not a point. First record from reward shaping (`c=0.10`, gate 75) and from `fc 320` chase-safe. 98/100 on the copy here. Added 2026-08-16; see below | `FC_LAYERS=320`, IS off (`SNEK_IS_WEIGHTS=0`), `td_error`, `DISCOUNT=0.9975`, `TARGET_UPDATE_PERIOD=1000`, forking on, `FOOD_DISTANCE_REWARD=0`, **`CHASE_SAFE_SHAPING=0.1`, gate 75**, `SNEK_SEED=2` |
| **`b24d-fc320noisseed4-ckpt1342000`** | **490/500 = 98.0%** (CI **96.4-98.9**) — the prior record (2026-08-13), within CI of `b29b`; it *rose* from 97.0/100 on re-measurement, the genuine-region signature. First non-default-arch record (`fc 320`). See below | `FC_LAYERS=320`, IS off (`SNEK_IS_WEIGHTS=0`), `td_error`, `DISCOUNT=0.9975`, `TARGET_UPDATE_PERIOD=1000`, forking on, `FOOD_DISTANCE_REWARD=0`, `SNEK_SEED=4` |
| `b24b-fc320noisseed2-ckpt2860000` | **490/500 = 98.0%** (CI 96.4-98.9) — ties the record from a second seed, a late checkpoint (2.86M). Added 2026-08-13 | same as above except `SNEK_SEED=2` |
| **`b18b-tgt1000seed2-ckpt1588000`** | **683/700 = 97.6%** (CI **96.1-98.5**) — the prior record and deepest-measured strong checkpoint; the first selected high in this project to survive re-measurement; see below | `TARGET_UPDATE_PERIOD=1000`, forking on, `DISCOUNT=0.9975`, `FOOD_DISTANCE_REWARD=0`, `td_loss`/no-IS, `SNEK_SEED=2` |
| `b17b-forkseed2-ckpt1248000` | **483/500 = 96.6%** (CI 94.6-97.9) on fresh episodes — 2nd-highest measured here; a 99/100 close-out read re-measured at 500, and **95/100 on the copy in this folder** (CI 88.8-97.8), confirming it loads and plays. Not the record — below `b18b` @1588k and CIs overlap — but higher than `b17b` @1190k, so the better step of this arm. Added 2026-08-12 | forking on (`FORK_BRANCHES=4`), `DISCOUNT=0.9975`, `FOOD_DISTANCE_REWARD=0`, `SNEK_SEED=2` |
| `b18b-tgt1000seed2-ckpt1601000` | 663/700 = **94.7%** (CI 92.8-96.1) — 13k steps after the record and 2.9 pp below it | same as above |
| `b17b-forkseed2-ckpt1190000` | **4825/5120 = 94.2%** (CI **93.6-94.8**) — the most heavily measured checkpoint in the project, see below | forking on (`FORK_BRANCHES=4`), `DISCOUNT=0.9975`, `FOOD_DISTANCE_REWARD=0`, `SNEK_SEED=2` |
| `b15b-nstep3seed2-ckpt3245000` | **97/100** selected, **182/200** on an independent re-run → **93.0% over 300** (CI 89.5-95.4) | `N_STEP_UPDATE=3`, `DISCOUNT=0.995`, `GUIDED_FRACTION=0.8`, `SNEK_SEED=2` |
| `b14a-disc9975seed1-ckpt3702000` | 96/100 selected, **91/100** on an independent re-run → **93.5% over 200** (CI 89.2-96.2) | `DISCOUNT=0.9975`, `GUIDED_FRACTION=0.8`, `SNEK_SEED=1` |
| `b11b-obs30seed2-ckpt855000` | 96/100 (CI 90.2-98.4), top-3 95.3%, **~94% shrunk** | `DISCOUNT=0.995`, `SNEK_SEED=2` |
| `b13d-shieldseed4-ckpt986000` | 95/100 (CI 88.8-97.8), top-3 93.3% | + eps handover 0.0125, `GUIDED_FRACTION=0.5`, `SNEK_SEED=4` |
| `b11a-obs30seed1-ckpt671000` | 94/100 (CI 87.5-97.2), top-3 93.3%, ~90% shrunk | `DISCOUNT=0.995`, `SNEK_SEED=1` |

**2026-08-12 — the shrink pattern held again.** Ten high close-out reads (five each from `b23b` and
`b17b`) re-measured at **500 fresh episodes**: every 96-99/100 read fell to **92.4-96.6%**. `b17b`
@1248k held highest (96.6%, added above); the new β-ladder arm `b23b` topped out at 95.2% (@771k), and
its close-out-*selected* step (@777k, 97/100) was the **worst** of its cluster on fresh episodes
(92.4%) — a clean example of why a selected high must be re-measured before it is trusted. Nothing beat
the 97.6% record.

> **‡ Re-measured 2026-08-11 on games none of these had seen, and the numbers above are a ceiling.**
> 100-200 fresh greedy episodes each, via `hyperparamTuning/perDiagnostics/behaviour_profile.py`:
> `b17b` @1190k read **95.5% /200** against its recorded 94.24% /5120 — agreement, and the check that
> validates the protocol. `b18b` @1588k read **94.0% /200** against its recorded 97.57% /700, about 2σ
> low. The other five read **2.0-7.5 pp** below their recorded figures, though they shared one game set
> that the two cross-checks show runs ~1-2 pp hard, so they are not independent observations.
>
> **Read the entries below as selected highs.** Selecting the maximum of 20-50 hundred-episode reads
> inflates by 5-6 pp on its own, which is enough to explain the whole shrinkage history without any
> other mechanism — see
> [`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md#-and-the-corollary-most-of-a-selected-high-is-selection).
> Nothing here needs changing: the entries are the right checkpoints to keep, and the ranking among
> them is roughly preserved. It is the absolute percentages that are optimistic.

### ‡ `b29b` @1447000 — the record, the first from reward shaping, and the first that is a *region*

Added 2026-08-16. Promoted under the standing rule: **a checkpoint that beats the all-time best on
re-measurement belongs here.** It came out of batch 29's auto HOF-500 (gate 98, early-abandon) — chase-safe
potential-based shaping at `c=0.10`, **gate 75**, otherwise b24's config on `fc 320`.

| measurement | result | why it exists |
|---|---|---|
| close-out, 100 episodes (desktop) | 98/100 = **98.0%** | the selected /100 read |
| **HOF-500, 500 fresh episodes, gate 98 (desktop, auto)** | 495/500 = **99.0%** (97.7-99.6) | the record figure — it *rose* 1 pp rather than shrinking |
| 100 fresh, **on the copy in this folder** (laptop) | 98/100 = **98.0%** (93.0-99.4), avg score 94.7 | confirms the copy loads via `arch.json` and plays like a champion |

**It edges `b24d` @1342k (98.0%/500) on the point estimate; the intervals overlap** (97.7-99.6 vs 96.4-98.9),
so it is a narrow lead taken as confirmed under this folder's 500-episode standard — the same call `b24d` got
over `b18b`. The tell that it is real, not a lucky draw, is the same as every record here: it **rose** on
re-measurement (98.0/100 → 99.0/500), the genuine-region signature.

**What is new is that it really is a region.** Where `b24d` and `b18b` were lone peaks whose neighbours fell
5-6 pp away, `b29b` holds **18 checkpoints at ≥98%/500 across 1446k-1529k**, and its seed-1 sibling `b29a`
holds 3 more (best 98.4%/500 @1347k) — 21 record-tier checkpoints across two seeds, where the prior
record-holder `b24` produced only 2 isolated ones across four. This is the first time the project's top tier
appears as a plateau rather than a spike. It loads only because `arch.json` (era `b09c616`, `obs_len` 30)
rebuilds the recorded 320-wide net. Full batch write-up:
[`../hyperparamTuning/completedRuns.md`](../hyperparamTuning/completedRuns.md#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region).

### ‡ `b24d` @1342000 — the prior record, and the first from a wide (`fc 320`) net

Added 2026-08-13. Promoted under the standing rule: **a checkpoint that beats the all-time best on
re-measurement belongs here.** It came out of batch 24's 500-episode HOF eval (gate 97, early-abandon), which
re-measured every one of the batch's **199 ≥97%/100 checkpoints** and kept only the 9 that held ≥97% over 500
fresh episodes.

| measurement | result | why it exists |
|---|---|---|
| close-out, 100 episodes (desktop) | 97/100 = **97.0%** | the selected /100 read — near the batch's best, not the flashiest |
| **HOF-500, 500 fresh episodes, gate 97 (desktop)** | 490/500 = **98.0%** (96.4-98.9) | the record figure — it *rose* 1 pp rather than shrinking |
| 100 fresh, **on the copy in this folder** (laptop) | 93/100 = **93.0%** (86.3-96.6), avg score 94.5 | confirms the copy loads via `arch.json` and plays like a champion |

**It edges `b18b` @1588k (97.6%/700) on the point estimate; the intervals overlap**, so it is a narrow lead
taken as confirmed under this folder's 500-episode standard (we did not re-run for the 700-vs-500 gap). The
tell that it is a real region, not a lucky draw, is that it **rose** on re-measurement (97.0/100 → 98.0/500)
— the same signature `b18b` showed (98/100 → 97.4/500), and the opposite of the batch's inflated highs: `b24a`
produced two 100.0%/100 checkpoints and **zero** survivors at 500 episodes. `b24b` @2860k ties this at
98.0%/500 from a second seed, so it is in the folder too. Both are **`fc 320`, the first non-default
architecture here** — they load only because `arch.json` (era `b09c616`, `obs_len` 30) rebuilds the recorded
320-wide net; a copy without it would fail loudly (`ArchMismatch`) rather than silently, per the 2026-08-11
sidecar. Full batch write-up:
[`../hyperparamTuning/completedRuns.md`](../hyperparamTuning/completedRuns.md#batch-24--fc-width-320-under-is-off-the-first-architecture-result-and-a-new-record).

### ‡ `b18b` @1588000 — the first selected high that did *not* shrink

Added 2026-08-09. Promoted under the standing rule: **a checkpoint that beats the all-time best on
re-measurement belongs here.** Two independent runs, 700 fresh episodes, no selection between them:

| measurement | result | why it exists |
|---|---|---|
| 500 fresh episodes, gate off, flat protocol | 487/500 = **97.4%** (95.6-98.5) | the verification of the close-out's 98/100 |
| 200 fresh, **on the copy in this folder** | 196/200 = **98.0%** (95.0-99.2) | confirms the copy loads and plays, per this folder's standard |
| **pooled** | **683/700 = 97.57%** (CI **96.1-98.5**) | the figure to quote |

**It beats `b17b`'s 94.24% with non-overlapping intervals** (96.1-98.5 against 93.6-94.8), so this is a
real move in the ceiling rather than a sampling artifact — the first since the 30-value vector landed.

**Why it matters more than the number.** Every previous selected high in this project shrank on
re-measurement: `b17b` 99/100 → 94.2%, `b15b` 97/100 → 93.0%, `b14a` 96/100 → 93.5%, `b11b` 96/100 →
~94%. This one was selected at 98/100 and re-measures at 97.6% over 7x the episodes — **a 0.6 pp
change**. It is the first checkpoint here whose close-out row was approximately correct.

**‡ It is a narrow peak, not a strong region, and that is the caveat on the entry.** The eight
neighbouring checkpoints measured in the same sweep are all well below it:

| step | 500-episode rate | distance from the record |
|---|---|---|
| **1588000** | **97.4%** | — |
| 1601000 | 95.0% | +13k |
| 1600000 | 92.8% | +12k |
| 1578000 | 91.6% | **−10k** |
| 561000 | 92.2% | far |

**1578000 is 10k steps away and reads 5.8 pp lower**, so nothing here licenses a claim about the
*region* — only about this checkpoint. That is the same lesson `b17b` taught, and it survives the
better result: a selected row describes the checkpoint the screen liked, never its neighbourhood.

### ‡ `b17b` @1190000 — the first entry measured deeply *before* being called a record

Added 2026-08-08. Three independent measurements, 700 episodes, no selection between them:

| measurement | result | why it exists |
|---|---|---|
| 500 fresh episodes, gate off | 475/500 = 95.0% (92.7-96.6) | the re-measurement that tested the close-out's claim |
| 100 fresh, position-chosen grid | 96/100 = 96.0% (90.2-98.4) | this step also fell on a blind every-10k grid, so it is *not* selection-conditioned |
| 100 fresh, **on the copy in this folder** | 91/100 = 91.0% (83.8-95.2) | confirms the copy loads and plays, per this folder's standard |
| 1000 fresh, independent collect path | 955/1000 = 95.5% (94.0-96.6) | half of the paired test that validated `EVAL_INDEPENDENT` |
| 1000 fresh, batched collect path | 939/1000 = 93.9% (92.2-95.2) | the other half — the two paths agree, p=0.110 |
| + 2,520 more from benchmark runs | — | every one a real 100-episode measurement, kept rather than discarded |
| **pooled** | **4825/5120 = 94.24%** (CI **93.6-94.8**) | the figure to quote |

**±0.6 pp is a tighter interval than any other entry in this folder has ever had**, and it exists as a
by-product rather than by design: validating the independent-worker change meant measuring one
checkpoint thousands of times, and those episodes are real measurements, so they were pooled instead
of thrown away. The earlier 700-episode figure of 94.57% is superseded by this, not contradicted —
94.57% sits inside 93.6-94.8.

**Split by collect path, as a check on the change**: batched **94.03%** (3479/3700) against independent
**94.79%** (1346/1420), difference +0.76 pp, **p=0.295**. No detectable difference, which is what
`eval_workers.py`'s quota design is meant to guarantee.

**Its arm's close-out originally read 99/100 at a different step (1205000), and that did not
survive.** Re-measured at 500 episodes, 1205k fell to **92.4%**, and all four of the arm's ≥98%
rows fell a mean of 5.05 pp. The reason is in
[`../hyperparamTuning/archive/runs-archive.md`](../hyperparamTuning/archive/runs-archive.md#-the-reasoning-error-a-selected-sample-cannot-defend-itself-against-selection):
an argument that the arm's 26 full-length rows averaging 96.2% proved the 99 was real was
**circular**, because those rows reached full depth by screening well. A position-chosen grid over
the same region reads **84.06%**, not 96.2% — a 12 pp selection effect, measured.

**1190000 is the entry rather than 1205000 for exactly that reason.** It is the step that reads
highest on *fresh* episodes and the only one corroborated by a sample that was not selected on the
quantity being measured. It also lands on the blind grid, which no other champion in this folder can
claim.

**It reached ~95% at 1.19M steps**, against `b15b`'s 3.2M and `b14a`'s 3.7M for ~93.5% — a third of
the training for a slightly better policy, and the clearest thing batch 17 produced. Note the batch
itself was a **null** on its pre-registered metric (`sef` -1.67 pp, p=0.875), so this is one arm's
achievement and not evidence that forking works.

**Read the record as ~94-95%, narrowly ahead of the 93-94% tie below**, with overlapping intervals.
Better, not a different class of policy.

**Two entries now have independent second measurements, and both told the same story.** `b14a` was
selected at 96/100 and re-read 91/100; `b15b` was selected at 97/100 and re-read 182/200 = 91.0%.
Neither original is wrong — the pooled figures, 93.5% and 93.0%, are simply the better estimates, and
the drop is the winner's curse, since each was the maximum over dozens to hundreds of full-length
measurements in its own arm. `b11b`'s ~94.0% comes from modelling rather than a re-run and should be
read the same way.

**So `b15b`, `b14a` and `b11b` should be read as a tie around 93-94%, not as a record and two
challengers.** Their intervals overlap almost entirely. `b17b` above sits narrowly ahead of all three
and overlaps them too, so **no batch since 11 has been shown to raise the ceiling by a clear
margin** — what has changed is how fast the frontier is reached (1.19M against 3.2-3.7M).

**‡ A high selected reading does not indicate a near-perfect policy**, and `b15b` is the clearest
demonstration in this folder. It produced **8 checkpoints reading ≥95%**, more than any other arm —
but its 94 full-length rows have **mean 90.7%**, and for a population centred at 90% noise alone
predicts ~5.4 readings of ≥95 per 94 rows. The cluster is a sampling tail. Always re-measure before
calling something a record.

**Verified by loading the copies in this folder**, not just the originals. `b15b`'s copy was
re-measured over **200** episodes and `b14a`'s over 100 — the strongest verification done here, and worth
preferring: it confirms the copy loads *and* sharpens the estimate. `b11b`'s copy re-measured
19/20 perfect (95.0%, avg score 94.8); `b13d`'s re-measured **17/20 perfect (85.0%, avg score 94.0,
min 87)** from `savedPolicies/champion_b13d/` before that scratch directory was removed. Worth doing
every time, given the silent-failure history below — and note what a 20-episode check is worth: 17/20
has a 95% interval of 64.0-94.8%, so it confirms the checkpoint *loads and plays like a champion*
without re-establishing its rate.

**`b17b` is the first entry whose *arm-level* result was a null.** Batch 17's forking change did not
beat its control; this checkpoint is here because a ~95% policy on the current environment is worth
preserving. The same was true of `b13d` below, and it is worth noticing the pattern: **two of the six
current-era entries come from batches that established nothing.** Champion quality and batch verdicts
are close to independent.

**`b13d` is the first entry from a post-epsilon-rewrite arm.** It did not raise the ceiling — batch 13
came back null against batch 11 on five metrics — so it is here because a 95% policy on the current
environment is worth preserving, not because anything improved.

The **shrunk** figure is the winner's-curse correction — each is the maximum over 148-204 full-length
measurements in its own arm, so some of the headline is luck. Shrinking against a Beta prior fitted
on each arm's *unselected* graph-100% rows gives 94.0% and 90.2%. Use those when comparing to
anything that was not selected the same way.

**That method no longer works for arms measured under an abandonment gate**, which is why `b14a` and
`b15b` have re-measurements instead of shrunk figures: the gate truncates the unselected graph-100%
rows the prior was fitted on, and their rates are biased low by optional stopping. For any future
champion, **re-measure at 100+ episodes** rather than trying to shrink. Prefer 200 — `b15b`'s pooled
interval is 89.5-95.4 against `b14a`'s 89.2-96.2 for the same effort per episode. See
[`../hyperparamTuning/archive/batches1-11.md`](../hyperparamTuning/archive/batches1-11.md#a-new-best-measured-checkpoint-b11b-855000-96100).

**Every arm here had long since peaked when it was stopped**, which is the entire reason this folder
exists. The two batch-11 arms were still healthy at 3.19M and 3.56M but `b11a` had given up 42 pp
since 678k; `b13d` lost 44 pp after ~1M; `b14a` lost 25 pp after 3.70M. The checkpoints preserved
here are from the peak, not from where the run ended.

**`b14a` is also the latest peak on record** — 3702000, against 855k, 671k, 986k and `b15b`'s 3245k.
That is the observation that moved `SNEK_MAX_STEPS` to 10M, and it means the habit of stopping arms
near 3.5M was costing checkpoints, not just steps. `b15b`'s own arm ran to 5.75M and had eight
checkpoints reading ≥95% spread from 3245k to 4594k, so the good region is wide, not a single spike.

**The 30-value era starts at `b09c616`** ("Two new observations, a three-stage close-out, and
`EVAL_CONFIRM_COUNT=100`"), which is the commit to check out if a future change moves the vector
again — the same role `450e66e` plays for the 26-value era and `e4514a8` for the 20-value one. Both
new blocks already carry their final polarity in that commit (1 = good at 26-28 and 29), so there is
no sub-era to worry about, and no checkpoint exists from the few hours the reversed version was live
because batch 11 launched after the flip.

**These five are also the only entries whose arm can still be resumed at all**, since every earlier
batch trained on a narrower vector. `SNEK_SEED` is recorded per arm in `runs/<policy>.md` (a
batch-11 first), which makes a run **describable, not reproducible** — measured 2026-08-07, two arms
on the same seed and config diverge in their weights within the first 1000 steps. See
[`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md#a-seed-does-not-reproduce-a-run-and-the-reason-is-the-replay-buffer).
A resume diverges for the additional reason that the buffer and RNG state are not checkpointed.

## The previous record: 95%, on the environment of 2026-08-02 (superseded 2026-08-03)

Batch 10 was a fresh baseline on the environment left by 2026-08-02's seven observation/reward
changes (fatal-move zeroing, the wall/body-hugging observation, normalized group count, the
corrected starve/length split, the terminal-discount fix, safe-to-chase-food, and the audit that
started the day) — the first arms ever to train end-to-end on it, since batch 9 predates all
seven. All four seeds beat every post-audit result on record before being stopped 2026-08-03 to
make room for further changes; two were measured with `eval_checkpoints.py` first:

| checkpoint | measured | config |
|---|---|---|
| `b10d-disc995seed4-ckpt1815000` | **95.0%** (95/100, CI 88.8-97.8), top-3 93.3%, pooled 74.5%/24600 | `DISCOUNT=0.995` |
| `b10b-disc995seed2-ckpt1157000` | 87.0% (87/100, CI 79.0-92.2), top-3 86.3%, pooled 70.4%/12000 | `DISCOUNT=0.995` |

**These no longer load on `master` either, as of 2026-08-03.** Two observations were added that
day — following-tail (26 -> 29) and food-space (29 -> 30) — so batch 10's checkpoints now fail the
same way every entry below does:

```
ValueError: Shapes (30, 50) and (26, 50) are incompatible
```

They held the "runs on master today" status for one day. To run one, check out a commit before
those landed — `450e66e` is the last one with a 26-value vector.

Both measurements were taken mid-run (the arms kept training for another ~2.5-2.9M steps
afterward) and the close-out re-measurement has since found better checkpoints in both arms —
90% @1501000 in `b10b` and 93% @1695000 in `b10d`, the latter within noise of the 95% above.
See [`../hyperparamTuning/completedRuns.md`](../hyperparamTuning/completedRuns.md) for the
batch's full results and whichever numbers are most current.

## The entries below predate 2026-08-02 and do not run on `master`

The observation vector is **30 values** and these were trained on 20, so the first layer's shape
no longer matches and restoring one fails immediately:

```
ValueError: Shapes (30, 50) and (20, 50) are incompatible
```

To run one by hand, check out the last commit whose observation matches them:

```
git checkout e4514a8    # "Fix head_with_tail to advance the tail"
```

**Do not assume a matching width means a working checkpoint.** For part of 2026-08-02 the vector
was coincidentally back at 20 while indices 18 and 19 meant entirely different things, and in that
window these checkpoints restored with no warning at all and played like beginners — the champion
below scored **0, 0, 1** over three episodes against **90.3%** at `e4514a8`. Nothing checks that
values still mean what they meant; only the length is checked. If the count ever returns to 20,
that silent failure comes back.

The `new env` column below was measured at `e4514a8`. The entries stay regardless: they are the
record of what this project achieved, and the weights are still the weights.

## Entries

**Superseded as the project's best by the batch-10 entries above** — this section's own
analysis of *why* is left as written, since it is still the correct read of these four
relative to each other and to the pre-08-02 environment. Read "best checkpoint this project
has" below as "best of these four", not literally.

Every entry is measured on **both** environments, because the 2026-08-01 audit changed two
observation components and the reward. `old env` is what the policy achieved when it was
trained; `new env` is the same weights re-measured on the environment that ran on 2026-08-02,
before the 21-value change. Read `new env` for what a checkpoint does at commit `e4514a8`.

| checkpoint | old env | new env | config |
|---|---|---|---|
| **`b8f-disc9975seed2-ckpt3149000`** | 84.3% pooled /300 | **82.0%** (82/100, CI 73.3-88.3), avg 92.0 | `DISCOUNT=0.9975` |
| `b8f-disc9975seed2-ckpt2816000` | **88.9% pooled /360**, peak 92.0% | 73.0% (73/100, CI 63.6-80.7), avg 86.8 | `DISCOUNT=0.9975` |
| `b8f-disc9975seed2-ckpt2581000` | 87.6% pooled /170 | 63.0% (63/100, CI 53.2-71.8) | `DISCOUNT=0.9975` |
| `b8d-disc995clip-ckpt2538000` | 73.3% pooled /300, peak 80.0% | 61.0% (61/100, CI 51.0-70.2), avg 86.5 | `DISCOUNT=0.995` + `GRADIENT_CLIPPING=10` |

"Perfect rate" is the share of episodes where the snake fills the board, over greedy episodes;
see [`../hyperparamTuning/findings.md`](../hyperparamTuning/findings.md). Old-env figures are
**pooled over every measurement of that step**, which is why `2816000` reads 88.9% rather than
the 92.0% single run it is famous for, and `2538000` reads 73.3% rather than 80.0% — both of
those were the high draw of three (92/88.3/83/92 and 70/80/70). The peak is given alongside.

> **The two columns rank the entries differently, and that is the point.** `3149000` is now the
> best checkpoint this project has, at 82%, despite ranking third on the old environment.
> `2816000`, the famous 92%, reads 73% today. Which checkpoint is "best" depends on the
> environment measuring it, so a record is only ever a record for one of them.
>
> The drop is not evidence the audit was harmful: these policies were trained to read features
> whose meaning then changed, so being off-distribution costs them. Across 72 checkpoints
> measured on both, the loss was ~10 points and 49 of 52 got worse — systematic, and roughly
> equal for both arms. Whether the corrected observations *train* better is what batch 9 tests.

Four entries, kept deliberately.

- **`3149000`** is the best policy that exists on the current environment, 82 of 100. Added
  2026-08-02 after the re-measurement, which is also when it stopped being just b8f's
  third-ranked checkpoint.
- **`2816000`** won 92 of 100 on the old environment and was the project record for a day. Kept
  as the historical high-water mark, and as the clearest single illustration of what the audit
  cost a policy trained before it.
- **`2581000`** is kept because it was independently corroborated on the old environment (88/100
  and 90/100) when reproducibility of the 100-episode measurement was still an open question.
- **`b8d`'s `2538000`** is the best from a **different config**, so the clipping-vs-no-clipping
  comparison stays reproducible if either is revisited.

## Running a checkpoint manually

These are `tf.train.Checkpoint` files, not a SavedModel, so they need the same network and
agent that produced them. The scripts in `snek2/` already build it.

The eval script locates checkpoints under `savedPolicies/<policy_name>/`, so copy an entry
into a policy directory of its own and evaluate it by explicit step:

```
cd /Users/tony_wang/Projects/Snek/snek2
conda activate snek

mkdir -p savedPolicies/champion
cp hallOfFame/b8f-disc9975seed2-ckpt2816000/* savedPolicies/champion/

EVAL_RENDER=1 PYTHONPATH=. python -u eval_checkpoints.py champion 2816000
```

**`EVAL_RENDER=1` is what opens the window** — worker 0 renders a game while the other nine
run headless. Watching one game is the point of running this by hand, so leave it on here.
Without it every worker is headless and you get numbers and no window.

Rendering is off by default because it is the slowest thing in an eval: 163us per game step
headless against 6050us in a real window, and since all workers step together the rendering
one paces the whole round. A 30-episode eval takes 14s headless and 70s with the window.
That is the right trade for an unattended close-out and the wrong one for watching a game.

Useful environment variables:

| variable | default | effect |
|---|---|---|
| `EVAL_EPISODES` | 100 | episodes to measure, rounded up to whole rounds |
| `EVAL_WORKERS` | 10 | parallel envs; `1` with `EVAL_RENDER=1` gives a single visible game |
| `EVAL_RENDER` | 0 | `1` shows a game in a window, at ~5x the wall clock |
| `EVAL_PERFECT_WAIT_MS` | 400 | pause on a win — raise to ~2000 to actually see it |
| `EVAL_OUT_SUFFIX` | none | appended to the output filename |

To just watch one game end to end, with a visible pause on the win:

```
EVAL_RENDER=1 EVAL_EPISODES=1 EVAL_WORKERS=1 EVAL_PERFECT_WAIT_MS=2000 \
  PYTHONPATH=. python -u eval_checkpoints.py champion 2816000
```

Results are written to `runs/champion_checkpoint_evals.json`.

### Things that look broken and are not

- **The window stops mid-game and closes.** A round runs until every worker finishes one
  episode; workers that finish early are auto-reset into extra episodes that are *not*
  counted, so the visible worker is usually part-way through a throwaway game when the round
  ends. Only each worker's first episode of a round is scored.
- **`OSError: Bad file descriptor` on exit.** Multiprocessing shutdown noise from
  `ParallelPyEnvironment`, printed after the results are written. Harmless.
- **A `Gym has been unmaintained since 2022` notice at startup.** Inert; the upgrade is not
  available for this dependency set.

### What is *not* here

**No replay buffer**, so these cannot be used to resume training — only to evaluate or watch.
Resuming needs `savedPolicies/<arm>/replay_buffer/` from the original run, and note that cpprb
does not persist priorities across save/restore anyway, so a resume starts from uniform
priorities regardless.

Each entry is three files, ~190 KB total: `ckpt-<step>.index`, `ckpt-<step>.data-00000-of-00001`
and `arch.json`. All three are required — **`arch.json` records the layer widths, action count and
observation era the weights were built for, and a restore now hard-fails without it** (see
`policy_arch.py`), which is what turns the two silent failures below into loud ones. The `checkpoint`
file from `savedPolicies/` is *not* needed — that one only records which checkpoint is "latest", and
restoring by explicit step does not consult it.

## Adding an entry

**Step 0 — confirm the checkpoint with a fresh re-measurement first. This is not optional.** A checkpoint's
close-out /100 read is a *selected* high, inflated ~5-6 pp by selection, and mostly does not survive: batch
24 offered 199 checkpoints at ≥97%/100 and only 9 held ≥97%/500, while `b24a`'s two 100.0%/100 highs held
**zero**. So a candidate earns its place on its **re-measured** number over **≥95 fresh episodes at the
folder's standard — the 500-episode HOF eval** (gate 97, early-abandon), which is exactly what admitted the
batch-24 entries:

```
cd /Users/tony_wang/Projects/Snek/snek2
EVAL_EPISODES=500 EVAL_MIN_ACHIEVABLE=97 EVAL_SCREEN_EPISODES=0 EVAL_INDEPENDENT=1 EVAL_OUT_SUFFIX=_hof500 \
  PYTHONPATH=. python -u eval_checkpoints.py <arm> <step> [<step> ...]
```

The gate-97 early-abandon stops any checkpoint the moment >15 losses put 97% out of reach, so shrinking
candidates bail in seconds and only genuine ≥97%/500 holders run full length. **A checkpoint that does not
clear the gate does not enter the folder.** (On the desktop this is a queued eval job with the same env — see
[`../desktop/README.md`](../desktop/README.md); batch 24's ran there as four parallel jobs.)

**The desktop now runs a `_hof500` re-measurement automatically** after every close-out (`auto_hof`, see
[`../desktop/README.md`](../desktop/README.md#the-eval-chain-training--closeout--hof-re-measure)): it
selects the close-out's **≥98%** checkpoints and re-runs them at 500 episodes with **gate 98**. So a
desktop arm usually already has its `<arm>_checkpoint_evals_hof500.json` waiting on the `results` branch —
step 0 is done for you *for the ≥98% band*. Two caveats before you trust it for admission: the auto run's
**gate is 98, not the folder's 97**, so a genuine 97–98%/500 holder is early-abandoned there and still needs
a hand re-measure at gate 97; and the automation only **measures** — it never copies anything in. Steps 1-2
below are always by hand.

**Step 1 — copy the confirmed checkpoint in:**

```
cd /Users/tony_wang/Projects/Snek/snek2
mkdir -p hallOfFame/<arm>-ckpt<step>
cp savedPolicies/<arm>/ckpt-<step>.index \
   savedPolicies/<arm>/ckpt-<step>.data-00000-of-00001 \
   savedPolicies/<arm>/arch.json \
   hallOfFame/<arm>-ckpt<step>/
```

**Step 2 — verify the *copy* loads and plays, not just the original.** Stage the copy under a throwaway
`savedPolicies/` name and re-measure ~100 episodes; it must read like a champion, not a beginner (the silent
arch/era load traps below both surfaced exactly here):

```
mkdir -p savedPolicies/hofverify && cp hallOfFame/<arm>-ckpt<step>/* savedPolicies/hofverify/
EVAL_EPISODES=100 EVAL_SCREEN_EPISODES=0 EVAL_MIN_ACHIEVABLE=0 EVAL_OUT_SUFFIX=_hofverify \
  PYTHONPATH=. python -u eval_checkpoints.py hofverify <step>
rm -rf savedPolicies/hofverify runs/hofverify_checkpoint_evals_hofverify.json
```

Then add a row to the table above with its **re-measured** rate (the HOF-500 figure), not a close-out /100 or
a graph point. A graph point is 10 episodes and reads in 10-point jumps; 90% graph points have measured
anywhere from 22% to 82%.

**Copy `arch.json` too — it is now required.** Without it the copy will not load at all
(`ArchMismatch: no arch.json`), which is the point: the width and observation era can no longer be
lost. It is one file per policy dir, so the same `arch.json` is correct for every step of that arm.

**‡ Still record a non-default `SNEK_FC_LAYERS` in the row, but `arch.json` now enforces it.** Entries were
all `50,100,50`, the default since batch 1, until batch 24's two `fc 320` records — so width now genuinely
varies between entries and the row must state it. A checkpoint rebuilt at the wrong width used to restore with
**no error**, leaving the mismatched layers unpopulated (`expect_partial()`) so it played like a
beginner — the same silent failure as the observation-vector era problem above. `arch.json` closes
it: the restorer reads the width (and observation length and era) from the sidecar and rebuilds the
*recorded* network, hard-failing on any mismatch, so **`SNEK_FC_LAYERS` is no longer read at eval or
watch time** and there is nothing to remember to set:

```
PYTHONPATH=. python -u watch.py <policy>
```

The row entry stays useful as human-readable documentation, but a forgotten one is no longer able to
mis-measure a checkpoint.

Note that training now **skips writing checkpoints below `SNEK_MIN_CHECKPOINT_SCORE`** (default
40), because `max_to_keep` is a rolling window and a dead arm used to evict good checkpoints
behind it. That reduces the risk of losing a record before it can be copied here, but does not
remove it — copy anything worth keeping as soon as it is measured.
