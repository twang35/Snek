# Completed runs

Every arm that has finished: config, final numbers, verdict. The table below is **canonical** —
it covers every arm ever run, including the batches whose narratives moved to
[`archive/batches1-11.md`](archive/batches1-11.md) (batches 1-11) and
[`archive/batches12-15.md`](archive/batches12-15.md) (batches 12-15). **Narratives here run from
batch 16 to the newest closed batch**; older ones are retired to keep this file readable, and nothing
is ever removed from the table itself.

Companion to [`runs.md`](runs.md) (what is running), [`findings.md`](findings.md) (conclusions) and
[`charts.md`](charts.md) (graphs, batch 12 onward). Nothing here should be re-run without a reason.

## All arms, ranked by best sustained perfect rate

**Use `best ckpt` for cross-arm comparison.** The `measured` column is a pooled figure whose
selection rule changed several times, so the episode count is spelled out per row (`/6300`, `/100`)
and rows with different counts are different statistics — `b8e` has 1 checkpoint behind its figure,
`b8f` 63. Read `measured` as a within-arm consistency check instead: best and pooled close together
means a strong *region*, which is the property being chased.

**Every graph-derived column misranks arms** — `b5c` is 2nd by best perfect-30 and last by
measurement. Read `best perfect-30` as description, not ranking.

**`best perfect-30` is no longer the primary metric.** From 2026-08-04 that is
`strong_eval_fraction`, the share of an arm's evals at >=80%, which has ~40% lower between-seed
variance; see [`hyperparamTuning.md`](hyperparamTuning.md#the-primary-metric-strong_eval_fraction-the-share-of-an-arms-evals-at-80).
Best perfect-30 stays in this table because every arm through batch 11 is recorded on it.

**‡ marks arms measured on the post-audit environment (2026-08-02 onward)**, whose numbers are
not comparable to any row without the mark. The audit changed two observation components and the
reward; the same checkpoint that scored 92% before reads 73% after. Compare ‡ rows only to each
other, or to `b8f`'s `3149000` re-measured at 82%.

**‡‡ marks arms trained on the environment after 2026-08-02's seven further fixes**
(fatal-move zeroing, wall/body hugging, normalized group count, the corrected starve/length
split, the terminal-discount fix, safe-to-chase-food, and the audit that started the day) — a
third, later environment than ‡. Batch 9 (‡) predates all seven; batch 10 (‡‡) is the first to
train on the result. Compare ‡‡ rows only to each other; they are not comparable to ‡ or
unmarked rows.

**‡‡‡ marks arms trained on the 30-value vector (2026-08-03 onward)** — a fourth environment,
adding the following-tail block (26-28) and food-space (29). Batches 11-15 all train on it,
and it is still current. ‡‡‡ rows compare to each other and, with care, to ‡‡ rows: batch 11 is
byte-identical to batch 10 in config, so that one cross-era comparison is *designed* and is written
up in [`archive/batches1-11.md`](archive/batches1-11.md). Everything else stays within its own era.
Batch 10's checkpoints stopped loading on `master` when this landed.

**§§ in the `measured` column marks a run measured under an abandonment gate**, whose figure is
`pooled_equal_effort` rather than the graph-100% tier — a gate censors the tier, and
`pooled_equal_effort` is exact regardless. Gates by batch: none through 13, **90** for 14 and 15,
**95** from 16. ***trunc*** in `best ckpt` means no full-length row survived the gate, so the figure
comes from a shorter row and reads noisier.

**§ marks arms trained after the epsilon rewrite of 2026-08-04.** Not an environment change —
the observation vector is untouched and every ‡‡‡ checkpoint still loads — but a config change,
and a large one: every arm above ran at epsilon exactly 0.0 for 96.8% of its steps, and § arms
explore for a meaningful fraction of the run.

**Every metric here is a greedy eval, so § rows are comparable to ‡‡‡ rows in exactly the way
batch 11 was comparable to batch 10 — as a controlled test of the one thing that changed.** An
earlier draft of this note claimed best-checkpoint survived the change while best-30 did not;
that was wrong, and the distinction does not exist. Epsilon only shapes the data, so it changes
*which policy you get at step N*, and every column then differs — but all of them still measure
the greedy policy's quality, which is what makes the comparison meaningful rather than broken.
What is not legitimate is treating a § arm as another seed of the ‡‡‡ baseline: it is a different
config, and pooling the two would hide the change instead of measuring it.

**Four environments in two days is the real cost being paid here**, and it is worth stating
plainly: every environment change resets the comparison set, so a batch's numbers are only ever
readable against its own siblings. That is why seed count inside a batch matters more than the
number of knobs tried across batches — see the note at the end of [`runs.md`](runs.md).

| policy | config change | final steps | best ckpt | top-3 | **measured** | best perfect-30 | verdict |
|---|---|---|---|---|---|---|---|
| `b29b-chase10g75seed2` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.10` **(gate 75)**, IS off, td_error | 2.00M | **99.0%** @1447k /500 | 100% | 87.14% /eq §§ | 97.3% | ‡‡‡ § **THE RECORD — highest /500 point estimate on record (99.0%, 495/500, CI 97.7-99.6)**, and the head of an **18-checkpoint ≥98%/500 band** (1446k-1529k) — a *region*, where b24's records are isolated points. The 99.0 vs 98.0 lead over b24 is inside the 500-ep CI; the region is not. **Promoted to [`../hallOfFame/`](../hallOfFame/README.md) 2026-08-16**, copy verified 98/100. See [Batches 28-29](#batches-28-29--chase-safe-dose-and-gate-the-gate-is-the-lever-and-gate-75-produces-a-record-region) |
| `b29a-chase10g75seed1` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.10` **(gate 75)**, IS off, td_error | 2.00M | **98.4%** @1347k /500 | 100% | 89.76% /eq §§ | 97.7% | ‡‡‡ § **3 checkpoints held ≥98%/500** (1339k/1347k/1414k), best 98.4% — record-tier, the second seed of the b29 gate-75 region. Not yet HOF-promoted |
| `b24d-fc320noisseed4` ‡‡‡ § | **fc 320**, IS off (`SNEK_IS_WEIGHTS=0`), td_error | 3.00M | **98.0%** @1342k /500 | 99.0% | 85.97% /eq §§ | 96.7% | ‡‡‡ § **THE NEW RECORD** — 98.0% over 500 fresh episodes (490/500, CI [96.4,98.9]), and it *rose* on re-measurement (97.0/100 → 98.0/500), the genuine-region signature. 3 of its 51 ≥97%/100 checkpoints held ≥97%/500 (early, 0.9-1.36M). In [`../hallOfFame/`](../hallOfFame/README.md) |
| `b24b-fc320noisseed2` ‡‡‡ § | **fc 320**, IS off, td_error | 3.00M | **98.0%** @2860k /500 | 99.0% | 88.84% /eq §§ | **96.7%** | ‡‡‡ § **new record (tied)** — 98.0%/500 (490/500, CI [96.4,98.9]); 3 of 59 ≥97%/100 checkpoints held ≥97%/500, all late (2.74-2.86M). In [`../hallOfFame/`](../hallOfFame/README.md) |
| `b24c-fc320noisseed3` ‡‡‡ § | **fc 320**, IS off, td_error | 3.00M | 97.4% @2982k /500 | 99.3% | 87.68% /eq §§ | 96.0% | ‡‡‡ § 3 of 46 ≥97%/100 checkpoints held ≥97%/500 (best 97.4%, CI [95.6,98.5], late 2.95-2.98M); **both its 100%/100 highs shrank below 97%/500**. Not added to HOF (below the record) |
| `b24a-fc320noisseed1` ‡‡‡ § | **fc 320** (one wide layer), IS off, td_error | 3.00M | — /500 (0 held) | 99.7% | **89.03%** /eq §§ | 95.3% | ‡‡‡ § **highest pooled ever at gate 95 (89.03)**, yet **0 of its 43 ≥97%/100 checkpoints held ≥97%/500** — its 100%/100 @1633k produced no survivor. The batch's cleanest selection-inflation lesson: strong consolidation, no record-tier peak |
| `b25c-fc200x100x100noisseed3-r2` ‡‡‡ § | **fc 200,100,100** (3 layers, ~1.6× params), IS off, td_error | 3.00M | 99.0% @818k /100 | 98.3% | **87.22%** /eq §§ | 93.7% | ‡‡‡ § best of b25 on pooled, and the batch's highest `sef` (66.9%). HOF-500 held nothing: best partial 95.3% @827k, abandoned at 258 episodes under gate 98 |
| `b25d-fc200x100x100noisseed4-r2` ‡‡‡ § | **fc 200,100,100**, IS off, td_error | 3.00M | 99.0% @2431k /100 | 98.3% | 85.92% /eq §§ | 94.3% | ‡‡‡ § HOF-500 best partial 96.4% @2431k (304 episodes, abandoned). Its peak arrives late — peak trailing 95.00 at 2426k |
| `b25a-fc200x100x100noisseed1-r2` ‡‡‡ § | **fc 200,100,100**, IS off, td_error | 3.00M | 99.0% @802k /100 | 98.7% | 85.59% /eq §§ | 93.7% | ‡‡‡ § its 99%/100 @802k re-measured to **92.7%** before gate 98 abandoned it — the batch's clearest selection-inflation row |
| `b25b-fc200x100x100noisseed2-r2` ‡‡‡ § | **fc 200,100,100**, IS off, td_error | 3.00M | 99.0% @890k /100 | 98.7% | 85.54% /eq §§ | **95.3%** | ‡‡‡ § **the batch's one plausible hall candidate**: @911k was still **97.2% at 392 episodes** when gate 98 stopped it. Gate 97 would have run it to 500; not promoted, since the auto chain's gate is 98 |
| `b30e-chase10fc200x100x100seed1` ‡‡‡ § | **fc 200,100,100**, chase-safe shaping `c=0.10` (gate 85), IS off, td_error | 2.00M | 99.0% @641k /100 | 99.0% | 83.75% /eq §§ | 93.7% | ‡‡‡ § shaped counterpart of `b25a`. 6 checkpoints ≥98%/100 re-measured at 500/gate 98: **0 held**, best 96.1% @651k (285 ep, ab.). No record |
| `b30g-chase10fc200x100x100seed3` ‡‡‡ § | **fc 200,100,100**, chase-safe shaping `c=0.10` (gate 85), IS off, td_error | 2.00M | 99.0% @738k /100 | 98.3% | 84.28% /eq §§ | 93.3% | ‡‡‡ § shaped counterpart of `b25c`. 3 checkpoints ≥98%/100 → **0 held**, best 95.3% @709k (233 ep, ab.). No record |
| `b30f-chase10fc200x100x100seed2` ‡‡‡ § | **fc 200,100,100**, chase-safe shaping `c=0.10` (gate 85), IS off, td_error | 2.00M | 98.0% @643k /100 | 97.0% | 84.32% /eq §§ | 92.3% | ‡‡‡ § shaped counterpart of `b25b`. 1 checkpoint ≥98%/100 → **0 held**, 90.9% @643k (132 ep, ab.). No record |
| `b30h-chase10fc200x100x100seed4` ‡‡‡ § | **fc 200,100,100**, chase-safe shaping `c=0.10` (gate 85), IS off, td_error | 2.00M | 97.0% @614k /100 | 96.3% | 81.00% /eq §§ | 92.3% | ‡‡‡ § shaped counterpart of `b25d`. **No ≥98%/100 checkpoint**, so the HOF-500 selected nothing. Lowest pooled of the four |
| `b29c-chase10g75seed3` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.10` (gate 75), IS off, td_error | 2.00M | 97.1% @1396k /378 *trunc* | 99.0% | 89.68% /eq §§ | 96.3% | ‡‡‡ § third b29 seed; **0 held** — best partial 97.1% abandoned at 378 ep under gate 98. Highest `sef` of b29 (67.9%). The two record seeds are `b29b`/`b29a` above |
| `b29d-chase10g75seed4` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.10` (gate 75), IS off, td_error | 2.00M | 90.6% @1443k /127 *trunc* | 98.0% | 84.75% /eq §§ | 94.3% | ‡‡‡ § the weak b29 seed; **0 held**, best partial 90.6% (127 ep, ab.). The gate-75 region rests on 2 of 4 seeds |
| `b28d-chase20g85seed4` ‡‡‡ § | **fc 320**, chase-safe shaping **`c=0.20`** (gate 85), IS off, td_error | 2.00M | 96.8% @1061k /341 *trunc* | 99.0% | 89.10% /eq §§ | 96.7% | ‡‡‡ § best of b28 (doubled dose, gate 85); **0 held** — best partial 96.8% ab. at 341 ep. Doubling `c` does not rescue gate 85 |
| `b28a-chase20g85seed1` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.20` (gate 85), IS off, td_error | 2.00M | 95.6% @1727k /275 *trunc* | 99.0% | 89.72% /eq §§ | 96.0% | ‡‡‡ § highest pooled of b28 (89.72); **0 held**, best partial 95.6% (275 ep, ab.) |
| `b28c-chase20g85seed3` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.20` (gate 85), IS off, td_error | 2.00M | — *trunc* (0 full-length) | 97.0% | 82.67% /eq §§ | 92.7% | ‡‡‡ § no checkpoint reached the gate; **0 held**. `sef` 33.4% |
| `b28b-chase20g85seed2` ‡‡‡ § | **fc 320**, chase-safe shaping `c=0.20` (gate 85), IS off, td_error | 2.00M | 90.0% @1127k /120 *trunc* | 96.0% | 80.15% /eq §§ | 90.7% | ‡‡‡ § weakest of b28; **0 held**, best partial 90.0% (120 ep, ab.) |
| `b26b-fc100x100noisseed2` ‡‡‡ § | **fc 100,100** (2 layers, 0.62× the b24 net), IS off, td_error | 3.00M | 97.0% @1948k /100 | 96.7% | **83.83%** /eq §§ | 93.7% | ‡‡‡ § best of b26 on every column; `sef` 58.0%. **No ≥98%/100 checkpoint**, so the auto HOF-500 selected nothing |
| `b26c-fc100x100noisseed3` ‡‡‡ § | **fc 100,100**, IS off, td_error | 3.00M | 97.0% @1969k /100 | 96.0% | 83.18% /eq §§ | 92.0% | ‡‡‡ § 2nd of b26; only 7 full-length rows cleared gate 95, against b25c's 64 — the shallow shape spends much less time in the measurable band |
| `b26a-fc100x100noisseed1` ‡‡‡ § | **fc 100,100**, IS off, td_error | 3.00M | 95.0% @2904k /100 | 95.0% | 80.02% /eq §§ | 88.0% | ‡‡‡ § just 2 full-length rows, both at the gate exactly. Peak trailing 94.92 — the first b24-family arm not to reach 95.00 |
| `b26d-fc100x100noisseed4` ‡‡‡ § | **fc 100,100**, IS off, td_error | 3.00M | — *trunc* (0 full-length) | — | 69.64% /eq §§ | 79.7% | ‡‡‡ § the weak seed of b26 (`sef` **13.8%**) — never died, never reached the measurable band. Below the b22 control's mean, which is what makes b26's small lift a shape result rather than a seed accident |
| `b18b-tgt1000seed2` ‡‡‡ § | **target period 1000**, forking on | 2.40M | **97.6%** @1588k /700 | **97.0%** | 78.52% /eq §§ | 86.0% | ‡‡‡ § **THE PROJECT RECORD** — 97.6% over 700 fresh episodes (CI 96.1-98.5), beating `b17b` by +3.33 pp, p=0.0002. **The first selected high here that did not shrink** (98/100 -> 97.4%/500). A narrow peak: @1578k reads 91.6%. In [`../hallOfFame/`](../hallOfFame/README.md) |
| `b18a-tgt1000seed1` ‡‡‡ § | **target period 1000**, forking on | 2.61M | 96% @1289k /100 | 95.3% | **81.87%** /eq §§ | 88.0% | ‡‡‡ § 2nd-highest eq-effort on record; `sef` **41.4%** at 2.61M |
| `b18d-tgt1000seed4` ‡‡‡ § | **target period 1000**, forking on | 2.60M | 96% @1105k /100 | 96.0% | 80.22% /eq §§ | **91.0%** | ‡‡‡ § **highest `sef` ever recorded, 47.9%** (inflated by run length); best-30 91.0%, and the batch's smallest drawdown |
| `b18c-tgt1000seed3` ‡‡‡ § | **target period 1000**, forking on | 2.51M | 92% @2168k *trunc* | — | 74.75% /eq §§ | 84.3% | ‡‡‡ § weakest of batch 18; **no full-length row survived the gate**; best window in its last 350k |
| `b23b-beta01seed2` ‡‡‡ § | **IS β anneal 0→0.1** (`SNEK_IS_BETA_FINAL=0.1`), td_error, fc 50,100,50 | 3.00M | **97%** @777k /100 | 96.3% | **82.07%** /eq §§ | 91.0% | ‡‡‡ § **best of the β ladder** — five full-length ckpts ≥95/100 clustered at ~777k, pooled 82.1 (highest eq-effort of any β-ladder arm). **Not a HOF candidate after all:** re-measured at 500 fresh episodes the selected @777k reads **92.4%**, the *worst* of its own cluster — textbook selection bias, well below the 97.6% record |
| `b23a-beta01seed1` ‡‡‡ § | **IS β anneal 0→0.1**, td_error, fc 50,100,50 | 3.00M | 80% @1039k /20 *trunc* | — | 77.17% /eq §§ | 87.0% | ‡‡‡ § 2nd on pooled; no full-length row cleared gate 95, so best row is a 20-ep screen. `sef` 23.5% |
| `b23d-beta01seed4` ‡‡‡ § | **IS β anneal 0→0.1**, td_error, fc 50,100,50 | 3.00M | 75% @603k /20 *trunc* | — | 72.11% /eq §§ | 82.3% | ‡‡‡ § just edges its control `b20d` (71.3), the control's strongest arm; `sef` 28.7% |
| `b23c-beta01seed3` ‡‡‡ § | **IS β anneal 0→0.1**, td_error, fc 50,100,50 | 3.00M | 75% @1393k /20 *trunc* | — | 71.45% /eq §§ | 83.0% | ‡‡‡ § weakest of b23 on pooled but still above b21's mean (64.3); `sef` 24.0% |
| `b22a-noisseed1` ‡‡‡ § | **IS off** (`SNEK_IS_WEIGHTS=0`), td_error, fc 50,100,50 | 3.00M | **96%** @1075k /100 | 95.0% | **78.89%** /eq §§ | 87.0% | ‡‡‡ § strongest of b22 and the batch's one clean full-length row; `sef` **46.3%**. Below the 97.6% record, not a HOF candidate |
| `b22d-noisseed4` ‡‡‡ § | **IS off**, td_error, fc 50,100,50 | 3.00M | 95% @2275k /100 | — | 76.27% /eq §§ | **88.3%** | ‡‡‡ § 2nd on pooled; a lone full-length 95/100 late at 2275k. Below the record |
| `b22c-noisseed3` ‡‡‡ § | **IS off**, td_error, fc 50,100,50 | 3.00M | 80% @866k /20 *trunc* | — | 75.04% /eq §§ | 86.7% | ‡‡‡ § no full-length row cleared gate 95, so best row is a 20-ep screen; `sef` 27.5% |
| `b22b-noisseed2` ‡‡‡ § | **IS off**, td_error, fc 50,100,50 | 3.00M | — *trunc* | — | 72.51% /eq §§ | 83.0% | ‡‡‡ § weakest of b22; no full-length row; `sef` 22.5% |
| `b20i-fc200x50seed1` ‡‡‡ § | **`FC_LAYERS=200,50`** (wide-early, 1.38× params), β anneal 300k | 3.00M | 87% @2818k *trunc* | — | 64.47% /eq §§ | 72.7% | ‡‡‡ § best of wave 2 on best-30; **the only seed where the shape beats its control**, and only because control `b20a` is the batch's weakest arm. `sef` 8.7% |
| `b20l-fc200x50seed4` ‡‡‡ § | **`FC_LAYERS=200,50`** (wide-early, 1.38× params), β anneal 300k | 3.00M | 86% @2940k *trunc* | — | **64.75%** /eq §§ | 72.3% | ‡‡‡ § highest `sef` (13.9%) and smallest drawdown (7.18) of wave 2; **best checkpoint at 2940k, its last 60k** — still climbing when capped |
| `b20j-fc200x50seed2` ‡‡‡ § | **`FC_LAYERS=200,50`** (wide-early, 1.38× params), β anneal 300k | 3.00M | 83% @2001k *trunc* | — | 59.31% /eq §§ | 69.3% | ‡‡‡ § mid of wave 2; `sef` 9.9% against its control's 16.2%; largest drawdown of the wave, 9.76 |
| `b20k-fc200x50seed3` ‡‡‡ § | **`FC_LAYERS=200,50`** (wide-early, 1.38× params), β anneal 300k | 3.00M | 72% @1073k *trunc* | — | 48.50% /eq §§ | 55.3% | ‡‡‡ § weakest of wave 2, `sef` 1.9%; only 20 checkpoints cleared selection and the deepest row is 25 episodes. Mirrors control `b20c` (2.2%) — seed 3 is weak under both shapes |
| `b20p-fc320seed4` ‡‡‡ § | **`FC_LAYERS=320`** (depth-1, 0.92× params), β anneal 300k | 3.00M | 89% @2321k *trunc* | — | **73.03%** /eq §§ | 80.0% | ‡‡‡ § best of the `320` seeds; `sef` 26.2%, peak 94.76. Edges its control `b20d` (71.3%) despite one fewer layer |
| `b20n-fc320seed2` ‡‡‡ § | **`FC_LAYERS=320`** (depth-1, 0.92× params), β anneal 300k | 3.00M | 87% @2896k *trunc* | — | 70.32% /eq §§ | **81.3%** | ‡‡‡ § **highest peak (94.86) and best-30 of the shape**; `sef` 23.1%. Beats its control `b20b` (62.7%) |
| `b20m-fc320seed1` ‡‡‡ § | **`FC_LAYERS=320`** (depth-1, 0.92× params), β anneal 300k | 3.00M | 82% @2935k *trunc* | — | 65.31% /eq §§ | 73.0% | ‡‡‡ § **the seed that lifts the shape's mean** — its control `b20a` is the batch's weak arm (33.2%), so this +32 pp gap is a control weakness, not a shape strength. `sef` 12.9% |
| `b20o-fc320seed3` ‡‡‡ § | **`FC_LAYERS=320`** (depth-1, 0.92× params), β anneal 300k | 3.00M | 69% @2807k *trunc* | — | 51.83% /eq §§ | 64.3% | ‡‡‡ § weakest of the `320` seeds, `sef` 3.9%, largest drawdown of the wave (11.96). Mirrors control `b20c` — seed 3 is weak under every shape |
| `b20w-fc60x30x30x30x30seed3` ‡‡‡ § | **`FC_LAYERS=60,30,30,30,30`** (deep-narrow, depth 5, 0.55× params), β 300k | 3.00M | 79% @1945k *trunc* | — | **62.67%** /eq §§ | **73.3%** | ‡‡‡ § best of the deep-narrow seeds; **the seed that matches its control `b20c`** (peak 94.66, `sef` 11.9%). The shape holds the ceiling here but drawdown is 10.9 vs the control's 5.1 |
| `b20x-fc60x30x30x30x30seed4` ‡‡‡ § | **`FC_LAYERS=60,30,30,30,30`** (deep-narrow, depth 5, 0.55× params), β 300k | 3.00M | 75% @1139k *trunc* | — | 59.19% /eq §§ | 63.0% | ‡‡‡ § 2nd of the shape; `sef` 4.6% against its control `b20d`'s 26.3%; drawdown 12.0 |
| `b20v-fc60x30x30x30x30seed2` ‡‡‡ § | **`FC_LAYERS=60,30,30,30,30`** (deep-narrow, depth 5, 0.55× params), β 300k | 3.00M | 77% @2436k *trunc* | — | 53.75% /eq §§ | 60.3% | ‡‡‡ § best row in its last ~560k — **still climbing when capped**; largest drawdown of the shape (13.58); `sef` 2.2% |
| `b20u-fc60x30x30x30x30seed1` ‡‡‡ § | **`FC_LAYERS=60,30,30,30,30`** (deep-narrow, depth 5, 0.55× params), β 300k | 3.00M | 56% @2534k *trunc* | — | 30.75% /eq §§ | 40.7% | ‡‡‡ § weakest of the shape; `sef` 0.1%, peak 93.60. Mirrors control `b20a` — seed 1 is weak under every shape |
| `b20r-fc25x50x25seed2` ‡‡‡ § | **`FC_LAYERS=25,50,25`** (small, depth 3, **0.29×** params), β 300k | 3.00M | 76% @1874k *trunc* | — | **57.44%** /eq §§ | **66.3%** | ‡‡‡ § best of the small-net seeds and **the only one to reach the control band** (peak 94.48); `sef` 6.3%. Even the best seed trails its control `b20b` (pooled 62.7%) |
| `b20t-fc25x50x25seed4` ‡‡‡ § | **`FC_LAYERS=25,50,25`** (small, depth 3, **0.29×** params), β 300k | 3.00M | 60% @1530k *trunc* | — | 45.25% /eq §§ | 57.0% | ‡‡‡ § 2nd of the shape; smallest drawdown of the small net (6.22); `sef` 2.0% against control `b20d`'s 26.3% |
| `b20s-fc25x50x25seed3` ‡‡‡ § | **`FC_LAYERS=25,50,25`** (small, depth 3, **0.29×** params), β 300k | 3.00M | 56% @739k *trunc* | — | 39.50% /eq §§ | 46.7% | ‡‡‡ § `sef` 0.2%; **peak 93.36, below the nine-batch band** — the capacity cut showing on the ceiling |
| `b20q-fc25x50x25seed1` ‡‡‡ § | **`FC_LAYERS=25,50,25`** (small, depth 3, **0.29×** params), β 300k | 3.00M | 44% @897k *trunc* | — | 30.25% /eq §§ | 41.3% | ‡‡‡ § **weakest arm in batch 20**: peak 93.08 (lowest), `sef` 0.0%, drawdown 16.3 (worst). 0.29× is not enough capacity here |
| `b20aa-fc93x93seed1` ‡‡‡ § | **`FC_LAYERS=93,93`** (iso-param depth-2, 1.00× params), β 300k | 3.00M | 82% @2908k *trunc* | — | 65.57% /eq §§ | 74.3% | ‡‡‡ § best of the `93,93` seeds, but **the seed flatters the shape** — its control `b20a` is the batch's weak arm (33.2 pooled), so the +33 best-30 gap is a control weakness. peak 94.78, `sef` 15.6% |
| `b20ad-fc93x93seed4` ‡‡‡ § | **`FC_LAYERS=93,93`** (iso-param depth-2, 1.00× params), β 300k | 3.00M | 79% @2912k *trunc* | — | 56.15% /eq §§ | 66.7% | ‡‡‡ § 2nd of the shape; peak 94.44, inside the control band; `sef` 10.0% |
| `b20ab-fc93x93seed2` ‡‡‡ § | **`FC_LAYERS=93,93`** (iso-param depth-2, 1.00× params), β 300k | 3.00M | 56% @2661k *trunc* | — | 42.00% /eq §§ | 54.0% | ‡‡‡ § `sef` 0.7%; behind its control `b20b` (62.7 pooled) |
| `b20ac-fc93x93seed3` ‡‡‡ § | **`FC_LAYERS=93,93`** (iso-param depth-2, 1.00× params), β 300k | 3.00M | 56% @2663k *trunc* | — | 42.25% /eq §§ | 50.7% | ‡‡‡ § weakest of the shape; `sef` 0.6%, peak 94.08. Mirrors control `b20c` — seed 3 is weak under both |
| `b20ah-fc100x200x100seed4` ‡‡‡ § | **`FC_LAYERS=100,200,100`** (3.69× params, depth 3), β 300k | 3.00M | 92% @1470k *trunc* | — | **74.57%** /eq §§ | **83.0%** | ‡‡‡ § best of the shape; peak 94.88, `sef` 32.9%; best window at 1.47M |
| `b20ae-fc100x200x100seed1` ‡‡‡ § | **`FC_LAYERS=100,200,100`** (3.69× params, depth 3), β 300k | 3.00M | 89% @1561k *trunc* | — | 67.76% /eq §§ | 77.7% | ‡‡‡ § **the seed that lifts the shape's mean** — control `b20a` is the batch's weak arm (33.2 pooled), so the +36 best-30 gap is a control weakness. `sef` 21.8% |
| `b20af-fc100x200x100seed2` ‡‡‡ § | **`FC_LAYERS=100,200,100`** (3.69× params, depth 3), β 300k | 3.00M | 83% @1918k *trunc* | — | 63.05% /eq §§ | 69.3% | ‡‡‡ § peak 94.60, `sef` 10.2%; ~flat against its control `b20b` |
| `b20ag-fc100x200x100seed3` ‡‡‡ § | **`FC_LAYERS=100,200,100`** (3.69× params, depth 3), β 300k | 3.00M | 64% @551k *trunc* | — | 48.25% /eq §§ | 55.3% | ‡‡‡ § weakest of the shape; `sef` 1.4%, peak 94.18. Mirrors control `b20c` — seed 3 weak under every shape |
| `b20ai-fc100x50x50seed1` ‡‡‡ § | **`FC_LAYERS=100,50,50`** (matched-capacity reshuffle, 0.92× params, depth 3), β 300k | 3.00M | 72% @2647k *trunc* | — | **52.08%** /eq §§ | **58.7%** | ‡‡‡ § best of the shape; peak 94.34, `sef` 4.5%, smallest-but-one drawdown (5.58). **Beats control `b20a` by +18.8 pooled — the batch's weak seed again**, not a shape effect |
| `b20al-fc100x50x50seed4` ‡‡‡ § | **`FC_LAYERS=100,50,50`** (matched-capacity reshuffle, 0.92× params, depth 3), β 300k | 3.00M | 64% @2836k *trunc* | — | 51.75% /eq §§ | 57.0% | ‡‡‡ § peak 94.36; **best row at 2836k, its last 165k** — still climbing when capped. `sef` 1.9% against control `b20d`'s 26.3% |
| `b20ak-fc100x50x50seed3` ‡‡‡ § | **`FC_LAYERS=100,50,50`** (matched-capacity reshuffle, 0.92× params, depth 3), β 300k | 3.00M | 68% @2692k *trunc* | — | 50.75% /eq §§ | 53.3% | ‡‡‡ § **dead level with control `b20c`** (50.8 vs 52.8 pooled, −2.0). peak 94.36, `sef` 1.0%, drawdown 5.32 |
| `b20aj-fc100x50x50seed2` ‡‡‡ § | **`FC_LAYERS=100,50,50`** (matched-capacity reshuffle, 0.92× params, depth 3), β 300k | 3.00M | 52% @1026k *trunc* | — | 30.75% /eq §§ | 36.0% | ‡‡‡ § **weakest arm of the shape and the whole reason its mean reads low**: `sef` 0.1%, peak 93.56 (below the nine-batch band), pooled 30.8 against control `b20b`'s 62.7. Seed 2 is the control's *second-best* arm, so this is a −31.9 pp seed swing at matched capacity |
| `b21b-beta05seed2` ‡‡‡ § | **partial IS** (`td_error` + IS, β 0.4→**0.5**), fc 50,100,50 | 3.00M | 89% @2525k *trunc* | — | **68.79%** /eq §§ | **80.7%** | ‡‡‡ § best of b21; peak 94.78, `sef` 21.6%. β→0.5 sits above the β→1.0 control on every column — see the batch 21 write-up |
| `b21d-beta05seed4` ‡‡‡ § | **partial IS** (`td_error` + IS, β 0.4→**0.5**), fc 50,100,50 | 3.00M | 84% @2392k *trunc* | — | 66.01% /eq §§ | 76.3% | ‡‡‡ § 2nd of b21; **peak 94.80, batch's highest**; `sef` 16.5% |
| `b21c-beta05seed3` ‡‡‡ § | **partial IS** (`td_error` + IS, β 0.4→**0.5**), fc 50,100,50 | 3.00M | 83% @2203k *trunc* | — | 62.80% /eq §§ | 70.7% | ‡‡‡ § peak 94.64, `sef` 11.1% |
| `b21a-beta05seed1` ‡‡‡ § | **partial IS** (`td_error` + IS, β 0.4→**0.5**), fc 50,100,50 | 3.00M | 82% @1263k *trunc* | — | 59.50% /eq §§ | 68.7% | ‡‡‡ § weakest of b21; peak 94.52, `sef` 8.2% |
| `b19d-stdperseed4` ‡‡‡ § | **standard PER** (`td_error` + IS on, β→1.0), period 1000 | 2.42M | 91% @1536k *trunc* | — | 75.46% /eq §§ | **85.7%** | ‡‡‡ § **the seed that escaped batch 19** — level with its `b18d` control on every column (`sef` 40.2 vs 41.6). No close-out run |
| `b19a-stdperseed1` ‡‡‡ § | **standard PER** (`td_error` + IS on, β→1.0), period 1000 | 2.19M | 77% @1485k *trunc* | — | 61.49% /eq §§ | 71.0% | ‡‡‡ § `sef` 8.0% against its control's 41.2%; **smallest drawdown in batches 18-19, 4.94**. No close-out run |
| `b19b-stdperseed2` ‡‡‡ § | **standard PER** (`td_error` + IS on, β→1.0), period 1000 | 2.12M | 76% @937k *trunc* | — | 51.59% /eq §§ | 66.7% | ‡‡‡ § slowest consolidator: pf30 ≥ 60% at 1861k against its control's 310k. **Still improving when stopped**. No close-out run |
| `b19c-stdperseed3` ‡‡‡ § | **standard PER** (`td_error` + IS on, β→1.0), period 1000 | 2.00M | 36% @169k *trunc* | — | 23.75% /eq §§ | 29.7% | ‡‡‡ § **`sef` 0.0% — not one eval ≥80% in 2005 evals**; never reached pf30 ≥ 40%; best window at 195k. Plateaued, never dead. No close-out run |
| `b17b-forkseed2` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.57M | **95.2%** @1190k /600 | 93.5% /500 | **82.42%** /eq §§ | **92.7%** | ‡‡‡ § **the record**, re-measured: selected reads of 99/100 fell to 92.4-95.0 over 500. Reached ~95% at **1.19M** |
| `b15b-nstep3seed2` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.75M | 97% @3245k | 96.3% | 79.7% /eq §§ | 89.3% | ‡‡‡ § previous best selected ckpt, but **93.0% over 300** |
| `b11b-obs30seed2` ‡‡‡ | disc 0.995, **fourth env** | 3.56M | **96%** @855k | **95.3%** | **81.0%** /10400 † | **91.7%** | ‡‡‡ 96/100 selected, ~94% shrunk |
| `b14a-disc9975seed1` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.17M | 96% @3702k | 93.0% | 72.4% /eq §§ | 79.7% | ‡‡‡ § 96/100 selected; 91/100 on re-measure, **93.5% over 200** |
| `b15a-nstep3seed1` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.79M | 95% @4697k | 94.3% | 77.7% /eq §§ | **89.7%** | ‡‡‡ § **peak trailing 95.00, highest on record**; still gaining at 5.8M |
| `b13d-shieldseed4` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.51M | **95%** @986k | 93.3% | 77.2% /4800 † | 83.3% | ‡‡‡ § **2nd best ckpt on record**; peaked ~1M then lost 44 pp |
| `b14d-disc9975seed4` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.46M | 93% @2559k | 93.0% | **77.6%** /eq §§ | **89.7%** | ‡‡‡ § **best arm on record for `strong_eval_fraction`, 39.3%**; only 11.7 pp drawdown |
| `b17c-forkseed3` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.52M | 93% @1424k *trunc* | 92.6% | 73.33% /eq §§ | 82.0% | ‡‡‡ § **still climbing when stopped** — peak and best window in its last 140k |
| `b16b-noshapeseed2` ‡‡‡ § | **shaping off** (`FOOD_DISTANCE_REWARD=0`), disc 0.9975 | 1.26M | 93% @913k *trunc* | — | **79.08%** /eq §§ | 85.0% | ‡‡‡ § best of batch 16, **the first non-null in six batches**; `sef` **30.5%**; peak trailing **94.98**, joint-highest on record |
| `b16a-noshapeseed1` ‡‡‡ § | **shaping off** (`FOOD_DISTANCE_REWARD=0`), disc 0.9975 | 1.25M | 92% @1203k /100 | — | 76.89% /eq §§ | **87.0%** | ‡‡‡ § 2nd of batch 16; `sef` 20.6%. **The only batch-16 arm with a genuine 100-episode row** — `eval_progress.best_of` prefers a 93.3% /90ep row over it, which is the half-depth relaxation showing its edge |
| `b13c-shieldseed3` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.67M | 92% @3367k | 90.7% | 75.9% /11400 † | **85.3%** | ‡‡‡ § best of batch 13 on the graph; best ckpt in its final 300k |
| `b13b-shieldseed2` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.70M | 91% @1166k | 90.3% | 74.8% /7800 † | 82.3% | ‡‡‡ § fastest start on record: 72.3% pf30 by 350k |
| `b14b-disc9975seed2` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.12M | 90% @2261k | — | 67.1% /eq §§ | 76.3% | ‡‡‡ § weakest of batch 14; **one full-length row survived the gate**; lost 47 pp |
| `b14c-disc9975seed3` ‡‡‡ § | disc **0.9975**, shield 0.8 | 4.16M | 90% @2099k | 89.3% | 71.3% /eq §§ | 87.7% | ‡‡‡ § **still improving at 4.5M** — the arm that moved the step cap to 10M |
| `b15d-nstep3seed4` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.81M | 91% @3671k | 90.3% | 73.5% /eq §§ | 86.3% | ‡‡‡ § **peak trailing at 5799k, its 2nd-to-last eval** — stopped mid-climb |
| `b13a-shieldseed1` ‡‡‡ § | + eps handover 0.0125, shield 0.5 | 3.39M | 80% @2044k | 80.0% | 70.5% /3100 † | 78.0% | ‡‡‡ § weakest of batch 13, and the batch's slowest starter |
| `b17d-forkseed4` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.51M | 86% @1260k *trunc* | 84.5% | 66.83% /eq §§ | 75.7% | ‡‡‡ § fastest starter of batch 17; best window at 679k then flat |
| `b16c-noshapeseed3` ‡‡‡ § | **shaping off** (`FOOD_DISTANCE_REWARD=0`), disc 0.9975 | 1.26M | 85% @979k *trunc* | — | 69.67% /eq §§ | 72.7% | ‡‡‡ § 3rd of batch 16; `sef` 10.5%; **peak trailing came at 1198k, its last 60k** — stopped mid-climb; deepest row is 40 episodes |
| `b15c-nstep3seed3` ‡‡‡ § | **n=3**, disc 0.995, shield 0.8 | 5.46M | 86% @3823k *trunc* | — | 66.4% /eq §§ | 75.7% | ‡‡‡ § weakest of batch 15; **no full-length row survived the gate** — best is a truncated 69/80 |
| `b16d-noshapeseed4` ‡‡‡ § | **shaping off** (`FOOD_DISTANCE_REWARD=0`), disc 0.9975 | 1.26M | 80% @1010k *trunc* | — | 63.27% /eq §§ | 73.0% | ‡‡‡ § weakest of batch 16 and **the arm that keeps the result at p=0.250** — `sef` 7.0%, level with its batch-14 control's 7.9%; deepest row is 30 episodes |
| `b11a-obs30seed1` ‡‡‡ | disc 0.995, **fourth env** | 3.19M | 94% @671k | 93.3% | 79.5% /4800 † | 85.7% | ‡‡‡ 2nd of batch 11; peaked at 678k, then lost 42 pp |
| `b11d-obs30seed4` ‡‡‡ | disc 0.995, **fourth env** | 3.59M | 88% @3507k | 86.7% | 69.3% /4000 † | 78.3% | ‡‡‡ only arm still near peak when stopped |
| `b11c-obs30seed3` ‡‡‡ | disc 0.995, **fourth env** | 3.23M | 87% @1706k | 84.7% | 69.0% /2300 † | 73.0% | ‡‡‡ weakest of batch 11 |
| `b17a-forkseed1` ‡‡‡ § | **forking on**, disc 0.9975, shaping off | 1.41M | 68% @916k *trunc* | 66.7% | 46.25% /eq §§ | 54.0% | ‡‡‡ § **the arm that made batch 17 a null**; never reached ε ≤ 0.003; 22 episodes is its deepest row |
| `b12d-eps002seed4` ‡‡‡ § | eps handover **0.05**, no shield | 1.09M | not measured | — | — | 6.3% | ‡‡‡ § **deadlocked**; abandoned at 1M of 2.5M |
| `b12a-eps002seed1` ‡‡‡ § | eps handover **0.05**, no shield | 1.12M | not measured | — | — | 6.3% | ‡‡‡ § deadlocked; peaked 87.0 trailing @214k then decayed to 55 |
| `b12c-eps002seed3` ‡‡‡ § | eps handover **0.05**, no shield | 0.98M | not measured | — | — | 1.7% | ‡‡‡ § deadlocked; 8 perfect games in 977 evals |
| `b12b-eps002seed2` ‡‡‡ § | eps handover **0.05**, no shield | 1.03M | not measured | — | — | **0.0%** | ‡‡‡ § deadlocked; **zero perfect games in 1032 evals** |
| `b12s-shield05seed1` ‡‡‡ § | eps handover **0.05** + shield 0.5 | 0.43M | not measured | — | — | 0.3% | ‡‡‡ § the probe that moved the handover; decay fixed, plateau not |
| `b10d-disc995seed4` ‡‡ | disc **0.995**, third env | 4.45M | **93%** @1695k | **93.3%** | **74.9%** /66000 | 84.3% | ‡‡ best of batch 10; held the record until `b11b` |
| `b10b-disc995seed2` ‡‡ | disc **0.995**, third env | 4.65M | 90% @1501k | 86.3% | 71.8% /62400 | 85.0% | ‡‡ 2nd of batch 10 |
| `b10a-disc995seed1` ‡‡ | disc **0.995**, third env | 4.29M | 85% @2344k | — | 67.2% /27200 | 78.3% | ‡‡ stopped healthy |
| `b10c-disc995seed3` ‡‡ | disc **0.995**, third env | 4.12M | 79% @3965k | — | 63.0% /4700 | 72.7% | ‡‡ weakest of batch 10 |
| `b9d-disc995b` ‡ | disc **0.995**, new env | 3.45M | **70%** @2544k | **66.3%** | 42.4% /1700 | 30.3% | ‡ best ceiling of batch 9 |
| `b9a-disc9975a` ‡ | disc **0.9975**, new env | 3.68M | 65% @1735k | 64.3% | **54.9%** /2000 | 56.0% | ‡ most consistent of batch 9 |
| `b9c-disc995a` ‡ | disc **0.995**, new env | 3.71M | 52% @2603k | 51.3% | 38.0% /2000 | 37.3% | ‡ weakest survivor |
| `b9b-disc9975b` ‡ | disc **0.9975**, new env | **10.47M** | not measured | — | — | 5.0% | ‡ **dead**, peaked at 328k |
| `b8f-disc9975seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.9975** | 5.47M | **92%** | **86.7%** | **66.3%** /5200 | **69.3%** | **project record**; declining when stopped |
| `b8d-disc995clip` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | **11.64M** | **80%** | 74.7% | 60.4% /2000 | 50.0% | 2nd by measurement, then **died at ~7M** |
| `b7f-disc995seed3` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.06M | 51% | 48.0% | 38.8% /1000 | 44.0% | best of batch 7, and survived |
| `b4c-schlongper` | alpha 0.8, `td_loss`, no IS | 1.06M | 50% | 46.7% | 37.1% /1000 | 34.0% | ties `b7f` on ceiling, **1 of 3 seeds survive** |
| `b7e-disc995seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.28M | 39% | 34.7% | 29.5% /1000 | 32.3% | strong, survived |
| `b6b-alpha06` | alpha 0.6, `td_loss`, no IS | 1.80M | — | — | 24.5% /1000 | 21.7% | old selector, **underestimate**; re-measure |
| `b7d-discount995` | alpha 0.6, `td_loss`, no IS, **disc 0.995** | 1.60M | 26% | 22.7% | 16.4% /1000 | 17.7% | survived, weakest of the three discount seeds |
| `b8e-clipseed2` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | 1.16M | **32%** | — | 32% /100 | 21.3% | one good ckpt, **no good region** — 1 above the floor |
| `b7a-a06seed2` | alpha 0.6, `td_loss`, no IS | 2.00M | 19% | 18.3% | 12.0% /1000 | 15.0% | survived to 2M, low ceiling |
| `b6a-alpha04` | alpha 0.4, `td_loss`, no IS | 1.41M | — | — | 8.1% /1000 | 14.3% | stable, never near death, low ceiling |
| `b5d-schlongTDE` | alpha 0.8, `td_error`, no IS | 2.07M | — | — | 6.6% /1000 | 10.7% | stable, low ceiling |
| `b5c-schlongIS` | alpha 0.8, `td_loss`, **IS on** | 2.31M | — | — | 2.1% /1000 | 17.0% | IS correction cancels the benefit; peak ckpt evicted |
| `b8c-disc9975` | alpha 0.6, `td_loss`, no IS, **disc 0.9975** | 1.75M | — | — | not measured | 14.7% | monotone decline to a stop; `b8f`'s sibling |
| `b8g-clipseed3` | disc 0.995 + **`GRADIENT_CLIPPING=10`** | 3.43M | — | — | **none >50%** | 30.0% | **died, recovered after 1.2M, died again** |
| `b7b-a06seed3` | alpha 0.6, `td_loss`, no IS | 1.78M | — | — | 0% | 7.7% | **died at 1162k** |
| `b7c-a06seed4` | alpha 0.6, `td_loss`, no IS | 1.74M | — | — | 0% | 9.7% | **died at 573k** |
| `b8a-disc999` | alpha 0.6, `td_loss`, no IS, **disc 0.999** | 1.11M | — | — | 0% | 0.7% | **died at 452k**, peak trailing only 63.1 |
| `b8b-disc999seed2` | alpha 0.6, `td_loss`, no IS, **disc 0.999** | 1.41M | — | — | 0% | 0.0% | **zero perfect games in 1.41M steps** |
| `b1a-base` | none (control) | 503k | — | — | — | 16.7% | collapsed at 265k; score recovered, skill did not |
| `b3a-epsfloor` | `MIN_EPSILON=0.001` | 545k | — | — | — | 11.0% | best of batch 3, degraded anyway |
| `b4b-unifbuf500k` | alpha 0 + 500k buffer | 1.23M | — | — | — | 9.3% | steadiest arm, but a low ceiling |
| `b4a-uniform` | alpha 0 | 1.25M | — | — | — | 8.7% | peaked ~575k, drifted down |
| `b3b-epsfloor2` | `MIN_EPSILON=0.001` | 549k | — | — | — | 8.3% | declined despite the floor — falsified hypothesis A |
| `b2a-base2` | none (repeat) | 999k | — | — | — | 7.0% | the 1M-step reference: no collapse, long oscillation, 1.1% at the end |
| `b3c-buf500k` | 500k buffer, alpha 0.6 | 4.81M | — | — | — | 5.7% | **died at ~750k**, score 0.0 for 4M steps |
| `b5a-schlong` | alpha 0.8, `td_loss`, no IS | 2.05M | — | — | — | 10.0% | **died at 272k**, `b4c` repeat |
| `b5b-schlong2` | alpha 0.8, `td_loss`, no IS | 1.92M | — | — | — | 7.7% | **died at 246k**, `b4c` repeat |
| `b1c-nstep3` | `N_STEP_UPDATE=3` | 1.14M | — | — | — | 1.7% | dead end |
| `b1b-tgt200` | `TARGET_UPDATE_PERIOD=200` | 106k | — | — | — | 1.0% | stopped early, verdict weak |
| `b2b-nstep2` | `N_STEP_UPDATE=2` | 580k | — | — | — | 0.7% | dead end |

**† batches 11 and 13's `measured` column is the graph-100% tier only, not the whole arm.** Batch 10 was
measured flat — every selected checkpoint at 100 episodes — so pooling all its rows is already an
equal-effort figure. Batch 11 ran the three-stage protocol, where the rows have different depths and
the deep ones were chosen *because* they screened well, so pooling them reads high by construction.
The equal-effort fix is to truncate every checkpoint to its first 20 episodes, and that is not
recoverable here: the output file stores per-checkpoint totals, not per-episode results, and these
four runs predate the `pooled_equal_effort` field that computes it in-process. What *is* clean in
both batches is the graph-100% tier — 100 episodes each, no screening applied in either — so that is
what these four rows report, and it is the column the batch 11 vs batch 10 comparison below uses.
The episode counts are correspondingly smaller (`/4800` against batch 10's `/27200`).

Batch 13 ran the same three-stage protocol and *does* have `pooled_equal_effort` (68.7 / 71.6 / 73.2
/ 74.8 across seeds 1-4). Its rows still report the graph-100% tier so the column means one thing
down the table and so the batch 11 comparison is like-for-like; the equal-effort figures are in the
batch 13 write-up below.

**§§ batch 14's `measured` column is `pooled_equal_effort` (`/eq`), because the graph-100% tier does
not exist for it.** Batch 14 is the first batch measured under `EVAL_MIN_ACHIEVABLE=90`, and the gate
truncates every tier member that falls below 90%, leaving the tier's own top slice — it reads 90.3%
against batch 13's 74.6% and means nothing. `pooled_equal_effort` truncates to the 20-episode screen
depth, which is at or below the abandon floor, so it is exact at any gate and is the **only** column
here that compares cleanly to batch 13 (72.07% against 72.08%). When comparing `measured` across the
† and §§ rows, don't — compare the batch write-ups, which state both figures.

**§§ Batch 17's gate is 95, the same as batch 16's**, so those two batches' `best ckpt` columns are
comparable to each other and to nothing above them. `b17b` is the exception in both directions: it is
the only arm in either batch with full 100-episode rows, and it has 26 of them.

**‡ `b17b-forkseed2` displaces `b15b` at the top of this table on every column at once** — best ckpt
99% vs 97%, top-3 98.7% vs 96.3%, eq-effort 82.42% vs 79.7%, best-30 92.7% vs 89.3% — and it did it at
**1.57M steps against 5.75M**. Its re-measurement is outstanding, so read the 99% as provisional; the
eq-effort figure is not provisional, and it is a record on its own.

`train` was a human-started run on committed defaults, stopped by the human. Never
touch `snek2/savedPolicies/train*`.

Four things this ranking makes visible that per-batch reading did not:

- **`DISCOUNT=0.995` ties the best ceiling and keeps it.** `b7f` (51%) and `b4c` (50%) are
  a dead heat, but `b4c`'s config dies in 2 of 3 seeds while all three discount seeds lived.
  Priced for survival that is 28.2% against 12.4% expected.
- **Everything above ~12% shares one config family**: alpha 0.6-0.8, `td_loss` priorities,
  no IS weights. The discount is the only addition that has helped on top of it.
- **`b1a-base`, a plain baseline, outranks four deliberate interventions**
  (`MIN_EPSILON`, both n-step values, the 500k buffer with PER). Most changes tried in
  this investigation made things worse.
- **The graph misranks arms badly.** `b5c-schlongIS` is 2nd of the batch-5/6 arms by best
  perfect-30 (17.0%) and **last by measurement** (2.1%). Any ranking built on 10-episode
  graph evals is unreliable; see [`hyperparamTuning.md`](hyperparamTuning.md).
- **Dying late is not the same as dying.** `b8g-clipseed3` sits below arms with a third of its
  best perfect-30 because it ended dead, but it spent 1.2M steps at zero and *recovered* to
  63.7 trailing before collapsing again. A ranking by endpoint hides that entirely — see
  [`findings.md`](findings.md).

**The top two rows were measured mid-run at ~2.6-2.9M steps**, not at the final steps shown, because
both arms kept training after measurement and then declined. Those are their best figures, and their
record checkpoints are preserved in [`../hallOfFame/`](../hallOfFame/README.md).

**Read the `final steps` column with care — it is not a quality signal.** `b8d`'s 11.64M is the
largest number in this table and it died; the same arm's best checkpoint came at 2538k. Both top
arms peaked at ~2.5-3M and were stopped well past it. Everything below them was stopped before
~2.1M, and the four next-best at ~1.06M, so **this ranking compares most configs at a horizon where
they had not finished improving** — see [`findings.md`](findings.md).

## Batch 30 — chase-safe shaping on `fc 200,100,100`, `c=0.10`: **null, and it completes the shaping×architecture 2×2**

**b27's shaping term on b25's net.** Potential-based chase-safe shaping — `c·(γΦ(s′) − Φ(s))` with Φ = 1
iff the head and tail share a free region that also holds the food and the snake is ≥85 long — on
`fc 200,100,100`, IS off, `td_error`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no
food-distance shaping, **2M cap**, seeds 1-4. The seed-matched control is **b25** (`-r2`). With b24
(`fc 320` control) and b27 (`fc 320` shaped) it closes a **2×2 of shaping × architecture**, so the shaping
result no longer rests on one net.

**Trained and closed out on the laptop, after a killed-and-resumed close-out.** The first close-out pass was
killed at ~80-100% screened; the relaunch with `EVAL_RESUME=1` reused ~75k banked episodes and finished the
rest, then a HOF-500 re-measure ran on the three arms with a ≥98%/100 checkpoint (only the valid post-fix
runs `b30e-h`; the first launches `b30a-d` trained under the perfect-counting bug and are discarded — see
[`findings.md`](findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter)).

**The result is a null, pointing the same way as b27.** Close-out pools **83.3** (eq-effort, gate 95),
**−2.7 under the b25 control's 86.0** — the identical gap and direction as b27's 85.2 vs b24's 87.9. And on
the metric that decides it, **0 of b30's 10 ≥98%/100 checkpoints held ≥98%/500**: every one deflated (best
`b30e` @651k 96.1%), exactly as b25's four arms did (best 97.2%). So on `fc 200,100,100` shaped and control
are a dead heat at zero records.

| arm | peak trail | best-30 | `sef` | pooled (eq) | HOF-500 |
|---|---|---|---|---|---|
| `b30e` seed1 | 95.00 | 93.7% | 58.4% | 83.75 | 6 ckpts, best 96.1% (ab.) — **0 held** |
| `b30g` seed3 | 95.00 | 93.3% | 58.3% | 84.28 | 3 ckpts, best 95.3% (ab.) — **0 held** |
| `b30f` seed2 | 94.92 | 92.3% | 55.9% | 84.32 | 1 ckpt, 90.9% (ab.) — **0 held** |
| `b30h` seed4 | 95.00 | 92.3% | 55.0% | 81.00 | no ≥98%/100 |
| **mean** | | **92.9%** | **56.9%** | **83.3** | **0 of 10 held** |

**The 2×2 verdict, at gate 85.** On `fc 320` the control (b24) produced two records and the shaped arm (b27)
none; on `fc 200,100,100` neither reaches a record. Chase-safe shaping at `c=0.10`, **gate 85**, produces no
record-tier checkpoint on either net and removes the control's records on the wider one. **b28** (`c=0.20`,
gate 85) later confirmed the dose is not the issue — also 0 records — as the Φ calibration
([findings.md](findings.md#-measured-the-chase-safe-potential-is-nearly-static-for-a-record-policy-and-busy-for-a-bad-one))
predicted, since Φ carries ~0 at the lengths a gate-85 term grades. **The lever turned out to be the gate, not
the dose: `b29` (`c=0.10`, gate 75) produced a 21-checkpoint ≥98%/500 region** — see Batches 28-29 below.
Full conclusion: [`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

**How it was judged: the control at a matched 2M horizon, seed by seed.** b25/b24 ran to 3M, so their
summaries were recomputed truncated at 2M (`run_report.build_summary`) for a like-for-like read against
b30's 2M cap:

| seed | b25 best-30 @2M | b25 `sef` @2M | b24 best-30 @2M | b24 `sef` @2M |
|---|---|---|---|---|
| 1 | 93.7 | 61.4 | 94.0 | 46.7 |
| 2 | 95.3 | 57.9 | 96.7 | 65.1 |
| 3 | 93.7 | 61.0 | 96.0 | 57.0 |
| 4 | 91.7 | 54.2 | 96.7 | 64.1 |
| **mean** | **93.6** | **58.6** | **95.8** | **58.2** |

`320` leads `200,100,100` on best-30 at 2M (95.8 vs 93.6) while `sef` is a dead heat — consistent with the
widest-layer ordering and `best_perfect30` being the sharper metric at this level. b30's best-30 (92.9)
sits just under its b25 control (93.6), the −0.7 of the shaping null; `b25d` was the weak seed of that wave
(91.7 / 54.2), so a b30 seed 4 that merely matched it would not have been a null.


## Batches 28-29 — chase-safe dose and gate: **the gate is the lever, and gate 75 produces a record region**

**Two arms extend the gate-85 nulls onto the two axes left untested — dose and gate.** Both are `fc 320`,
IS off, `td_error`, `TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance
shaping, **2M cap**, seeds 1-4, seed-matched control **b24** (the record-holder). `b28` raises the dose to
**`c=0.20`** at gate 85; `b29` drops the gate to **75** at `c=0.10`. Both trained, closed out and HOF-500
re-measured on the desktop.

**`b28` (`c=0.20`, gate 85) is a null — the dose is not the issue.** Same direction as b27/b30: pooled
**85.4**, ~2.5 under the b24 control's 87.9, and **0 of 4 arms** produced a checkpoint holding ≥98%/500.

| arm | peak trail | best-30 | `sef` | pooled (eq) | HOF-500 |
|---|---|---|---|---|---|
| `b28d` seed4 | 95.00 | 96.7% | 68.7% | 89.10 | best 96.8% @1061k (341 ep, ab.) — **0 held** |
| `b28a` seed1 | 95.00 | 96.0% | 54.0% | 89.72 | best 95.6% @1727k (275 ep, ab.) — **0 held** |
| `b28c` seed3 | 95.00 | 92.7% | 33.4% | 82.67 | none reached the gate — **0 held** |
| `b28b` seed2 | 94.94 | 90.7% | 47.9% | 80.15 | best 90.0% @1127k (120 ep, ab.) — **0 held** |
| **mean** | | **94.0%** | **51.0%** | **85.4** | **0 of 4 held** |

**`b29` (`c=0.10`, gate 75) produces a record region — this is the positive result.** Pooled **87.8**, a dead
heat with the b24 control (87.9), and best-30 **96.4** (the highest of any shaped batch) — but the tell is the
HOF-500: **21 checkpoints held ≥98%/500 across two of the four seeds**, where the record-holding control b24
only ever produced 2 isolated ones across all four. `b29b` alone carries an **18-checkpoint contiguous band**
(1446k-1529k), peaking at **`b29b` @1447k = 99.0%/500 (495/500)** — a point estimate *above* the project
record b24d/b24b at 98.0%/500.

| arm | peak trail | best-30 | `sef` | pooled (eq) | HOF-500 (≥98%/500) |
|---|---|---|---|---|---|
| `b29a` seed1 | 95.00 | 97.7% | 60.3% | 89.76 | **3 held**, best 98.4% @1347k |
| `b29c` seed3 | 95.00 | 96.3% | 67.9% | 89.68 | 0 held (best 97.1%, 378 ep ab.) |
| `b29b` seed2 | 95.00 | 97.3% | 55.6% | 87.14 | **18 held** (1446k-1529k), best **99.0% @1447k** |
| `b29d` seed4 | 95.00 | 94.3% | 59.7% | 84.75 | 0 held (best 90.6%, 127 ep ab.) |
| **mean** | | **96.4%** | **60.9%** | **87.8** | **21 held, in 2 of 4 seeds** |

**The verdict: the gate is the lever, not the dose or the net.** Gate 85 is null on `fc 320` (b27), on
`fc 200,100,100` (b30) and at doubled dose (b28); gate 75 (b29) matches the control's pooled/best-30 *and*
produces a record region the control never did. The Φ calibration is why — the potential carries ~0 at
lengths 98-99, so a gate-85 term grades the flat final approach, while gate 75 turns the term on ten meals
earlier, in the packing decisions that decide whether the endgame is winnable. **`b29b` @1447k is now the
project record**, promoted to [`../hallOfFame/`](../hallOfFame/README.md) on 2026-08-16 (rsynced off the
desktop, the copy re-measured 98/100 on fresh laptop episodes). The 99.0 vs 98.0 lead over `b24d` is inside
the 500-episode CIs — a narrow point lead, taken as the record under the folder's 500-episode standard — but
the *region* (21 held across 2 seeds, an 18-wide band) is the signal outside noise, and the first time the
top tier appears as a plateau rather than a spike. Full conclusion:
[`findings.md`](findings.md#-chase-safe-reward-shaping-null-at-gate-85-at-any-dose-records-at-gate-75--the-gate-is-the-lever).

## Batch 26 — FC `100,100` under IS-off: **the shallow shape does not carry the lift**

Design, and what it isolates: b22's exact IS-off config (`SNEK_IS_WEIGHTS=0`, `td_error` priority,
`TARGET_UPDATE_PERIOD=1000`, `DISCOUNT=0.9975`, `FORK_BRANCHES=4`, no food-distance shaping, 3M cap,
seeds 1-4) with the network changed to a **two-layer `100,100`** — **13,604 parameters, 1.14× the
control and 1.21× b24's `320`**. It is the third shape in the width follow-up, after b24 (`320`,
+12.2) and b25 (`200,100,100`, +10.3), and it asks the one question those two leave open: **is the
consolidation lift bought by size, or by the width of the widest layer?**

**The shallow shape does not carry it** — pooled **79.2** is only **+3.5 over the b22 control's
75.7**, a quarter of b24's gain, and one seed (`b26d`, `sef` 13.8%) lands *below* the control mean.
No arm produced a ≥98%/100 checkpoint, so the auto HOF-500 selected nothing and the record stays
b24's.

**And this is the arm that separates the two explanations, against what b25's write-up first
concluded.** `100,100` has *more* parameters than b24's `320` and gets a quarter of the lift, while
`320` is the *smallest* of the four nets and gets the most. Parameter count is not even monotone with
the result; the widest layer is:

| shape | params | ×control | widest layer | pooled lift vs b22 |
|---|---|---|---|---|
| `320` | 11,204 | **0.94×** | **320** | **+12.2** |
| `200,100,100` | 36,804 | 3.09× | 200 | +10.3 |
| `100,100` | 13,604 | 1.14× | 100 | +3.5 |
| `50,100,50` (control) | 11,904 | 1.00× | 50 | — |

Counted with `under_the_hood.build_q_net` at `obs_len=30`, `num_actions=4`. **So "the lift tracks
capacity" is wrong and is retracted** — see
[`findings.md`](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count).
The next architecture arm this implies is a *wider* first layer (`512`), not a bigger net.

| mean of 4 | b22 control (`50,100,50`) | b24 (`320`) | b25 (`200,100,100`) | **b26 (`100,100`)** |
|---|---|---|---|---|
| peak trailing | 94.88 | 95.00 | 95.00 | **94.93** |
| `sef` (training) | 30.5 | 66.0 | 63.8 | **42.1** |
| best-30 | 86.2 | 96.2 | 94.3 | **88.4** |
| close-out pooled (eq-effort, gate 95) | 75.7 | 87.9 | 86.0 | **79.2** |
| ≥98%/500 checkpoints | — | **2** | 0 | **0** |

Per seed — close-out pooled: b 83.8 · c 83.2 · a 80.0 · d 69.6. Full-length rows clearing gate 95:
15 · 7 · 2 · 0, against b25's 81 · 64 · 50 · 47 — the shallow net simply spends far less of the run
in the band a close-out can measure.

**The reading for the ceiling is unchanged and now three-shapes deep**: peak trailing is 94.84-95.00
across `50,100,50`, `100,100`, `200,100,100` and `320`. Architecture buys time near the ceiling and
never the ceiling itself — the result batch 20 reached under β→1.0 and this family confirms under
IS-off. Charts: [`charts.md`](charts.md).

## Batch 25 — FC `200,100,100` under IS-off: **the lift replicates at a second shape — but no record**

Design: identical to batch 26 above except the net is a **3-layer `200,100,100`** — 36,804
parameters, **3.09× the control** (not the "~1.6×" this folder said before the count was actually
run) — launched as `-r2` after the first four arms failed at launch. It asks whether b24's `320`
result survives at a shape other than one wide layer.

**It does.** Close-out pools **86.0**, +10.3 over the control and within 1.9 of b24's 87.9. Peak is
unmoved at 95.00 on all four seeds. **The original reading of this — "so the gain tracks capacity,
not width" — did not survive b26**, which has more parameters than b24 and gets +3.5; what the two
arms share is a wide first layer (200 and 320). See the b26 section above and
[`findings.md`](findings.md#-corrected-2026-08-14-the-is-off-architecture-lift-tracks-the-widest-layer-not-the-parameter-count).

**But the batch produced no hall entry, and the reason is the gate rather than the arms.** The
HOF-500 runs at gate 98; every ≥98%/100 candidate was abandoned before 500 episodes. The strongest,
`b25b` @911k, was still **97.2% at 392 episodes** when the gate stopped it — a plausible ~97%/500
holder that the folder's own gate-97 standard would have measured to completion. It is the one
outstanding hand re-measure this batch leaves behind.

| arm | peak trail | best-30 | `sef` | close-out pooled | HOF-500 (gate 98) |
|---|---|---|---|---|---|
| `b25c` | 95.00 | 93.7% | **66.9%** | **87.2** | best partial 95.3% @827k (ab.) |
| `b25d` | 95.00 | 94.3% | 62.2% | 85.9 | best partial 96.4% @2431k (ab.) |
| `b25a` | 95.00 | 93.7% | 63.2% | 85.6 | best partial 92.7% @802k (ab.) |
| `b25b` | 95.00 | **95.3%** | 62.7% | 85.5 | best partial **97.2%** @911k, 392 ep (ab.) |
| **mean — b25** | **95.00** | **94.3%** | **63.8%** | **86.0** | 0 of 4 held ≥98%/500 |
| **mean — b22 control** | 94.88 | 86.2% | 30.5% | 75.7 | — |

**This batch was also the first end-to-end run of the auto-HOF chain** (training → close-out →
HOF-500, `auto_hof` on by default since 2026-08-13), and the chain worked: it queued, measured and
published without a hand step. Its one lesson is the gate — a chain fixed at 98 cannot promote a
97%-class checkpoint, so a strong arm can finish with an empty HOF job and still be worth a manual
gate-97 pass. Charts: [`charts.md`](charts.md).

## Batch 24 — FC width `320` under IS-off: **the first architecture result, and a new record**

**Trained on the desktop, four arms `b24a`-`b24d` at seeds 1-4, all to the 3M cap; closed out on the desktop
under gate 95, `EVAL_WORKERS=4`, then HOF-500 re-measured (every ≥97%/100 checkpoint at 500 fresh episodes,
gate-97 early-abandon).** Charts in [`charts.md`](charts.md) (batch 24 section). The record and any HOF claim
come from the HOF-500 rows, not the /100 close-out rows.

### The design

Batch 22's exact IS-off config — `td_error` priority α=0.6, `SNEK_IS_WEIGHTS=0`, target 1000, disc 0.9975,
guided 0.8, fork 4/0.5/85/60, food-distance off — with the network widened to a single **`320`** layer (batch
20's `320` shape). Width is the only change. The seed-matched control is b22 (`50,100,50`, IS off). It asks the
one architecture question batch 20 could not: **does width matter once the prioritisation is fixed at IS-off?**
Batch 20 asked width under the β→1.0 control and found nothing — nine shapes, no ceiling movement, the
consolidation columns exposed as seed noise.

### The result — vs the b22 control (IS off, `50,100,50`)

| mean of 4 | b22 control (fc 50,100,50) | b24 (fc 320) | b24 − control |
|---|---|---|---|
| peak trailing | 94.88 | 95.00 | +0.12 |
| `sef` (primary) | 30.5% | 66.0% | +35.5 |
| best-30 | 86.2% | 96.2% | +10.0 |
| close-out pooled (eq-effort, gate 95) | 75.7% | **87.9%** | **+12.2** |

**Width raises consolidation under IS-off, and the close-out backs the graph — pooled 87.9, +12.2 over the
control and higher on all four seeds** (seed-for-seed 89.03 / 88.84 / 87.68 / 85.97 vs control 78.89 / 72.51 /
75.04 / 76.27; exact sign-test p=0.0625, the n=4 floor). That is the highest pooled of any gate-95 arm on
record — above the b18b record's 78.5. **Peak trailing is unmoved at 95.00**, the top of the flat 94.7-95.0
band that has held since batch 11: width does not raise the ceiling, only how much of the run sits near it.
This is the project's first sign that **width and prioritisation interact** — width paid nothing under β→1.0
(batch 20) and pays here under IS-off.

### The HOF-500 — a new record, and the selection-inflation lesson at scale

The /100 close-out was flashy — two 100.0% full-length checkpoints (`b24a` @1633k, `b24c` @2126k) and **199
checkpoints ≥97%/100** across the batch. The HOF-500 re-measured all 199 at 500 fresh episodes and shows how
much of that was selection: **only 9 held ≥97%/500.**

| arm | ≥97%/500 held / ≥97%/100 candidates | best HOF-500 row |
|---|---|---|
| `b24d` | 3 / 51 | **98.0%** @1342k (490/500, CI [96.4,98.9]) |
| `b24b` | 3 / 59 | **98.0%** @2860k (490/500, CI [96.4,98.9]) |
| `b24c` | 3 / 46 | 97.4% @2982k (CI [95.6,98.5]) |
| `b24a` | **0** / 43 | — (all abandoned) |

**`b24d` @1342k is the new record: 98.0% over 500 fresh episodes**, narrowly ahead of `b18b` @1588k
(97.6%/700) — a lead on the point estimate with overlapping intervals, taken as confirmed under the folder's
500-episode standard. `b24b` @2860k ties it at 98.0%/500. The tell that both are real regions, not lucky
draws: `b24d` @1342k *rose* on re-measurement (97.0/100 → 98.0/500), the same signature that marked `b18b`
(98/100 → 97.4/500). Both are in [`../hallOfFame/`](../hallOfFame/README.md).

**And the counter-lesson: `b24a` produced the batch's two flashiest /100 numbers and zero survivors.** Its
100%/100 @1633k held 0 of 43 ≥97%/100 checkpoints at 500 episodes — a clean demonstration that a selected
100%/100 is mostly selection. The survivors cluster by seed (b24b/b24c late at 2.7-3.0M, b24d early at
0.9-1.36M), so the strong region's location is seed-dependent, not a batch-wide step.

**Caveats on the consolidation finding.** n=4 cannot resolve below ~10 pp, and batch 20's `320` looked strong
on the graph before its pooled gap proved to be seed noise. But that was a within-batch iso-capacity confound;
here the gap is +12.2 higher on all four **seed-matched** controls, and the HOF-500 confirms the batch owns 9
genuine ≥97%/500 checkpoints rather than only inflated /100 highs. Still a consolidation gain at n=4, not a
raised ceiling — the peak is unmoved at 95.00.

## Batch 23 — β annealed 0→0.1: **the best point on the β ladder, near the no-IS extreme**

**Trained on the laptop, four arms `b23a`-`b23d` at seeds 1-4, all to the 3M cap; checkpoints rsynced to
the desktop and closed out there under gate 95, `EVAL_WORKERS=4`.** Charts in
[`charts.md`](charts.md) (batch 23 section).

### The design

One step further down the IS-β ladder than b21: importance-sampling β annealed from **0 to 0.1** over 300k
(`SNEK_IS_BETA=0`, `SNEK_IS_BETA_FINAL=0.1`), `td_error` priority and α=0.6 unchanged (fc 50,100,50). At the
anneal target the update keeps **α·(1−β)=0.54** of the priority signal on the gradient — between b21's 0.30
(β→0.5) and the full 0.6 with IS off (b22). Seeds match b21/b22 for a seed-for-seed compare.

### The result — vs the β→1.0 control (`b20a-d`) and vs b21 (β→0.5)

| mean of 4 | control β→1.0 | b21 β→0.5 | b23 β→0.1 | b23 − control | b23 − b21 |
|---|---|---|---|---|---|
| peak trailing | 94.44 | 94.69 | 94.90 | +0.46 | +0.21 |
| `sef` (primary) | 11.2% | 14.3% | 33.1% | +21.9 | +18.8 |
| best-30 | 64.0% | 74.1% | 85.8% | +21.8 | +11.7 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 64.3% | 75.7% | +20.7 | +11.4 |

**β→0.1 is the strongest point measured on the β ladder, and the close-out backs the graph — pooled 75.7,
+20.7 over the control and +11.4 over b21, higher on all four seeds than either** (seed-for-seed pooled
77.2 / 82.1 / 71.5 / 72.1 vs control 33.2 / 62.7 / 52.8 / 71.3 and vs b21 59.5 / 68.8 / 62.8 / 66.0; exact
sign-test p=0.0625 each, the n=4 floor). It closes most of the gap to batch 18 (`td_loss`, IS off, ESS/N
0.21: pooled ~78.8, best-30 87.3%, `sef` 34.6%), the strongest consolidation on record. The four points now
measured make a monotone ladder — pooled control 55.0 (ESS/N ≈1.0) → b21 64.3 (0.86) → **b23 75.7 (0.54)** →
b18 ~78.8 (0.21): **more prioritisation on the gradient tracks better learning**, and even a little IS
correction (β=0.1) barely costs anything relative to none. The last point, **b22** (`td_error`, IS off,
ESS/N ≈0.38), came in at **pooled 75.7 — a dead heat with b23** (see the b22 section below), so the gain
saturates by β→0.1: removing the last of the correction adds nothing.

**`b23b` produced the batch's standout region on the graph — five full-length checkpoints ≥95/100 clustered
at ~777k (best 97/100, top-3 96.3%), pooled 82.1**, the highest eq-effort of any β-ladder arm. It looked
like a genuine hall-of-fame candidate, but the folder's re-measurement protocol falsified it: at **500 fresh
episodes the close-out-selected @777k reads 92.4%** — the *worst* of its own cluster, a textbook selection
artifact, and well below the 97.6% record. So no b23 checkpoint enters the hall of fame. The other three
seeds cleared no full-length row under gate 95 (best rows are 20-episode screens, 75-80%), so their `pooled`
is the figure to read. Peak trailing 94.78-95.00 across the batch — the ceiling is unmoved, as it has been
since batch 11; only consolidation differs.

## Batch 22 — IS off: **a dead heat with β→0.1 — the consolidation gain saturates**

**Trained on the desktop, four arms `b22a`-`b22d` at seeds 1-4, all to the 3M cap; closed out on the
desktop under gate 95, `EVAL_WORKERS=4`.** Charts in [`charts.md`](charts.md) (batch 22 section).

### The design

The bottom rung of the IS-β ladder short of the no-IS extreme: importance sampling **off**
(`SNEK_IS_WEIGHTS=0`), so the gradient carries the full `|δ|^0.6` prioritisation with no correction
(ESS/N ≈0.38). `td_error` priority, α=0.6, otherwise batch 20's control (fc 50,100,50). Seeds match
b21/b23 for a seed-for-seed compare. It answers the one question b23 left open: does taking the last of
the IS correction off keep buying consolidation, or has the gain flattened?

### The result — vs b23 (β→0.1) and the β→1.0 control

| mean of 4 | control β→1.0 | b23 β→0.1 | b22 IS off | b22 − b23 | b22 − control |
|---|---|---|---|---|---|
| peak trailing | 94.44 | 94.90 | 94.88 | −0.02 | +0.44 |
| `sef` (primary) | 11.2% | 33.1% | 30.5% | −2.6 | +19.3 |
| best-30 | 64.0% | 85.8% | 86.2% | +0.4 | +22.2 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 75.7% | 75.7% | ±0.0 | +20.7 |

**It is a dead heat with b23.** Pooled 75.7 vs 75.7, best-30 86.2 vs 85.8, `sef` 30.5 vs 33.1 — every
column is inside seed noise, and the seed-for-seed pooled figures (78.9 / 72.5 / 75.0 / 76.3 for b22 vs
77.2 / 82.1 / 71.5 / 72.1 for b23) split 2–2. So the ladder — control 55.0 (ESS/N ≈1.0) → b21 64.3 (0.86)
→ **{b23 75.7 (0.54), b22 75.7 (0.38)}** → b18 ~78.8 (0.21) — **flattens at the bottom**: most of the
consolidation is bought by the time β reaches 0.1, and removing the residual IS correction adds nothing
measurable. (b18's higher point is a different base — `td_loss` priority, not `td_error` — so it is not
the next rung of this ladder.)

**No hall-of-fame candidate.** `b22a` produced the batch's one clean full-length row — 96/100 @1075k,
pooled 78.9, `sef` 46.3% — and `b22d` a lone 95/100 late at 2275k; both are below the 97.6% record and,
by the shrink pattern that has held for every selected high but `b18b`, are not record candidates. The
other two seeds cleared no full-length row under gate 95 (best rows are 20-episode screens), so their
`pooled` is the figure to read. Peak trailing 94.78-94.94 — the ceiling is unmoved, as since batch 11.

## Batch 21 — partial IS (β→0.5): **beats the β→1.0 control, still far behind no-IS**

**Trained on the laptop, four arms `b21a`-`b21d` at seeds 1-4, all to the 3M cap; checkpoints rsynced to
the desktop and closed out there under gate 95, `EVAL_WORKERS=4`.** Charts in
[`charts.md`](charts.md) (batch 21 section).

### The design

One knob off batch 20's control: importance-sampling β annealed to **0.5** instead of 1.0, `td_error`
priority and α=0.6 unchanged (fc 50,100,50, `SNEK_IS_BETA_FINAL=0.5`). At β=1 the IS weight cancels the
prioritised sampling and the gradient is near-uniform (ESS/N ≈0.95-1.0); at β=0.5 it leaves `|δ|^{0.30}`
of prioritisation on the gradient (**ESS/N ≈0.86**, measured on the four end-of-run buffers, tight
0.856-0.866). The aim: keep IS's anti-forgetting effect while letting the valuable end-game transitions
pull harder.

### The result — vs the β→1.0 control (`b20a-d`)

| mean of 4 | control β→1.0 | b21 β→0.5 | delta | p (16 flips) | favour b21 |
|---|---|---|---|---|---|
| peak trailing | 94.44 | 94.69 | +0.25 | 0.375 | 3/4 |
| `sef` (primary) | 11.2% | 14.3% | +3.1 | 0.625 | 3/4 |
| best-30 | 64.0% | 74.1% | +10.1 | 0.375 | 3/4 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 64.3% | +9.3 | 0.250 | 3/4 |

**β→0.5 is directionally better than β→1.0 — best-30 +10 pp, close-out pooled +9 pp, 3 of 4 seeds — but
n=4 cannot resolve it** (p 0.25-0.63, seed 4 flips on the graph metrics). The close-out agrees with the
training graph, so the edge is not a graph artefact. It is nowhere near batch 18 (`td_loss`, IS off:
best-30 87.3%, `sef` 34.6%, ESS/N 0.21), the strongest consolidation on record. Across the three points
now measured — β→1.0 (ESS/N ≈1.0) < β→0.5 (0.86) ≪ no-IS (0.21) — **more prioritisation on the gradient
tracks better learning**. That is what **batch 22** (`td_error`, IS off, ESS/N ≈0.38, queued on the
desktop) is meant to pin down. Best checkpoints 82-89% (all under the 95 gate, so `*trunc*`); peak
trailing is flat across all three points (94.4-94.9), so the ceiling is unmoved — only consolidation
differs.

## Batch 20 — the design of the nine-shape sweep (complete 2026-08-12)

Moved here from [`runs.md`](runs.md) when the last shape closed out. **Read this before reading any
individual batch-20 write-up below** — it is what says whether a given result was informative or was
never going to answer anything. Conclusions are in
[`findings.md`](findings.md#network-shape-the-sweep-is-complete--nine-shapes-and-architecture-never-raises-the-ceiling).

**The question was the one thing nine batches of optimiser knobs had not moved: the ceiling.** Network
architecture had **never been varied in this project** — `FC_LAYERS` sat at `(50, 100, 50)` from batch 1
to batch 19 with not a single measurement behind it.

### The shapes, and what each one was for

30 inputs → 3 actions (`left/right/forward`), ReLU, He init. Parameter counts include biases. **Each
shape owned a fixed block of batch letters** (`b20<letter>-fc<shape>seed<N>`, one letter per seed),
assigned up front so two hosts never reused a letter — which happened once anyway, `b20q-t` naming
*both* `25,50,25` and `60,30,30,30,30`, resolved by renaming the latter to `u-x`.

| shape | params | vs control | depth | arms | what it isolated — and what it found |
|---|---|---|---|---|---|
| `50,100,50` (control) | 11,853 | 1.00× | 3 | `b20a-d` | the re-baseline at β=300k; every shape reads against it |
| `200,100,50` | 31,503 | **2.66×** | 3 | `b20e-h` | capacity up, shape preserved — **null** (peak +0.26, 3/4, p 0.375) |
| `200,50` | 16,403 | 1.38× | 2 | `b20i-l` | **wide-early**: conjunctions of engineered features — **null** (peak +0.12, `sef` −2.6) |
| `320` | 10,883 | 0.92× | **1** | `b20m-p` | **depth, at matched capacity** — **null; depth buys nothing** |
| `25,50,25` | 3,428 | **0.29×** | 3 | `b20q-t` | is capacity binding at all — **yes: ceiling drops, 4/4 seeds** |
| `60,30,30,30,30` | 6,573 | 0.55× | **5** | `b20u-x` | deep and narrow — matches the ceiling, forgets ~2× more |
| `93,93` | 11,907 | 1.00× | 2 | `b20aa-ad` † | the iso-param depth-2 rung — **null (peak −0.03)** |
| `100,200,100` | 43,703 | **3.69×** | 3 | `b20ae-ah` † | escalation above 2.66× — **null (peak +0.17, p 0.875)** |
| `100,50,50` | 10,853 | 0.92× | 3 | `b20ai-al` † | reshuffle at matched capacity/depth — **null, and it priced the noise floor** |

**† past `x` the letters roll into double letters (`aa`, `ab`, …), still batch 20.** Nine shapes at four
seeds is 36 arms, more than the 26 single letters. `batch_prefix` groups `[a-z]*`, so `b20aa-` reads as
`b20` (fixed 2026-08-10, with a test); `y,z` were left spare so each shape kept a clean four-letter block.

**Wide-early was the one with a mechanism behind it.** The observation is already high-level — per-action
safety triples, tail-following flags, food direction — so what the net needs is *conjunctions* of those
features ("safe left AND food left AND tail not adjacent"), not a deep hierarchy. Conjunctions want width
in the first layer, which is what `200,50` and `200,100,50` supplied. It came out null like the rest.

**`60,30,30,30,10` was proposed and deliberately widened to `...,30`.** A 10-unit final layer forces all
three Q-values through 10 ReLU units, where dead units cost a large share of the representation
permanently.

### Why batch 19 was the base, even though it lost

At the user's direction, and the reasoning was sound. Batch 19 was falsified on level but it **cut max
drawdown from 55.52 to 8.76, 4/4 seeds** — the strongest anti-forgetting result on record. The premise
was that **consistency and ceiling are separable**: keep the config that learns steadily, change the
function approximator, and see whether a different shape lifts the level without giving the steadiness
back. That made the target explicit — **a higher ceiling at batch-19 drawdown** — and it means an arm
that raised `sef` while drawdown climbed back toward batch 18's ~57 had not answered the question.

**Drawdown held across the whole sweep** (5.4-12.3 against batch 18's ~57), so the premise survived even
though no shape raised the ceiling.

### ‡ Two variables moved against batch 19, on purpose

**β anneal went to 300k**, the then-current default, at the user's direction. Batch 19 ran the **1M**
schedule, so **batch 20 differed from batch 19 in two ways, not one**: `FC_LAYERS` and
`BETA_ANNEAL_STEPS`.

> **No batch-20 vs batch-19 difference can be attributed to architecture alone.** What *is* clean is the
> comparison **among batch-20 shapes**, since they share β=300k and differ only in `FC_LAYERS`.

**That is why wave 1 re-baselined.** Running the control shape at β=300k cost four arms and bought two
things: it isolated the β change against batch 19 (300k was a new default no arm had run), and it became
the true seed-matched control. Without it every architecture result in the batch would be confounded.

```
SNEK_FC_LAYERS=<shape>  SNEK_SEED=1..4
SNEK_BETA_ANNEAL_STEPS=300000
SNEK_TARGET_UPDATE_PERIOD=1000  SNEK_FOOD_DISTANCE_REWARD=0
SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5  SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60
# PER otherwise at defaults, as batch 19 ran it: td_error priority, IS on, beta 0.4 -> 1.0
```

**‡ β=300k was a wash on ceiling and slightly *better* on drawdown** — matched at batch 19's 2.004M
horizon, so the β schedule is the only difference:

| | control (β=300k) | batch 19 (β=1M) |
|---|---|---|
| peak trailing | 94.41 | 94.16 |
| best-30 | 64.05 | 63.27 |
| `sef` | 12.17% | 13.82% |
| max drawdown | **5.27** | 8.76 |

Every gap is well inside seed noise at n=4, so the β default change is neither validated nor falsified on
the ceiling. What it does show is that **batch 19's anti-forgetting property survived both the β change
and 800k extra steps**.

### The control at 3M — the numbers every shape is read against

| arm | peak trailing | best-30 | `sef` | max drawdown | close-out pooled | best row |
|---|---|---|---|---|---|---|
| `b20a` | 93.80 | 41.3 | 0.2% | 7.66 | 33.2% | 52.0% @2000k (n=25) |
| `b20b` | 94.84 | 78.3 | 16.2% | 4.26 | 62.7% | 83.3% @1802k (n=36) |
| `b20c` | 94.34 | 56.3 | 2.2% | 5.06 | 52.8% | 68.0% @1919k (n=25) |
| `b20d` | 94.76 | 80.3 | 26.3% | 4.68 | **71.3%** | 89.7% @384k (n=58) |
| mean | **94.44** | 64.05 | 11.2% | **5.42** | 55.0% | — |

**No arm produced a full-length row**, in the control or in any of the eight treatments: at gate 95 every
measurement was abandoned, deepest 58 of 100 episodes. So the best-row column is a *bound* throughout
batch 20 and is not comparable across arms or with earlier batches — `pooled_equal_effort` is the exact
column.

**Seeds 1 and 3 are the *control's* two weakest arms** (pooled 33.2 and 52.8, against 62.7 and 71.3), and
that is the single fact most needed to read the per-shape tables: a shape only has to be ordinary on those
two seeds to show a large positive mean, and nearly every apparent shape "win" in the batch is that.
**The weakness is not a property of the seeds themselves** — seed 3 beats its control by +19.4 pooled under
`200,100,50` and +9.9 under `60,30,30,30,30`, where it is the *best* of its own four arms. It is the
control's draw that is unlucky, which is why this batch is read on paired per-seed differences and not on
means.

### Pre-registration

| item | value |
|---|---|
| **control** | wave 1's `50,100,50` at β=300k, seed-matched — the only comparison that isolates `FC_LAYERS`. Batch 19 a *secondary* reference, differing by β as well |
| horizon | run to `max_steps` 2,500,000 so arms could pass batch 19's ~2.0M paired read; later raised to 3M for every shape |
| **co-primary (ceiling)** | **peak trailing** and `best_perfect30`. Batch 19: peak 94.16 mean; batches 11-18 all 94.8-95.0 |
| **co-primary (must not regress)** | **max drawdown**, running-max definition. Batch 19 mean **8.76**. A shape raising the ceiling with drawdown above ~25 would not have answered the question |
| secondary | `strong_eval_fraction` — batch 19 was 13.82% at 2.004M against batch 18's 31.60% |
| test | exact paired permutation over the 16 sign flips, as batches 16-19 |
| abandon | ≥3 of 4 arms not crossing pf30 ≥ 40% by 800k, or drawdown above ~25 on 3+ arms |

### ‡ The width trap, and why it is no longer a manual step

`restore()` is called with `expect_partial()`, so a checkpoint trained at one width and rebuilt at another
**loads with no error** and leaves the mismatched layers unpopulated — the policy then plays like a
beginner. This is the same failure class that once took a 90.3% champion down to scoring 0, 0, 1, and
during batch 20 it meant every eval and `watch.py` run had to be passed the arm's `SNEK_FC_LAYERS` by
hand.

**`arch.json` closed that hole on 2026-08-11** (`policy_arch.py`): every policy dir carries
`fc_layer_params`, `num_actions`, `obs_len`, `obs_era`, and training / `eval_checkpoints` / `eval_workers`
/ `watch.py` all rebuild the *recorded* network and hard-fail (`ArchMismatch`) on a disagreement.
`SNEK_FC_LAYERS` is no longer read at eval or watch time. Older dirs were backfilled by
`backfill_arch.py`. **So the manual override is only needed for a batch-20 checkpoint directory that
predates the sidecar and was never backfilled** — and there it fails loudly rather than silently. **Copy
`arch.json` with any checkpoint** (into `hallOfFame/`, or rsynced between hosts) or it will not load.

## Batch 20 — matched-capacity reshuffle `100,50,50`: **null, and it closes the nine-shape sweep**

**Ran on the desktop `the-claw-den`, four arms `b20ai`-`b20al` at seeds 1-4, all to the 3M cap; close-out
2026-08-12, gate 95, `EVAL_WORKERS=4`, ~5.5 min per arm (597 episodes of 2,400 planned — 45 of 45
measurements abandoned early).** Charts in
[`charts.md`](archive/charts-archive.md#batch-20--matched-capacity-reshuffle-1005050-against-the-control-both-to-3m).
**This was the ninth and last shape**, so batch 20 is complete.

### The design

`FC_LAYERS=100,50,50` — **10,853 params, 0.92× the control, depth 3**. Same depth and effectively the
same capacity as `50,100,50`; the only change is *where* the width sits (front-loaded instead of in the
middle). Pre-registered as an expected null: with depth and capacity both held, there was no mechanism
predicted to move anything. Its value is as the fourth corner of the design — the sweep already had
capacity up (`200,100,50`, `100,200,100`), capacity down (`25,50,25`, `60,30,30,30,30`) and depth varied
at fixed capacity (`320`, `93,93`), but no pure *reshuffle*.

### The result

| mean of 4 | control `50,100,50` | `100,50,50` | delta | seeds favouring | p (16 flips) |
|---|---|---|---|---|---|
| peak trailing | 94.44 | 94.16 | −0.28 | 2 of 4 | 0.750 |
| `sef` (primary) | 11.2% | 1.9% | −9.35 | 1 of 4 | 0.375 |
| best-30 | 64.0% | 51.2% | −12.80 | 1 of 4 | 0.375 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 46.3% | −8.66 | 1 of 4 | 0.375 |
| max drawdown | 5.42 | 6.35 | +0.93 | 1 of 4 | 0.625 |

**Null on the ceiling, as designed.** Peak trailing sits inside the flat 94.4-95.0 band on three seeds;
only `b20aj` (93.56) drops below it. Drawdown matches the control, so the base's anti-forgetting property
is intact — as it is for every shape at or above the control's capacity.

**The consolidation columns read lower, and it is one seed.** Paired pooled differences are
**+18.8 / −31.9 / −2.0 / −19.6**: `b20ak` is level, `b20ai` is well *ahead* (its control `b20a` being the
batch's weak arm — the seed-1 pattern), and the mean is carried by `b20aj`, which nearly failed to
consolidate at all (`sef` 0.1%, best-30 36.0). Its control `b20b` is the control's second-*best* arm, so
that single pair swings 31.9 pp. At 1 of 4 seeds and p=0.375 there is no effect here in either direction.

**The seed-1 pattern, priced across the whole batch:** control `b20a` sits at pooled 33.2, and **six of
the eight treatment shapes beat it on seed 1 by +18.8 to +34.5 pp** — every shape at or above ~0.9× the
control's capacity. The two exceptions are the sub-capacity shapes (`25,50,25` −3.0, `60,30,30,30,30`
−2.5), which cannot reach the level at all. So a seed-1 gap of +30 pp in this batch carries no information
about a shape; it is the size of the control's own bad draw.

### ‡ What this arm actually contributes: it prices the sweep's noise floor

`100,50,50` (10,853 params) and `320` (10,883 params) differ in capacity by **0.3%**, and they land on
**pooled 46.3 and 65.1** — an 18.8 pp spread that **brackets the control from both sides**:

| shape | params | pooled | vs control |
|---|---|---|---|
| `100,50,50` | 10,853 | 46.3% | **−8.7** |
| `50,100,50` (control) | 11,853 | 55.0% | — |
| `320` | 10,883 | 65.1% | **+10.1** |

Two nets of the same size disagree by more than any shape in the sweep disagrees with the control. So
**the consolidation columns across batch 20 are not measuring capacity** — they are measuring which seeds
an arm happened to draw. That is an independent confirmation of the ‡ downgrades already applied to
`320`'s +10.1 and `200,100,50`'s +9.6 by the per-seed test, and it is the reason the batch's conclusion
rests on peak trailing and drawdown rather than on pooled.

**No full-length rows** (deepest 25-28 of 100 under gate 95), so best-checkpoint is a bound and `pooled`
is the exact column. Nothing near 95%, no hall-of-fame candidate.

## Batch 20 wave 2 — wide-early `200,50`: **null on the ceiling, behind the control on the primary metric**

**Ran 2026-08-09 15:12 to 22:54, four arms `b20i`-`b20l` at seeds 1-4, all self-terminating at the 3M
cap after ~7h40m; close-out 22:54-23:19 (25 min, four parallel, `EVAL_WORKERS=4`).** Charts in
[`charts.md`](archive/charts-archive.md#batch-20-wave-2--wide-early-20050-against-the-control-both-to-3m). Launched and
closed out unattended by a watcher that waited for the cap, so no arm was stopped by hand.

### The design

`FC_LAYERS=200,50` — two layers, wide first, 16,403 params against the control's 11,853 (1.38×). One
variable against wave 1's control, which is seed-matched and ran to the same 3M under the same β=300k.

```
SNEK_FC_LAYERS=200,50  SNEK_SEED=1..4  SNEK_BETA_ANNEAL_STEPS=300000
SNEK_TARGET_UPDATE_PERIOD=1000  SNEK_FOOD_DISTANCE_REWARD=0  SNEK_DISCOUNT=0.9975
SNEK_GUIDED_FRACTION=0.8  SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5
SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60  SNEK_MAX_STEPS=3000000
```

**What it isolated:** whether *width early* buys anything, on the argument that the observation is
already high-level so the net needs conjunctions of engineered features rather than a deep hierarchy.

### The result

| mean of 4 | control `50,100,50` | `200,50` | delta |
|---|---|---|---|
| peak trailing | 94.44 | 94.55 | +0.11 |
| `sef` (primary) | 11.2% | **8.60%** | **−2.6** |
| best-30 | 64.0% | 67.4% | +3.4 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 59.3% | +4.3 |
| max drawdown | 5.41 | 8.56 | +3.15 |

**Null on the ceiling and negative on the primary metric.** Peak trailing moves 0.11 inside a band nine
batches have held, and `sef` — the metric with the lowest between-seed variance, so the one that resolves
the smallest effect — goes the *wrong* way.

**‡ The best-30 and pooled deltas are a single seed, and it is a control weakness rather than a shape
strength.** Paired per-seed differences, exact paired permutation over 16 sign flips:

| metric | by seed | mean | p | favouring `200,50` |
|---|---|---|---|---|
| `sef` | +8.5 / −6.3 / −0.3 / −12.4 | −2.62 | 0.625 | 1 of 4 |
| best-30 | +31.4 / −9.0 / −1.0 / −8.0 | +3.35 | 1.000 | 1 of 4 |
| pooled | +31.3 / −3.4 / −4.3 / −6.5 | +4.27 | 1.000 | 1 of 4 |

Seed 1 supplies every positive mean, and seed 1's control (`b20a`) is the weakest arm in the batch
(`sef` 0.2%, pooled 33.2%). **The control's own seed spread — `sef` 0.2-26.3%, pooled 33.2-71.3% — is
wider than any between-shape gap batch 20 has produced.** That is the finding worth carrying forward:
at n=4 this design cannot see an architecture effect smaller than its control's seed variance, and none
of the three shapes tried has produced one that large.

**No full-length rows again** (deepest 25-46 of 100 under gate 95), so every `best ckpt` in the canonical
rows is truncated and is a bound. Nothing reached 95%, so no hall-of-fame candidate.

Drawdown rose from 5.41 to 8.56 — still batch-19 territory (8.76) and nowhere near batch 18's ~57, so
the base's anti-forgetting property holds under this shape.

## Batch 20 wave 2 — depth-1 `320`: **depth contributes nothing at matched capacity**

**Ran 2026-08-09/10 on the desktop `the-claw-den`, four arms `b20m`-`b20p` at seeds 1-4, all to the 3M
cap; close-out ~32 min (four parallel, `EVAL_WORKERS=4`).** One operational wrinkle: the arms were
relaunched once at ~240k, resuming from their checkpoints, to adopt the crisp-chart env — no training was
lost. Charts in
[`charts.md`](archive/charts-archive.md#batch-20-wave-2--depth-1-320-against-the-control-both-to-3m).

### The design

`FC_LAYERS=320` — a single hidden layer, 10,883 params against the control's 11,853 (**0.92×**). It holds
capacity roughly constant and removes all depth (3 layers → 1), so it is the cleanest single architecture
finding available: **match the control and depth is contributing nothing here.** One variable against
wave 1's seed-matched control, same 3M, same β=300k.

```
SNEK_FC_LAYERS=320  SNEK_SEED=1..4  SNEK_BETA_ANNEAL_STEPS=300000
SNEK_TARGET_UPDATE_PERIOD=1000  SNEK_FOOD_DISTANCE_REWARD=0  SNEK_DISCOUNT=0.9975
SNEK_GUIDED_FRACTION=0.8  SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5
SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60  SNEK_MAX_STEPS=3000000
```

### The result

| mean of 4 | control `50,100,50` | `320` | delta |
|---|---|---|---|
| peak trailing | 94.44 | 94.67 | +0.23 |
| `sef` (primary) | 11.2% | 16.5% | +5.3 |
| best-30 | 64.1% | 74.7% | +10.6 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 65.1% | +10.1 |
| max drawdown | 5.41 | 7.44 | +2.0 |

**Depth-1 matches the control — the cleanest read in batch 20.** Peak trailing moves 0.23 inside the band
nine batches have held; all four seeds land in 94.44-94.86. Removing every hidden layer but one, at 92% of
the parameters, cost nothing on the ceiling. Unlike `200,50`, the consolidation columns all tick the
*right* way and 3 of 4 seeds favour the shape — but none of it is significant.

**‡ Every gap is inside the control's own seed spread, and the magnitude is one seed.** Paired per-seed
differences, exact paired permutation over 16 sign flips:

| metric | by seed (1/2/3/4) | mean | p | favouring `320` |
|---|---|---|---|---|
| peak trailing | +0.82 / +0.02 / +0.10 / +0.00 | +0.23 | 0.250 | 4 of 4 |
| `sef` | +12.7 / +6.9 / +1.7 / −0.1 | +5.30 | 0.250 | 3 of 4 |
| best-30 | +31.7 / +3.0 / +8.0 / −0.3 | +10.60 | 0.250 | 3 of 4 |
| pooled | +32.1 / +7.6 / −1.0 / +1.7 | +10.10 | 0.250 | 3 of 4 |

The consistency is real — 3-4 of 4 seeds favour the shape on every column, better than `200,50`'s 1 of 4 —
but **p bottoms out at 0.250** (n=4 cannot reach significance with one delta at ~0), and seed 1 supplies
the bulk of every mean. Seed 1's control (`b20a`) is the batch's weakest arm (`sef` 0.2%, pooled 33.2%),
so its +32 pp pooled gap is a control weakness; **excluding seed 1 the pooled edge is +2.8**. This is the
same lesson `200,50` taught: at n=4 the design cannot resolve an architecture effect smaller than the
control's own seed variance (pooled 33.2-71.3%), and none of the three shapes has produced one that large.

**No full-length rows** (deepest 26-56 of 100 under gate 95), so every `best ckpt` is a truncated bound.
Nothing reached 95%, so no hall-of-fame candidate. Drawdown rose to 7.44 — one seed (`b20o`, 11.96) drives
it, still batch-19 territory and far from batch 18's ~57, so the base's anti-forgetting property holds.

## Batch 20 wave 3 — small-capacity `25,50,25`: **capacity finally binds — the ceiling drops**

**Ran 2026-08-10 on the desktop `the-claw-den`, four arms `b20q`-`b20t` at seeds 1-4, all to the 3M cap;
close-out under gate 95, `EVAL_WORKERS=4`.** Charts in
[`charts.md`](archive/charts-archive.md#batch-20-wave-3--small-capacity-255025-against-the-control-both-to-3m).

### The design

`FC_LAYERS=25,50,25` — the control's shape scaled down to **3,428 params, 0.29× the control**, depth
unchanged at 3. This arm asks the direct question the earlier shapes only implied: *is capacity binding at
all?* If under a third of the parameters holds the ceiling, it is not — and after `200,100,50` (2.66× did
not help) and `320` (depth-1 matched), that was the way the evidence pointed. One variable against the
seed-matched control, same 3M, same β=300k.

### The result

| mean of 4 | control `50,100,50` | `25,50,25` | delta |
|---|---|---|---|
| peak trailing | 94.44 | **93.75** | **−0.69** |
| `sef` (primary) | 11.2% | 2.1% | −9.1 |
| best-30 | 64.0% | 52.8% | −11.2 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 43.1% | −11.9 |
| max drawdown | 5.41 | 11.13 | +5.7 |

**This is the first shape in batch 20 to move the ceiling, and it moves it down.** Peak trailing drops 0.69
— outside the flat 94.4-94.9 band every batch since 11 has held — and two of the four seeds (`b20q` 93.08,
`b20s` 93.36) sit clearly below it. Every consolidation column falls with it and drawdown roughly doubles.
**All four seeds favour the control on every column.**

**‡ The cleanest directional result batch 20 has produced.** Exact paired permutation over 16 sign flips:

| metric | by seed (1/2/3/4) | mean | p | favouring `25,50,25` |
|---|---|---|---|---|
| peak trailing | −0.72 / −0.36 / −0.98 / −0.70 | −0.69 | 0.125 | 0 of 4 |
| `sef` | −0.2 / −9.9 / −2.0 / −24.3 | −9.10 | 0.125 | 0 of 4 |
| best-30 | 0.0 / −12.0 / −9.6 / −23.3 | −11.23 | 0.250 | 0 of 4 |
| pooled | −3.0 / −5.2 / −13.3 / −26.1 | −11.89 | 0.125 | 0 of 4 |
| drawdown | +8.6 / +6.6 / +6.1 / +1.5 | +5.71 | 0.125 | 0 of 4 (all worse) |

p is at the n=4 floor (0.125) on four of five metrics — the strongest this design can register — and every
seed points the same way. Where `320` and `200,50` produced seed-driven noise straddling zero, `25,50,25`
is uniform and directional. **At 0.29× the capacity is binding: the net can no longer reach the ceiling the
control reaches, and it consolidates and holds worse while trying.**

**No full-length rows** (deepest 25-29 of 100 under gate 95), so best-checkpoint is a bound; `pooled` is
exact. Nothing near 95%, no hall-of-fame candidate. Drawdown mean 11.13 is the batch's worst but still
batch-19 territory, far from batch 18's ~57 — the base's anti-forgetting property is dented, not broken.

## Batch 20 wave 3 — deep-narrow `60,30,30,30,30`: **matches the ceiling, forgets more**

**Trained on the laptop, seeds 1-4 `b20u`-`b20x` (renamed from a `q-t` collision with the desktop's
`25,50,25`), all to the 3M cap; checkpoints rsynced to the desktop and closed out there under gate 95,
`EVAL_WORKERS=4`.** Charts in
[`charts.md`](archive/charts-archive.md#batch-20-wave-3--deep-narrow-6030303030-against-the-control-both-to-3m).

### The design

`FC_LAYERS=60,30,30,30,30` — **6,573 params, 0.55× the control**, but depth **5**, the deepest shape in the
sweep. It tests depth from the other side of `320`: where `320` removed all depth at matched capacity, this
stacks five narrow layers below capacity. The final layer was widened from a proposed `,10` to `,30` so the
three Q-values are not forced through 10 ReLU units where dead units would cost the representation
permanently.

### The result

| mean of 4 | control `50,100,50` | `60,30,30,30,30` | delta |
|---|---|---|---|
| peak trailing | 94.44 | 94.25 | −0.18 |
| `sef` (primary) | 11.2% | 4.7% | −6.5 |
| best-30 | 64.0% | 59.3% | −4.7 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 51.6% | −3.4 |
| max drawdown | 5.41 | 12.31 | +6.9 |

**It matches the control on the ceiling and forgets about twice as much.** Peak trailing 94.25 is inside
the band (−0.18, p 0.375), and pooled/`sef`/best-30 are all within the control's own seed spread — one seed
(3) favours the shape, the rest do not. The one column that separates cleanly is drawdown: **worse on all
four seeds, +6.9, p 0.125.**

**‡ Only drawdown clears noise.** Exact paired permutation over 16 sign flips:

| metric | by seed (1/2/3/4) | mean | p | favouring the shape |
|---|---|---|---|---|
| peak trailing | −0.20 / −0.44 / +0.32 / −0.40 | −0.18 | 0.375 | 1 of 4 |
| `sef` | −0.1 / −14.0 / +9.7 / −21.7 | −6.53 | 0.375 | 1 of 4 |
| best-30 | −0.6 / −18.0 / +17.0 / −17.3 | −4.73 | 0.375 | 1 of 4 |
| pooled | −2.5 / −8.9 / +9.9 / −12.1 | −3.41 | 0.500 | 1 of 4 |
| drawdown | +5.1 / +9.3 / +5.8 / +7.4 | +6.89 | 0.125 | 0 of 4 (all worse) |

At 0.55× and depth 5 the ceiling still holds — so the knee where capacity starts to bind is between this
and `25,50,25`'s 0.29×, not at the control. Depth 5 buys nothing and costs steadiness: the same
higher-drawdown signature `25,50,25` shows, so it tracks narrowness/depth rather than capacity alone.

**No full-length rows** (deepest 25-36 of 100), best-checkpoint a bound, nothing near 95%.

**What batch 20 now shows across five shapes.** At or above the control's capacity — 2.66×
(`200,100,50`), 1.38× wide-early (`200,50`), depth-1 (`320`, 0.92×) — the ceiling does not move, and depth
1 vs 3 makes no difference. Going below: at 0.55× and depth 5 (`60,30,30,30,30`) the ceiling still holds; at
0.29× (`25,50,25`) it finally drops, 4/4 seeds, p 0.125. **So capacity is not the binding constraint
anywhere at or above the control — the control sits comfortably above the knee, which lies between 0.29×
and 0.55× — and architecture (width, depth, shape) does not raise the ceiling within the range that
preserves it.** Two narrow shapes also consolidate worse (drawdown 11-12 vs 5.4, 4/4 seeds each), so
shrinking or deepening the net costs steadiness before it costs the ceiling. The ceiling nine batches of
optimiser knobs could not move is not an approximation-capacity limit.

## Batch 20 — capacity escalation `100,200,100` (3.69×): **null on the ceiling, the seed-1 pattern again**

**Ran on the desktop `the-claw-den`, four arms `b20ae`-`b20ah` at seeds 1-4, all to the 3M cap; close-out
under gate 95, `EVAL_WORKERS=4`.** Charts in [`charts.md`](charts.md) (100,200,100 section).

### The design

`FC_LAYERS=100,200,100` — 43,703 params, **3.69× the control** at depth 3, the largest net in the sweep.
It escalates above the 2.66× (`200,100,50`) arm to ask whether *any* amount of extra capacity moves the
ceiling, given 2.66× did not.

### The result

| mean of 4 | control `50,100,50` | `100,200,100` | delta | p (16 flips) | favour shape |
|---|---|---|---|---|---|
| peak trailing | 94.44 | 94.61 | +0.17 | 0.875 | 2/4 |
| `sef` (primary) | 11.2% | 16.6% | +5.4 | 0.625 | 2/4 |
| best-30 | 64.0% | 71.3% | +7.3 | 0.750 | 2/4 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 63.4% | +8.4 | 0.625 | 3/4 |

**Null on the ceiling — peak +0.17 stays inside the flat 94.4-94.9 band.** The consolidation means rise,
but the rise is one seed: `b20ae` (+36 best-30, +35 pooled) against control arm `b20a`, the batch's weak
seed (33.2 pooled). The other three seeds are flat-to-mixed (best-30 −9 / −1 / +3) and the p-values sit
far from the 0.125 floor. **3.69× behaves like the control**, the same reading `200,100,50` (2.66×) gave.
This closes the capacity question: across 0.92× / 1.00× / 1.38× / 2.66× / 3.69× nothing moved the ceiling
— only cutting to 0.29× (`25,50,25`) did, downward.

**No full-length rows** (deepest 25-73 of 100 under gate 95), so best-checkpoint is a bound and `pooled`
is exact. Nothing at the 95 gate; no hall-of-fame candidate.

## Batch 20 wave 3 — iso-param depth-2 `93,93`: **null — matches the control on every column**

**Ran on the desktop `the-claw-den`, four arms `b20aa`-`b20ad` at seeds 1-4, all to the 3M cap; close-out
under gate 95, `EVAL_WORKERS=4`.** Charts in
[`charts.md`](archive/charts-archive.md#batch-20-wave-3--iso-param-depth-2-9393-against-the-control-both-to-3m).

### The design

`FC_LAYERS=93,93` — two hidden layers of 93, **11,907 params, 1.00× the control**, at depth 2 against the
control's depth 3. It fills the iso-param depth rung between the control (depth 3) and `320` (depth 1):
hold capacity fixed, cut one layer — does depth matter? After `320` matched the control, the prediction
was null.

### The result

| mean of 4 | control `50,100,50` | `93,93` | delta | p (16 flips) |
|---|---|---|---|---|
| peak trailing | 94.44 | 94.41 | −0.03 | 1.000 |
| `sef` (primary) | 11.2% | 6.7% | −4.5 | 0.375 |
| best-30 | 64.0% | 61.4% | −2.6 | 0.875 |
| close-out pooled (eq-effort, gate 95) | 55.0% | 51.5% | −3.5 | 0.875 |

**A null — the ceiling does not move (peak −0.03) and no metric separates from the control at n=4.** Only
1 of 4 seeds favours `93,93` on each column, but the means are small and the p-values sit far from the
0.125 floor, so the direction is noise, not a deficit. The one large swing is seed 1 (`b20aa` +33 best-30)
and it is a **control weakness**: `b20a` is the batch's weak arm (33.2 pooled), the same seed-1 pattern
every b20 shape shows. **Depth 2 at matched capacity behaves like the control**, exactly as `320` (depth 1)
did — depth is not the lever, and only `25,50,25`'s 0.29× capacity cut has moved the ceiling.

**No full-length rows** (deepest 25-38 of 100 under gate 95), so best-checkpoint is a bound and `pooled` is
exact. Nothing near 95%, no hall-of-fame candidate.

## Batch 19 — standard PER: **falsified, 4/4 seeds, on every comparable metric**

**Ran 2026-08-08 15:54 to 23:19, four arms `b19a`-`b19d` at seeds 1-4, stopped by hand at 2.00-2.42M
after 7h25m.** Charts in
[`charts.md`](archive/charts-archive.md#batch-19--standard-per-td_error-priority--is-on-falsified-stopped-at-200-242m).
**No close-out evals were run**, at the user's direction — the training comparison is decisive on its
own and the arms are not champion candidates.

### The design, and why batch 18 is a clean control

**One knob-group changed:** the priority signal went `td_loss` → **`td_error`** (so priority is
`|TD error|`) and importance sampling turned **on**, β annealing 0.4 → 1.0 over 1M steps. Huber stayed
the network loss (`element_wise_huber_loss`, unchanged). Everything else is batch 18 byte-for-byte, so
**batch 18 is the seed-matched control**. No code edit was needed — `td_error`, `IS_WEIGHTS=1` and
`IS_BETA=0.4` are all `snek2.py` defaults, so the launch just *dropped* batch 18's two PER overrides,
confirmed by the startup logs showing no `PRIORITY_SIGNAL`/`IS_WEIGHTS` override on any arm.

```
SNEK_SEED=1..4  SNEK_TARGET_UPDATE_PERIOD=1000
SNEK_FOOD_DISTANCE_REWARD=0  SNEK_DISCOUNT=0.9975  SNEK_GUIDED_FRACTION=0.8
SNEK_FORK_BRANCHES=4  SNEK_FORK_PROB=0.5  SNEK_FORK_MIN_LENGTH=85  SNEK_FORK_MAX_STEPS=60
# PER left at snek2.py defaults: td_error priority, IS on, beta 0.4 -> 1.0 over 1M
```

**What it isolated:** whether standard proportional PER (`|δ|` priority + full IS correction) beats the
effective-α≈1.2 `td_loss`/no-IS config the project has run since batch 5. LR was kept at the default
1e-5 because IS weights are mean-normalised, so the tuned LR is preserved. **β anneal:** these arms ran
the **1M** schedule, the default at launch; the default was changed to 300k later the same day, which
does not affect them — they were fully annealed by ~1M.

### The pre-registered comparison, at a matched 2.004M horizon

`b19c` is the shortest arm at 2.004M, so both batches truncate there. Exact paired permutation over all
16 sign flips.

| metric | b18 (`td_loss`, no IS) | b19 (standard PER) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **31.60%** | **13.82%** | **-17.78 pp** | **0.125** (4/4) |
| `best_perfect30` | 85.52% | 63.27% | **-22.25 pp** | **0.125** (4/4) |
| mean perfect, back half | 68.86% | 48.88% | **-19.98 pp** | **0.125** (4/4) |
| peak trailing | 94.85 | 94.16 | **-0.69** | **0.125** (4/4) |
| **max drawdown** | 55.52 | **8.76** | **-46.76** | **0.125** (4/4) |
| steps to pf30 ≥ 40% | 299.5k | 324.7k (3 arms) | slower 3/3 | — |

0.125 is the **floor** at n=4, so every seed moved the same way on all five pooled metrics.
**`b19c` never reached pf30 ≥ 40%**, so that row has no fourth pair and is left unpooled rather than
imputed; the three seeds that did reach it were all slower.

| seed | step | peak trailing | best-30 | `sef` | recent-30 | max drawdown |
|---|---|---|---|---|---|---|
| 4 | 2423k | **94.86** | **85.7%** | **40.2%** | **56.0%** | 12.84 |
| 1 | 2192k | 94.66 | 71.0% | 8.0% | 43.3% | **4.94** |
| 2 | 2116k | 94.40 | 66.7% | 4.6% | 51.3% | 10.20 |
| 3 | 2004k | 92.72 | 29.7% | **0.0%** | 14.3% | 7.04 |
| **mean** | | **94.16** | **63.3%** | **13.2%** | **41.2%** | **8.76** |

### What it establishes

**The pre-registered "clearly worse" branch fired, so this reproduces `b5c` cleanly for the first
time and closes the partial-IS-correction candidate.** `b5c` had paired IS with `td_loss` + alpha 0.8
*and* a fast anneal, so it never isolated IS; this batch does, and the long-standing `td_loss`/no-IS
default now has a real defence rather than an inherited one.

**‡ The ceiling moved for the first time in nine batches — downward.** Peak trailing read 94.8-95.0
for every batch from 11 through 18 regardless of config. Here it is 94.16 mean, lower on 4/4. The
magnitude is small but the invariance breaking at all is the notable part.

**‡ The drawdown result is the real finding hiding in a negative batch.** Max drawdown fell 55.52 →
**8.76**, 4/4 — the largest movement in the table and the strongest anti-forgetting result on record.
It is not one bad control seed: three of four batch-18 arms drop 24-85 points, `b18b` collapsing to
13.9 trailing over 1714k-2283k before recovering to 86.8. **But the arms buy it by sitting lower, not
by holding a high level** — `sef` more than halved. Since reducing catastrophic forgetting is a *means*
here and not the goal, that does not pay for −17.78 pp. **It does make the β anneal a candidate to pair
with any future change that raises the level**, which is the one thing worth carrying forward.

**Seed 4 is the exception and bounds the claim.** `b19d` is level with its control on every column, so
the finding is a distribution shifting left rather than a mechanism that cannot work. One seed in four
paid nothing.

**What this is not:** a verdict on `td_error` alone or IS alone. The two moved together, so a config
wanting `|δ|` priority *without* the IS correction is untested — and given the drawdown result, that is
the version of the experiment with a live hypothesis behind it.

**‡ Amended 2026-08-10: the label "standard PER" overstates what ran.** Measuring the saved buffers
found that the expected update is proportional to `raw^(α(1−β))`, so **at β=1.0 prioritization cancels
exactly** — realised ESS/N 0.951 against a 0.975 uniform noise floor. These arms were fully annealed
by ~1M and every one of them peaked after that, so past the anneal this batch compared batch 18's
effective-α≈1.2 buffer against **uniform replay**, not against a milder prioritization. The result
above stands and got stronger — batch 20 wave 1 reproduces it on four fresh seeds at a matched 2.401M
(`sef` −21.39 pp, 4/4) — but read it as *uniform replay is worse here, 8 seeds*. Full measurement in
[`findings.md`](findings.md#-measured-batches-19-20-compared-aggressive-per-against-uniform-replay).

## Batch 18 — `TARGET_UPDATE_PERIOD` 1000: the strongest speed result, and 20 rows ≥95%

**Ran 2026-08-08 00:50 to 09:04, four arms stopped by hand at 2.40-2.61M; close-out evals finished
2026-08-08 17:50.** Design, the paired training comparison against batch 17 and the per-seed table are
in [`charts.md`](archive/charts-archive.md#batch-18--target_update_period1000-forking-retained-stopped-at-240-261m)
and [`archive/runs-archive.md`](archive/runs-archive.md#closed-batch-18--target_update_period-1000-forking-retained);
only the close-out is new here. Gate **95**, so `measured` is `pooled_equal_effort`.

| arm | rows | episodes | eq-effort | 100-ep rows | ≥90% | ≥95% | top-3 | best full row |
|---|---|---|---|---|---|---|---|---|
| `b18a` | 599 | 21,101 | **81.87%** | 6 | 6 | 5 | 95.3% | 96.0% @1289k |
| `b18b` | 310 | 11,429 | 78.52% | **9** | **9** | **9** | **97.0%** | **98.0% @1588k** |
| `b18c` | 275 | 8,245 | 74.75% | **0** | 0 | 0 | — | none — best is 92.3% /91ep @2168k |
| `b18d` | 693 | 21,584 | 80.22% | 6 | 6 | 6 | 96.0% | 96.0% @1105k |

**Two things stand out, and both held up.**

### ‡ `b18b` @1588000 is the project record: 97.6% over 700 fresh episodes

**Verified 2026-08-09.** The close-out row read 98.0%/100 and was written up here as a candidate that
"must not be called a record", on the grounds that every selected high in this project had shrunk —
`b17b` 99/100 → 94.2% over 5,120, `b15b` 97/100 → 93.0%, `b11b` 96/100 → ~94%. **That expectation was
wrong for this checkpoint**, and it is the first time:

| measurement | result |
|---|---|
| 500 fresh episodes, gate off, flat protocol | 487/500 = **97.4%** (95.6-98.5) |
| 200 fresh, on the hall-of-fame copy | 196/200 = **98.0%** (95.0-99.2) |
| **pooled** | **683/700 = 97.57%** (CI **96.1-98.5**) |

**Against `b17b`'s 94.24%/5,120: +3.33 pp, z=3.67, p=0.0002, intervals non-overlapping.** Promoted to
[`../hallOfFame/`](../hallOfFame/README.md#-b18b-1588000--the-first-selected-high-that-did-not-shrink)
with the copy verified to load and play.

**All nine of the batch's >95% checkpoints were re-measured at 500 episodes each**, which is what makes
@1588000 readable as an outlier rather than a loose protocol — the other eight shrank by 2.0 to 10.6 pp:

| checkpoint | selected | 500 ep | delta |
|---|---|---|---|
| `b18b` @1588000 | 98.0 | **97.4** | **−0.6** |
| `b18b` @1601000 | 97.0 | 95.0 | −2.0 |
| `b18b` @1600000 | 96.0 | 92.8 | −3.2 |
| `b18b` @561000 | 96.0 | 92.2 | −3.8 |
| `b18b` @1578000 | 96.0 | 91.6 | −4.4 |
| `b18d` @1111000 | 96.0 | 90.4 | −5.6 |
| `b18d` @1105000 | 96.0 | 89.6 | −6.4 |
| `b18d` @2292000 | 96.0 | 85.6 | −10.4 |
| `b18a` @1289000 | 96.0 | 85.4 | −10.6 |

Mean shrinkage **−5.2 pp**; pooled over all nine, 4100/4500 = **91.11%**.

**‡ It is a narrow peak, not a strong region.** `@1578000` sits 10k steps earlier and reads 91.6%, and
`@1600000` 12k later reads 92.8%. So `TARGET_UPDATE_PERIOD=1000` is not demonstrated to produce a
better *region* — only this checkpoint. A blind every-10k grid over 1.55-1.62M is the outstanding test,
the same one that deflated `b17b`'s apparent region to 84%.

**The consistency across seeds is the real result: eq-effort 74.75-81.87, a 7.1-point spread.** Batch
17 spanned 46.25-82.42 and batch 14 67.1-77.6. Batch 18's mean of **78.84%** sits just under `b17b`'s
single 82.42% while every arm clears 74% — the first batch where the weakest seed is not a write-off.
`pooled_equal_effort` is the one column selection cannot inflate, so this comparison is safe.

**`b18c` produced no full-length row at all**, so its 92.3% comes from a 91-episode abandoned row and
reads noisier than the rest. It is still the batch's weakest arm on every other column, so the gate is
reporting a real difference rather than hiding one.

## Batch 17 — forked endgame collection: **a null that produced the project record**

**Ran 2026-08-07 19:55 to 2026-08-08 00:45, four arms stopped by hand at 1.41-1.57M, measured
2026-08-08.** `SNEK_FORK_BRANCHES=4 SNEK_FORK_PROB=0.5 SNEK_FORK_MIN_LENGTH=85
SNEK_FORK_MAX_STEPS=60` on batch 16's config exactly — same four seeds, same discount 0.9975, same
`GUIDED_FRACTION=0.8`, shaping off in both, `N_STEP_UPDATE=1`. **Batch 16 is the seed-matched control
and differs only by `FORK_BRANCHES=1`.** The first change in this project aimed at the *collect
distribution* rather than the optimiser or the reward. Graphs in
[`charts.md`](archive/charts-archive.md#batch-17--forked-endgame-collection-snek_fork_-a-null-stopped-at-141-157m).

| seed | step | peak trailing | best-30 | `sef` | best ckpt | eq-effort |
|---|---|---|---|---|---|---|
| 1 | 1245k | 93.86 @1144k | 54.0% @1166k | **1.3%** | 68.2% @916k †22ep | **46.25%** |
| 2 | 1245k | 94.92 @1068k | **92.7%** @1223k | **32.0%** | **99.0%** @1205k **100ep** | **82.42%** |
| 3 | 1245k | 94.38 @1230k | 71.3% @954k | 9.9% | 93.5% @1424k †92ep | 73.33% |
| 4 | 1245k | 94.60 @1007k | 75.7% @679k | 18.6% | 86.0% @1260k †50ep | 66.83% |
| **mean** | | **94.44** | **73.4%** | **15.4%** | | **67.21%** |

**Closed out 2026-08-08 at `EVAL_MIN_ACHIEVABLE=95`, matching batch 16's protocol exactly.** An
earlier launch at 90 was stopped after ~4 minutes with no completed rows and restarted at 95: when the
whole point is a paired comparison against batch 16, running the *same* protocol as the control beats
running a better one. † marks a truncated row; `b17b`'s is the only full 100 in the batch, and there
are 26 of them.

### The pre-registered comparison, at a matched 1.245M horizon

`b16a` stopped at 1245k, so that is the common horizon — 1246 evals per arm on both sides. Exact
paired permutation over all 16 sign flips.

| metric | b16 (fork off) | b17 (fork on) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **17.11%** | **15.45%** | **-1.67 pp** | 0.875 |
| `best_perfect30` | 79.42% | 73.42% | -6.00 pp | 0.875 |
| mean perfect, back half | 62.22% | 56.37% | -5.85 pp | 0.750 |
| mean perfect | 42.80% | 42.07% | -0.73 pp | 1.000 |
| peak trailing | 94.71 | 94.44 | -0.27 | 0.250 |
| mean trailing | 85.13 | 80.16 | -4.97 | 0.250 |
| max drawdown | 66.92 pp | 73.97 pp | +7.04 pp | 0.500 |
| steps to pf30 ≥ 40% | 424k | 402k | **-21.8k** | 0.750 |

**The primary metric points the wrong way and nothing is significant.** The only direction worth
anything is speed — 22k earlier to pf30 ≥ 40%, 2 of 4 seeds — which is the metric with the largest
per-seed spread here (+110k, -68k, +7k, -136k).

### ‡ Unlike batch 16, the sweep is flat, so this is not a horizon artifact either way

| horizon | `sef` delta | `best30` delta |
|---|---|---|
| 300k | +0.00 pp | +4.67 pp |
| 500k | +2.30 pp | +4.17 pp |
| 700k | +3.67 pp | +6.83 pp |
| 900k | +0.97 pp | -6.00 pp |
| 1.0M | -1.15 pp | -9.75 pp |
| **1.245M** | **-1.67 pp** | **-6.00 pp** |

Positive and rising to ~700k, then negative and stable. Batch 16's signal grew monotonically to its
horizon; this one reverses, which is what a noise effect looks like. **The 900k interim read called it
a null for the same reason it later stayed a null, by luck** — the 500-700k slice would have called it
a win.

### ‡ One seed is the whole result, and that has to be said in both directions

| metric | all four seeds | dropping `b17a` |
|---|---|---|
| `sef` | **-1.67 pp** | **+4.23 pp** |
| `best_perfect30` | -6.00 pp | +3.03 pp |
| mean perfect, back half | -5.85 pp | +4.07 pp |

`b17a-forkseed1` is -19.3 pp on `sef` and -33.0 on best-30; the other three are +1.7 / -0.6 / +11.6.
**This is not a licence to drop it** — it is the honest statement that n=4 cannot separate "forking
destabilises some seeds" from "seed 1 drew badly." Two facts argue for the second: `b13a` failed the
same way with no forking at all, and `b17a`'s fork counters are entirely normal (9683 forks, 27%
share, retired == created). It is also the only arm of the eight that **never reached ε ≤ 0.003**,
where all seven others did between 373k and 974k — its schedule is gated on a trailing-30 perfect
rate it never sustained, so the instability and the high epsilon feed each other. That is a
[failure mode](failureModes.md), not a forking mechanism.

### The mechanism engaged and held for 1.5M steps — this is a real null, not a null test

| arm | forks | branch share | truncated / ended | eligible skipped, slots full |
|---|---|---|---|---|
| `b17a` | 9,683 | 27% | 5,725 / 3,958 | 11,670 |
| `b17b` | 8,614 | 24% | 5,730 / 2,881 | 12,228 |
| `b17c` | 10,268 | 29% | 6,537 / 3,731 | 13,772 |
| `b17d` | 10,544 | 29% | 6,691 / 3,853 | 12,921 |

Inside the pre-registered 25-60% band, `forks == retired` on all four (no leaked branch envs, RSS flat
at 496-659 MB), `main_steps + branch_steps == global_step` exactly, and 0 violations of the buffer
integrity invariant.

**But the delivered dose was below design, and that is the one loose end.** Branch share landed at
**24-29% against the predicted ~46%**, and 12-14k eligible fork points per arm were skipped for want
of a slot — roughly **30% of all eligible points**. The **cap of 4 is what binds, not `FORK_PROB=0.5`**.
So the finding is "counterfactual coverage at endgame decision points does not help *at this dose*",
and `SNEK_FORK_BRANCHES=6-8` is the version of the experiment that would actually test the hypothesis.
The batch nonetheless landed inside its own pre-registered band, so by its own terms it is a fair null.

### ‡ The close-out: `b17b` is the best policy this project has measured, by a clear margin

| arm | rows | episodes | saved | eq-effort | deepest | rows ≥90% | rows ≥95% | best row |
|---|---|---|---|---|---|---|---|---|
| `b17a` | 20 | 440 | 1,560 | **46.25%** | **22 ep** | 0 | 0 | 68.2% @916k |
| `b17b` | **306** | **13,078** | 10,642 | **82.42%** | 100 ep (**26 of them**) | **69** | **25** | **99.0% @1205k** |
| `b17c` | 117 | 3,662 | 8,038 | 73.33% | 92 ep | 6 | 0 | 93.5% @1424k |
| `b17d` | 131 | 3,096 | 9,524 | 66.83% | 50 ep | 0 | 0 | 86.0% @1260k |

**`pooled_equal_effort` 82.42% is the highest ever recorded here**, against `b15b`'s 79.7%, and it is
the one column that cannot be inflated by selection — it truncates every checkpoint to the 20-episode
screen depth, so it is exact at any gate.

### ‡ Re-measured the same day over 6,600 fresh episodes: the 99/100 was mostly selection

The close-out produced 26 rows at the full 100 episodes, **25 of them ≥95%**, mean 96.2% — which looked
like a ~96% region rather than a lucky max. **It was not.** Four processes, `EVAL_MIN_ACHIEVABLE=0`,
no screening:

| step | close-out | re-measured /500 | change |
|---|---|---|---|
| 1190k | 98/100 | **95.0%** (92.7-96.6) | -3.0 |
| 1205k | **99/100** | **92.4%** (89.7-94.4) | **-6.6** |
| 1231k | 98/100 | 92.6% (90.0-94.6) | -5.4 |
| 1248k | **99/100** | 93.8% (91.3-95.6) | -5.2 |
| **mean of the ≥98% group** | **98.5%** | **93.45%** | **-5.05** |
| mean of the six 97% rows, /200 | 97.0% | **95.25%** | -1.75 |

**The ranking inside the top rows was noise**: the ≥98% group re-measures *below* the 97% group. And a
**blind grid** — 17 checkpoints every 10k across 1110-1270k, chosen by position, 100 fresh episodes
each — prices the selection effect directly:

| sample of the same region | mean | pooled | CI |
|---|---|---|---|
| position-chosen grid | **84.06%** | 1429/1700 | 82.2-85.7 |
| the close-out's selected full-length rows | **96.2%** | — | — |
| `b16b` control grid, 840-1000k | **80.41%** | 1367/1700 | 78.5-82.2 |

**Three results, in order of how much they matter:**

1. **The reasoning that defended the 99 was circular** — the 26 deep rows reached full depth *because*
   they screened well, so their mean is inflated by the same mechanism as the max. Full account in
   [`archive/runs-archive.md`](archive/runs-archive.md#-the-reasoning-error-a-selected-sample-cannot-defend-itself-against-selection).
2. **The record is real but small.** `b17b` @1190k pools to **95.17% over 600 fresh episodes** (CI
   93.1-96.6) against `b14a`'s 93.5%/200 and `b15b`'s 93.0%/300 — better, overlapping intervals, and
   reached at **1.19M steps** against 3.7M and 3.2M. **The speed is the unambiguous part.**
3. **The region beats its control by +3.65 pp** (84.06% vs 80.41%) on position-chosen samples — which
   independently reproduces this arm's `pooled_equal_effort` delta of **+3.34 pp**. Two different
   protocols, same answer, and it is the same ~+3.5 pp that seeds 3 and 4 showed.

**And the region is not a plateau.** `1140k` reads **12.0%** on the blind grid — a complete collapse
30k steps from a 95% checkpoint, invisible to the close-out because it never screened well enough to be
selected. Excluding it the grid mean is 88.56%, and quoting that instead would be the same selection
error one level up, so the 84.06% stands as the region figure.

**And `b17a`'s close-out is nearly worthless, which is worth recording as a protocol result**: 20 rows,
deepest 22 episodes, 440 episodes total. The gate is arithmetic and correct — that arm has no
checkpoint that could reach 95% — but it means the batch's mean eq-effort (67.21%) is one real
measurement and three shallow ones plus one near-empty one.

### The paired close-out comparison, and why it says the opposite of the mean

| seed | b16 eq-effort | b17 eq-effort | delta |
|---|---|---|---|
| 1 | 76.89% | 46.25% | **-30.64** |
| 2 | 79.08% | **82.42%** | **+3.34** |
| 3 | 69.67% | 73.33% | **+3.66** |
| 4 | 63.27% | 66.83% | **+3.56** |
| **mean** | **72.23%** | **67.21%** | **-5.02** (p=1.000) |

**Seeds 2, 3 and 4 move the same way by nearly the same amount** — +3.34, +3.66, +3.56, a spread of
0.32 pp on a metric whose between-seed sd is several points. That is a more consistent signal than any
graph metric in this batch produced. And seed 1 reads -30.64, which swamps it.

**Do not resolve this by dropping seed 1.** Two readings survive the data: forking gives ~+3.5 pp and
seed 1 failed for unrelated reasons, or forking destabilises some fraction of seeds and seed 1 is that
fraction. **n=4 cannot separate them**, and that — not the sign of the mean — is the batch's real
finding. The re-run that could separate them is `SNEK_FORK_BRANCHES=6-8` at n=8.

### What it establishes, and what it does not

- **Do not adopt forking on this evidence.** The primary metric is negative at p=0.875.
- **Do not call the hypothesis falsified either.** At 30% of eligible points skipped, the dose is
  ~60% of design, and three of four seeds moved the right way.
- **`b17b-forkseed2` is the most interesting single arm this project has produced on the graph**:
  `best_perfect30` **92.7%**, the highest of any arm ever run here, at **1.22M steps** against
  `b11b`'s 91.7% at 3.56M and `b15a`'s 89.7% at 5.79M — and its full-length peak trailing reads
  **95.00**, tying `b15a` for the highest on record. `best_perfect30` is a max statistic with sd 8.6
  across identical seeds, which is exactly why this file demoted it, so **one arm is a curiosity and
  not evidence**. Whether it holds a champion checkpoint is the close-out's job.
- **The ceiling still has not moved.** Peak trailing 94.44 against 94.71 / 95.00 / 94.90 / 94.80 for
  batches 16 / 15 / 14 / 13 — seven batches, everything inside 0.6 points.

## Batch 16 — the food-distance shaping ablated: **the first non-null in six batches**

**Ran 2026-08-07 09:5x to ~19:00, four arms stopped by hand at ~1.25M, measured 2026-08-07.**
`SNEK_FOOD_DISTANCE_REWARD=0` on batch 14's config exactly — same four seeds, same discount 0.9975,
same `GUIDED_FRACTION=0.8`, same `td_loss` + `IS_WEIGHTS=0`, `N_STEP_UPDATE=1` in both. **One live
variable and no reliance on a previous null**, which no batch since 13 has managed. Graphs in
[`charts.md`](archive/charts-archive.md#batch-16--food_distance_reward0-the-shaping-term-ablated-stopped-at-125m).

| seed | step | peak trailing | best-30 | `sef` | best ckpt | eq-effort |
|---|---|---|---|---|---|---|
| 1 | 1245k | 94.82 @837k | 87.0% @850k | 20.6% | 93.3% @838k †90ep | 76.89% |
| 2 | 1261k | **94.98** @816k | 85.0% @919k | **30.5%** | 93.3% @913k †90ep | **79.08%** |
| 3 | 1257k | 94.36 @1198k | 72.7% @1221k | 10.5% | 85.0% @979k †40ep | 69.67% |
| 4 | 1256k | 94.68 @946k | 73.0% @1032k | 7.0% | 80.0% @1010k †30ep | 63.27% |
| **mean** | | **94.71** | **79.4%** | **17.2%** | | **72.23%** |

**† Every "best ckpt" here is a truncated measurement, not a full 100 episodes** — see the gate
section below, which is a protocol result in its own right.

### The paired comparison, at a matched 1.25M horizon

Batch 16 stopped at ~1.25M while batch 14 ran to 4.2M, and `strong_eval_fraction` is a fraction of an
arm's *own* evals — so **every figure below truncates both batches to steps ≤ 1.25M**, giving 1251
evals per arm on both sides. Exact paired permutation over all 16 sign flips.

| metric | b14 (shaping on) | b16 (shaping off) | delta | p |
|---|---|---|---|---|
| **`strong_eval_fraction`** (primary) | **5.80%** | **17.15%** | **+11.35 pp** | 0.250 |
| `best_perfect30` | 66.83% | 79.42% | **+12.58 pp** | **0.125** (4/4 seeds) |
| mean perfect, back half | 51.15% | 62.27% | +11.13 pp | 0.250 |
| mean perfect | 36.52% | 42.87% | +6.35 pp | 0.250 |
| peak trailing | 94.31 | 94.71 | +0.41 | 0.250 |
| mean trailing | 85.96 | 85.15 | -0.82 | 0.625 |
| max drawdown | 65.75 pp | 66.92 pp | +1.18 pp | 0.750 |
| steps to pf30 ≥ 40% | 429k | 424k | **-6k** | 0.875 |

Per-seed `sef`: b14 `5.6 / 6.9 / 2.8 / 7.9` against b16 `20.6 / 30.5 / 10.5 / 7.0`. 0.125 is the
**floor** at n=4 — it means all four seeds moved the same way, which is what `best_perfect30` did.

### ‡ It is not a horizon artifact: the gap widens monotonically

The obvious objection is that 1.25M was chosen because that is where batch 16 stopped. Swept:

| horizon | `sef` b14 | `sef` b16 | delta | `best30` delta |
|---|---|---|---|---|
| 400k | 0.62% | 0.37% | -0.25 pp | +6.33 pp |
| 600k | 1.41% | 2.79% | +1.37 pp | +11.00 pp |
| 800k | 2.81% | 6.49% | +3.68 pp | +6.58 pp |
| 1.0M | 4.45% | 12.61% | +8.17 pp | +13.50 pp |
| **1.25M** | **5.80%** | **17.15%** | **+11.35 pp** | **+12.58 pp** |

A noise effect would wander; this one is absent at 400k and grows at every step after. That is the
signature of arms consolidating earlier, not of a lucky slice.

### What it establishes, and what it does not

**The mechanism is consolidation, not speed and not the ceiling** — and that is exactly the
pre-registered "handover unchanged, level up" outcome:

- **Not speed.** Steps to pf30 ≥ 40% is flat (-6k, p=0.875), and the epsilon handover was flat too
  (11.5k against 12.5k, three of four seeds in the same eval).
- **Not the ceiling.** Peak trailing +0.41 with batches 11-15 holding 94.8-95.0 anyway.
- **It is time spent near the top.** `sef` is literally the share of evals at ≥80% perfect, and it
  roughly **tripled**. Arms without the shaping reach a high perfect rate at the same moment and then
  *hold* it, where the shaping-on arms oscillate back down.

That reading fits the mechanism proposed before launch: 0.001 per retreating move is a small
permanent tax on exactly the detours a 93% endgame requires, so it does not stop an arm learning to
win — it stops it staying there.

**What this is not:** established. One batch at n=4, p=0.125-0.250, and **the arms were stopped at
1.25M**, so whether the advantage holds, closes or reverses past that point is untested — batch 14's
own curve was still climbing there and its full-run `sef` is 21.53% against batch 16's 17.15% at 1.25M.
It needs a replication at a longer horizon before it goes into [`findings.md`](findings.md) as more
than a lead. **Do not restore the shaping** in the meantime: nothing here argues for it, and batch 17
onward runs with it off.

### ‡ The 95% gate saved 65% of the work and cost the ability to rank checkpoints

A protocol result, and the first close-out run at `EVAL_MIN_ACHIEVABLE=95`:

| arm | rows | episodes | vs flat 100 | deepest row | rows ≥90% | rows ≥95% |
|---|---|---|---|---|---|---|
| b16a | 127 | 4,680 | 36.9% | 100 ep (one row) | 9 | **0** |
| b16b | 196 | 7,080 | 36.1% | 90 ep | 15 | **0** |
| b16c | 45 | 1,390 | 30.9% | **40 ep** | 0 | **0** |
| b16d | 26 | 760 | 29.2% | **30 ep** | 0 | **0** |
| **total** | 394 | **13,910** | **35.3%** | | 24 | **0** |

The saving beat the predicted 31%. The cost is that **no arm produced a full-length measurement worth
ranking**: b16c's best is 34/40 and b16d's 24/30, so their headline numbers carry ±15 pp intervals, and
`best_full_length_row`'s half-depth fallback is now the normal path rather than the exception it was
written as. Two consequences:

- **Cross-batch best-checkpoint comparison is dead at this gate.** Batch 14's bests are 100-episode
  measurements; batch 16's are 30-90 episode ones. Use `pooled_equal_effort` (exact at any gate) or the
  training-graph metrics, which do not touch the eval protocol at all.
- **The gate is calibrated above the population.** Zero of 394 rows reached 95% and only 24 reached
  90%, so at this skill level a 95 gate is not selecting good checkpoints — it is declining to measure
  almost all of them. See
  [`hyperparamTuning.md`](hyperparamTuning.md#measuring-a-policy-properly-eval_checkpointspy) for the
  standing recommendation that follows.
