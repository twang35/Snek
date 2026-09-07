# Reading the sweep — the analysis plan

**Written 2026-09-06, after b18 closed the one-knob sweep; built the same day** — `tools/sweep_analysis.py`, `tools/sweep_figures.py`, `viewer/sweep.html`, `docs/sweep.md`, `charts/sweep/`. Decisions taken at build time: b3–b8 stay out (not part of the sweep); the traces overlay cells, with the arm view one click away. b9–b21 are done: 13 batches, 116 cells,
464 arms, every arm at 50M with its stage B, `hof5000` and `hof30k` files in `runs/`. What exists so
far is one table per batch in `results.md` and a paragraph of reading, and one PNG per arm. What does
not exist is a **picture of any knob's curve**, a **cross-batch view** of which knobs are levers, or
any reading of the **training diagnostics** every stage-A row has carried since PPO landed — value
loss, explained variance, approx KL, clip fraction, entropy — which nobody has plotted once.

This plan is two deliverables from one reducer:

| deliverable | what | where |
|---|---|---|
| **the report** | one Markdown file, one section per knob, each with the knob's curve figure and its trace figure, then a cross-batch levers section | `docs/sweep.md`, figures in `charts/sweep/` |
| **the page** | a local static page beside the chart viewer: pick a batch, see every metric as a row of small charts, flip cells, hover a point for its arm | `viewer/sweep.html`, data in `viewer/sweep.js` (local build, gitignored like `manifest.js`) |
| **the reducer** | walks `runs/` once, writes the numbers both draw from, so the page and the report cannot disagree | `tools/sweep_analysis.py` |

The order below is the order to build it: reducer, report, page. The report alone answers the
questions asked; the page is what makes the 116 cells browsable.

## 1. What the data holds

| file | rows per arm | what each row has | what it answers |
|---|---:|---|---|
| `<arm>_evals.json` (stage A) | 3,052 (one per rollout) | `step`, `avg_score`, `perfect_percent` (100 episodes), and **`ppo`**: `value_loss`, `explained_variance`, `approx_kl`, `clip_fraction`, `entropy`, `policy_loss`, `epochs_run`, `clip`, `learning_rate` | the whole training trace: how fast, how high, how stable, and *why* — the diagnostics are the only view into the optimiser |
| `<arm>_checkpoint_evals.json` (stage B) | ~1,000 (every checkpoint at ≥97/100) | `step`, `perfect_percent` /500, CI | the record region: how wide, how high, and **when** in training |
| `_hof5000.json` | 0–200 (every stage-B row ≥99) | the same at 5,000 episodes | what survives re-measurement |
| `_hof30k.json` | 0–40 (every hof5000 row ≥99) | the same at 30,000 episodes, seed 7 | "truly over 99%" — readout 6, the north star |
| `plans/hyperparam-sweep.json` | — | each cell's `env` and `prediction` | the numeric knob value for the x axis, and what the cell was expected to do |
| `viewer/references.json` | — | each batch's reference arms and their knob value | the control, slotted into the curve at its own x |

Two cautions the reducer has to encode rather than the reader remember:

- **Two references.** b9–b14 read against b9's λ 0.99 cell (`b9bw`–`b9bz`); b15–b21 read against b7's
  λ 0.98 cell (`b7aa`–`b7ad`). `references.json` already says which; every figure draws its batch's
  own reference and says so in the caption.
- **Not every diagnostic is comparable across a batch.** `value_loss` is Huber except b19's `mse` cell
  (13.4 late against the base's 1.5 — a different function, not a worse critic), and its scale moves
  with γ (b10: longer horizons mean larger value targets). `explained_variance` is the scale-free
  version and is the one to rank on; `value_loss` is drawn but not ranked on b10 or b19.

## 2. The metrics — what to compute per arm

Grouped by the question each answers. Every scalar is per arm; the cell's value is the four seeds
shown as dots plus their median as the line, never a mean of four alone (`protocol.md`: lead with
the sign test). **Bold** rows are the pre-registered readouts from `hyperparam-sweep.md` §4.

### Outcome — how good, at depth

| metric | definition | source | note |
|---|---|---|---|
| **density98** | share of stage-B rows ≥98 /500 | stage B | readout 1, the primary |
| density99 | share ≥99 /500 | stage B | the `hof5000` cut; steeper, noisier |
| best row | max stage-B row | stage B | a selected high; reported, not ranked on |
| **hof5000 mean, best, count ≥98.73** | over the pass's rows | hof5000 | 98.73 is the snek2 champion |
| **hof30k best, mean** | over the pass's rows | hof30k | readout 6. Empty for most arms — a cell with *any* row here is already a finding |
| **best30** | `summary.best_perfect30` | stage A | readout 5. One number per arm; the user's rank-1 statistic |
| best30 step | where the peak was | stage A | early peaks (b21l at 7.7M) are a different story from late ones |
| **stage-A ≥98 share, post-onset** | share of evals ≥98 after competence | stage A | readout 3; the free proxy for density98 (+0.80) |

### Stability — does it hold

| metric | definition | source | note |
|---|---|---|---|
| **drawdown <50, <80** | share of post-onset evals below 50 / 80 | stage A | readout 2, the collapse share. Onset = first eval ≥80, as `viewer_manifest` computes it |
| worst post-onset eval | min `perfect_percent` after onset | stage A | how deep the worst collapse went |
| end drop | `best30 − trailing_now` at the cap | stage A | did it keep what it had |
| late sd | sd of `perfect_percent` over the last 300 evals (~5M) | stage A | the user's rank-2 statistic made numeric |

### Speed — how fast

| metric | definition | source | note |
|---|---|---|---|
| **onset** | first step with trailing-30 ≥90 (the plan's definition) *and* first ≥80 eval (the manifest's) | stage A | readout 4. Compute both, draw the plan's. "Not reached" is a value, drawn as an open marker at the cap |
| step to 95, to 98 (trailing-30) | first crossing | stage A | speed at the top, where it matters |

### Training diagnostics — why

All from the `ppo` block, over the **last 20% of the run** (610 rollouts, ~10M) unless stated, so
they describe the endgame the record region lives in rather than the climb.

| metric | definition | what it says |
|---|---|---|
| value loss, late mean | mean `value_loss` | the critic's fit. The "TD loss" of PPO. Not comparable across b10 or into b19's mse cell (§1) |
| explained variance, late mean | mean `explained_variance` | the critic's fit, scale-free. The one to rank on |
| approx KL, late mean and **p99** | over all post-onset updates | how far each update moves the policy; the p99 is the tail the collapse hypothesis is about |
| clip fraction, late mean | share of samples the clip bound | whether the trust region binds at all (0.4 clip should read near zero) |
| entropy, at the cap | final `entropy` | how committed the policy is; ln 3 = 1.099 is uniform |
| epochs run, mean | b16 only | whether target KL fired |

**Two cross-metric questions worth one figure each**, because they are what the tables cannot show:

- **Does the critic predict the record?** Scatter of explained variance (late) against density98 over
  all 464 arms, coloured by batch. If the b19 finding (collapses live in the critic) generalises, the
  correlation is there; if not, the critic is a red herring.
- **Does the KL tail predict collapses?** Scatter of approx-KL p99 against drawdown <80. b18 said
  gradient clipping is a no-op, which argues against the tail hypothesis; this tests it with 464
  points instead of 24.

## 3. The figures

### 3a. The curve — one per batch, the headline (readout 7)

x = the knob value (log axis for lr, minibatch, rollout, lanes, entropy; linear for λ, γ, epochs,
clip, grad clip; categorical for b17's anneals, b19's switches, b21's gates). One panel per metric,
stacked in a column, sharing x, so the eye reads down: where density peaks, does best30 peak there
too, does drawdown dip there. Four seed dots per cell, median line through them, the reference cell
drawn at its own x as a **hollow gold marker** with a shaded band across the panel at its median
(the "inside the noise" test made visible). Cells that did not reach onset get an open marker on
the onset panel and no point elsewhere.

Panels, top to bottom: density98 · hof5000 best · hof30k best (with a "no rows" tick) · best30 ·
stage-A ≥98 · drawdown <80 · drawdown <50 · onset · explained variance · approx-KL p99 · entropy at
cap. Eleven panels, each short (~1.2 in), one figure ~14 in tall at the report's width.

### 3b. The traces — one per batch, everything against step

The picture the user described, generalised: **one row per metric, x = step, every cell of the
batch overlaid as one line, coloured by knob value on a sequential palette** (dark = low value,
bright = high; categorical batches get a categorical palette). A monotone knob shows as an ordered
colour gradient; a flat knob shows as an unordered tangle; a cliff shows as one colour leaving the
band. Each line is the cell's **median over its 4 seeds at each step bin** — seeds are the next
figure's job. The reference is a thick grey line.

Rows: trailing-30 perfect rate · average score · **stage-B density over time** (rows ≥98 per 2.5M
bin, as a share of rows in the bin — this is what separates b17's hold-at-floor cells, which lift
*late* density, from a knob that lifts it everywhere) · value loss (log y) · explained variance ·
approx KL (log y) · clip fraction · entropy.

### 3c. The cell — 4 seeds across, metrics down

Exactly the user's 4x1 layout: pick one cell, columns are its four seeds, rows are the same metrics
as 3b, each panel a raw trace (thin) with its trailing average over it, the reference's four seeds
in grey behind. This is the figure that shows *one collapsed seed* — the b21 finding that seed 1 is
the weak seed in five of six cells came from reading 24 PNGs by hand, and this makes it one glance.
In the report only for the cells the reading names; on the page for every cell.

### 3d. The levers — one figure for the whole sweep

A forest plot: one row per knob (13), x = the best cell's delta from its reference in pp of
density98, drawn as the four seeds' range with the median marked, the reference at zero. Beside it
the same for drawdown <80 and for hof5000 best. Sorted by the density delta. This is the picture
the corner grid is designed from, and it makes the n=4 spread the visible thing rather than the
mean.

### 3e. The peaks — where each metric peaked, per knob

A table (and a heatmap on the page): rows = knobs, columns = metrics, cell = the knob value that
won that metric, coloured by how far outside the reference's seed range it sits (0 = inside, so a
no-op batch is a grey row). Generated, not hand-written: it is the direct answer to "where did each
setting peak on best30, value loss, drawdown, perfect rate". Beside each winner, the Mann-Whitney
p against the reference (0.029 is the floor at n=4, and the plan's bar for "separated").

### 3f. The two scatters (§2)

Explained variance against density98, and approx-KL p99 against drawdown <80, all 464 arms plus the
64 of b7, coloured by batch, with Spearman ρ in the corner.

## 4. The page — `viewer/sweep.html`

Same shape as the chart viewer: one HTML file, no server, no dependency, opened from disk, state in
the URL hash so a view is a link. The charts are inline SVG drawn by the page from `sweep.js`, not
PNGs — hover gives the arm and its number, a click on a trace opens that arm's existing PNG in the
lightbox, and a cell in 3a is a link to its 3c view.

| control | choices |
|---|---|
| batch | b9 … b21, newest first, as the viewer |
| view | **curve** (3a) · **traces** (3b) · **cell** (3c, plus a cell picker) · **levers** (3d + 3e, batch-independent) · **scatter** (3f) |
| metrics | a checklist of the rows to show, default the eleven above; the choice persists in the hash |
| x | step or transitions (b13/b14/b20 change the rollout, so the eval cadence differs — transitions is the honest axis and the default) |
| smoothing | trailing window for the trace rows: 1, 10, 30 |

Keyboard: `←`/`→` previous/next batch, `↑`/`↓` previous/next cell in the cell view, `1`–`5` the
views, `r` toggles the reference overlay.

**The data file.** 464 arms × 3,052 rows × 8 metrics is ~60 MB of JSON, too much for a page load.
The reducer bins each trace to **200 bins of 250k steps**, writing per bin the mean and, for the
perfect rate, the **min** too — a collapse is one eval at 0% among fourteen at 95%, and a bin mean
alone hides exactly the thing the drawdown rows exist to show. That is ~4 MB as `sweep.js`, which
opens from `file://` in under a second. Scalars, stage-B density-over-time and the hof rows ride in
the same file. `window.SNEK_SWEEP = {...}` like the manifest, for the same `file://` reason.

The page could later go on the `site` branch through `publish_pages` with no change, since it is
static; that is not part of this plan.

## 5. The report — `docs/sweep.md`

Newest-first does not apply: this is a reference document in knob order, written once.

1. **The levers** — figure 3d, table 3e, the two scatters, and the one-paragraph reading of each: which
   knobs separated, which are flat, which are confounded (b20 against b14, the two references).
2. **One section per knob, b9 → b21** — figure 3a, figure 3b, the batch's peaks row from 3e, then a
   short reading that names the cell the corner grid should take and links the `results.md` section
   for the per-arm table. The 3c figure only where a seed effect is the story.
3. **What the diagnostics said** — the first reading of the `ppo` block across the sweep: does
   explained variance track density, does the KL tail track collapses, what entropy the record
   cells settle at.
4. **How this was made** — the reducer command, the bin size, the two references, the value-loss
   caveats. Short.

Figures are PNGs under `charts/sweep/<batch>-curve.png`, `<batch>-traces.png`, `levers.png`,
`peaks.png`, `scatter-*.png`, drawn by matplotlib through the object API (as `progress_chart.py`
does, for the same leak reason) and committed with the doc — `charts/` is where one-off diagnostic
figures a finding refers to already live.

## 6. The reducer — `tools/sweep_analysis.py`

```
PYTHONPATH=. python -m tools.sweep_analysis reduce            # runs/ -> viewer/sweep.js (+ a .json twin for the renderer)
PYTHONPATH=. python -m tools.sweep_analysis figures [b9 ...]  # sweep.json -> charts/sweep/*.png
PYTHONPATH=. python -m tools.sweep_analysis peaks             # the 3e table as Markdown, to paste into docs/sweep.md
```

- **Reuses, does not redefine.** `viewer_manifest.py`'s `onset`, `drawdown`, `stage_a_share`,
  `batch_of`, `knob_of`, `seed_of` and `references()`; `progress_update.py`'s `knob_value` for the
  numeric x from a cell's env. A metric defined in two places is a metric that will disagree.
- **The knob value comes from the sweep manifest**, not the arm name: `hyperparam-sweep.json` cell
  `env` → the one key that differs from `base` → its value. b17/b19/b21 cells with two keys or a
  categorical value keep their slug as the label and get an ordinal x.
- **Bins by transitions**, so b13/b14/b20 line up with the rest.
- **Tests**, same pass: onset/drawdown against a hand-built trace with one collapse (the min-per-bin
  rule), the knob-value extraction for a numeric, an anneal and a switch cell, and the bin count for a
  50M arm. Mutation-check the min-per-bin: replace min with mean and the collapse test must fail.

Estimated size: the reducer ~300 lines, the figures ~350, the page ~500 of HTML+JS. Reduce runs in
~1 min (464 × 1.6 MB stage-A files plus ~2.8 MB stage-B each — ~2 GB read once).

## 7. Order of work

| step | produces | ~time |
|---|---|---|
| 1 | reducer + tests, `sweep.json` | 1.5 h |
| 2 | figures 3d, 3e, 3f — the sweep-wide answer first | 1 h |
| 3 | figures 3a, 3b for all 13 batches | 1 h |
| 4 | `docs/sweep.md` with the readings | 1.5 h |
| 5 | `viewer/sweep.html` | 2 h |

Steps 1–4 are the report the user asked for; step 5 is the browsing tool. Steps 2 and 3 load the
`dataviz` skill before the first line of chart code.

## 8. Decisions taken, and the ones left

Taken here, say so if wrong:

- **Median of four, not mean**, as the cell's line; the four dots always shown.
- **Transitions as the x axis**, step as the alternative.
- **Late = last 20% of the run** for the diagnostics.
- **The report goes in `docs/`**, as a reference document in knob order, not newest-first.

Left to the user:

- Whether the page should also carry the **b3–b8 batches** (different bases, different horizons —
  browsable but not on any curve). Recommendation: no, they are in the chart viewer already.
- Whether 3b should overlay **cells** (one median line each, 16 lines for b9) or **arms** (64 lines).
  Recommendation: cells, with the arm view one click away in 3c.
