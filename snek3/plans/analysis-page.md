# Snek analysis: rename the sweep page, cover every batch, publish it on the site

**Status: planned 2026-09-08, not started.** Written with the user after b26/b27; implement in the order of work below. Supersedes the TODO in `docs/runs.md` about adding the ladder to the sweep manifest.


## Context

`viewer/sweep.html` (data `viewer/sweep.js`, from `python -m tools.sweep_analysis reduce`) is the interactive
twin of `docs/sweep.md`: pick a batch, see each knob's curve with its seeds against the reference band, click a
cell for its seeds and traces, click a seed for the arm's chart. It covers b9-b21 from `plans/hyperparam-sweep.json`
and b26 from `plans/sweep-extra.json` (added 2026-09-07). The user wants it to be the project's analysis page:
renamed **snek analysis**, covering **every batch** (b1-b8, the b22-b25 ladder, b26, b27 and whatever comes next),
and **on GitHub Pages** beside the chart viewer rather than a local build only.

Three things stand in the way today, all found in the exploration:

- **One horizon for everything.** `HORIZON = 50_003_968` and the bin sizes are module constants in
  `tools/sweep_analysis.py:52-58`; both binners clamp into the last bin, so a 100M or 200M arm has everything past
  50.25M crushed into bin 200, and the page reads one scalar `S.horizon` and takes its step grid from an arbitrary
  arm (`sweep.html:330,371`). b4-b6 (200-270M), b8 (100M), b24/b25 (200M), b27 (100M) and b26's own reference
  (b24) are all wrong on the current grid; b1-b3 (3-20M) would sit in the first few percent of the axis.
- **The page only shows manifest batches.** `references.json` is not the gate; b27 has a reference entry and is
  in neither manifest. b22-b25 and b27 have their cells and env only in the specs on `origin/ops`; b1-b8 have no
  recoverable spec (ops holds b17-b27 only) and b3 has the old name shape (no `-seedN`, two-token cells, seven
  arms still under their pre-rename `p0*` names).
- **Not publishable as is.** `publish_pages.publish` copies a fixed list (`index.html`, `manifest.js`, `charts/`);
  `sweep.html:532` hardcodes `'../runs/' + policy + '.png'` for the arm lightbox where the site has `charts/`;
  the data file is 7.1 MB and stamps `generated` so it would change on every build of a branch that is one
  amended, force-pushed snapshot rebuilt every network cycle.

## Part A -- the rename, everywhere at once, old names gone

| today | after |
|---|---|
| `viewer/sweep.html`; `viewer/sweep.js` / `sweep.json`, `window.SNEK_SWEEP` | `viewer/analysis.html`; `viewer/analysis.js` / `analysis.json`, `window.SNEK_ANALYSIS` |
| `tools/sweep_analysis.py` (`reduce | figures | peaks`), `tools/sweep_figures.py` | `tools/analysis.py`, `tools/analysis_figures.py`; CLI `python -m tools.analysis reduce` |
| `plans/sweep-extra.json` | `plans/analysis-batches.json` -- every batch the page shows that the plan does not (Part B). `plans/hyperparam-sweep.json` stays: it is the sweep's design |
| `charts/sweep/*.png` | `charts/analysis/*.png`; the 25+ image links in `docs/sweep.md` follow |
| `docs/sweep.md` | stays, as the b9-b21 *report*; its interactive-twin paragraph (L9-12), §4 and §5 point at the new names and say the page now covers every batch |
| page strings: `<title>Snek sweep</title>`, "Every arm of the sweep" (levers), help line, the missing-data message naming the command, the "b9 first" comment | "Snek analysis", "every arm", the new command |
| `.gitignore:81-87`; `snek3/CLAUDE.md` rows L35 (viewer layout), L213 (docs table), L238 (tools table); `docs/runs.md:114-116` (the TODO this supersedes); `plans/sweep-analysis.md` (status line at the top: renamed and extended, date) | updated |
| `tests/test_sweep_analysis.py`, `tests/test_sweep_figures.py` | `tests/test_analysis.py`, `tests/test_analysis_figures.py` |

Not renamed: `tools/sweep_specs.py` (it expands the sweep plan into job specs), the word "sweep" in
`progress_update.py:642`'s per-batch `charts.md` heading (a batch is a sweep of one knob), `site_build.py`'s
unrelated "sweep jobs" filter. `git mv` for the tracked files so history follows.

## Part B -- every batch, each on its own horizon

### B1. Horizon and bins per batch (`tools/analysis.py`)

- `build` computes each batch's `horizon` = max `summary.step` over its arms (reference arms excluded), rounded up
  to a whole bin, and picks bin sizes that keep ~200 trace bins and ~20 stage-B bins: `bin = horizon / 200`
  rounded to a clean step (250k at 50M, 500k at 100M, 1M at 200M, 12.5k at 3M), `stage_b_bin = 10 * bin`.
  `arm_record(policy, runs_dir, horizon, bin, stage_b_bin)` takes them as arguments; `bin_trace` and
  `stage_b_density` lose their module-constant defaults.
- The batch dict carries `horizon`, `bin`, `stage_b_bin`; top-level `horizon`/`bin` go away. A reference arm from
  another horizon (b26 <- b24 at 200M, b8 <- b4 at 100M) is binned on the *reference's own* horizon and stored
  once per arm; the page draws the reference's traces on the batch's x domain, clipped, and labels the band with
  the reference's horizon -- the curve view compares scalars only, as today.
- Page (`analysis.html`): x domain from `b.horizon`; step grid from the batch's own arms (never
  `S.arms[Object.keys(S.arms)[0]]`); smoothing menu built from `b.bin` (1x / 2x / 4x) instead of the
  hard-coded `250k / 500k / 1M`; the status line prints the batch's bin.
- `late` stays a fraction of each arm's own evals (`LATE_FRACTION`), which already scales.

### B2. One list of batches: `plans/analysis-batches.json`

Renamed from `sweep-extra.json`, same shape as the plan's `batches` (`batch`, `knob`, `control_value`, `cells[]`
of `slug`/`env`/`prediction`, `notes`), read after the plan's manifest as today (`manifest_batches`). Two small
extensions to the cell shape, both optional:

- `cells[].arms`: an explicit arm list, used instead of the name-token match. Needed for b3 only (its two-token
  cells and the seven arms still named `p0l`-`p0r`); `batch_of`/`knob_of` are not changed.
- `cells[].control: true`: this cell is the batch's reference when `references.json` has no entry for the batch.
  Needed for b7 (its own `fc320` cell) and available for b4-b6 (see table). Where `references.json` has an entry,
  it wins, so b9-b27 are unchanged.

Entries to write, from `docs/results.md`, `plans/hyperparam-sweep.md` and the ops specs:

| batch | knob | cells (axis order) | reference |
|---|---|---|---|
| b1 | DDQN baseline, 3M | `baseline` (4 seeds) | none: `control: true` on its one cell |
| b2 | snek2-b29 reproduction (DQN), 3M | `b29repro` | same |
| b3 | PPO tuning pass, seed 1 only, ~10M | 10 cells by explicit `arms` (incl. `p0l`-`p0r`), the base config first as control | its own base cell |
| b4 / b5 / b6 | fc (200,100) + 8 ep; fc 320 + 8 ep; fc (200,100) 4 ep, at 200-270M | one cell each, 8 seeds | b4 and b5 read against b6 (`references.json`: arms = b6's), b6 against itself |
| b7 | fc layout, 50M | 8 layouts, categorical, `fc320` marked `control` | its own `fc320` cell |
| b8 | stability knobs, 100M | `ent003`, `entanneal`, `kl02`, `lam95` | already in `references.json` (b4 at 100M) |
| b22 | ladder rungs 1-2, 50M | `g999`, `g999roll512` | in `references.json` (b9 λ 0.99) |
| b23 | rungs 3-4 | `g999roll512mse`, `g999roll512msehold` | in `references.json` (b22 `g999roll512`) |
| b24 | horizon anneal, 200M | `hzanneal50` | in `references.json` |
| b25 | ladder top, 200M | `laddertop` | in `references.json` |
| b27 | move history, 100M | `hist0`, `hist4`, `hist8` | in `references.json` (b26 `pen01`) |

b22-b25's cells are multi-key, so they render categorical in manifest order (b19's path), with the reference at
index 0 -- right for a ladder. `env` for b22-b27 is copied from the specs (`git show origin/ops:snek3/desktop/
queue/pending/<arm>.json`), as a delta off the batch's base; b1-b8 from `results.md`.

**Helper, so no one types env dicts again:** `python -m tools.analysis describe b27` prints a ready-to-paste
entry -- cells from the distinct name tokens, `env` as the keys that differ between cells, `control_value` as the
reference cell's values -- using `tools/progress_update.py`'s existing `ops_specs()` / `spec_envs()` /
`knob_key()`. It works for any batch whose specs are on `ops` (b17 onward), which is every future batch; the
`queue-batch` skill gets one line telling the next session to run it after queueing a batch.

### B3. Degrade, never throw

- A cell with no arms draws an empty panel with its label; a batch with no non-reference cells is not possible
  once every batch has at least one cell, but `render()` guards it anyway.
- PPO-only rows (`value_loss`, `ev`, `kl`, `clipfrac`, `entropy`) are hidden for a batch whose arms carry no `ppo`
  block (b1, b2); `arm_record` sets `algo: 'dqn' | 'ppo'` from the row shape and the batch carries the set.
- The x axis is labelled from what the rows carry: `transitions` (b3 onward) or `steps` (b1, b2), stated once
  under the batch title so the DQN unit is never mistaken for game moves.
- Scalars missing a pass (`density98` with no stage B, `hof_*` with no hof file) stay `None`, as today; the peaks
  and levers views skip them, as today.

## Part C -- publish on the site

- **`publish_pages.publish`** copies `viewer/analysis.html` beside `index.html` and writes `analysis.js` beside
  `manifest.js`, both into the site worktree (never the build dir -- `prune_build_dir` deletes non-arm files
  there). The payload gets `charts_dir` (`'charts/'` on the site, absent locally) exactly as
  `viewer_manifest.render` does; `zoom()` in the page reads it with the `'../runs/'` fallback.
- **`site_build.build`** runs the reducer against the build dir, which holds every arm's `_evals.json` and every
  pass's merged JSON from both feeds (so the site's copy sees b5's hof5000 rows, which `runs/` lacks). It is gated
  on a fingerprint of the reducer's inputs -- `(name, size, mtime)` of the `_evals.json` and `_checkpoint_evals*`
  files -- kept beside `.feeds.json`; unchanged inputs mean no re-reduce and no changed file, so the ~7-10 MB blob
  is pushed only when a wave or pass actually landed. `generated` in the payload is set from the newest input
  mtime, not the clock, for the same reason. The reducer's ~1 min stays inside `SITE_BUILD_TIMEOUT` (900 s).
- **Nav:** a link each way in the two page headers (`index.html` has none today).
- One-time: remove nothing from `site` by hand; the first build writes `analysis.html`/`analysis.js`, and there
  is no `sweep.*` on the branch to clean up. Locally, `git rm` nothing -- the old data files are gitignored.
- The reducer stays runnable on the laptop (`python -m tools.analysis reduce` -> `viewer/analysis.js`, open
  `viewer/analysis.html` from disk), which is how the figures for `docs/sweep.md` are drawn.

## Files

| file | change |
|---|---|
| `tools/sweep_analysis.py` -> `tools/analysis.py` | rename; per-batch horizon/bins; `cells[].arms`, `cells[].control`; `algo` per arm; `charts_dir`; `describe`; `generated` from input mtime |
| `tools/sweep_figures.py` -> `tools/analysis_figures.py` | rename; read `batch['horizon']`, batch-own step grid; `FIGURES_DIR = charts/analysis` |
| `viewer/sweep.html` -> `viewer/analysis.html` | rename; strings; per-batch horizon/bin; PPO rows gated; x-axis unit; `charts_dir`; guards; nav link |
| `viewer/index.html` | nav link to `analysis.html` |
| `plans/sweep-extra.json` -> `plans/analysis-batches.json` | rename; entries for b1-b8, b22-b25, b27 |
| `viewer/references.json` | entries for b4, b5 (arms = b6's) |
| `tools/publish_pages.py`, `tools/site_build.py` | copy the page, write the data, fingerprint gate |
| `tests/test_sweep_analysis.py` -> `tests/test_analysis.py`, `tests/test_sweep_figures.py` -> `tests/test_analysis_figures.py`, `tests/test_publish_pages.py`, `tests/test_site_build.py` | renamed and extended (below) |
| `docs/sweep.md`, `docs/runs.md`, `snek3/CLAUDE.md`, `plans/sweep-analysis.md`, `.gitignore`, `skills/queue-batch/SKILL.md` | names, the TODO, the `describe` step |
| `charts/sweep/` -> `charts/analysis/` | `git mv`, then regenerate |

## Order of work

1. Rename (Part A) as one commit: `git mv` the tracked files, rewrite names and strings, run the suite. Nothing
   behavioural changes yet; the page still works from disk.
2. Per-batch horizon (B1) with its tests; regenerate and check b24-b26 draw to 200M.
3. Manifest extensions and the batch entries (B2, B3), `describe`, `references.json`; regenerate; open every
   batch in the page from disk and confirm nothing throws (b1, b3, b5 are the hard cases).
4. Publish (Part C): `publish_pages`, `site_build`, nav, tests; deploy to the desktop (`desktop-deploy`), trigger,
   open https://twang35.github.io/Snek/analysis.html.
5. Docs: `docs/sweep.md` §4/§5 and figures, `runs.md`, `CLAUDE.md`, the skill line.

Code commits wait for approval per the repo's git rule; docs commit as they are done.

## Verification

- `pytest tests/test_analysis.py tests/test_analysis_figures.py tests/test_publish_pages.py tests/test_site_build.py`,
  then the full suite. New tests: horizon/bins per batch (a 50M and a 200M batch in one build get different
  `bin` and a 200M arm's last trace bin is not the whole tail); `cells[].arms` overrides the name match;
  `cells[].control` makes the reference without a `references.json` entry; a DQN arm's `algo` and the PPO rows
  hidden; `charts_dir` present when publishing and absent locally; `publish` copies `analysis.html` and writes
  `analysis.js` (sentinel bytes, parse the `window.SNEK_ANALYSIS = ` prefix); `site_build` skips the reduce when
  the fingerprint is unchanged and re-reduces when an `_evals.json` changes; `describe` on a fake ops tree prints
  the expected entry.
- `PYTHONPATH=. python -m tools.analysis reduce` then open `viewer/analysis.html` from disk: every batch in the
  picker, b24/b25 traces reach 200M, b1/b2 show no PPO rows and say "steps", b3 shows 10 cells, clicking a seed
  opens its chart.
- After deploy and `trigger`: the live page loads `analysis.js`, a seed click opens `charts/<policy>.png`, and a
  second `trigger` with nothing changed does not rewrite `analysis.js` (check the `site` commit's file list).

## Decisions taken (say if any should go the other way)

- b1-b3 are included as browsable batches with degraded views, since the user asked for all; the alternative
  (leave them to the chart viewer, as `plans/sweep-analysis.md:253` recommended) is a one-line removal from the
  batch list.
- The data file is regenerated by the desktop's site build rather than committed; ~7-10 MB per push when a wave
  or pass lands, nothing otherwise.
- `docs/sweep.md` keeps its name and scope (the b9-b21 report). If a report on the other batches is wanted, it is
  a new file, not a widening of this one.
