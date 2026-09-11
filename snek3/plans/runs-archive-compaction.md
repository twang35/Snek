# Compacting the `runs/` archive: what each file carries, who reads it, and five ways to shrink it

**Status: proposal 2026-09-11, for review.** Nothing built. The versions in §4 are alternatives, not
steps; the recommendation is §5.

## 1. What is asked

The b28/b29 close-out commit added 1.26 GB of JSON to master and GitHub warned on 14 files over 50 MB
(the largest, `b28n`'s `hof30k` file, is 78 MB; the hard limit that rejects a push is 100 MB). The
request: a per-run summary JSON that every chart can be redrawn from, carrying any other cheap metric
that cannot be recomputed from it, and a plan with several versions to choose between.

## 2. What the archive is, measured

`runs/` on the laptop is **5.7 GB** (the desktop's 4.0 GB). By file kind:

| file | per arm | total | what it is | share that is `episode_scores` |
|---|---:|---:|---|---:|
| `<arm>_evals.json` | 3.3 MB (200M arm) | **3.1 GB** | stage A: one row per trainer self-eval (6,104 rows at 200M) | 0% (no arrays; the `ppo` block is ~60%) |
| `<arm>_checkpoint_evals.json` | 10-13 MB | 1.8 GB | stage B: every screened checkpoint at 500 episodes | 87% |
| `…_hof5000.json` | 57-76 MB | 1.5 GB | 5,000 episodes per row, early-stopped at 99.6 | 98% |
| `…_hof30k.json` | 55-78 MB | 1.1 GB | 30,000 per row, early-stopped at 99.8 | 99% |
| `.png` + `.md` | | 82 MB | drawn from the above | — |

**Git compresses the arrays ~50x, so the pack is the smaller problem.** The b28/b29 commit's 1,261 MB
raw is 29.6 MB in the pack; master's whole pack is 552 MB, of which `snek3/runs` is 371 MB (6.3 GB
raw). The feeds carry the same files: `results` 293 MB compressed / 5.8 GB raw, `laptop-results` 144 MB.

So the costs, in order of how much they bite:

1. **The 100 MB per-file limit.** A denser or longer pass than b28's crosses it and the push is
   rejected; the file then cannot be archived at all without a format change under pressure.
2. **Checkout and working-tree size**: 5.7 GB on every clone, of which >95% is never read.
3. **Tool time**: `viewer_manifest` parses every arm's four files on every site build and progress
   update (~1 GB of JSON per build today); the laptop's git wrapper adds ~1 s per call on top.
4. **Pack growth**, ~30 MB per 8-arm 200M batch. Real but slow.

## 3. Who reads what — the schema each reader actually needs

Checked by grep across `tools/`, `viewer/`, `train.py`, `ppo/`, `dqn/` (tests excluded).

| field | in | readers | verdict |
|---|---|---|---|
| `episode_scores` | stage B, hof | **none.** `eval_plan.perfect_flags` derives win flags from it and has no caller; `median_score` is computed from it at write time. The stated reasons for keeping it (`eval_plan.py`, `engine.py`): a median does not pool, and resumability | **replace with a histogram** (§4). A histogram of scores is exact for the median, mean, min, max, percentiles and the perfect count — every order-independent statistic — so nothing the rationale protects is lost. Resumption is by step and rows are written only when complete (`shard.py`), so no reader needs the per-episode order |
| `perfect_ci95` | stage B, hof | `stage_b_chart` (error bars) | derivable: Wilson interval of `perfect_games` / `episodes`. Keep for the chart or recompute; 20 B a row either way |
| `median_score`, `avg_score`, `min_score`, `max_score`, `avg_reward` | stage B, hof | none outside the row's own one-line log | derivable from the histogram (except `avg_reward`); droppable or kept, ~60 B a row |
| `step`, `episodes`, `episodes_planned`, `abandoned`, `perfect_games`, `perfect_percent`, `seconds`, `stage_a_percent` | stage B, hof | `stage_b_chart`, `viewer_manifest`, `progress_update`, `eta` (`seconds` for pass ETAs), `sweep_analysis` | **the core row.** ~120 B |
| `step`, `perfect_percent`, `avg_score` | stage A | `progress_chart` (the arm's graph), `viewer_manifest` (drawdown, ≥98 share, onset), `run_report`, `train.py` on resume | **the core series** |
| `trailing_avg_score`, `transitions` | stage A | `run_report`, `train.py` | derivable (trailing mean of `avg_score`; `transitions` equals `step` for PPO) |
| `min_score`, `max_score`, `avg_reward`, `steps_per_second`, `entropy_coef` | stage A | `run_report`, `eta` (`steps_per_second`) | keep; not recomputable |
| `ppo` block (12 diagnostics: `approx_kl`, `clip_fraction`, `explained_variance`, `value_loss`, `policy_loss`, `entropy`, `epochs_run`, `stopped_early`, `episodes`, `perfect_games`, `rollouts`, plus the four live knob values) | stage A | `ppo/algo.py` log lines while training; nothing offline | **not recomputable and the most interesting diagnostic in the archive** (a collapse's KL and value-loss signature). Keep, but as rounded columns: ~60% of the stage-A file today |
| `summary` block | stage A | `viewer_manifest` (best30, sef, trailing, evals) | keep as is |

Everything the viewer, the site build, the progress tables and the reports need is in the two **bold**
rows plus the histogram. The charts are already redrawn from JSON by `site_build`, so a format change is
one reader (`stage_b_chart` / `progress_chart`) plus `viewer_manifest`.

## 4. The versions

Sizes are measured on `b28i` (a 200M arm: 4,067 hof5000 rows, 5,839 stage-B rows, 6,104 stage-A rows),
raw JSON and gzip-6 (a proxy for what git stores).

| | change | hof5000 file | stage-B file | stage-A file | `runs/` total (est.) | what is lost | effort |
|---|---|---:|---:|---:|---:|---|---|
| today | — | 57.0 MB / 0.62 gz | 10.2 / 0.23 | 3.3 / 0.36 | 5.7 GB | — | — |
| **A. prune** | `prune_runs arrays` also strips `episode_scores`; nothing else changes | 1.05 / 0.08 | 1.36 / 0.09 | 3.3 / 0.36 | ~3.3 GB | the median beyond what is already stored; any future per-episode analysis | half a day |
| **B. histogram** | `build_row` writes `score_counts: {score: n}` instead of the array; a `prune_runs histogram` converts every old file in place; `perfect_flags` and the median read the histogram | 1.76 / 0.21 | 1.66 / 0.14 | 3.3 / 0.36 | ~3.4 GB | only episode order, which nothing reads | a day: writer, two readers, converter, tests |
| **C. B + lean rows** | B, and derived fields dropped from rows (`perfect_ci95`, `median/avg/min/max_score`, `perfect_percent` recomputed by readers) | 1.26 / 0.18 | 0.95 / 0.10 | 3.3 / 0.36 | ~3.3 GB | nothing; every dropped field is a function of the kept ones | B + touching every reader of `perfect_percent` (many) — **not worth it**: 0.5 MB an arm |
| **D. B + columnar stage A** | B, and `_evals.json` stored as columns (`step: […], perfect_percent: […], …, ppo.approx_kl: […]`), floats rounded, `transitions`/`trailing_avg_score` dropped | 1.76 / 0.21 | 1.66 / 0.14 | **1.06 / 0.26** | **~1.4 GB** | nothing (the `ppo` diagnostics stay, rounded to 4 d.p.) | B + `train.py`'s writer and resume reader, `progress_chart`, `run_report`, `viewer_manifest`, `eta` — the stage-A schema has ~8 readers |
| **E. one summary file per arm** | D, and the four files collapse into `<arm>_runs.json`: `stage_a` (columns), `stage_b` / `hof5000` / `hof30k` (row tables with histograms), `summary`; full-row files stay on the box's disk, gitignored | one file ~3.5 MB | | | ~1.2 GB, and 1 file an arm instead of 4-7 | nothing in git; the raw rows survive only on the box that measured them until its disk is pruned | D + the feed (`results_feed`), the importer (`progress_update`), the manifest's file-existence checks, `charts.md` links — two to three days, and every skill that names a file |

Considered and set aside:

- **`.json.gz` in git.** Git already gets the 50x; a gzipped blob defeats delta compression and
  `git diff`, and the per-file limit is on the stored blob, which gzip does shrink — but B gets the same
  per-file win in plain text.
- **Git LFS.** Not installed on either box, every reader would need it, and the problem is
  the files' content, not where they live.
- **Rewriting history** to drop the arrays already pushed. Destructive to every clone and to the
  results feeds; the 371 MB it would recover is a one-off. Not proposed.
- **Thinning the stage-A series** (every other eval). It is the graph; and 3.3 MB an arm is 60% `ppo`
  block, which D removes without thinning.

## 5. Recommendation: B now, D after, E only if the file count itself becomes the problem

**B is the change that removes the risk** — the per-file limit, the parse time, 5.7 → 3.4 GB — with
zero information loss that anyone can name, one writer and two readers, and the same shape of
migration the project already did once (`prune_runs arrays`, 2026-09-01, 0.96 GB). Its histogram is
also the one new metric worth having (§6). **D is the second half of the win** (3.4 → 1.4 GB) and
touches the stage-A schema, which has the most readers; do it as its own change after B lands and
is verified. **C is not worth its churn** and **E is a re-architecture** for a problem B and D
have already solved; hold it until the arm count makes four files an arm the pain.

Migration for B, in order: (1) `build_row` writes `score_counts` and stops writing `episode_scores`;
(2) `perfect_flags` and the one-line log read either form; (3) `prune_runs histogram --apply`
converts every file under `runs/` on the laptop (dry run first, byte counts printed), one commit
"runs: episode_scores → score_counts (lossless)"; (4) the same command on both boxes' `runs/`, and
the feeds carry the new form from then on; (5) tests: a converted row's median, mean, min, max and
perfect count equal the originals over every stored row (the 2026-09-01 change verified 2.25M
episodes the same way). Deploy the writer to the desktop before its next pass, since a pass writes
its files at the end.

## 6. Metrics worth adding, because they are cheap and not recomputable

| metric | where | bytes | why |
|---|---|---:|---|
| **score histogram per row** (`score_counts`) | stage B, hof rows | ~100-300 B | it *is* version B — and it answers a question the arrays never did in practice: where a checkpoint's failures die (the distribution of non-perfect scores), which separates "starves at 60" from "loops at 90" collapse modes. `is_perfect_score` makes the perfect count one entry of it |
| mean score, failure count, worst score per row | derived from the histogram | 0 | free once B lands; no need to store |
| per-arm stage-A `ppo` diagnostics | already stored; D keeps them as columns | ~0.4 MB an arm | the collapse signature — KL, clip fraction, explained variance at the eval before a drawdown — is the one analysis the archive cannot do today because nothing reads the block; making it columnar makes it readable |
| wall-clock per row (`seconds`) and per arm (`wall_seconds`) | `seconds` is on every pass row; the arm's wall time is `arch.json` mtime → last eval, which does not travel | 8 B an arm | put `started` and `finished` ISO stamps in the stage-A `summary`, so §"how long does a batch take" stops needing `ssh` |
| eval seed, shard count, stop target, config string | already on the pass file header | 0 | keep |

Not worth storing: per-episode rewards or lengths (no reader, and the 2026-09-01 removal already
decided this), per-step anything from training (the trainer's own log has it).

## 7. Questions for the review

1. B alone, or B and D together in one pass? (Recommendation: two changes, B first.)
2. Does anything you do by hand read `episode_scores` — a notebook, an ad-hoc script? Nothing in the
   tree does.
3. Should the converter also run on the desktop's `savedPolicies` side (no — those are checkpoints,
   not rows) and on `snek2/runs` (no — frozen)?
4. `started` / `finished` stamps in the stage-A summary: yes?
5. Do we want the histogram keyed by score (`"95": 4915`) or as a 96-long array (`counts[score]`)?
   The dict is smaller for near-perfect rows (a 30k row has ~40 distinct scores) and reads as data;
   the array is simpler to index. The measurements above are the dict.
