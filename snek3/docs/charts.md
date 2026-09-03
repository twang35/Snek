# Charts — one graph per arm

**Batch b9 — the λ sweep — closed on the desktop 2026-09-02**, all 64 arms with stage B complete, 127
panels below. **Batch b10, the γ sweep, is training there now**; its charts arrive at close-out.
**Batch b8 closed 2026-09-01**, all sixteen arms with stage B complete, charts below.
The window to watch the box live is below.

**Batch b7 closed all four waves 2026-09-01** — 32 arms, 8 layouts x 4 seeds, charts imported from the
`results` branch the same day and below. **Batch b4's charts are here too**, imported and redrawn
2026-09-01.

**Batches b5 and b6 closed 2026-08-30** — eight seeds each, stage B complete on both, charts below.
**Batch b2 closed 2026-08-29.** Every arm's chart is committed here on every progress update, live
desktop arms included (rule changed 2026-09-02; the box's `desktop/deploy` redraws over them), and the
same PNGs are what the GitHub-Pages viewer at [`../viewer/index.html`](../viewer/index.html) pages
through, 8 at a time. Only a live arm's JSON waits for the `results` branch at close-out.

## Watching them live

**One window per box, opened by the trainings themselves**, showing every arm running there. Nothing
launches it; the first arm to start opens it and the last one to finish takes it away — and a panel
stays for the rest of the wave once it appears, so a batch with one arm left still shows all four. To
put it back
after closing it: `PYTHONPATH=. python -m tools.chart_window`. Killing it, closing it and relaunching
it are all free — no training reads it, waits on it, or reopens it ([`findings.md`](findings.md) on why
it took three attempts to get there).

## Batch b9 — the λ sweep off b7's `fc (320,)`, 4 seeds each, 50M

**All eight waves closed 2026-09-02**, stage A and stage B for all 64 arms; stage-B panels follow each
arm's chart. b9 changes **only** `ppo_gae_lambda` off b7's winning cell — `fc (320,)`, 4 epochs,
entropy 0.01, lr 3e-4, γ 0.99, clip 0.2, same 50M cap — so **b7aa-b7ad below are this sweep's λ 0.98
arms**, and `fc (320,)`'s panels there are the reference to read these against.

| λ | ≥98%/500 | best30 | drawdown < 50% | | λ | ≥98%/500 | best30 | drawdown < 50% |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0.00 | 0.0% | 88.28 | 11.29% | | 0.94 | 4.6% | 96.80 | 0.00% |
| 0.50 | 0.0% | 94.25 | 0.05% | | 0.95 | 5.3% | 96.97 | 0.02% |
| 0.80 | 0.8% | 96.12 | 0.02% | | 0.96 | 8.3% | 97.30 | 0.03% |
| 0.85 | 0.8% | 95.90 | 0.00% | | 0.97 | 11.7% | 97.88 | 0.03% |
| 0.90 | 3.9% | 96.75 | 0.07% | | 0.98 (b7) | 17.3% | 97.75 | 0.29% |
| 0.91 | 4.3% | 96.88 | 0.02% | | **0.99** | **27.3%** | 98.33 | 0.77% |
| 0.92 | 4.8% | 96.92 | 0.02% | | 0.995 | 27.3% | 98.40 | 1.03% |
| 0.93 | 5.4% | 97.17 | 0.02% | | 0.999 | 25.6% | 98.45 | 1.26% |
| | | | | | **1.00** | **29.5%** | 98.40 | 2.10% |

**What to look at.** λ 0.00 next to λ 1.00 first — the whole dynamic range of the knob, best30 88.28
against 98.40. Then λ 0.98 against λ 0.99: the density jump that changes the base every later batch
should be built on, and, in the stage-A traces, the first visible drawdown — the top four groups get
noisier as they get richer, λ 1.00 most of all. The stage-B panels for λ 0.99-1.00 are where the
100/500 rows are: `b9ch-lam999-seed4` at 47.2-47.3M and `b9bw-lam99-seed1` at 48.4M. `b9ab-lam0-seed2`
has no stage-B panel and never will: it screened **zero** checkpoints, which is the finding rather
than a gap.

**λ 0.00** — `b9aa`-`b9ad`:

![b9aa-lam0-seed1](../runs/b9aa-lam0-seed1.png)
![b9aa-lam0-seed1 stage B](../runs/b9aa-lam0-seed1_checkpoint_evals.png)
![b9ab-lam0-seed2](../runs/b9ab-lam0-seed2.png)
![b9ac-lam0-seed3](../runs/b9ac-lam0-seed3.png)
![b9ac-lam0-seed3 stage B](../runs/b9ac-lam0-seed3_checkpoint_evals.png)
![b9ad-lam0-seed4](../runs/b9ad-lam0-seed4.png)
![b9ad-lam0-seed4 stage B](../runs/b9ad-lam0-seed4_checkpoint_evals.png)

**λ 0.50** — `b9ae`-`b9ah`:

![b9ae-lam50-seed1](../runs/b9ae-lam50-seed1.png)
![b9ae-lam50-seed1 stage B](../runs/b9ae-lam50-seed1_checkpoint_evals.png)
![b9af-lam50-seed2](../runs/b9af-lam50-seed2.png)
![b9af-lam50-seed2 stage B](../runs/b9af-lam50-seed2_checkpoint_evals.png)
![b9ag-lam50-seed3](../runs/b9ag-lam50-seed3.png)
![b9ag-lam50-seed3 stage B](../runs/b9ag-lam50-seed3_checkpoint_evals.png)
![b9ah-lam50-seed4](../runs/b9ah-lam50-seed4.png)
![b9ah-lam50-seed4 stage B](../runs/b9ah-lam50-seed4_checkpoint_evals.png)

**λ 0.8** — `b9ai`-`b9al`:

![b9ai-lam80-seed1](../runs/b9ai-lam80-seed1.png)
![b9ai-lam80-seed1 stage B](../runs/b9ai-lam80-seed1_checkpoint_evals.png)
![b9aj-lam80-seed2](../runs/b9aj-lam80-seed2.png)
![b9aj-lam80-seed2 stage B](../runs/b9aj-lam80-seed2_checkpoint_evals.png)
![b9ak-lam80-seed3](../runs/b9ak-lam80-seed3.png)
![b9ak-lam80-seed3 stage B](../runs/b9ak-lam80-seed3_checkpoint_evals.png)
![b9al-lam80-seed4](../runs/b9al-lam80-seed4.png)
![b9al-lam80-seed4 stage B](../runs/b9al-lam80-seed4_checkpoint_evals.png)

**λ 0.85** — `b9am`-`b9ap`:

![b9am-lam85-seed1](../runs/b9am-lam85-seed1.png)
![b9am-lam85-seed1 stage B](../runs/b9am-lam85-seed1_checkpoint_evals.png)
![b9an-lam85-seed2](../runs/b9an-lam85-seed2.png)
![b9an-lam85-seed2 stage B](../runs/b9an-lam85-seed2_checkpoint_evals.png)
![b9ao-lam85-seed3](../runs/b9ao-lam85-seed3.png)
![b9ao-lam85-seed3 stage B](../runs/b9ao-lam85-seed3_checkpoint_evals.png)
![b9ap-lam85-seed4](../runs/b9ap-lam85-seed4.png)
![b9ap-lam85-seed4 stage B](../runs/b9ap-lam85-seed4_checkpoint_evals.png)

**λ 0.9** — `b9aq`-`b9at`:

![b9aq-lam90-seed1](../runs/b9aq-lam90-seed1.png)
![b9aq-lam90-seed1 stage B](../runs/b9aq-lam90-seed1_checkpoint_evals.png)
![b9ar-lam90-seed2](../runs/b9ar-lam90-seed2.png)
![b9ar-lam90-seed2 stage B](../runs/b9ar-lam90-seed2_checkpoint_evals.png)
![b9as-lam90-seed3](../runs/b9as-lam90-seed3.png)
![b9as-lam90-seed3 stage B](../runs/b9as-lam90-seed3_checkpoint_evals.png)
![b9at-lam90-seed4](../runs/b9at-lam90-seed4.png)
![b9at-lam90-seed4 stage B](../runs/b9at-lam90-seed4_checkpoint_evals.png)

**λ 0.91** — `b9au`-`b9ax`:

![b9au-lam91-seed1](../runs/b9au-lam91-seed1.png)
![b9au-lam91-seed1 stage B](../runs/b9au-lam91-seed1_checkpoint_evals.png)
![b9av-lam91-seed2](../runs/b9av-lam91-seed2.png)
![b9av-lam91-seed2 stage B](../runs/b9av-lam91-seed2_checkpoint_evals.png)
![b9aw-lam91-seed3](../runs/b9aw-lam91-seed3.png)
![b9aw-lam91-seed3 stage B](../runs/b9aw-lam91-seed3_checkpoint_evals.png)
![b9ax-lam91-seed4](../runs/b9ax-lam91-seed4.png)
![b9ax-lam91-seed4 stage B](../runs/b9ax-lam91-seed4_checkpoint_evals.png)

**λ 0.92** — `b9ay`-`b9bb`:

![b9ay-lam92-seed1](../runs/b9ay-lam92-seed1.png)
![b9ay-lam92-seed1 stage B](../runs/b9ay-lam92-seed1_checkpoint_evals.png)
![b9az-lam92-seed2](../runs/b9az-lam92-seed2.png)
![b9az-lam92-seed2 stage B](../runs/b9az-lam92-seed2_checkpoint_evals.png)
![b9ba-lam92-seed3](../runs/b9ba-lam92-seed3.png)
![b9ba-lam92-seed3 stage B](../runs/b9ba-lam92-seed3_checkpoint_evals.png)
![b9bb-lam92-seed4](../runs/b9bb-lam92-seed4.png)
![b9bb-lam92-seed4 stage B](../runs/b9bb-lam92-seed4_checkpoint_evals.png)

**λ 0.93** — `b9bc`-`b9bf`:

![b9bc-lam93-seed1](../runs/b9bc-lam93-seed1.png)
![b9bc-lam93-seed1 stage B](../runs/b9bc-lam93-seed1_checkpoint_evals.png)
![b9bd-lam93-seed2](../runs/b9bd-lam93-seed2.png)
![b9bd-lam93-seed2 stage B](../runs/b9bd-lam93-seed2_checkpoint_evals.png)
![b9be-lam93-seed3](../runs/b9be-lam93-seed3.png)
![b9be-lam93-seed3 stage B](../runs/b9be-lam93-seed3_checkpoint_evals.png)
![b9bf-lam93-seed4](../runs/b9bf-lam93-seed4.png)
![b9bf-lam93-seed4 stage B](../runs/b9bf-lam93-seed4_checkpoint_evals.png)

**λ 0.94** — `b9bg`-`b9bj`:

![b9bg-lam94-seed1](../runs/b9bg-lam94-seed1.png)
![b9bg-lam94-seed1 stage B](../runs/b9bg-lam94-seed1_checkpoint_evals.png)
![b9bh-lam94-seed2](../runs/b9bh-lam94-seed2.png)
![b9bh-lam94-seed2 stage B](../runs/b9bh-lam94-seed2_checkpoint_evals.png)
![b9bi-lam94-seed3](../runs/b9bi-lam94-seed3.png)
![b9bi-lam94-seed3 stage B](../runs/b9bi-lam94-seed3_checkpoint_evals.png)
![b9bj-lam94-seed4](../runs/b9bj-lam94-seed4.png)
![b9bj-lam94-seed4 stage B](../runs/b9bj-lam94-seed4_checkpoint_evals.png)

**λ 0.95** — `b9bk`-`b9bn`:

![b9bk-lam95-seed1](../runs/b9bk-lam95-seed1.png)
![b9bk-lam95-seed1 stage B](../runs/b9bk-lam95-seed1_checkpoint_evals.png)
![b9bl-lam95-seed2](../runs/b9bl-lam95-seed2.png)
![b9bl-lam95-seed2 stage B](../runs/b9bl-lam95-seed2_checkpoint_evals.png)
![b9bm-lam95-seed3](../runs/b9bm-lam95-seed3.png)
![b9bm-lam95-seed3 stage B](../runs/b9bm-lam95-seed3_checkpoint_evals.png)
![b9bn-lam95-seed4](../runs/b9bn-lam95-seed4.png)
![b9bn-lam95-seed4 stage B](../runs/b9bn-lam95-seed4_checkpoint_evals.png)

**λ 0.96** — `b9bo`-`b9br`:

![b9bo-lam96-seed1](../runs/b9bo-lam96-seed1.png)
![b9bo-lam96-seed1 stage B](../runs/b9bo-lam96-seed1_checkpoint_evals.png)
![b9bp-lam96-seed2](../runs/b9bp-lam96-seed2.png)
![b9bp-lam96-seed2 stage B](../runs/b9bp-lam96-seed2_checkpoint_evals.png)
![b9bq-lam96-seed3](../runs/b9bq-lam96-seed3.png)
![b9bq-lam96-seed3 stage B](../runs/b9bq-lam96-seed3_checkpoint_evals.png)
![b9br-lam96-seed4](../runs/b9br-lam96-seed4.png)
![b9br-lam96-seed4 stage B](../runs/b9br-lam96-seed4_checkpoint_evals.png)

**λ 0.97** — `b9bs`-`b9bv`:

![b9bs-lam97-seed1](../runs/b9bs-lam97-seed1.png)
![b9bs-lam97-seed1 stage B](../runs/b9bs-lam97-seed1_checkpoint_evals.png)
![b9bt-lam97-seed2](../runs/b9bt-lam97-seed2.png)
![b9bt-lam97-seed2 stage B](../runs/b9bt-lam97-seed2_checkpoint_evals.png)
![b9bu-lam97-seed3](../runs/b9bu-lam97-seed3.png)
![b9bu-lam97-seed3 stage B](../runs/b9bu-lam97-seed3_checkpoint_evals.png)
![b9bv-lam97-seed4](../runs/b9bv-lam97-seed4.png)
![b9bv-lam97-seed4 stage B](../runs/b9bv-lam97-seed4_checkpoint_evals.png)

**λ 0.99** — `b9bw`-`b9bz`:

![b9bw-lam99-seed1](../runs/b9bw-lam99-seed1.png)
![b9bw-lam99-seed1 stage B](../runs/b9bw-lam99-seed1_checkpoint_evals.png)
![b9bx-lam99-seed2](../runs/b9bx-lam99-seed2.png)
![b9bx-lam99-seed2 stage B](../runs/b9bx-lam99-seed2_checkpoint_evals.png)
![b9by-lam99-seed3](../runs/b9by-lam99-seed3.png)
![b9by-lam99-seed3 stage B](../runs/b9by-lam99-seed3_checkpoint_evals.png)
![b9bz-lam99-seed4](../runs/b9bz-lam99-seed4.png)
![b9bz-lam99-seed4 stage B](../runs/b9bz-lam99-seed4_checkpoint_evals.png)

**λ 0.995** — `b9ca`-`b9cd`:

![b9ca-lam995-seed1](../runs/b9ca-lam995-seed1.png)
![b9ca-lam995-seed1 stage B](../runs/b9ca-lam995-seed1_checkpoint_evals.png)
![b9cb-lam995-seed2](../runs/b9cb-lam995-seed2.png)
![b9cb-lam995-seed2 stage B](../runs/b9cb-lam995-seed2_checkpoint_evals.png)
![b9cc-lam995-seed3](../runs/b9cc-lam995-seed3.png)
![b9cc-lam995-seed3 stage B](../runs/b9cc-lam995-seed3_checkpoint_evals.png)
![b9cd-lam995-seed4](../runs/b9cd-lam995-seed4.png)
![b9cd-lam995-seed4 stage B](../runs/b9cd-lam995-seed4_checkpoint_evals.png)

**λ 0.999** — `b9ce`-`b9ch`:

![b9ce-lam999-seed1](../runs/b9ce-lam999-seed1.png)
![b9ce-lam999-seed1 stage B](../runs/b9ce-lam999-seed1_checkpoint_evals.png)
![b9cf-lam999-seed2](../runs/b9cf-lam999-seed2.png)
![b9cf-lam999-seed2 stage B](../runs/b9cf-lam999-seed2_checkpoint_evals.png)
![b9cg-lam999-seed3](../runs/b9cg-lam999-seed3.png)
![b9cg-lam999-seed3 stage B](../runs/b9cg-lam999-seed3_checkpoint_evals.png)
![b9ch-lam999-seed4](../runs/b9ch-lam999-seed4.png)
![b9ch-lam999-seed4 stage B](../runs/b9ch-lam999-seed4_checkpoint_evals.png)

**λ 1.00** — `b9ci`-`b9cl`:

![b9ci-lam100-seed1](../runs/b9ci-lam100-seed1.png)
![b9ci-lam100-seed1 stage B](../runs/b9ci-lam100-seed1_checkpoint_evals.png)
![b9cj-lam100-seed2](../runs/b9cj-lam100-seed2.png)
![b9cj-lam100-seed2 stage B](../runs/b9cj-lam100-seed2_checkpoint_evals.png)
![b9ck-lam100-seed3](../runs/b9ck-lam100-seed3.png)
![b9ck-lam100-seed3 stage B](../runs/b9ck-lam100-seed3_checkpoint_evals.png)
![b9cl-lam100-seed4](../runs/b9cl-lam100-seed4.png)
![b9cl-lam100-seed4 stage B](../runs/b9cl-lam100-seed4_checkpoint_evals.png)

## Batch b8 — the stability knobs, 4 knobs x 4 seeds, 100M each

**Both waves closed 2026-09-01**, stage A and stage B for all sixteen arms.

The batch exists to fix b4's collapse, so **read these on the drawdown axis** — and then read the
right-hand column, which is why the batch failed anyway. Drawdown is the median share of
post-competence stage-A evals below 50% perfect; density is the ≥98%/500 stage-B rate, both against b4
as the control truncated to the same 100M cap:

| arms | knob | drawdown < 50% | ≥98%/500 |
|---|---|---:|---:|
| `b8m`-`b8p` | λ 0.95 | **2.2%** | **2.3%** |
| `b8e`-`b8h` | entropy 0.01 → 0.001 | 3.5% | 5.0% |
| `b8a`-`b8d` | entropy 0.003 | 3.7% | 4.3% |
| `b8i`-`b8l` | `target_KL` 0.02 | 5.9% | **6.0%** |
| `b4a`-`b4h` | **control** | 8.4% | 5.7% |

**All four knobs cut the drawdown and not one of them beat the control on record density** — and the
two columns run *opposite* to each other, λ 0.95 steadying the curve most while banking the fewest
records. Stage A measures the argmax, so what these curves show falling is the deployed policy, not
eval noise. The trade-off and its caveats are in [`findings.md`](findings.md); per-arm numbers and the
5,000-episode re-measurement in [`results.md`](results.md).

**Both of the never-exercised knobs did fire**, which is what their smoke tests were for.
`target_KL` 0.02 stopped the epoch loop on **1.9-3.3%** of recorded updates per arm with `epochs_run`
median still 8 — binding on the tail and not the body, which is what it was set for — and the anneal
ran 0.0100 → 0.0010 and completed exactly at the cap.

![b8a-ent003-seed1](../runs/b8a-ent003-seed1.png)
![b8a stage B](../runs/b8a-ent003-seed1_checkpoint_evals.png)
![b8b-ent003-seed2](../runs/b8b-ent003-seed2.png)
![b8b stage B](../runs/b8b-ent003-seed2_checkpoint_evals.png)
![b8c-ent003-seed3](../runs/b8c-ent003-seed3.png)
![b8c stage B](../runs/b8c-ent003-seed3_checkpoint_evals.png)
![b8d-ent003-seed4](../runs/b8d-ent003-seed4.png)
![b8d stage B](../runs/b8d-ent003-seed4_checkpoint_evals.png)
![b8e-entanneal-seed1](../runs/b8e-entanneal-seed1.png)
![b8e stage B](../runs/b8e-entanneal-seed1_checkpoint_evals.png)
![b8f-entanneal-seed2](../runs/b8f-entanneal-seed2.png)
![b8f stage B](../runs/b8f-entanneal-seed2_checkpoint_evals.png)
![b8g-entanneal-seed3](../runs/b8g-entanneal-seed3.png)
![b8g stage B](../runs/b8g-entanneal-seed3_checkpoint_evals.png)
![b8h-entanneal-seed4](../runs/b8h-entanneal-seed4.png)
![b8h stage B](../runs/b8h-entanneal-seed4_checkpoint_evals.png)

**Wave 2:**

![b8i-kl02-seed1](../runs/b8i-kl02-seed1.png)
![b8i stage B](../runs/b8i-kl02-seed1_checkpoint_evals.png)
![b8j-kl02-seed2](../runs/b8j-kl02-seed2.png)
![b8j stage B](../runs/b8j-kl02-seed2_checkpoint_evals.png)
![b8k-kl02-seed3](../runs/b8k-kl02-seed3.png)
![b8k stage B](../runs/b8k-kl02-seed3_checkpoint_evals.png)
![b8l-kl02-seed4](../runs/b8l-kl02-seed4.png)
![b8l stage B](../runs/b8l-kl02-seed4_checkpoint_evals.png)
![b8m-lam95-seed1](../runs/b8m-lam95-seed1.png)
![b8m stage B](../runs/b8m-lam95-seed1_checkpoint_evals.png)
![b8n-lam95-seed2](../runs/b8n-lam95-seed2.png)
![b8n stage B](../runs/b8n-lam95-seed2_checkpoint_evals.png)
![b8o-lam95-seed3](../runs/b8o-lam95-seed3.png)
![b8o stage B](../runs/b8o-lam95-seed3_checkpoint_evals.png)
![b8p-lam95-seed4](../runs/b8p-lam95-seed4.png)
![b8p stage B](../runs/b8p-lam95-seed4_checkpoint_evals.png)

## Batch b7 — the fc-layout sweep, 8 layouts x 4 seeds, 50M each

Closed 2026-09-01, all four waves. Pooled **10.9%** of stage-B rows in the ≥98%/500 record region
over 28,006 rows; the winner is `fc (320,)` at **17.3%** and the loser `fc (400,200)` at 5.1%. Numbers
and the exact tests in [`results.md`](results.md), the reading in [`findings.md`](findings.md).

| wave | layouts | arms | ≥98%/500 |
|---:|---|---|---|
| 1 | `fc 320`, `fc 200,100` | `b7aa`-`b7ad`, `b7ae`-`b7ah` | **17.3%**, 11.8% |
| 2 | `fc 300,100`, `fc 400,200` | `b7ai`-`b7al`, `b7am`-`b7ap` | 6.8%, 5.1% |
| 3 | `fc 160,160`, `fc 100,100` | `b7aq`-`b7at`, `b7au`-`b7ax` | 8.3%, 11.3% |
| 4 | `fc 100,200,100`, `fc 200,100,50` | `b7ay`-`b7bb`, `b7bc`-`b7bf` | 11.6%, 10.8% |

**What to look at, given 64 panels.** Compare wave 1's two layouts first — that is the whole result —
then `fc (400,200)` in wave 2 for what too much width looks like: fewer screened checkpoints and a
thinner record region, not a visibly worse curve. Stage-A progress chart then stage-B pass, per arm,
in wave order.

![b7ae-fc200x100-seed1](../runs/b7ae-fc200x100-seed1.png)
![b7ae stage B](../runs/b7ae-fc200x100-seed1_checkpoint_evals.png)
![b7af-fc200x100-seed2](../runs/b7af-fc200x100-seed2.png)
![b7af stage B](../runs/b7af-fc200x100-seed2_checkpoint_evals.png)
![b7ag-fc200x100-seed3](../runs/b7ag-fc200x100-seed3.png)
![b7ag stage B](../runs/b7ag-fc200x100-seed3_checkpoint_evals.png)
![b7ah-fc200x100-seed4](../runs/b7ah-fc200x100-seed4.png)
![b7ah stage B](../runs/b7ah-fc200x100-seed4_checkpoint_evals.png)
![b7aa-fc320-seed1](../runs/b7aa-fc320-seed1.png)
![b7aa stage B](../runs/b7aa-fc320-seed1_checkpoint_evals.png)
![b7ab-fc320-seed2](../runs/b7ab-fc320-seed2.png)
![b7ab stage B](../runs/b7ab-fc320-seed2_checkpoint_evals.png)
![b7ac-fc320-seed3](../runs/b7ac-fc320-seed3.png)
![b7ac stage B](../runs/b7ac-fc320-seed3_checkpoint_evals.png)
![b7ad-fc320-seed4](../runs/b7ad-fc320-seed4.png)
![b7ad stage B](../runs/b7ad-fc320-seed4_checkpoint_evals.png)
![b7ai-fc300x100-seed1](../runs/b7ai-fc300x100-seed1.png)
![b7ai stage B](../runs/b7ai-fc300x100-seed1_checkpoint_evals.png)
![b7aj-fc300x100-seed2](../runs/b7aj-fc300x100-seed2.png)
![b7aj stage B](../runs/b7aj-fc300x100-seed2_checkpoint_evals.png)
![b7ak-fc300x100-seed3](../runs/b7ak-fc300x100-seed3.png)
![b7ak stage B](../runs/b7ak-fc300x100-seed3_checkpoint_evals.png)
![b7al-fc300x100-seed4](../runs/b7al-fc300x100-seed4.png)
![b7al stage B](../runs/b7al-fc300x100-seed4_checkpoint_evals.png)
![b7am-fc400x200-seed1](../runs/b7am-fc400x200-seed1.png)
![b7am stage B](../runs/b7am-fc400x200-seed1_checkpoint_evals.png)
![b7an-fc400x200-seed2](../runs/b7an-fc400x200-seed2.png)
![b7an stage B](../runs/b7an-fc400x200-seed2_checkpoint_evals.png)
![b7ao-fc400x200-seed3](../runs/b7ao-fc400x200-seed3.png)
![b7ao stage B](../runs/b7ao-fc400x200-seed3_checkpoint_evals.png)
![b7ap-fc400x200-seed4](../runs/b7ap-fc400x200-seed4.png)
![b7ap stage B](../runs/b7ap-fc400x200-seed4_checkpoint_evals.png)
![b7au-fc100x100-seed1](../runs/b7au-fc100x100-seed1.png)
![b7au stage B](../runs/b7au-fc100x100-seed1_checkpoint_evals.png)
![b7av-fc100x100-seed2](../runs/b7av-fc100x100-seed2.png)
![b7av stage B](../runs/b7av-fc100x100-seed2_checkpoint_evals.png)
![b7aw-fc100x100-seed3](../runs/b7aw-fc100x100-seed3.png)
![b7aw stage B](../runs/b7aw-fc100x100-seed3_checkpoint_evals.png)
![b7ax-fc100x100-seed4](../runs/b7ax-fc100x100-seed4.png)
![b7ax stage B](../runs/b7ax-fc100x100-seed4_checkpoint_evals.png)
![b7aq-fc160x160-seed1](../runs/b7aq-fc160x160-seed1.png)
![b7aq stage B](../runs/b7aq-fc160x160-seed1_checkpoint_evals.png)
![b7ar-fc160x160-seed2](../runs/b7ar-fc160x160-seed2.png)
![b7ar stage B](../runs/b7ar-fc160x160-seed2_checkpoint_evals.png)
![b7as-fc160x160-seed3](../runs/b7as-fc160x160-seed3.png)
![b7as stage B](../runs/b7as-fc160x160-seed3_checkpoint_evals.png)
![b7at-fc160x160-seed4](../runs/b7at-fc160x160-seed4.png)
![b7at stage B](../runs/b7at-fc160x160-seed4_checkpoint_evals.png)
![b7ay-fc100x200x100-seed1](../runs/b7ay-fc100x200x100-seed1.png)
![b7ay stage B](../runs/b7ay-fc100x200x100-seed1_checkpoint_evals.png)
![b7az-fc100x200x100-seed2](../runs/b7az-fc100x200x100-seed2.png)
![b7az stage B](../runs/b7az-fc100x200x100-seed2_checkpoint_evals.png)
![b7ba-fc100x200x100-seed3](../runs/b7ba-fc100x200x100-seed3.png)
![b7ba stage B](../runs/b7ba-fc100x200x100-seed3_checkpoint_evals.png)
![b7bb-fc100x200x100-seed4](../runs/b7bb-fc100x200x100-seed4.png)
![b7bb stage B](../runs/b7bb-fc100x200x100-seed4_checkpoint_evals.png)
![b7bc-fc200x100x50-seed1](../runs/b7bc-fc200x100x50-seed1.png)
![b7bc stage B](../runs/b7bc-fc200x100x50-seed1_checkpoint_evals.png)
![b7bd-fc200x100x50-seed2](../runs/b7bd-fc200x100x50-seed2.png)
![b7bd stage B](../runs/b7bd-fc200x100x50-seed2_checkpoint_evals.png)
![b7be-fc200x100x50-seed3](../runs/b7be-fc200x100x50-seed3.png)
![b7be stage B](../runs/b7be-fc200x100x50-seed3_checkpoint_evals.png)
![b7bf-fc200x100x50-seed4](../runs/b7bf-fc200x100x50-seed4.png)
![b7bf stage B](../runs/b7bf-fc200x100x50-seed4_checkpoint_evals.png)

## Batch b4 — `fc (200,100)` + 8 epochs, seeds 1-8

Closed 2026-08-31, charts imported from the `results` branch 2026-09-01 and **redrawn**, because the
published PNGs carried the pre-rename `p1` titles. Pooled **7.3%** in the record region, the weakest
of the three 8-seed batches ([`results.md`](results.md)).

**Read these for the drawdowns, which is what makes b4 different.** The red trace repeatedly falls
from ~95% to near zero, and it is the *greedy* policy doing it — stage A measures the argmax, not a
sample. Past its competence onset b4 spends a median **9.1%** of its evals below 50% perfect, against
0.7% for b6 and 0.0% for b5.

**‡ Corrected 2026-09-01: it is not a "late-run" effect, it is an effect with an onset.** Truncated to
a matched 611-eval horizon (~10M) the three batches are indistinguishable — but b4's rate is **8.5% in
its first 100M and 9.0% in its second**, so it is fully developed well before the cap and the second
half adds nothing to it. The earlier reading, that it "develops over the 200M", was drawn from the
short-horizon comparison alone and did not check the halves. This is what sets [batch b8](runs.md)'s
budget at 100M. Where the second 100M *does* matter is the record region: **65%** of b4's ≥98%/500
rows land after 100M, so a density comparison against b4 must truncate b4 to the same cap.

![b4a-fc200x100ep8-seed1](../runs/b4a-fc200x100ep8-seed1.png)
![b4a stage B](../runs/b4a-fc200x100ep8-seed1_checkpoint_evals.png)
![b4b-fc200x100ep8-seed2](../runs/b4b-fc200x100ep8-seed2.png)
![b4b stage B](../runs/b4b-fc200x100ep8-seed2_checkpoint_evals.png)
![b4c-fc200x100ep8-seed3](../runs/b4c-fc200x100ep8-seed3.png)
![b4c stage B](../runs/b4c-fc200x100ep8-seed3_checkpoint_evals.png)
![b4d-fc200x100ep8-seed4](../runs/b4d-fc200x100ep8-seed4.png)
![b4d stage B](../runs/b4d-fc200x100ep8-seed4_checkpoint_evals.png)
![b4e-fc200x100ep8-seed5](../runs/b4e-fc200x100ep8-seed5.png)
![b4e stage B](../runs/b4e-fc200x100ep8-seed5_checkpoint_evals.png)
![b4f-fc200x100ep8-seed6](../runs/b4f-fc200x100ep8-seed6.png)
![b4f stage B](../runs/b4f-fc200x100ep8-seed6_checkpoint_evals.png)
![b4g-fc200x100ep8-seed7](../runs/b4g-fc200x100ep8-seed7.png)
![b4g stage B](../runs/b4g-fc200x100ep8-seed7_checkpoint_evals.png)
![b4h-fc200x100ep8-seed8](../runs/b4h-fc200x100ep8-seed8.png)
![b4h stage B](../runs/b4h-fc200x100ep8-seed8_checkpoint_evals.png)

## Batch b6 — `fc (200,100)`, 4 epochs, seeds 1-8

Closed 2026-08-30. Pooled **12.8%** of stage-B rows in the >=98%/500 record region; best rows
99.4-99.8. Stage-A progress chart then stage-B pass, per arm. Numbers and the b5 comparison —
including why the two batches are **not** a clean network-shape test — in [`results.md`](results.md).

![b6a-fc200x100-seed1](../runs/b6a-fc200x100-seed1.png)
![b6a stage B](../runs/b6a-fc200x100-seed1_checkpoint_evals.png)
![b6b-fc200x100-seed2](../runs/b6b-fc200x100-seed2.png)
![b6b stage B](../runs/b6b-fc200x100-seed2_checkpoint_evals.png)
![b6c-fc200x100-seed3](../runs/b6c-fc200x100-seed3.png)
![b6c stage B](../runs/b6c-fc200x100-seed3_checkpoint_evals.png)
![b6d-fc200x100-seed4](../runs/b6d-fc200x100-seed4.png)
![b6d stage B](../runs/b6d-fc200x100-seed4_checkpoint_evals.png)
![b6e-fc200x100-seed5](../runs/b6e-fc200x100-seed5.png)
![b6e stage B](../runs/b6e-fc200x100-seed5_checkpoint_evals.png)
![b6f-fc200x100-seed6](../runs/b6f-fc200x100-seed6.png)
![b6f stage B](../runs/b6f-fc200x100-seed6_checkpoint_evals.png)
![b6g-fc200x100-seed7](../runs/b6g-fc200x100-seed7.png)
![b6g stage B](../runs/b6g-fc200x100-seed7_checkpoint_evals.png)
![b6h-fc200x100-seed8](../runs/b6h-fc200x100-seed8.png)
![b6h stage B](../runs/b6h-fc200x100-seed8_checkpoint_evals.png)

## Batch b5 — `fc (320,)`, 8 epochs, seeds 1-8

Closed 2026-08-30. Pooled **9.6%** in the record region, and the single best row in either
batch — **100.0%/500** at b5b/184M.

![b5a-ep8-seed1](../runs/b5a-ep8-seed1.png)
![b5a stage B](../runs/b5a-ep8-seed1_checkpoint_evals.png)
![b5b-ep8-seed2](../runs/b5b-ep8-seed2.png)
![b5b stage B](../runs/b5b-ep8-seed2_checkpoint_evals.png)
![b5c-ep8-seed3](../runs/b5c-ep8-seed3.png)
![b5c stage B](../runs/b5c-ep8-seed3_checkpoint_evals.png)
![b5d-ep8-seed4](../runs/b5d-ep8-seed4.png)
![b5d stage B](../runs/b5d-ep8-seed4_checkpoint_evals.png)
![b5e-ep8-seed5](../runs/b5e-ep8-seed5.png)
![b5e stage B](../runs/b5e-ep8-seed5_checkpoint_evals.png)
![b5f-ep8-seed6](../runs/b5f-ep8-seed6.png)
![b5f stage B](../runs/b5f-ep8-seed6_checkpoint_evals.png)
![b5g-ep8-seed7](../runs/b5g-ep8-seed7.png)
![b5g stage B](../runs/b5g-ep8-seed7_checkpoint_evals.png)
![b5h-ep8-seed8](../runs/b5h-ep8-seed8.png)
![b5h stage B](../runs/b5h-ep8-seed8_checkpoint_evals.png)

## Arms

Every arm gets a `### <policy> — <what it changes>` section with a stats line, a short reading, and
its image. **Newest first, in this section and in the batch sections above it** — a new batch goes
directly under `Watching them live`, and `Imported policies` stays at the bottom. **Images are linked straight from `../runs/`** — there is no copy step and no separate
chart directory to keep in sync. That duplication is what snek2 needed a `refresh_charts.sh` and a
completeness-check snippet for, and it still drifted to 12 undocumented arms.

```markdown
### b1a-example — what this arm changes

step 2.00M · peak trailing 88.4 · best30 41.2% · sef 0.31 · best 500-ep 97.8% · ≥98%/500 x 0

One or two sentences: what the curve does, and what it means for the hypothesis.

![b1a-example](../runs/b1a-example.png)
```

### b3 — the PPO tuning sweep, 15 arms, one knob each

**Read these as a set, and read them at 10M rather than at 3M.** Every arm is seed 1 on b2's reward
function, one knob off a reference of lr 3e-4 / γ 0.99 / λ 0.98 / entropy 0.01 / fc 320 / 128x128
rollout / 4 epochs / minibatch 256. Nine of the fifteen finished inside **0.8 pp** of each other, so
the curves matter more than the ranking: what to look at is *where each one turns up*, not which one
ends highest.

**The two the batch exists for.** `b3a` is the reference and `b3g`/`b3e` are the two that were 6th and
7th at 3M and 1st and 2nd at 10M — put those three side by side and the cap-inversion finding is
visible as a shape rather than a table
([`findings.md`](findings.md)).

step 10.01M · best30 96.4 - 97.2 across nine arms · sd30 1.8 - 3.2 · best stage B **98.6% / 500**,
re-measured **96.6% / 3,000**

The laptop half — λ, entropy and the learning-rate bracket:

![b3a-lr3e4-g99](../runs/b3a-lr3e4-g99.png)
![b3g-ent003](../runs/b3g-ent003.png)
![b3e-lam95](../runs/b3e-lam95.png)
![b3j-lr5e4](../runs/b3j-lr5e4.png)
![b3i-lr1e4](../runs/b3i-lr1e4.png)
![b3f-lam100](../runs/b3f-lam100.png)
![b3h-ent03](../runs/b3h-ent03.png)

And the three arms that stopped at the 3M cap, kept because they are what the inversion is measured
against — the learning-rate extremes and γ 0.9975:

![b3b-lr1e3-g99](../runs/b3b-lr1e3-g99.png)
![b3c-lr3e3-g99](../runs/b3c-lr3e3-g99.png)
![b3d-lr3e4-g9975](../runs/b3d-lr3e4-g9975.png)

**The desktop half's eight charts — the four fc shapes, γ 0.995, and the three update-shape knobs —
arrive on the `results` branch at close-out**, and this section gets their `![]` lines in the same
pass. They are the half that found the one axis that moved: `b3q-ep8` at 97.2 and `b3r-mb1024` at
89.7 are the two ends of it.

### ppo-smoke — the phase-6b PPO gate arm, untuned defaults

Not a batch arm, and not seed-matched to anything: it exists to show `ppo/` learns. Read it against
b1 at a *matched transition count* (b1's step x 6), not against b1's endpoint.

step 508k transitions · trailing score 62.5 · avg score 79.55/500 eps · perfect 1.2%/500 · ev 0.90 ·
entropy 1.086 → 0.27 · clip fraction 0.03

Score climbs monotonically to ~80 and then flattens while the perfect rate sits near 1% — the shape
[`../plans/ppo.md`](../plans/ppo.md) §8 predicted for a short GAE horizon against a win ~950 moves
away, though at this budget it is equally just an untuned learning rate. `clip_fraction` 0.03 says the
update is not being constrained, so the rate is the first thing b3 moves.

![ppo-smoke](../runs/ppo-smoke.png)

### b1 — the DDQN baseline at every default, seeds 1-4

The batch changes nothing but `SNEK_SEED`, so these four curves are this codebase's noise floor as
much as they are a result. Read them together.

step 3.00M · trailing score 92.3-94.2 · peak best30 42.1 / 58.3 / 56.7 / **81.9%** · ≥95/100: **0**

**Every one of the four is still climbing at its cap, and that is the finding.** b1a goes ~20% at
500k to ~40% at 3M; b1d goes 0% to ~80% with its highest band in the final 500k. Neither plateaued,
so 3M steps measures how fast this config climbs and not what it converges to. The seed spread of
**39.8 pp** at this horizon is four times what n=4 can resolve.

The early spike and crash all four share is a separate, real feature of the task rather than a port
artefact: snek2's `b13d-shieldseed4` scored 91.2 with **80% perfect at step 20,000** and finished a
3.5M-step arm at 70%. Early competence here is cheap and unstable.

The weakest and the strongest seed, which between them are the whole story — the same config, the
same cap, 42.1% against 81.9%:

![b1a-baseline-seed1](../runs/b1a-baseline-seed1.png)
![b1d-baseline-seed4](../runs/b1d-baseline-seed4.png)

The middle two sit between them and add nothing a reader needs:

![b1b-baseline-seed2](../runs/b1b-baseline-seed2.png)
![b1c-baseline-seed3](../runs/b1c-baseline-seed3.png)

**Refresh this file in the same pass as any doc edit or progress update**, whether or not the arms
have finished — a running batch with no chart entry is a bug, not a "wait until it closes" state.

`../charts/` holds only the one-off diagnostic figures a finding refers to, never per-arm charts.

## Imported policies

Not arms — snek2 checkpoints converted to torch and measured by snek3, so these charts describe
**snek3's environment and measurement** rather than snek3 as a learner. See
[`results.md`](results.md).

### b45a-import — the phase-2 A/B, every checkpoint of snek2's `b45a-lowlr8-b29b`

3,222 rows · 100 ep each · pooled 97.29% · 1,576 rows ≥98% · widest ≥98% run 9

The arm's real shape only exists as the trailing mean: a single 100-episode row is quantised to whole
percent and carries 1.6 pp of noise, which is why the points form bands. Read the dark line — `b45a`
peaks near 98% between 1.8M and 2.4M and sags to ~96.5% near 3.9M. The green rug marks which
checkpoints cleared 98%.

![b45a-import seed 0](../runs/b45a-import_checkpoint_evals_ab3222.png)

The second seed reproduces the **shape** — high through ~2.6M, lower after ~3.0M — and not the
individual dips, which is exactly right: a 40-row trailing mean has 0.26 pp of noise, so a ±0.5 pp
wiggle is two standard deviations and should not repeat. This pass is also how the 100/100-count
discrepancy was settled ([`findings.md`](findings.md)).

![b45a-import seed 1](../runs/b45a-import_checkpoint_evals_ab3222seed1.png)
