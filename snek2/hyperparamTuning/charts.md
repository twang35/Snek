# Charts

Progress graphs, **batch 9 onward**. Per-arm numbers live in
[`completedRuns.md`](completedRuns.md); this file is images plus a short reading of each.
Batch 1-8 captions moved to [`archive/batches1-8.md`](archive/batches1-8.md) — the PNGs are all
still in `charts/`.

In every chart: **blue is average score** (food eaten, out of 95) on the left axis, **red is
perfect-game percentage** on the right. Grey dashed vertical lines mark resumes; faint red dashed
horizontals mark 20/40/60/80% on the right axis, because the perfect rate is the objective and
was unreadable against left-axis ticks.

**Newest batch first.** Within a batch, best result first.

## These are snapshots, on purpose

The images are **copies** from `snek2/runs/`, not links. The live graphs there are rewritten every
eval and would be lost if that directory were cleaned out, silently blanking this file. Refresh with
`refresh_charts.sh`, which re-copies every `runs/*.png` into `charts/` and prints each one's step.

**The script does not touch this file** — it copies images only, so a new arm gets a PNG and no
entry unless one is written by hand. That drifted once, to 12 undocumented arms across batches 5-7,
because a successful `refresh_charts.sh` looked like the charts were handled. Check both this file
and the archive, since captions now live in two places:

```
cd snek2/hyperparamTuning
ls charts/*.png | sed 's|.*/||;s|\.png||' | sort > /tmp/have
grep -ho 'charts/[a-zA-Z0-9-]*\.png' charts.md archive/batches1-8.md \
  | sed 's|charts/||;s|\.png||' | sort -u > /tmp/doc
comm -23 /tmp/have /tmp/doc   # anything listed is an undocumented arm
```

## Batch 9 — two discounts on the post-audit environment

Four arms, two seeds each of `DISCOUNT=0.995` and `0.9975`, and the first batch to run after the
2026-08-02 audit. Its own lesson was methodological: it compared two values before either had a
baseline on the environment being measured, which is why batch 10 spent all four slots on one
config instead.

### ‡ These arms are measured on a different environment

The 2026-08-02 audit changed two observation components and the reward, and every arm below the
batch-9 rows was measured before it. A batch-9 number is **not comparable** to a pre-audit one:
the same checkpoint that scored 92% on the old environment reads 73% on this one. Compare batch-9
arms to each other, and to `b8f`'s re-measured 82% if you want a pre-audit reference. The audit's
own measurements are in
[`archive/findings-superseded.md`](archive/findings-superseded.md).

All three survivors were still training when measured, at 3.4-3.6M steps, so these are mid-run
close-outs rather than final ones.

### b9a-disc9975a — `DISCOUNT=0.9975`, new env

Step 3.61M (running) · peak score **89.8** (at 3277k) · **best 30-eval perfect 56.0%** (at 1738k) · max single eval **90%** · **best measured checkpoint 65.0%** (at 1735k), pooled **54.9%** /2000

**The most consistent arm of batch 9**, and the only 0.9975 seed that survived. Its pooled 54.9%
over 20 checkpoints is well clear of both 0.995 seeds, and 18 of 20 checkpoints measured above 40%.

Its best checkpoint sits at **1735k, less than half way to its trailing peak at 3277k** — the
graph keeps climbing after the measurable quality stops improving, which is the same pattern
`b8f` showed and a reason not to read peak trailing as a proxy for peak policy.

![b9a-disc9975a progress](charts/b9a-disc9975a.png)

### b9b-disc9975b — `DISCOUNT=0.9975`, new env, dead

Step **10.47M** (stopped 2026-08-02) · peak score 72.1 (at **328k**) · best 30-eval perfect 5.0% (at 221k) · max single eval 20% · not measured

**The overrun failure the 3-3.5M stop rule exists to prevent.** It peaked at step **328k** — before
any sibling had warmed up — then declined and ran a further 10.1M steps producing nothing, with
`zero_since` at 9.92M. Nothing was watching it for eight hours overnight.

Not measured: no checkpoint came near the 60% selector floor, so a close-out would have had
nothing to evaluate.

![b9b-disc9975b progress](charts/b9b-disc9975b.png)

### b9c-disc995a — `DISCOUNT=0.995`, new env

Step 3.64M (running) · peak score 86.4 (at 2599k) · best 30-eval perfect 37.3% (at 2608k) · max single eval **80%** · **best measured checkpoint 52.0%** (at 2603k), pooled 38.0% /2000

The weakest of the three survivors on every measure, and the reason 0.995 cannot be called the
better value on this environment despite going 2 for 2 on survival. Its good region is narrow:
best and second-best are 2603k and 2626k, 23k apart.

![b9c-disc995a progress](charts/b9c-disc995a.png)

### b9d-disc995b — `DISCOUNT=0.995`, new env

Step 3.40M (running) · peak score 86.7 (at 1232k) · best 30-eval perfect 30.3% (at 1324k) · max single eval **90%** · **best measured checkpoint 70.0%** (at 2544k), pooled 42.4% /1700

**The best single checkpoint of batch 9** at 70%, and the arm that inverts the interim reading —
at 12 of 17 checkpoints its best was 49% and it looked like the weakest arm in the batch. The
remaining five checkpoints contained its top three. A partial close-out is not a small version of
a complete one.

Its trailing average peaked at **1232k** and had fallen to 49.4 by 3.4M, while its best measured
checkpoint is at 2544k — the graph and the measurement disagree about when this arm was good.

![b9d-disc995b progress](charts/b9d-disc995b.png)

---
