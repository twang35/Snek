# Charts — archived batch sections

Chart sections retired from [`../charts.md`](../charts.md), newest-retired first. **History only —
do not read into context during normal work**, same as everything else in this folder.

`charts.md` keeps the **six most recent batches**. When a batch is stopped and added there, the
oldest section moves here, so the live file stays short enough to actually read. The PNGs are *not*
moved: every image stays in `../charts/`, so the captions here still render.

| retired | batch | why it went |
|---|---|---|
| 2026-08-08 | 12 | batch 18 landed; batch 12 became the seventh-newest |

---

## Batch 12 — the epsilon rewrite, and the deadlock it found

Four seeds of batch 11's config plus the two-phase epsilon schedule, **stopped at ~1M of a
planned 2.5M** because all four failed the same way: epsilon pinned at the refinement ceiling
0.05 and the perfect rate never left 0. Read these four charts as one shape repeated four times —
a fast climb to 81-87 trailing between 214k and 479k, then a slow decay to 53-63 that never
recovers. Both numbers are greedy evals, so that decay is the learned policy getting worse, not
an exploration tax on the measurement.

**`strong_eval_fraction` is 0.0% in all four arms**, against 25.2 / 30.5 / 0.0 / 8.2% for batch 11
at the same 1M steps. The mechanism, the fix, and the two wrong turns taken diagnosing it are in
[`completedRuns.md`](../completedRuns.md#-the-new-schedule-deadlocks-all-four-arms-are-failing-44-at-1m-steps). These
arms are kept as the measured cost of sitting at epsilon 0.05: not a wasted batch, a negative
result with four seeds behind it.

### b12s-shield05seed1 — the exploration shield at handover 0.05, seed 1

![b12s](../charts/b12s-shield05seed1.png)

Step 0.43M (stopped) · trailing 83.1 at stop · best 30-eval perfect 0.3% · max single eval 10% · not measured

**The arm that moved the handover.** A verification run, `SEED=1` so it pairs with `b12a`, with the
one-step exploration shield on and the handover still at 0.05. It **fixed the decay** — `b12a` fell
83.8 → 74.2 between 200k and 400k while this one was still rising — and **did not fix perfect
games**: 2 perfect-game evals in 431, plateauing at trailing ~83 where the perfect rate is ~0,
improving at 4.7 points per 100k against `b11a`'s 11.1.

Kept because it is the whole argument for dropping the handover to 0.0125: a one-step mask prevents
blunders but not self-trapping, so the collect policy still never finishes a board and the buffer
never contains the last ten food. Read against `b12a` below and `b11a` above.

### b12a-eps002seed1 — two-phase epsilon, seed 1

![b12a](../charts/b12a-eps002seed1.png)

Step 1.12M (stopped) · peak score 89.1, peak trailing 87.02 (at 214k) · best 30-eval perfect **6.3%** (at 213k) · max single eval 40% · not measured

The best of a bad batch, and the arm that makes the decay unambiguous. It read trailing **87.0**
with 6.3% best-30 at 214k — a genuinely promising arm — then fell to 59.6 over the next 900k steps
**at exactly the same epsilon**. Same exploration rate, worse policy, so nothing about the
measurement explains it.

41 of its 1122 evals contained a perfect game, which was enough to nudge epsilon to 0.0388 at its
best and never enough to escape: the refinement phase needs 20-40% to reach the floor, and 0.05
makes that unreachable.

### b12d-eps002seed4 — two-phase epsilon, seed 4

![b12d](../charts/b12d-eps002seed4.png)

Step 1.09M (stopped) · peak score 87.2, peak trailing 86.36 (at 479k) · best 30-eval perfect 6.3% (at 29k) · max single eval 30% · not measured

The latest peak in the batch at 479k, and the only arm whose best-30 came in its first 30k steps —
during the bootstrap phase, before the ceiling took hold. Everything after is decline.

### b12c-eps002seed3 — two-phase epsilon, seed 3

![b12c](../charts/b12c-eps002seed3.png)

Step 0.98M (stopped) · peak score 85.3, peak trailing 82.46 (at 360k) · best 30-eval perfect 1.7% (at 369k) · max single eval 10% · not measured

8 perfect games in 977 evals, and a max single eval of 10% — it reached the endgame often enough
to prove the policy was not hopeless and never often enough for the schedule to notice.

### b12b-eps002seed2 — two-phase epsilon, seed 2

![b12b](../charts/b12b-eps002seed2.png)

Step 1.03M (stopped) · peak score 82.8, peak trailing 81.4 (at 259k) · **best 30-eval perfect 0.0%** · max single eval **0%** · not measured

**The cleanest demonstration of the deadlock in the project: zero perfect games in 1032 evals.**
Epsilon reached 0.05 at step 11000 and sat there for the remaining 942k steps, because the signal
that would have lowered it requires finishing a game and 3.3% random actions never let it. An arm
that peaked at 81.4 trailing was never once measured completing the board.

---
