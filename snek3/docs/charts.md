# Charts — one graph per arm

**Batch b2 is running** — b29's record config, seeds 1-4, all four on the desktop. Its charts arrive
on the `results` branch at close-out; a live desktop arm's `runs/` files must not be committed from
here, because the box rewrites those paths every eval and a committed copy aborts its next ff-merge.

**Batch b1 closed 2026-08-29.** All four charts are committed now that the arms have stopped — while
a desktop arm is *live* its `runs/` files must not be committed from here, because the box rewrites
those paths every eval and a committed copy makes its next `git merge --ff-only` abort.

## Watching them live

**One window per box, opened by the trainings themselves**, showing every arm running there. Nothing
launches it; the first arm to start opens it and the last one to finish takes it away — and a panel
stays for the rest of the wave once it appears, so a batch with one arm left still shows all four. To
put it back
after closing it: `PYTHONPATH=. python -m tools.chart_window`. Killing it, closing it and relaunching
it are all free — no training reads it, waits on it, or reopens it ([`findings.md`](findings.md) on why
it took three attempts to get there).

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

## Arms

Every arm gets a `### <policy> — <what it changes>` section with a stats line, a short reading, and
its image. **Images are linked straight from `../runs/`** — there is no copy step and no separate
chart directory to keep in sync. That duplication is what snek2 needed a `refresh_charts.sh` and a
completeness-check snippet for, and it still drifted to 12 undocumented arms.

```markdown
### b1a-example — what this arm changes

step 2.00M · peak trailing 88.4 · best30 41.2% · sef 0.31 · best 500-ep 97.8% · ≥98%/500 x 0

One or two sentences: what the curve does, and what it means for the hypothesis.

![b1a-example](../runs/b1a-example.png)
```

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
