# Charts — one graph per arm

**Batch b1 is running.** Only `b1a`'s chart is committed: it is the laptop's own arm. **A running
desktop arm's `runs/` files must not be committed from here** — the box writes those same paths every
eval, so a committed copy makes its next `git merge --ff-only` abort and blocks every deploy. The
desktop three arrive on the `results` branch at close-out and get their images then.

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

**All four show the same early spike and crash**, which is the one thing worth reading now.
`b1b` reached **69% perfect at step 5,000** and fell to ~14% by 10,000; `b1a` scored 89 at ~5,000,
sagged to 65 by 15,000, and recovered to ~85 by 40,000. That is not a port artefact — snek2's
`b13d-shieldseed4` scored 91.2 with **80% perfect at step 20,000** and finished a 3.5M-step arm at
70%. Early competence in this task is cheap and unstable; the last few percent is what 3M steps buy.

![b1a-baseline-seed1](../runs/b1a-baseline-seed1.png)

The desktop's three at 2026-08-28 23:47, from `status.json` and their own summaries — images at
close-out:

| arm | step | trailing | recent30 | max single eval | epsilon |
|---|---:|---:|---:|---:|---:|
| `b1b-baseline-seed2` | 19,000 | 60.20 | 5.2% | 70% | 0.01109 |
| `b1c-baseline-seed3` | 19,000 | 58.32 | 0.1% | 1% | 0.01248 |
| `b1d-baseline-seed4` | 17,000 | 56.47 | 0.7% | 4% | 0.01230 |

**Refresh this file in the same pass as any doc edit or progress update**, whether or not the arms
have finished — a running batch with no chart entry is a bug, not a "wait until it closes" state.

`../charts/` holds only the one-off diagnostic figures a finding refers to, never per-arm charts.
