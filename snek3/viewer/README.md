# The chart viewer and the other pages

| page | live | what |
|---|---|---|
| **chart viewer** | https://twang35.github.io/Snek/ | every arm's stage-A, stage-B, hof5000 and hof30k chart, batch by batch. Reloads itself at :00 of every ten minutes while the tab is visible |
| **Deep RL Atlas** | https://twang35.github.io/Snek/atlas | the Atari scores view and the table of published agents, `pages/atlas.html` |
| sweep page | local only: `sweep.html` beside `sweep.json` | one knob's traces across a batch |
| **b45 reads** | https://twang35.github.io/Snek/reads-b45 | one-off: the sixteen acting rules of `plans/quantile-reads.md` over Group A's best checkpoints, paired against the mean read -- a matrix, the algorithm view with the control pinned, one read across algorithms, a checkpoint drill, the table. `pages/reads-b45.html`, data inlined by a scratch reducer over the b45 pass files; rebuilt by hand when the batch closed, not by the site build |

The site is the `site` branch, built by the desktop daemon from both boxes' results feeds
(`tools/site_build.py`, root `CLAUDE.md`). To see the viewer locally,
`PYTHONPATH=. python -m tools.viewer_manifest` then open `index.html`, which reads `../runs/` directly.

Update cadence, on the wall clock: the laptop publishes its live pictures at :08, the desktop fetches
both feeds and rebuilds the site at :09, the viewer reloads at :00 (`desktop/daemon/cadence.py`).
