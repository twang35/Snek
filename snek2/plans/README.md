# Approved designs, not yet implemented

One file per proposed change that is worth writing down before it is worth building — a design the
user has read and agreed with, parked because something else is running.

Distinct from [`../hyperparamTuning/runs.md`](../hyperparamTuning/runs.md)'s standing backlog, which
is a one-line prior per idea. A file lands here when the design is settled enough that a future
session could implement it without redoing the reasoning: the exact edits, the tests, the
pre-registered success criteria, and the risks that were already argued through.

**Every plan here has a row in that backlog table**, so nobody has to know this directory exists to
find its contents. When a plan ships, mark it done in the file's status line and leave the file —
the measurement design in it is what a later replication needs.

| plan | status |
|---|---|
| [`chase-safe-reward-shaping.md`](chase-safe-reward-shaping.md) | approved 2026-08-11, **revised 2026-08-14 and ready to execute** — the hold expired (20-26 closed, both hosts idle), the control moved to batch 24, and the fork/snapshot gap is fixed in the design |
| [`distributional-c51.md`](distributional-c51.md) | **proposed 2026-08-15, awaiting review** — a categorical (C51) value head behind `SNEK_ALGO=c51`. A feasibility probe against this repo's own specs is already done and is what fixes the two design choices a naive port would get wrong (IS weights are silently dropped by `tf_agents`' C51 loss; the PER priority has to be the KL, not the cross-entropy) |
