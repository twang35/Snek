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
| [`distributional-c51.md`](distributional-c51.md) | **proposed 2026-08-15; phases 0-2 done, phase 3 armed, phase 4 held for the user.** Support measured and settled (`[-5, 120]`, 51 atoms, 2.5 spacing); `categorical_agent.py` and its edits are written and **uncommitted pending review** (suite 24 modules / 643 tests / 0 failed, 18 mutants all killed); a c51 smoke arm trains, resumes, watches and evaluates. Two design choices come from a feasibility probe rather than from the paper: `tf_agents`' C51 loss silently drops IS weights, and the PER priority has to be the KL, not the cross-entropy |
