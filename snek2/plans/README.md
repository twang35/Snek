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
| [`vectorized-eval.md`](vectorized-eval.md) | **approved 2026-08-23, implementation started.** Eval engine only — no training-loop change, no PyTorch. Lives in `snek2/vectorized/`, tests in `snek2/tests/`, and stays as a record when `snek3/` gets a copy. **TensorFlow stays for inference**: batched TF is 3.3M rows/s against the ~176k/s the env can consume, so the network is not worth porting and no weight conversion can silently break the policy. Parity is deductive — bit-identical observations under forced food imply identical greedy episodes — with four pre-registered layers (L1 observation, L2 trajectory, L3 mutation, L4 vs b45's own non-abandoned rows). Gate: **>= 40x** per lane on a 500-episode measurement |
| [`snek3-pytorch-rewrite.md`](snek3-pytorch-rewrite.md) | **investigated 2026-08-23, measured, not built.** Where the wall clock actually goes: **99.65% of an arm's ~1.45B env steps are measurement**, the in-training eval is ~80% of training wall clock, and the flood fill is only 19% of an env step (running 1.92x per step, not 3x) so memoising it is worth ≤5%. A parity-exact vectorised numpy prototype — 100% on all 30 indices over 18,053 states, 12 mutants killed — reaches **176,830 env-steps/s**, i.e. 500 episodes in 2.3 s against 183 s, projecting **~42x per arm**. Training alone is capped at 5.8x by the optimizer unless the replay ratio changes. Not approved; step 3 (reproducing a b45-class arm) is the gate |
| [`eval-wave-controller.md`](eval-wave-controller.md) | **proposed 2026-08-19, scope approved, not built.** Infrastructure rather than a training change: one `eval_wave.py` controller owns a wave of policies, hands whole-checkpoint units to lanes, and is launched identically by an agent on the laptop and by the daemon on the desktop. Deletes `chain_closeout_after_training.sh` and the gate constants duplicated between it and `runner.py`. Measured prize: **1.7-2.8x on a HOF wave**, 1.0-1.2x on a recent close-out (up to 3.0x on the historically uneven ones) |
