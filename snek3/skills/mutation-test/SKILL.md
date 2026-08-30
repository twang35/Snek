---
name: mutation-test
description: Verify that snek3's tests actually cover a change, by mutating the implementation and confirming a test fails. Use for "does this have coverage", "run mutation testing", "check the tests catch this", or after adding logic worth pinning down.
---

# Mutation testing

**A passing suite is not coverage of the change you just made.** snek2 took a third signature for its
observation grouper and all 24 existing tests passed before and after, because every fixture was an
open board where old and new answers agree. Mutate the implementation and confirm a test fails.

## Use `tools/mutate.py`. Do not write the shell version

```
cd snek3
PYTHONPATH=. python -m tools.mutate tests/mut_<name>.json
```

Every hazard below is one an ad-hoc harness walked into in this repository, and all four are handled:

| hazard | what an ad-hoc harness does | why it is silent |
|---|---|---|
| a stale `.pyc` | `mv` the backup back, restoring an older mtime | bytecode is revalidated on mtime **and size**, and `3` → `1` changes neither |
| a pattern that does not match | abort *after* checking, before writing a backup | the next restore puts the previous file's backup over the wrong module |
| the harness is killed | nothing | `finally` unwinds on an exception, **not on a signal** — the mutation in flight survives |
| the mutant hangs | wait forever | an outer command timeout then kills the harness, which is hazard 3 |

The last two are one chain and it fired twice in one session: dropping the decrement from a
`while debt >= 1.0:` accumulator makes an infinite loop, the 2-minute command timeout kills the
harness, and `train.py` is left holding the mutation. `mutate.py` raises on `SIGTERM`/`SIGINT` so the
restore runs, and allows each mutant 6x the baseline's own wall clock before calling it hung. **A hung
mutant counts as killed** — the tests noticed.

## The committed specs — re-run rather than re-derive

| spec | mutants | covers |
|---|---:|---|
| `tests/mut_ppo.json` | 14 | `ppo/` — GAE's episode gate, `min` vs `max`, the entropy sign, the ratio's direction, γ/λ order, the bootstrap's state |
| `tests/mut_seam.json` | 15 | the `train.py` algorithm seam and `dqn/algo.py` |
| `tests/mut_window.json` | 20 | the chart window's `flock`, the viewer, `closeout.py`, the desktop runner |
| `tests/mut_trans.json` | 8 | the `transitions` column, prefill to summary block |
| `tests/mut_shaping.json` | 3 | `shaping_discount` reaching the collect env |

`ls tests/mut_*.json` rather than trusting this table — it is a summary, not the source. Add mutants
to the matching spec rather than starting a new one, unless the target is a new module.

A spec's `tests` key scopes the run to the relevant test files, so a small spec is fast:
`mut_shaping.json` (3 mutants against `tests/test_train.py`) finishes in well under a minute. Verify
the restore afterwards — `git status --porcelain` over the mutated files should be empty.

## Reading the result

**A survivor names a test-writing mistake, and usually the same one: the fixture's subject was a copy
of the line rather than the line.** Two PPO mutants survived their first run. The clipped-objective
fixtures rebuilt the surrogate from the same three statements the agent uses — pinning the arithmetic
and leaving `min` → `max` *in the agent* undetected — and `ppo/collect.py` had no test file, so a
bootstrap taken off the last stored value instead of the state after it passed 100 tests. **The fix in
both cases was a fixture that calls the production entry point.**

**Check the failure *type*.** Thirteen of snek2's tests were dead for two signature generations: they
called a function with an argument it had stopped taking, so they raised `TypeError` rather than
failing an assertion, and a `TypeError` looks like noise if nobody is watching.

**A fixture whose subject cannot violate it is not a fixture.** A frame-rate fixture asserted a
"naive" formula that rounded exactly the way the real one does, so it failed while the code was right;
a palette fixture asserted a *bound* on frames that only had six colours, which any cap satisfies, so
it passed with the knob ignored.

For refactors, also diff observations against a fixed-seed run — byte-identical output over a few
thousand steps catches what assertions do not.
