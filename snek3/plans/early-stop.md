# Early stop in the deep passes: retire a checkpoint once the target is out of reach

**Status: built 2026-09-09**, the same day it was planned, in the configuration the user chose after §5:
**no stop in stage B, hof5000 stops at 99.6, hof30k stops at 99.8** (`eta.HOF5000_STOP`, `eta.HOF30K_STOP`;
`--stop` / `--no-stop` on `tools.closeout`). §3 is as built, with one change: the sweep page shows the
counts at the gates (`hof_996`, `hof30k_998`) in place of the hof means (user). The estimates in §4 were
the basis; the 99.7 column there is the asked-for one, the 99.8 hof30k figures are in its text.

## 1. What is asked

The three passes measure every selected checkpoint to full length -- 500, 5,000 and 30,000 episodes -- and
a checkpoint that failed its 61st episode of 30,000 keeps playing the other 29,000 although it can no
longer read 99.8. snek2 had an abandon gate (`EVAL_MIN_ACHIEVABLE`); snek3 dropped it on purpose
(`plans/archive/pytorch-port.md` §"the gate is the load-bearing removal"). This plan brings it back in the
one shape that keeps the reasons for dropping it answered.

**The rule is arithmetic, never predictive**: a checkpoint is retired only when `perfect_so_far +
episodes_not_yet_played < target * episodes`, i.e. when even a perfect remainder cannot reach the target.
A checkpoint that would have reached the target is never stopped, and a stopped row's rate is by
construction below the target.

| pass | episodes | selector (gate in) | proposed stop target | what a stopped row can still do |
|---|---:|---|---:|---|
| stage B | 500 | screen (stage-A >= 97) | **99.2** (the hof5000 gate) | nothing downstream selects it; **it is lost to density98** (§5) |
| hof5000 | 5,000 | above:99.2 | **99.6** (the hof30k gate) | cannot be promoted; drags `hof_mean` unless readers become episode-aware |
| hof30k | 30,000 | above:99.6:hof5000 | **99.7** | last pass, nothing selects from it; only `hof30k_mean` reads it besides `best` |

Each target is one knob: `SNEK_EVAL_STOP_<pass>` or a `--stop <pct>` on `tools.closeout`, absent = no
stop (today's behaviour). The stop target of a pass must not sit below the next pass's gate, or a stopped
row could be promoted -- snek2's `assert DEFAULT_MIN_ACHIEVABLE < HOF_GATE`, this time asserted in
`closeout.PASSES` at import and pinned by a test.

## 2. Why the engine can do it cheaply

`vectorized/engine.py` runs one env of 1,024 lanes per shard with 8-12 checkpoints resident; a lane that
finishes an episode is reassigned to whichever resident checkpoint still has episodes to *start*, and a
new checkpoint is loaded only when none does (`take_job`, `assign`). So retiring a job frees its unstarted
quota at once and the batch stays full: there is no drain penalty, which is where snek2's per-unit abandon
lost part of its gain. Results are banked by **start slot** (`_Job.record`), so a prefix is a fair sample.

Two things the engine has today that the change must respect:

- **`_Job.held()` refuses a partial sample** (`banked != episodes` raises) and hard-codes
  `'abandoned': False`; `tests/test_engine.py` pins "no abandon gate". Both become: a job is either full
  or *stopped*, and a stopped job hands out exactly the episodes it banked.
- **In-flight lanes.** When the stop fires, a few of the job's lanes are mid-episode, and the ones still
  playing are disproportionately the long, perfect games (`engine.py` §"length bias"). **Let them finish**
  -- stop *starting* episodes for the job, retire it when its last lane completes -- rather than dropping
  them. It costs a handful of lanes for one episode length and removes a bias snek2 never had to face.

## 3. The change

| where | what |
|---|---|
| `vectorized/engine.py` | `measure_stream(..., stop_target=None)`. `_Job` gets `stop_target`, `started`, and `out_of_reach()`: `perfect + (episodes - started) < ceil(target/100 * episodes)`. `take_job` skips a job that is out of reach (no new starts); `finished()` is `done >= episodes or (out_of_reach() and done == started)`. `held()` compacts `scores`/`perfect`/`rewards` to the banked slots and carries `episodes_planned`, `episodes` (banked), `abandoned`, `stop_target` |
| `tools/eval_plan.py` | `build_row` divides by banked episodes, writes `episodes_planned`, `abandoned`, and the Wilson CI on the banked count. The test that asserts `abandoned` is *not* a field flips to asserting it is, with `false` on a full row |
| `tools/results.py` | the file header carries `stop_target` (null when off); `merge` keeps the longer sample as it does and refuses to pool two files with different `stop_target`s -- snek2's "four gate eras" trap, refused rather than remembered |
| `tools/shard.py`, `tools/eval_wave.py`, `tools/closeout.py` | thread `--stop` through; `closeout.PASSES` gets `stop` per pass from the knob; the import-time assert on stop >= next gate |
| `tools/stage_b_chart.py` | `pooled_percent` and the `>= 95/98/99` counts over **full rows only**, and the caption says how many rows stopped; `episodes_per_row` prints planned depth, not the banked set |
| `tools/viewer_manifest.py`, `tools/sweep_analysis.py` | `density98`, `hof_mean`, `hof30k_mean` over full rows only; `hof_best`, `hof30k_best`, counts `>= gate` unaffected (a stopped row is below every gate) |
| `tools/eta.py` | per-checkpoint rates are learnt from the ledger, so they re-learn; `pass_progress` counts rows, unchanged |
| `docs/protocol.md`, `docs/invariants.md`, `CLAUDE.md` "the eval protocol is one stage" | the sentence "every row is full length" becomes "every row is full length **or stopped below the target, and says so**"; the stop targets and the ordering rule |
| tests | `test_engine`: a job stopped at the right episode, in-flight lanes finished, a job that reaches the target never stopped, full rows unchanged with the knob off; `test_eval_plan`: the row fields; `test_results`: the header refusal; `test_closeout`: the ordering assert; `test_stage_b_chart` / `test_viewer_manifest`: stopped rows excluded from the pooled statistics |

Order of work: engine + eval_plan + results with tests, then the pass plumbing, then the readers, then docs.
The knob defaults to **off**, so every existing file and every pass in flight is unchanged until a target is set.

## 4. Estimates: b27 and b28 on the desktop

Wall time at the desktop's measured rates (16 shards): 0.27 s per stage-B checkpoint, 2.7 s per hof5000
checkpoint, 16.3 s per hof30k checkpoint (`desktop/runs/.live/.durations.json`, hist8's passes). A stopped
checkpoint is charged the fraction of its episodes played, failures taken as uniformly spread. The model
reproduces hist8's actual passes (1.71 / 9.37 / 8.34 h) to within 3%. b27's three cells are measured to
the end; **b28 is measured through stage B and predicted onward** from hist8's conditional distributions
(hof5000 score given stage-B score, hof30k given hof5000) -- the 944 b28 hof5000 rows in so far promote
14.7% to 99.6, exactly hist8's share.

| cell | rows B / 5k / 30k | stage B: now -> stop 99.2 | hof5000: now -> stop 99.6 | hof30k: now -> stop 99.7 | total: now -> proposed |
|---|---|---|---|---|---|
| b27 hist0 | 15,271 / 1,128 / 124 | 1.15 -> 0.61 h | 0.8 -> 0.3 h | 0.6 -> 0.2 h | 2.6 -> 1.1 h (57%) |
| b27 hist4 | 22,102 / 11,288 / 1,355 | 1.66 -> 1.44 h | 8.5 -> 5.3 h | 6.1 -> 4.2 h | 16.3 -> 11.0 h (32%) |
| b27 hist8 | 22,192 / 12,479 / 1,836 | 1.66 -> 1.48 h | 9.4 -> 6.0 h | 8.3 -> 5.8 h | 19.3 -> 13.4 h (31%) |
| b28 hist8a25 (predicted) | 46,471 / 31,630 / ~5,000 | 3.49 -> 3.24 h | 23.7 -> 15.6 h | 22.8 -> 16.0 h | 50.0 -> 34.9 h (30%) |
| **all four** | | | | | **88 -> 60 h (32%)** |

Where the saving is: **hof5000 with the 99.6 stop gives 8 of the 28 hours**, hof30k with 99.7 gives 12,
stage B with 99.2 gives 1. A hof30k stop at **99.8** instead of 99.7 would take hof30k to 2.9 / 3.9 / 10.8 h
(hist4 / hist8 / b28) -- another 10 hours over the four cells -- at the price of not knowing which
checkpoints read 99.7, which the hall-of-fame promotion does not need but a "how wide is the 99.7 region"
reading would.

## 5. Where this plan disagrees with the configuration asked for

**The stage-B stop at 99.2 saves about 1 hour in 88 and costs the batch's primary metric.** `density98` --
the share of stage-B rows at or above 98/500, readout 1 of the sweep and the number every batch is judged
on -- counts rows *between* 98 and 99.2, and those are exactly the rows a 99.2 stop retires unfinished.
Under the stop a batch's density98 is not computable, and neither is the stage-B chart's pooled rate or the
`>= 95 / 98 / 99` table. The recommendation is **no stop in stage B** (or a stop at 98.0, which preserves
every row density98 counts and saves 0.02 h per wave -- not worth the knob). The hof5000 and hof30k stops
carry no such cost: nothing reads a hof5000 row below 99.6 except `hof_mean`, and nothing reads a hof30k
row below 99.7 except `hof30k_mean`, both of which §3 restricts to full rows.

**Sequential stopping biases a stopped row's rate downward** (the decision to stop is taken when the
row is having a bad run). That is fine for rows that are only ever "below the target", and it is why the
readers must not average them in. It is also why `stop_target` has to be in the file header: a file's
stopped rows mean something only against the target they were stopped under.

## 6. Open decisions

- Stage B: off (recommended) or 99.2 (asked). §5.
- hof30k target: 99.7 (asked; keeps the 99.7 band measured) or 99.8 (10 more hours over b27+b28).
- Whether `hof_mean` / `hof30k_mean` over full rows only is still a useful statistic once most rows stop,
  or whether the sweep page should show "count >= gate" in its place.
