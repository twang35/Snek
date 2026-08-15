# Diagnostic result payloads

The measurements behind two findings, kept because the run they replaced had not been kept.

`point_of_no_return.py`'s original 2026-08-10 run recorded neither its payloads nor its shard seeds,
so when its `geom` reading was retracted four days later the retraction had to run on **70 fresh
losses instead of re-examining the same 75** — the same two checkpoints and protocol, a different food
draw, and no way to compare loss for loss. That is the cost this directory exists to avoid. **Keep the
payload and the seed for anything a doc quotes.**

| files | script | what it backs |
|---|---|---|
| `eat_and_survive-b18b-{101..104}.json`, `-b20d-{101,102}.json` | [`../eat_and_survive.py`](../eat_and_survive.py) | [the retraction](../../findings.md#-retracted-2026-08-14-the-positions-are-trapped--geom-counts-routes-that-eat-and-die) |
| `endgame_packing-{b18b,b24d,b20d}-{201,202}.json` | [`../endgame_packing.py`](../endgame_packing.py) | [the packing finding](../../findings.md#-the-packing-property-the-records-keep-their-free-space-in-one-piece-and-it-separates-them-by-87-points) |

Each payload carries its own `target`, `checkpoint`, `global_step`, `seed`, `episodes`, `outcomes` and
`mismatches`, so a shard is reproducible from the file alone. **`mismatches` is 0 in all twelve** —
that is the guard that the search was walking the same game the policy played.

## Two things to know before quoting these

**`states` is stripped from the packing payloads.** It held one row per step, which is the
*time-weighted* table: a policy that dithers 900 steps at length 98 contributes 900 rows against a
good policy's 30, so it describes where a policy spends its endgame rather than how well it packs.
No published figure uses it. The per-meal `meals` rows are kept in full and are what `findings.md`
reports. Each file records how many rows were dropped in `states_stripped`.

**The `eat_and_survive` shards predate `ROUTE_NODE_CAP`**, so they ran with an unbounded route
enumeration — more exhaustive than the committed script, not less, since the caps added afterwards can
only turn a negative verdict inexact. **Checked rather than assumed:** `b20d` seed 101 was re-run under
the committed code as `eat_and_survive-b20d-101-recheck.json`, and every `at_last_geom` record is
**identical loss for loss** — same 27 losses, same 13 survivable, same 23 with no open neighbour, 0
mismatches both times. The caps never bound on these boards. Re-running the other five shards would
cost about ten minutes if a future session wants the same assurance for them.
