# snek3 docs

The investigation. Snek3 starts these fresh — **none of snek2's 3,300 lines of findings or its arm
table come across**, because snek3 is for new ideas (PPO first) and a result about snek2's
hyperparameters under TF-Agents is not evidence about snek3's.

What *did* come across is [`invariants.md`](invariants.md): nine facts about the game and the
instrumentation that are properties of the problem rather than results.

| file | contents |
|---|---|
| [`runs.md`](runs.md) | what is running, what to run next. **Start here** |
| [`protocol.md`](protocol.md) | how to judge a run: metrics, stop criteria, the eval protocol |
| [`running.md`](running.md) | how to launch things, and every `SNEK_*` knob |
| [`environment.md`](environment.md) | the game, the observation vector, the reward terms |
| [`invariants.md`](invariants.md) | the nine |
| [`findings.md`](findings.md) | what is established, what is falsified |
| [`results.md`](results.md) | every arm: config, final numbers, verdict |
| [`charts.md`](charts.md) | one graph per arm |

**Keep the split clean.** `runs.md` is current state and forward plan only; results go to
`results.md`, conclusions to `findings.md`, anything about *how to measure or judge* to
`protocol.md`. snek2's equivalent grew to 950 lines of interleaved status and stopped being usable.

Designs that are worth writing down before they are worth building go in
[`../plans/`](../plans/), one file each.
