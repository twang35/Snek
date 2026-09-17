"""The distributional value heads: C51, QR-DQN, IQN and FQF, behind `algos/dqn/`'s replay, collector
and schedules. Group A of `plans/algoExploration/a-return-tail.md`.

The four rungs differ in three places and only three -- the head, the loss, and how a greedy action is
read off the head -- so this package is one `net.py` (the heads), one `losses.py`, one `agent.py` (a
`DdqnAgent` whose `update` fits a distribution) and one `algo.py` (the seam, parameterised by the rung),
with four one-screen modules `c51.py`, `qrdqn.py`, `iqn.py`, `fqf.py` giving each rung its `NAME` for
`train.ALGOS` and `tools/restore.ALGORITHMS`. `algos/dqn/` is not changed by this package existing.
"""
