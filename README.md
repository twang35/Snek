# Snek
Different agents that play snake, named by Josh

## TheSchlort
Random movements to learn how to control the snek.

## TheSchmid
Human-coded agent that tries. ~20% perfect game rate.


![theSchmid](https://user-images.githubusercontent.com/5852883/214492458-7fef1c79-1abb-4907-a2bb-cb3ddd4fdd7a.gif)


## TheSchlong
Reinforcement learning agent. ~76% perfect game rate.


![theSchlong](https://user-images.githubusercontent.com/5852883/214492576-f493ec8a-afc0-4e86-9ed5-35e7c6b8be19.gif)

## Snek2

TensorFlow + TF-Agents. Batches 1-47, peaking at a **98.7% perfect-game rate**.
**Frozen 2026-08-28** — kept runnable for A/B against snek3, not developed further.

```
conda activate snek
cd snek2 && python snek2.py <policy_name>
```

Its manual is [`snek2/CLAUDE.md`](snek2/CLAUDE.md), the investigation is
[`snek2/hyperparamTuning/`](snek2/hyperparamTuning/), and the record checkpoints and their
recordings are in [`snek2/hallOfFame/HOF.md`](snek2/hallOfFame/HOF.md).

## Snek3

PyTorch, and the actively-developed one. Same game and same 30-value observation as snek2, so a
snek2 champion's weights convert straight across; a clean-slate implementation of everything else.

```
conda activate snek3
cd snek3 && PYTHONPATH=. python -u train.py <policy_name>
```

The argument is the policy name. It doubles as the checkpoint directory under
`snek3/savedPolicies/<policy_name>/` and as the prefix for the run's own graph and report in
`snek3/runs/`, so several policies train independently without overwriting each other.

Every eval writes `runs/<policy>.png` (the graph, covering the policy's whole history across
restarts), `runs/<policy>.md` (graph, config and eval table, generated from the values the run
actually used) and `runs/<policy>_evals.json` (the measurements later sessions read).

Training never opens a window. To watch a policy play or record it:

```
PYTHONPATH=. python -u watch.py <policy_name>
PYTHONPATH=. python -u record_gif.py <policy_name>
```

Its manual is [`snek3/CLAUDE.md`](snek3/CLAUDE.md), the design is
[`snek3/plans/pytorch-port.md`](snek3/plans/pytorch-port.md), and the investigation is
[`snek3/docs/`](snek3/docs/).
