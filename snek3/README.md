# snek3

A reinforcement-learning snake in PyTorch. Same game and same 30-value observation as
[`../snek2/`](../snek2/), a clean-slate implementation of everything else.

```
conda activate snek3
cd snek3
PYTHONPATH=. python -u train.py <policy>                 # train
PYTHONPATH=. python -u evaluate.py <policy|batch> [sel]   # measure saved checkpoints
PYTHONPATH=. python -u watch.py <policy> [step]           # a live window
PYTHONPATH=. python -u record_gif.py <policy|hof>         # -> gifs/
PYTHONPATH=. python -m pytest -q                          # the suite
```

**Status: phase 0.** The environment and the measurement engine; no learning code yet. The plan and
its gates are [`plans/pytorch-port.md`](plans/pytorch-port.md).

## What is where

| | |
|---|---|
| [`env/`](env/) | the scalar game — the parity reference, and the only package that may import pygame |
| [`vectorized/`](vectorized/) | `VecSnake`, N games in lockstep in pure numpy, plus the measurement engine and wave |
| [`dqn/`](dqn/), [`ppo/`](ppo/) | learning algorithms |
| [`tools/`](tools/) | the tools and the libraries behind them |
| [`desktop/`](desktop/) | the git-bus job queue for `the-claw-den` |
| [`docs/`](docs/) | the investigation. **[`docs/runs.md`](docs/runs.md) first** |
| [`plans/`](plans/) | designs |
| `runs/`, `evals/`, `charts/`, `hallOfFame/` | committed output |
| `savedPolicies/`, `gifs/` | gitignored output |

Instructions for working here: [`CLAUDE.md`](CLAUDE.md).

## Why PyTorch

snek2's measured bottleneck was never the arithmetic. A TF-Agents `policy.action` at batch 1 costs
217 us, of which the network is ~1%. What snek3 buys is a framework where a custom loss is fifty
lines rather than a subclass of a library agent — which is what PPO and anything after it needs — and
a codebase where the cost of a change is not dominated by reading.

The throughput gain is real but smaller than it looks, because snek2 already banked the eval half:
its vectorised engine shipped 2026-08-24 and reaches ~196,000 env-steps/s. The remaining prize is the
training loop, honestly 5-30x.
