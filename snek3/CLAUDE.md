# snek3 — PyTorch

The active era. Read the repository's [`../CLAUDE.md`](../CLAUDE.md) first for the collaboration
rules, the conda envs, the git workflow and the two compute hosts; this file is what is true of
snek3 specifically.

**The design is [`plans/pytorch-port.md`](plans/pytorch-port.md).** It carries the decisions, the
measurements behind them, and the phase gates. Read it before proposing a change to the structure.

**snek2 is frozen.** Copy from it freely — that is what it is for — but never edit it, and never
move a file out of it.

## Environment

```
conda activate snek3          # or /opt/miniconda3/envs/snek3/bin/python directly
```

Python 3.12, torch, numpy 2.x, pygame, matplotlib, pillow, imageio, pytest. **No TensorFlow.**
numpy 2 changed integer-promotion rules (NEP 50), which matters in exactly one place: `VecSnake`'s
packed-uint64 bitboards. Every shift there goes through an explicit `np.uint64` operand for that
reason.

## Layout

| directory | contents | may import pygame? | may import torch? |
|---|---|---|---|
| `env/` | the scalar game: constants, drawing, `Game`, the reference observation builder | **yes, and only here** | no |
| `vectorized/` | `VecSnake` (N games in lockstep, pure numpy) plus the measurement engine and wave | no | no |
| `dqn/`, `ppo/` | learning algorithms | no | yes |
| `tools/` | the tools and the libraries behind them: `eval_plan`, `run_report`, `arch`, charts | no | no |
| `desktop/` | the git-bus job queue. stdlib only, imports nothing from this project | no | no |
| `docs/` | the investigation | | |
| `plans/` | designs | | |
| `tests/` | | | |

**The pygame column is an invariant, not a convention.** `pygame.init()` opens a real CoreAudio
stream per process and `SDL_VIDEODRIVER=dummy` does not affect audio. Keeping the vectorised and
eval paths pygame-free removes the trap instead of guarding against it. `env/constants.py` holds no
pygame objects for the same reason — that is what `env/render.py` is for.

**`torch` appears only inside a `policy_fn`.** The engine's seam is one function of shape
`(m, 30) float32 -> (m,) int64`, which is what lets the whole measurement stack be tested against a
hand-written heuristic policy with no framework imported. Do not thread a tensor through
`vectorized/`.

## Running things

```
cd snek3
PYTHONPATH=. python -u train.py <policy>                 # policy name doubles as the checkpoint dir
PYTHONPATH=. python -u evaluate.py <policy|batch> [sel]   # the eval wave
PYTHONPATH=. python -u watch.py <policy> [step]           # a live window, follows the newest checkpoint
PYTHONPATH=. python -u record_gif.py <policy|hof>         # -> gifs/, throwaway
```

**Always pass `smoke` for verification runs**, so output is isolated in `savedPolicies/smoke/` and is
safe to delete. Hyperparameters come from `SNEK_*` env vars, so variants run side by side without
editing files; `grep 'hyperparameter override:'` on a log confirms an arm got its config.

**Training never draws.** A display flip costs ~5.2 ms and the game flips once per step — a round
trip to the window server, not our drawing code. `watch.py` and `record_gif.py` are the only ways to
see a game, and they run in their own processes so they cost training nothing.

## The eval protocol is one stage

| stage | who | selection | episodes |
|---|---|---|---:|
| **A** | the trainer, in-process, every 1,000 steps | every checkpoint | 100 |
| **B** | a wave of shard processes, after the arm stops | every checkpoint at **≥95/100** in stage A | 500 |

**Stage B is the hall-of-fame measurement.** There is no third stage, no tiered selection, no
screen/confirm split and no min-achievable gate — so **every row in every result file is full length
and directly comparable**, which was not true of snek2's files. `min_achievable` is absent, not null
with a number to check.

Three consequences worth holding on to:

- **Stage A is measurement, not a progress readout.** It must run at 100 episodes on every
  checkpoint or a checkpoint exists that no screen can select. That makes it ~90% of a training
  arm's wall clock, and an arm ~2 h — which is the cost of the protocol, not waste.
- **`perfect_percent` is not only a report.** It feeds the epsilon refinement schedule, so breaking
  the measurement changes the training. In snek2 a shaping term silenced the perfect-game counter and
  eight arms trained with epsilon pinned at its ceiling for 300k+ steps while reading 0%.
- **The maximum over stage B is a selected high.** A *record* claim needs a fresh measurement of the
  single winner at 1,000+ episodes. snek2's 99.0%/500 champion re-measured at 97.5%/1000, and its
  four best hall-of-fame entries fell a mean 1.4 pp on re-measurement.

## Invariants

[`docs/invariants.md`](docs/invariants.md) holds the nine facts about this game and its measurement
that are cheap to state and expensive to rediscover — the perfect-game definition, the `arch.json`
era guard, the observation conventions, the reward/discount coupling, and the noise floor. **Read it
before changing the observation, the reward, or anything that counts a perfect game.**

## Tests

pytest is installed here, unlike in the `snek` env:

```
cd snek3 && PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -m pytest -q
```

**When a change has logic worth pinning down, add a test in the same pass.** Worth a fixture whenever
a change involves a conditional whose branches could later collapse, an index or coordinate
convention, a rule someone could "simplify" without seeing why it exists, or an edge case that took
thinking to get right — if it needed reasoning, it needs a fixture, because the reasoning does not
survive in the diff.

**A passing suite is not coverage of the change you just made. Mutate the implementation and confirm
a test fails.** snek2 took a third signature for its observation grouper and all 24 existing tests
passed before and after, because every fixture was an open board where old and new answers agree.

**Check the failure *type*.** Thirteen of snek2's tests were dead for two signature generations —
they called a function with an argument it had stopped taking, so they raised `TypeError` rather than
failing an assertion, and a `TypeError` looks like noise if nobody is watching.

**A fixture whose subject cannot violate it is not a fixture.** Two real examples: a frame-rate
fixture asserted a "naive" formula that rounded exactly the way the real one does, so it failed while
the code was right; and a palette fixture asserted a *bound* on frames that only had six colours,
which any cap satisfies, so it passed with the knob ignored.

For refactors, also diff observations against a fixed-seed run — byte-identical output over a few
thousand steps catches what assertions do not.

## Docs

| file | contents |
|---|---|
| [`docs/runs.md`](docs/runs.md) | what is running, what to run next. **Start here** |
| [`docs/protocol.md`](docs/protocol.md) | how to judge a run: metrics, stop criteria, how to launch |
| [`docs/findings.md`](docs/findings.md) | what is established, what is falsified |
| [`docs/results.md`](docs/results.md) | every arm: config, final numbers, verdict |
| [`docs/environment.md`](docs/environment.md) | the game, the observation vector, the reward terms |
| [`docs/invariants.md`](docs/invariants.md) | the nine |
| [`docs/charts.md`](docs/charts.md) | one graph per arm, linked straight from `runs/` |
| [`docs/running.md`](docs/running.md) | every `SNEK_*` knob and what it does |

Keep the split clean: `runs.md` is current state and forward plan only, results go to `results.md`,
conclusions to `findings.md`, anything about *how to measure or judge* to `protocol.md`. snek2's
equivalent grew to 950 lines of interleaved status and stopped being usable.

**`docs/charts.md` links `../runs/<policy>.png` directly.** There is no copy step and no separate
chart directory to keep in sync — that duplication is what snek2 needed `refresh_charts.sh` and a
completeness-check snippet for, and it still drifted to 12 undocumented arms. `charts/` holds only
the one-off diagnostic figures a finding refers to.

**Any time you touch the docs or run a progress update, refresh `charts.md` in the same pass** —
whether or not the arms have finished. A running batch with no chart entry is a bug, not a "wait
until it closes" state.

## "Progress update" means look, don't touch

A progress update is **read-only with respect to running processes**: analyse, update docs, report.
**Do not kill, stop or restart any arm** — not even one that looks finished, is past its cap, or is
clearly failing. Deciding a run is done is the user's call. If a run looks finished and no slot is
needed, say so as a recommendation.

**Before killing or relaunching any arm, check its wall-clock runtime and step, and never call an arm
"fresh" from a hunch.** Elapsed session time is not real time — an arm that feels seconds old to a
session can have trained for hours. This nearly killed a 3.5-hour, near-record snek2 arm whose config
change would have been reverted for the loss, because the launch "felt recent". Run
`ps -o etime,lstart -p <pid>` and read `summary.step` from `runs/<policy>_evals.json` *first*.

## Reading status

Training runs quiet — one compact line per 10 evals. `SNEK_DEBUG=1` restores verbose output; use it
for debugging, not status. Read status from `runs/<policy>_evals.json`'s precomputed `summary` block.

**`strong_eval_fraction` is the primary metric** (share of an arm's evals at ≥80% perfect) — it has
the lowest between-seed variance of the candidates. It is a fraction of each arm's own evals, so
**compare only at a common step horizon.** And it is a *threshold-crossing* statistic, so it rewards
noise: it is not comparable across a change in episodes per eval, where a *rate* is. snek3 runs 100
episodes throughout, so this only bites when comparing against a snek2 number.

**Use `zero_since`, not `dead_since`, to ask whether an arm is dead now.** `dead_since` is the
earliest sustained-zero stretch and is history; `zero_since` is the current unbroken stretch.
Neither is a verdict — a snek2 arm recovered from 1.2M steps near zero to 63.7 trailing.

## Two rules that are easy to get wrong

- **Never run more than 4 trainers at once on the laptop**, counting human-started ones. Check with
  `ps -Ao pid=,command= | grep '[t]rain.py'`, not `pgrep -f` — see the root file on why a process
  scan both over- and under-reports.
- **This domain is very noisy** — the same snek2 config produced 62.5 and 18.0. Never conclude from a
  single run; repeat promising configs 2-3 times. **n=4 cannot resolve an effect below ~10 pp.**
