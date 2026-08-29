# The desktop — `the-claw-den`

A stdlib-only systemd daemon on a dedicated Linux box that runs trainings and evals unattended. It
**imports nothing from this project** — it shells out to `train.py` and `evaluate.py` — and it talks
to the laptop only through three single-writer git branches. That decoupling is the design's best
property: the bus works from anywhere, `ssh` is a convenience, and the daemon cannot be broken by a
change to the trainer.

**snek3's daemon owns the box as of 2026-08-28**, replacing snek2's. The two eras share the `ops`
branch, which is why every spec carries a required `project` field.

| branch | writer | payload |
|---|---|---|
| `ops` | **laptop** | `snek3/desktop/queue/pending/*.json` specs, `snek3/desktop/config/runtime.json` |
| `ops-status` | **desktop** | `status.json` — heartbeat, running jobs, ledger, `at_a_glance` |
| `results` | **desktop** | `results/<job-id>/*` artifacts |

One writer per branch, so every push is `--force-with-lease` and nothing ever merges.

## Queue a job

Three steps, and the third is what makes it start now rather than within ten minutes.

```
git checkout ops
cp snek3/desktop/queue/examples/train.json snek3/desktop/queue/pending/b1b.json   # edit it
git add snek3/desktop/queue/pending/b1b.json && git commit -m 'queue b1b' && git push origin ops
ssh the-claw-den 'Snek/snek3/desktop/trigger'
```

**Pushing to `ops` starts real work on another machine**, so it needs the user's approval for *that*
job — see the root [`CLAUDE.md`](../../CLAUDE.md).

`queue/examples/` holds one worked spec per job type. The fields:

| field | required | means |
|---|---|---|
| `project` | **yes, no default** | must be `snek3`. The guard against `ops`'s ~150 retired snek2 specs |
| `id` | yes | unique; the ledger key and the log name. `b<n><letter>-...` groups arms into a batch |
| `type` | yes | `train`, `smoke`, `benchmark`, `eval` |
| `policy` | train/eval | the checkpoint directory name |
| `policies` | eval only | a **wave**: every arm of a batch in one process |
| `max_steps` | no | `SNEK_MAX_STEPS`. Defaults per type for smoke/benchmark |
| `env` | no | any `SNEK_*` knob; wins over the runtime defaults. See [`../docs/running.md`](../docs/running.md) |
| `selector`, `episodes` | eval, no | **omit them.** Absent means `evaluate.py`'s own defaults, which *are* the protocol |
| `eval_shards` | no | shard processes for this wave; defaults to the runtime config's 16 |
| `priority` | no | lower runs first. Default 100, auto-queued stage B is 10 |
| `label` | no | one line for `at_a_glance` |
| `notes` | no | free text, for the reader |

A malformed spec is **recorded against its filename and skipped**, never raised into the loop, so one
bad commit cannot stop the box.

## Read the box

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

**`git fetch` is not optional, and leaving it out is the single most repeated mistake in this
project's history with the desktop.** `git show origin/ops-status:…` reads a local remote-tracking
ref, so without the fetch you are shown an arbitrarily old snapshot *with no indication that it is
old* — and the payload contains a timestamp, so the natural misreading is "the daemon died at 08:33"
when the truth is "my ref is from 08:33". That has produced three false alarms. The ladder, in order:
fetch and re-read; `trigger`, which makes the daemon publish *now* and reports whether it is polling
at all; then `ssh -o ConnectTimeout=8 -o BatchMode=yes`. Only after all three fail is the box worth
calling unreachable.

**A `status.json` up to 10 minutes old is a healthy daemon** — see `git_seconds` below.

Start with `at_a_glance`: one line per running batch with a percentage, one per queued batch-phase,
and an `attention` list for anything needing a human — a failed eval, a stuck push, a rejected spec.
`counts`, `running`, `ledger`, `disk_free_gb` and `load_avg` are underneath it.

Ledger states, and the one worth knowing:

| state | means |
|---|---|
| `queued` → `running` → `done` | the normal path |
| `failed` | non-zero exit, or a traceback in the log |
| **`interrupted`** | the box rebooted under it. **Non-terminal, so it is relaunched**, and a training resumes from its checkpoint |

`interrupted` exists because the same situation used to read `done`, which **published truncated arms
as finished and silently consumed their measurements.** It is detected by comparing the record's boot
id against `/proc/sys/kernel/random/boot_id`. Read it as "lost wall clock, nothing else".

## Tune it while it runs

`config/runtime.json` is committed on `ops` and re-read every network cycle. **A malformed file is
rejected whole and the last known-good config kept** — the box has no ssh backstop in normal
operation, so a bad commit must never be able to wedge it, and a *partially* applied config is worse
than a rejected one because it looks like it worked. Values are then clamped to `host.env`'s
`HARD_MAX_*`, with anything clamped noted in `status.json` rather than raised.

| knob | default | notes |
|---|---:|---|
| `max_trainers` | 4 | concurrent train/smoke/benchmark jobs |
| `max_evals` | 1 | concurrent eval *jobs*. A job is a whole wave, so 1 is normal |
| `eval_shards` | 16 | **this is the knob that fills the box**, not `max_evals` |
| `poll_seconds` | 30 | the local half: reap, read the fetched ref, dispatch. Off-network, so it stays fast |
| `git_seconds` | 600 | the network half: one fetch, one status push, one retry of any local-only commit |
| `torch_threads` | 1 | measured, not cautious. `SNEK_TORCH_THREADS` |
| `omp_num_threads` | 1 | same, for numpy's BLAS underneath |
| `nice` | 0 | |
| `disk_min_gb` | 5 | refuse to launch below this much free |
| `paused` / `drain` | false | finish what is running, start nothing new |
| `auto_stage_b` | true | a finished training auto-queues its stage-B wave at priority 10 |

**Cores are the binding constraint, not memory** — which is a change from snek2. Measured 2026-08-28
on this code: an eval shard peaks at **202 MB** (193 of it torch's import) and a trainer at **290
MB**, so a full box is **4.4 GB of 15,030**, 29%. snek2's memory-driven three-way clamp therefore
collapses into one ceiling tied to the box's **16 SMT threads** (Ryzen 7 9700X, 8 physical cores): 16
shards at one intra-op thread each is the measured optimum, and 18 loses 6-10%. `eval_shards` gives
way to that ceiling, never `max_evals` — how many waves run is a scheduling decision, and quietly
running fewer than asked would be a surprising way to honour a thread limit.

`git_seconds` is separate from `poll_seconds` because at 30 s the box made ~2,880 fetches and ~2,880
pushes to github a day — enough sustained machine-shaped traffic from a home connection to be worth
not making. 600 s cuts both to 144 while costing nothing locally, and `trigger` covers the case where
a batch should start *now*.

## Set the box up

```
ssh the-claw-den
cd ~/Snek && git fetch origin && git merge --ff-only origin/master

# the two worktrees the daemon writes its branches through, outside the main checkout
git worktree add /home/claw/snek-bus/status  ops-status
git worktree add /home/claw/snek-bus/results results

cp snek3/desktop/config/host.env.example snek3/desktop/config/host.env   # edit for this box
sudo cp snek3/desktop/systemd/snek3-runner.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now snek3-runner
```

`host.env` is **not** in git — only the example is. It holds machine identity (paths, branches, the
two `ops` locations) and the hard ceilings; everything tunable at runtime is in `runtime.json`
instead.

The daemon runs on **base** python, not the conda env, so it can start before `snek3` is built;
`PYTHON_BIN` is the env python and is what actually needs torch. `KillMode=process` means a deploy
can restart the daemon with four arms mid-run — jobs are launched detached with `setsid`, they
self-terminate at `SNEK_MAX_STEPS`, and the daemon re-adopts them by pid on the next poll.

Deploying new code is the same `ff-only` merge plus `sudo systemctl restart snek3-runner`. **Untracked
run artifacts on the box abort the fast-forward**, and piping the merge to `tail` hides the failure.

## What the port changed, and why each one is an incident

| change | the incident behind it |
|---|---|
| a required `project` field | `ops` holds ~150 retired snek2 specs whose `script` would resolve to a TensorFlow trainer that does not exist here |
| one eval stage, not two | `training → closeout → HOF` becomes `training → stage B`, so the `auto_hof` hop, the `-hof` id handling and the legacy phase branch all go |
| `_ensure_viewer` deleted (~200 lines) | snek3's trainer writes PNGs and never opens a window. An X11 session under memory pressure once XIO-crashed all four arms |
| `publish_results` reports its push, and `push_unpushed` retries | a failed push left the commit local while the ledger said `done` — **indistinguishable from a pass that legitimately found nothing.** It hid four 500-episode result files, one a 98.2% checkpoint, for hours |
| a `failed` eval reaches `attention` | silently never retrying one cost snek2's batch 46 wave 1 its whole measurement |
| no episode count or gate in the launcher | snek2's daemon carried five protocol numbers as a second copy of what `eval_plan.py` defines, and they drifted |
| `clear_stale_locks` | kill the daemon inside a git write and `index.lock` outlives it. Every later publish then fails while jobs keep running — a frozen heartbeat over healthy work, and it needed a human with ssh to delete a file |

Carried across unchanged, each also a documented incident rather than a preference:

- **The wave barrier.** Trainings and evals never overlap, and nothing backfills until a wave drains.
- **`EVAL_RELEVANT_ENV`.** A wave is keyed only on the settings that can reach a measurement of an
  already-trained checkpoint — shaping and reward knobs, not seeds or learning rates. Keying on the
  whole inherited env once split one batch into three waves of 2/1/1 arms.
- **`trigger`.** One ssh round trip that forces a fetch, a dispatch and a publish, and reports whether
  the daemon is polling at all — so one command both starts queued work and answers "did it start?".
  Exit 0 healthy, 2 not polling, 1 unreachable.
