# Desktop runner — unattended training/eval driven by git

A dedicated box (Ryzen 7 9700X, 8c/16t, **14 GiB RAM**) runs snek trainings and
evals on its own, driven entirely through git. You never log in to it in normal
use: you commit a job, it runs it, it reports back.

**It is set up and live** — `the-claw-den`, user `claw`, daemon `active`, verified
2026-08-08. Setup details and the backstop SSH command are in
[`SETUP.md`](SETUP.md); nothing there needs running again.

**This box is a second compute host, not a replacement.** The laptop's own 4-trainer
limit and the desktop's limits are separate pools — see
[counting slots across two machines](#counting-slots-across-two-machines).

**The GPU is not used.** snek2 disables it (`CUDA_VISIBLE_DEVICES=-1`) and the
net is a tiny MLP, so this is a CPU workload — the value of the box is its cores
and that it runs 24/7, not the RTX 5070.

## How it works — three single-writer git branches

Each branch has exactly one writer, so there are never merge conflicts.

| branch | writer | carries |
|---|---|---|
| `ops` | laptop | job specs in `queue/pending/`, and `config/runtime.json` |
| `ops-status` | desktop | `status.json` — heartbeat, running jobs, steps/sec, ledger |
| `results` | desktop | each finished job's `runs/<policy>*` artifacts, pushed at completion |

`master` stays your curated log; the bus branches carry the churn. The desktop
reads `ops` straight from the fetched ref (never checks it out) and writes its two
branches through dedicated worktrees, so nothing races.

The daemon (`runner/`) polls every `poll_seconds`: fetch → re-read `runtime.json`
→ reap finished jobs → launch pending ones up to the concurrency limits → publish
status. Jobs are launched **detached**, so a daemon restart never kills a running
trainer; they self-terminate via `SNEK_MAX_STEPS`. A local ledger makes launches
idempotent across restarts (a job id never runs twice).

## Driving it from the laptop

**Launch jobs** — drop one JSON file per job into `queue/pending/` on `ops` and
push. See [`queue/examples/`](queue/examples/). The daemon drains them as slots
free, lowest `priority` number first.

```
git checkout ops
cp snek2/desktop/queue/examples/train.json snek2/desktop/queue/pending/b20a-seed1.json
# edit it, then:
git add snek2/desktop/queue/pending/b20a-seed1.json && git commit -m "queue b20a" && git push origin ops
```

**Tune it live** — edit `config/runtime.json` on `ops`, commit, push. The change
applies within one poll, no restart:

| key | meaning |
|---|---|
| `max_trainers` / `max_evals` | concurrent trainer / eval jobs (capped by `HARD_MAX_*`) |
| `eval_workers` | `EVAL_WORKERS` per eval job |
| `poll_seconds` | poll cadence (floored by `MIN_POLL_SECONDS`) |
| `tf_intraop_threads` / `omp_num_threads` | TF / oneDNN threads per job — the main throughput lever |
| `nice` | launch priority |
| `disk_min_gb` | refuse to launch below this much free space |
| `paused` / `drain` | finish current jobs, start nothing new |

A malformed `runtime.json` is rejected — the daemon keeps the last good config
and reports the error in `status.json`, so a bad commit can't wedge the box.

**Watch it** — read `status.json` on `ops-status`:

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

**Pull results** — `git fetch origin results && git checkout results -- results/<job-id>`.

## Measured capacity — memory is the limit

**Measured on the box 2026-08-08, four concurrent evals of one checkpoint:**

| config | peak RAM used | verdict |
|---|---|---|
| 4 evals × **10** `EVAL_WORKERS` | **12,770 MB** of 14,336 | **too close** — ~1.5 GB headroom, no OOM but nothing else fits |
| 4 evals × **4** `EVAL_WORKERS` | **7,296 MB** of 14,336 | comfortable |

No OOM kills occurred in either run, so 12.8 GB is a real measurement rather than a
survived near-miss — but it leaves no room for a trainer alongside. **That is why
`HARD_MAX_EVALS=1`**: one eval at `eval_workers: 4` sits well inside budget and
leaves the cores for trainers.

**Cores are not the constraint, and this inverts the laptop's rule.** On the laptop
`EVAL_WORKERS` is close to free and lowering it wastes CPU
([eval cost](../../CLAUDE.md)); here each worker is a process holding its own
TensorFlow arena, so worker count is the memory dial. Raise `max_evals` only with a
`free -m` measurement in hand.

### Capacity testing, if you re-measure

Queue several `benchmark` jobs (short fixed runs that report steps/sec), then
sweep `max_trainers` × `tf_intraop_threads` in `runtime.json` and watch aggregate
steps/sec in `status.json`. The knee is the box's real capacity. `HARD_MAX_*` in
`host.env` is the guardrail so a probe can't thrash the machine. **Sample RAM while
it runs** — steps/sec alone will happily lead you past the memory cliff:

```bash
ssh -i ~/.ssh/snek_desktop claw@the-claw-den \
  'for k in $(seq 1 150); do free -m | awk "NR==2{print \$3}"; sleep 1; done' | sort -n | tail -1
```

## Counting slots across two machines

`CLAUDE.md`'s **"never more than 4 trainers"** rule and its `pgrep` check are
**laptop-local** — they cannot see desktop jobs, and desktop jobs cannot see them.
Treat the two as separate pools:

| host | limit | how to check |
|---|---|---|
| laptop | 4 trainers | `pgrep -fl "python -u snek2.py"` |
| desktop | `max_trainers` (≤ `HARD_MAX_TRAINERS=4`), `max_evals` (≤ 1) | `git show origin/ops-status:status.json` |

**Neither check covers the other host**, so a status report that says "4 arms
running" must say *which box*. The desktop's `counts` and `running` fields in
`status.json` are the only authority for its side.

## Getting a finished job into the analysis workflow

`git checkout results -- results/<job-id>` lands artifacts at
`results/<job-id>/<policy>*`. **The tuning tooling reads `snek2/runs/`**, so a
finished job needs one manual move before `refresh_charts.sh`, `eval_progress.py`
or any of the summary scripts will see it:

```bash
git fetch origin results
git checkout origin/results -- results/<job-id>
cp results/<job-id>/<policy>* snek2/runs/
```

Then treat it exactly like a locally-run arm. **Do not skip the copy and point tools
at `results/`** — `refresh_charts.sh` globs `runs/*.png` only, so a job left in
`results/` silently gets no chart and no caption, which is the same drift the
[charts checklist](../hyperparamTuning/hyperparamTuning.md#when-you-stop-a-batch-of-arms)
exists to prevent.

## Job spec

```json
{
  "id": "b20a-seed1",           // unique, [A-Za-z0-9._-]; the ledger key
  "type": "train",              // train | smoke | benchmark | eval
  "policy": "b20a-seed1",       // required for train/eval; checkpoint dir + runs/ prefix
  "env": {"SNEK_SEED": "1"},    // passed straight through as SNEK_* overrides
  "max_steps": 2000000,         // -> SNEK_MAX_STEPS (self-terminate)
  "eval_args": ["top20"],       // eval only: extra argv for eval_checkpoints.py
  "eval_workers": 10,           // eval only: overrides runtime eval_workers
  "priority": 10,               // lower runs first; default 100
  "notes": "..."
}
```

`smoke`/`benchmark` force `SNEK_MIN_CHECKPOINT_SCORE=0` and default the policy to
`smoke` / `bench-<id>` so they stay throwaway.

## Files

```
desktop/
  README.md  SETUP.md  environment.yml
  runner/    config.py job.py launch.py gitbus.py runner.py
  systemd/   snek-runner.service
  config/    host.env.example  runtime.json
  queue/     pending/  examples/
  tests/     test_runner.py
```
