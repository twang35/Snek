# Desktop runner — unattended training/eval driven by git

A dedicated box (Ryzen 7 9700X) runs snek trainings and evals on its own, driven
entirely through git. You never log in to it in normal use: you commit a job, it
runs it, it reports back. Setup is in [`SETUP.md`](SETUP.md).

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

## Capacity testing

Queue several `benchmark` jobs (short fixed runs that report steps/sec), then
sweep `max_trainers` × `tf_intraop_threads` in `runtime.json` and watch aggregate
steps/sec in `status.json`. The knee is the box's real capacity. `HARD_MAX_*` in
`host.env` is the guardrail so a probe can't thrash the machine.

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
