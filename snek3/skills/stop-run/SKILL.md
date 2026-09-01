---
name: stop-run
description: Stop a snek3 training, eval wave or stage-B close-out on the laptop or the desktop, and clean up the child processes it leaves behind. Use for "kill that arm", "stop the batch", "free a slot", "cancel the eval wave".
---

# Stop a run

## 0. Check before you kill, every time

**Elapsed session time is not wall-clock time.** An arm that feels seconds old to this session can
have trained for hours. This nearly killed a 3.5-hour near-record snek2 arm because the launch "felt
recent".

```
ps -o pid=,etime=,lstart=,stat= -p <pid>
python3 -c "import json;print(json.load(open('runs/<policy>_evals.json'))['summary'])"
```

Read `etime` and `summary.step` **before** deciding. If the user asked for a progress update rather
than a stop, do not kill anything — that is a different skill.

## 1. Laptop

**Use `kill -9`.** A trainer may not stop on SIGTERM. This is safe: every durable file is written
`.partial` then `os.replace`d, and checkpoints land every 1,000 steps.

### ‡ Kill only pids you derived from the job you were asked to stop

**Never sweep by a shape — `ppid == 1`, "all snek3 pythons", a bare module name.** Measured
2026-08-30: an orphan sweep on `ppid == 1` killed a live 8-arm stage-B close-out along with its
target, because `tools/closeout.py` had been launched detached and so had ppid 1 exactly like the
orphans. **A detached parent and an orphaned child are indistinguishable by ppid.** List the pids,
read the full command line of each one, and pass that list.

`ps -o pid=,command=` truncation hides which arm a pid belongs to. Print enough of the line to tell
`b6b` from `b6c` before killing anything.

### One arm

```
ps -Ao pid=,etime=,command= | grep '[t]rain.py'      # read this, pick the pids, then:
PIDS=(<the pids you just read>)
kill -9 "${PIDS[@]}"
ps -Ao pid=,stat=,command= | grep '[t]rain.py'        # verify
```

- **`kill $PIDS` is one argument in zsh** — it does not word-split. Use an array, as above.
- **Do not test liveness with `kill -0`** — it succeeds on a zombie. Read `ps -o stat=`.
- **A `ps`/`pgrep` pattern matches the shell that runs it**, and bracketing does not save you inside
  the tool's `zsh -c` wrapper. When the answer must be trustworthy, match on the interpreter path
  (`envs/snek3/bin/python`) and add `grep -v 'zsh -c'`.

### What a kill leaves behind

Take the children before the parent — `pgrep -P <pid>` lists them, and it is safe because it is
scoped to the one job:

```
kill -9 $(pgrep -P "$PID") "$PID"
```

| you killed | it orphans | what to do |
|---|---|---|
| a **trainer** | its 6 `tools.eval_worker` processes and its chart viewer (measured: all 7 reparented to pid 1) | they **exit after 300 s idle** on their own. Kill them only to free the cores now, and only by pid |
| a **close-out or eval wave** | its `tools.shard` processes — 16 orphans held ~3.7 GB once — and its chart viewer | kill them by pid; nothing merges their files without the controller |

**Nothing is lost either way.** Each shard rewrites its own output file after every completed
measurement, and rerunning the identical command resumes every shard where it stopped. So the repair
for a close-out killed by mistake is: let its shards finish the arm they are on (their logs print an
`eta`), then relaunch the same command.

The chart window is disposable — its own session, never read from or waited on. Killing it cannot
touch a run. `PYTHONPATH=. python -m tools.chart_window` puts it back.

## 2. Desktop `the-claw-den`

**Pause the queue first, or the freed slot refills within one poll (30 s).** Set `paused: true` in
`snek3/desktop/config/runtime.json` on the `ops` branch, push, and `trigger` — see the
`desktop-batch` skill. Then:

```
ssh the-claw-den "ps -Ao pid=,etime=,command= | grep '[t]rain.py'"
ssh the-claw-den 'kill -9 <pids>'
```

Unpause when the box should take work again. Do not restart the daemon to stop a job: jobs are
launched detached with `setsid` and `KillMode=process`, so a restart leaves them running and the
daemon re-adopts them by pid.

**Expect the killed job in `attention` as `failed`.** The daemon reads the negative return code, so a
hand-killed job records `failed` (`rc=-9`), is **not** retried, and does not auto-queue its stage B.
That is the right outcome and it is visible rather than silent — but **to run it again you must delete
its ledger record**, or the id counts as already measured. Say so in your report; do not leave a
`failed` line in `attention` unexplained.

## 3. After stopping

Update `docs/charts.md` and `docs/results.md` in the same pass, and move the batch's rationale out of
`docs/runs.md` into `results.md`. **Without the rationale a later session cannot tell a surprising
result from an arm that was never going to answer anything.**
