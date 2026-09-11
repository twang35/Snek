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

**Never sweep by a shape — `ppid == 1`, "all snek3 pythons", a bare module name.** Measured: an orphan sweep on `ppid == 1` killed a live 8-arm stage-B close-out along with its
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
| a **trainer** | nothing of its own under the scheduler (the workers and the viewer are the scheduler's); a bare `train.py` orphans its `tools.eval_worker` processes | workers **exit after 300 s idle** on their own. Kill them only to free the cores now, and only by pid |
| a **close-out or eval wave** | its `tools.shard` processes — 16 orphans held ~3.7 GB once | kill them by pid; nothing merges their files without the controller |
| the **scheduler** (`tools.scheduler`) | nothing: its arms and its close-out run in their own sessions and finish; its viewer exits by itself once its parent is gone | rerun the same command to resume; it adopts live arms |

**Nothing is lost either way.** Each shard rewrites its own output file after every completed
measurement, and rerunning the identical command resumes every shard where it stopped. So the repair
for a close-out killed by mistake is: let its shards finish the arm they are on (their logs print an
`eta`), then relaunch the same command.

The chart window is disposable — its own session, never read from or waited on. Killing it cannot
touch a run. The scheduler reopens it at its next launch; `PYTHONPATH=. python -m tools.scheduler
--reopen-window` does it now.

## 2. Desktop `the-claw-den`

**Pause the queue first, or the scheduler relaunches the arm inside its wave.** Set `paused: true` in
`snek3/desktop/config/runtime.json` on the `ops` branch, push, and `trigger` — see the
`queue-batch` skill. Then:

```
ssh the-claw-den "ps -Ao pid=,etime=,command= | grep '[t]rain.py'"
ssh the-claw-den 'kill -9 <pids>'
```

Unpause when the box should take work again. Do not restart the daemon to stop a job: jobs are
launched detached with `setsid` and `KillMode=process`, so a restart leaves them running and the
daemon re-adopts them by pid.

**A killed arm short of its cap is relaunched by its scheduler**, up to three times inside its wave,
resuming from `resume.pt` -- unless the box is paused, in which case the relaunch waits for the hold to
lift. So "stop an arm for good" is: pause, kill, and then either remove its spec from `ops` (the
`queue-batch` worktree) or release its wave (below) before unpausing. A close-out killed by hand is
relaunched twice and then marked `.failed-<label>` beside the batch's specs in the box's queue mirror;
delete the marker to retry. Say which you did in your report.

## 2b. The wave's claim

**A wave belongs to the box that claimed it** (`tools/claims.py`), and the claim outlives its processes:
the box resumes the wave at its next scheduler start. Stopping the processes is therefore not stopping the
work unless you also decide what happens to the claim:

| you want | do |
|---|---|
| the wave to resume here later (a deploy, a reboot) | nothing: the next scheduler on this box adopts or relaunches its arms |
| the wave to run on the other box instead | kill its arms here, then `PYTHONPATH=. python -m tools.claims release <batch>-w<N>`: the arms return to the pool unclaimed, the other box claims them as a fresh wave (numbered past this one), and **they retrain from scratch** -- a box cannot resume another's checkpoints |
| the arms never to run again | kill them, remove their specs from `ops` (`queue-batch`), release the wave |

`PYTHONPATH=. python -m tools.claims show` prints who holds what. A claim whose box has gone quiet is
also named under `attention` after two hours (`queue-batch`, "Pin, unpin, release").

## 3. After stopping

Update `docs/charts.md` and `docs/results.md` in the same pass, and write the batch's `Learned`
paragraph in `docs/runs.md` saying it was stopped and why, under its `Why`. **Without the rationale a
later session cannot tell a surprising result from an arm that was never going to answer anything.**
