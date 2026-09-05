# The desktop — `the-claw-den`

A stdlib-only systemd daemon on a dedicated Linux box that runs trainings and evals unattended. It
**imports nothing from this project** and **schedules nothing itself** (since 2026-09-05): it mirrors the
specs on `ops` into a local queue, starts `tools/scheduler.py` — the same scheduler the laptop runs — over
that queue, and publishes what the scheduler finishes. It talks to the laptop only through four
single-writer git branches. That decoupling is the design's best property: the bus works from anywhere,
`ssh` is a convenience, and the daemon cannot be broken by a change to the trainer or the scheduler.
The design and its decisions are [`../plans/scheduler.md`](../plans/scheduler.md).

**snek3's daemon owns the box as of 2026-08-28**, replacing snek2's. The two eras share the `ops`
branch, which is why every spec carries a required `project` field.

| branch | writer | payload |
|---|---|---|
| `ops` | **laptop** | `snek3/desktop/queue/pending/*.json` specs, `snek3/desktop/config/runtime.json` |
| `ops-status` | **desktop** | `status.json` — heartbeat, running jobs, ledger, `at_a_glance` |
| `results` | **desktop** | `results/<job-id>/*` artifacts |
| `laptop-status` | **laptop** (`tools/laptop_status.py`, from the queue driver) | the laptop's `status.json`, same `at_a_glance` shape; the daemon reads it each network cycle and publishes it inside its own as `at_a_glance.laptop_running`, `laptop_queued`, `laptop_iso` |

One writer per branch, so every push is `--force-with-lease` and nothing ever merges. The laptop's queue
reaches `ops-status` *through* the daemon rather than by the laptop writing there, for exactly that reason
(2026-09-05).

## Queue a job

The procedure — the `ops` worktree, validating a spec against `parse_job` before pushing, the push
and the trigger — is [`../skills/desktop-batch`](../skills/desktop-batch/SKILL.md).

**Pushing to `ops` starts real work on another machine**, so it needs the user's approval for *that*
job — see the root [`CLAUDE.md`](../../CLAUDE.md). The trigger is what makes it start now rather than
within ten minutes:

```
ssh the-claw-den 'Snek/snek3/desktop/trigger'
```

`queue/examples/` holds one worked spec per job type. The fields:

| field | required | means |
|---|---|---|
| `project` | **yes, no default** | must be `snek3`. The guard against `ops`'s ~150 retired snek2 specs |
| `id` | yes | unique; the ledger key and the log name. `b<n><letter>-...` groups arms into a batch |
| `type` | yes | `train`, `smoke`, `benchmark`, `eval` — or the two **actions**, `deploy` and `restart`, which the daemon runs itself on the poll that sees them (below, "Deploy over the bus") |
| `policy` | train/eval | the checkpoint directory name |
| `policies` | eval only | a **wave**: every arm of a batch in one process |
| `max_steps` | no | `SNEK_MAX_STEPS`. Defaults per type for smoke/benchmark |
| `env` | no | any `SNEK_*` knob; wins over the runtime defaults. See [`../docs/running.md`](../docs/running.md) |
| `selector`, `episodes` | eval, no | **omit them.** Absent means `tools/closeout.py`'s own defaults, which *are* the protocol |
| `eval_shards` | no | shard processes for this wave; defaults to the runtime config's 16 |
| `priority` | no | lower runs first. Default 100; the auto-queued passes are 10 (stage B), 11 (hof5000), 12 (hof30k) |
| `label` | no | one line for `at_a_glance` |
| `notes` | no | free text, for the reader |

A malformed spec is **recorded against its filename and skipped**, never raised into the loop, so one
bad commit cannot stop the box.

## Read the box — and the laptop

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

`at_a_glance.running` / `queued` / `attention` are the box; `laptop_running` / `laptop_queued` are the
laptop's queue driver, as of `laptop_iso` (the laptop's own clock). The driver's last publish before it
exits is **empty**, so empty lists mean an idle laptop, and lines under a `laptop_iso` hours old mean the
driver died — the two are meant to read differently. `python -m tools.laptop_status` on the laptop
publishes an empty status by hand after a killed driver. The laptop's lines are as fresh as the box's last
network cycle: `trigger` refreshes both now.

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
and an `attention` list for anything needing a human — a failed pass, a stuck push, a rejected spec,
a scheduler that exited non-zero. While the scheduler runs, the lines are **its own** (`runs/.live/
.status.json` on the box, the same file its chart window follows); the daemon adds its attention lines,
the hold notice, and the laptop's lines. Underneath:

| key | means |
|---|---|
| `scheduler` | `alive`, `pid`, `spawned`, `last_exit`, `log`, and `status_iso` — the scheduler's own timestamp |
| `running` | the scheduler's running jobs: `id`, `type`, `policies`, `step`/`max_steps` for an arm |
| `ledger` | `{job id: queued / running / done / failed}`, derived for the tools that read one (`tools/progress_update.py`, `tools/viewer_manifest.py`); `done` means published to `results` |
| `runtime`, `config_notes`, `disk_free_gb`, `head`, `load_avg` | the box |

**The scheduler's state is the filesystem, and so is the daemon's memory of it.** An arm is finished
when its `_evals.json` has reached the cap, a pass when its merged file exists, a failed pass leaves a
`.failed-<id>` marker beside its spec (`tools/scheduler.py`). The daemon keeps one small `state.json`
beside the ledger: the scheduler's pid and boot id, the queue it was started on, the ids it saw running
and the ids it has published. A reboot is detected by the boot id: the scheduler is gone, so the next
poll starts a fresh one and every arm resumes from its checkpoint, every pass from its shard files. Old
`interrupted` records and the boot-id check per job are gone with the per-job ledger.

## Tune it while it runs

`config/runtime.json` is committed on `ops` and re-read every network cycle. **A malformed file is
rejected whole and the last known-good config kept** — the box has no ssh backstop in normal
operation, so a bad commit must never be able to wedge it, and a *partially* applied config is worse
than a rejected one because it looks like it worked. Values are then clamped to `host.env`'s
`HARD_MAX_*`, with anything clamped noted in `status.json` rather than raised.

| knob | default | notes |
|---|---:|---|
| `max_trainers` | 8 | the scheduler's `--wave` and `--max-trainers`: arms per wave, and the cap counting anything else on the box |
| `eval_shards` | 16 | the scheduler's `--shards`: the stage-B shard pool, spread over the wave's arms |
| `poll_seconds` | 30 | the local half: mirror the queue, relay the hold, publish what finished. Off-network, so it stays fast |
| `git_seconds` | 600 | the network half: one fetch, one status push, one retry of any local-only commit |
| `torch_threads` | 1 | measured, not cautious. `SNEK_TORCH_THREADS` |
| `omp_num_threads` | 1 | same, for numpy's BLAS underneath |
| `nice` | 0 | |
| `disk_min_gb` | 5 | refuse to launch below this much free |
| `paused` / `drain` | false | finish what is running, start nothing new. **Relayed at once** as the scheduler's hold marker, `desktop/runs/.live/.paused` |
| `auto_stage_b` | true | the chain: every wave is followed by its stage B, hof5000 and hof30k — see below. Off is the scheduler's `--no-stage-b` |
| `viewer` | true | the chart window; off reaches the scheduler as `SNEK_CHART_WINDOW=0` |

**Every knob but `paused`/`drain` applies when the scheduler is next started**, because they reach it
as flags and environment at spawn. A scheduler is started when the mirrored queue changes, when a hold
is lifted, or when `trigger` asks and none is running; a change of `max_trainers` alone does not
restart a running one — pause, wait for the wave, unpause, and the next scheduler has the new value.


### The automatic chain: training → stage B → hof5000 → hof30k

Every batch gets all three passes without anyone queueing them, and **the scheduler runs the chain**
— on the laptop and on the box, the same code (`tools/scheduler.py`; before 2026-09-05 the daemon had
its own copy, which minted the pass jobs into its ledger). A batch's arms run in waves of
`max_trainers`; after each wave, `tools.closeout <arms> --pass stageb`, then `hof5000`, then `hof30k`
over that wave's arms, with the ids `b15-stageb`, `b15-hof5000`, `b15-hof30k` for wave 1 and `-w2`,
`-w3`, … after. An `eval` spec in the queue runs once, after the batch's waves, as the command it
spells, and a pass the arm already has the file for is skipped.

| pass | id | selects | episodes | seed | writes |
|---|---|---|---|---:|---|
| stage B | `<batch>-stageb` | `screen:97` — every checkpoint at ≥97/100 in stage A | 500 | 0 | `runs/<arm>_checkpoint_evals.json` |
| hof5000 | `<batch>-hof5000` | `above:99` — stage-B rows at ≥99/500 | 5,000 | 0 | `…_checkpoint_evals_hof5000.json` |
| hof30k | `<batch>-hof30k` | `above:99:hof5000` — hof5000 rows at ≥99/5,000 | 30,000 | 7 | `…_checkpoint_evals_hof30k.json` |

**The numbers are in neither the daemon nor the scheduler.** They are `tools/closeout.py`'s `PASSES`;
the scheduler passes the name and the close-out's preset does the rest — snek2's daemon carried five
protocol numbers as a second copy and they drifted. A pass with no candidates in an arm is not an
error: the shards exit at once and an empty labelled file is written, so the next pass selects nothing
in turn.

**Success is required at every hop.** hof5000 selects from the file stage B wrote, so a failed stage B
earns no hof5000. A failed pass is marked `.failed-<id>` beside the batch's specs in the local queue
(`desktop/queue-local/<batch>/`), named under `attention`, and never retried on its own; delete the
marker and the chain resumes from there at the scheduler's next run.

**The scheduler also starts the stage-A eval workers** a wave needs (`tools.eval_queue`), sized from
the wave's specs, before it launches the arms — so a wave never trains against an empty worker pool.

`status.json`'s queue shows a batch's owed passes as **one line**, `b16 evals | … | queued (8 arms)`;
the running line names the pass that is running.

### One wave at a time is the scheduler's, and no limit enforces it

The scheduler launches a wave, waits for every arm of it, runs the wave's passes, then launches the
next wave. Nothing new starts until the wave and its passes are done, a freed slot is never backfilled
mid-wave, and trainings and passes therefore never overlap. `max_trainers` only caps how *wide* a
wave is and how many trainers the box may hold counting any launched by hand; `eval_shards` is the
shard pool a pass spreads over its arms. It was 4 because snek2's TensorFlow workers cost 230 MB of
arena each; on this code a trainer is 290 MB and the ceiling is threads (Ryzen 7 9700X, 8 cores / 16
SMT threads), so 8 trainers beside the stage-A queue's workers is the same shape as the 16-shard eval
wave. Raised to 8 on 2026-08-29.

**Cores are the binding constraint, not memory.** Measured 2026-08-28: an eval shard peaks at 202 MB
and a trainer at 290 MB, so a full box is 4.4 GB of 15,030. 16 shards at one intra-op thread each is
the measured optimum, and 18 loses 6-10%; `eval_shards` is clamped to `HARD_MAX_EVAL_SHARDS` (16).

**Removed settings, kept as history so nobody re-adds them.** `max_evals` and `HARD_MAX_EVALS` went on
2026-08-29 (an eval job is a whole wave, so the count's only legal value was 1); `clamp_total_shards`
went with them; `HARD_MAX_TRAINERS` went because wave width is not a safety property and a silently
clamped request ran 4 arms while looking like 8. A `runtime.json` still naming `max_evals` is rejected
whole and the last-known-good config kept, with a note in `status.json`. Extra keys in `host.env` are
ignored, so an unchanged box keeps working.

`git_seconds` is separate from `poll_seconds` because at 30 s the box made ~2,880 fetches and ~2,880
pushes to github a day. 600 s cuts both to 144 while costing nothing locally, and `trigger` covers the
case where a batch should start *now*.

## The chart window

**One window per box, owned by the scheduler** (2026-09-05; the root `CLAUDE.md` has the history of the
three designs before it). The scheduler opens it at its first launch — `tools.chart_viewer --follow
desktop/runs/.live/.status.json` — repoints it at each wave's arms and then at each pass's, and closes
it when it exits. Nothing else on the box opens a window: not the arms, not the close-outs, not the
daemon. So "a pass is running and there is no window" has one cause now, the scheduler's viewer died
or was closed, and one remedy:

```
ssh the-claw-den 'cd Snek/snek3 && SNEK_RUNS_DIR=~/Snek/snek3/desktop/runs PYTHONPATH=. ~/miniconda3/envs/snek3/bin/python -m tools.scheduler --reopen-window'
```

which drops a request file the running scheduler picks up within a poll. **A window you close stays
closed until the scheduler's next launch** (user, 2026-09-05), and closing it is safe: the window reads
a status file and some PNGs, and nothing in a trainer reads it, waits on it or reopens it. A daemon
restart does not touch it (the scheduler is detached); a scheduler restart replaces it (the old one is
killed by pid, after checking the pid is still a viewer).

**The daemon owes the window two things.** `DISPLAY` and `XAUTHORITY` from `host.env` reach the
scheduler's environment — the daemon runs outside the graphical session — and `runtime.json`'s
`viewer: false` reaches it as `SNEK_CHART_WINDOW=0`. Those two keys are the only optional ones in
`host.env`; without them the scheduler runs headless and no window appears.

**Why not the daemon's own window**, the first design: it drew a fixed grid of the wave it launched, had
to be reopened whenever an arm joined or finished, and left the laptop with no window at all. Why not
the arms' own, the second: eight arms racing for one `flock` opened five windows on 2026-08-29 and a
stale winner held the slot for 15 hours on 2026-09-03. The scheduler is the one process that knows what
is running, on both boxes, so it holds the one handle.

**The window is sized from the display, not from a number of inches.** It fills 95% x 88% of whatever
screen it opens on — 3086x1951 on this box's 3840x2160 panel. `SNEK_CHART_WINDOW_SCALE` is a fraction of
that budget and `SNEK_CHART_WINDOW_MAX_PX` caps the width, in the scheduler's environment.

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
`PYTHON_BIN` is the env python, and it is what runs the scheduler and everything under it.
`KillMode=process` means a deploy can restart the daemon mid-batch — the scheduler is launched
detached with `setsid`, it carries on, and the daemon re-adopts it by pid on the next poll. Its log is
`LOG_DIR/scheduler-<stamp>.log`; the arms' and passes' logs are the scheduler's, under
`desktop/runs/../logs` as on the laptop.

**Every job on the box writes under `snek3/desktop/runs/`, gitignored, never `snek3/runs/`** (2026-09-03).
`launch.runs_dir(host)` is the one place the path is defined; it goes to the scheduler as
`SNEK_RUNS_DIR` and from there to every arm and pass, and the daemon collects a finished job's artifacts
for `results` from the same place. The mirrored queue is beside it, `snek3/desktop/queue-local/`, also
gitignored. So the box's checkout of master holds nothing under a path master tracks, and the laptop is
free to commit every chart. A tool run by hand on the box (`tools.scheduler --reopen-window`,
`tools.closeout`) needs `SNEK_RUNS_DIR=~/Snek/snek3/desktop/runs` exported or it looks in the empty
`runs/`.

### Deploy over the bus

```
snek3/desktop/queue_action deploy            # from the laptop: fetch + ff-merge on the box, restart iff desktop/runner or systemd changed
snek3/desktop/queue_action deploy --restart  # restart regardless
snek3/desktop/queue_action restart           # restart only
```

**Since 2026-09-05 a deploy is a job type, not an ssh.** The script commits a `deploy-<stamp>.json`
(`type: deploy`) to `ops`, triggers, and waits for the ledger. The daemon runs actions in the poll that
sees them: **ahead of dispatch, beside running jobs, and under a pause** (a pause is how a deploy that
must not race a wave is done). A deploy runs the box's own `desktop/deploy`, records `head_before`,
`head_after`, the script's last lines and `rc` in the ledger, and restarts when the merge touched
`desktop/runner/` or `desktop/systemd/` (or the spec says `"restart": true`). A **restart is the daemon
recording the action as done, publishing, and exiting 0**; systemd's `Restart=always` relaunches it in
10 s on the code now in the checkout, and it re-adopts the running scheduler by pid. No sudo anywhere, which is
the point: the laptop's permission classifier refuses `sudo` over ssh, and the box's passwordless sudo
was never the limit. `status.json` now carries `head`, the commit the box runs, so the laptop can check
a deploy landed. A deploy that exits 3 (a differing JSON; nothing touched) is `failed` in the ledger and
under `attention`, and is never retried: fix it and queue a new id.

**The one-time bootstrap.** A daemon older than this change marks a `deploy` spec malformed (`failed`,
unknown type) and the new code then never runs it, so the first daemon that understands actions has to
be started the old way: `ssh the-claw-den 'Snek/snek3/desktop/deploy'` and then, by the user at the
prompt, `ssh the-claw-den 'sudo systemctl restart snek3-runner'`.

By hand, the old way still works: `snek3/desktop/deploy` over ssh (a fast-forward that settles any
leftover from before the move) plus `sudo systemctl restart snek3-runner` when `desktop/runner/*`
changed. Piping it to `tail` hides the failure and its exit code.

## Reach the box from outside the home LAN

**Status 2026-09-04: done and verified end to end.** `ssh the-claw-den` resolves by mDNS on the home LAN and
the box sits behind the router's NAT, so before this the alias, `deploy`, `trigger`, `journalctl` and the
live-chart `rsync` were home-LAN only; the git bus worked from anywhere regardless. The
constraint is **nothing new on the laptop** (Tailscale was removed 2026-08-13 for that reason), so the path is
plain OpenSSH through a port forward. The box itself may run anything.

| # | where | change | done |
|---|---|---|---|
| 1 | desktop | `ssh the-claw-den 'bash -s' < snek3/desktop/harden_ssh.sh` — rewrites the sshd hardening file (key-only, no root, `AllowUsers claw`, `MaxAuthTries 3`) and **asserts the effective config with `sshd -T`, refusing to touch the firewall if anything is not key-only or `authorized_keys` is empty**; then fail2ban (5 failures / 10 min → 1 h, LAN exempt) and ufw with the LAN trusted and port 22 rate-limited | **done 2026-09-04**: all seven asserts ok, fail2ban jail `sshd` active, ufw active, fresh login and mDNS verified afterwards |
| 2 | router | DHCP reservation for the desktop at `192.168.0.79`; port forward WAN **2222** → `192.168.0.79:22` TCP; TP-Link DDNS as **`clawden.tplinkdns.com`** | done 2026-09-04 |
| 3 | router | confirm the WAN status page shows the same address as `curl -4 ifconfig.me` from the box (75.164.174.153 on 2026-09-04). A 100.64–100.127 or 10.x address there means carrier NAT and the forward cannot work; the fallback is then a reverse tunnel from the box to a small VPS | done: WAN page shows 75.164.174.153, same as the box sees |
| 4 | laptop | the `Match` block below in `~/.ssh/config`, so the same alias falls back to the DDNS name when mDNS does not answer | done; backup at `~/.ssh/config.bak-2026-09-04` |

```
# BEFORE the Host block: ssh keeps the first value set for an option, so a Match
# placed after it is silently ignored (found on the first hotspot test, 2026-09-04).
Match host the-claw-den !exec "nc -z -w 1 -G 1 the-claw-den.local 22 2>/dev/null"
  HostName clawden.tplinkdns.com
  Port 2222

Host the-claw-den
  HostName the-claw-den.local
  HostKeyAlias the-claw-den
  User claw
  IdentityFile ~/.ssh/snek_desktop
  IdentitiesOnly yes
```

`HostKeyAlias` is already in the alias, so the known-hosts entry is shared by both paths and the fallback
does not prompt. Verified 2026-09-04: on the LAN `ssh -G the-claw-den` picks `.local:22`; the forwarded path
logged in by name and by address (the BE600 does NAT loopback, so it is testable from inside too), and
**from a phone hotspot the plain alias logged in through the forward** — the real off-LAN test. A DDNS name
that `dig` resolves but `ssh` does not is the Mac's negative resolver cache from before the record existed —
`dscacheutil -flushcache`. Every existing command then works unchanged from anywhere. What stays home-LAN is the box
reaching the laptop (only the diagnostic ping in `plans/laptop-wifi.md`), since the laptop's address is
whatever its current network gave it.

**A `deploy` job type covers the commonest off-LAN need without ssh at all** — done 2026-09-05,
`queue_action` above. Named actions only (deploy, restart), never arbitrary shell: anything on that branch
runs as `claw` on the box.

## What the port changed, and why each one is an incident

| change | the incident behind it |
|---|---|
| a required `project` field | `ops` holds ~150 retired snek2 specs whose `script` would resolve to a TensorFlow trainer that does not exist here |
| one eval stage, not two | `training → closeout → HOF` becomes `training → stage B`, so the `auto_hof` hop, the `-hof` id handling and the legacy phase branch all go. **Partly reversed 2026-09-04:** the chain now runs hof5000 and hof30k after stage B — but as deeper *re-measures* of the same protocol's file, named passes of one close-out, not snek2's tiered selection with its own gates and episode counts |
| `_ensure_viewer` shrinks from ~200 lines to ~50 | snek2 needed a process registry, an `O_EXCL` claim lock, a grace period, zombie detection and a dedupe because four peer trainers each tried to open one shared window knowing nothing about each other. One process starts the arms here, so one process starts the window |
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
