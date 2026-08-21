# Desktop runner — unattended training/eval driven by git

A dedicated box (Ryzen 7 9700X, 8c/16t, **15,030 MB RAM** per `free -m`) runs snek trainings and
evals on its own, driven entirely through git. You never log in to it in normal
use: you commit a job, it runs it, it reports back.

**It is set up and live** — `the-claw-den`, user `claw`, daemon `active`, verified
2026-08-08. Setup details and the backstop SSH command are in
[`SETUP.md`](SETUP.md); nothing there needs running again.

## Reaching the box

**The git bus works from anywhere; the SSH backstop is home-LAN only.** Tailscale was
removed on 2026-08-13, so `ssh the-claw-den` resolves by mDNS (`the-claw-den.local`)
and only works on the home network. The alias, the no-config fallback command and the
key-recovery path are in
[`SETUP.md`](SETUP.md#laptop-side-ssh-access-and-how-to-rebuild-it).

Off the home network you can still do almost everything, because none of it touches
SSH — queue jobs, edit `runtime.json`, read `status.json`, pull `results`. What you
cannot do until you are home is [deploy code](#deploying-a-code-change-to-the-desktop),
read `journalctl`, or sample `free -m`.

**So a box that looks unreachable from the office is usually just off-LAN, not broken.**
Check the `iso` heartbeat in `status.json` before assuming anything is wrong — if it is
current, the daemon is fine and only your shell access is absent.

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
| `ops-status` | desktop | `status.json` — heartbeat, running jobs, steps/sec, ledger (incl. the queue) |
| `results` | desktop | each finished job's `runs/<policy>*` artifacts, pushed at completion |

`master` stays your curated log; the bus branches carry the churn. The desktop
reads `ops` straight from the fetched ref (never checks it out) and writes its two
branches through dedicated worktrees, so nothing races.

The daemon (`runner/`) polls every `poll_seconds`: fetch → re-read `runtime.json`
→ reap finished jobs → launch pending ones up to the concurrency limits → publish
status. Jobs are launched **detached**, so a daemon restart never kills a running
trainer; they self-terminate via `SNEK_MAX_STEPS`. A local ledger makes launches
idempotent across restarts (a job id never runs twice).

## The eval chain: training → closeout → HOF re-measure

The daemon synthesizes two follow-on evals, each off a ledger marker, never as spec files:

| link | trigger | synthesized job | what it runs |
|---|---|---|---|
| **closeout** | a training finishes OK (`auto_closeout`) | `<policy>-closeout`, priority 10 | `eval_checkpoints.py <policy> top20` at `EVAL_MIN_ACHIEVABLE=96` |
| **HOF re-measure** | a closeout finishes OK (`auto_hof`) | `<policy>-hof`, priority 11 | `eval_checkpoints.py <policy> above:98` at 500 episodes, flat |

The closeout pins its abandonment gate at **`EVAL_MIN_ACHIEVABLE=96`** (`CLOSEOUT_EVAL_ENV`),
overriding whatever the training env carried rather than falling to `eval_checkpoints`' default of
95. It stays **below** the HOF gate of 98 on purpose: HOF's `above:98` selects only rows the
closeout measured at full length, and a gate above 98 would abandon exactly those checkpoints and
starve the re-measure.

**A reboot mid-eval resumes rather than restarts.** When `_reattach` marks a running closeout or
HOF `interrupted` (boot-id mismatch) and the job is re-synthesized on the next dispatch, its env
carries **`EVAL_RESUME=1`** — so the relaunch keeps the full-length rows already on disk and
re-measures only the checkpoint that was in flight when the power was lost, instead of redoing the
whole pass. A first-time eval (no prior ledger record) never gets the flag, so it always measures
from scratch. The flag is only set on the `interrupted` path, so a `done`/`failed` eval is never
resumed from a stale file.

The HOF re-measure reconfirms the checkpoints a closeout already found excellent. `above:98` reads
the closeout's own `runs/<policy>_checkpoint_evals.json` and takes every checkpoint measured there
at **≥98%**; the job runs those at **`EVAL_EPISODES=500 EVAL_SCREEN_EPISODES=0 EVAL_INDEPENDENT=1
EVAL_MIN_ACHIEVABLE=98`** and writes **`EVAL_OUT_SUFFIX=_hof500`** — a separate file, so it never
clobbers the closeout's result and the closeout's 100-episode rows are never mistaken for finished
HOF work. Worker count is the runtime's `eval_workers` (the usual 4), like any eval.

**‡ A batch closes out as *one* wave, and what defines "one" is the measurement-relevant env**
(2026-08-21). `_auto_closeout_jobs` groups the pending markers by `(batch, closeout_group_env(env))`,
where `closeout_group_env` keeps only `EVAL_RELEVANT_ENV` — the shaping and reward knobs an eval can
actually see. It used to key on the *whole* inherited training env, so `b45`, whose four arms differ
only in `SNEK_SEED`, split into three waves: `b45-closeout` `{a,c}`, `b45-closeout-w2` `{b}`,
`b45-closeout-w3` `{d}`. Two costs, and the second is the real one:

- The chart window showed 2 panels, then 1, then 1 — a finished arm's chart was gone before anyone
  came back to read it (see `eval_batch_pngs` and the `keep_batches` exemption in
  `eval_plan.archive_existing_eval_pngs`).
- **Three sequential waves of 2/1/1 arms measure a batch at a quarter of the intended 4 lanes.** For a
  continuation batch, whose close-out is already priced in hours, that is the difference between one
  pass and three.

The wave runs under `agreed_env(...)` — every key the group's arms agree on — so a looser group key
never attributes one arm's seed or learning rate to its siblings. `runner.EVAL_RELEVANT_ENV` is a
**copy** of `eval_wave.EVAL_RELEVANT_ENV` (the runner cannot import that module: TensorFlow), and
`tests/test_runner.py::test_eval_relevant_env_matches_eval_wave` parses the real tuple out of
`eval_wave.py` and fails if the two drift.

Three properties that matter:

- **The HOF never races its own closeout.** The `hof: pending` marker is set only when the closeout
  *reaps*, so its result file is complete (atomic `os.replace`) before the HOF job can be synthesized.
- **No HOF-of-a-HOF, no HOF off a hand-queued eval.** The trigger keys on the `<policy>-closeout`
  id the daemon itself mints; a `<policy>-hof` id and a manual `top20` eval both fail the check.
- **Most arms produce an empty HOF.** If no closeout checkpoint reached 98%, `above:98` selects
  nothing and the job exits 0 (marked `done`, not `failed`) — the common case.

Turn either link off in `runtime.json` (`auto_closeout` / `auto_hof`); both default on. The
`_hof500` files come back on the `results` branch like any other artifact. **Promotion into
`hallOfFame/` is still the manual, verified process** in [`hallOfFame/README.md`](../hallOfFame/README.md)
— this automation only produces the re-measurement; it never copies a checkpoint in.

## Rebooting the box, and what recovers by itself

**The safe way is to drain first**: set `"drain": true` in `runtime.json` on `ops`, wait for the
running wave to finish, then reboot. Nothing below applies if you do that.

If the box goes down mid-job — a reboot, a power cut, an OOM — **the data is safe and the job
recovers itself, but you lose the steps since the last checkpoint.** Everything durable is written
`<path>.partial` then `os.replace`d, which is atomic, so a reader never sees a torn file and a
crash leaves either the old complete file or the new one:

| what | written every | at risk |
|---|---|---|
| agent checkpoint | 1,000 steps (score ≥ 40) | ~10 s of training |
| replay buffer (~20 MB) | 10,000 steps, same gate | ~100 s, and only warm-starts the resume |
| `runs/<policy>_evals.json`, `.md`, `.png` | every eval | one eval |
| `_checkpoint_evals.json` | every checkpoint measured | one checkpoint's episodes |
| the runner's ledger | every state change | nothing |

`max_to_keep=10000` means no checkpoint is ever evicted, so even a damaged newest one leaves
thousands of earlier ones intact — and `initialize_or_restore` fails loudly rather than restoring
something half-written.

**On the next boot the daemon comes back (`Restart=always`, `WantedBy=multi-user.target`) and
classifies each job the ledger says was running, by comparing the record's boot id against
`/proc/sys/kernel/random/boot_id`:**

| record's boot | pid | verdict | what happens next |
|---|---|---|---|
| **differs** (machine rebooted) | not consulted | `interrupted` | relaunched on the next dispatch — a training **resumes from its checkpoint** (`SNEK_MAX_STEPS` is absolute), an eval re-runs |
| same (daemon restarted only) | alive | `running` | re-adopted by pid, exactly as before |
| same | dead | `done` | the detached job ran to its own end; a training earns its closeout |

So a reboot costs wall clock and nothing else. Watch `restarts` in the ledger entry: it counts
interruptions and carries across the relaunch, so a box rebooting in a loop shows a climbing
number instead of looking freshly started each time.

**Abandoning a wave: pause first, then `kill -9`, and the exit code is what suppresses the chain**
(done 2026-08-14, when batch 27 had to be thrown away mid-run for
[the perfect-game counter bug](../hyperparamTuning/findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter)).
Three things that pass unnoticed if the order is wrong:

- **Pause before killing.** Slots free within one poll, and an unpaused daemon fills them from
  `queue/pending/` immediately — which is how the *next* batch inherits whatever made you abandon
  this one. Set `"paused": true` in `runtime.json`, wait for `status.json` to echo it back
  (`runtime.paused` is published, so this is checkable), and only then kill.
- **`kill -9`, not `kill`.** The trainer does not stop on SIGTERM — TF-Agents' worker layer swallows
  it — and it keeps stepping while you assume it is shutting down. Confirmed on both hosts the same
  day: four laptop arms and four desktop arms all advanced ~25k further steps after SIGTERM.
- **A non-zero exit marks the job `failed`, and `failed` suppresses the eval chain.** The four
  killed arms' `-closeout` and `-hof` entries vanished from the ledger, because `_auto_closeout_jobs`
  fires off a `done` marker. That is the outcome you want when the run is being discarded — a
  graceful stop would instead have earned four close-outs of arms you no longer trust. It also means
  `failed` in the ledger does not always mean something broke; check whether a human killed it.

Forked self-eval workers outlive the parent for a moment, so finish with `pkill -9 -f "snek2.py <prefix>"`
and confirm with `free -m`. **`pgrep -f snek2.py` will match its own shell** — the two "leftovers" after
that cleanup were `bash -c ps -o … $(pgrep …)`, not trainers.

**The boot id is load-bearing in two ways, and both were silent bugs before 2026-08-13.** Without
it a dead pid after a reboot read as `done`, so (1) a truncated training was published to `results`
as a finished arm *and* spent its `closeout: pending` measuring the partial checkpoint set, and
(2) a truncated closeout marked itself terminal, which made `_auto_closeout_jobs` skip that arm
forever — it was never evaluated again, and nothing said so. It also closes a pid-reuse hole: pids
restart low after a boot, so a stored low pid could match an unrelated process, and re-adopting
that phantom would idle the box permanently behind the wave barrier.

**A killed git leaves a lock, and that used to freeze the heartbeat.** `publish_status` commits and
pushes every poll, so the daemon is inside a git write for a slice of every 30 seconds; killed
there, the `index.lock` outlives it and every later `git add` fails. The daemon kept running and
kept dispatching, but `status.json` never updated again — indistinguishable from a dead daemon.
`gitbus.clear_stale_locks` now sweeps locks older than 60 s from both bus worktrees before each
write. The age gate is what makes it safe: a live git holds the lock for milliseconds.

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
| `auto_closeout` | a finished training auto-queues its `top20` closeout eval (default on) |
| `auto_hof` | a finished closeout auto-queues a 500-episode HOF re-measure of its ≥98% checkpoints (default on) |

A malformed `runtime.json` is rejected — the daemon keeps the last good config
and reports the error in `status.json`, so a bad commit can't wedge the box.

**Watch it** — read `status.json` on `ops-status`:

```
git fetch origin ops-status && git show origin/ops-status:status.json
```

`running` is what is on the box now; the **`ledger`** map carries the run history and, **at its
end, the pending queue** as `queued` entries **in the order the wave-barrier scheduler will
actually launch them** — including the **closeout eval each queued training will spawn** and the
**HOF re-measure each closeout will spawn**, slotted where they will run (a closeout has priority
10 and a HOF 11, both below a training's 100, so a batch's closeouts then its HOFs always form the
next waves before the following training batch). So the tail of the ledger reads `batchA
trainings → batchA closeouts → batchA HOFs → batchB trainings …`, even though those closeout and
HOF specs do not exist as files yet. A launched job moves from `queued` to `running` the same
poll; a malformed spec never shows as `queued` — it lands in the ledger as `failed`.

**Pull results** — `git fetch origin results && git checkout results -- results/<job-id>`.

## Measured capacity — memory is the limit

**Measured on the box 2026-08-08, four concurrent evals of one checkpoint:**

Total RAM is **15,030 MB** as `free -m` reports it (`MemTotal` 15,390,836 kB, ~14.7 GiB).

| config | spawned workers | peak RAM used | verdict |
|---|---|---|---|
| 4 evals × **10** `EVAL_WORKERS` | 40 | **12,770 MB** of 15,030 | **at the ceiling** — only ~2.3 GB free, below the ≥3 GB target |
| 4 evals × **4** `EVAL_WORKERS` | 16 | **7,296 MB** of 15,030 | comfortable, and the current setting |

No OOM kills occurred in either run, so 12.8 GB is a real measurement rather than a
survived near-miss — but surviving is not the bar.

**The dial is total spawned workers, not the number of eval jobs.** Each standalone eval
worker is its own process holding its own TensorFlow arena at ~230 MB, so what matters is
the product `max_evals × eval_workers`. Three points fix the scale:

| total spawned workers | what happens |
|---|---|
| ~52 | **OOM-killed** in a scaling test |
| ~40 | survives, but only ~2.3 GB headroom — a **ceiling, not an operating point** |
| **≤32** | the operating band: `HARD_MAX_EVALS=4` at `eval_workers` ≤ ~8, keeping ≥3 GB headroom |

Live setting is 4 × 4 = **16**, comfortably inside. `HARD_MAX_EVALS=4` guards the job count;
nothing guards the *product*, so that is the number to check by hand before raising either knob.

**Cores are not the constraint, and this inverts the laptop's rule.** On the laptop
`EVAL_WORKERS` is close to free and lowering it wastes CPU
([eval cost](../../CLAUDE.md)); here worker count is the memory dial. Raise `max_evals` or
`eval_workers` only with a `free -m` measurement in hand.

**Training self-eval workers are different — they are forked, not spawned**, so Linux
COW-shares the parent's TF pages and they are nearly free: 4 trainers × 10 self-eval
workers ≈ 4.2 GB total. Only the standalone `eval_checkpoints.py` workers cost ~230 MB each.

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
| desktop | `max_trainers` (≤ `HARD_MAX_TRAINERS=4`), `max_evals` (≤ `HARD_MAX_EVALS=4`) | **`git fetch origin ops-status &&`** `git show origin/ops-status:status.json` |

**Neither check covers the other host**, so a status report that says "4 arms
running" must say *which box*. The desktop's `counts` and `running` fields in
`status.json` are the only authority for its side.

**The `git fetch` is load-bearing and omitting it has caused three false alarms**
(2026-08-12, twice on 2026-08-17). `git show origin/ops-status:…` reads a *local*
remote-tracking ref, so without a fetch you get an old snapshot whose embedded
`iso` timestamp then reads as a dead daemon. A stale-looking heartbeat is your own
ref until you have fetched and re-read it.

## ‡ A `done` in the ledger does not mean the results were published

**`publish_results` has no retry, and this box's DNS for `github.com` flaps.** The job finishes, its artifacts
are written to `~/Snek/snek2/runs/`, the ledger records **`done`**, the push fails once with
`ssh: Could not resolve hostname github.com: Temporary failure in name resolution` — and nothing tries again.
Measured 2026-08-18: **14 `publish_results` failures since 2026-08-17** (plus 122 for `publish_status`, which
*does* retry every poll and so recovers on its own).

**Nothing is lost, because a failed push leaves the commit local and the next successful results push carries
the backlog with it.** That is why `b35`'s and three of `b37`'s close-outs eventually appeared while all four of
`b40`'s HOF-500 files and `b40b`'s entire close-out did not: no results job completed after 07:27, and the queue
then went empty, so nothing came along to carry them.

**The trap is that absence looks like a result.** Most HOF jobs legitimately publish nothing — an arm with no
≥98% checkpoint exits `done` with nothing measured — so *no file* and *an empty measurement* are the same
absence over the git bus. `b40` looked like four empty HOF passes; it was in fact one held ≥98%/500 checkpoint
at 98.2%, sitting unpublished on the box. **So before concluding a HOF pass found nothing, check the branch
against the box:**

```bash
git fetch origin results
git ls-tree --name-only origin/results:results | grep <policy>     # what was published
ssh the-claw-den 'ls ~/Snek/snek2/runs/<policy>*'                  # what exists
```

**Recovery is a direct copy** — the git bus is for job control, not for large files, and rsync does not care
about DNS on the box:

```bash
rsync -a "the-claw-den:Snek/snek2/runs/<policy>_checkpoint_evals*.json" snek2/runs/
```

Note the quoted **relative** path: `~/…` inside the quotes is not expanded by either shell and rsync fails with
`change_dir … failed`.

**The proper fix is in the daemon and has not been made** — either retry `publish_results` on the next poll, or
reconcile at idle by pushing whenever the local `results` worktree is ahead of `origin/results`. Either one is a
code change. Until then, treat a `done` HOF or close-out whose file never arrived as **unpublished, not empty**.

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

**Delete the staging copy once it is in `runs/`.** `git checkout` also *stages* what it
writes, so an untracked-and-staged `results/` sitting in a checkout is what aborts the
ff-merge in [the deploy section](#deploying-a-code-change-to-the-desktop) below. It is
redundant the moment the `cp` lands — byte-identical to `snek2/runs/`, and still on the
`results` branch either way. `/results/` is gitignored at the repo root since 2026-08-13,
so it no longer shows up in `git status`; that hides the clutter, it does not reclaim the
disk (~2.5 MB per arm), so still `rm -rf results/` when done. Unstage first if you have
not committed: `git restore --staged results/`.

**That ignore rule cannot reach this box's bus worktrees — and the obvious alternative
can.** The daemon publishes with `git add -A results/<job-id>` and then tests `git status
--porcelain`; both skip ignored files, so a rule that *did* reach `RESULTS_WORKTREE` would
make `_commit_and_push` find nothing staged and return early — **results would stop being
published with no error anywhere**, while the ledger kept saying `done`. Two things keep
that from happening: gitignore lookup never walks above a working tree's own root, and the
`results` branch is orphan-style (just `results/`, no `.gitignore` of its own). Verified by
running the daemon's exact sequence inside a real worktree of `origin/results`.

**So the rule belongs in the tracked `.gitignore`, never in `.git/info/exclude`.** That one
lives in the *common* git dir and is shared by every linked worktree, so the same pattern
there does reach `RESULTS_WORKTREE` and does break publishing — measured, not assumed.

## Staging a laptop-trained policy for a desktop eval

**The git bus carries job specs and results, never checkpoints** — weights are far too
large for it. So evaluating a policy that trained on the *laptop* needs the files copied
over first, and that copy is the one routine task that requires the LAN.

```bash
# checkpoints -- exclude the replay buffer: an eval never reads it, and it changes live
rsync -a --exclude='replay_buffer' \
  snek2/savedPolicies/<policy> the-claw-den:/home/claw/Snek/snek2/savedPolicies/

# the graph JSON the top20 selector reads
rsync -a snek2/runs/<policy>_evals.json the-claw-den:/home/claw/Snek/snek2/runs/
```

Then push an `eval` spec (`type: eval`, `eval_args: ["top20"]`, `priority: 10`) to
`queue/pending/` on `ops` as usual.

Three things to know, each of which has cost a failed job:

- **`arch.json` must ride along**, or the restore hard-fails with `ArchMismatch`
  (`policy_arch.py`). `rsync -a <policy>` carries it because it lives in the policy dir —
  so never add it to `--exclude`.
- **Auto-closeout does not fire for these**, and so neither does the auto-HOF that chains off a
  closeout. Both trigger on a desktop job finishing, and a laptop-trained arm never trained here.
  Queue the eval by hand (and, if you want the HOF re-measure, run the `hallOfFame/README.md`
  recipe by hand too).
- **A ~4-arm wave is ~2.3 GB** (~574 MB/arm), a couple of minutes over the LAN.

## Deploying a code change to the desktop

The daemon runs `runner.runner` from the **`master` checkout** at `/home/claw/Snek`. So code
reaches the box the same way results come back — through git, not scp.

```bash
# laptop: push the commit
git push origin master

# desktop: fast-forward the checkout, then confirm HEAD moved
ssh -i ~/.ssh/snek_desktop claw@the-claw-den
cd /home/claw/Snek
git fetch origin master && git merge --ff-only origin/master
git log --oneline -1
```

**Restart only for daemon code.** A change under `runner/` needs
`sudo systemctl restart snek-runner` (passwordless) to take effect. The restart is safe for
running jobs — the unit is `KillMode=process` and jobs are detached and re-adopted from the
ledger, so no trainer or eval dies. A change to trainer/eval code (`snek2.py`,
`under_the_hood.py`, a `SNEK_*` default) needs **no restart**: each job is a fresh process that
reads the on-disk code at launch. **A job already running keeps the old code** until it stops;
the next job gets the change.

### The fast-forward often aborts — two collisions, both lossless to fix

The desktop writes `runs/<policy>*` artifacts, and the laptop later commits the same files to
`master`. So the incoming commit and the desktop working tree hold the same paths, and the
merge stops. **Remove or stash a file only after you show it is byte-identical to the incoming
version. Back up any that differ — that is real run data.**

- **"untracked working tree files would be overwritten"** — the desktop copy is *untracked*;
  the commit adds it as *tracked*. Per path, confirm identical, then remove:
  ```bash
  git show origin/master:"$f" | cmp -s - "$f" && rm "$f"   # silent = identical
  ```
- **"local changes would be overwritten by merge"** — the path is *tracked* and the desktop
  wrote a fuller version than its HEAD held (e.g. an eval file grown to the 3M-step cap). If the
  working copy already matches the incoming bytes, stash it (recoverable) and merge:
  ```bash
  git show origin/master:"$f" | cmp -s - "$f"              # confirm identical first
  git stash push -m backup -- "$f" && git merge --ff-only origin/master
  ```

**`| tail` hides a failed merge.** A pipe drops the command's exit code, so a scripted deploy
sails past an abort and restarts the daemon on the *old* code. Read `git log --oneline -1` after
the merge — do not trust piped output.

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
  "label": "b40: free space + chase-safe shaping, gate=75, c=0.10",  // short, human; for at-a-glance
  "notes": "..."
}
```

`smoke`/`benchmark` force `SNEK_MIN_CHECKPOINT_SCORE=0` and default the policy to
`smoke` / `bench-<id>` so they stay throwaway.

**Always give a training batch a `label`.** It is a short, human one-liner naming the batch and its
key knobs (e.g. `"b40: free space + chase-safe shaping, gate=75, c=0.10"`), and it is what the
`at_a_glance` block at the top of `status.json` shows for that batch — so the box reads at a glance
without parsing the ledger. All arms of a batch share one label; the auto-spawned closeout and HOF
evals inherit it by batch. It is optional and defaults to `""`, but a batch queued without one shows
only its id in the summary.

### Reading `status.json` at a glance

`status.json` opens with an `at_a_glance` block — `{"running": [...], "queued": [...]}`, one line per
batch-phase, each running line carrying the mean percent done across that batch's arms
(`"b41 -- b29 re-run (same seeds), gate=75 c=0.10 -- training 50% (2 arms)"`). Below it, the `ledger`
is ordered **newest/active first**: the pending queue on top (in launch order, so the next job to run
is highest), then the running jobs, then the finished history most-recent-first. This is the reverse of
the on-disk `ledger.json`, whose insertion order is oldest-first — only the published view is reordered,
so the authoritative ledger and every restart path are untouched.

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
