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

A finished training auto-queues **one** eval job for its whole batch, and the HOF re-measure is a
second *stage inside that job* rather than a job of its own:

| link | trigger | what runs |
|---|---|---|
| **closeout** | a training finishes OK (`auto_closeout`) | one `<batch>-closeout` job, priority 10, carrying every arm |
| **HOF re-measure** | the closeout's stage A finishes | stage B of the *same* process (`--chain`) |

**The engine defaults to the vectorised one** (2026-08-24): `vectorized/vec_wave.py`, which measures
~40x faster than the TF path and was validated against it at four levels, ending in a 24-checkpoint ×
500-episode head-to-head that agreed to −0.058 pp (z = −0.28). Set `eval_engine: "scalar"` in
`runtime.json`, or `SNEK_EVAL_ENGINE` in a single job spec's `env`, to force `eval_wave.py` — the only
way to reproduce a pre-switch measurement, and the answer to a regression here that does not need a
deploy. `runtime.json` validates it as an enum and rejects the whole file on a typo, so the daemon keeps
its last-known-good config rather than failing every eval dispatch one job at a time. **A c51 batch
needs no opt-out either** — `vec_wave` measures categorical arms itself since 2026-08-24, validated on
six `b38a` checkpoints at 200 episodes per engine (−0.17 pp, z = −0.10). The split that used to hand them
to `eval_wave.py` is gone.

**The daemon carries no eval-protocol numbers.** It used to pin the closeout gate, the HOF gate, 500
episodes, the flat-screen flag and the `_hof500` suffix, plus its own copy of the
`closeout gate < HOF gate` assertion. All of that is `eval_plan.py`'s now, and the daemon *removes* the
protocol keys from the env it inherits so the tool's own defaults decide (`EVAL_PROTOCOL_KEYS`). It
cannot simply import the originals: it runs on base miniconda python so it can start before the `snek`
env exists. For a vec wave `EVAL_WORKERS`/`EVAL_LANES` are not set at all — they size TF worker
processes and that engine has none; `VEC_WAVE_PROCS` is its analogue, and unset means cores minus two.

**A reboot mid-eval resumes rather than restarts.** When `_reattach` marks a running closeout or
HOF `interrupted` (boot-id mismatch) and the job is re-synthesized on the next dispatch, its env
carries **`EVAL_RESUME=1`** — so the relaunch keeps the full-length rows already on disk and
re-measures only the checkpoint that was in flight when the power was lost, instead of redoing the
whole pass. A first-time eval (no prior ledger record) never gets the flag, so it always measures
from scratch. The flag is only set on the `interrupted` path, so a `done`/`failed` eval is never
resumed from a stale file.

The HOF re-measure reconfirms the checkpoints the closeout already found excellent. `above:98` reads
each arm's own `runs/<policy>_checkpoint_evals.json` and takes every checkpoint measured there at
**≥98%**, then runs those at 500 episodes, flat, into **`_hof500`** — a separate file, so it never
clobbers the closeout's result and the closeout's 100-episode rows are never mistaken for finished HOF
work. The whole recipe is `eval_plan.hof_settings`, derived from stage A's settings rather than restated,
which is what retired the copies that used to live here and in the laptop's chain script. An arm whose
stage A did not come out `complete` is skipped rather than selected from a truncated file.

**‡ A batch closes out as *one* wave, and what defines "one" is the measurement-relevant env**
(2026-08-21). `_auto_closeout_jobs` groups the pending markers by `(batch, closeout_group_env(env))`,
where `closeout_group_env` keeps only `EVAL_RELEVANT_ENV` — the shaping and reward knobs an eval can
actually see. It used to key on the *whole* inherited training env, so `b45`, whose four arms differ
only in `SNEK_SEED`, split into three waves: `b45-closeout` `{a,c}`, `b45-closeout-w2` `{b}`,
`b45-closeout-w3` `{d}`. Two costs, and the second is the real one:

- The chart window showed 2 panels, then 1, then 1 — a finished arm's chart was gone before anyone
  came back to read it (see `eval_batch_pngs`; the sweep that removed those charts is itself gone
  since 2026-08-24).
- **Three sequential waves of 2/1/1 arms measure a batch at a quarter of the intended 4 lanes.** For a
  continuation batch, whose close-out is already priced in hours, that is the difference between one
  pass and three.

The wave runs under `agreed_env(...)` — every key the group's arms agree on — so a looser group key
never attributes one arm's seed or learning rate to its siblings. `runner.EVAL_RELEVANT_ENV` is a
**copy** of `eval_wave.EVAL_RELEVANT_ENV` (the runner cannot import that module: TensorFlow), and
`tests/test_runner.py::test_eval_relevant_env_matches_eval_wave` parses the real tuple out of
`eval_wave.py` and fails if the two drift.

Three properties that matter:

- **The HOF cannot race its own closeout, structurally.** Stage B is the next statement in the same
  process, so there is nothing to schedule and no `hof: pending` marker any more — that marker, and the
  window in which a second job could read a half-written result file, both went away with the chained
  stage.
- **No HOF-of-a-HOF, no HOF off a hand-queued eval.** `--chain` is set by the daemon only on the
  `<batch>-closeout` job it mints itself; a hand-queued eval spec has to ask for it.
- **Most arms produce an empty HOF.** If no closeout checkpoint reached 98%, `above:98` selects nothing
  and the stage reports that and exits 0 — the common case, and not a failure.

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

**‡ Over ssh, that `pkill` kills the shell running it, and the kill still works** (2026-08-25, abandoning
`b46`'s first wave). `ssh the-claw-den 'kill -9 …; pkill -9 -f "snek2.py b46"; ps …'` returned **exit 255**
with no output, because the remote shell's own command line contains `snek2.py b46` — so `pkill` matched it
and SIGKILLed the session before the verification could run. **Read this as "cannot confirm", never as
"failed"**: a follow-up `ps` showed 0 b46 processes and `free -m` back to 11 GB. Same family as the `pgrep`
self-match above, but worse in one way — `pkill` *acts* on the match instead of merely counting it. Put the
verification in a **separate** ssh invocation, or bracket the pattern (`sne[k]2.py b46`).

**‡ A killed job cannot be re-queued under its own id, and this is the thing to know before abandoning a
batch you intend to relaunch.** `_launch` writes `failed` for a non-zero exit, `failed` is in `TERMINAL`, and
`_scan_pending` drops any spec whose id is already terminal — so pushing the same spec again is silently a
no-op. It never dispatches, nothing reports an error, and the arm simply does not appear. Three ways out,
in order of preference:

1. **Give the relaunched arms new ids.** No daemon surgery, and the ledger keeps an honest record of both
   attempts. Costs you the naming continuity, which matters if the ids encode the experiment (`…seed2`).
2. **Delete the ledger entries.** `systemctl stop snek-runner` **first** — the daemon holds the ledger in
   memory and `_save_ledger()` will overwrite an edit made underneath it — then remove the keys from
   `~/.snek-runner/ledger.json`, keep a `.bak`, and start the service again. This is what `b46`'s restart
   did, and it is right when the ids are part of the experiment's design.
3. Not `interrupted`. It is non-terminal and does get relaunched, but a training *resumes from its
   checkpoint*, so it is the wrong tool unless you are also deleting the checkpoints.

**Deleting a discarded batch's data is four paths**, and none of it reaches `results` if the arms never
finished: `savedPolicies/<policy>/`, `runs/<policy>_evals.json` + `.md` + `.png`, `evals/<policy>*.png`, and
`desktop/logs/train-<id>.log`.

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

## ‡ A failed close-out is never retried, and `b46` wave 1 lost its measurement to that

**The bug, for the record.** `vectorized/vec_wave.py` lives in `snek2/vectorized/`, so Python seeds
`sys.path[0]` with *that* directory, not the `snek2/` above it where `chart_viewer`, `eval_plan` and
`eval_wave` live. Every documented invocation passes `PYTHONPATH=.` from `snek2/` — which is exactly
what hid it, because the laptop always worked. **The runner passes no `PYTHONPATH`, and never had to:**
`eval_wave.py` and `eval_checkpoints.py` sit *in* `snek2/`, so their own script directory is already
right. The moment `vec_wave.py` became the desktop's default eval (2026-08-24) the next close-out died
2 seconds in on `ModuleNotFoundError: No module named 'chart_viewer'`. `b46`'s wave 1 was the first
close-out queued after that switch, so it was the first to hit it. Fixed by a `sys.path` bootstrap
inside **both** `vec_wave.py` and `vec_eval.py` — the shards need it too, since the parent spawns them
as `[sys.executable, '-u', 'vectorized/vec_eval.py', ...]`.

**The expensive half is what happened next, and it is a design decision working as intended.**
`_measured_policies` counts a **`failed`** wave as *measured*, documented as "a wave that failed is not
retried automatically, because the reason is usually not transient". So:

| what you see | what it means |
|---|---|
| `<batch>-closeout` = `failed` in the ledger | the wave ran and died |
| every arm still `closeout: pending` | the marker is never cleared; it means "was trained", not "needs measuring" |
| **no `-w2` in `queued`** | `_measured_policies` covers those policies, so `_auto_closeout_jobs` skips the group entirely |

**So a batch can train for 21 hours, fail its close-out in 2 seconds, and go on to the next wave with
nothing measured and nothing queued** — `status.json` shows a healthy box the whole time. The `failed`
state is the only tell, and it is in the ledger rather than in `at_a_glance`.

**Check for it after any eval-path deploy**, and any time a batch's `closeout eval` line disappears
from `at_a_glance` without an eval having run:

```
ssh the-claw-den 'python3 -c "
import json
d = json.load(open(\"/home/claw/.snek-runner/ledger.json\"))
print([k for k, v in d.items() if k.endswith(\"closeout\") and v.get(\"state\") == \"failed\"])"'
ssh the-claw-den 'tail -5 ~/.snek-runner/logs/eval-<batch>-closeout.log'
```

**Recovering one is a manual spec, not a nudge.** Nothing will re-derive it, so queue an eval job with
an id that ends in `-closeout` (`_CLOSEOUT_ID_RE` is `-closeout(-w\d+)?$`) so that once it runs
`_measured_policies` covers its policies and nothing double-measures them. Key it on the *wave's* arm
prefix rather than the batch — `b46a-closeout`, not `b46-closeout-w2` — so it cannot collide with the
series `_closeout_id` hands to later waves. Give it **priority 9**, one ahead of
`AUTO_CLOSEOUT_PRIORITY`, so the wave barrier runs it before any further training. Carry the arms'
inherited env minus whatever they disagree on, and **do not drop
`SNEK_FOOD_DISTANCE_REWARD`** — its default is `0.001`, so omitting it silently changes `avg_reward`.
`snek2/desktop/queue/pending/b46a-closeout.json` on `ops` is the worked example.

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
  "policies": ["b20a", "b20b"], // eval only: the whole wave in one job
  "eval_args": ["top50"],       // eval only: the selector, and anything before the policies
  "chain": true,                // eval only: run the HOF re-measure as stage B of the same process
  "eval_workers": 10,           // eval only: vec shards, or scalar EVAL_WORKERS, per engine
  "priority": 10,               // lower runs first; default 100
  "label": "b40: free space + chase-safe shaping, gate=75, c=0.10",  // short, human; for at-a-glance
  "notes": "..."
}
```

`smoke`/`benchmark` force `SNEK_MIN_CHECKPOINT_SCORE=0` and default the policy to
`smoke` / `bench-<id>` so they stay throwaway.

`"env": {"SNEK_EVAL_ENGINE": "scalar"}` forces one eval job onto `eval_wave.py` without touching
`runtime.json`, which is one setting for the whole box. It works in the other direction too, so a box
configured for the scalar path can still be handed a vec wave.

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
