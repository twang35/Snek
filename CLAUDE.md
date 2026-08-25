# Snek

## Work as a collaborator

**Ask questions when a request is ambiguous** in a way that changes the work — which arms to stop,
what "best" means, how long to let something run. Don't ask what the code or docs already answer.

**Volunteer better approaches**, before starting and as they occur mid-task. Give a recommendation
rather than a menu, then proceed with it unless redirected. This is explicitly wanted: the largest
result in this project came from an unrequested suggestion to diff `snek2` against the old
`theSchlong` implementation instead of continuing to sweep hyperparameters.

**Say so when evidence undercuts the plan**, including a conclusion written in these docs. This
investigation has overturned several of its own findings and each retraction was worth more than
the section it replaced.

**Be succinct in the docs.** Detail belongs in the summary at the end of a reply; the docs should
be scannable. Lead a section covering a *set* of things with a compact table, then explain
underneath.

## Environment

Dedicated conda env named `snek` (Python 3.10). Activate before running anything:
`conda activate snek`, or `conda run -n snek <command>` non-interactively.

**`conda run` buffers stdout**, even with `python -u`. A backgrounded run's log can stay empty for
90+ seconds while the process is fine, and `kill -9` then discards the buffer permanently. For any
background run where you need live output, call the env's python directly:

```
/opt/miniconda3/envs/snek/bin/python -u snek2.py smoke > log 2>&1 &
```

Reserve `conda run` for short one-shot commands where only the final output matters.

**`snek2/` is the only directory to edit.** `theSchlong/`, `theSchlongCardinalDirs/`,
`humanPlayer/` etc. are kept as-is for posterity.

## Git workflow

**Code: leave finished work uncommitted.** Do not `git commit`, `git push` or even `git add` a
**code** change until the user explicitly approves *that* change — "push" or "commit this" is the
go-ahead and applies only to the change in front of you. The reason is review: the user reads diffs
in their editor, and committing or staging moves the change out of the working tree where it is
visible. So the loop for code is: make the edit, describe what changed, stop and wait.

**Documentation and instructions: commit and push without waiting** (standing authorization,
2026-08-14). Any change that is *only* Markdown — `CLAUDE.md`, the tuning docs (`runs.md`,
`charts.md`, `completedRuns.md`, `findings.md`, …) and their chart PNGs, the `README.md`s and
`SETUP.md` — is committed and pushed as soon as it is done, no confirmation needed. This includes
progress-update doc edits and instruction changes like this one. **The exception is docs-only:** the
moment a commit also touches code, config, or anything that changes program behaviour, the whole
change reverts to the code rule above and waits for approval. Chart images that accompany a
`charts.md` edit ride along with the docs commit.

**Diagnostics and tests: commit and push without waiting** (standing authorization, 2026-08-14, for
the stated reason that the user does not read these diffs). Two directories plus one file pattern:

| path | why it is safe to push unreviewed |
|---|---|
| `snek2/hyperparamTuning/perDiagnostics/*.py` | measurement only; nothing in the training, eval or watch path imports them |
| `snek2/hyperparamTuning/diagnostics/*.py` | same, and frozen besides |
| `snek2/tests/*.py` | assertions about behaviour, never behaviour |

**The same only-if rule applies, and it is the important half.** A commit qualifies only when it
touches *nothing else*. A test that arrives alongside the change it pins — which is the normal case,
since CLAUDE.md asks for a test in the same pass as the logic — is part of a **code** change and
waits with it. So the split is: a diagnostic or a test written to *measure or pin existing*
behaviour goes straight up; a test written for behaviour being changed in the same working tree does
not.

Two things this does not license. A diagnostic still may not write into anything in the
never-delete table below, and a **new** script gets its row in
[`perDiagnostics/README.md`](snek2/hyperparamTuning/perDiagnostics/README.md) in the same push —
an undocumented tool is how `refresh_charts.sh` drifted to 12 undocumented arms.

Read-only git commands are fine at any time.

## Training runs

**`SNEK_MAX_STEPS` (default 10,000,000) is an absolute cap**, so a wave self-terminates and frees
its slots instead of needing someone to notice. Absolute means "stop when `global_step` reaches
this", not "run this many more" — `global_step` is restored on resume, so a relative count would
let an arm resumed at 4M run to 9M. An arm already at its cap prints `already at or past the
N-step cap` and exits after its opening eval; raise the knob to continue it.

The default is generous on purpose — a backstop against an unattended arm, not a planned horizon.
`b9b-disc9975b` ran **10.1M steps past its peak** overnight before this existed. Raised 5M → 10M on
2026-08-06: in batch 14 two of four arms peaked past 3.5M and one was still gaining at 4.0-4.5M, so
the old "arms peak between ~1M and ~3.4M" rule partly described where humans stopped them.

**An arm that never clears `SNEK_MIN_CHECKPOINT_SCORE` (default 40) never writes a checkpoint**, so
it cannot resume and its cap counts from 0 on each launch. That matters for short smoke runs, which
score ~0 — set `SNEK_MIN_CHECKPOINT_SCORE=0` to test resume behaviour.

## Smoke tests

`policy_name` (the arg to `snek2.py`) doubles as the checkpoint directory under
`snek2/savedPolicies/`. **Always pass `smoke`** for verification runs, so output is isolated in
`savedPolicies/smoke/`, which is safe to delete afterward. The user's real run is `train` — never
delete or overwrite `savedPolicies/train/`. **Never `rm -rf snek2/savedPolicies/` wholesale.**

## What must never be deleted

| path | rule |
|---|---|
| `snek2/savedPolicies/train/` | the user's own run |
| `snek2/hallOfFame/` | record checkpoints, outside the `max_to_keep` rotation |
| `snek2/runs/` | every real arm's graph, report and measurements |
| `snek2/evals/archive/` | every chart the retired eval-chart sweep took; nothing writes here since 2026-08-24 |

A wrongly kept file costs a few KB; a wrongly deleted one costs a training run. When in doubt,
keep it, and `grep -rn <name> --include='*.md' .` first — the tuning docs link files by name.

**Throwaway output is fine to delete**: smoke tests, speed benchmarks, `champion_*` and
`bench-*` verification evals. Judge by what produced it, not where it lives.

**‡ `snek2/evals/` accumulates now, and nothing moves a chart out of it** (2026-08-24). Each eval
writes `evals/<policy>_eval_progress.png` and overwrites its own file by name. That is the whole
mechanism — there is no reset, no archive step, and **no way for one eval to affect another's chart.**

**The removed thing was `archive_existing_eval_pngs`, and it is worth knowing why**, because eight
months of this file warned about it and the warnings were the tell. Every eval entry point used to
sweep every PNG at the top of `evals/` into `evals/archive/<timestamp>/` before doing anything else,
so the folder would show "only the most recently completed work". The costs:

- **A one-checkpoint verification eval displaced a whole batch's finished panels.** Batch 11's four
  charts, then batch 13's, then `b43`'s and `k1000`'s twelve — the last of those *while the removal
  was being written*, which is a fair summary of the problem.
- **`EVAL_OUT_SUFFIX` protected the results file and not the charts**, because the chart path has no
  suffix in it and the sweep ran before any setup. There was no flag that made an eval harmless.
- **A batch measured as several waves erased its own earlier waves.** `keep_batches` was added to
  patch that, which made the rule *"an eval archives every chart except those of batches this
  particular process happens to be measuring"* — unholdable, and it still displaced other batches.
- **Restoring was lossless but manual**, and had to be remembered.

**What it was protecting is free without it.** An arm rewrites its own chart by name, so `evals/`
is self-correcting; it accumulates instead of resetting, and a stale entry is a stale chart of a real
arm rather than a missing chart of a current one. That is the strictly better failure.

**`evals/archive/` stays** — it holds every chart the sweep took, it is in the never-delete table
above, and nothing writes there any more. Restoring an old chart from it is still just a `cp`.

**Two consequences to carry.** `tests/test_evals_dir_is_never_swept.py` is a tripwire that fails if
any eval entry point regains a `shutil.move`/`rmtree` — the absence is the invariant now. And **the
viewer's panel cap became load-bearing**: the sweep was what kept `--glob evals/b20*` down to a
wave's four files, so `chart_viewer.newest_glob_files` now caps at `MAX_WAVE_PANELS` (8) and selects
by **mtime**, since a running arm rewrites its chart every round. Alphabetical would have shown
`b20a`-`b20h` regardless of what was live.

**`<policy>_checkpoint_evals<suffix>.json.previous` is a safety net.** Written before overwriting
an existing *complete* result, because the first write of a new run destroys whatever was at that
path. Check it isn't the only copy of something real before removing it.

Deleting `<policy>_history.json` also throws away the graph's history, so the next run restarts
the curve instead of continuing it.

## `snek2/hallOfFame/` — record checkpoints

Copies of the best policies, outside `savedPolicies/` so the `max_to_keep` rotation cannot evict
them. That has already destroyed evidence once (`b5c-schlongIS`'s 17.0% peak). **Never delete
anything here**, and add an entry whenever a run produces a checkpoint worth keeping — two files,
~190 KB. The README has the copy-in-and-evaluate procedure; verify the *copy* loads and plays, not
just the original.

**Matching widths do not mean a working checkpoint, and this has bitten.** A checkpoint restores
whenever the vector is the same *length*; nothing checks the values still mean what they meant. On
2026-08-02 the vector was briefly back at its original 20 values with two indices repurposed, and
every hall-of-fame checkpoint restored with no warning and played like a beginner — the champion
went from **90.3%** to scoring **0, 0, 1**. An input that was previously constant is the specific
trap: `game_over` sat at 0 in every state a policy acts in, so its weights were unconstrained, and
the index it occupied now carried board-fill.

So **when the observation changes, record which indices changed meaning in the hall-of-fame README
and name the last commit whose observation matches those checkpoints.** Era markers: `e4514a8` =
20 values, `450e66e` = 26, `b09c616` = the current 30.

**`arch.json` now makes both traps loud (`policy_arch.py`, 2026-08-11).** Every policy dir carries a
one-file sidecar — `fc_layer_params`, `num_actions`, `obs_len`, `obs_era` — written once at training
start (not per checkpoint). Training, `eval_checkpoints`, `eval_workers` and `watch.py` all rebuild
the *recorded* network from it and **hard-fail** (`ArchMismatch`) if it is missing or if the live
env's observation length or era disagrees; a resume also fails if `SNEK_FC_LAYERS` disagrees with the
sidecar. So `SNEK_FC_LAYERS` is no longer read at eval/watch time, and the era must be bumped in one
place — `snake_environment.OBS_ERA` — whenever the observation's *meaning* changes at constant length
(the `game_over` trap). Existing dirs were backfilled from their own checkpoint tensor shapes by
`backfill_arch.py`, which read the true per-era lengths (b2-b9 = 20, b10 = 26, b11+ = 30). **Copy
`arch.json` with any checkpoint** — into `hallOfFame/`, or rsynced to the desktop — or it will not
load.

**Append new per-action blocks after the existing ones**, never interleaved. The frozen diagnostics
in `hyperparamTuning/diagnostics/` read `head_with_tail` at `obs[9 + 2 * i]`. That is why the
vector's order is chronological rather than logical. **Not every block is a per-action triple** —
food-space, starve and board-fill are single values, so don't assume index arithmetic in threes.

**1 means good or safe throughout the vector.** Two caveats on the newest blocks: a *fatal* move
reads 1 at indices 26-28 (the flag only asks "is this the tail's cell" — combine with 6-8), and
index 29 sits at 1 in **99.95%** of states, so it is nearly a constant and **its weights are not
meaningfully trained** — same shape as the `game_over` trap above.

**Indices 18-20 (`perfect_game_move`) are worse: nonzero in 0.000-0.025% of states** (measured
2026-08-16, 12,000 greedy states on two arms). `perfect_game_obs` returns `[0, 0, 0]` unless
`snake_len == PERFECT_SCORE - 1`, so the "this move wins" flag only exists on the single step before a
win. Forcing it to 1 moves `Q` for that action by **+0.53** on one arm and **−0.94** on another — the
wrong sign on the arm that wins 92% of its games. **No policy in this project has ever learned to win
from that input**; the ones that win do it through board-fill (index 22), which is **rank 1 of 30** by
saliency in every arm measured. Two consequences: don't credit an endgame result to indices 18-20, and
**don't try to fix an endgame by adding an input that only fires in the endgame's last step** — an input
that is on in 0.01% of states cannot be trained however informative it looks. Measure occupancy with
[`perDiagnostics/endgame_gradient.py`](snek2/hyperparamTuning/perDiagnostics/endgame_gradient.py) before
adding a block.

**A terminal reward is a potential, not a prize, and it has a threshold.** With `k` steps per meal and
`f = γ^k`, one meal of progress is worth `f^(m-1)·[W(1-f) - 1]`, so progress only raises value when
**`W > 1/(1 - γ^k)`** — at `DISCOUNT=0.9975` and this project's 7-12 steps per meal, **34-58**.
`PERFECT_GAME_REWARD=100` clears that by 2-3×; batch 33 cut it to 10, missed by 3-6×, and the agents
correctly learned to avoid finishing (see
[`findings.md`](snek2/hyperparamTuning/findings.md#-falsified-2026-08-16-shrinking-the-win-reward-100--10-does-not-buy-c51-stability--it-teaches-the-agent-that-winning-is-a-mistake)).
**So `PERFECT_GAME_REWARD` and `DISCOUNT` cannot be tuned independently** — lowering the win requires
lowering γ to match.

**`tests/test_observation_spec.py` is the guard.** It asserts `observation_spec()`'s length equals
what `get_observations` builds across three hand-built boards, and pins each recent block to its
index range by comparing against the producing function rather than a literal — an ordering bug
passed a literal-based version because two blocks coincidentally held the same values.
`test_state_helpers.py` hardcodes the count as a tripwire, so adding a block fails until the count,
the layout docstring and the era markers are all updated together.

## A perfect game is identified by its **score**, never by its reward

`state_helpers.is_perfect_score(score)` is the single definition, and `Snake.check_perfect_game` plus
all three counters — `under_the_hood.compute_avg_return`, `eval_workers`, `eval_checkpoints` — go
through it. **Never compare a final reward with `PERFECT_GAME_REWARD`**; an `ast` tripwire in
`tests/test_perfect_game_counting.py` fails if that comparison reappears.

The rule is written in the cost of learning it. All three counters used to test
`final_reward == PERFECT_GAME_REWARD`, so the moment `CHASE_SAFE_SHAPING` shipped — it pays `−c·Φ(s)`
at the winning step, as potential-based shaping must — a perfect game paid **99.9 instead of 100** and
every counter read **0%**. Eight arms across two hosts trained blind for 300k+ steps while filling
boards from step 9k, and because `training.epsilon_for`'s refinement phase is driven by the trailing
perfect rate, **epsilon stayed pinned at its 0.0125 ceiling** instead of annealing — so the runs were
handicapped, not just mismeasured. The tell was in their own reports: `max_score` read `95/95`, and 95
*is* a filled board.

Two general lessons worth carrying past this bug. **A reward is a sum of terms**, so anything derived
from it breaks silently when a term is added — identify outcomes from state, not from reward. And
**`perfect_percent` is not only a report**: it feeds the exploration schedule, so breaking the
measurement changes the training. Full account:
[`findings.md`](snek2/hyperparamTuning/findings.md#-a-perfect-game-was-identified-by-its-final-reward-and-the-shaping-term-silenced-every-counter).

## Hyperparameter tuning

An ongoing, resumable investigation into configs that reach the highest perfect-game percentage
while learning consistently. Reducing catastrophic forgetting is a means to that, not the goal.
Everything is in **`snek2/hyperparamTuning/`**:

| file | contents |
|---|---|
| `runs.md` | what is running, what to run next. **Start here** |
| `hyperparamTuning.md` | the protocol: metrics, stop criteria, how to judge, how to launch, knobs |
| `findings.md` | what is established, what is falsified. Read before proposing an experiment |
| `completedRuns.md` | every arm: config, final numbers, verdict. The canonical arm table |
| `failureModes.md` | the four ways a policy degrades and how to tell them apart |
| `charts.md` + `charts/` | progress graph per arm, the six newest batches (older captions in `archive/charts-archive.md`) |
| `archive/` | retired batch narratives (1-11, 12-15), retired chart sections, superseded findings. **History only — do not read into context** |

Keep the split clean: `runs.md` is current state and forward plan only, results go to
`completedRuns.md`, conclusions to `findings.md`, anything about *how to measure or judge* to
`hyperparamTuning.md`. `runs.md` once grew to 950 lines of interleaved status and stopped being
usable.

**Every running batch keeps its description in `runs.md`** — why it is shaped that way, what each
arm isolates, what each outcome would mean — while *any* arm is still running, then moves to
`completedRuns.md` when the last one stops. Without the rationale a future session cannot tell a
surprising result from an arm that was never going to answer anything.

### "Progress update" means look, don't touch

A progress update is **read-only with respect to running processes**: analyse, refresh charts,
update docs, report. **Do not kill, stop or restart any arm** — not even one that looks finished,
is past its cap, or is clearly failing. Deciding a run is done is the user's call. If a run looks
finished and no slot is needed, say so as a recommendation.

**Refresh the charts first — before the analysis, not after it** (standing instruction, 2026-08-19).
The order is: pull any finished desktop results into `snek2/runs/`, run `refresh_charts.sh`, run the
completeness check, *then* analyse and write. Two reasons it has to be first. The user reads the chart
window and `charts.md` while the reply is still being written, so stale images are the one artifact that
is visibly wrong for the whole update; and the refresh is what reveals **which** arms have new data —
including arms whose results landed on the `results` branch, or never landed and need the rsync recovery.
Doing it last has repeatedly meant discovering a finished batch after the analysis was already written.
`refresh_charts.sh` is **zsh**, not bash (`${0:a:h}`), so `bash scripts/refresh_charts.sh` dies with
`a: unbound variable`.

Stop an arm only when asked, or when the user asks for something that plainly needs a free slot —
and then say which ones are stopping and why.

**Before killing or relaunching any arm, check its wall-clock runtime and step, and never call an
arm "fresh" from a hunch.** Elapsed session time is not real time — an arm that feels seconds old to
a session can have trained for hours. This nearly killed a **3.5-hour, near-record b19 arm** whose
config change would have been reverted for the loss: the fix was assumed clean because the launch
"felt recent". Run `ps -o etime,lstart -p <pid>` and read `summary.step` from `runs/<policy>_evals.json`
*first*, and let the numbers — not the sense of how long ago you launched it — decide whether it is safe.

It also means updating **`charts.md` as well as `runs.md`**, and **this is not deferrable to batch
close.** **Any time you touch the tuning docs or run a progress update, refresh `charts.md` in the
same pass — whether or not the arms have finished.** An in-progress batch gets its section *now*,
carrying whatever readings exist (training self-eval peak/best-30/`sef`), with the close-out pooled
and any HOF-500 marked *running* until they land. The file's whole job is to show every current arm's
graph **in one place**, so a running batch with no chart entry is a bug, not a "wait until it closes"
state — this has been missed repeatedly, on the mistaken theory that charts belong only to batch close.
`refresh_charts.sh` only copies PNGs, so a new arm silently ends up with an image and no entry. Every
arm needs a `### <policy> — <change>` section with a stats line, a short reading, and the image. Older
captions live in `archive/batches1-11.md` and `archive/charts-archive.md`, so check all three:

```
cd snek2/hyperparamTuning
ls charts/*.png | sed 's|.*/||;s|\.png||' | sort > /tmp/have
grep -ho 'charts/[a-zA-Z0-9-]*\.png' charts.md archive/batches1-11.md archive/charts-archive.md \
  | sed 's|.*charts/||;s|\.png||' | sort -u > /tmp/doc
comm -23 /tmp/have /tmp/doc   # anything listed is undocumented
```

**Both details in that snippet are load-bearing.** `archive/charts-archive.md` has to be in the grep or
every retired arm reads as undocumented, and the `sed` must be `s|.*charts/||` — archived captions link
`../charts/x.png`, which the shorter `s|charts/||` leaves as `../x`. **A few PNGs will always be listed** —
six as of 2026-08-17 (`champion-vs-mediocre`, `drawdown-b23b-vs-b18`, `per-b18-vs-b20-priorities`,
`plasticity-metrics`, `best30-drivers`, `gate-behavior-b27-vs-b29`): they are diagnostic figures referenced
from `findings.md`, not arm charts. **The authoritative list is the one in `charts.md`'s own completeness note**, which sits next to the
snippet and is updated when a figure is added; this count read "four" for a while because `best30-drivers`
was added there and not here.

This drifted once: batches 5-7 reached 12 undocumented arms because `refresh_charts.sh` succeeding
looked like the charts were handled.

**Retiring a section needs the `../` link repair, and it was skipped every time until 2026-08-12** —
sixteen links inside `archive/charts-archive.md` pointed at `charts/…` and `completedRuns.md` from one
directory too deep, and five links *into* retired sections still pointed at `charts.md`. Both classes
render as normal text or 404 rather than erroring, so nothing surfaces them. After any move, re-resolve
every `](file.md#anchor)` in the tuning docs against the real headings.

### Two rules that are easy to get wrong

- **Never run more than 4 snek trainers at once on this laptop**, counting human-started ones. Check
  with `pgrep -fl "python -u snek2.py"`. **Not** `grep "[s]nek2.py"` — Airbnb git telemetry fires
  `curl` processes whose JSON payload contains `snek2/snek2.py`, which read 6 trainers when 4 were
  running. The same class of trap caught `pgrep -fl watch` matching `watchdogd` and `watchman`.
  **That `pgrep` is laptop-local and cannot see the desktop** — see below.
- **‡ A `pgrep` pattern matches the shell that runs it, and this is the default outcome, not an
  edge case** (2026-08-20). `pgrep -f <pat>` scans full command lines, and the invoking shell's own
  command line contains `<pat>` verbatim — so the check counts itself. It cost two immortal
  processes: wait-loops written as `until [ "$(pgrep -cf 'eval_checkpoints.py k1000')" = "0" ]; do
  sleep 10; done` never saw zero, because the loop's own argv holds that string. They spun on
  `sleep` for six hours until killed by pid.

  Three things follow. **Bracket every pattern in the command, not just one** — a compound check
  that brackets `eval_check[p]oints` and then greps a bare `chart_viewer` on the next line still
  self-matches on the second pattern. **Prefer `ps -Ao pid=,command= | grep <bracketed>`** over
  `pgrep -f`, or hold the pattern in a variable the argv never spells out. And **a self-matching
  count is 1, not 0**, so it reads as "still running" — which fails safe for a liveness check and
  fails *open* for a wait-loop, where it means "never finish". Never write a wait-loop whose
  condition greps for a string its own command line contains.

  Note this is the *inverse* failure from
  [`_matching_commands`](snek2/chart_viewer.py), where a `pgrep` that errored read as "nothing is
  running" and closed a live window. Both directions are real: **a process scan can over-report and
  under-report, so never treat its output as authority without checking the pattern against the
  scanner itself.**
- **This domain is very noisy** — the same config has produced 62.5 and 18.0. Never conclude from a
  single run; repeat promising configs 2-3 times. **n=4 cannot resolve an effect below ~10 pp.**

Hyperparameters come from `SNEK_*` env vars (see `tuned()` in `snek2.py`), so variants run side by
side without editing files.

### There are two compute hosts — say which one you mean

Since 2026-08-08 a dedicated desktop (`the-claw-den`) also runs trainings and evals, driven entirely
by git: you commit a job spec, it runs it, it pushes results back. Full docs in
[`snek2/desktop/README.md`](snek2/desktop/README.md).

| | laptop | desktop `the-claw-den` |
|---|---|---|
| limit | **4 trainers** | `max_trainers` ≤ 4, `max_evals` ≤ 4 |
| check | `pgrep -fl "python -u snek2.py"` | **`git fetch origin ops-status && git show origin/ops-status:status.json`** |
| queue work | launch by hand | commit a JSON spec to `queue/pending/` on the `ops` branch |

**`git fetch` is not optional in that command, and leaving it out is the single most repeated mistake
in this project's history with the desktop.** `git show origin/ops-status:…` reads a **local
remote-tracking ref**, which changes only when you fetch. Without the fetch you are shown an
arbitrarily old snapshot *with no indication that it is old* — and because the payload contains a
timestamp, the natural misreading is "the daemon stopped at 08:33" when the truth is "my ref is from
08:33". This has now produced three false alarms: **2026-08-12** (17 hours stale, four finished evals
reported as still running), and **2026-08-17** twice in one session — a session that reported a
10-hour-dead daemon and a batch that had failed to dispatch, while the box was in fact healthy, had
already finished that batch *and* its close-outs, and had moved on to the next one.

**So a stale-looking `iso` is your own ref until you have fetched and re-read it.** Never report the
desktop as down, drained, stuck or off-LAN from an unfetched read. The ladder, in order:

1. **`git fetch origin ops-status`**, then re-read. This resolves it almost every time.
2. **Try `ssh the-claw-den` — actually run it**, with `-o ConnectTimeout=8 -o BatchMode=yes` so a
   genuine failure returns in seconds instead of hanging. One command settles reachability.
3. Only after 1 and 2 both fail is the box worth calling unreachable.

**Neither check sees the other host.** The `pgrep` rule above is laptop-local, and desktop jobs never
appear in it — so **"N arms running" is meaningless without naming the box**, and a progress report
has to check both. The desktop's `running` and `counts` fields in a **freshly fetched** `status.json`
are the only authority for its side.

**The git bus works from anywhere, and `ssh the-claw-den` is home-LAN only** — Tailscale was removed
on 2026-08-13, so the name resolves by mDNS. Queueing, retuning, `status.json` and `results` are
unaffected; deploying code, `journalctl` and `free -m` need you to be home. **But "probably off-LAN"
is a conclusion, not a starting assumption** — it was wrong on 2026-08-17, when `ssh` answered
immediately and the earlier guidance to infer liveness from the heartbeat instead of just trying the
connection is what produced the wrong call. Alias, no-config fallback and key recovery:
[`snek2/desktop/SETUP.md`](snek2/desktop/SETUP.md#laptop-side-ssh-access-and-how-to-rebuild-it).

**Memory is the desktop's binding constraint, not cores** — **15,030 MB** as `free -m` reports it
(`MemTotal` 15,390,836 kB, ~14.7 GiB). Measured 2026-08-09, and the two worker kinds differ sharply:

- **Standalone eval workers (`eval_checkpoints.py`) are *spawned*** — each loads its own TensorFlow
  arena, **~230 MB/worker**. A single scaling eval hit the OOM-killer at ~52 workers, and **~40 is the
  ceiling, not the operating point**: the `12.8 GB at 4×10` measurement was 40 spawned workers and
  left only **~2.3 GB** of the 15,030 MB free. **Operate at ≤32 total** — `HARD_MAX_EVALS=4` at
  `eval_workers` ≤ ~8 — which is what keeps the ≥3 GB headroom this band is chosen for.
- **Training self-eval workers are *forked*** (Linux COW-shares the parent's TF pages), so they are
  nearly free: **4 trainers × 10 self-eval workers ≈ 4.2 GB total**. The overnight OOM was the
  cv2/XIO chart cascade plus orphan accumulation, *not* steady memory.
- **‡ The desktop's vec-eval optimum is 16 shards with `TF_NUM_INTRAOP_THREADS=1`, and the box is 8
  cores, not 16** (swept 2026-08-24 by
  [`perDiagnostics/vec_wave_sweep.py`](snek2/hyperparamTuning/perDiagnostics/vec_wave_sweep.py); full
  table in [`snek2/vectorized/README.md`](snek2/vectorized/README.md)). It is a Ryzen 7 9700X: **8
  physical cores, 16 SMT threads**, and `os.cpu_count()` reports the threads — so `DEFAULT_PROCS =
  cores − 2` has always meant *threads* − 2 here, and this file's "16-core box" is where the 14 came
  from. Throughput at 240 checkpoints × 100 episodes, noise **under 1%** on this box:

  | procs | intra-op | episodes/s |
  |---:|---:|---:|
  | 8 | default | 326.3 |
  | 12 | default | 344.1 |
  | 14 (the default) | default | 358.5 |
  | 16 | default | 361.2 |
  | 14 | **1** | 371.4 |
  | **16** | **1** | **373.9** |
  | 18 | 1 | 337.5 |

  **Three things to carry.** The plausible inference from the topology — that 14 shards on 8 cores is
  the oversubscribed regime the laptop measured as 20% slow — is **false**: throughput climbs past the
  physical cores to 16, so SMT is worth ~+10% here. **The cliff is at `cpu_count`**: 18 loses 6-10%,
  20 and 24 are 12-13% down, so running it harder than 16 is strictly worse, and the peak already sits
  at **~6% idle** — the last few percent is not slack that can be converted. And **`TF_NUM_INTEROP_THREADS=1`
  is the biggest free win, +3.6% and ~33 MB a shard**: a shard is 95% single-threaded numpy, but TF
  sizes its executor pool to the machine, so 14 shards dispatching across 16-thread pools put ~800
  threads on 16 hardware threads. **Isolated to that one variable** — intra-op alone is +0.3%,
  `OMP_NUM_THREADS` alone is −0.3%, inter-op alone is the whole effect. **Do not check it with
  `tf.config.threading.get_intra_op_parallelism_threads()`** — it reads the `ConfigProto` field and
  returns 0 either way, because TF reads the env var at pool creation; count the process's threads.

  **‡ On the laptop the same change is null, and two traps live in measuring it there.**
  `OMP_NUM_THREADS=1` *undoes* the thread reduction on arm64 (30 threads at the default, 17 with
  inter-op pinned, **30 again with all three pinned**, reproducible — where the desktop goes 50 → 5),
  so a laptop A/B run at "all three" compares 30 threads with 30 and measures nothing. Pinning
  inter-op alone does shrink the pool there (30 → 17), and it **buys CPU rather than speed**:
  throughput is flat (**+0.55%** across four pairs, RSS unchanged) while **CPU idle goes 9.4% → 15.0%**
  — the same work for ~0.8 fewer of the 14 cores, which is the more useful half on the interactive
  machine. The desktop is CPU-bound at 14 shards on 8 cores so the saving becomes throughput; the
  laptop's 12 shards on 14 real cores each own a core already, so it becomes idle. **Idle resolves it
  where throughput cannot** — 15.0%/15.0% against 9.4%, while `episodes/s` scatters ±4%. And **the laptop needs paired configs**: its spread is **±4%** against the desktop's
  **under 1%**, and a laptop sweep in launch order carries a warm-up drift that inverted the ranking
  between two rounds. Alternate A/B/A/B there; the desktop does not need it.

  **The memory arithmetic this section used to carry was pessimistic by 3-4 GB.** It subtracted
  `procs × 690 MB` from `MemAvailable`, which double-counts reclaimable page cache. Measured: a shard
  is **553 MB at `intraop=1` and 585-601 MB at TF's default**, and a 16-shard wave peaks at **8.9 GB
  resident leaving 6.7 GB available** — where the old arithmetic predicted ~1.6 GB. Even 24 shards
  (14.0 GB) leave 3.7 GB. So the "one configuration that can OOM" is less sharp than it read, though
  four trainers plus a 16-shard wave is still 13.1 GB of 15,030 MB; **14 shards at `intraop=1` cost
  0.7% and save 1.1 GB**, which is the trade for a wave overlapping live training. Both RSS figures are
  still short-run (64-101 s), so a multi-hour wave's memory is unmeasured — **watch `free -m` through
  the first long close-out.**

  **`runtime.json`'s `tf_intraop_threads` cannot deliver that win, and reaching for it would be a
  mistake.** `launch.py` applies the threading knobs to **every** job it starts, trainings included, so
  setting it to 1 pins trainers to one intra-op thread as well — untested, and training is the half of
  the box where batched gradient work can actually use a pool. The eval-only place for it is
  `vec_wave.child_env`.

  Unrelated to the count but easy to trip on: the wave's shards are `eval_workers`-insensitive —
  that knob sizes TF worker processes, which this engine has none of — but a job spec that still sets
  it *caps* `VEC_WAVE_PROCS` at its value (`launch.py` reads
  `job.eval_workers or runtime['vec_wave_procs']`), so an old spec carrying `eval_workers: 4` runs a
  4-process wave whatever the box.

Still measure with `free -m` before pushing `max_evals`/`eval_workers` past those bands.

**A finished desktop job is not in `snek2/runs/`** — it arrives on the `results` branch and needs one
copy before any tuning tool can see it. The exact commands are in
[`snek2/desktop/README.md`](snek2/desktop/README.md#getting-a-finished-job-into-the-analysis-workflow).

**The desktop chains two evals off every training: `training → closeout (top50) → HOF re-measure`.**
When a closeout finishes, `auto_hof` (default on) queues a `<policy>-hof` job that re-runs the
closeout's **≥98%** checkpoints at **500 episodes, flat, `EVAL_MIN_ACHIEVABLE=98`**, writing
`_hof500`. It only produces the re-measurement — **promotion into `hallOfFame/` is still the manual,
verified process**. Most arms have no ≥98% checkpoint, so the HOF job exits `done` with nothing
measured; that is normal, not a failure. Turn it off with `auto_hof: false`. Full mechanism:
[`snek2/desktop/README.md`](snek2/desktop/README.md#the-eval-chain-training--closeout--hof-re-measure).

**‡ A batch's close-out is one wave per *measurement-relevant* env, and it used to be one per env**
(fixed 2026-08-21). `_auto_closeout_jobs` keyed its groups on the whole inherited training env, so
`b45` — four arms differing only in `SNEK_SEED` — split into **three** waves, `{a,c}`, `{b}`, `{d}`.
The cost was throughput, not just a short window: three sequential waves of 2/1/1 arms measure a
batch at a quarter of the intended 4 lanes, against the standing "4 processes × ≥4 workers" rule. The
key is now `runner.EVAL_RELEVANT_ENV`, a copy of `eval_wave.EVAL_RELEVANT_ENV` with a test that fails
if the two drift, and the wave runs under `agreed_env` — every setting its arms share, so no arm's
seed is attributed to the others. **A seed, a learning rate, a target-update period cannot reach a
measurement of an already-trained checkpoint**; shaping and reward knobs can, and those still split.

**‡ The desktop viewer's panel set is the batch's charts for an eval wave, the running arms for a
training one** (2026-08-21). Stickiness within a wave (`sticky_wave_pngs`) covers a training wave,
whose arms are dispatched together, and cannot cover a batch measured in several waves — which is how
`b45` showed 2 panels, then 1, then 1. `eval_batch_pngs` widens an eval wave to **every chart in
`evals/` whose batch this wave is measuring**. Membership is *the file existing on disk*, the same
rule as the laptop's `--glob`, and that is load-bearing twice over: `chart_viewer` deliberately has
no per-panel title, so a path for an un-started arm would be an unlabelled empty box; and it bounds
the set without a TTL, so a 36-arm batch like `b20` cannot open a window taller than the screen. A
*training* wave is deliberately **not** widened — it is the one case that could over-report.

**‡ But "nothing measured" and "never published" look identical, and `publish_results` has no retry.** The
box's DNS for `github.com` flaps; a failed push leaves the commit local, the ledger still says `done`, and only
the *next* successful results push carries the backlog. On 2026-08-18 all four of `b40`'s HOF-500 files plus
`b40b`'s whole close-out sat unpublished for hours — one of them a **98.2%/500** checkpoint — while
`status.json` read `done` and the branch showed nothing. **So before concluding a HOF pass was empty, compare
the branch with the box** (`git ls-tree --name-only origin/results:results | grep <policy>` against
`ssh the-claw-den 'ls ~/Snek/snek2/runs/<policy>*'`) and recover with
`rsync -a "the-claw-den:Snek/snek2/runs/<policy>_checkpoint_evals*.json" snek2/runs/`. Full account:
[`snek2/desktop/README.md`](snek2/desktop/README.md#-a-done-in-the-ledger-does-not-mean-the-results-were-published).

**Pushing to `ops` starts real work on another machine**, so it falls under the git rule above: queue a
job only when the user has approved *that* job.

**Every training spec carries a `label`** — a short, human one-liner naming the batch and its key knobs
(e.g. `"b40: free space + chase-safe shaping, gate=75, c=0.10"`). It is what `status.json`'s `at_a_glance`
block shows for the batch, so the box reads at a glance. All arms of a batch share one label and the
auto-spawned closeout/HOF evals inherit it; it defaults to `""`, so a batch queued without one shows only
its id. **Copying an existing spec is how a batch ends up label-less** — the field post-dates most specs on
`ops` — so set it deliberately. Spec format and the `at_a_glance`/ledger layout:
[`snek2/desktop/README.md`](snek2/desktop/README.md#job-spec).

**A reboot mid-job recovers by itself, and `interrupted` is the state that says so.** The daemon
compares each running record's boot id against `/proc/sys/kernel/random/boot_id`: a mismatch means
the machine rebooted, so the job is marked **`interrupted`** — non-terminal, therefore relaunched on
the next dispatch, with a training resuming from its checkpoint. Same boot plus a dead pid still
means `done`, because detached jobs really do outlive a daemon restart. **Read `interrupted` as
"lost wall clock, nothing else"** and check the entry's `restarts` count; before 2026-08-13 the same
situation read `done`, which published truncated arms as finished and silently consumed their
close-outs. Draining first (`"drain": true` in `runtime.json`) avoids the whole path. Full table:
[`snek2/desktop/README.md`](snek2/desktop/README.md#rebooting-the-box-and-what-recovers-by-itself).

### Reading status: the summary block, not the log

Training runs **quiet** — one compact line per 10 evals. `SNEK_DEBUG=1` restores the original
verbose output; use it for debugging, not status. The one log line worth grepping is
`hyperparameter override:` at startup, which confirms an arm got its intended config.

Read status from `runs/<policy>_evals.json`'s precomputed `summary` block: `step, evals,
trailing_now, peak_trailing, best_perfect30, strong_eval_fraction, recent_perfect30,
max_single_eval, dead_since, zero_since, epsilon`.

**`strong_eval_fraction` is the primary metric** (share of the arm's evals at >=80% perfect) — it
has the lowest between-seed variance of the candidates, so it resolves ~40% smaller effects than
`best_perfect30`, which is kept for continuity with batches 1-11. It is a fraction of each arm's
own evals, so **compare only at a common step horizon**.

**Use `zero_since`, not `dead_since`, to ask whether an arm is dead now.** `dead_since` is the
earliest sustained-zero stretch and is history; `zero_since` is the current unbroken stretch and is
`null` if the latest eval is above threshold. `b8d-disc995clip` carried `dead_since=275000` while
going on to a 36% best-30 window.

Neither is a verdict. `b8g-clipseed3` recovered from **1.2M steps** near zero to 63.7 trailing
before collapsing for good. Read `zero_since` against `step` for duration, and only call an arm
dead after hundreds of thousands of steps pinned there *and* no recovery arc in progress.

**Check `zero_since` is actually present before trusting it.** Arms launched before the field
existed overwrite the backfill from their old in-memory `run_report`, so the key is absent — and
`summary.get('zero_since')` returns `None`, which reads identically to "alive". Use
`'zero_since' in summary`, or recompute with `run_report.build_summary(rows['evals'])`.

## Markdown traps

Two that look fine in source and render wrong. The user reads these rendered.

- **`A.`/`a.` are not list markers** — only digits and `-`/`*`. Lettered items collapse into one
  run-together paragraph. Use `#### A. Thing` headings when the letters are cross-referenced.
- **Duplicate numbers in one list renumber silently** — two items both written `6.` render as 6 and
  7, shifting everything after and breaking prose that refers to "item 12". Cross-reference items
  by name, not number.

Worth a `grep -n '^[A-Za-z]\.\s'` and a duplicate-number check after editing any md file here.

## No audio — never call bare `pygame.init()`

`pygame.init()` starts *every* subsystem including `pygame.mixer`, which opens a real CoreAudio
stream **per process**. `SDL_VIDEODRIVER=dummy` does not affect audio, so headless workers were
still holding audio streams: 10 idle env processes drove `coreaudiod` to **15% CPU**, and evals run
several 10-worker processes at once.

Nothing here plays sound. `Snake.Game.__init__` inits only `pygame.display` and `pygame.font`, and
`snek2.py` and `eval_checkpoints.py` both set `SDL_AUDIODRIVER=dummy` before any pygame import. If
a future change needs another subsystem, init that subsystem by name. Verify with
`ps -o %cpu= -p $(pgrep -x coreaudiod)` while workers run; it should read 0.0.

## Tests: `snek2/tests/` — write them, and run them

**When a change has logic worth pinning down, add a test in the same pass.** Worth a fixture
whenever a change involves a conditional whose branches could later collapse, an index or
coordinate convention, a rule someone could "simplify" without seeing why it exists, or an edge
case that took thinking to get right — if it needed reasoning, it needs a fixture, because the
reasoning does not survive in the diff.

**pytest is not installed**, so run them directly:

**Discover the modules with a glob, never a hardcoded list.** The list this snippet used to carry had
drifted to 18 of the 22 modules — it silently skipped `test_eat_and_survive` and the three
`test_plasticity*` files, 51 tests, and reported "515 tests, 0 failed" as if that were the suite:

```
cd snek2
PYTHONPATH=. /opt/miniconda3/envs/snek/bin/python -c "
import sys, glob, os; sys.path.insert(0, 'tests'); import importlib
mods = [os.path.basename(p)[:-3] for p in sorted(glob.glob('tests/test_*.py'))]
assert len(mods) >= 30, 'found only {0} test modules - run this from snek2/'.format(len(mods))
total = fails = 0
for name in mods:
    mod = importlib.import_module(name)
    for t in [x for x in dir(mod) if x.startswith('test')]:
        total += 1
        try: getattr(mod, t)()
        except Exception as e: print('FAIL', name, t, type(e).__name__, e); fails += 1
print(len(mods), 'modules,', total, 'tests,', fails, 'failed')"
```

**No exact count is pinned here on purpose.** The suite grows with almost every change, so a figure
in this file is false within the week and reads as a failure when it is only stale. The `assert`
carries the check instead, and it is the only thing that ever needed checking: run from the wrong
directory and the glob finds a handful of modules or none, which the floor catches while any real
count passes. What matters in the output is **`0 failed`**.

**A passing suite is not coverage of the change you just made.** `group_obs` took a third signature
and all 24 existing tests passed before and after, because every fixture was an open board where
old and new answers agree. **So mutate the implementation and confirm a test fails.** Every
behaviour change in this project since has been checked that way, and it has repeatedly found
tests that assert nothing.

**Check the failure *type*.** 13 tests were dead for two signature generations — they called
`group_obs` with an argument it had stopped taking, so they raised `TypeError` rather than failing
an assertion, and a `TypeError` looks like noise if nobody is watching. **Every test covering
`group_obs` was dead at the point `group_obs` changed.** A `TypeError` means the test is stale, not
that the code is fine.

For refactors, also diff observations against a fixed-seed run — byte-identical output over a few
thousand steps catches what assertions do not.

## Rendering is off by default — use `watch.py` to see a game

A game window costs **~5.2ms per frame** and the game flips once per step. That is a round trip to
the macOS window server, not our drawing code, so dirty rects do not help. Everything else
`render()` does is 2-4us.

**Two chart-write bugs fixed 2026-08-09 — the cause of the desktop OOM, and the "frozen" viewer.**
(1) `under_the_hood.display_progress` built a figure with `plt.subplots()` every eval; pyplot's global
figure manager kept the artists alive despite `plt.close()`, leaking ~0.45 MB/eval (×2.4 at
`SNEK_CHART_SCALE=3.07`) → ~3 GB per 3M-step arm, which OOM'd the desktop. Fixed by building through the
OO API (`Figure` + `FigureCanvasAgg`, no pyplot) so the figure is GC'd on return — verified with
tracemalloc (growth → reclamation). Do not revert it to `plt.subplots`. (2) `eval_checkpoints.update_chart`
used one `off` flag for both "no window" and "stop entirely", so `SNEK_CHART_WINDOW=0` wrote the eval PNG
*once* and it looked frozen in `chart_viewer`; split into `window_off` (keeps writing the PNG) vs `off`.

**To watch a policy play, run `watch.py` — never turn a window on inside a run:**

```
cd snek2
PYTHONPATH=. python -u watch.py <policy_name>          # follows the newest checkpoint
PYTHONPATH=. python -u watch.py <policy_name> <step>    # pins one checkpoint
```

It renders in its own process, reloads the newest checkpoint between episodes so a live arm's
progress shows up, and costs training nothing. `WATCH_FPS` caps the frame rate (default 90; drop to
20-30 to follow the moves).

**A laptop training launch opens its own chart window** — `snek2.main()` calls
`chart_viewer.spawn_for_policy()`, so no one has to remember to start one. One window per *wave*: the
first arm to start opens it and the other three share it, and the dedupe keys on the batch prefix so a
second batch still gets its own.

**A laptop *eval* now does the same** (2026-08-15) — `eval_checkpoints.main()` calls
`chart_viewer.spawn_for_eval()`, which globs `evals/<prefix>*_eval_progress.png` and watches
`eval_checkpoints.py <prefix>`, so a four-arm close-out opens **one** window (own `<prefix>-eval` lock
namespace) and it self-exits when the last arm stops. Darwin-only via `viewer_enabled()` and skipped for
verification evals (`smoke`/`champion_*`/`bench-*`), so the desktop is untouched — its runner daemon
stays the sole viewer owner there. **Two knobs make the laptop charts crisp on the Retina panel, where
the Tk backend reports `device_pixel_ratio` 1** (so matplotlib renders at 1x and macOS upscales →
blur): the viewer renders its figure at `viewer_dpi()` = **200** on darwin (100 elsewhere; `--dpi`
overrides), and `eval_checkpoints` raises the source PNG to `SNEK_EVAL_CHART_DPI=220` on darwin (default
110, so desktop/standalone are unchanged). The *display* dpi is the real lever — a higher source alone
still looks soft because `chart_viewer` downsamples it into the display figure.

**The dedupe needs the `O_EXCL` claim lock, not just the `pgrep` check — a check-then-spawn dedupe
shipped and opened four windows.** A wave's four trainers launch inside the same second, so all four ran
the `pgrep` check before any had spawned, all four saw nothing, and all four opened a window.
`claim_viewer_slot()` closes that race because create-and-test is one operation. Both checks are kept:
`pgrep` catches a viewer started by hand or by an earlier wave whose lock has aged out of tmp. The lock
holds the **viewer's** pid, not the trainer's — pointed at the trainer it would keep the claim alive
after the window was killed and drop it when the trainer merely finished — and a failed spawn releases
it. A stale lock naming a dead pid is taken over, so nothing suppresses the window permanently.

**‡ The claim lock still lets two windows through, and a laptop close-out opened two on 2026-08-19.**
`b43`'s four evals spawned **two** viewers in the same second (pids 89235/89236, byte-identical
`--glob evals/b43*_eval_progress.png` argv, both appending to the one `*-eval.log`). The hole is that
**`claim_viewer_slot` writes the pid as a second step after the `O_EXCL` create**, and
`hold_viewer_slot` later rewrites the lock with `open(lock, 'w')`, which truncates before writing —
so there are two windows in which the lock file **exists but is empty**. An empty lock parses as
`int('' or 0)` → `holder = 0`, and `if holder and pid_alive(holder)` is then False, so the second
claimant reads a *young* lock as a **stale** one and takes it over. That is the only path in that
function where two claimants both win. Not yet fixed; the mechanism is from reading the code, not a
reproduction.

Two things this implies. **`O_EXCL` alone is not atomic enough when the payload is written
separately** — the claim is only as atomic as its slowest write, so an empty-but-present lock has to
count as *held*, not stale (a create-time mtime check, or writing the pid inside the same `os.open`
via a temp-and-`rename`, would close it). And **a duplicate viewer is safe to retire by explicit pid
with plain `kill`** — SIGTERM reaches `exit_now()`, so it costs no crash report; verified 0 before and
0 after. **Never retire one with `pkill -f chart_viewer`**, which kills every window including the
survivor, and prefer the pid the lock names as the one to keep, so the claim stays consistent. Expect
the killed one to sit in `ZN` until the eval that spawned it exits.
**Sequential calls in one process cannot test this**; race it with several live processes that hold the
claim, the way a trainer does. It refreshes every 1s at 2x size, watches `snek2.py <prefix>` and exits
on its own once the batch stops. `SNEK_CHART_VIEWER=0` turns it off; smoke runs and `eval_only` never
open one.

**`--arms <prefix>` shows the arms that are *running*, not `runs/<prefix>*.png`.** The glob ages badly —
by batch 20 wave 2 it matched eight finished arms plus four live ones, which at 2x scale is a window
taller than the screen. `chart_viewer.live_arms()` reads policy names off the process list each refresh
and the set is **sticky**: an arm that reaches its cap keeps its panel while its siblings run, which is
the point, since the finished curve is the reference. Two traps it has to dodge, both with fixtures:
`pgrep -f snek2.py` matches any command line *containing* the string (git pathspecs, the telemetry
`curl`), so a name is only taken from a line that also runs `python`; and membership is
`batch_prefix(policy) == prefix`, never `startswith`, because `'b20a'.startswith('b2')` is true.

**A process scan alone cannot decide the panel set, and b30 opened on 3 of 4 arms because it did**
(2026-08-14). Whether an arm had a panel depended on a `pgrep`/`ps` snapshot landing after that arm's
`exec` — for four trainers launched inside one second, with the window opened by whichever of them
reached `main()` first, that is a race with **no repair path**: an arm the scan missed reappears only if
a later scan happens to see it. So every trainer now **registers itself** in
`<tmpdir>/snek_chart_viewer_<prefix>.arms` (`register_arm`, one `<epoch>\tpolicy` line, `O_APPEND` so
four simultaneous writers cannot clobber each other), and `wave_files` unions the registry with the
process scan. Each source covers the other's blind spot — the registry cannot miss an arm that launched,
the scan still catches one resumed by hand. **Registration happens before the dedupe returns**, because
the three arms that *lose* the lock are exactly the ones that must still appear in the window the winner
opens. `MAX_WAVE_PANELS` (8) caps the window whatever the file says.

**What separates one wave from the next is liveness, not age — a TTL cannot do it, and b30 opened on
*eight* arms because it tried** (2026-08-14, the same evening as the 3-of-4 fix above). The rule was "admit
any entry inside `ARM_REGISTRY_TTL`", 12 h; b30 was killed and relaunched **71 minutes** later, so the
registry offered the four dead arms alongside the four new ones. No age threshold can tell those apart —
71 minutes is equally plausible as "a wave that started slowly" and as "the wave I replaced". So
`registered_arms` now admits an entry only if it is **younger than `ARM_REGISTRY_GRACE` (120 s)** — a
starting arm, which no scan can see yet, and the registry's entire reason to exist — **or its policy has a
live trainer**, since after those first seconds the process list is the authority on "running".
`ARM_REGISTRY_TTL` stays as a backstop on file growth only.

**That does not weaken the sticky panel, because stickiness never lived in the registry.** `wave_files`
accumulates into its caller's `known` list and never prunes it, so an arm admitted while running keeps its
panel after it finishes — which is why "drop anything not running" is the right rule *here* and would be
the wrong rule there. Both halves have fixtures, and both mutants (removing the liveness test, removing the
grace period) fail tests.

**Relaunching a batch under the same prefix needs the tmp state cleared, and the window started by hand.**
`rm ${TMPDIR}snek_chart_viewer_<prefix>.{arms,lock,log}` before relaunching. A trainer only tries to spawn
a viewer once, at startup, and `viewer_running_for` matches **any** command line containing
`chart_viewer.py --arms <prefix>` — including a shell loop *waiting* for the viewer, which is how a wave
ends up with no window at all. To open one afterwards, go through the real path rather than hand-rolling
argv, so the lock and the registry stay consistent:
`PYTHONPATH=. python -c "import chart_viewer; chart_viewer.spawn_for_policy('<one-arm>')"`.

**The desktop is immune to this whole class, by a different design.** The runner daemon passes the viewer
**explicit PNG paths** for the running jobs (`_ensure_viewer` + `sticky_wave_pngs`) and never `--arms`, so
it reads no registry; the set resets when the box goes idle between waves or when the running arms are
disjoint from the previous set. Its window showed 4 panels through the same relaunch that gave the laptop 8.

**A killed viewer stays a zombie for hours, and `kill -0` calls a zombie alive.** The viewer is spawned
by a trainer that never `wait()`s for it, so it sits in state `ZN` until that trainer exits. Both dedupe
paths believed it: the claim lock read "a live viewer owns this batch" and `viewer_running_for` matched
its still-intact `--arms b30` argv, so **killing a window locked the batch out of ever reopening one** —
the exact opposite of the "nothing suppresses the window permanently" property above. `pid_state()` /
`zombie()` fix both sites. When restarting a window by hand, expect the old pid to linger in `pgrep`
output; that is the zombie, not a second viewer, and `ps -o stat=` tells them apart.

**‡ But `pgrep -f` does *not* always show the zombie, and the two tools disagree** (2026-08-19). A
zombie's argv is no longer readable, so `pgrep -f chart_viewer` returned **nothing** while
`ps -o stat= -p 89236` read `ZN` for the same pid — the b43 eval viewer, unreaped because the eval
that spawned it never `wait()`s. So a process scan can under-report as easily as over-report, and
**"pgrep found nothing" is not proof a pid is gone**; read `ps -o pid=,stat= -p <pid>` before
concluding it. The claim lock survives this because it checks `zombie()` on the pid it names rather
than scanning, which is what let the batch reopen instead of staying locked out.

**‡ A failed `pgrep` used to read as "nothing is running", and it closed a live window**
(2026-08-19). `pgrep` exits 0 for a match, **1** for no match and **>= 2** for an error, and all of
those produce empty stdout — so `chart_viewer._matching_commands`, which read only stdout, turned a
check that *failed* into the strongest possible answer. `_training_alive` then returns False, and six
of those in a row close the window: the `b43` window exited at 13:59 while `b43b-lowlr-b29a` still had
five hours to run. It now raises on `>= 2`, which is what its docstring always claimed. **The cause of
that particular exit was never reproduced** — the check passed on the same pattern minutes later, and
the obvious suspect (a sibling pid dying between the `pgrep` and the `ps`) was tested and falsified, so
treat the fix as closing a real hole rather than as a diagnosis. `ps` is deliberately left alone: it
exits 1 both when every listed pid is gone and on a bad pid, so there is nothing to tell apart.

**A `kill` that seems not to work may have worked.** `kill -0 <pid>` succeeds on a zombie, so a
wait-for-exit loop written on it never finishes. Read `ps -o stat=` instead.

**A test in `tests/test_chart_viewer.py` must stub `subprocess.Popen`, not only `subprocess.run`.** A
fixture that asserted a spawn was *blocked* opened three real b30 windows on the laptop when the
assertion failed — on top of a wave that was training. Stub both, so a wrong test cannot open a window.

It is off by default anywhere but darwin, because on the desktop the runner daemon owns the viewer
(`desktop/runner/runner.py::_ensure_viewer`) — it injects the graphical session's
`DISPLAY`/`XAUTHORITY`, which a systemd-launched trainer does not have. **Don't add a trainer-side
launch there**; two owners means two windows per wave.

**The viewer must exit through `chart_viewer.exit_now()`, and its signal handler must be installed
*after* `subplots()`.** Otherwise killing it aborts the process — macOS pops a "python quit
unexpectedly" crash-report dialog. Two separate causes, both real, both measured on 2026-08-09:

- **Tk owns the OS-level SIGTERM handler**, because it installs its own while creating the figure's
  window — inside the first `subplots()` call, not at import. Tcl's handler runs `Tcl_Exit` straight
  off the signal trampoline, which destroys the windows, fires their `<Destroy>` bindings, and calls
  back into Python with no thread state. An install that happens *before* the figure is dead code:
  5 of 5 kills still aborted. `make_figure()` exists to keep the two together.
- **Interpreter shutdown with a live Tk window** ends the same way — a late Tk event finds a
  half-finalised interpreter. So `exit_now()` closes the figures *first*, while the interpreter is
  fully alive, then `os._exit()`s to skip finalisation entirely. The viewer owns no state, so
  skipping cleanup costs nothing.

Both surface as `_Py_FatalError_TstateNULL` under `PythonCmd` in the crash report, so read the frames
*below* it to tell them apart — `Tcl_Exit`/`_sigtramp` means the signal path. **It is race-dependent:
one clean exit proves nothing.** Verify with a loop of ~5 kills and `ls ~/Library/Logs/DiagnosticReports
| grep -c python` before and after. Retiring a viewer that predates this fix needs `kill -9`, which the
kernel handles without reaching Python at all.

**Window size is `SNEK_TILE_PIXELS` and is cosmetic only.** Every pixel constant derives from it.
Observations are built from tile positions, verified by a fixed-seed hash coming out identical at
10, 20, 25 and 40 pixels per tile. **It must be set in the environment before `snake_constants` is
imported** — `from snake_constants import *` binds a copy, so a later assignment never reaches
`Snake.py`.

**Training cannot draw at all — there is no switch.** `snek2.main()` selects
`SDL_VIDEODRIVER=dummy` unconditionally. `eval_checkpoints.py` keeps `EVAL_RENDER=1` for debugging
by hand (worker 0 only, ~5x the wall clock); that plus `watch.py` are the only ways to see a game.

**A process that will not draw must select the dummy driver, not merely skip drawing.**
`Game.__init__` calls `set_mode()` unconditionally and `reset()` blits and flips, so a process with
a real driver and `display=False` opens a window, paints it white once and never touches it again —
which looks like a broken window rather than an absent one. `main()` does this, not module scope:
`eval_checkpoints.py` and `watch.py` both import `snek2` for `build_q_net`, and an import-time
`setdefault` would suppress *their* windows too.

**When something here seems slow, profile the critical path rather than the hot-looking code.** An
11x speedup of the observation code and a 6.8x speedup of policy inference both moved eval wall
clock by approximately nothing, because neither was the slowest worker. And **measure a render
change per-episode with a fixed policy, not by timing a training run** — an earlier "83s → 56s"
claim was learning-speed variance attributed to a flag that was a no-op at the time.

## Stopping a batch: file the charts before anything else

**A trainer does not stop on SIGTERM — use `kill -9`, and verify.** Measured 2026-08-14 on both hosts:
eight arms took a plain `kill` and kept stepping, the laptop's four advancing another ~25k steps over two
minutes while the session assumed they were shutting down. Nothing in `snek2.py` installs a handler; the
TF-Agents worker layer swallows it. `kill -9` is safe because every durable file is written `.partial`
then `os.replace`d, and checkpoints land every 1,000 steps. Also **do not test liveness with `kill -0`**
— it succeeds on a zombie; read `ps -o stat=`, and expect forked self-eval workers to outlive the parent
briefly (`pkill -9 -f "python -u snek2.py <prefix>"` finishes the job). On the desktop, **pause the queue
before killing** or the freed slots refill within one poll:
[procedure](snek2/desktop/README.md#rebooting-the-box-and-what-recovers-by-itself).

**When arms are killed, `charts.md` is updated in the same pass** — refresh the images, add the new
batch's section at the top, and retire the oldest so it holds at most six batches (moved verbatim to
`hyperparamTuning/archive/charts-archive.md`, PNGs left in `charts/`). Full checklist, including what
each caption has to say and why the cap is six:
[`snek2/hyperparamTuning/hyperparamTuning.md`](snek2/hyperparamTuning/hyperparamTuning.md#when-you-stop-a-batch-of-arms).

**Stop time is the *finalization*, not the only time `charts.md` moves.** Per the progress-update rule
above, the file is kept current throughout a batch's life, so by the time arms are killed the section
usually already exists with training-only numbers — stopping just fills in the close-out/HOF-500 figures
and retires the oldest batch. Never treat an unfinished batch as a reason to leave `charts.md` alone.

**`refresh_charts.sh` does not edit `charts.md`.** It copies images only, so a clean run of it looks
like the charts are handled when no caption has been written. That drifted once to **12 undocumented
arms across batches 5-7**. The completeness check at the top of `charts.md` is what catches it, and it
has to be run against the archive files too.

## Eval cost

**‡ The default engine is the vectorised one, on both hosts** (2026-08-24). A close-out or HOF pass is
one command, and everything in the rest of this section describes the **scalar** path it replaced:

```
cd snek2
PYTHONPATH=. python -u vectorized/vec_wave.py --chain top50 b45     # a batch's whole measurement
```

It takes `eval_wave.py`'s CLI — same selectors, same `--chain`, a bare batch id expands to its arms —
because it imports `eval_wave`'s own argv functions rather than copying them. It writes the **canonical**
`runs/<policy>_checkpoint_evals.json` and `evals/<policy>_eval_progress.png`, so nothing downstream
changes. Measured **~40x** the scalar path's throughput (348 vs 8.55 episodes/s machine-wide), and
validated against it at four levels ending in a 24-checkpoint × 500-episode head-to-head that agreed to
**−0.058 pp (z = −0.28)**. Full account: [`snek2/vectorized/README.md`](snek2/vectorized/README.md).

Four things to carry:

- **It is flat and ungated.** No screen/confirm tiers, no `EVAL_MIN_ACHIEVABLE`. So every row in one of
  its files is full length and directly poolable, and `pooled_equal_effort` / `min_achievable` are
  `null` rather than a number to check. **That is a file-format boundary like the gate boundaries
  below** — a vec file's rows are not censored, so its graph-100% tier *is* comparable across arms in a
  way a gated file's is not.
- **`EVAL_WORKERS` and `EVAL_LANES` do nothing to it**, and both hosts stop setting them. They size TF
  worker processes; this engine has none. The analogue is `VEC_WAVE_PROCS`, default `os.cpu_count()`
  **− 2**, and **each host's optimum is its own** — 12 on the laptop (2-6% idle; 16 processes reach 0%
  idle and are 20% *slower*), **16 with `TF_NUM_INTRAOP_THREADS=1` on the desktop**, whose 16 is SMT
  threads over 8 cores. Neither number transfers; sweep a new host with `vec_wave_sweep.py`.
- **‡ c51 runs on it too, since 2026-08-24 — there is no longer a fallback.** `vec_eval` used to refuse
  categorical policies and `vec_wave` used to hand them to `eval_wave.py`; both are gone, because the
  engine never reads a Q head. It builds through `eval_agent.build_eval_agent` (which picks the agent
  class off `arch.json`) and calls `policy.action(...)`, and a categorical agent's greedy policy reduces
  over its own support. Validated on six `b38a-c51fc320eps3125seed1` checkpoints, 200 episodes per
  checkpoint per engine: **−0.17 pp, z = −0.10**. So **every** batch is measured by one engine now, and
  the only reason to reach for `eval_wave.py` is the opt-out below.
- **‡ A GPU cannot help this eval, and MPS silently breaks it** (tested 2026-08-24). The policy is
  **8.2%** of a step at width 1024 and the numpy observation build is 4296 us of 5050 us, so **1.09x is
  the ceiling for any accelerator, however fast** — the bottleneck is a bitboard flood fill, not a
  tensor program. `tensorflow-metal` is also **2.4x slower** on the policy call at that width (a fixed
  ~900 us a call against a 24 MFLOP batch), i.e. ~10% slower end to end. **But the reason it is
  disqualified is correctness:** four hall-of-fame champions measuring 97-98% perfect measured **0.0%**
  on MPS, with no error raised and a *faster* wall clock, because the composed `GreedyPolicy(QPolicy)`
  graph disagrees with `argmax` over its own Q-values on 23 of 64 states — by a **median 0.64** in
  reward units, so not float tie-breaking, and every component (`argmax` eager and graph,
  `tfp.Categorical.mode`, the restored weights, the forward pass to 5.7e-06) is individually correct.
  **Run [`perDiagnostics/eval_device_split.py`](snek2/hyperparamTuning/perDiagnostics/eval_device_split.py)
  `--verify` before trusting any new device, accelerator build or TF version** — the failure mode is a
  silent zero, which reads as a bad arm. Full account:
  [`snek2/vectorized/README.md`](snek2/vectorized/README.md).
- **The opt-out is `SNEK_EVAL_ENGINE=scalar`** — laptop script env, desktop `runtime.json`'s
  `eval_engine`, or a single job spec's `env`. Kept because it is the only way to reproduce a
  pre-switch measurement, and because a regression here has to be answerable without a deploy.

**Run a close-out or a HOF eval as 4 parallel processes, each with at least 4 workers** (standing
instruction, 2026-08-15, and **scalar-path only** — a vec wave shards itself). Give every arm in a wave its own `eval_checkpoints.py` process and start all
four at the same time. Set `EVAL_WORKERS` to 4 or more on each. The HOF-500 re-measure runs the same
way. This is the measured throughput point — 4 processes × 4 workers fill the 14 cores (~12.7 busy), and
it is how the desktop already closes out. **Do not run the arms one after another**; that leaves most
cores idle and runs several times slower. 16 spawned workers hold ~3.7 GB, inside the band on both hosts.
Each process writes its own `<policy>_checkpoint_evals.json`, so the results do not collide; only the
`evals/` chart PNG is shared, and that is cosmetic.

**`EVAL_WORKERS` is close to free, and lowering it to save CPU does the opposite.** Measured
seconds per episode: **1.03 at 2 workers, 0.33 at 10, 0.30 at 20.** TensorFlow's thread pool costs
about a core whether its batch has 2 rows or 20, so a small count pays full inference overhead for
a fraction of the work. Batch 10's close-out was launched at 2 workers to hit a ~50% CPU target and
ran 2.8x slower for *more* CPU per episode.

**Evals are allowed to run the machine hot, from 2026-08-08.** Close-out latency is the bottleneck
after many batches, so a close-out may saturate the box and slow any trainer sharing it — training is
resumable and loses only wall clock, while an un-analysed batch blocks the next launch. Defaults are
now `EVAL_INDEPENDENT=1` and `EVAL_WORKERS=4`; see
[`snek2/hyperparamTuning/hyperparamTuning.md`](snek2/hyperparamTuning/hyperparamTuning.md#-evals-run-hot-on-purpose).
This replaces the older "to be gentler on the machine, run fewer arms, not fewer workers" — the
measurement it rested on still holds, the priority does not.

**`EVAL_EPISODES` no longer has to divide `EVAL_WORKERS`.** The independent path splits the remainder
one episode at a time (`eval_workers.split_quota`), so a request is exact. The batched path still
rounds up to whole rounds, so the old preference applies only with `EVAL_INDEPENDENT=0`.

**Do not trust `resource.getrusage(RUSAGE_CHILDREN)` for a `ParallelPyEnvironment`.** It only counts
reaped children and the workers live for the whole run, so it reported roughly the parent alone —
undercounting a 20-worker eval by ~3x. Read total CPU from `top -l 2 -n 0`.

**The close-out runs in three stages** (`EVAL_SCREEN_EPISODES`, on by default): every checkpoint
whose graph point was 100% gets the full 100 episodes immediately (uncapped), everything else
selected gets 20, and the best `EVAL_CONFIRM_COUNT` *of those screened* get 80 more. ~2.4x fewer
episodes on a large arm — but the saving collapses to 1.0x when the screened pool is smaller than
the confirm count, which happened on `b11c`.

**`top50` is not a budget — "N is a target, not a quota", and a *good* arm blows through it.**
`select_top_checkpoints` measures **every checkpoint whose graph eval reached `ALWAYS_EVAL_SINGLE`**, past
N; only the fill band down to `MIN_EVAL_SINGLE` is limited by the count. A normal arm climbs from 0, so few
checkpoints qualify and `top50` really does measure a few dozen. **An arm continued from an
already-excellent checkpoint spends its entire run above the mandatory threshold**, so nearly every
checkpoint qualifies: `b43`'s four continuation arms selected **791, 1196, 803 and 826** checkpoints, and
that close-out ran **~15 hours**. `b42` is the contrast from the other side — the same selector, but it
*decayed*, so only 261-373 qualified and it finished overnight.

**The dominant cost is the uncapped full-length tier, not the selection count.** A checkpoint at
`ALWAYS_FULL_SINGLE` skips the screen and takes the whole `EVAL_EPISODES`; on `b43`/`b44` that tier was
**791-1300 checkpoints per arm**, i.e. essentially the entire bill. Raising `num_eval_episodes` to 20 is
what cut it: the threshold is a *rate*, so 95% now means 19 or 20 perfect games out of 20 rather than
collapsing onto 10 of 10.

**‡ `ALWAYS_FULL_SINGLE` is 95, equal to `ALWAYS_EVAL_SINGLE` — the mandatory tier *is* the full-length
tier, and that is affordable only because of the gate.** The obvious fear is that an uncapped
full-length tier admitting every 19/20 checkpoint would multiply the bill. It does not: simulated on
b43/b44's own curves at 20-episode graph evals, moving this threshold from 100 to 95 changed total
close-out episodes by **-1%**, every arm within ±3%. The reason is `EVAL_MIN_ACHIEVABLE=97` — a 19/20
checkpoint whose true rate is under the gate is abandoned after 4 failures, often at the 20-episode
floor, so it costs about what a screen would have; one that survives deserved the measurement. **The
result is conditional on the gate being stricter than the tier**, so never loosen the gate and leave
this at 95. `tests/test_selection_tiers.py` fails if that ordering breaks.

What it buys is coverage: under the old 100 threshold, **427-575 checkpoints per arm** sat at 19/20 on
the graph, were screened to 20 episodes, and were capped by `EVAL_CONFIRM_COUNT` — so an arm's best
checkpoint could finish on a 20-episode row. Now every one gets a full-length attempt bounded by the
gate rather than by a quota.

Measured on `b43`/`b44`'s own curves, the retune (20-episode graph, tiers 95/90, `top50`, gate 97) cuts
close-out episodes by **~25%**, against **+15,500 self-eval episodes per arm** on the training side —
roughly a 2:1 trade in the close-out's favour, and the training half is *not* negligible.

So **still budget a continuation batch's close-out in hours, not minutes, and never read one still running
after 7 hours as hung.** The remaining lever is the **selector, not the worker count** —
`above:<threshold>` reads a prior close-out's 100-episode measurements instead of the noisy graph, which
is what the HOF pass already does. Raising `EVAL_WORKERS` does not help much here, because the cost is
checkpoint *count* times per-checkpoint restore, not episodes per checkpoint.

**‡ But part of those hours was never measurement, and the diagnostic is two `grep -c`s** (fixed
2026-08-20, `adbec2904`). A wave's *lane* prints `N episodes in Xs` when it finishes a measurement; the
*controller* prints the `[ n/N ]` row when it folds one. **Those counts should track each other. When
lane completions run ahead, the controller is the bottleneck, not the box:**

```
grep -c "episodes in" <log>          # measurements the lanes have finished
grep -cE "^\[ *[0-9]+/[0-9]+" <log>  # measurements the controller has folded
```

The gap was real on both hosts. `on_round` fired once per `EVAL_WORKERS` episodes — 125 times for a
500-episode measurement — and each call rebuilt every banked row and re-serialised the whole result file,
five payloads deep in a wave because `wave_eta_seconds` prices the wave off *all* the arms. Cost is
O(rows) per write and O(episodes) writes, so the two multiply: **58 s of single-threaded bookkeeping per
measurement at 552 rows, against the 46 s four lanes need to produce one.** So the controller overtakes
its own lanes partway through a long arm, and once the queue drains it is the only thing running —
b43's HOF pass finished all 767 measurements and then folded alone for 90 minutes with 16 workers idle
and the machine 95% free. b44's, at 2,235 measurements, was heading for ~30 h of which half was
bookkeeping.

`eval_plan.WriteGate` now bounds progress writes by wall clock and `eval_plan.RowCache` memoises
`build_row` per step, taking a write from 468 ms to 43 ms and the per-measurement cost from 58 s to 1 s.
Both write paths use them, so this applies to `eval_checkpoints.py` too. Three things to carry:

- **A running eval that looks slow is not necessarily slow.** Check the two counts before adding workers.
- **All 16 workers idle at 0% while the controller burns a core is the signature.** `ps -o time` deltas
  and `sample`/`top` on a worker settle it in a minute; a worker parked in `read` is waiting for a
  command that its lane is too busy to send.
- **A killed wave loses only what has not been folded**, because a finished measurement forces its
  write — the unfolded ones live in the controller's outbox and nowhere else. Read the gap before
  killing: it is exactly what a resume will have to re-measure.

**`EVAL_MIN_ACHIEVABLE=97` abandons a checkpoint mid-measurement** once it cannot reach 97% even if
every remaining episode is perfect — at 100 episodes, once more than 3 have failed (it was 95, and "more
than 5", until 2026-08-19). Full-length work drops further than the **31%** of a flat pass measured at the
95 gate (52% at 90, 71% at 85); the 97 figure has not been measured yet, so do not quote one. **No ranking
among rows that reach the gate can change**, because the test is arithmetic rather than predictive.
Abandoned rows carry `abandoned: true`, are shorter, and are **not comparable with full-length rows**.
`pooled_equal_effort` is exact at any gate. `EVAL_MIN_ACHIEVABLE=0` turns it off.

**97 leaves exactly one point of headroom under the HOF selection gate of 98**, and that invariant is
load-bearing: HOF reads `above:98` out of the close-out's own file, and only rows reaching the close-out
gate are measured full length, so a close-out gate at or above 98 would abandon precisely the rows the
re-measure needs and starve it silently. **`tests/test_selection_tiers.py` is now the only thing that
pins it**, and that is deliberate: `runner.py` and the laptop chain script each carried a copy of the
assertion, and both copies are gone along with the gate numbers they asserted — the daemon strips the
protocol keys from the env it inherits and the chain script sets none, so `eval_plan.py` is the single
definition. **Do not raise either number without re-reading the other.** The invariant is vacuous for a
`vec_wave` file, which has no gate to abandon rows with, and it still governs every scalar close-out.

**At 97 most arms will have no full-length row**, since few checkpoints clear 97%.
`best_full_length_row` then relaxes to **half-depth** rows and prints `[truncated]`. It must never
relax to *all* rows — that hands the title to a 20-episode screen on a lucky 20/20.

**A file's gate is in its payload as `min_achievable`; check it before pooling anything.** Batches 11
and 13 have no gate, batch 14 has 90, batches 15-44 have 95 (96 where the desktop or the chain script
pinned it), and **batch 45 onward has 97** — **except that a file measured by `vec_wave` has no gate at
all** (`min_achievable: null`), which is the third era, not a missing field. Read the payload, never the
batch number. Cross-batch best-checkpoint stays valid for "did this arm
produce a ≥`gate`% checkpoint", since anything at or above a gate is measured full length under it — but
the graph-100% tier is censored by any gate and must not be compared across them.

**‡ 2026-08-19 is also a *graph* boundary, not only a gate one, and it biases one metric.**
`training.num_eval_episodes` went 10 → 20, so batches 1-44 report `perfect_percent` in multiples of 10 and
batch 45 onward in multiples of 5. **Banded mean perfect rate stays comparable** — a 20-episode estimate of
the same true rate has the same expectation. **`best_perfect30` and `max_single_eval` do not**: they are
maxima over a noisy statistic, and halving the noise lowers them systematically, so a 20-episode arm looks
slightly *worse* on those than a 10-episode arm of identical quality. Compare those two metrics only within
an era, and prefer banded means or the close-out across the boundary.

Two consequences for reading the output file: rows have **different episode counts**, so pooling
them over-weights the winners and reads high — use the equal-effort figure the run prints, or the
graph-100% tier if the run predates that field — and best-checkpoint must come from full-length rows
only, which `eval_progress.best_of()` enforces.

**The full-length tier is a coverage guarantee, not a shortlist of champions.** 6 of the 8 arms
measured across batches 10-11 found their best checkpoint *below* the graph-100% tier — which is
exactly why `ALWAYS_FULL_SINGLE` came down to 95, so those checkpoints are measured properly instead of
competing for `EVAL_CONFIRM_COUNT` slots. Below 95 that job is still the confirm count's.
