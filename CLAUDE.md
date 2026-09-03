# Snek

A reinforcement-learning snake, in three eras. Each is a separate directory and each keeps its own
instructions; **this file is only what is true of the whole repository.**

| directory | era | state |
|---|---|---|
| `theSchlong/`, `theSchlongCardinalDirs/`, `theSchmid/`, `theSchlort/`, `baddieStuff/`, `humanPlayer/` | the originals | kept as-is for posterity. Do not edit |
| [`snek2/`](snek2/) | TensorFlow + TF-Agents, batches 1-47 | **frozen 2026-08-28.** Runnable for A/B, not edited. Its manual is [`snek2/CLAUDE.md`](snek2/CLAUDE.md) |
| [`snek3/`](snek3/) | PyTorch | **active.** Its manual is [`snek3/CLAUDE.md`](snek3/CLAUDE.md), its design [`snek3/plans/pytorch-port.md`](snek3/plans/pytorch-port.md) |

**Work in `snek3/`.** snek2 is copied *from*, never edited — code duplication between the two is
deliberate. If a rule you need is missing from `snek3/CLAUDE.md`, it may be in `snek2/CLAUDE.md`
describing snek2's version of the same thing; carry it across rather than editing snek2.

**`README.md` is for humans and stays barebones** — the eras, a sentence each, a gif each. Anything
an agent needs belongs here or in an era's own manual, not there. Details that were moved out of it
on 2026-08-29 are the two sections below.

## What each era is, and how to run it

| era | headline | env | train |
|---|---|---|---|
| snek2 | TensorFlow + TF-Agents, batches 1-47, peaking at a **98.7% perfect-game rate**. **Frozen 2026-08-28** — runnable for A/B against snek3, not developed further | `snek` | `cd snek2 && python snek2.py <policy_name>` |
| snek3 | PyTorch, the active one. Same game and same 30-value observation as snek2, so a snek2 champion's weights convert straight across; a clean-slate implementation of everything else | `snek3` | `cd snek3 && PYTHONPATH=. python -u train.py <policy_name>` |

The argument is the policy name, and in both eras it doubles as the checkpoint directory under
`savedPolicies/<policy_name>/` and as the prefix for the run's own graph and report in `runs/`, so
several policies train independently without overwriting each other.

Every snek3 eval writes `runs/<policy>.png` (the graph, covering the policy's whole history across
restarts), `runs/<policy>.md` (graph, config and eval table, generated from the values the run
actually used) and `runs/<policy>_evals.json` (the measurements later sessions read).

To watch a snek3 policy play, or record it:

```
PYTHONPATH=. python -u watch.py <policy_name>
PYTHONPATH=. python -u record_gif.py <policy_name>
```

Where the rest is written: snek2's investigation is [`snek2/hyperparamTuning/`](snek2/hyperparamTuning/)
and its record checkpoints and recordings are [`snek2/hallOfFame/HOF.md`](snek2/hallOfFame/HOF.md);
snek3's investigation is [`snek3/docs/`](snek3/docs/) and its records are
[`snek3/hallOfFame/HOF.md`](snek3/hallOfFame/HOF.md).

## Every training opens a chart window, and no agent launches one by hand

**One window per box, showing every training running on it**, on the laptop and the desktop alike.
Every arm asks for it, the viewer settles which one becomes it, later arms join it, and it closes
itself a few minutes after the last one finishes. Nothing has to be launched, and a `runs/*.png` glob
is no longer the way to watch a batch.

**The "one" is enforced by an `flock` the viewer holds, not by the arms agreeing.** Every arm of a
wave spawns a viewer; the losers exit in ~0.3 s having drawn nothing. That is deliberate: a launcher
that has to *decide* whether to spawn is a claim protocol, and this project has now had two of those
go wrong — snek2's 500 lines of `pgrep` machinery, and snek3's own `O_EXCL`-plus-takeover, which
opened **five windows on the desktop on 2026-08-29** and a mean of 6.6 per 8-arm batch when measured.
A lock the kernel holds cannot be taken twice, is released however its holder dies, and leaves no
state behind for the next window to misread. **Do not reintroduce a launcher-side claim.**

**A panel stays for the rest of the wave once it appears**, so a batch with one arm left still shows
all four — three finished arms are most of what a glance is for. The accumulated set is dropped when
the box goes quiet and a new arm appears, so the next batch does not draw its predecessor's charts.

**A stage-B close-out gets the same treatment**, in a second window of its own: `tools/closeout.py`
opens one panel per arm of the batch it is measuring and it closes when the pass ends. Same viewer,
same launcher, its own slot — so a box can show a batch training and a batch being measured at once,
and `SNEK_CHART_WINDOW=0` still silences both.

The window is **disposable and the training is not**: it runs in its own session, no training reads
from or waits on it, and no training reopens it. So killing it, closing it, or relaunching it with
`cd snek3 && PYTHONPATH=. python -m tools.chart_window` cannot affect a run — which is what makes it
safe to fix a window while four arms are training.

**The window is sized from the monitor, from the charts, and from an optional cap — smallest wins.**
It probes the display it opens on, so the laptop and the desktop need no separate settings; it never
draws a panel wider than the source PNG, so a window with one small chart in it opens small rather
than upscaling to fill the screen; and `SNEK_CHART_WINDOW_MAX_PX` caps the width outright on either
box. `SNEK_CHART_WINDOW_SCALE` still takes a fraction of the screen budget.

`SNEK_CHART_WINDOW=0` in a training's environment
turns it off; the mechanism is [`snek3/tools/chart_window.py`](snek3/tools/chart_window.py),
[`snek3/tools/eval_window.py`](snek3/tools/eval_window.py) and
[`snek3/tools/live_runs.py`](snek3/tools/live_runs.py).

## snek3's procedures are skills, not prose

Launching a training, queueing a batch on the desktop, stopping an arm, deploying code, running a
progress update: each is one skill in [`snek3/skills/`](snek3/skills/), invoked by name. Use the
skill rather than reconstructing the steps — it carries the commands in order and the traps that
break a run, and **a skill that fails is a bug in the skill**: fix the situation, then edit the skill
so the next session does not hit it. [`snek3/skills/README.md`](snek3/skills/README.md) has the
conventions.

The files under `snek3/docs/` stay what they are: what is *true*, and the incidents behind each rule.

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

One conda env per era. Activate before running anything, and **say which one you mean** — snek3 has
no TensorFlow and snek2 has no torch, so a command run in the wrong env fails at import.

| project | env | python | key packages |
|---|---|---|---|
| snek3 | `snek3` | 3.12 | torch, numpy 2.x, pygame, matplotlib, pillow, imageio, pytest |
| snek2 | `snek` | 3.10 | tensorflow, tf-agents, numpy 1.26, cpprb, pygame, cv2 |

**`conda run` buffers stdout**, even with `python -u`. A backgrounded run's log can stay empty for
90+ seconds while the process is fine, and `kill -9` then discards the buffer permanently. For any
background run where you need live output, call the env's python directly:

```
/opt/miniconda3/envs/snek3/bin/python -u train.py smoke > log 2>&1 &
```

Reserve `conda run` for short one-shot commands where only the final output matters.

## Git workflow

**Code: leave finished work uncommitted.** Do not `git commit`, `git push` or even `git add` a
**code** change until the user explicitly approves *that* change — "push" or "commit this" is the
go-ahead and applies only to the change in front of you. The reason is review: the user reads diffs
in their editor, and committing or staging moves the change out of the working tree where it is
visible. So the loop for code is: make the edit, describe what changed, stop and wait.

**Documentation and instructions: commit and push without waiting** (standing authorization,
2026-08-14). Any change that is *only* Markdown — this file, `snek3/docs/*.md`, `plans/*.md`, the
`README.md`s — is committed and pushed as soon as it is done, no confirmation needed. This includes
progress-update doc edits and instruction changes like this one. **The exception is docs-only:** the
moment a commit also touches code, config, or anything that changes program behaviour, the whole
change reverts to the code rule above and waits for approval. Chart images that accompany a docs
edit ride along with the docs commit.

**Tests: commit and push without waiting** (standing authorization, 2026-08-14, for the stated
reason that the user does not read these diffs). `snek3/tests/*.py` — assertions about behaviour,
never behaviour.

**The same only-if rule applies, and it is the important half.** A commit qualifies only when it
touches *nothing else*. A test that arrives alongside the change it pins — which is the normal case,
since these instructions ask for a test in the same pass as the logic — is part of a **code** change
and waits with it. So the split is: a test written to *pin existing* behaviour goes straight up; a
test written for behaviour being changed in the same working tree does not.

**Pushing to `ops` starts real work on another machine**, so it falls under the code rule: queue a
job only when the user has approved *that* job.

Read-only git commands are fine at any time.

## What must never be deleted

| path | rule |
|---|---|
| `snek3/runs/`, `snek2/runs/` | every real arm's graph, report and measurements |
| `snek3/hallOfFame/`, `snek2/hallOfFame/` | record checkpoints, outside any rotation |
| `snek2/savedPolicies/train/` | the user's own run |
| `snek2/evals/archive/` | every chart snek2's retired eval-chart sweep took |
| anything under `snek2/` | the whole directory is frozen |

**A wrongly kept file costs a few KB; a wrongly deleted one costs a training run.** When in doubt,
keep it, and `grep -rn <name> --include='*.md' .` first — the docs link files by name.

**Throwaway output is fine to delete**: smoke tests, speed benchmarks, `snek3/gifs/`, and
verification evals. Judge by what produced it, not by where it lives.

## There are two compute hosts — say which one you mean

Since 2026-08-08 a dedicated desktop (`the-claw-den`) also runs trainings and evals, driven entirely
by git: you commit a job spec, it runs it, it pushes results back. **snek3's daemon owns the box as
of 2026-08-28**; full docs in [`snek3/desktop/README.md`](snek3/desktop/README.md).

| | laptop | desktop `the-claw-den` |
|---|---|---|
| limit | **8 trainers** | `max_trainers` (8; no host ceiling), `eval_shards` ≤ 16 |
| check | `ps -Ao pid=,command= \| grep '[t]rain.py'` | **`git fetch origin ops-status && git show origin/ops-status:status.json`** |
| queue work | launch by hand | commit a JSON spec to `queue/pending/` on the `ops` branch, then trigger |
| start it now | — | `ssh the-claw-den 'Snek/snek3/desktop/trigger'` |

**Every progress update commits every arm's `runs/<policy>.png` and `.md`, live desktop arms included**
(rule changed 2026-09-02; before that live arms' charts were never committed). Those two files are
pictures of the JSON, redrawn on every eval, so a committed copy is simply a snapshot that the next
update overwrites — and it is what the GitHub-Pages chart viewer at `snek3/viewer/` shows, so a live
batch is visible there between close-outs. A live desktop arm's charts are pulled from the box first:

```
rsync -a --include='<batch>*.png' --include='<batch>*.md' --exclude='*' the-claw-den:Snek/snek3/runs/ snek3/runs/
```

**The JSON is the opposite: never commit a *live* desktop arm's `runs/<policy>_evals.json` or
`_checkpoint_evals.*`.** `_evals.json` is single-writer (the trainer) and is what the arm's chart and
report are rebuilt from across restarts; the stage-B file is a pass in progress. They arrive on the
`results` branch at close-out, and only then are they committed here. A laptop arm's own files are fine.

**The box's deploy expects the collision this creates and settles it by what each file is.** A
committed chart the box also holds would abort `git merge --ff-only` as "untracked working tree files
would be overwritten", so `snek3/desktop/deploy` runs on the box instead of a bare merge: it **keeps
the box's own pictures** (saves the bytes, merges, writes them back — the box drew every chart the
laptop ever committed, so the committed copy is always the older snapshot and a finished arm's final
chart is never replaced by a mid-training one), stages any other colliding file whose bytes match the
incoming blob (a closed batch imported from `results`), and **stops with exit 3, touching nothing, if
any JSON differs** — that is a live arm's file committed by mistake, and the fix is `git rm --cached`
on master, never overwriting the box. The `desktop-deploy` skill has the procedure.

**One implementation for both boxes, wherever the behaviour wanted is the same.** The two hosts
differ in what they *have* — cores, a monitor, a queue — and almost never in what the code should
*do*, so a laptop path and a desktop path for the same job is two places to fix every bug and one of
them gets forgotten. Where a difference is real, read it at runtime (probe the display, count the
cores) rather than branching on which box you are; where a knob is wanted, make it one knob both
sides honour. `tools/chart_viewer.py` draws both boxes' windows and both windows on each box, and the
sizing knobs live in `chart_window.sizing` for exactly this reason — the eval window had already
drifted into parsing the environment for itself, and had stopped honouring one of the two knobs as a
result. **When you fix something on one box, check whether the other has its own copy of it, and if
it does, delete the copy rather than fixing it twice.**

**Neither check sees the other host**, so **"N arms running" is meaningless without naming the box**,
and any progress report has to check both.

**`git fetch` is not optional in that command, and leaving it out is the single most repeated mistake
in this project's history with the desktop.** `git show origin/ops-status:…` reads a **local
remote-tracking ref**, which changes only when you fetch. Without the fetch you are shown an
arbitrarily old snapshot *with no indication that it is old* — and because the payload contains a
timestamp, the natural misreading is "the daemon stopped at 08:33" when the truth is "my ref is from
08:33". That has produced three false alarms, one of which reported a 10-hour-dead daemon and a
failed dispatch while the box was healthy and had already finished the batch.

So **a stale-looking timestamp is your own ref until you have fetched and re-read it.** Never report
the desktop as down, drained or off-LAN from an unfetched read. The ladder, in order: fetch and
re-read; `ssh the-claw-den 'Snek/snek3/desktop/trigger'`, which makes the daemon publish *now* and
reports whether it is polling at all; then `ssh the-claw-den -o ConnectTimeout=8 -o BatchMode=yes`,
which settles reachability in seconds. Only after all three fail is the box worth calling
unreachable. The git bus works from anywhere; `ssh` is home-LAN only (mDNS, no Tailscale since
2026-08-13) — but "probably off-LAN" is a conclusion, not a starting assumption, and it has been
wrong.

**‡ `pgrep -f` and `pkill -f` match the shell that runs them, and this is the default outcome, not an
edge case.** `-f` scans full command lines, and the invoking shell's own command line contains the
pattern verbatim, so the scan always includes itself.

**The destructive half comes first, because it is the half that has cost work.** `pkill` *acts* on
the match instead of counting it, so it SIGKILLs its own shell:

| when | what was typed | what happened |
|---|---|---|
| 2026-08-25, abandoning `b46`'s first wave | `ssh the-claw-den 'kill -9 …; pkill -9 -f "snek2.py b46"; ps …'` | the remote shell's own command line contained `snek2.py b46`, so `pkill` matched it and killed the session: **exit 255, no output**. The kill had in fact worked — a follow-up `ps` showed 0 b46 processes and `free -m` was back to 11 GB. Read exit 255 here as "cannot confirm", never as "failed" |
| 2026-09-01, relaunching the desktop chart window | the relaunch was piped through `tail -3`, then `pkill -f "tail -3"` | matched its own `ssh` command line, killed the pipeline, and gave the relaunched viewer SIGPIPE |

So **never kill by pattern**: list first with `ps`, read the pids, then kill those pids explicitly —
`PIDS=(<the pids>) ; kill -9 "${PIDS[@]}"`, an array because `kill $PIDS` is one argument in zsh.
`pgrep -P <pid>` is by *parent* pid and cannot self-match, so it is the safe way to take a job's
children with it. And put a verification in a **separate** ssh invocation from the kill.

**The reading half costs time rather than work.** A self-matching count is **1, not 0**, so it reads
as "still running" — which fails safe for a liveness check and fails *open* for a wait-loop: two
wait-loops never saw zero and spun for six hours. Never write a wait-loop whose condition greps for a
string its own command line contains. Two things follow: **bracket every pattern in the command, not
just one**, and **prefer `ps -Ao pid=,command= | grep <bracketed>`** over the `-f` forms.

**A hook now refuses both in command position**, because prose did not work — this rule was written
in 66 places across 13 files and was still repeated twice. It is
[`.claude/hooks/block_self_matching_pattern.py`](.claude/hooks/block_self_matching_pattern.py), wired
as a `PreToolUse` hook on Bash in `.claude/settings.json`, with its cases in
`test_block_self_matching_pattern.py` beside it. It denies only *invocations*: a commit message, a
`grep` through these docs, a `pgrep -x` on a process name, and a heredoc that writes a paragraph
about the trap all still run. **If it blocks something legitimate that is a bug in the regex** — fix
the hook and add the case, rather than routing around it. (Both exemptions above were written after
the hook blocked something real: this section's own edit was the second.)

**And bracketing does not save you when the pattern sits inside an enclosing `zsh -c` string.**
`[t]ools.shard` protects the `grep` process itself, but the tool runs commands through
`zsh -c '... grep '[t]ools.shard' ...'`, and *that* shell's command line contains the bracketed
pattern **verbatim** — brackets and all — so `ps -Ao command=` matches the wrapper and the check
reads "still running" forever. Measured 2026-08-28: a `[t]ools.shard` check printed two matches when
nothing was running, both of them its own wrapper. When the answer must be trustworthy, **match on
something the scanner cannot contain** — the interpreter path (`snek3/bin/python`) plus a `grep -v
'zsh -c'`, or a pid file the job writes.

**A wait-loop needs a sleep in every loop body, including its guard.** This one burned a full core
for 2 h 08 m:

    until [ ! -e /proc/self ] && false; do :; done   # never exits, never sleeps

On macOS `/proc/self` does not exist, so the test is true, `&& false` makes the condition false, and
`until` spins on `:` at 100% CPU without ever reaching the `sleep 20` loop below it. It was not the
*condition* that was wrong — it was that a loop existed with no sleep in it at all. **Prefer the
`Monitor` tool over a hand-written wait-loop**, and if you write one, put the `sleep` first.

The inverse is equally real: a `pgrep` that *errored* once read as "nothing is running" and closed a
live chart window with five hours left to run, because `pgrep` exits 0 on a match, 1 on no match and
≥2 on an error, and all three produce empty stdout. **A process scan can over-report and
under-report, so never treat its output as authority without checking the pattern against the
scanner itself.**

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
several worker processes at once.

Nothing here plays sound. Init only the subsystems you need, **by name** — `pygame.display` and
`pygame.font` — and set `SDL_AUDIODRIVER=dummy` before any pygame import. Verify with
`ps -o %cpu= -p $(pgrep -x coreaudiod)` while workers run; it should read 0.0.

In snek3 the vectorised and eval paths import no pygame at all, which removes the trap rather than
guarding against it. Only `env/` may import pygame.
