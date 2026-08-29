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
| limit | **4 trainers** | `max_trainers` ≤ 4, `max_evals` ≤ 4 |
| check | `ps -Ao pid=,command= \| grep '[t]rain.py'` | **`git fetch origin ops-status && git show origin/ops-status:status.json`** |
| queue work | launch by hand | commit a JSON spec to `queue/pending/` on the `ops` branch, then trigger |
| start it now | — | `ssh the-claw-den 'Snek/snek3/desktop/trigger'` |

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

**‡ A `pgrep` pattern matches the shell that runs it, and this is the default outcome, not an edge
case.** `pgrep -f <pat>` scans full command lines and the invoking shell's own command line contains
`<pat>` verbatim, so the check counts itself. It cost two immortal processes: wait-loops that never
saw zero and spun for six hours. Three things follow. **Bracket every pattern in the command, not
just one.** **Prefer `ps -Ao pid=,command= | grep <bracketed>`** over `pgrep -f`. And **a
self-matching count is 1, not 0**, so it reads as "still running" — which fails safe for a liveness
check and fails *open* for a wait-loop. Never write a wait-loop whose condition greps for a string
its own command line contains.

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
