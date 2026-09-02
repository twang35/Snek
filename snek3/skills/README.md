# snek3 skills

Procedures an agent runs often, moved out of the docs so they cost nothing until they are used. The
docs say *what is true*; a skill says *what to do, in order*.

| skill | use it for |
|---|---|
| [`progress-update`](progress-update/SKILL.md) | what is running on both boxes, then bring the docs up to date. Read-only with respect to processes |
| [`laptop-run`](laptop-run/SKILL.md) | start a training, a batch or a stage-B close-out here |
| [`desktop-batch`](desktop-batch/SKILL.md) | queue work on `the-claw-den`, and pause/retune it |
| [`stop-run`](stop-run/SKILL.md) | kill an arm or a wave on either box, and clean up its children |
| [`desktop-deploy`](desktop-deploy/SKILL.md) | get the box on new code |
| [`mutation-test`](mutation-test/SKILL.md) | prove the tests cover a change |
| [`hof-remeasure`](hof-remeasure/SKILL.md) | re-measure a batch's stage-B winners at 5,000 episodes — the `hof5000` pass |
| [`hof-promote`](hof-promote/SKILL.md) | put a checkpoint in `hallOfFame/`, on a confirmed fresh measurement |

## Which model runs a skill

`desktop-deploy` and `progress-update` carry `model: sonnet` in their frontmatter (2026-09-02): both are
procedures with the judgement already written down, so the cheaper model runs them for the turn and
the session's own model resumes afterwards. Add the field to any skill that is a checklist rather than
an investigation; leave it off the ones that weigh evidence (`hof-promote`, `hof-remeasure`).

## How they are discovered

The files live here. `/.claude/skills/<name>` at the repository root is a **symlink** to each one, and
that is what Claude Code reads:

```
ln -s ../../snek3/skills/<name> .claude/skills/<name>
```

**A new skill needs its symlink**, or nothing will find it. Claude Code has no settings key for extra
skill directories, and it does follow a symlinked skill directory (the alternative,
`claude --add-dir snek3`, only helps if the flag is passed every session).

The listing is rescanned during a session, not only at startup, so a skill added now becomes
invocable a short while later — an `Unknown skill` straight after creating one means "not yet", not
"wrong symlink". Verify the symlink resolves (`head -4 .claude/skills/<name>/SKILL.md`) rather than
retrying the invocation.

## Writing one

- **Frontmatter is `name` and `description` only.** The description is the whole basis for picking the
  skill, so write it as the phrases a user actually types — "kill that arm", "queue this on the
  desktop" — not as a category.
- **Keep it short. Every line is paid on every invocation.** State the commands and the traps that
  break a run. Send background, measurements and history to the docs by link. Much over ~100 lines and
  it is doing a doc's job.
- **A trap earns its place by having happened.** These skills carry the incident behind each rule
  because that is what stops a later session from "simplifying" it away. A rule with no incident
  behind it belongs in the docs.
- **Do not duplicate a doc.** Link `docs/running.md` for the knob list, `docs/protocol.md` for how a
  run is judged, `desktop/README.md` for the daemon's design.

## When a skill breaks

**Fix the skill, not just the immediate problem.** If a skill's steps fail, are ambiguous, or leave
something behind:

1. Work out what actually happened — do not retry the same command and hope.
2. Fix the situation.
3. **Edit the skill** so the next invocation does not hit it: correct the command, add the check that
   would have caught it, or name the trap.
4. Say in your reply that you changed the skill and why.

A skill that failed once and was not updated will fail the same way again, and the next session will
not have the context this one did. Skills are Markdown, so the edit commits and pushes with the
docs-only standing authorization — unless the same change also touches code, which sends the whole
thing back to waiting for approval.
