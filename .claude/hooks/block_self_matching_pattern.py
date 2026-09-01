#!/usr/bin/env python3
"""PreToolUse(Bash): refuse `pgrep -f` and `pkill -f` in command position.

`-f` matches the FULL command line, and the invoking shell's own command line contains the pattern
verbatim, so the scan always includes itself. As a check it reads 1 instead of 0. As a kill it
SIGKILLs the shell running it — over ssh that is exit 255 with no output, which looks exactly like
the remote work having failed.

This repository has hit it at least three times (two six-hour wait-loops; 2026-08-25 abandoning
b46's first wave; 2026-09-01 relaunching a chart window) while documenting the rule in 66 places
across 13 files. Prose was not the missing piece, so this is a gate rather than a 67th sentence.

**Only command position is blocked, and that boundary is the whole design.** A command that merely
*mentions* the flag has to keep working: a commit message about this trap, a `grep` through the docs
for it, and the test file beside this one all contain the string and none of them is an invocation.
So command position is the start of the command, a shell separator, or a keyword that introduces a
command — not a bare word, and not a quote.

Except when the command hands a string to another shell. `ssh host 'pkill -f x'` and `bash -c
"pkill -f x"` are real invocations whose only prefix is a quote, and the ssh form is the one that
actually cost a wave. So when a nesting indicator is present, quotes count as command position too.
That is narrower than treating every quote that way, which blocked this file's own tests.

A heredoc body is data for the same reason, and is stripped before scanning — `cat > doc.md <<'EOF'`
writing a paragraph *about* this trap has to work, and documenting it means quoting the incident
verbatim, separator and all. The exception is a body fed to a shell (`bash <<'EOF'`), where the lines
really are commands. Both carve-outs were added after this hook blocked something real: its own tests
first, then the CLAUDE.md section describing it.

Run the tests after editing:  python3 .claude/hooks/test_block_self_matching_pattern.py
"""
import json
import re
import sys

# What may precede a real invocation.
SEPARATORS = r"""(?:^|[;&|(`\n]|\$\()"""
KEYWORDS = (r"(?:if|elif|then|else|fi|while|until|do|done|not|sudo|nohup|setsid|time|exec|ssh|"
            r"xargs|command)")
PREFIX = r"(?:%s|\b%s)[ \t]*" % (SEPARATORS, KEYWORDS)
# The same, plus a quote — used only when the command nests a shell (see the module docstring).
NESTED_PREFIX = r"(?:%s|['\"]|\b%s)[ \t]*" % (SEPARATORS, KEYWORDS)
# `pkill -f`, `pkill -9 -f`, `pgrep -af`, `pkill -fx` — a flag bundle *containing* `f` anywhere in
# it, which is why the `f` is not anchored to the end of the bundle: `-fx` is a real spelling.
INVOCATION = r"(?P<tool>pgrep|pkill)(?:[ \t]+-[A-Za-z0-9]+)*[ \t]+-[A-Za-z]*f[A-Za-z]*\b"

PATTERN = re.compile(PREFIX + INVOCATION)
NESTED_PATTERN = re.compile(NESTED_PREFIX + INVOCATION)
# `ssh` as a command word, or an explicit `-c` string handed to an interpreter.
NESTS_A_SHELL = re.compile(r"(?:^|[;&|(`\n]|\$\(|[ \t])ssh[ \t]|(?:ba|z|)sh[ \t]+-c[ \t]|[ \t]-c[ \t]")

# A heredoc body is DATA, not command position: `cat > doc.md <<'EOF' … EOF` writing a paragraph
# about this trap is not an invocation of it, and this file's own arrival was blocked by the earlier
# version that scanned one. The exception is a body fed to a shell — `bash <<'EOF'` — where every
# line of it really is a command.
HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
FEEDS_A_SHELL = re.compile(r"(?:^|[;&|(]|\$\()[ \t]*(?:sudo[ \t]+)?(?:ssh\b|(?:/\S+/)?(?:ba|z|k|)sh\b)")


def strip_heredocs(command):
    """`command` with every heredoc body removed, except a body handed to a shell."""
    out, position = [], 0
    while True:
        match = HEREDOC.search(command, position)
        if not match:
            out.append(command[position:])
            return ''.join(out)
        newline = command.find('\n', match.end())
        if newline < 0:                            # the body never started; nothing to strip
            out.append(command[position:])
            return ''.join(out)
        line_start = command.rfind('\n', 0, match.start()) + 1
        introducer = command[line_start:match.start()]
        delimiter, body_start = match.group(2), newline + 1
        end = re.compile(r"^[ \t]*%s[ \t]*$" % re.escape(delimiter), re.M).search(command, body_start)
        body_end = end.start() if end else len(command)
        keep = bool(FEEDS_A_SHELL.search(introducer))
        out.append(command[position:body_start if keep else newline + 1])
        if keep:
            out.append(command[body_start:body_end])
        position = body_end

REASON = """Blocked: `{tool} -f` scans FULL command lines, and this command's own shell contains the pattern verbatim — so the scan includes itself. As a check it reads 1 instead of 0; as a kill it SIGKILLs its own shell, which over ssh returns exit 255 with no output and looks like the work failed.

Bracketing does NOT save you: the enclosing `zsh -c` / `bash -c` string holds the bracketed pattern too, so `[t]ools.shard` matched its own wrapper and read "still running" when nothing was.

  LIST   ps -Ao pid=,etime=,command= | grep <bracketed> | grep -v 'zsh -c'
         more trustworthy: match something the scanner cannot contain, such as the
         interpreter path `envs/snek3/bin/python`, or a pid file the job writes.
  KILL   list first, read the pids, then kill those pids explicitly:
           PIDS=(<the pids you just read>) ; kill -9 "${{PIDS[@]}}"
         (`kill $PIDS` is ONE argument in zsh — use the array)
         one job's children: kill -9 $(pgrep -P <pid>) <pid>    # -P is by parent pid, safe
  WAIT   use the Monitor tool, or Bash run_in_background — not a hand-written loop.

CLAUDE.md: "A `pgrep` pattern matches the shell that runs it, and this is the default outcome".

Refused: {cmd}"""


def decide(command):
    """The reason to deny, or None to allow. Pure, so it is testable without the harness."""
    scanned = strip_heredocs(command)
    pattern = NESTED_PATTERN if NESTS_A_SHELL.search(scanned) else PATTERN
    match = pattern.search(scanned)
    if not match:
        return None
    return REASON.format(tool=match.group('tool'), cmd=command)


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0                                  # not our business; never block on a bad payload
    command = (payload.get('tool_input') or {}).get('command') or ''
    reason = decide(command)
    if reason is None:
        return 0
    json.dump({'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': reason,
    }}, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
