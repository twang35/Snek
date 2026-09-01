#!/usr/bin/env python3
"""Tests for the `pgrep -f` / `pkill -f` gate.

    python3 .claude/hooks/test_block_self_matching_pattern.py

**This file exists as a file rather than as a heredoc for a load-bearing reason.** Its cases contain
the very strings the hook blocks, so a Bash command carrying them inline is itself refused — the
first attempt at these tests was blocked by the hook under test. Running a file keeps the Bash
command clean (`python3 <path>`) and the patterns on disk where nothing scans them.

Every BLOCK case below is a command shape that has actually been written in this repository, or the
obvious neighbour of one. Every ALLOW case is a command that must keep working, and the last three
are the ones that made the first version of this hook unusable.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BLOCK = [
    # 2026-09-01, relaunching the desktop chart window: killed its own pipeline.
    'pkill -f "tail -3"',
    # 2026-08-25, abandoning b46's first wave: exit 255, no output, read as a failed kill.
    'ssh the-claw-den \'kill -9 1; pkill -9 -f "snek2.py b46"; ps -Ao args=\'',
    # The wait-loops that spun for six hours: the condition greps a string its own line contains.
    'until pgrep -f train.py; do sleep 5; done',
    'while pgrep -f shard >/dev/null; do sleep 20; done',
    'if pgrep -f train.py; then echo up; fi',
    # Plain forms, and the flag-bundle spellings.
    'pgrep -f train.py',
    'pgrep -af chart_viewer',
    'pkill -fx exactname',
    'pkill -9 -f viewer',
    'sudo pkill -f daemon',
    'echo hi && pkill -f viewer',
    'x=$(pgrep -f train.py)',
    # A quote is the only prefix, but a shell is being nested — the form that cost a wave.
    "ssh host 'pkill -f foo'",
    'bash -c "pkill -f foo"',
    # A heredoc body IS scanned when it is fed to a shell: every line of it is a command.
    "bash <<'EOF'\npkill -f viewer\nEOF",
    'ssh the-claw-den bash <<EOF\npkill -9 -f "snek2.py b46"\nEOF',
]

ALLOW = [
    # The safe replacements the deny message recommends.
    'ps -Ao pid=,etime=,command= | grep "[t]rain.py" | grep -v \'zsh -c\'',
    'ps -Ao pid=,command= | grep envs/snek3/bin/python',
    # `-P` is by parent pid: it cannot match the scanner, and the docs recommend it.
    'pgrep -P 1234',
    'kill -9 $(pgrep -P 555) 555',
    # No `-f`, so the match is on the process NAME and a shell called bash/zsh cannot collide.
    'pkill chart_viewer',
    'pgrep -x coreaudiod',
    'ps -o %cpu= -p $(pgrep -x coreaudiod)',
    # Mentions, not invocations. These three are why the hook checks command position at all.
    'git commit -m "a note about pkill -f in the docs"',
    'grep -rn "the pgrep -f trap" docs/',
    'echo "never use pkill -f for this"',
    # 2026-09-01: writing THIS RULE into CLAUDE.md was blocked by the version before heredoc
    # stripping. A heredoc body is data — and documenting the trap means quoting it verbatim,
    # separator and all. The second case is the exact prose that fired.
    "cat > doc.md <<'EOF'\nNever run `pkill -f x`; list with ps first.\nEOF",
    ("cat > new_rule.md <<'MDEOF'\n| 2026-08-25 | "
     "`ssh the-claw-den 'kill -9 1; pkill -9 -f \"snek2.py b46\"; ps …'` | exit 255 |\nMDEOF"),
    # Python (or anything that is not a shell) reading a heredoc: still data.
    'python3 - <<PYEOF\nprint("pgrep -f is the trap")\nPYEOF',
]


def load():
    path = os.path.join(HERE, 'block_self_matching_pattern.py')
    spec = importlib.util.spec_from_file_location('hook', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    hook = load()
    failures = []
    for command, want in [(c, True) for c in BLOCK] + [(c, False) for c in ALLOW]:
        reason = hook.decide(command)
        got = reason is not None
        if got != want:
            failures.append((command, want, got))
        print('%s  %-5s %s' % ('ok  ' if got == want else 'FAIL',
                               'BLOCK' if got else 'allow', command))
        if got and 'Refused:' not in reason:
            failures.append((command, 'a reason naming the command', reason))

    print('\n%d block case(s), %d allow case(s)' % (len(BLOCK), len(ALLOW)))
    if failures:
        print('\n%d FAILURE(S):' % len(failures))
        for command, want, got in failures:
            print('  %r  wanted %s, got %s' % (command, want, got))
        return 1
    print('all pass')
    return 0


if __name__ == '__main__':
    sys.exit(main())
