"""Mutation testing: break the implementation on purpose and check that a test notices.

[`../CLAUDE.md`](../CLAUDE.md) asks for this on every change, because a passing suite is not coverage
of the change you just made — snek2 took a third signature for its observation grouper and all 24
tests passed before and after. This module exists because the obvious shell version of the same idea
is dangerous, and two different ad-hoc harnesses damaged this working tree before it was written:

- **A `.pyc` outlives its own restore.** Bytecode is revalidated on the source's *mtime and size*,
  and `cp` / `mv` of a backup restores an older mtime while a one-character mutation like `3` to `1`
  has the same size. The mutated bytecode is then reused and the *next* result is silently wrong.
  Here every run sets `PYTHONDONTWRITEBYTECODE=1`, so no `.pyc` is written at all.
- **A pattern that does not match must abort, not fall through.** A harness that wrote its backup
  after checking the pattern left no backup when the check failed, then restored the previous
  mutation's backup over the wrong file — replacing one module's source with another's. Here **every
  pattern is validated against every file before anything is modified**, the originals are held in
  memory, and the restore is in a `finally`.

    PYTHONPATH=. python -m tools.mutate mutations.json

The spec is JSON: a list of tests to run and a list of mutations to try.

    {"tests": ["tests/test_replay.py"],
     "mutations": [{"desc": "priorities ignore alpha",
                    "file": "dqn/replay.py",
                    "from": "priorities ** self.alpha",
                    "to": "priorities"}]}

A mutation is **KILLED** when the tests fail with it applied — which is the outcome you want — and
**SURVIVED** when they still pass, meaning nothing tests that line. A survivor is either a missing
fixture or dead code, and both are worth knowing.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpecError(Exception):
    """The spec is unusable. Raised before any file is touched."""


def load_spec(path):
    with open(path) as handle:
        spec = json.load(handle)
    tests = list(spec.get('tests') or [])
    mutations = list(spec.get('mutations') or [])
    if not tests:
        raise SpecError('the spec names no tests to run')
    if not mutations:
        raise SpecError('the spec names no mutations')
    return tests, mutations


def validate(mutations):
    """Checks every pattern against its file and returns the originals, keyed by path.

    **All of them, before any file is written.** A typo in the twentieth mutation must not be found
    after nineteen have already been applied and restored.
    """
    originals = {}
    problems = []
    for index, mutation in enumerate(mutations):
        for field in ('desc', 'file', 'from', 'to'):
            if field not in mutation:
                problems.append('mutation {0} has no {1!r}'.format(index, field))
        if problems:
            continue
        path = os.path.join(ROOT, mutation['file'])
        if not os.path.exists(path):
            problems.append('{0}: no such file'.format(mutation['file']))
            continue
        if path not in originals:
            with open(path) as handle:
                originals[path] = handle.read()
        found = originals[path].count(mutation['from'])
        if found != 1:
            problems.append('{0}: {1!r} occurs {2} times, need exactly 1 — {3}'.format(
                mutation['file'], mutation['from'][:60], found, mutation['desc']))
        if mutation['from'] == mutation['to']:
            problems.append('{0}: from and to are identical, so nothing is mutated'.format(
                mutation['desc']))
    if problems:
        raise SpecError('\n'.join(problems))
    return originals


# A mutant's tests get this multiple of the baseline's own wall clock before they are called hung.
# Generous, because a mutation legitimately changes how much work the tests do.
TIMEOUT_FACTOR = 6.0
MIN_TIMEOUT = 30.0


def run_tests(tests, quiet=True, timeout=None):
    """`'passed'`, `'failed'` or `'timeout'`. No bytecode is written — see the module docstring.

    **A hang is a distinct outcome, not a variety of failure, and it needs its own clock.** A
    mutation can turn a bounded loop into an unbounded one: dropping the decrement from the training
    loop's gradient-budget accumulator makes `while debt >= 1.0: ... continue` spin forever. Without
    a timeout the harness inherits that hang, and the enclosing command timeout then kills the
    harness itself — which is how a mutation came to outlive its own restore twice in one session.
    """
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'pytest', '-q', '-x', '--no-header', '-p', 'no:cacheprovider']
    command.extend(tests)
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, timeout=timeout,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return 'timeout'
    if not quiet:
        sys.stdout.write(result.stdout.decode('utf-8', 'replace'))
    return 'passed' if result.returncode == 0 else 'failed'


def apply_one(path, original, mutation):
    with open(path, 'w') as handle:
        handle.write(original.replace(mutation['from'], mutation['to']))


def restore(originals):
    for path, text in originals.items():
        with open(path, 'w') as handle:
            handle.write(text)


def restore_on_signal():
    """Turns SIGTERM and SIGINT into an exception, so the `finally` restore actually runs.

    **Without this a killed run leaves the tree mutated.** A `finally` block unwinds on an
    exception, not on a signal — Python installs no SIGTERM handler, so the default action
    terminates the process outright and the mutation in flight survives its own harness. That
    happened: a 2-minute command timeout during a 31-mutation pass left `train.py` holding mutation
    5, and the only clue was a test suite that still passed, because the mutation was one nothing
    covered yet.

    Raising `KeyboardInterrupt` rather than exiting, because it is the one exception `finally`
    handles identically to any other and pytest already treats as an interrupt.
    """
    def handler(number, frame):
        raise KeyboardInterrupt('signal {0} during a mutation run'.format(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, handler)


def run(spec_path, verbose=False):
    tests, mutations = load_spec(spec_path)
    originals = validate(mutations)
    restore_on_signal()

    started = time.time()
    baseline = run_tests(tests, quiet=not verbose)
    if baseline != 'passed':
        # A red baseline makes every mutation look killed, which is the most misleading possible
        # outcome — it reads as complete coverage.
        raise SpecError('the tests {0} before any mutation is applied; fix that first'.format(
            'time out' if baseline == 'timeout' else 'fail'))
    # Derived from this suite's own measured wall clock rather than fixed, so the same factor works
    # for a 0.2 s unit-test file and a 20 s one.
    timeout = max(MIN_TIMEOUT, TIMEOUT_FACTOR * (time.time() - started))

    survivors = []
    try:
        for mutation in mutations:
            path = os.path.join(ROOT, mutation['file'])
            apply_one(path, originals[path], mutation)
            outcome = run_tests(tests, quiet=not verbose, timeout=timeout)
            restore({path: originals[path]})
            # A hang counts as killed: the mutation changed observable behaviour and the tests
            # noticed, which is all "killed" claims. Reported separately because a hung mutant is
            # worth reading — it usually means an unbounded loop rather than a wrong value.
            label = {'passed': 'SURVIVED', 'failed': 'KILLED  ', 'timeout': 'KILLED  '}[outcome]
            print('{0}  {1}{2}'.format(
                label, mutation['desc'], ' (hung)' if outcome == 'timeout' else ''), flush=True)
            if outcome == 'passed':
                survivors.append(mutation['desc'])
    finally:
        restore(originals)

    print('\n{0} mutation(s), {1} killed, {2} survived'.format(
        len(mutations), len(mutations) - len(survivors), len(survivors)))
    for desc in survivors:
        print('  survived: {0}'.format(desc))
    return 0 if not survivors else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('spec', help='path to the JSON spec')
    parser.add_argument('--verbose', action='store_true', help='show pytest output')
    args = parser.parse_args(argv)
    try:
        return run(args.spec, args.verbose)
    except SpecError as error:
        print('mutation spec is unusable:\n{0}'.format(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
