"""The mutation harness itself, because two ad-hoc versions damaged this working tree.

The failures it exists to prevent are all *silent*: a pattern that does not match and falls through
to restore the wrong file, a stale `.pyc` that outlives its own restore, and a killed run that leaves
a mutation behind. None of them raises; each makes the *next* result wrong instead.
"""

import json
import os
import signal

import pytest

from tools import mutate


@pytest.fixture(autouse=True)
def keep_the_signal_handlers():
    """`run` installs process-wide handlers, so they are put back before the next test.

    Without this a later test's interrupt would hit this module's handler.
    """
    saved = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)


def write_spec(tmp_path, mutations, tests=('tests/test_mutate.py',)):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'tests': list(tests), 'mutations': mutations}))
    return str(spec)


# --- the spec ---------------------------------------------------------------------------------

def test_a_spec_with_no_tests_is_refused(tmp_path):
    with pytest.raises(mutate.SpecError, match='names no tests'):
        mutate.load_spec(write_spec(tmp_path, [{'desc': 'x'}], tests=()))


def test_a_spec_with_no_mutations_is_refused(tmp_path):
    with pytest.raises(mutate.SpecError, match='names no mutations'):
        mutate.load_spec(write_spec(tmp_path, []))


def test_every_pattern_is_checked_before_any_file_is_touched(tmp_path, monkeypatch):
    """The failure that replaced one module's source with another's.

    A harness that validated each pattern as it went wrote no backup when the check failed, then
    restored the *previous* mutation's backup over the wrong file. So validation is all-or-nothing.
    """
    good = tmp_path / 'good.py'
    good.write_text('KEEP = 1\n')
    bad = tmp_path / 'bad.py'
    bad.write_text('OTHER = 2\n')
    spec = write_spec(tmp_path, [
        {'desc': 'fine', 'file': str(good), 'from': 'KEEP = 1', 'to': 'KEEP = 9'},
        {'desc': 'no such text', 'file': str(bad), 'from': 'ABSENT', 'to': 'X'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    monkeypatch.setattr(mutate, 'run_tests', lambda tests, quiet=True, timeout=None: 'passed')
    with pytest.raises(mutate.SpecError, match='occurs 0 times'):
        mutate.run(spec)
    assert good.read_text() == 'KEEP = 1\n', 'a file was modified despite an unusable spec'


def test_a_pattern_that_matches_twice_is_refused(tmp_path):
    # Ambiguous: `str.replace` would change both, so the mutation is not the one described.
    target = tmp_path / 'twice.py'
    target.write_text('A = 1\nA = 1\n')
    with pytest.raises(mutate.SpecError, match='occurs 2 times'):
        mutate.validate([{'desc': 'd', 'file': str(target), 'from': 'A = 1', 'to': 'A = 2'}])


# --- the run ----------------------------------------------------------------------------------

def test_a_red_baseline_stops_the_run(tmp_path, monkeypatch):
    # Every mutation would read as killed, which is the most misleading outcome there is: it looks
    # exactly like complete coverage.
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    monkeypatch.setattr(mutate, 'run_tests', lambda tests, quiet=True, timeout=None: 'failed')
    with pytest.raises(mutate.SpecError, match='fail before any mutation'):
        mutate.run(spec)


def test_a_completed_run_leaves_the_file_as_it_found_it(tmp_path, monkeypatch):
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    seen = []

    def run_tests(tests, quiet=True, timeout=None):
        seen.append(target.read_text())
        return 'passed' if len(seen) == 1 else 'failed'

    monkeypatch.setattr(mutate, 'run_tests', run_tests)
    assert mutate.run(spec) == 0, 'a killed mutation is a pass'
    assert seen == ['VALUE = 3\n', 'VALUE = 1\n'], 'the mutation never reached the file'
    assert target.read_text() == 'VALUE = 3\n'


def test_a_surviving_mutation_is_a_non_zero_exit(tmp_path, monkeypatch):
    # So a CI step or a shell `&&` chain notices, rather than the survivor scrolling past.
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    monkeypatch.setattr(mutate, 'run_tests', lambda tests, quiet=True, timeout=None: 'passed')
    assert mutate.run(spec) == 1


def test_bytecode_writing_is_off_for_every_test_run(monkeypatch):
    """The `.pyc` trap, pinned at its source.

    Bytecode is revalidated on the source's mtime *and size*, and a one-character mutation like `3`
    to `1` changes neither in a way that invalidates it — so a stale `.pyc` made a mutation outlive
    its own restore and the *next* measurement was wrong with no indication.
    """
    captured = {}

    class Result(object):
        returncode = 0
        stdout = b''

    def fake_run(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
        captured.update(env)
        return Result()

    monkeypatch.setattr(mutate.subprocess, 'run', fake_run)
    assert mutate.run_tests(['tests/test_mutate.py']) == 'passed'
    assert captured['PYTHONDONTWRITEBYTECODE'] == '1'


def test_a_signal_mid_run_still_restores_the_tree(tmp_path, monkeypatch):
    """A killed harness must not leave a mutation behind.

    `finally` unwinds on an exception, not on a signal, so without a handler a `kill` or a command
    timeout leaves the file mutated — and the tell is a suite that still passes, because the
    surviving mutation is by definition one nothing covers. This happened: a 2-minute command
    timeout during a 31-mutation pass left `train.py` holding mutation 5.
    """
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    calls = []

    def killed(tests, quiet=True, timeout=None):
        calls.append(tests)
        if len(calls) == 1:
            return 'passed'                    # a green baseline
        os.kill(os.getpid(), signal.SIGTERM)   # ...then die with the mutation applied

    monkeypatch.setattr(mutate, 'run_tests', killed)
    with pytest.raises(KeyboardInterrupt):
        mutate.run(spec)
    assert target.read_text() == 'VALUE = 3\n', 'the mutation outlived its own harness'


def test_a_mutant_that_hangs_counts_as_killed(tmp_path, monkeypatch):
    """An unbounded loop is the mutation being detected, but only if the harness has a clock.

    Without one the harness inherits the hang and the enclosing command timeout kills *it* — which
    is how a mutation outlived its own restore twice in one session. Reported as hung rather than
    merely failed, because the cause is almost always a loop whose exit condition was mutated away.
    """
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    calls = []

    def hangs(tests, quiet=True, timeout=None):
        calls.append(timeout)
        return 'passed' if len(calls) == 1 else 'timeout'

    monkeypatch.setattr(mutate, 'run_tests', hangs)
    assert mutate.run(spec) == 0, 'a hung mutant is killed, so the run is a pass'
    assert calls[0] is None, 'the baseline is what the timeout is measured from, so it has none'
    assert calls[1] >= mutate.MIN_TIMEOUT


def test_a_baseline_that_times_out_stops_the_run(tmp_path, monkeypatch):
    # Otherwise every mutation times out too and reads as perfect coverage.
    target = tmp_path / 'subject.py'
    target.write_text('VALUE = 3\n')
    spec = write_spec(tmp_path, [{'desc': 'd', 'file': str(target),
                                  'from': 'VALUE = 3', 'to': 'VALUE = 1'}])
    monkeypatch.setattr(mutate, 'ROOT', '')
    monkeypatch.setattr(mutate, 'run_tests', lambda tests, quiet=True, timeout=None: 'timeout')
    with pytest.raises(mutate.SpecError, match='time out'):
        mutate.run(spec)


def test_a_real_infinite_loop_is_caught_rather_than_inherited(tmp_path, monkeypatch):
    """End to end through the actual subprocess, because the timeout has to reach `subprocess.run`.

    The stubs above pin the harness's bookkeeping; this pins that the clock is really wired up.
    """
    target = tmp_path / 'test_spins.py'
    target.write_text('def test_it():\n    while False:\n        pass\n')
    spec = write_spec(tmp_path, [{'desc': 'the loop never exits', 'file': str(target),
                                  'from': 'while False:', 'to': 'while True:'}],
                      tests=[str(target)])
    monkeypatch.setattr(mutate, 'ROOT', str(tmp_path))
    monkeypatch.setattr(mutate, 'MIN_TIMEOUT', 3.0)
    assert mutate.run(spec) == 0
    assert target.read_text().count('while False:') == 1, 'the hung mutation was not restored'
