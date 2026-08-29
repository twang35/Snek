"""Fixtures for the two properties the `eval_plan.py` extraction exists to create.

Neither is about what the moved functions compute — `test_eval_checkpoints.py` and
`test_selection_tiers.py` already pin that, and they were not touched by the move. These pin the
*structural* facts that a future edit would otherwise break silently:

  * `eval_plan` imports no TensorFlow. That is the whole point — `eval_wave.py` plans a wave and
    writes the files without paying for a ~230 MB arena, and one stray `import tensorflow` (or an
    import of a module that pulls it, like `eval_agent` or `snake_environment`) undoes it with no
    other symptom than a slower, fatter controller.
  * `eval_checkpoints` re-exports every public name. That is the compatibility surface for the 90
    fixtures in `test_eval_checkpoints.py`, which reach `eval_checkpoints.build_row` and friends.
    A name dropped from the re-export fails there as an `AttributeError`, which reads like noise;
    this fails as a named assertion instead.
"""
import ast
import os
import subprocess
import sys

import eval_plan

SNEK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _defined_names(module):
    """Every name `module` *defines* at top level, read from its source rather than its namespace.

    Parsing is deliberate. The obvious version filters `dir(module)` on `__module__`, and it is
    wrong in a way that is easy to miss: a plain value imported from elsewhere — `EVALS_ARCHIVE_DIR`
    here, a `str` from `snake_constants` — has `__module__` of None, indistinguishable from a value
    defined locally, so it slips into the set and the re-export fixture then demands that
    `eval_checkpoints` re-export somebody else's constant. The first draft of this file did exactly
    that and failed on it. Top-level `def` and assignment targets are the precise question being
    asked, so ask it precisely.
    """
    tree = ast.parse(open(module.__file__).read())
    names = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            names[node.name] = getattr(module, node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith('_'):
                    names[target.id] = getattr(module, target.id)
    return names


def test_eval_plan_imports_without_tensorflow():
    # A subprocess, not this interpreter: the rest of the suite imports TensorFlow, so an in-process
    # check would pass no matter what eval_plan does.
    code = ('import sys; import eval_plan; '
            'bad = [m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow.") '
            'or m == "tf_agents" or m.startswith("tf_agents.")]; '
            'print(len(bad)); print(sorted(bad)[:5])')
    out = subprocess.run([sys.executable, '-c', code], cwd=SNEK_DIR, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=SNEK_DIR))
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().split('\n')
    assert lines[0] == '0', (
        'eval_plan pulled TensorFlow in: {0}. That is the one property this module exists for — '
        'see its docstring.'.format(lines[1] if len(lines) > 1 else '?'))


def test_eval_plan_defines_the_whole_protocol_surface():
    # Not an exhaustive list-of-29 (that would just restate the import block); the point is that the
    # four groups a caller needs are all here, so nothing had to stay behind in the TF-bound module.
    for name in ('select_top_checkpoints', 'select_checkpoints_above'):          # selection
        assert callable(getattr(eval_plan, name)), name
    for name in ('plan_stages', 'pick_finalists', 'skips_screening'):            # the stage plan
        assert callable(getattr(eval_plan, name)), name
    for name in ('make_abandon_test', 'achievable_percent'):                     # the gate
        assert callable(getattr(eval_plan, name)), name
    for name in ('build_row', 'held_from_row', 'load_finished_results'):         # the record
        assert callable(getattr(eval_plan, name)), name


def test_eval_checkpoints_reexports_every_eval_plan_name_by_identity():
    import eval_checkpoints
    missing, rebound = [], []
    for name, value in _defined_names(eval_plan).items():
        if not hasattr(eval_checkpoints, name):
            missing.append(name)
        elif getattr(eval_checkpoints, name) is not value:
            rebound.append(name)
    assert not missing, 'not re-exported by eval_checkpoints: {0}'.format(sorted(missing))
    # Identity, not equality: a re-defined copy would satisfy `==` for the float thresholds while
    # silently diverging from eval_plan's value the next time one of them is retuned.
    assert not rebound, 'rebound to a different object: {0}'.format(sorted(rebound))


def test_the_gate_ordering_invariant_lives_in_one_place():
    # CLOSEOUT gate must stay strictly below the HOF selection gate: HOF reads `above:98` out of the
    # close-out's own file, and only rows reaching the close-out gate are measured full length, so a
    # close-out gate at or above 98 abandons exactly the rows the re-measure needs and starves it.
    # Pinned here as well as in test_selection_tiers.py because eval_plan is now the single
    # definition both hosts import.
    assert eval_plan.DEFAULT_MIN_ACHIEVABLE < eval_plan.DEFAULT_ABOVE_THRESHOLD
    assert eval_plan.DEFAULT_MIN_ACHIEVABLE == 97.0 and eval_plan.DEFAULT_ABOVE_THRESHOLD == 98.0
