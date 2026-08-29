"""Which checkpoints a stage-B wave measures.

Three selectors, and the list is closed on purpose. snek2 had `full`, `screen`, `flat`, `confirm`,
`top50`, `ALWAYS_EVAL_SINGLE`, `ALWAYS_FULL_SINGLE` and a min-achievable gate, and the cost was not
the code: it was that **two rows from the same file were not necessarily comparable**, so reading one
began with working out which tier and which gate era produced it. snek3's rows are all 500 episodes,
so a selector chooses *which* checkpoints and never *how deeply*.

| selector | means | reads |
|---|---|---|
| `screen:<n>` | every checkpoint whose stage-A eval was ≥ n perfect. **The protocol's default**, n=95 | `runs/<name>_evals.json` |
| `above:<n>[:<label>]` | every checkpoint above n in a *prior stage-B* pass — the record re-measure | `runs/<name>_checkpoint_evals[_<label>].json` |
| `steps:<path>` | an explicit list, one step per line | that file |
| `all` | every checkpoint present | the policy directory |

`screen` is the one that matters and `all` is the one to be careful with: a 3M-step arm has ~3,000
checkpoints, so `all` at 500 episodes is 1.5M episodes.

**Named `step_selectors`, not `selectors`, and that is not fussiness.** A module called `selectors.py`
shadows the standard library's, and the standard library's is imported by `subprocess` — so running
any script from inside `tools/` (which puts `tools/` at the head of `sys.path`) made `import
subprocess` load this file instead, which imports torch, which imports `multiprocessing`, which
imports `subprocess` again. The error names a circular import in `subprocess` and points nowhere
near here. `tests/test_module_layering.py` guards the whole tree against the same trap.
"""

import os

from tools import checkpoints
from tools import results

DEFAULT_SCREEN = 95


class SelectorError(Exception):
    pass


def parse(token):
    """`(kind, value)` from a selector string. `value`'s type depends on the kind."""
    if token in (None, '', 'all'):
        return 'all', None
    kind, _, rest = token.partition(':')
    if kind == 'screen':
        return 'screen', float(rest) if rest else float(DEFAULT_SCREEN)
    if kind == 'above':
        threshold, _, label = rest.partition(':')
        if not threshold:
            raise SelectorError('above: needs a threshold, e.g. above:98')
        return 'above', (float(threshold), label or None)
    if kind == 'steps':
        if not rest:
            raise SelectorError('steps: needs a path, e.g. steps:runs/ab.txt')
        return 'steps', rest
    raise SelectorError(
        'unknown selector {0!r}. Known: screen:<n>, above:<n>[:<label>], steps:<path>, all'.format(
            token))


def _read_steps_file(path):
    """Steps from a file, one per line. Blank lines and `#` comments are skipped.

    A plain list rather than JSON, because the thing that produces one is usually a shell pipeline
    over another result file, and because a human writes them by hand.
    """
    if not os.path.exists(path):
        raise SelectorError('no step list at {0}'.format(path))
    steps = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            text = line.split('#', 1)[0].strip()
            if not text:
                continue
            try:
                steps.append(int(text))
            except ValueError:
                raise SelectorError('{0}:{1}: {2!r} is not a step'.format(path, number, text))
    if not steps:
        raise SelectorError('{0} lists no steps'.format(path))
    return steps


def _screened(policy, threshold):
    payload = results.read(results.stage_a_path(policy))
    if payload is None:
        raise SelectorError(
            'no stage-A file at {0}. `screen:` reads the trainer\'s own evals, so a policy that '
            'was never trained here — an imported checkpoint, say — has to be selected with '
            '`steps:` or `all`'.format(results.stage_a_path(policy)))
    return [int(eval_['step']) for eval_ in payload.get('evals', ())
            if eval_.get('perfect_percent') is not None
            and float(eval_['perfect_percent']) >= threshold]


def _above(policy, threshold, label):
    path = results.stage_b_path(policy, label)
    payload = results.read(path)
    if payload is None:
        raise SelectorError('no stage-B file at {0} to select above'.format(path))
    return [int(row['step']) for row in results.rows_of(payload)
            if float(row['perfect_percent']) >= threshold]


def resolve(policy_dir, token, policy=None):
    """`(steps, description)` — the checkpoints to measure, ascending, and a line naming the choice.

    **Every selected step is checked against what is on disk, and a miss is an error.** A selector
    that silently drops a step it cannot find turns a dispatch bug into a short result file, which is
    indistinguishable from a checkpoint that was never good enough to select.
    """
    policy = policy if policy is not None else policy_dir
    kind, value = parse(token)
    available = checkpoints.steps(policy_dir)
    if not available:
        raise SelectorError('no checkpoints in {0}'.format(policy_dir))

    if kind == 'all':
        wanted, description = list(available), 'every checkpoint present'
    elif kind == 'screen':
        wanted = _screened(policy, value)
        description = 'stage-A perfect >= {0:g}'.format(value)
    elif kind == 'above':
        threshold, label = value
        wanted = _above(policy, threshold, label)
        description = 'stage-B perfect >= {0:g} in {1}'.format(threshold, label or 'the main pass')
    else:
        wanted = _read_steps_file(value)
        description = 'the {0} steps listed in {1}'.format(len(wanted), value)

    present = set(available)
    missing = sorted(step for step in set(wanted) if step not in present)
    if missing:
        raise SelectorError(
            '{0} selected step(s) have no checkpoint in {1}: {2}{3}'.format(
                len(missing), policy_dir, missing[:8], ' ...' if len(missing) > 8 else ''))
    return sorted(set(wanted)), description


def slice_for(steps, shard, shards):
    """Shard `shard` of `shards`, interleaved rather than blocked.

    **Interleaved because cost is not uniform along an arm.** A strong checkpoint plays a ~1,800-step
    perfect game and a weak one dies in 40, so contiguous blocks hand the shard covering the trained
    end of the arm several times the work of the shard covering the start — the wave then finishes
    when its slowest shard does, at maybe 40% utilisation. Striding mixes early and late checkpoints
    into every shard, which is the cheapest possible balance and needs no scheduler.
    """
    if not 0 <= shard < shards:
        raise SelectorError('shard {0} is not in 0..{1}'.format(shard, shards - 1))
    return steps[shard::shards]
