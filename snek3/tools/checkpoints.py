"""Reading and writing `savedPolicies/<policy>/ckpt-<step>.pt`, and finding what is there.

One file per checkpoint, named by agent step, holding a `state_dict` and the step that produced it.
Torch makes this small enough not to need a format: a `30 -> 320 -> 3` net is ~45 KB, so a 3M-step
arm checkpointing every 1,000 steps is ~135 MB — which is why `savedPolicies/` is gitignored while
`runs/` and `hallOfFame/` are not.

**The step is in the filename *and* in the payload, and they are checked against each other.** They
come apart in exactly one way that matters: a checkpoint copied into `hallOfFame/` under a new name.
Reading the step from the name alone is what let snek2 report a hall-of-fame measurement against the
wrong step.

`arch.json` is not written here — it belongs to the *directory*, not to each checkpoint, and
`tools/arch.py` owns it.
"""

import glob
import os
import re

import torch

PREFIX = 'ckpt-'
SUFFIX = '.pt'
_NAME = re.compile(r'^' + PREFIX + r'(\d+)' + re.escape(SUFFIX) + r'$')


class CheckpointError(Exception):
    pass


def path(policy_dir, step):
    return os.path.join(policy_dir, '{0}{1}{2}'.format(PREFIX, int(step), SUFFIX))


def step_of(checkpoint_path):
    """The step in a checkpoint's *filename*, or None if the name is not one of ours."""
    match = _NAME.match(os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else None


def steps(policy_dir):
    """Every step present, ascending. Empty for a directory with no checkpoints."""
    found = [step_of(p) for p in glob.glob(os.path.join(policy_dir, PREFIX + '*' + SUFFIX))]
    return sorted(step for step in found if step is not None)


def latest_step(policy_dir):
    present = steps(policy_dir)
    return present[-1] if present else None


def save(policy_dir, step, net, extra=None):
    """Writes `ckpt-<step>.pt`, atomically.

    Atomically because the trainer writes these while an eval wave may be reading the same
    directory, and a half-written file is indistinguishable from a complete one by name. Rename
    within a directory is atomic on every filesystem this runs on.
    """
    os.makedirs(policy_dir, exist_ok=True)
    payload = {'step': int(step), 'model': net.state_dict()}
    if extra:
        overlap = set(extra) & set(payload)
        if overlap:
            raise CheckpointError('extra may not shadow {0}'.format(sorted(overlap)))
        payload.update(extra)
    final = path(policy_dir, step)
    staging = final + '.partial'
    torch.save(payload, staging)
    os.replace(staging, final)
    return final


def load(checkpoint_path, net=None, device='cpu'):
    """The payload, with `net` populated from it if given.

    `strict=True` on purpose. A `state_dict` that half-loads is the failure mode `arch.json` exists
    to prevent, and torch will refuse a shape mismatch here — so the two together cover both shape
    and meaning.
    """
    if not os.path.exists(checkpoint_path):
        raise CheckpointError('no checkpoint at {0}'.format(checkpoint_path))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    named_step = step_of(checkpoint_path)
    if named_step is not None and int(payload['step']) != named_step:
        raise CheckpointError(
            '{0} records step {1}. A checkpoint renamed to a step it did not come from would be '
            'measured and reported against the wrong step'.format(checkpoint_path, payload['step']))
    if net is not None:
        net.load_state_dict(payload['model'], strict=True)
    return payload
