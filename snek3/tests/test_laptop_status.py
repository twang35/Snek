"""`tools/laptop_status.py`: the laptop's status branch."""

import json
import os
import subprocess

from tools import laptop_status


def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, text=True, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def test_build_is_the_daemons_shape_with_the_laptops_own_timestamp():
    running = [{'id': 'b16aa-kl003-seed1', 'type': 'train', 'policy': 'b16aa-kl003-seed1',
                'policies': ['b16aa-kl003-seed1'], 'label': 'b16: kl003, seed 1 of 4 -- wave 1 of 5',
                'step': 25000000, 'max_steps': 50000000}]
    queued = [{'id': 'b16-stageb', 'type': 'eval', 'policies': ['b16aa-kl003-seed1']}]
    status = laptop_status.build(running, queued, now=1788600000.0)
    assert status['box'] == 'laptop' and status['ts'] == 1788600000.0
    assert status['iso'].startswith('2026-09-05T')
    assert status['at_a_glance']['running'] == ['b16 | kl003 -- wave 1 of 5 | training 50% (1 arm)']
    assert status['at_a_glance']['queued'] == ['b16 evals | kl003 | queued (1 arm)']
    assert status['running'][0]['step'] == 25000000 and status['queued_ids'] == ['b16-stageb']
    json.dumps(status)     # what goes on the branch


def test_ensure_worktree_makes_an_empty_rooted_branch_and_is_idempotent(tmp_path):
    """No remote and no branch yet: the branch starts from an empty tree, so the worktree never holds
    a copy of the source, and a second call finds it and does nothing."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(['init', '-q'], str(repo))
    _git(['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'root'], str(repo))
    host = laptop_status.host(repo=str(repo), worktree=str(tmp_path / 'bus' / 'status'),
                              branch='laptop-status', remote='origin')
    path = laptop_status.ensure_worktree(host)
    assert os.path.exists(os.path.join(path, '.git'))
    assert _git(['rev-parse', '--abbrev-ref', 'HEAD'], path) == 'laptop-status'
    assert _git(['ls-tree', 'HEAD'], path) == ''                      # the empty root
    assert laptop_status.ensure_worktree(host) == path
    # and a publish through gitbus lands a commit on it (no remote: the push fails, the commit stays)
    from desktop.runner import gitbus
    with open(os.path.join(path, '.git')):
        pass
    _git(['config', 'user.name', 't'], path)
    _git(['config', 'user.email', 't@t'], path)
    assert gitbus.publish_status(host, '{"iso": "x"}') is False
    assert _git(['show', 'HEAD:status.json'], path) == '{"iso": "x"}'
