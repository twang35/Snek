"""`tools/site_build.py`: both feeds flattened, this box's files on top, one snapshot commit, incremental."""
import json
import os
import subprocess

import pytest

from tools import site_build


def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def _commit_feed(remote, feed, files, message):
    """Adds `files` ({'results/<job>/<name>': bytes}) to `feed` on the bare remote through a scratch clone."""
    scratch = remote + '-scratch-' + feed
    if not os.path.isdir(scratch):
        subprocess.run(['git', 'clone', '-q', remote, scratch], check=True, capture_output=True)
        _git(['config', 'user.email', 't@t'], scratch); _git(['config', 'user.name', 't'], scratch)
        if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'origin/' + feed], cwd=scratch, capture_output=True).returncode == 0:
            _git(['checkout', '-q', feed], scratch)
        else:
            _git(['checkout', '-q', '--orphan', feed], scratch)
            subprocess.run(['git', 'rm', '-rfq', '--cached', '.'], cwd=scratch, capture_output=True)
            for name in os.listdir(scratch):
                if name != '.git':
                    path = os.path.join(scratch, name)
                    (os.remove if os.path.isfile(path) else __import__('shutil').rmtree)(path)
    for path, body in files.items():
        os.makedirs(os.path.dirname(os.path.join(scratch, path)), exist_ok=True)
        with open(os.path.join(scratch, path), 'wb') as handle:
            handle.write(body)
    _git(['add', '-A'], scratch)
    _git(['commit', '-q', '-m', message], scratch)
    _git(['push', '-q', 'origin', feed], scratch)


def _evals(step):
    return json.dumps({'summary': {'step': step, 'best_perfect30': {'rate': 90.0}}, 'evals': [], 'resumes': []}).encode()


@pytest.fixture
def world(tmp_path):
    remote = str(tmp_path / 'remote.git')
    subprocess.run(['git', 'init', '-q', '--bare', remote], check=True)
    repo = str(tmp_path / 'laptop')
    subprocess.run(['git', 'init', '-q', repo], check=True)
    _git(['config', 'user.email', 't@t'], repo); _git(['config', 'user.name', 't'], repo)
    _git(['remote', 'add', 'origin', remote], repo)
    open(os.path.join(repo, 'README'), 'w').close()
    _git(['add', '-A'], repo); _git(['commit', '-q', '-m', 'root'], repo)
    _git(['push', '-q', 'origin', 'HEAD:master'], repo)
    viewer = str(tmp_path / 'viewer'); os.makedirs(viewer)
    with open(os.path.join(viewer, 'index.html'), 'w') as handle:
        handle.write('<html>viewer</html>')
    runs = str(tmp_path / 'runs'); os.makedirs(runs)
    png = b'\x89PNG local'
    return {'remote': remote, 'repo': repo, 'viewer': viewer, 'runs': runs, 'png': png,
            'worktree': str(tmp_path / 'bus' / 'site'), 'build': str(tmp_path / 'bus' / 'site-build')}


def _build(world, **kwargs):
    lines = []
    result = site_build.build(repo=world['repo'], remote='origin', branch='site', worktree=world['worktree'],
                              feeds=site_build.FEEDS, runs_dir=world['runs'], build_dir=world['build'],
                              viewer_dir=world['viewer'], log=lines.append, **kwargs)
    return result, lines


def test_both_feeds_are_flattened_the_local_runs_win_and_the_site_is_one_snapshot_commit(world):
    _commit_feed(world['remote'], 'results', {
        'results/b1a-x-seed1/b1a-x-seed1.png': b'\x89PNG desktop', 'results/b1a-x-seed1/b1a-x-seed1_evals.json': _evals(10),
        'results/b1-stageb/b1a-x-seed1_checkpoint_evals.json': json.dumps({'rows': []}).encode(),
        'results/b1-stageb/b1a-x-seed1_checkpoint_evals-s1of2.json': b'shard',
        'results/b1-stageb/b1a-x-seed1_checkpoint_evals.png': b'\x89PNG stageb',
        'results/b3a-lr3e4-g99/b3a-lr3e4-g99.png': b'\x89PNG no seed in the name, snek3 all the same',
        'results/smoke-3/smoke.png': b'\x89PNG smoke', 'results/sw1w4-a/sw1w4-a.png': b'\x89PNG sweep',
        'results/sse8/sse8.png': b'\x89PNG sweep', 'results/p1c-fc200x100ep8-seed3/p1c-fc200x100ep8-seed3.png': b'\x89PNG old name'},
        'desktop wave')
    _commit_feed(world['remote'], 'laptop-results', {
        'results/b2a-y-seed1/b2a-y-seed1.png': b'\x89PNG laptop', 'results/b2a-y-seed1/b2a-y-seed1_evals.json': _evals(10)}, 'laptop arm')
    # this box (the desktop) has a newer picture of its own arm
    with open(os.path.join(world['runs'], 'b1a-x-seed1.png'), 'wb') as handle:
        handle.write(world['png'])
    result, lines = _build(world)
    assert result['built'] and result['pushed'] is True and result['arms'] == 3, 'the three snek3 arms, nothing else'
    assert os.path.exists(os.path.join(world['build'], 'b3a-lr3e4-g99.png'))
    for name in ('smoke.png', 'sw1w4-a.png', 'sse8.png', 'p1c-fc200x100ep8-seed3.png'):
        assert not os.path.exists(os.path.join(world['build'], name)), name
    # a stray file an earlier build left is pruned on the next one
    open(os.path.join(world['build'], 'p2a-ep8-seed1.png'), 'wb').write(b'x')
    assert not os.path.exists(os.path.join(world['build'], 'b1a-x-seed1_checkpoint_evals-s1of2.json')), 'shards stay out'
    with open(os.path.join(world['worktree'], 'charts', 'b1a-x-seed1.png'), 'rb') as handle:
        assert handle.read() == world['png'], 'the local copy, newer, wins over the feed'
    assert os.path.exists(os.path.join(world['worktree'], 'charts', 'b2a-y-seed1.png'))
    assert os.path.exists(os.path.join(world['worktree'], 'charts', 'b1a-x-seed1_checkpoint_evals.png'))
    assert os.path.exists(os.path.join(world['worktree'], '.nojekyll'))
    with open(os.path.join(world['worktree'], 'manifest.js')) as handle:
        assert 'charts/' in handle.read()
    assert _git(['rev-list', '--count', 'origin/site'], world['repo']) == '1', 'a snapshot: one commit'
    # the stray file is pruned, which is a build; then nothing moved: skipped, no push
    result, lines = _build(world)
    assert result['built'] and not os.path.exists(os.path.join(world['build'], 'p2a-ep8-seed1.png'))
    result, lines = _build(world)
    assert not result['built'] and 'skipped' in lines[-1]
    # a new laptop wave: only its files are read from the feed (incremental), and the branch is still one commit
    _commit_feed(world['remote'], 'laptop-results', {
        'results/b2b-y-seed2/b2b-y-seed2.png': b'\x89PNG laptop2', 'results/b2b-y-seed2/b2b-y-seed2_evals.json': _evals(10)}, 'laptop arm 2')
    result, lines = _build(world)
    assert result['built'] and result['flattened'] == 2 and result['arms'] == 4
    assert _git(['rev-list', '--count', 'origin/site'], world['repo']) == '1'
    # --force builds with nothing moved; the manifest's timestamp changes, so it is a push, and still one commit
    result, lines = _build(world, force=True)
    assert result['built'] and result['pushed'] is True
    assert _git(['rev-list', '--count', 'origin/site'], world['repo']) == '1'


def test_a_file_deleted_from_a_feed_leaves_the_site(world):
    _commit_feed(world['remote'], 'results', {'results/b1a-x-seed1/b1a-x-seed1.png': b'\x89PNG',
                                              'results/b41a-b29repro-seed1-closeout/b41a-b29repro-seed1.png': b'\x89PNG snek2'}, 'wave')
    result, _ = _build(world, push=False)
    assert result['arms'] == 2
    scratch = str(world['remote']) + '-rm'
    _git(['clone', '-q', '-b', 'results', str(world['remote']), scratch], None)
    _git(['rm', '-rq', 'results/b41a-b29repro-seed1-closeout'], scratch)
    _git(['commit', '-q', '-m', 'snek2 out'], scratch)
    _git(['push', '-q', 'origin', 'results'], scratch)
    result, _ = _build(world, push=False)
    assert result['built'] and result['flattened'] == 1 and result['arms'] == 1
    assert not os.path.exists(os.path.join(world['build'], 'b41a-b29repro-seed1.png'))


def test_a_feed_is_one_parentless_commit_whose_tree_keeps_every_earlier_job(world):
    """`gitbus.publish_results` / `publish_jobs`: every publish rewrites the branch as a new root commit,
    force-pushed with a lease; the persistent worktree means the tree still holds every job published
    before; a publish that changes nothing pushes nothing; and a branch with history collapses to one
    commit on its first publish after the change."""
    from desktop.daemon import gitbus
    _commit_feed(world['remote'], 'laptop-results', {'results/b1a-x-seed1/b1a-x-seed1.png': b'old'}, 'old history')
    _commit_feed(world['remote'], 'laptop-results', {'results/b1a-x-seed1/b1a-x-seed1_evals.json': b'{}'}, 'more history')
    _git(['fetch', '-q', 'origin', 'laptop-results'], world['repo'])
    assert _git(['rev-list', '--count', 'origin/laptop-results'], world['repo']) == '2'
    host = {'REPO_PATH': world['repo'], 'RESULTS_WORKTREE': str(world['worktree']) + '-results',
            'RESULTS_BRANCH': 'laptop-results', 'GIT_REMOTE': 'origin'}
    gitbus.ensure_worktree(host['REPO_PATH'], host['RESULTS_WORKTREE'], host['RESULTS_BRANCH'], host['GIT_REMOTE'])
    _git(['config', 'user.email', 't@t'], host['RESULTS_WORKTREE']); _git(['config', 'user.name', 't'], host['RESULTS_WORKTREE'])
    png = os.path.join(world['runs'], 'b30a-h1-seed1.png')
    with open(png, 'wb') as handle:
        handle.write(b'\x89PNG t1')

    class Job(object):
        id = 'b30a-h1-seed1'
    assert gitbus.publish_results(host, Job(), [png]) is True
    tree = lambda: sorted(_git(['ls-tree', '-r', '--name-only', 'origin/laptop-results'], world['repo']).splitlines())
    assert _git(['rev-list', '--count', 'origin/laptop-results'], world['repo']) == '1', 'history collapsed'
    assert tree() == ['results/b1a-x-seed1/b1a-x-seed1.png', 'results/b1a-x-seed1/b1a-x-seed1_evals.json',
                      'results/b30a-h1-seed1/b30a-h1-seed1.png'], 'the earlier jobs are still in the tree'
    first = _git(['rev-parse', 'origin/laptop-results'], world['repo'])
    assert gitbus.publish_results(host, Job(), [png]) is True
    assert _git(['rev-parse', 'origin/laptop-results'], world['repo']) == first, 'unchanged: no new commit'
    # live pictures of a wave: several jobs, one commit; the final later overwrites the same path
    with open(png, 'wb') as handle:
        handle.write(b'\x89PNG t2')
    png2 = os.path.join(world['runs'], 'b30b-h1-seed2.png')
    with open(png2, 'wb') as handle:
        handle.write(b'\x89PNG b')
    pushed, copied = gitbus.publish_jobs(host, {'b30a-h1-seed1': [png], 'b30b-h1-seed2': [png2]}, 'live')
    assert (pushed, copied) == (True, 2)
    assert _git(['rev-list', '--count', 'origin/laptop-results'], world['repo']) == '1'
    assert subprocess.run(['git', 'show', 'origin/laptop-results:results/b30a-h1-seed1/b30a-h1-seed1.png'],
                          cwd=world['repo'], capture_output=True, check=True).stdout == b'\x89PNG t2'
    assert 'results/b30b-h1-seed2/b30b-h1-seed2.png' in tree()
    # and the site build reads the rewritten feed: its anchor commit is gone, so the feed is read whole
    result, _ = _build(world, push=False)
    assert result['arms'] == 3
    result, _ = _build(world, push=False)
    assert result['built'] is False
