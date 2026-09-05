"""`desktop/deploy` on the box: the box's pictures survive the merge, identical JSON is staged, differing
JSON stops the merge. Two real git repos under tmp_path — a bare "origin" and the "box" clone."""

import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'desktop'))

from daemon import deploy  # noqa: E402


def _git(repo, *args):
    return subprocess.run(['git'] + list(args), cwd=repo, check=True, capture_output=True, text=True).stdout


def _commit_files(repo, files, message):
    for path, content in files.items():
        full = os.path.join(repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as handle:
            handle.write(content)
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', message)


@pytest.fixture
def repos(tmp_path):
    origin = str(tmp_path / 'origin.git'); laptop = str(tmp_path / 'laptop'); box = str(tmp_path / 'box')
    _git(str(tmp_path), 'init', '-q', '--bare', '-b', 'master', origin)
    _git(str(tmp_path), 'clone', '-q', origin, laptop)
    _commit_files(laptop, {'snek3/train.py': 'v1\n'}, 'base')
    _git(laptop, 'push', '-q', 'origin', 'master')
    _git(str(tmp_path), 'clone', '-q', origin, box)
    return laptop, box


def _laptop_commits(laptop, files):
    _commit_files(laptop, files, 'progress update')
    _git(laptop, 'push', '-q', 'origin', 'master')


def _box_writes(box, files):
    for path, content in files.items():
        full = os.path.join(box, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as handle:
            handle.write(content)


def test_the_boxs_newer_pictures_survive_the_merge(repos):
    laptop, box = repos
    _box_writes(box, {'snek3/runs/b10aa-g70-seed1.png': 'box-newer', 'snek3/runs/b10aa-g70-seed1.md': 'box-newer',
                      'snek3/runs/b10aa-g70-seed1_evals.json': '{"live": 1}'})
    _laptop_commits(laptop, {'snek3/runs/b10aa-g70-seed1.png': 'laptop-older', 'snek3/runs/b10aa-g70-seed1.md': 'x',
                             'snek3/train.py': 'v2\n'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    decision = deploy.plan(box)
    assert sorted(decision['keep']) == ['snek3/runs/b10aa-g70-seed1.md', 'snek3/runs/b10aa-g70-seed1.png']
    assert decision['stage'] == [] and decision['differs'] == []   # the live JSON was never committed
    out = io.StringIO()
    assert deploy.apply(box, decision, out=out) == 0
    assert open(os.path.join(box, 'snek3/train.py')).read() == 'v2\n'
    assert open(os.path.join(box, 'snek3/runs/b10aa-g70-seed1.png')).read() == 'box-newer'
    assert open(os.path.join(box, 'snek3/runs/b10aa-g70-seed1_evals.json')).read() == '{"live": 1}'
    assert 'kept 2' in out.getvalue()
    # the kept pictures are now tracked-and-modified; a second deploy over them still goes through
    _laptop_commits(laptop, {'snek3/runs/b10aa-g70-seed1.png': 'laptop-later-still', 'snek3/train.py': 'v3\n'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    decision = deploy.plan(box)
    assert sorted(decision['keep']) == ['snek3/runs/b10aa-g70-seed1.md', 'snek3/runs/b10aa-g70-seed1.png']
    assert deploy.apply(box, decision, out=io.StringIO()) == 0
    assert open(os.path.join(box, 'snek3/train.py')).read() == 'v3\n'
    assert open(os.path.join(box, 'snek3/runs/b10aa-g70-seed1.png')).read() == 'box-newer'


def test_identical_json_is_staged_and_the_index_ends_clean(repos):
    laptop, box = repos
    _box_writes(box, {'snek3/runs/b9aa-lam0-seed1_evals.json': '{"same": 1}'})
    _laptop_commits(laptop, {'snek3/runs/b9aa-lam0-seed1_evals.json': '{"same": 1}', 'snek3/runs/b9aa-lam0-seed1.png': 'p'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    decision = deploy.plan(box)
    assert decision == {'keep': [], 'stage': ['snek3/runs/b9aa-lam0-seed1_evals.json'], 'differs': []}
    assert deploy.apply(box, decision, out=io.StringIO()) == 0
    assert _git(box, 'status', '--porcelain').strip() == ''


def test_a_differing_json_stops_everything_and_names_itself(repos):
    laptop, box = repos
    _box_writes(box, {'snek3/runs/b10aa-g70-seed1_evals.json': '{"live": 2}', 'snek3/runs/b10aa-g70-seed1.png': 'b'})
    _laptop_commits(laptop, {'snek3/runs/b10aa-g70-seed1_evals.json': '{"live": 1}', 'snek3/runs/b10aa-g70-seed1.png': 'l',
                             'snek3/train.py': 'v2\n'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    decision = deploy.plan(box)
    assert decision['differs'] == ['snek3/runs/b10aa-g70-seed1_evals.json']
    out = io.StringIO()
    assert deploy.apply(box, decision, out=out) == deploy.EXIT_DIFFERS
    assert 'DIFFERS  snek3/runs/b10aa-g70-seed1_evals.json' in out.getvalue()
    # nothing moved: the picture is still the box's, the code is still v1
    assert open(os.path.join(box, 'snek3/runs/b10aa-g70-seed1.png')).read() == 'b'
    assert open(os.path.join(box, 'snek3/train.py')).read() == 'v1\n'


def test_dry_run_changes_nothing(repos):
    laptop, box = repos
    _box_writes(box, {'snek3/runs/b10aa-g70-seed1.png': 'b'})
    _laptop_commits(laptop, {'snek3/runs/b10aa-g70-seed1.png': 'l'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    out = io.StringIO()
    assert deploy.apply(box, deploy.plan(box), dry_run=True, out=out) == 0
    assert 'would merge' in out.getvalue()
    assert open(os.path.join(box, 'snek3/runs/b10aa-g70-seed1.png')).read() == 'b'


def test_snek2_stragglers_are_settled_like_snek3_runs(repos):
    # 2026-09-03: the frozen era's runs/ and evals/ leftovers were committed on the laptop while the box
    # held untracked copies; the merge refused on 75 files the script had not looked at (exit 4).
    laptop, box = repos
    _box_writes(box, {'snek2/runs/b46a-c51batch512seed1_evals.json': '{"same": 1}',
                      'snek2/runs/b46a-c51batch512seed1.png': 'box', 'snek2/evals/b45a_eval_progress.png': 'box'})
    _laptop_commits(laptop, {'snek2/runs/b46a-c51batch512seed1_evals.json': '{"same": 1}',
                             'snek2/runs/b46a-c51batch512seed1.png': 'laptop', 'snek2/evals/b45a_eval_progress.png': 'laptop',
                             'snek3/train.py': 'v2\n'})
    _git(box, 'fetch', '-q', 'origin', 'master')
    decision = deploy.plan(box)
    assert decision['stage'] == ['snek2/runs/b46a-c51batch512seed1_evals.json']
    assert sorted(decision['keep']) == ['snek2/evals/b45a_eval_progress.png', 'snek2/runs/b46a-c51batch512seed1.png']
    assert deploy.apply(box, decision, out=io.StringIO()) == 0
    assert open(os.path.join(box, 'snek3/train.py')).read() == 'v2\n'
    assert open(os.path.join(box, 'snek2/runs/b46a-c51batch512seed1.png')).read() == 'box'
    assert _git(box, 'status', '--porcelain', '--', 'snek2/runs/b46a-c51batch512seed1_evals.json') == ''
