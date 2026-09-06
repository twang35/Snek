"""`tools/claims.py`: the shared queue. The pure rules on dicts, and the race on a real bare remote.

The race is the point of the module, so it is exercised for real: two clones of one bare `origin`, each
with its own claims worktree, both compute the same next wave from the same fetched state and both push.
Exactly one push lands; the loser resets, re-reads, and claims the next wave instead.
"""

import json
import os
import subprocess

import pytest

from tools import claims


# ---------------------------------------------------------------- fixtures

def _spec(spec_id, priority=100, kind='train', policies=None, box=None):
    spec = {'project': 'snek3', 'id': spec_id, 'type': kind, 'priority': priority, 'env': {}, 'label': ''}
    if kind == 'train':
        spec.update({'policy': spec_id, 'max_steps': 10})
    else:
        spec['policies'] = list(policies or [])
    if box:
        spec['box'] = box
    return spec


def _batch(batch, n, priority=100, cell='x'):
    return {s['id']: s for s in (_spec('{0}{1}-{2}-seed{3}'.format(batch, chr(97 + i), cell, i % 4 + 1), priority)
                                 for i in range(n))}


def _claim(batch, wave, box, arms):
    return claims.wave_record(batch, wave, box, arms, now=1000.0)


# ---------------------------------------------------------------- the pure rules

def test_a_wave_is_the_next_arms_of_the_first_batch_in_priority_order_numbered_past_the_highest_claim():
    specs = dict(_batch('b21', 24, 220), **_batch('b18', 24, 240))
    record = claims.next_claim(specs, [], 'desktop', 8)
    assert record['batch'] == 'b21' and record['wave'] == 1 and record['box'] == 'desktop'
    assert record['arms'] == sorted(_batch('b21', 24))[:8]
    held = [_claim('b21', 1, 'desktop', record['arms']), _claim('b21', 2, 'laptop', sorted(_batch('b21', 24))[8:16])]
    nxt = claims.next_claim(specs, held, 'desktop', 8)
    assert (nxt['batch'], nxt['wave'], nxt['arms']) == ('b21', 3, sorted(_batch('b21', 24))[16:])
    held.append(nxt)
    assert claims.next_claim(specs, held, 'laptop', 8)['batch'] == 'b18', 'b21 is all claimed; the next batch'
    everything = held + [_claim('b18', w, 'laptop', sorted(_batch('b18', 24))[(w - 1) * 8: w * 8]) for w in (1, 2, 3)]
    assert claims.next_claim(specs, everything, 'laptop', 8) is None


def test_a_released_wave_number_is_never_reused_and_its_arms_go_back_to_the_pool():
    specs = _batch('b21', 16, 220)
    arms = sorted(specs)
    held = [_claim('b21', 1, 'desktop', arms[:8]), _claim('b21', 2, 'laptop', arms[8:])]
    # w2 released: the laptop went quiet. `release` leaves a tombstone -- no arms, the number kept --
    # so the arms come back but the next wave is w3, never a second w2 (its pass ids are on a feed).
    tombstone = dict(held[1], kind='released', arms=[])
    released = [held[0], tombstone]
    assert claims.claimed_ids(released) == set(arms[:8]), 'a tombstone covers no arm'
    record = claims.next_claim(specs, released, 'desktop', 8)
    assert record['wave'] == 3 and record['arms'] == arms[8:]
    # a tombstone holds nothing in the pool view and is nothing to mirror
    view = claims.pool_view(specs, released)
    assert [h['id'] for h in view['held']['laptop']] == [] if 'laptop' in view['held'] else True
    assert view['lines'] == ['b21 training | 8/16 arms', 'desktop holds b21-w1 (8 arms)']
    # with every live claim gone but the tombstones the numbering still continues
    record = claims.next_claim(specs, [dict(held[0], kind='released', arms=[]), tombstone], 'desktop', 8)
    assert record['wave'] == 3 and record['arms'] == arms[:8]
    # and with wave 3 still standing a release of 1 and 2 numbers the next wave 4
    record = claims.next_claim(specs, [_claim('b21', 3, 'laptop', [])], 'desktop', 8)
    assert record['wave'] == 4


def test_a_pinned_spec_is_only_that_boxs_and_a_batch_of_pins_for_the_other_box_is_passed_over():
    specs = dict(_batch('b21', 8, 220), **_batch('b18', 8, 240))
    for spec in _batch('b21', 8).values():
        specs[spec['id']]['box'] = 'laptop'
    assert claims.next_claim(specs, [], 'desktop', 8)['batch'] == 'b18'
    assert claims.next_claim(specs, [], 'laptop', 8)['batch'] == 'b21'
    specs['b21a-x-seed1']['box'] = None
    record = claims.next_claim(specs, [], 'desktop', 8)
    assert record['batch'] == 'b21' and record['arms'] == ['b21a-x-seed1'], 'a wave of the eligible arms only'


def test_a_wave_is_at_most_wave_size_arms_and_a_short_tail_is_its_own_wave():
    specs = _batch('b21', 11, 220)
    first = claims.next_claim(specs, [], 'desktop', 8)
    assert len(first['arms']) == 8
    second = claims.next_claim(specs, [first], 'laptop', 8)
    assert len(second['arms']) == 3 and second['wave'] == 2


def test_an_eval_spec_waits_for_its_arms_and_needs_their_checkpoints_here():
    specs = _batch('b7', 4, 130)
    specs['b7-hof30k-confirm'] = _spec('b7-hof30k-confirm', 15, 'eval', policies=['b7a-x-seed1', 'b7b-x-seed2'])
    # b7's arms are unclaimed training specs: the eval is passed over and the wave comes first
    record = claims.next_claim(specs, [], 'laptop', 8, has_checkpoints=lambda p: True)
    assert record['kind'] == 'wave' and record['batch'] == 'b7'
    held = [record]
    # arms claimed (by anyone) and checkpoints here: the eval is claimable
    ev = claims.next_claim(specs, held, 'laptop', 8, has_checkpoints=lambda p: True)
    assert ev['kind'] == 'eval' and ev['id'] == 'b7-hof30k-confirm' and ev['arms'] == ['b7a-x-seed1', 'b7b-x-seed2']
    # checkpoints elsewhere: not this box's to take
    assert claims.next_claim(specs, held, 'desktop', 8, has_checkpoints=lambda p: False) is None
    # a batch whose eval is at priority 15 sorts first among batches, ahead of a training batch at 100
    specs.update(_batch('b30', 8, 100))
    ev = claims.next_claim(specs, held, 'laptop', 8, has_checkpoints=lambda p: True)
    assert ev['kind'] == 'eval'
    assert claims.next_claim(specs, held + [ev], 'laptop', 8, has_checkpoints=lambda p: True)['batch'] == 'b30'


def test_the_eval_probe_is_the_checkpoint_directory(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), 'b7a-x-seed1'))
    assert claims.default_has_checkpoints(['b7a-x-seed1'], policy_dir=str(tmp_path))
    assert not claims.default_has_checkpoints(['b7a-x-seed1', 'b7b-x-seed2'], policy_dir=str(tmp_path))


def test_the_pool_view_lists_unclaimed_work_by_batch_each_boxs_open_holdings_and_a_stranded_claim():
    specs = dict(_batch('b21', 16, 220), **_batch('b18', 8, 240))
    specs['b7-hof30k-confirm'] = _spec('b7-hof30k-confirm', 15, 'eval', policies=['b7a-x'], box='laptop')
    arms = sorted(_batch('b21', 16))
    records = [_claim('b21', 1, 'desktop', arms[:8]), _claim('b21', 2, 'laptop', arms[8:])]
    published = set(arms[:8]) | {'b21-stageb', 'b21-hof5000', 'b21-hof30k'}
    view = claims.pool_view(specs, records, published=published, running={arms[9]: 'laptop'},
                            status_ages={'laptop': 3 * 3600, 'desktop': 10.0})
    assert view['lines'] == ['b7 eval | b7-hof30k-confirm | pinned laptop',
                             'b18 training | 8/8 arms',
                             'laptop holds b21-w2 (8 arms, running)'], \
        'a line naming a batch and no box is unclaimed by construction; the count is out of the batch on ops'
    assert view['held']['desktop'][0]['done'] is True and view['held']['laptop'][0]['done'] is False
    assert view['attention'] == ['** laptop holds b21-w2 but its status is 3.0h old; if it is not coming back, '
                                 '`python -m tools.claims release <id>` returns the work to the pool']
    assert view['ledger'][arms[0]] == 'done' and view['ledger'][arms[9]] == 'running'
    assert view['ledger'][arms[10]] == 'queued' and view['ledger']['b21-stageb'] == 'done'
    assert view['ledger']['b18a-x-seed1'] == 'queued'
    fresh = claims.pool_view(specs, records, published=published, status_ages={'laptop': 60.0})
    assert fresh['attention'] == [], 'a box that published a minute ago is not stranded'
    assert claims.pool_view(specs, records, malformed=[('bad.json', 'no max_steps')])['attention'] == [
        '** spec bad.json on ops is malformed and is not run: no max_steps']


def test_the_mirror_writes_this_boxs_waves_with_their_numbers_and_removes_what_it_no_longer_holds(tmp_path):
    queue = str(tmp_path / 'queue')
    specs = dict(_batch('b21', 16, 220), **_batch('b18', 8, 240))
    specs['b21-confirm'] = _spec('b21-confirm', 15, 'eval', policies=['b21a-x-seed1'])
    arms = sorted(_batch('b21', 16))
    os.makedirs(os.path.join(queue, 'b18'))
    with open(os.path.join(queue, 'b18', 'b18a-x-seed1.json'), 'w') as handle:        # a stale mirror entry
        json.dump(specs['b18a-x-seed1'], handle)
    open(os.path.join(queue, 'b18', '.failed-b18-stageb'), 'w').close()               # a marker: kept
    mine = [_claim('b21', 2, 'laptop', arms[8:]), claims.eval_record('b21', 'b21-confirm', 'laptop', ['b21a-x-seed1'])]
    mirrored = claims.mirror(queue, specs, mine, log=lambda m: None)
    assert mirrored == set(arms[8:]) | {'b21-confirm'}
    with open(os.path.join(queue, 'b21', arms[8] + '.json')) as handle:
        assert json.load(handle)['_wave'] == 2
    with open(os.path.join(queue, 'b21', 'b21-confirm.json')) as handle:
        assert '_wave' not in json.load(handle)
    assert not os.path.exists(os.path.join(queue, 'b18', 'b18a-x-seed1.json'))
    assert os.path.exists(os.path.join(queue, 'b18', '.failed-b18-stageb'))
    # a held arm dequeued from ops is logged and not mirrored; a second mirror with nothing changed rewrites nothing
    lines = []
    del specs[arms[9]]
    claims.mirror(queue, specs, mine, log=lines.append)
    assert any(arms[9] in line and 'no longer on ops' in line for line in lines)
    assert not os.path.exists(os.path.join(queue, 'b21', arms[9] + '.json'))


# ---------------------------------------------------------------- the race, for real

def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def _clone(remote, path):
    subprocess.run(['git', 'clone', '-q', remote, path], check=True, capture_output=True)
    _git(['config', 'user.email', 't@t'], path)
    _git(['config', 'user.name', 't'], path)
    return path


def _push_ops(remote, specs, scratch):
    """The `ops` branch on the bare remote: `snek3/desktop/queue/pending/<id>.json` per spec."""
    if not os.path.isdir(scratch):
        _clone(remote, scratch)
        if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'origin/ops'], cwd=scratch,
                          capture_output=True).returncode == 0:
            _git(['checkout', '-q', 'ops'], scratch)
        else:
            _git(['checkout', '-q', '--orphan', 'ops'], scratch)
            subprocess.run(['git', 'rm', '-rfq', '--cached', '.'], cwd=scratch, capture_output=True)
            for name in os.listdir(scratch):
                if name != '.git':
                    os.remove(os.path.join(scratch, name))
    pending = os.path.join(scratch, claims.QUEUE_DIR)
    os.makedirs(pending, exist_ok=True)
    for name in os.listdir(pending):
        os.remove(os.path.join(pending, name))
    for spec_id, spec in specs.items():
        with open(os.path.join(pending, spec_id + '.json'), 'w') as handle:
            json.dump(spec, handle)
    _git(['add', '-A'], scratch)
    _git(['commit', '-q', '--allow-empty', '-m', 'ops'], scratch)
    _git(['push', '-q', 'origin', 'ops'], scratch)


@pytest.fixture
def bus(tmp_path):
    remote = str(tmp_path / 'origin.git')
    subprocess.run(['git', 'init', '-q', '--bare', remote], check=True)
    seed = _clone(remote, str(tmp_path / 'seed'))
    open(os.path.join(seed, 'README'), 'w').close()
    _git(['add', '-A'], seed)
    _git(['commit', '-q', '-m', 'root'], seed)
    _git(['push', '-q', 'origin', 'HEAD:master'], seed)
    specs = dict(_batch('b21', 16, 220), **_batch('b18', 8, 240))
    _push_ops(remote, specs, str(tmp_path / 'ops-scratch'))
    boxes = {}
    for box in ('desktop', 'laptop'):
        repo = _clone(remote, str(tmp_path / box))
        boxes[box] = claims.Store(repo=repo, remote='origin', branch='claims',
                                  worktree=str(tmp_path / 'bus' / box / 'claims'), log=lambda m: None)
    return {'remote': remote, 'specs': specs, 'stores': boxes, 'tmp': tmp_path}


def _read(store):
    return lambda **kw: claims.read_specs(repo=store.repo, remote='origin')


def test_two_boxes_racing_for_the_same_wave_end_up_with_different_waves(bus):
    desktop, laptop = bus['stores']['desktop'], bus['stores']['laptop']
    # both read the same (empty) state and compute the same first wave
    desktop.sync(); laptop.sync()
    specs, _ = claims.read_specs(repo=desktop.repo, remote='origin')
    wanted_d = claims.next_claim(specs, desktop.records, 'desktop', 8)
    wanted_l = claims.next_claim(specs, laptop.records, 'laptop', 8)
    assert wanted_d['arms'] == wanted_l['arms'] and wanted_d['wave'] == wanted_l['wave'] == 1
    assert desktop.try_claim(wanted_d) == 'won'
    assert laptop.try_claim(wanted_l) == 'lost', 'the second push is not a fast-forward'
    # the loser's worktree is back on the remote's head, holding the winner's claim and not its own
    laptop.sync()
    assert [r['box'] for r in laptop.records] == ['desktop']
    assert _git(['status', '--porcelain'], laptop.worktree) == ''
    # and its recomputation is the next wave
    nxt = claims.next_claim(specs, laptop.records, 'laptop', 8)
    assert nxt['wave'] == 2 and nxt['arms'] == sorted(_batch('b21', 16))[8:]
    assert laptop.try_claim(nxt) == 'won'
    desktop.sync()
    assert [(r['batch'], r['wave'], r['box']) for r in desktop.records] == [('b21', 1, 'desktop'), ('b21', 2, 'laptop')]


def test_claim_next_recomputes_after_a_lost_race_and_returns_none_when_the_pool_is_empty(bus):
    desktop, laptop = bus['stores']['desktop'], bus['stores']['laptop']
    laptop.sync()                                   # the laptop's read predates the desktop's claim
    won = claims.claim_next(desktop, 'desktop', 8, read_specs=_read(desktop), log=lambda m: None)
    assert (won['batch'], won['wave']) == ('b21', 1)
    # the laptop's first sync inside claim_next is made a no-op, so it computes on its stale worktree and
    # pushes from the old parent; the server rejects that, and the retry's real sync recomputes
    real_sync, calls = laptop.sync, {'n': 0}

    def stale_then_real():
        calls['n'] += 1
        return laptop.head if calls['n'] == 1 else real_sync()
    laptop.sync = stale_then_real
    won = claims.claim_next(laptop, 'laptop', 8, read_specs=_read(laptop), log=lambda m: None)
    assert (won['batch'], won['wave']) == ('b21', 2), 'lost once, recomputed, took the next wave'
    assert calls['n'] == 2
    laptop.sync = real_sync
    third = claims.claim_next(laptop, 'laptop', 8, read_specs=_read(laptop), log=lambda m: None)
    assert (third['batch'], third['wave']) == ('b18', 1)
    assert claims.claim_next(desktop, 'desktop', 8, read_specs=_read(desktop), log=lambda m: None) is None


def test_release_returns_the_arms_to_the_pool_and_the_next_claim_numbers_past_it(bus):
    desktop, laptop = bus['stores']['desktop'], bus['stores']['laptop']
    first = claims.claim_next(desktop, 'desktop', 8, read_specs=_read(desktop), log=lambda m: None)
    second = claims.claim_next(laptop, 'laptop', 8, read_specs=_read(laptop), log=lambda m: None)
    assert (first['wave'], second['wave']) == (1, 2)
    desktop.sync()
    assert desktop.try_release('b21-w2') == 'won'
    assert desktop.try_release('b21-w2') == 'missing'
    laptop.sync()
    assert [(r['wave'], r['kind']) for r in laptop.records] == [(1, 'wave'), (2, 'released')], \
        'the release leaves a tombstone in place of the claim'
    assert laptop.records[1]['arms'] == [] and laptop.records[1]['released_iso']
    again = claims.claim_next(laptop, 'laptop', 8, read_specs=_read(laptop), log=lambda m: None)
    assert again['wave'] == 3 and again['arms'] == second['arms'], 'the released number is never reused'
    view = claims.pool_view(bus['specs'], laptop.records)
    assert [h['id'] for h in view['held']['laptop']] == ['b21-w3'], 'the tombstone is not a holding'
    # a tombstone cannot be released again, and the eval-less pool has nothing left of b21 unclaimed
    laptop.sync()
    assert laptop.try_release('b21-w2') == 'missing'


def test_a_failed_push_drops_the_local_commit_so_the_worktree_tracks_the_remote(bus, monkeypatch):
    desktop = bus['stores']['desktop']
    desktop.sync()
    monkeypatch.setattr(claims.gitbus, 'push_fast_forward', lambda worktree, branch, remote: 'failed')
    record = claims.wave_record('b21', 1, 'desktop', ['b21a-x-seed1'])
    assert desktop.try_claim(record) == 'failed'
    assert _git(['status', '--porcelain'], desktop.worktree) == ''
    assert claims.read_records(desktop.worktree) == []
    assert claims.claim_next(desktop, 'desktop', 8, read_specs=_read(desktop), log=lambda m: None) is None


def test_seed_claims_the_first_waves_as_the_scheduler_cut_them(bus, capsys):
    desktop = bus['stores']['desktop']
    assert claims._seed(desktop, 'b21', 'desktop', 2, 8) == 0
    desktop.sync()
    assert [(r['wave'], r['box'], len(r['arms'])) for r in desktop.records] == [(1, 'desktop', 8), (2, 'desktop', 8)]
    assert desktop.records[0]['arms'] == sorted(_batch('b21', 16))[:8]
    assert claims._seed(desktop, 'b21', 'laptop', 1, 8) == 0 and 'already claimed' in capsys.readouterr().out


def test_gather_reads_the_pool_from_the_refs(bus):
    desktop = bus['stores']['desktop']
    claims.claim_next(desktop, 'desktop', 8, read_specs=_read(desktop), log=lambda m: None)
    view = claims.gather(repo=desktop.repo, remote='origin', store=desktop)
    assert view['lines'] == ['b21 training | 8/16 arms', 'b18 training | 8/8 arms',
                             'desktop holds b21-w1 (8 arms)']
    assert view['heads']['claims'] == desktop.head and view['heads']['ops']
    assert view['published'] == {} and view['running'] == {}
    assert set(view['ledger']) == set(bus['specs'])
