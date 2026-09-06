"""Where a batch stands across both boxes, from the shared queue's view: which box holds each wave, what
is trained, what is measured, what is unclaimed.

    PYTHONPATH=. python -m tools.batch_state b21          # one line per wave, and the pool's share

The view is `tools.claims.gather()` -- the specs on `ops`, the claims, both feeds' published ids, both
statuses' running ids -- so this reads the same facts the daemon publishes and the scheduler acts on.
A batch may span both boxes wave by wave (`plans/archive/shared-queue.md`), so the unit here is the wave: the
box that claimed it, its arms at cap (on a feed), its arms live (in a status), and its three passes
(each on a feed, running, or owed). `tools/progress_update.py` reads this for its state lines and its
counts; the old per-box readings (`batch_state` off the desktop ledger, `laptop_state_line` off the
laptop's `runs/`) are gone with it.
"""

import collections
import sys

from desktop.daemon.daemon import batch_of
from tools import claims

PASSES = ('stageb', 'hof5000', 'hof30k')


def pass_id(batch, pass_name, wave):
    return '{0}-{1}{2}'.format(batch, pass_name, '' if int(wave) == 1 else '-w{0}'.format(int(wave)))


def batch(view, name):
    """`{'arms': [ids], 'waves': [...], 'unclaimed': [ids], 'evals': [...]}` for one batch.

    Each wave: `number`, `box`, `arms`, `trained` (ids at their cap, on a feed), `live` (ids a status
    lists as running), `passes` (`{name: 'done' | 'running' | 'owed'}`) and `state`: `closed` (every arm
    trained and every pass done), `measuring` (arms trained, a pass owed or running), `training`.
    """
    specs = view.get('specs') or {}
    published = set(view.get('published') or {})
    running = view.get('running') or {}
    arms = sorted(spec_id for spec_id, spec in specs.items()
                  if batch_of(spec_id) == name and spec.get('type', 'train') == 'train')
    records = [r for r in view.get('records') or [] if r.get('batch') == name]
    taken = claims.claimed_ids(records)
    waves = []
    for record in sorted((r for r in records if r.get('kind', 'wave') == 'wave'), key=lambda r: r['wave']):
        wave_arms = list(record['arms'])
        trained = [a for a in wave_arms if a in published]
        live = [a for a in wave_arms if a in running]
        passes = {}
        for pass_name in PASSES:
            pid = pass_id(name, pass_name, record['wave'])
            passes[pass_name] = 'done' if pid in published else ('running' if pid in running else 'owed')
        if len(trained) == len(wave_arms) and all(state == 'done' for state in passes.values()):
            state = 'closed'
        elif len(trained) == len(wave_arms):
            state = 'measuring'
        else:
            state = 'training'
        waves.append({'number': int(record['wave']), 'box': record['box'], 'arms': wave_arms, 'trained': trained,
                      'live': live, 'passes': passes, 'state': state, 'claimed_iso': record.get('claimed_iso')})
    evals = [{'id': r['id'], 'box': r['box'], 'done': r['id'] in published, 'running': r['id'] in running}
             for r in records if r.get('kind') == 'eval']
    return {'arms': arms, 'waves': waves, 'unclaimed': [a for a in arms if a not in taken], 'evals': evals}


def counts(view, name):
    """`(arms, waves)` Counters over `done` / `running` / `queued`, the shape `tools/progress_update.py`
    tables and estimates from. An unclaimed arm is `queued`; a wave is `done` when its stage B is."""
    state = batch(view, name)
    arms, waves = collections.Counter(), collections.Counter()
    for wave in state['waves']:
        for arm in wave['arms']:
            arms['done' if arm in wave['trained'] else ('running' if arm in wave['live'] else 'queued')] += 1
        stage_b = wave['passes']['stageb']
        waves[stage_b if stage_b in ('done', 'running') else 'queued'] += 1
    arms['queued'] += len(state['unclaimed'])
    return arms, waves


def closed(view, name):
    """Every arm claimed and trained, every wave's passes done, nothing unclaimed: the batch is finished."""
    state = batch(view, name)
    return bool(state['waves']) and not state['unclaimed'] and all(w['state'] == 'closed' for w in state['waves'])


def line(view, name, percent=None):
    """One line: `b21: w1 desktop closed, w2 laptop training (3 of 8 at cap), w3 unclaimed (8 arms)`.

    `percent` maps an arm id to its training percent (from a status's `step`/`max_steps`), shown as the
    mean over a wave's live arms when given. `Not on ops` when the batch has no spec there.
    """
    state = batch(view, name)
    if not state['arms'] and not state['waves']:
        return '{0}: not on ops (no spec); nothing to read from the shared queue'.format(name)
    parts = []
    for wave in state['waves']:
        if wave['state'] == 'closed':
            text = 'closed'
        elif wave['state'] == 'measuring':
            running = [p for p, s in wave['passes'].items() if s == 'running']
            owed = [p for p, s in wave['passes'].items() if s == 'owed']
            text = ('{0} running'.format(running[0]) if running else
                    '{0} owed'.format(', '.join(owed)) if owed else 'measuring')
        else:
            text = 'training ({0} of {1} at cap'.format(len(wave['trained']), len(wave['arms']))
            if percent:
                shown = [percent[a] for a in wave['live'] if a in percent]
                if shown:
                    text += ', live at {0}%'.format(int(round(sum(shown) / len(shown))))
            text += ')'
        parts.append('w{0} {1} {2}'.format(wave['number'], wave['box'], text))
    if state['unclaimed']:
        parts.append('{0} arm{1} unclaimed'.format(len(state['unclaimed']), '' if len(state['unclaimed']) == 1 else 's'))
    for ev in state['evals']:
        parts.append('eval {0} {1} {2}'.format(ev['id'], ev['box'], 'done' if ev['done'] else ('running' if ev['running'] else 'owed')))
    verdict = 'Closed' if closed(view, name) else 'In flight'
    return '{0}: {1}'.format(verdict, ', '.join(parts))


def live_batches(view):
    """Batches with anything still to do: an unclaimed arm, a wave not closed, an eval owed."""
    names = {batch_of(spec_id) for spec_id in (view.get('specs') or {})}
    names |= {r['batch'] for r in view.get('records') or []}
    return sorted((name for name in names if not closed(view, name)),
                  key=lambda b: int(''.join(ch for ch in b if ch.isdigit()) or 0))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: python -m tools.batch_state <batch>...')
        return 2
    view = claims.gather()
    for name in argv:
        print(line(view, name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
