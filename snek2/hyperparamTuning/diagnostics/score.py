"""Ranks candidate observations by how often they would have flagged the losing move."""
import collections
import glob
import json
import sys

shards = [json.load(open(path)) for path in sorted(glob.glob(sys.argv[1]))]
blames = [b for shard in shards for b in shard['blames']]
outcomes = collections.Counter()
saturation = collections.Counter()
for shard in shards:
    outcomes.update(shard['outcomes'])
    saturation.update(shard['saturation'])

print('shards {0}, outcomes {1}'.format(len(shards), dict(outcomes)))
print('attributed losses: {0}'.format(len(blames)))
late = saturation['late_steps']
print('late steps (len>=50): {0}; some move keeps the tail reachable — '
      'static {1:.1f}%, time-aware {2:.1f}%'.format(
          late, 100.0 * saturation['late_static_tail_any'] / max(late, 1),
          100.0 * saturation['late_timed_tail_any'] / max(late, 1)))

NUMERIC = ('timed_area', 'timed_depth', 'static_area', 'degree', 'corridor')
BOOLEAN = ('timed_tail', 'static_tail')

usable = [b for b in blames if b.get('chosen')]
print('\ncomparable decisions (chosen move was legal): {0}\n'.format(len(usable)))
print('{0:14s} {1:>7s} {2:>7s} {3:>7s}   {4}'.format(
    'candidate', 'right', 'tie', 'wrong', 'median chosen vs survivor'))
rows = []
for name in NUMERIC + BOOLEAN:
    right = tie = wrong = 0
    chosen_values = []
    survivor_values = []
    for blame in usable:
        mine = blame['chosen'][name]
        # Best over surviving branches: the feature only has to rank one of them above.
        theirs = max(survivor[name] for survivor in blame['survivors'])
        chosen_values.append(int(mine))
        survivor_values.append(int(theirs))
        if theirs > mine:
            right += 1
        elif theirs == mine:
            tie += 1
        else:
            wrong += 1
    chosen_values.sort()
    survivor_values.sort()
    rows.append((right, name, tie, wrong,
                 chosen_values[len(chosen_values) // 2],
                 survivor_values[len(survivor_values) // 2]))
for right, name, tie, wrong, med_chosen, med_survivor in sorted(rows, reverse=True):
    total = right + tie + wrong
    print('{0:14s} {1:6.1f}% {2:6.1f}% {3:6.1f}%   {4} vs {5}'.format(
        name, 100.0 * right / total, 100.0 * tie / total, 100.0 * wrong / total,
        med_chosen, med_survivor))

print('\n-- combinations, as a policy would use them --')


def rank_right(key):
    right = tie = wrong = 0
    for blame in usable:
        mine = key(blame['chosen'])
        theirs = max(key(survivor) for survivor in blame['survivors'])
        if theirs > mine:
            right += 1
        elif theirs == mine:
            tie += 1
        else:
            wrong += 1
    return right, tie, wrong


for label, key in (
        ('timed_tail then timed_area', lambda f: (f['timed_tail'], f['timed_area'])),
        ('timed_area alone', lambda f: f['timed_area']),
        ('timed_depth then timed_area', lambda f: (f['timed_depth'], f['timed_area'])),
        ('static_tail then static_area', lambda f: (f['static_tail'], f['static_area'])),
        ('timed_area >= len flag', lambda f: f['timed_area'] >= f['len']),
):
    right, tie, wrong = rank_right(key)
    total = right + tie + wrong
    print('  {0:30s} right {1:5.1f}%  tie {2:5.1f}%  wrong {3:5.1f}%'.format(
        label, 100.0 * right / total, 100.0 * tie / total, 100.0 * wrong / total))

print('\n-- where the losing branch is a trap: timed_area of chosen vs survivor --')
buckets = collections.Counter()
for blame in usable:
    mine = blame['chosen']['timed_area']
    theirs = max(survivor['timed_area'] for survivor in blame['survivors'])
    ratio = 'survivor >=2x' if theirs >= 2 * max(mine, 1) else (
        'survivor bigger' if theirs > mine else 'equal' if theirs == mine else 'chosen bigger')
    buckets[ratio] += 1
for key, count in buckets.most_common():
    print('  {0:18s} {1:3d}  {2:5.1f}%'.format(key, count, 100.0 * count / len(usable)))

leads = sorted(b['lead'] for b in blames)
print('\nlead times: median {0}, p90 {1}, max {2}'.format(
    leads[len(leads) // 2], leads[int(len(leads) * 0.9)], leads[-1]))
lens = sorted(b['len'] for b in blames)
print('length at the fatal decision: median {0}, range {1}-{2}'.format(
    lens[len(lens) // 2], lens[0], lens[-1]))
