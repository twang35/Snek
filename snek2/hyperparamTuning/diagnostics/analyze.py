"""Merges the diag.py shards into the numbers the recommendation rests on."""
import collections
import glob
import json
import sys

shards = [json.load(open(path)) for path in sorted(glob.glob(sys.argv[1]))]
print('shards: {0}'.format(len(shards)))

outcomes = [record for shard in shards for record in shard['outcomes']]
counters = collections.Counter()
choice = collections.Counter()
for shard in shards:
    counters.update(shard['counters'])
    choice.update(shard['choice'])

kinds = collections.Counter(record['outcome'] for record in outcomes)
print('\n== episodes: {0} =='.format(len(outcomes)))
for kind, count in kinds.most_common():
    print('  {0:10s} {1:4d}  {2:5.1f}%'.format(kind, count, 100.0 * count / len(outcomes)))

losses = [record for record in outcomes if record['outcome'] != 'perfect']
bands = collections.Counter()
for record in losses:
    score = record['score']
    band = ('<50' if score < 50 else '50-79' if score < 80 else
            '80-89' if score < 90 else '90-94')
    bands[(record['outcome'], band)] += 1
print('\n== where losses happen (score at the end) ==')
for (kind, band), count in sorted(bands.items()):
    print('  {0:10s} {1:6s} {2:3d}'.format(kind, band, count))

print('\n== steps ==')
steps = counters['steps']
for key in ('steps', 'steps_late', 'choices', 'choices_area_spread5',
            'choices_tail_differs', 'food_sealed', 'food_sealed_late'):
    base = counters['steps_late'] if key.endswith('_late') else steps
    print('  {0:22s} {1:8d}  {2:5.1f}% of {3}'.format(
        key, counters[key], 100.0 * counters[key] / max(base, 1),
        'late steps' if key.endswith('_late') else 'all steps'))

print('\n== which move the policy takes when the free area differs by >=5 ==')
total_choice = sum(choice.values())
for key, count in choice.most_common():
    print('  {0:24s} {1:6d}  {2:5.1f}%'.format(key, count, 100.0 * count / max(total_choice, 1)))

print('\n== failure attribution ({0} losses) =='.format(len(losses)))
blamed = [record['blame'] for record in losses if record.get('blame')]
print('  pinpointed inside the last 40 decisions: {0} of {1}'.format(len(blamed), len(losses)))
print('  no survivable branch in that window:     {0}'.format(len(losses) - len(blamed)))
final_safe = collections.Counter(record['final_safe_moves'] for record in losses)
print('  safe moves available on the final step:  ' +
      ', '.join('{0}: {1}'.format(k, v) for k, v in sorted(final_safe.items())))

if blamed:
    leads = sorted(b['lead'] for b in blamed)
    print('\n  how many steps before the end the mistake was made:')
    print('    min {0}, median {1}, p90 {2}, max {3}'.format(
        leads[0], leads[len(leads) // 2], leads[int(len(leads) * 0.9)], leads[-1]))
    print('    lead 0 (final decision): {0}, 1-3: {1}, 4-10: {2}, 11+: {3}'.format(
        sum(1 for x in leads if x == 0), sum(1 for x in leads if 1 <= x <= 3),
        sum(1 for x in leads if 4 <= x <= 10), sum(1 for x in leads if x > 10)))

    print('\n  what distinguished the surviving move from the chosen one:')
    signals = collections.Counter()
    for blame in blamed:
        if not blame['chosen_safe']:
            signals['chosen move was already flagged unsafe'] += 1
            continue
        if blame['chosen_tail'] != blame['alt_tail']:
            signals['tail-reachable flag differed (already observed)'] += 1
        elif blame['alt_area'] - blame['chosen_area'] >= 5:
            signals['free area differed by >=5 (not observed)'] += 1
        elif blame['alt_area'] != blame['chosen_area']:
            signals['free area differed by <5 (not observed)'] += 1
        elif blame['chosen_regions'] != blame['alt_regions']:
            signals['region count differed (already observed)'] += 1
        elif blame['chosen_degree'] != blame['alt_degree']:
            signals['landing-cell degree differed (not observed)'] += 1
        else:
            signals['nothing among these differed'] += 1
    for key, count in signals.most_common():
        print('    {0:48s} {1:3d}  {2:5.1f}%'.format(key, count,
                                                     100.0 * count / len(blamed)))
    lens = sorted(b['len'] for b in blamed)
    print('\n  snake length at the fatal decision: median {0}, range {1}-{2}'.format(
        lens[len(lens) // 2], lens[0], lens[-1]))

print('\n== observation aliasing (per shard, so the spread across shards shows stability) ==')
for name in ('alias', 'geom_alias'):
    print('  {0}:'.format(name))
    for field in ('classes', 'repeated_steps', 'len_spread_median', 'len_spread_p90',
                  'len_spread_max', 'area_spread_median', 'area_spread_p90'):
        values = [shard[name][field] for shard in shards]
        print('    {0:20s} {1}'.format(field, values))
    totals = sum(shard[name]['steps'] for shard in shards)
    classes = sum(shard[name]['classes'] for shard in shards)
    repeated = sum(shard[name]['repeated_steps'] for shard in shards)
    print('    {0:20s} {1} steps, {2} classes, {3:.1f}% of steps share a vector'.format(
        'totals', totals, classes, 100.0 * repeated / max(totals, 1)))
