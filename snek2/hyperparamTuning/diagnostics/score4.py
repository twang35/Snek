"""Scores the five tail-test variants on the losing decisions."""
import collections, glob, json, sys
VARIANTS = ('observed', 'holding', 'newtail', 'both', 'timed')
shards = [json.load(open(p)) for p in sorted(glob.glob(sys.argv[1]))]
c = collections.Counter(); o = collections.Counter(); blames = []
for s in shards:
    c.update(s['counters']); o.update(s['outcomes']); blames += s['blames']
print('outcomes', dict(o))
print('steps {0}, legal actions {1}, observed_mismatch {2}'.format(
    c['steps'], c['legal_actions'], c.get('observed_mismatch', 0)))
print()
print('== over every legal action ==')
print('{0:9s} {1:>10s} {2:>18s}'.format('variant', 'says yes', 'differs from timed'))
for n in VARIANTS:
    print('{0:9s} {1:10d} {2:18d}'.format(n, c.get(n + '_true', 0),
                                          c.get(n + '_vs_timed_differ', 0)))
print('holding vs newtail differ: {0}  (newtail-only {1}, holding-only {2})'.format(
    c.get('holding_vs_newtail_differ', 0), c.get('newtail_true_holding_false', 0),
    c.get('holding_true_newtail_false', 0)))
print()
usable = [b for b in blames if b['chosen']['legal']]
print('== at the {0} losing decisions with a legal chosen move =='.format(len(usable)))
print('{0:9s} {1:>8s} {2:>8s} {3:>10s} {4:>11s} {5:>10s}'.format(
    'variant', 'flags', 'flags%', 'both true', 'both false', 'wrong way'))
for n in VARIANTS:
    right = sum(1 for b in usable if any(s[n] for s in b['survivors']) and not b['chosen'][n])
    both = sum(1 for b in usable if b['chosen'][n] and any(s[n] for s in b['survivors']))
    neither = sum(1 for b in usable if not b['chosen'][n]
                  and not any(s[n] for s in b['survivors']))
    wrong = sum(1 for b in usable if b['chosen'][n] and not any(s[n] for s in b['survivors']))
    print('{0:9s} {1:8d} {2:7.1f}% {3:10d} {4:11d} {5:10d}'.format(
        n, right, 100.0 * right / len(usable), both, neither, wrong))
print()
pairs = collections.Counter()
for b in usable:
    h = (b['chosen']['holding'], any(s['holding'] for s in b['survivors']))
    t = (b['chosen']['newtail'], any(s['newtail'] for s in b['survivors']))
    if h != t:
        pairs[(h, t)] += 1
print('decisions where holding and newtail disagree: {0} {1}'.format(sum(pairs.values()),
                                                                    dict(pairs)))
