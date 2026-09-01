---
name: hof-remeasure
description: Re-measure a batch's stage-B winners at 5,000 episodes — the hof5000 pass. Use for "run the 5000 eval on b4", "re-measure the close-out winners deeply", "which checkpoints are actually good", "run hof5000 on that batch".
---

# The `hof5000` pass — 5,000 episodes on the close-out's winners

Stage B measures every screened checkpoint at 500 episodes. **A 500-episode maximum is a selected
high, not a rate** — pooled over b6, the ≥98.5 %/500 rows averaged 98.80 there and **97.86** at
5,000, a −0.94 pp regression. This pass buys back most of that.

It **feeds** [`hof-promote`](../hof-promote/SKILL.md); it does not replace its confirm step. The
episodes here are genuinely different from the selecting 500 — verified on 40 b6a checkpoints, **0
reused the same episode sequence** — but the *selection* is still the 500-ep number, so this pass's
own maximum is a selected high in turn. Promotion needs one more measurement at an unused seed.

## 1. Size it first

```
python3 - <<'EOF'
import json, glob, os
tot = 0
for f in sorted(glob.glob('runs/<batch>?-*_checkpoint_evals.json')):
    arm = os.path.basename(f).split('_checkpoint')[0]
    n = sum(1 for r in json.load(open(f))['rows'] if r['perfect_percent'] >= 98.5)
    tot += n; print(arm, n)
print('total', tot, '=', tot * 5000, 'episodes')
EOF
```

**~1,100 episodes/s across 8 shards**, so b4's 274 candidates (1.37M episodes) is ~25 min and b6's
1,299 (6.5M) took ~1 h 45 m. Anything past ~2 h belongs on the desktop instead.

## 2. The checkpoints have to be on this box, and probably are not

A batch trained on `the-claw-den` has its `runs/` JSONs here (they ride the `results` branch) and its
**weights only there** — `savedPolicies/` is not on the git bus. Check `ls -d savedPolicies/<arm>`.

**Pull only the candidates.** b4's arms hold 12,200 checkpoints each; the pass needs 17-64 of them, so
this is 1.8 MB per arm instead of 1.3 GB:

```
python3 - <<'EOF'      # one file list per arm
import json, glob, os
for f in sorted(glob.glob('runs/<batch>?-*_checkpoint_evals.json')):
    arm = os.path.basename(f).split('_checkpoint')[0]
    rows = json.load(open(f))['rows']
    with open('/tmp/files-%s.txt' % arm, 'w') as out:
        out.write('arch.json\n')
        for s in sorted({r['step'] for r in rows if r['perfect_percent'] >= 98.5}):
            out.write('ckpt-%d.pt\n' % s)
EOF

mkdir -p savedPolicies/<local-arm>
rsync -a --files-from=/tmp/files-<local-arm>.txt \
  the-claw-den:Snek/snek3/savedPolicies/<remote-arm>/ savedPolicies/<local-arm>/
```

- **‡ The box may hold the arm under a different name.** Batches were renamed p0-p3 → b3-b6 on the
  laptop 2026-08-31 and the desktop was not redeployed, so **b4 is `p1` there** and b5 is `p2`. Map
  the letter, not the whole name: `b4c` ← `p1c`. `ls -d savedPolicies/*<config-fragment>*` on the box
  finds them.
- **`--files-from` is the form that works.** Multi-source rsync over ssh takes neither `'host:a b'`
  nor `'host:{a,b}'` — both fail as one absurd path.

Then confirm the selector agrees with what landed, **before** launching. `resolve` raises on a step it
cannot find rather than quietly measuring fewer, so this is the whole safety net:

```
PYTHONPATH=. python3 -c "
import glob, os
from tools import step_selectors as sel
for f in sorted(glob.glob('runs/<batch>?-*_checkpoint_evals.json')):
    arm = os.path.basename(f).split('_checkpoint')[0]
    steps, desc = sel.resolve(os.path.join('savedPolicies', arm), 'above:98.5', arm)
    print(arm, len(steps), desc)"
```

Its signature is `resolve(policy_dir, token, policy)` — three arguments, `policy_dir` first.

## 3. Launch

```
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -u -m tools.closeout <arm...> \
    --selector above:98.5 --episodes 5000 --label hof5000 --shards 8 \
    > logs/<batch>-hof5000.log 2>&1 &
```

**‡ `--label hof5000` is not cosmetic and omitting it destroys the input.** The output path is
`runs/<arm>_checkpoint_evals[_<label>].json`, so with no label this pass overwrites the 500-episode
close-out — which is the file `above:98.5` reads. You would lose the selection, the comparison
baseline and any chance of repeating the pass, and the wreckage looks like a normal short result file.

Call the env python directly, never `conda run` — it buffers a backgrounded log for 90+ seconds.

## 4. Watch it

The controller prints `[done/total] N shard(s) alive` per arm and opens its own eval-chart window
(second slot, so a training's window is untouched). Per-shard truth is
`logs/<arm>_checkpoint_evals_hof5000-s<i>of<n>.log`.

**A killed pass loses nothing.** Each shard rewrites its own file after every completed measurement,
and the identical command resumes every shard where it stopped.

## 5. Reading the result

```
PYTHONPATH=. python3 -c "
import json, statistics as st
h = json.load(open('runs/<arm>_checkpoint_evals_hof5000.json'))
r = sorted(h['rows'], key=lambda x: -x['perfect_percent'])
print('n', len(r), 'best', r[0]['perfect_percent'], '@', r[0]['step'], r[0]['perfect_ci95'])
print('mean', round(st.mean(x['perfect_percent'] for x in r), 2))"
```

Sanity check the merge: **merged rows must equal the shard sum and equal the ≥98.5 %/500 count.** All
sixteen b5/b6 arms satisfied both, which is what made those passes trustworthy.

Then judge a candidate by its **basin**, not its peak — the mean of its neighbours within ±1M
transitions predicts its true rate better than its own score does. [`hof-promote`](../hof-promote/SKILL.md)
has the method and the numbers.
