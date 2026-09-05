---
name: hof-remeasure
description: Re-measure a batch's stage-B winners at 5,000 episodes — the hof5000 pass. Use for "run the 5000 eval on b4", "re-measure the close-out winners deeply", "which checkpoints are actually good", "run hof5000 on that batch".
---

# The `hof5000` pass — 5,000 episodes on the close-out's winners

**Since 2026-09-04 this pass and the `hof30k` after it run by themselves**, on both boxes, after every
wave's stage B — the desktop daemon and `tools.laptop_batch` chain `tools.closeout <arms> --pass
hof5000` then `--pass hof30k`, and a batch that finished under the chain needs nothing from this skill.
Use it for a batch measured before that date, a driver started before it, or a re-run; `--pass hof5000`
is the shorthand for the `--selector above:99 --episodes 5000 --label hof5000` below and the
desktop's own spec is `tools/closeout.py`'s `PASSES`.

Stage B measures every screened checkpoint at 500 episodes. **A 500-episode maximum is a selected
high, not a rate** — pooled over b6, the ≥98.5 %/500 rows averaged 98.80 there and **97.86** at
5,000, a −0.94 pp regression. This pass buys back most of that.

**The candidate cut is ≥99 %/500** (raised from 98.5 on 2026-09-02, user's decision). At b9's
density a 98.5 cut was 2,289 candidates against 727 at 99, and with the usual −1 pp regression a
98.5/500 row almost never reaches the 98.73 record region at 5,000 — b7's 766 candidates yielded 6.

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
    n = sum(1 for r in json.load(open(f))['rows'] if r['perfect_percent'] >= 99)
    tot += n; print(arm, n)
print('total', tot, '=', tot * 5000, 'episodes')
EOF
```

**~1,100 episodes/s across 8 shards** (the laptop default until 2026-09-04; 12 is expected to be
~25-40% faster and has not been timed yet), so b4's 274 candidates (1.37M episodes) was ~25 min at 8
and b6's 1,299 (6.5M) took ~1 h 45 m. Anything past ~2 h belongs on the desktop instead.

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
        for s in sorted({r['step'] for r in rows if r["perfect_percent"] >= 99}):
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
    steps, desc = sel.resolve(os.path.join('savedPolicies', arm), 'above:99', arm)
    print(arm, len(steps), desc)"
```

Its signature is `resolve(policy_dir, token, policy)` — three arguments, `policy_dir` first.

## 3. Launch

```
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -u -m tools.closeout <arm...> \
    --selector above:99 --episodes 5000 --label hof5000 --shards 12 \
    > logs/<batch>-hof5000.log 2>&1 &
```

**‡ `--label hof5000` is not cosmetic and omitting it destroys the input.** The output path is
`runs/<arm>_checkpoint_evals[_<label>].json`, so with no label this pass overwrites the 500-episode
close-out — which is the file `above:99` reads. You would lose the selection, the comparison
baseline and any chance of repeating the pass, and the wreckage looks like a normal short result file.

Call the env python directly, never `conda run` — it buffers a backgrounded log for 90+ seconds.

**‡ Do not pass the arm list through a zsh variable.** `ARMS="a b c"; ... closeout $ARMS` hands the
controller **one** policy named `a b c` (zsh does not word-split a parameter), and it exits status 1 in
0.0 min having opened a window for a nonexistent arm. Type the names, or use `$(cat list.txt)` — an
unquoted command substitution *is* split. Cost one relaunch on b9, 2026-09-02.

## 4. Watch it — and make sure the window is actually up

The controller prints `[done/total] N shard(s) alive` per arm and asks for the stage-B chart window
(second slot, so a training's window is untouched). **Asking is not having: always confirm the window
opened, and open it yourself if it did not:**

```
ps -Ao pid=,etime=,command= | grep '[c]hart_viewer' | grep -v 'zsh -c'      # expect one line titled stage B
PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -m tools.eval_window <arm...> --label hof5000 \
    --watch-pid <close-out pid> > logs/<batch>-hof5000-window.log 2>&1 &   # only if the ps shows none;
                                                    # without --watch-pid the window never closes itself
```

The window is one per box by an `flock`, so a pass launched while a *stale* window holds the lock draws
nothing, and killing the stale one afterwards leaves the pass with no window at all — which is exactly
what happened on b9 (the mislaunch above had opened one). Opening it by hand is free: nothing reads it
or waits on it, and `--label` must match the pass's label or it watches the wrong files.

Per-shard truth is `logs/<arm>_checkpoint_evals_hof5000-s<i>of<n>.log`.

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

Sanity check the merge: **merged rows must equal the shard sum and equal the ≥99 %/500 count.** All
sixteen b5/b6 arms satisfied both, which is what made those passes trustworthy.

Then judge a candidate by its **basin**, not its peak — the mean of its neighbours within ±1M
transitions predicts its true rate better than its own score does. [`hof-promote`](../hof-promote/SKILL.md)
has the method and the numbers.
