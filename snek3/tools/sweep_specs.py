"""Expand one batch of `plans/hyperparam-sweep.json` into desktop job specs.

    PYTHONPATH=. python -m tools.sweep_specs b9                       # dry run: print what would be written
    PYTHONPATH=. python -m tools.sweep_specs b9 --out "$OPS/snek3/desktop/queue/pending"
    PYTHONPATH=. python -m tools.sweep_specs b9 --smoke smoke-b9.sh   # the laptop smoke script for its never-exercised cells

One spec per arm, every cell at every seed, ids in the project's shape — `b9a-lam0-seed1` — with
the letter running over arms as b7 and b8 did (single letters up to 26 arms, two above). Each wave of
`wave_size` arms gets its own priority so the box takes the batch in the manifest's order, and every
spec is validated against the daemon's own `parse_job` before it is written, because a malformed spec
is skipped silently on the box and would not tell you until after the push.

**This writes files and nothing else.** The push to `ops` is the `desktop-batch` skill's step 2 and
needs the user's approval for that batch. Cells whose knob does not exist yet (`requires_code`) are
dropped with a note unless `--include-uncoded` is given, so a batch cannot be queued with an arm the
trainer would refuse by name.
"""

import argparse
import json
import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, 'plans', 'hyperparam-sweep.json')


def arm_letters(n_arms):
    """`a`..`z` for up to 26 arms, `aa`, `ab`, ... above — b8's and b7's schemes respectively."""
    letters = string.ascii_lowercase
    if n_arms <= 26:
        return list(letters[:n_arms])
    out = []
    for i in range(n_arms):
        out.append(letters[i // 26] + letters[i % 26])
    return out


def select_cells(batch, include_uncoded):
    kept, dropped = [], []
    for c in batch['cells']:
        if c.get('requires_code') and not include_uncoded:
            dropped.append(c['slug'])
        else:
            kept.append(c)
    return kept, dropped


def build_specs(manifest, batch, cells):
    seeds = manifest['seeds']
    wave = int(manifest.get('wave_size', 8))
    base = manifest['base']
    n_arms = len(cells) * len(seeds)
    letters = arm_letters(n_arms)
    n_waves = max(1, -(-n_arms // wave))
    specs = []
    for ci, cell in enumerate(cells):
        for si, seed in enumerate(seeds):
            index = ci * len(seeds) + si
            wave_no = index // wave + 1
            env = dict(base['env'])
            env.update(cell['env'])
            env['SNEK_SEED'] = str(seed)
            arm_id = '{0}{1}-{2}-seed{3}'.format(batch['batch'], letters[index], cell['slug'], seed)
            notes = ('Batch {b}: the {knob} sweep, one knob off the frozen base (b7\'s fc (320,) reference: '
                     'PPO defaults, b2 reward, 4 epochs), seeds 1-4 pinned to the arm letter, at b7\'s 50M cap '
                     'so b7aa-b7ad are the control. Design: plans/hyperparam-sweep.md. '
                     'This cell: {cell} -> {env}. Prediction: {pred}'
                     ).format(b=batch['batch'], knob=batch['knob'], cell=cell['slug'],
                              env=json.dumps(cell['env'], sort_keys=True), pred=cell.get('prediction', ''))
            if batch.get('notes'):
                notes += ' Batch note: ' + batch['notes']
            specs.append({
                'project': base.get('project', 'snek3'),
                'id': arm_id,
                'type': base.get('type', 'train'),
                'policy': arm_id,
                'max_steps': int(manifest['horizon']['max_steps']),
                'env': env,
                'priority': int(batch['priority']) + wave_no - 1,
                'label': '{0}: {1}, seed {2} of {3} -- wave {4} of {5}'.format(
                    batch['batch'], cell['slug'], seed, len(seeds), wave_no, n_waves),
                'notes': notes,
            })
    return specs


def validate(specs):
    """Run each spec through the daemon's parser; returns a list of error strings."""
    sys.path.insert(0, os.path.join(ROOT, 'desktop'))
    from runner.job import parse_job, JobError  # stdlib-only, runs on the laptop
    errors = []
    for spec in specs:
        try:
            parse_job(json.dumps(spec), source=spec['id'])
        except JobError as e:
            errors.append('{0}: {1}'.format(spec['id'], e))
    return errors


def smoke_script(manifest, batch, cells):
    """A shell script that runs each `smoke: true` cell for a few rollouts on the laptop.

    Every never-exercised value gets one before its batch is pushed, because a knob the trainer
    silently ignores costs four arms. Each run prints its `hyperparameter override:` lines and its
    first eval; a ramp is read at a known fraction of the short cap. Policy name `smoke`, per
    `CLAUDE.md`, and the directory is removed between runs so each starts clean.
    """
    base = manifest['base']['env']
    cap = int(manifest['smoke']['max_steps'])
    lines = ['#!/bin/zsh', '# smoke runs for {0}: {1}. Generated by tools/sweep_specs.py; run from snek3/ in the snek3 env.'.format(
        batch['batch'], batch['knob']), 'set -u', 'setopt null_glob',
        '[ -f train.py ] || { echo "run this from snek3/"; exit 1; }', '']
    for cell in cells:
        if not cell.get('smoke'):
            continue
        env = dict(base)
        env.update(cell['env'])
        env.update({'SNEK_SEED': '1', 'SNEK_MAX_STEPS': str(cap), 'SNEK_CHART_WINDOW': '0',
                    'SNEK_EVAL_QUEUE': '0', 'SNEK_MIN_CHECKPOINT_SCORE': '0'})
        assigns = ' '.join('{0}={1}'.format(k, v) for k, v in sorted(env.items()))
        lines += ["echo '=== {0}: {1} -> {2}'".format(batch['batch'], cell['slug'], json.dumps(cell['env'], sort_keys=True)),
                  'rm -rf savedPolicies/smoke runs/smoke*',
                  'env {0} PYTHONPATH=. /opt/miniconda3/envs/snek3/bin/python -u train.py smoke 2>&1 | '
                  'grep -E "hyperparameter override:|reward config:|Traceback|Error|perfect|step" | head -40'.format(assigns),
                  '']
    lines.append('rm -rf savedPolicies/smoke runs/smoke*')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('batch', help='b9, b10, ... as named in the manifest')
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--out', default=None, help='directory to write specs into; omit for a dry run')
    parser.add_argument('--smoke', default=None, metavar='FILE.sh',
                        help='write the laptop smoke script for this batch\'s never-exercised cells and exit')
    parser.add_argument('--include-uncoded', action='store_true',
                        help='also emit cells marked requires_code (only once the knob exists)')
    args = parser.parse_args(argv)

    with open(args.manifest) as f:
        manifest = json.load(f)
    batches = {b['batch']: b for b in manifest['batches']}
    if args.batch not in batches:
        sys.exit('no batch {0} in {1}; have {2}'.format(args.batch, args.manifest, sorted(batches)))
    batch = batches[args.batch]

    cells, dropped = select_cells(batch, args.include_uncoded)
    if args.smoke:
        with open(args.smoke, 'w') as f:
            f.write(smoke_script(manifest, batch, cells))
        n = sum(1 for c in cells if c.get('smoke'))
        print('wrote {0} smoke run(s) for {1} to {2}'.format(n, batch['batch'], args.smoke))
        return
    specs = build_specs(manifest, batch, cells)
    errors = validate(specs)

    wave = int(manifest.get('wave_size', 8))
    print('{0}: {1} cells x {2} seeds = {3} arms, {4} wave(s) of {5}, priorities {6}-{7}'.format(
        batch['batch'], len(cells), len(manifest['seeds']), len(specs),
        -(-len(specs) // wave), wave, batch['priority'], batch['priority'] + -(-len(specs) // wave) - 1))
    if len(specs) % wave:
        print('WARNING: {0} arms is not a multiple of {1}; the last wave straddles nothing but runs '
              'short'.format(len(specs), wave))
    if dropped:
        print('dropped (requires_code, pass --include-uncoded once the knob exists): ' + ', '.join(dropped))
    for spec in specs:
        print('  {0:<28} p{1:<4} {2}'.format(spec['id'], spec['priority'], spec['label']))
    if errors:
        print('\n'.join(['VALIDATION FAILED:'] + errors))
        sys.exit(1)

    if args.out is None:
        print('dry run; pass --out <dir> to write')
        return
    os.makedirs(args.out, exist_ok=True)
    for spec in specs:
        path = os.path.join(args.out, spec['id'] + '.json')
        with open(path, 'w') as f:
            json.dump(spec, f, indent=2)
            f.write('\n')
    print('wrote {0} specs to {1}'.format(len(specs), args.out))


if __name__ == '__main__':
    main()
