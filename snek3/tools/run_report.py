"""Stage-A history: the eval series, its precomputed summary, and the markdown report beside it.

`runs/<name>_evals.json` is written after every training eval, so it always describes the run as it
currently stands and a finished run leaves its own documentation behind. `runs/<name>.md` is rewritten
alongside it, with the config passed in rather than transcribed, so the record cannot drift from what
ran.

**The summary block exists so a progress check is one read.** Every question a check asks — how far,
how good, is it dead — is answered here rather than recomputed from thousands of points, which also
makes "is this arm dead" one consistent definition instead of a judgement made afresh each time.
"""

import json
import os

# The share of an arm's evals at or above this perfect rate is `strong_eval_fraction`.
STRONG_EVAL_THRESHOLD = 80.0

EVAL_COLUMNS = (('step', 'step'),
                ('avg_score', 'avg score'),
                ('trailing_avg_score', 'trailing avg'),
                ('min_score', 'min score'),
                ('max_score', 'max score'),
                ('avg_reward', 'avg reward'),
                ('perfect_percent', 'perfect %'),
                ('epsilon', 'epsilon'))

# A 3M-step arm has ~3,000 evals, which is unreadable inline and bloats the file. The full series
# lives in the JSON; the table shows the opening rows and the recent ones.
HEAD_ROWS = 3
TAIL_ROWS = 12


def history_path(runs_dir, name):
    return os.path.join(runs_dir, '{0}_evals.json'.format(name))


def load_history(path):
    """`(eval_rows, resume_steps)` from earlier runs of this policy.

    The graph is rebuilt from this, so stopping and restarting an arm continues the same curve
    instead of starting a new one at the current step. A corrupt history is reported and treated as
    empty rather than raising: losing a graph is bad, losing a training run to a graph is worse.
    """
    if not os.path.exists(path):
        return [], []
    try:
        with open(path) as handle:
            saved = json.load(handle)
    except (ValueError, OSError) as error:
        print('could not read {0} ({1}); starting a fresh graph'.format(path, error))
        return [], []
    return saved.get('evals', []), saved.get('resumes', [])


def strong_eval_fraction(perfect, threshold=STRONG_EVAL_THRESHOLD):
    """Percentage of this arm's evals that reached `threshold` perfect or better.

    **The primary cross-arm metric.** It has the lowest between-seed variance of the candidates
    measured on snek2's four identical seeds — sd 5.8 pp against 8.6 for a best-30 window — and
    variance is what decides how small an effect a batch can resolve. Same data, ~40% tighter
    detectable effect for no extra compute:

    | metric | sd across 4 identical seeds | detects at n=8 |
    |---|---|---|
    | fraction of evals >= 80% | **5.8** | **7.2 pp** |
    | mean perfect over the last half | 6.3 | 7.9 pp |
    | best checkpoint, 100 episodes | 7.3 | 9.1 pp |
    | best 30-eval window | 8.6 | 10.7 pp |

    Two reasons it behaves better. A best-window figure is a **max statistic**, and maxima inflate
    variance — it reports the single luckiest window an arm ever had. And this measures *sustained*
    competence, which is closer to the goal than one good window is.

    It is a fraction of an arm's **own** evals, so it is comparable only between arms read at the
    same step horizon: the denominator grows with run length and a long tail of decline drags it
    down. It is also a *threshold-crossing* statistic, so it is not comparable across a change in
    episodes per eval — see `docs/invariants.md` invariant 8. snek3 runs 100 throughout.
    """
    if not perfect:
        return 0.0
    return round(100.0 * sum(1 for value in perfect if value >= threshold) / len(perfect), 1)


def build_summary(eval_rows, perfect_window=30, dead_window=30, dead_threshold=1.0):
    """The precomputed answers, or `{}` for an arm with no evals yet.

    **Two death fields, because one is not enough:**

    - `dead_since` — the first step of the *earliest* window of `dead_window` consecutive evals all
      below `dead_threshold`. History: this arm hit a wall at least once.
    - `zero_since` — the start of the *current* unbroken sub-threshold stretch, or None if the latest
      eval is above it. **This is the one that answers "is it dead now".**

    Neither is a verdict. snek2 arms recovered from trailing 0.3, one after ~400k steps near zero,
    and `b8d-disc995clip` carried `dead_since=275000` while going on to a 36% best-30 window. Read
    `zero_since` against `step` for the duration of the current stretch, and only call an arm dead
    after hundreds of thousands of steps.
    """
    if not eval_rows:
        return {}

    trailing = [row['trailing_avg_score'] for row in eval_rows]
    perfect = [row['perfect_percent'] for row in eval_rows]
    last = eval_rows[-1]

    best_perfect, best_perfect_step = 0.0, last['step']
    if len(perfect) >= perfect_window:
        for index in range(len(perfect) - perfect_window + 1):
            window = sum(perfect[index:index + perfect_window]) / perfect_window
            if window > best_perfect:
                best_perfect = window
                best_perfect_step = eval_rows[index + perfect_window - 1]['step']

    dead_since = None
    for index in range(len(trailing) - dead_window + 1):
        if all(value < dead_threshold for value in trailing[index:index + dead_window]):
            dead_since = eval_rows[index]['step']
            break

    zero_since = None
    for index in range(len(trailing) - 1, -1, -1):
        if trailing[index] >= dead_threshold:
            break
        zero_since = eval_rows[index]['step']

    recent = perfect[-perfect_window:]
    peak = max(trailing)
    return {
        'step': last['step'],
        'evals': len(eval_rows),
        'trailing_now': round(trailing[-1], 2),
        'peak_trailing': {'value': round(peak, 2), 'step': eval_rows[trailing.index(peak)]['step']},
        'best_perfect30': {'value': round(best_perfect, 1), 'step': best_perfect_step},
        'strong_eval_fraction': strong_eval_fraction(perfect),
        'recent_perfect30': round(sum(recent) / len(recent), 1),
        'max_single_eval': max(perfect),
        'dead_since': dead_since,
        'zero_since': zero_since,
        'epsilon': last.get('epsilon'),
    }


def merge_eval_row(eval_rows, row):
    """Adds a row in step order, replacing any existing row for the same step.

    Resuming re-evaluates at the step the previous run ended on, which would otherwise put two
    points at the same x position and draw a vertical segment in the graph.
    """
    for index, existing in enumerate(eval_rows):
        if existing['step'] == row['step']:
            eval_rows[index] = row
            break
    else:
        eval_rows.append(row)
    eval_rows.sort(key=lambda entry: entry['step'])
    return eval_rows


def save_history(path, eval_rows, resume_steps=()):
    """Writes the eval series plus its summary, atomically. Returns the summary."""
    summary = build_summary(eval_rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    staging = path + '.partial'
    with open(staging, 'w') as handle:
        json.dump({'summary': summary, 'evals': eval_rows, 'resumes': list(resume_steps)}, handle)
    os.replace(staging, path)
    return summary


def _elided(rows):
    """Head rows, an ellipsis marker, then tail rows — or everything if it is short enough."""
    if len(rows) <= HEAD_ROWS + TAIL_ROWS + 1:
        return [(row, False) for row in rows]
    return ([(row, False) for row in rows[:HEAD_ROWS]] + [(None, True)]
            + [(row, False) for row in rows[-TAIL_ROWS:]])


def write_run_report(path, name, run_config, eval_rows, graph_filename=None, resume_steps=(),
                     stage_b_rows=None):
    """Rewrites `runs/<name>.md`. Returns the summary it reports.

    `stage_b_rows`, when a wave has produced them, adds the measured region — which is what a
    hall-of-fame promotion reads, since stage B *is* the record measurement in snek3.
    """
    summary = build_summary(eval_rows)
    lines = ['# {0}'.format(name), '']

    if summary:
        lines += ['step **{0:,}** · {1} evals · trailing **{2}** · peak **{3}** @{4:,} · '
                  'sef **{5}** · best30 **{6}** @{7:,}'.format(
                      summary['step'], summary['evals'], summary['trailing_now'],
                      summary['peak_trailing']['value'], summary['peak_trailing']['step'],
                      summary['strong_eval_fraction'], summary['best_perfect30']['value'],
                      summary['best_perfect30']['step']),
                  '']
        if summary['zero_since'] is not None:
            lines += ['**Below threshold since step {0:,}** — {1:,} steps ago. Not a verdict; arms '
                      'have recovered from longer.'.format(
                          summary['zero_since'], summary['step'] - summary['zero_since']), '']
    else:
        lines += ['*No evals yet.*', '']

    if run_config:
        lines += ['## Config', '', '| | |', '|---|---|']
        lines += ['| {0} | {1} |'.format(key, value) for key, value in sorted(run_config.items())]
        lines.append('')

    if resume_steps:
        lines += ['## Resumes', '',
                  'Resumed at ' + ', '.join('{0:,}'.format(step) for step in resume_steps), '']

    if stage_b_rows:
        best = max(stage_b_rows, key=lambda row: row['perfect_percent'])
        strong = [row for row in stage_b_rows if row['perfect_percent'] >= 98.0]
        lines += ['## Stage B — {0} checkpoint(s) at {1} episodes'.format(
            len(stage_b_rows), stage_b_rows[0]['episodes']), '',
            'best **{0}%** @{1:,} (CI {2}) · **{3}** row(s) at >=98%'.format(
                best['perfect_percent'], best['step'], best['perfect_ci95'], len(strong)), '',
            '**The best row is a selected high.** A record claim needs a fresh measurement of that '
            'one checkpoint at 1,000+ episodes — snek2\'s 99.0%/500 champion re-measured at 97.5%.',
            '']

    if graph_filename:
        lines += ['![{0}]({1})'.format(name, graph_filename), '']

    lines += ['## Evals', '',
              '| ' + ' | '.join(label for _, label in EVAL_COLUMNS) + ' |',
              '|' + '---|' * len(EVAL_COLUMNS)]
    for row, elided in _elided(eval_rows):
        if elided:
            lines.append('|' + ' ... |' * len(EVAL_COLUMNS))
            continue
        lines.append('| ' + ' | '.join(str(row.get(key, '')) for key, _ in EVAL_COLUMNS) + ' |')
    lines.append('')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    return summary
