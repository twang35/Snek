"""Writes a markdown summary of a training run beside its progress graph.

Rewritten on every eval, so runs/<policy_name>.md always describes the run as it
currently stands and a finished run leaves its own documentation behind. Config
values are passed in from snek2.py rather than transcribed by hand, so they can't
drift away from what actually ran.
"""
import json
import os

EVAL_COLUMNS = [
    ('step', 'step'),
    ('avg_score', 'avg score'),
    ('trailing_avg_score', 'trailing avg'),
    ('min_score', 'min score'),
    ('max_score', 'max score'),
    ('avg_reward', 'avg reward'),
    ('perfect_percent', 'perfect %'),
    ('epsilon', 'epsilon'),
]

# A million-step run has a thousand evals, which is unreadable inline and bloats
# the file. The full series lives in <policy>_evals.json; the table shows the
# opening rows and the recent ones.
HEAD_ROWS = 3
TAIL_ROWS = 12


def history_path(runs_dir, policy_name):
    return os.path.join(runs_dir, '{0}_evals.json'.format(policy_name))


def load_history(path):
    """Returns (eval_rows, resume_steps) from earlier runs of this policy.

    The graph is rebuilt from this, so stopping and restarting a policy continues
    the same curve instead of starting a new one at the current iteration.
    """
    if not os.path.exists(path):
        return [], []
    try:
        with open(path) as handle:
            saved = json.load(handle)
    except (ValueError, OSError) as error:
        # A corrupt history shouldn't stop training; the graph just restarts.
        print('could not read graph history {0} ({1}), starting a fresh graph'.format(path, error))
        return [], []
    return saved.get('evals', []), saved.get('resumes', [])


def build_summary(eval_rows, perfect_window=30, dead_window=30, dead_threshold=1.0):
    """Precomputed answers to the questions every progress check asks.

    These were previously recalculated by hand from the full eval series each time an arm
    was checked on — peak, best sustained perfect rate, whether it has died. Storing them
    beside the rows means a status check is one read instead of a pass over thousands of
    points, and it makes "is this arm dead" a single consistent definition rather than a
    judgement re-made each time.

    `dead_since` is the first step of the earliest window of `dead_window` consecutive evals
    whose trailing score never reaches `dead_threshold`. Note it says *since*, not *dead*:
    arms have recovered from trailing 0.3, and one recovered after 400k steps near zero, so
    the field is the onset of a sustained-zero stretch and the length of that stretch is
    what makes it a verdict. Compare against the last step to get the duration.
    """
    if not eval_rows:
        return {}

    trailing = [row['trailing_avg_score'] for row in eval_rows]
    perfect = [row['perfect_percent'] for row in eval_rows]
    last = eval_rows[-1]

    peak_index = trailing.index(max(trailing))

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

    recent = perfect[-perfect_window:]
    return {
        'step': last['step'],
        'evals': len(eval_rows),
        'trailing_now': round(trailing[-1], 2),
        'peak_trailing': {'value': round(max(trailing), 2), 'step': eval_rows[peak_index]['step']},
        'best_perfect30': {'value': round(best_perfect, 1), 'step': best_perfect_step},
        'recent_perfect30': round(sum(recent) / len(recent), 1),
        'max_single_eval': max(perfect),
        'dead_since': dead_since,
        'epsilon': last['epsilon'],
    }


def save_history(path, eval_rows, resume_steps):
    """Writes the eval series plus a precomputed summary. Returns the summary."""
    summary = build_summary(eval_rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + '.partial'
    with open(partial, 'w') as handle:
        json.dump({'summary': summary, 'evals': eval_rows, 'resumes': resume_steps}, handle)
    os.replace(partial, path)
    return summary


def merge_eval_row(eval_rows, row):
    """Adds a row in step order, replacing any existing row for the same step.

    Resuming re-evaluates at the step the previous run ended on, which would
    otherwise put two points at the same x position.
    """
    for index, existing in enumerate(eval_rows):
        if existing['step'] == row['step']:
            eval_rows[index] = row
            break
    else:
        eval_rows.append(row)
    eval_rows.sort(key=lambda entry: entry['step'])


def write_run_report(path, policy_name, run_config, eval_rows, graph_filename, resume_steps=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ['# {0}'.format(policy_name), '']
    lines += ['![{0} progress]({1})'.format(policy_name, graph_filename), '']
    lines += ['Blue is average score (food eaten) on the left axis, red is '
              'perfect-game percentage on the right.', '']

    if eval_rows:
        lines += ['Latest eval: step {0}, avg score {1}, perfect games {2}%.'.format(
            eval_rows[-1]['step'], eval_rows[-1]['avg_score'], eval_rows[-1]['perfect_percent']), '']

    if resume_steps:
        lines += ['Training was resumed at step {0} (the dashed lines on the graph).'.format(
            ', '.join(str(resume_step) for resume_step in resume_steps)), '']

    lines += ['## Config', '']
    lines += ['| setting | value |', '|---|---|']
    for key, value in run_config.items():
        lines.append('| {0} | {1} |'.format(key, value))
    lines.append('')

    lines += ['## Evals', '']
    lines += ['{0} evals so far. Full series in [`{1}_evals.json`]({1}_evals.json).'.format(
        len(eval_rows), policy_name), '']
    lines += ['| ' + ' | '.join(label for _, label in EVAL_COLUMNS) + ' |']
    lines += ['|' + '---|' * len(EVAL_COLUMNS)]
    for row in _elided(eval_rows):
        if row is None:
            lines.append('| ... |' + ' |' * (len(EVAL_COLUMNS) - 1))
            continue
        lines.append('| ' + ' | '.join(str(row[key]) for key, _ in EVAL_COLUMNS) + ' |')
    lines.append('')

    partial = path + '.partial'
    with open(partial, 'w') as report:
        report.write('\n'.join(lines))
    os.replace(partial, path)


def _elided(rows):
    """Head rows, an ellipsis marker, then tail rows -- or everything if short."""
    if len(rows) <= HEAD_ROWS + TAIL_ROWS:
        return list(rows)
    return list(rows[:HEAD_ROWS]) + [None] + list(rows[-TAIL_ROWS:])
