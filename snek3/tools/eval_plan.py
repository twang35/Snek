"""A measured checkpoint, as a result row.

**Every row here is full length and directly comparable.** snek3's protocol is one stage — stage A
is the trainer's own 100-episode self-eval, stage B measures every checkpoint that reached ≥95/100
at 500 episodes — so there is no screen/confirm split, no tiered selector and no min-achievable
gate. snek2's files carried `selected_by`, `abandoned` and a nullable `min_achievable` for exactly
those, and half of comparing two of its rows was working out whether they were comparable at all.

The per-episode lists are stored, not just the summaries, and that is not redundancy: the summaries
pool but **the median does not**, so a row rebuilt from two summaries carries a quietly wrong
median. It is also what makes a killed run resumable — snek2 lost 192 rows and 7,534 episodes in one
incident before it stored them.
"""

import numpy as np


def wilson_interval(successes, trials, z=1.96):
    """95% confidence interval for a rate, by Wilson's score interval.

    Not the normal approximation, which breaks down at exactly the counts that matter here: a rate
    near 1.0 over 500 trials, where the symmetric interval runs past 100%.
    """
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5) / denominator
    # Clamped to contain `p`, which is exact rather than a fudge: the Wilson interval is the set of
    # rates the score test does not reject, and `p` has a test statistic of 0, so it is always
    # inside. Floating point disagrees at the ends — at 500/500 the algebra cancels to exactly 1.0
    # and the arithmetic returns 1 - 1.1e-16, so an unclamped interval excludes its own point
    # estimate at precisely the rate snek3 measures most often.
    return max(0.0, min(p, centre - spread)), min(1.0, max(p, centre + spread))


def build_row(step, held, stage_a_percent=None):
    """One result row from a checkpoint's accumulated episodes.

    `stage_a_percent` is the training self-eval that selected this checkpoint, carried through so
    stage A can be checked against stage B on the same weights — a 100-episode screen at ≥95% and a
    500-episode measurement of the same checkpoint should agree within the interval, and a
    systematic gap between them would mean the two measurement paths differ.
    """
    scores = held['scores']
    perfect = int(sum(held['perfect']))
    episodes = len(scores)
    low, high = wilson_interval(perfect, episodes)
    return {
        'step': int(step),
        'stage_a_percent': stage_a_percent,
        'episodes': episodes,
        'perfect_games': perfect,
        'perfect_percent': round(100.0 * perfect / episodes, 1),
        'perfect_ci95': [round(100.0 * low, 1), round(100.0 * high, 1)],
        'avg_score': round(float(np.mean(scores)), 2),
        'median_score': round(float(np.median(scores)), 1),
        'min_score': round(float(np.min(scores)), 1),
        'max_score': round(float(np.max(scores)), 1),
        'avg_reward': round(float(np.mean(held['rewards'])), 2),
        # Wall clock per checkpoint, so a progress readout gives an ETA from this run's own
        # throughput. Strong policies play longer episodes and measure slower, so a fixed estimate
        # is wrong in both directions.
        'seconds': round(held['seconds'], 1),
        'episode_scores': [int(score) for score in scores],
        'episode_perfect': [int(bool(flag)) for flag in held['perfect']],
        'episode_rewards': [round(float(reward), 2) for reward in held['rewards']],
    }


def held_from_row(row):
    """A `held` sample rebuilt from a stored row — the inverse of the three `episode_*` fields.

    What makes a killed measurement resumable: the shard tops the sample up to full length instead
    of discarding it and starting again.
    """
    return {'scores': list(row['episode_scores']),
            'perfect': [bool(flag) for flag in row['episode_perfect']],
            'rewards': list(row['episode_rewards']),
            'seconds': row.get('seconds', 0.0),
            'abandoned': False}


def one_line(row, label=''):
    """A row as one line, for a log or a terminal."""
    low, high = row['perfect_ci95']
    return ('{0}step {1:>9}  {2:>5.1f}% perfect  [{3:.1f}, {4:.1f}]  {5}/{6} episodes  '
            'score avg {7:.2f} median {8:.1f} max {9:.0f}  reward {10:.2f}  {11:.0f}s'.format(
                '{0}  '.format(label) if label else '', row['step'], row['perfect_percent'],
                low, high, row['perfect_games'], row['episodes'], row['avg_score'],
                row['median_score'], row['max_score'], row['avg_reward'], row['seconds']))
