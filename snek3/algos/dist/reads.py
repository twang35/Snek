"""Acting on a return distribution other than by its mean: the reads `plans/quantile-reads.md` §1 defines.

Every distributional head can state its distribution per action as `(values, masses)`, two tensors of
shape `(m, actions, n)` with the masses summing to 1 along the last axis -- C51's atoms with their
probabilities, QR-DQN's quantiles at 1/n each, IQN's at fixed fractions, FQF's at its proposed
fractions with the bin widths as masses. A *read* turns that into one score per action, `(m, actions)`,
and the policy is the argmax of the score, with an optional second score to break ties.

| variant | score | tie-break |
|---|---|---|
| `mean:fixed` | sum(w v) | -- |
| `leastneg[:t]` | partial expectation of `v < t` (t defaults to 0): the losses' share of the mean, mass kept; 0 when there are none, so no loss at all is best | partial expectation of `v >= 0` |
| `leastnegmean[:t]` | conditional mean of `v < t`, mass discarded; 0 when the set is empty | conditional mean of `v >= 0` |
| `mix:a` | a * cond-mean(v < 0) + (1 - a) * cond-mean(v >= 0) | -- |
| `mixmass:a` | a * partial(v < 0) + (1 - a) * partial(v >= 0) | -- |
| `above:t` | partial expectation of `v > t`; the plain mean for a state where every action scores 0 | -- |
| `abovemean:t` | conditional mean of `v > t`; the plain mean where every action scores 0 | -- |

`cvar:a` stays in `net.py`, on each head's own `cvar_values`, and `mean` (no variant) stays the head's
`q_values` -- IQN's with sampled fractions. `mean:fixed` is the same arithmetic on `distribution()`,
which for IQN means fixed fractions, and is the control a reads pass runs beside the variants.

**A set with almost no mass is treated as empty** (`MIN_MASS`). The conditional mean divides by the
set's mass, so on a categorical head -- where a softmax puts some mass on every atom -- `leastneg`
would otherwise average a handful of 1e-6 probabilities on the negative atoms into a confident loss.
Quantile heads carry mass in multiples of 1/n, so at n <= 100 the floor never touches them.
"""

import torch

MIN_MASS = 0.01
EPS = 1e-6

# The parsed form: (kind, argument). `kind` is one of KINDS; the argument's meaning is per kind.
KINDS = ('cvar', 'mean', 'leastneg', 'leastnegmean', 'mix', 'mixmass', 'above', 'abovemean')


def parse_variant(variant):
    """`None`, `''`, `'mean'` -> None (the head's own mean read); otherwise `(kind, argument)`.

    `cvar:<alpha>` with alpha in (0, 1]; `mean:fixed`; `leastneg` / `leastnegmean`, each with an optional `:<t>`; `mix:<a>` and
    `mixmass:<a>` with a in [0, 1]; `above:<t>` and `abovemean:<t>`. Anything else names itself.
    """
    if variant is None or variant == '' or variant == 'mean':
        return None
    kind, _, value = str(variant).partition(':')
    if kind not in KINDS:
        raise ValueError('unknown policy variant {0!r}; the distributional heads know {1}'.format(
            variant, ', '.join(KINDS)))
    if kind == 'cvar':
        alpha = float(value)
        if not 0.0 < alpha <= 1.0:
            raise ValueError('cvar alpha must be in (0, 1], got {0}'.format(value))
        return kind, alpha
    if kind == 'mean':
        if value != 'fixed':
            raise ValueError("the mean's only variant is mean:fixed, got {0!r}".format(variant))
        return kind, 'fixed'
    if kind in ('leastneg', 'leastnegmean'):
        return kind, float(value) if value else 0.0
    if kind in ('mix', 'mixmass'):
        weight = float(value)
        if not 0.0 <= weight <= 1.0:
            raise ValueError('{0} weight must be in [0, 1], got {1}'.format(kind, value))
        return kind, weight
    if not value:
        raise ValueError('{0} needs a threshold, e.g. {0}:30'.format(kind))
    return kind, float(value)


# ---------------------------------------------------------------- the blocks

def partial_expectation(values, masses, member):
    """sum(w v) over the members: the set's share of the mean. `(m, actions)`."""
    return (values * masses * member).sum(dim=2)


def set_mass(masses, member):
    return (masses * member).sum(dim=2)


def conditional_mean(values, masses, member):
    """sum(w v) / sum(w) over the members, 0 where the set's mass is under `MIN_MASS`."""
    mass = set_mass(masses, member)
    partial = partial_expectation(values, masses, member)
    empty = mass < MIN_MASS
    return torch.where(empty, torch.zeros_like(partial), partial / torch.where(empty, torch.ones_like(mass), mass))


def mean(values, masses):
    return (values * masses).sum(dim=2)


# ---------------------------------------------------------------- the reads

def scores(values, masses, parsed):
    """`(score, tie_break)` for a parsed non-cvar variant, each `(m, actions)`; `tie_break` may be None."""
    kind, argument = parsed
    negative = (values < 0.0).to(values.dtype)
    positive = 1.0 - negative
    if kind == 'mean':
        return mean(values, masses), None
    if kind == 'leastneg':
        below = (values < argument).to(values.dtype)
        return partial_expectation(values, masses, below), partial_expectation(values, masses, positive)
    if kind == 'leastnegmean':
        below = (values < argument).to(values.dtype)
        return conditional_mean(values, masses, below), conditional_mean(values, masses, positive)
    if kind == 'mix':
        return (argument * conditional_mean(values, masses, negative)
                + (1.0 - argument) * conditional_mean(values, masses, positive)), None
    if kind == 'mixmass':
        return (argument * partial_expectation(values, masses, negative)
                + (1.0 - argument) * partial_expectation(values, masses, positive)), None
    tail = (values > argument).to(values.dtype)
    if kind == 'above':
        score = partial_expectation(values, masses, tail)
    elif kind == 'abovemean':
        score = conditional_mean(values, masses, tail)
    else:
        raise ValueError('no read for {0!r}'.format(kind))
    # A state where no action has anything in the tail is read by the mean, not by a tie among zeros.
    silent = (set_mass(masses, tail) < MIN_MASS).all(dim=1, keepdim=True)
    return torch.where(silent, mean(values, masses), score), None


def choose(score, tie_break=None):
    """Argmax of `score`; among actions within EPS of the best, the highest `tie_break`. `(m,)` int64."""
    if tie_break is None:
        return score.argmax(dim=1).to(torch.int64)
    best = score.max(dim=1, keepdim=True).values
    tied = score >= best - EPS
    ranked = torch.where(tied, tie_break, torch.full_like(tie_break, float('-inf')))
    return ranked.argmax(dim=1).to(torch.int64)


def actions(values, masses, parsed):
    """The read's greedy action per row for a parsed non-cvar variant."""
    score, tie_break = scores(values, masses, parsed)
    return choose(score, tie_break)
