"""Architecture sidecar for a saved policy — ``arch.json``, one per policy directory.

A TF checkpoint stores variable *values*, never the architecture that produced them, and
``restore(...).expect_partial()`` populates whatever layers it can match and stays silent about the
rest. So a checkpoint trained with ``FC_LAYERS=100,200,100`` restored into a ``50,100,50`` net loads
without a word and plays like a beginner — the failure this repo has been bitten by more than once
(see CLAUDE.md's hall-of-fame section and ``eval_agent.py``).

``arch.json`` closes that. It records the few facts that decide whether restored weights *mean* what
the network thinks they mean — the FC layer widths, the action count, and the observation vector's
length and meaning-era — written once when a policy is created and read by everything that restores
it (training resume, ``eval_checkpoints``, ``eval_workers``, ``watch.py``). A restore *requires* the
file and fails loudly on any mismatch, rather than half-loading.

It lives beside the checkpoints (``savedPolicies/<policy>/arch.json``), not inside them, so it is
written once rather than copied into all ~10000 checkpoints, and it travels with the weights when a
checkpoint is copied into ``hallOfFame/`` or rsynced to the desktop.
"""
import json
import os

import numpy as np

import snake_constants

ARCH_FILENAME = 'arch.json'

# Read for `algo` when the field is absent. Every policy directory written before C51 existed has no
# such key, and there are ~100 of them plus every `hallOfFame/` entry, so "missing means ddqn" is what
# keeps them all loading. New directories record the field explicitly either way.
DEFAULT_ALGO = 'ddqn'
CATEGORICAL_ALGO = 'c51'
# Only meaningful for the categorical algorithm, and only written for it — a scalar arm's sidecar does
# not carry three null fields.
CATEGORICAL_FIELDS = ('num_atoms', 'v_min', 'v_max')


class ArchMismatch(Exception):
    """A restore's environment or config disagrees with the recorded architecture.

    Loud on purpose: the whole point of the sidecar is that this can no longer pass silently.
    """


def arch_path(policy_dir):
    return os.path.join(policy_dir, ARCH_FILENAME)


def build_arch(fc_layer_params, num_actions, obs_len, obs_era, algo=DEFAULT_ALGO,
               num_atoms=None, v_min=None, v_max=None, perfect_game_reward=None):
    """The canonical dict.

    ``fc_layer_params`` is stored as a list — JSON has no tuples and every reader iterates it, so
    the round trip compares list-to-list.

    ``algo`` is always recorded, so a future third algorithm cannot be confused with an old file. The
    three categorical fields are written **only** for ``c51``, because a scalar arm carrying
    ``"num_atoms": null`` is noise and because their absence is then a positive signal rather than an
    ambiguity.

    ``v_min``/``v_max`` are as load-bearing as the layer widths, and this is the only place they are
    durably written. A categorical policy's greedy action is ``argmax_a sum_i z_i p_i(s, a)``, so
    restoring correct weights against the wrong support yields a **different policy** with no shape
    mismatch anywhere — the silent-failure class this file exists for, arriving through a new door.

    ``perfect_game_reward`` is recorded for **both** algorithms and defaults to
    ``snake_constants.DEFAULT_PERFECT_GAME_REWARD`` when omitted, which is what every arm before batch
    33 trained with. It is not a weight shape and not an observation, so nothing about restoring
    breaks when it disagrees — which is exactly why it is written down. An arm trained to value a win
    at 10 and resumed at 100 restores cleanly and then optimises a different objective, and for a
    categorical arm the atoms are calibrated to the old return range on top of that. Absent from a
    pre-batch-33 sidecar, and ``reward_scale_of`` reads that absence as the default rather than as
    unknown.
    """
    arch = {
        'fc_layer_params': [int(width) for width in fc_layer_params],
        'num_actions': int(num_actions),
        'obs_len': int(obs_len),
        'obs_era': str(obs_era),
        'algo': str(algo),
    }
    if str(algo) == CATEGORICAL_ALGO:
        if num_atoms is None or v_min is None or v_max is None:
            raise ValueError('a {0} arch needs num_atoms, v_min and v_max'.format(CATEGORICAL_ALGO))
        arch['num_atoms'] = int(num_atoms)
        arch['v_min'] = float(v_min)
        arch['v_max'] = float(v_max)
    arch['perfect_game_reward'] = float(
        snake_constants.DEFAULT_PERFECT_GAME_REWARD if perfect_game_reward is None
        else perfect_game_reward)
    return arch


def reward_scale_of(arch):
    """What a win paid for this checkpoint, defaulting to the pre-batch-33 value.

    A missing field means the sidecar predates the knob, and every such arm trained at the default —
    so this is a known value rather than an unknown one, and the guards can compare it.
    """
    return float(arch.get('perfect_game_reward', snake_constants.DEFAULT_PERFECT_GAME_REWARD))


def algo_of(arch):
    """The algorithm this checkpoint was trained with, ``ddqn`` for a pre-C51 sidecar."""
    return arch.get('algo', DEFAULT_ALGO)


def is_categorical(arch):
    return algo_of(arch) == CATEGORICAL_ALGO


def support_from_arch(arch):
    """The atom support as a numpy array, or ``None`` for a scalar arm.

    The one place the support is reconstructed, so every reader — eval, watch, diagnostics — gets the
    grid the checkpoint was *trained* on rather than whatever the environment is currently configured
    for. Pair it with ``under_the_hood.expected_q``.
    """
    if not is_categorical(arch):
        return None
    return np.linspace(float(arch['v_min']), float(arch['v_max']), int(arch['num_atoms']))


def refuse_categorical(policy_dir, tool):
    """Stop `tool` from running against a c51 policy, loudly. Returns the arch when it is scalar.

    For the diagnostics whose measurement is *about* a scalar head — an absolute TD error, a Huber
    loss, a teacher-student fit against a same-shaped scalar network. Each would run on a categorical
    checkpoint and print numbers, because a logit is a float like any other, and those numbers would
    mean nothing. Refusing is the cheap half of porting them; a wrong plot in `findings.md` is the
    expensive half.
    """
    arch = require_arch(policy_dir)
    if is_categorical(arch):
        raise SystemExit(
            '{0} does not support SNEK_ALGO={1} policies ({2}). Its measurement assumes a scalar Q '
            'head — there is no TD error to take the absolute value of, and a same-shaped fresh '
            'network is a different object. Port it or pick a ddqn arm.'
            .format(tool, CATEGORICAL_ALGO, policy_dir))
    return arch


def read_arch(policy_dir):
    """The recorded arch, or ``None`` if the policy has no sidecar (a pre-``arch.json`` dir)."""
    path = arch_path(policy_dir)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def write_arch(policy_dir, arch):
    """Write ``arch.json`` atomically.

    The dir is normally the checkpointer's own ``ckpt_dir``, created before this runs; making it
    here too keeps a fresh run that writes arch before its first checkpoint safe.
    """
    os.makedirs(policy_dir, exist_ok=True)
    path = arch_path(policy_dir)
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(arch, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def require_arch(policy_dir):
    """The recorded arch, or a loud ``ArchMismatch`` if the sidecar is missing."""
    arch = read_arch(policy_dir)
    if arch is None:
        raise ArchMismatch(
            'no {0} in {1}: this policy predates the architecture sidecar, or the sidecar was not '
            'copied with the checkpoint. Run backfill_arch.py (or write one) before restoring.'
            .format(ARCH_FILENAME, policy_dir))
    return arch


def _categorical_problems(arch):
    """Internal consistency of the algorithm fields, independent of the live environment.

    A hand-edited or half-written sidecar is the case this catches: ``algo: c51`` with no support is
    unusable, and support fields on a ``ddqn`` arm mean someone changed the algorithm without
    retraining. Both are cheap to detect and expensive to debug from the resulting policy.
    """
    problems = []
    algo = algo_of(arch)
    present = [field for field in CATEGORICAL_FIELDS if arch.get(field) is not None]
    if algo == CATEGORICAL_ALGO:
        missing = [field for field in CATEGORICAL_FIELDS if arch.get(field) is None]
        if missing:
            problems.append('algo {0!r} but {1} missing'.format(algo, ', '.join(missing)))
        else:
            if int(arch['num_atoms']) < 2:
                problems.append('num_atoms {0} < 2'.format(arch['num_atoms']))
            if float(arch['v_min']) >= float(arch['v_max']):
                problems.append('v_min {0} >= v_max {1}'.format(arch['v_min'], arch['v_max']))
    elif present:
        problems.append('algo {0!r} but carries categorical fields {1} — the sidecar was edited '
                        'without retraining'.format(algo, ', '.join(present)))
    return problems


def assert_restorable(policy_dir, num_actions, obs_len, obs_era):
    """Require ``arch.json`` and check every field that decides weight shape or meaning against the
    live environment.

    Returns the arch so the caller builds the network from ``arch['fc_layer_params']`` — the
    recorded shape is authoritative, which is what removes the reliance on ``SNEK_FC_LAYERS`` being
    set correctly at eval time.
    """
    arch = require_arch(policy_dir)
    problems = []
    if arch.get('num_actions') != int(num_actions):
        problems.append('num_actions {0} != env {1}'.format(arch.get('num_actions'), num_actions))
    if arch.get('obs_len') != int(obs_len):
        problems.append('obs_len {0} != env {1} (observation vector changed length)'
                        .format(arch.get('obs_len'), obs_len))
    if arch.get('obs_era') != str(obs_era):
        problems.append('obs_era {0!r} != env {1!r} (observation meaning changed at constant '
                        'length)'.format(arch.get('obs_era'), obs_era))
    problems.extend(_categorical_problems(arch))
    if problems:
        raise ArchMismatch('{0} in {1} does not match this environment: {2}'
                           .format(ARCH_FILENAME, policy_dir, '; '.join(problems)))
    return arch


def assert_config_matches(policy_dir, fc_layer_params, algo=DEFAULT_ALGO, num_atoms=None,
                          v_min=None, v_max=None, perfect_game_reward=None):
    """On a training *resume*, the checkpoint's arch is authoritative and the env knobs are the thing
    that might be wrong.

    Fail loudly if they disagree — a resume under a changed knob is how a near-record arm was almost
    reverted for a loss (CLAUDE.md). Returns the arch on success.

    The support is checked as well as the shape, and it is the *more* dangerous of the two: changing
    ``SNEK_FC_LAYERS`` at least produces a shape error somewhere downstream, whereas changing
    ``SNEK_V_MAX`` on a resume restores every weight cleanly and silently relabels what the atoms
    mean. Only ``num_atoms`` would be caught by shape alone.
    """
    arch = require_arch(policy_dir)
    problems = []
    recorded = [int(width) for width in arch['fc_layer_params']]
    wanted = [int(width) for width in fc_layer_params]
    if recorded != wanted:
        problems.append('recorded fc_layer_params {0} != SNEK_FC_LAYERS {1}'.format(recorded, wanted))
    if algo_of(arch) != str(algo):
        problems.append('recorded algo {0!r} != SNEK_ALGO {1!r}'.format(algo_of(arch), str(algo)))
    elif is_categorical(arch):
        for field, wanted_value, knob in (('num_atoms', num_atoms, 'SNEK_NUM_ATOMS'),
                                          ('v_min', v_min, 'SNEK_V_MIN'),
                                          ('v_max', v_max, 'SNEK_V_MAX')):
            if wanted_value is None:
                problems.append('resuming a {0} arm needs {1}'.format(CATEGORICAL_ALGO, knob))
            elif float(arch[field]) != float(wanted_value):
                problems.append('recorded {0} {1} != {2} {3}'
                                .format(field, arch[field], knob, wanted_value))
    wanted_reward = (snake_constants.PERFECT_GAME_REWARD if perfect_game_reward is None
                     else float(perfect_game_reward))
    if reward_scale_of(arch) != float(wanted_reward):
        problems.append('recorded perfect_game_reward {0} != SNEK_PERFECT_GAME_REWARD {1} — the '
                        'checkpoint was trained for a different objective'
                        .format(reward_scale_of(arch), wanted_reward))
    if problems:
        raise ArchMismatch(
            'resuming {0}: {1}. The checkpoint was trained with the recorded values; drop the '
            'override or resume a matching run.'.format(policy_dir, '; '.join(problems)))
    return arch
