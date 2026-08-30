"""The reward preset: which objective an arm optimises when nothing names one.

**`SNEK_ALGO=ppo` shapes by default and a DQN arm does not** (2026-08-29), so the question "what
reward function is this arm on" now has an answer that depends on the algorithm. That is a deliberate
choice and it carries one hazard worth a fixture of its own: **a bare DQN arm and a bare PPO arm
optimise different objectives**, so a comparison between them is only meaningful if both name the
knobs. The last fixture here asserts that an explicit knob wins over the preset, which is the
mechanism any such comparison relies on.

These are import-time reads, so every case needs a clean interpreter — by the time this module runs,
another test has already imported `env.constants` and the values are fixed.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROGRAM = """
import json
from env import constants as c
print(json.dumps({'preset': c.REWARD_PRESET,
                  'chase': c.CHASE_SAFE_SHAPING,
                  'gate': c.CHASE_SAFE_GATE,
                  'dist': c.FOOD_DISTANCE_REWARD,
                  'free': c.FREE_SPACE_SHAPING,
                  'perfect': c.PERFECT_GAME_REWARD}))
"""


def resolved(**env):
    """The reward constants a fresh interpreter resolves under `env`."""
    environ = dict(os.environ)
    for key in list(environ):
        if key.startswith('SNEK_'):
            del environ[key]           # a test-runner's own SNEK_* must not leak in
    environ.update({k: str(v) for k, v in env.items()})
    environ['PYTHONPATH'] = ROOT
    out = subprocess.run([sys.executable, '-c', PROGRAM], cwd=ROOT, env=environ,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


# --- the two defaults ----------------------------------------------------------------------------

def test_a_dqn_arm_at_no_knobs_is_unshaped():
    """**b1 is the no-shaping baseline and its config has to keep meaning what it meant.**

    Every b1 conclusion is stated against "snek3's bare defaults". If the preset had moved DQN's
    defaults too, b1 would silently stop being reproducible from its own write-up.
    """
    out = resolved(SNEK_ALGO='dqn')
    assert out['preset'] == 'snek3'
    assert (out['chase'], out['gate'], out['dist']) == (0.0, 85, 0.001)


def test_no_algo_at_all_is_the_dqn_default():
    # `SNEK_ALGO` is itself defaulted, and an eval process may not set it.
    assert resolved()['preset'] == 'snek3'


def test_a_ppo_arm_at_no_knobs_gets_b2s_reward_function():
    """b2's = b29's = the configuration snek2 set every record with."""
    out = resolved(SNEK_ALGO='ppo')
    assert out['preset'] == 'b2'
    assert (out['chase'], out['gate'], out['dist']) == (0.1, 75, 0.0)


def test_the_two_algorithms_bare_defaults_genuinely_differ():
    """**The hazard, asserted rather than left in a comment.**

    This is the failure `plans/ppo.md` is shaped to avoid, now deliberately accepted: a bare
    `SNEK_ALGO=dqn` arm and a bare `SNEK_ALGO=ppo` arm are not an algorithm A/B. Anyone who deletes
    this fixture because it "asserts a difference nobody wants" should read the preset's comment
    first — and anyone who makes the two agree again has to delete it, which is the point.
    """
    dqn, ppo = resolved(SNEK_ALGO='dqn'), resolved(SNEK_ALGO='ppo')
    assert (dqn['chase'], dqn['gate'], dqn['dist']) != (ppo['chase'], ppo['gate'], ppo['dist'])


# --- naming one explicitly -----------------------------------------------------------------------

@pytest.mark.parametrize('algo', ['dqn', 'ppo'])
def test_naming_the_preset_overrides_the_algorithms_default_in_both_directions(algo):
    """**The mechanism a matched A/B runs on.** Either algorithm can ask for either objective."""
    b2 = resolved(SNEK_ALGO=algo, SNEK_REWARD_PRESET='b2')
    assert (b2['chase'], b2['gate'], b2['dist']) == (0.1, 75, 0.0)
    bare = resolved(SNEK_ALGO=algo, SNEK_REWARD_PRESET='snek3')
    assert (bare['chase'], bare['gate'], bare['dist']) == (0.0, 85, 0.001)


@pytest.mark.parametrize('knob,value,field', [
    ('SNEK_CHASE_SAFE_SHAPING', '0.25', 'chase'),
    ('SNEK_CHASE_SAFE_GATE', '90', 'gate'),
    ('SNEK_FOOD_DISTANCE_REWARD', '0.002', 'dist'),
])
def test_an_explicit_knob_beats_the_preset(knob, value, field):
    """Or a sweep over the shaping dose would silently be pinned at the preset's value.

    Batch p0's arms all pass these explicitly; so will p1's. A preset that could not be overridden
    would make every shaping experiment a no-op that reports the dose it was asked for.
    """
    out = resolved(SNEK_ALGO='ppo', **{knob: value})
    assert str(out[field]) == str(float(value)) or out[field] == int(value)


def test_a_knob_the_preset_does_not_mention_is_untouched():
    # The preset names three knobs. Free-space shaping and the perfect-game reward are not among them.
    out = resolved(SNEK_ALGO='ppo')
    assert out['free'] == 0.0
    assert out['perfect'] == 100.0


def test_an_unknown_preset_is_refused_by_name():
    """**Silence here would be the `arch.json` silent-default failure in a third costume**: a typo'd
    preset falling back to the unshaped default would make a whole batch's reward function a mystery.
    """
    environ = {k: v for k, v in os.environ.items() if not k.startswith('SNEK_')}
    environ.update({'SNEK_ALGO': 'ppo', 'SNEK_REWARD_PRESET': 'b29', 'PYTHONPATH': ROOT})
    out = subprocess.run([sys.executable, '-c', PROGRAM], cwd=ROOT, env=environ,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0
    assert 'SNEK_REWARD_PRESET' in out.stderr and 'b29' in out.stderr


def test_the_preset_reaches_a_subprocess_that_inherits_the_environment():
    """`env/constants.py`'s own rule: eval shards inherit the environment, and an assignment into the
    module would not reach them. So the preset has to be a *fallback for a knob*, resolved per
    process — which is what makes stage A and stage B agree for a queued PPO arm."""
    out = resolved(SNEK_ALGO='ppo')
    assert out['preset'] == 'b2'
    # And the resolution is a pure function of the environment, so two processes agree.
    assert resolved(SNEK_ALGO='ppo') == out
