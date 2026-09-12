"""Move history: two bits per past move, read off the body (plans/archive/obs-history.md, built 2026-09-07).

The block is gated by `SNEK_OBS_HISTORY` at import, so the layout and parity suites run a second time
under the knob from a subprocess here; the value tests below need no knob because they call the
producing functions directly.
"""
import os
import subprocess
import sys

import numpy as np

from env.observations import move_history_obs, turn_between
from tools import sidecar_env
from vectorized import vec_env as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------- the reference

def test_a_straight_body_reads_all_forward():
    body = [(5, 3), (4, 3), (3, 3), (2, 3), (1, 3)]          # the opening body, heading right
    assert move_history_obs(body, 4) == [0, 0] * 4


def test_the_most_recent_move_comes_first_and_left_right_are_distinct_bits():
    # Heading right along y=3, then turned up: head (4, 2) above (4, 3). The latest move was 'left'
    # (right -> up is a left turn); the ones before were straight.
    body = [(4, 2), (4, 3), (3, 3), (2, 3), (1, 3)]
    assert move_history_obs(body, 3) == [1, 0, 0, 0, 0, 0]
    # Mirror it: turned down instead, a right turn.
    body = [(4, 4), (4, 3), (3, 3), (2, 3), (1, 3)]
    assert move_history_obs(body, 3) == [0, 1, 0, 0, 0, 0]


def test_a_spiral_reads_a_left_on_every_move():
    # Four left turns in a row: right, up, left, down, right ... read back from the head.
    body = [(3, 3), (3, 4), (4, 4), (4, 3), (4, 2), (3, 2), (2, 2)]
    # head (3,3) <- (3,4): heading up; (3,4) <- (4,4): left; (4,4) <- (4,3): down; (4,3) <- (4,2): down
    # moves: up after left = right turn? Work it through turn_between instead of by hand:
    dirs = ['up', 'left', 'down', 'down', 'right', 'right']
    expected = []
    for j in range(1, 5):
        action = turn_between(dirs[j], dirs[j - 1])
        expected += [int(action == 'left'), int(action == 'right')]
    assert move_history_obs(body, 4) == expected
    assert expected[:2] == [0, 1], 'left -> up is a right turn'


def test_a_body_too_short_for_the_depth_reads_forward_for_the_missing_moves():
    # Three cells show exactly one move; the other three read forward, as the plan specifies.
    body = [(4, 2), (4, 3), (3, 3)]
    assert move_history_obs(body, 4) == [1, 0] + [0, 0] * 3
    assert move_history_obs(None, 2) == [0, 0, 0, 0]
    assert move_history_obs([], 2) == [0, 0, 0, 0]


def test_turn_between_is_the_inverse_of_the_direction_map():
    from env.constants import CURRENT_DIRECTION_MAPS
    for heading, table in CURRENT_DIRECTION_MAPS.items():
        for action, result in table.items():
            assert turn_between(heading, result) == action
    assert turn_between('up', 'down') == 'forward', 'a reversal cannot occur; it reads forward'


# ------------------------------------------------------------- the vectorised form

def test_move_history_bits_agrees_with_the_reference_on_driven_lanes():
    """Three lanes driven by hand through `VecSnake`, the buffer read back as positions for the
    reference. Lane 1's four lefts end in a self-collision, which is fine: the observation of the final
    board is still defined and still has to agree."""
    env = V.VecSnake(3, seed=1)
    seqs = [[2, 2, 2, 2], [0, 0, 0, 0], [0, 1, 0, 1]]
    for t in range(4):
        env.step(np.array([s[t] for s in seqs]), autoreset=False)
    bits = V.move_history_bits(env.body, env.hp, env.length, 4)
    for lane in range(3):
        L, hp = int(env.length[lane]), int(env.hp[lane])
        cells = [int(env.body[lane, (hp - k) % V.CAP]) for k in range(L)]
        positions = [(c % V.GRID, c // V.GRID) for c in cells]
        assert bits[lane].tolist() == move_history_obs(positions, 4), (lane, seqs[lane])
    assert bits[0].tolist() == [0.0] * 8, 'four forwards'
    assert bits[1].tolist() == [1, 0] * 4, 'four lefts'
    assert bits[2].tolist() == [0, 1, 1, 0, 0, 1, 0, 0], 'zigzag; the oldest move is off the body'


def test_move_history_bits_is_zero_width_at_depth_zero():
    env = V.VecSnake(2, seed=1)
    assert V.move_history_bits(env.body, env.hp, env.length, 0).shape == (2, 0)


# ------------------------------------------------------------- the era and the sidecar

def test_the_era_carries_the_depth_and_the_parser_reads_it_back():
    assert sidecar_env.obs_history_of_era('obs26-20260907') == 0
    assert sidecar_env.obs_history_of_era('obs26-20260907-hist4') == 4
    assert sidecar_env.obs_history_of_era('obs26-20260907-hist12') == 12
    assert sidecar_env.obs_history_of_era('b09c616') == 0


def test_adopt_sets_the_knob_from_a_sidecar_but_never_overrides_an_explicit_value(tmp_path):
    import json
    directory = tmp_path / 'hist-arm'
    directory.mkdir()
    (directory / 'arch.json').write_text(json.dumps({'obs_era': 'obs26-20260907-hist4', 'obs_len': 34,
                                                     'algo': 'ppo', 'fc_layer_params': [320],
                                                     'num_actions': 3}))
    environ = {}
    assert sidecar_env.adopt(str(directory), environ=environ) == 4
    assert environ['SNEK_OBS_HISTORY'] == '4'
    environ = {'SNEK_OBS_HISTORY': '2'}
    assert sidecar_env.adopt(str(directory), environ=environ) is None
    assert environ['SNEK_OBS_HISTORY'] == '2'
    # argv form: flags are skipped, the first token that names a sidecar decides
    environ = {}
    argv = ['shard.py', '--shards', '4', str(directory), '--episodes', '500']
    assert sidecar_env.adopt_from_argv(argv, environ=environ) == 4
    assert sidecar_env.adopt_from_argv(['x', str(tmp_path / 'nope')], environ={}) is None


def test_a_sidecar_without_the_suffix_adopts_depth_zero(tmp_path):
    import json
    directory = tmp_path / 'plain-arm'
    directory.mkdir()
    (directory / 'arch.json').write_text(json.dumps({'obs_era': 'obs26-20260907'}))
    environ = {}
    assert sidecar_env.adopt(str(directory), environ=environ) == 0
    assert environ['SNEK_OBS_HISTORY'] == '0'


# ------------------------------------------------------------- the whole vector, under the knob

def test_the_layout_and_parity_suites_hold_at_depth_four():
    """`SNEK_OBS_HISTORY` is read at import, so the suites that pin the vector run again from a
    subprocess with the knob set: the block table sums to 34, the era carries `-hist4`, and the
    vectorised observation equals the reference elementwise over the parity states."""
    env = {**os.environ, 'SNEK_OBS_HISTORY': '4', 'SNEK_CHART_WINDOW': '0'}
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         'tests/test_observation_layout.py',
         'tests/test_vec_parity.py::test_observation_parity_elementwise',
         'tests/test_observations.py::test_observation_spec_matches_what_the_game_emits'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    probe = subprocess.run(
        [sys.executable, '-c', 'from env import constants as c; print(c.OBS_LEN, c.OBS_ERA, c.OBS_BLOCKS[-1])'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert probe.stdout.strip() == "34 obs26-20260907-hist4 ('move_history', 8)", probe.stdout + probe.stderr


def test_the_scheduler_reads_one_depth_per_wave_and_none_when_mixed():
    from tools.scheduler import wave_obs_history
    uniform = [{'env': {'SNEK_OBS_HISTORY': '4'}}, {'env': {'SNEK_OBS_HISTORY': '4'}}]
    unset = [{'env': {}}, {'env': {'SNEK_OBS_HISTORY': '0'}}]
    mixed = [{'env': {'SNEK_OBS_HISTORY': '4'}}, {'env': {}}]
    assert wave_obs_history(uniform) == '4'
    assert wave_obs_history(unset) == '0'
    assert wave_obs_history(mixed) is None
