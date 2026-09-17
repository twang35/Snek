"""The four rungs behind the seam, the linear epsilon schedule and the sidecar's head.
`algos/dist/algo.py`, `algos/dqn/algo.py`, `algos/dqn/schedules.py`, `tools/arch.py`, `tools/restore.py`."""

import json
import os

import pytest

import train
from algos.dqn import schedules
from env import constants
from tools import arch as arch_tools
from tools import restore

RUNGS = ('c51', 'qrdqn', 'iqn', 'fqf')


def small_config(monkeypatch, algo, **env):
    monkeypatch.setenv('SNEK_ALGO', algo)
    for key, value in env.items():
        monkeypatch.setenv('SNEK_' + key, str(value))
    config = train.build_config()
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40,
                   'batch_size': 8})
    return config


def build(monkeypatch, algo, **env):
    config = small_config(monkeypatch, algo, **env)
    module = train.ALGOS[algo]
    extra = getattr(module, 'arch_fields', lambda config: {})(config)
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=algo, **extra)
    return module.build(config, arch), config, arch


# ---------------------------------------------------------------- the registry

def test_the_four_rungs_are_registered_under_their_own_names():
    assert set(RUNGS) <= set(train.ALGOS)
    for rung in RUNGS:
        assert train.ALGOS[rung].NAME == rung
        assert rung in restore.ALGORITHMS


@pytest.mark.parametrize('rung', RUNGS)
def test_a_rung_records_its_head_in_the_sidecar_and_the_head_names_the_rung(rung, monkeypatch):
    algo, config, arch = build(monkeypatch, rung)
    assert arch['head']['type'] == {'c51': 'c51', 'qrdqn': 'quantile', 'iqn': 'iqn', 'fqf': 'fqf'}[rung]
    assert algo.rung == rung and algo.agent.head_type == arch['head']['type']


@pytest.mark.parametrize('rung', RUNGS)
def test_a_rung_prefills_advances_and_reports_like_dqn(rung, monkeypatch):
    algo, config, arch = build(monkeypatch, rung)
    assert algo.prefill() >= 40
    steps, transitions = algo.advance()
    assert steps == 1 and transitions >= 1
    assert set(algo.fields()) == {'epsilon', 'guided_fraction', 'fork'}
    assert rung in algo.describe()


def test_the_rung_knobs_are_read_and_reach_the_head():
    monkeypatch = pytest.MonkeyPatch()
    try:
        algo, config, arch = build(monkeypatch, 'c51', DIST_ATOMS=21, DIST_V_MIN=-5, DIST_V_MAX=105)
        assert arch['head'] == {'type': 'c51', 'atoms': 21, 'v_min': -5.0, 'v_max': 105.0}
        assert algo.agent.net.atoms == 21
        algo, config, arch = build(monkeypatch, 'qrdqn', DIST_QUANTILES=16, DIST_KAPPA=0.5)
        assert arch['head']['n'] == 16 and algo.agent.kappa == 0.5
        algo, config, arch = build(monkeypatch, 'iqn', DIST_TAU_SAMPLES=4, DIST_TAU_PRIME_SAMPLES=6,
                                   DIST_POLICY_SAMPLES=2, DIST_RISK_ALPHA=0.25, DIST_RISK_TRAIN=1)
        assert (algo.agent.n_tau, algo.agent.n_tau_prime) == (4, 6) and arch['head']['k'] == 2
        assert algo.agent.risk_train and algo.agent.risk_alpha == 0.25
    finally:
        monkeypatch.undo()


def test_a_bad_support_or_risk_alpha_is_refused(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'c51')
    monkeypatch.setenv('SNEK_DIST_V_MAX', '-20')
    with pytest.raises(ValueError):
        train.build_config()
    monkeypatch.delenv('SNEK_DIST_V_MAX')
    monkeypatch.setenv('SNEK_DIST_RISK_ALPHA', '0')
    with pytest.raises(ValueError):
        train.build_config()


def test_every_dist_config_key_is_its_knob_lowercased():
    import re
    from algos.dist import algo as dist_algo
    source = open(dist_algo.__file__).read()
    knobs = {name.lower() for name in re.findall(r"tuned\('([A-Z0-9_]+)'", source)}
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv('SNEK_ALGO', 'iqn')
        keys = {key for key in train.build_config() if key.startswith('dist_')}
    finally:
        monkeypatch.undo()
    assert keys <= knobs, sorted(keys - knobs)


def test_ppo_refuses_the_new_dqn_family_knobs(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    for knob in ('SNEK_DIST_ATOMS', 'SNEK_MUNCHAUSEN_ALPHA', 'SNEK_EPSILON_SCHEDULE'):
        monkeypatch.setenv(knob, '1')
        with pytest.raises(ValueError):
            train.build_config()
        monkeypatch.delenv(knob)


# ---------------------------------------------------------------- the linear schedule

def test_the_linear_schedule_is_a_straight_line_then_a_hold():
    assert schedules.linear_epsilon(0, 1.0, 0.01, 1000) == 1.0
    assert abs(schedules.linear_epsilon(500, 1.0, 0.01, 1000) - 0.505) < 1e-9
    assert schedules.linear_epsilon(1000, 1.0, 0.01, 1000) == 0.01
    assert schedules.linear_epsilon(10 ** 9, 1.0, 0.01, 1000) == 0.01
    assert schedules.linear_epsilon(5, 1.0, 0.01, 0) == 0.01, 'a zero anneal is the floor from move 0'


def test_the_default_schedule_is_the_eval_driven_one_and_an_unknown_is_refused(monkeypatch):
    assert train.build_config()['epsilon_schedule'] == 'eval'
    monkeypatch.setenv('SNEK_EPSILON_SCHEDULE', 'cosine')
    with pytest.raises(ValueError):
        train.build_config()


def test_under_the_linear_schedule_epsilon_follows_the_move_count_and_survives_a_resume(monkeypatch):
    algo, config, arch = build(monkeypatch, 'dqn', EPSILON_SCHEDULE='linear', INITIAL_EPSILON='1.0',
                               MIN_EPSILON='0.1', EPSILON_ANNEAL_STEPS='100', FORK_BRANCHES='1',
                               GUIDED_FRACTION='0.3')
    assert algo.collector.guided_fraction == 0.3, 'no bootstrap phase: the shield is on from move 0'
    algo.prefill()
    moves = 0
    for _ in range(30):
        _, transitions = algo.advance()
        moves += transitions
    assert algo.moves == moves
    expected = schedules.linear_epsilon(moves - transitions, 1.0, 0.1, 100)
    assert abs(algo.epsilon - expected) < 1e-9, 'the step ran under the epsilon of the moves before it'
    # on_eval reports the ramp and does not consult the eval history.
    reported = algo.on_eval([], {'avg_reward': 100.0, 'perfect': 1.0})
    assert abs(reported['epsilon'] - round(schedules.linear_epsilon(moves, 1.0, 0.1, 100), 5)) < 1e-9
    # A resume carries the move count.
    fresh, _, _ = build(monkeypatch, 'dqn', EPSILON_SCHEDULE='linear', INITIAL_EPSILON='1.0',
                        MIN_EPSILON='0.1', EPSILON_ANNEAL_STEPS='100', FORK_BRANCHES='1')
    fresh.load_state_dict(algo.state_dict())
    assert fresh.moves == moves


def test_under_the_eval_schedule_nothing_changed(monkeypatch):
    algo, config, arch = build(monkeypatch, 'dqn')
    algo.prefill()
    before = algo.epsilon
    algo.advance()
    assert algo.epsilon == before == config['initial_epsilon']
    assert algo.collector.guided_fraction == 0.0, 'bootstrap phase: the shield waits for the handover'
    reported = algo.on_eval([], {'avg_reward': 0.0, 'perfect': 0.0})
    assert reported['epsilon'] == round(schedules.epsilon_for(0.0, 0.0, 0.4, 0.002), 5)


# ---------------------------------------------------------------- the sidecar's head

def test_a_head_is_written_when_present_and_absent_otherwise(tmp_path):
    plain = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA)
    headed = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c51',
                                   head={'type': 'c51', 'atoms': 51, 'v_min': -10, 'v_max': 110})
    assert 'head' not in plain
    arch_tools.write_arch(str(tmp_path / 'p'), plain)
    arch_tools.write_arch(str(tmp_path / 'h'), headed)
    assert 'head' not in json.load(open(arch_tools.arch_path(str(tmp_path / 'p'))))
    on_disk = arch_tools.read_arch(str(tmp_path / 'h'))
    assert on_disk['head'] == headed['head']
    assert arch_tools.signature(on_disk) == arch_tools.signature(headed)


def test_two_heads_of_different_shape_are_different_networks(tmp_path):
    a = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c51',
                              head={'type': 'c51', 'atoms': 51, 'v_min': -10, 'v_max': 110})
    b = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c51',
                              head={'type': 'c51', 'atoms': 101, 'v_min': -10, 'v_max': 110})
    plain = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c51')
    assert arch_tools.signature(a) != arch_tools.signature(b)
    assert arch_tools.signature(a) != arch_tools.signature(plain)
    with pytest.raises(arch_tools.ArchMismatch):
        arch_tools.assert_same_network(a, b)
    directory = str(tmp_path / 'p')
    arch_tools.write_arch(directory, a)
    with pytest.raises(arch_tools.ArchMismatch):
        arch_tools.write_arch(directory, b)


# ---------------------------------------------------------------- restore's variant

def test_restore_refuses_a_variant_on_a_scalar_head_and_serves_it_on_a_distributional_one():
    plain = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA)
    net = restore.build_net(plain)
    assert callable(restore.policy_fn_for(plain, net))
    with pytest.raises(arch_tools.ArchMismatch):
        restore.policy_fn_for(plain, net, variant='cvar:0.25')
    headed = arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='qrdqn',
                                   head={'type': 'quantile', 'n': 8})
    net = restore.build_net(headed)
    assert callable(restore.policy_fn_for(headed, net, variant='cvar:0.25'))
    with pytest.raises(ValueError):
        restore.policy_fn_for(headed, net, variant='var:0.25')


def test_an_unknown_algo_in_the_sidecar_still_names_itself():
    with pytest.raises(arch_tools.ArchMismatch):
        restore.build_net(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='c52'))
