"""Shrink-and-perturb resets: `algos/dqn/resets.py`, the `maybe_reset` hook on both agents, the knobs.
Group D, row D1 (`plans/algoExploration/d-data-efficiency.md`)."""

import numpy as np
import pytest
import torch

import train
from algos.dist import net as dist_net
from algos.dist.agent import DistAgent
from algos.dqn import resets
from algos.dqn.agent import DdqnAgent
from env import constants
from tools import arch as arch_tools

HEADS = {
    'c51': {'type': 'c51', 'atoms': 11, 'v_min': -10.0, 'v_max': 110.0},
    'qrdqn': {'type': 'quantile', 'n': 8},
    'fqf': {'type': 'fqf', 'embedding': 16, 'n': 8},
}


def an_arch(head=None, algo='dqn'):
    return arch_tools.build_arch([32, 16], constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                 algo=algo, head=head)


def a_batch(m=16, seed=0):
    rng = np.random.default_rng(seed)
    done = rng.random(m) < 0.25
    return {'obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'next_obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'action': rng.integers(0, constants.NUM_ACTIONS, m).astype(np.int64),
            'reward': np.where(done, rng.choice([-5.0, 100.0], m), rng.choice([0.0, 1.0], m)).astype(np.float32),
            'discount': np.where(done, 0.0, 0.99).astype(np.float32)}


def params(net):
    return {name: p.detach().clone() for name, p in net.named_parameters()}


def an_agent(head=None, **kwargs):
    if head is None:
        return DdqnAgent(an_arch(), seed=1, learning_rate=1e-2, **kwargs)
    return DistAgent(an_arch(HEADS[head], algo=head), seed=1, learning_rate=1e-2, **kwargs)


# ---------------------------------------------------------------- the schedule

def test_the_schedule_fires_on_multiples_of_the_interval_and_not_past_the_stop():
    schedule = resets.ResetSchedule(interval=5, alpha=0.5, stop_after=12)
    assert [step for step in range(0, 30) if schedule.due(step)] == [5, 10]
    never_stops = resets.ResetSchedule(interval=5, alpha=0.5, stop_after=0)
    assert [step for step in range(0, 30) if never_stops.due(step)] == [5, 10, 15, 20, 25]
    off = resets.ResetSchedule(interval=0)
    assert not off.enabled and not any(off.due(step) for step in range(0, 30))


def test_a_bad_schedule_names_its_knob():
    with pytest.raises(ValueError, match='reset_alpha'):
        resets.ResetSchedule(interval=5, alpha=1.5)
    with pytest.raises(ValueError, match='reset_interval'):
        resets.ResetSchedule(interval=-1)
    with pytest.raises(ValueError, match='reset_stop_after'):
        resets.ResetSchedule(interval=5, stop_after=-3)


# ---------------------------------------------------------------- the reset itself

def test_the_partition_puts_every_hidden_linear_in_the_trunk_and_the_rest_in_the_head():
    trunk, head = resets.partition(an_agent().net)
    assert sorted(trunk) == ['hidden.0.bias', 'hidden.0.weight', 'hidden.1.bias', 'hidden.1.weight']
    assert sorted(head) == ['head.bias', 'head.weight']
    trunk, head = resets.partition(an_agent('fqf').net)
    assert all(name.startswith('qnet.hidden.') for name in trunk) and len(trunk) == 4
    assert sorted(head) == ['fraction.bias', 'fraction.weight', 'qnet.head.bias', 'qnet.head.weight',
                            'tau_embed.bias', 'tau_embed.weight']


def test_alpha_one_leaves_the_trunk_and_still_reinitialises_the_head():
    agent = an_agent()
    before = params(agent.net)
    fresh = agent.fresh_net(seed=99)
    resets.shrink_and_perturb(agent.net, fresh, alpha=1.0)
    after = params(agent.net)
    for name in ('hidden.0.weight', 'hidden.1.weight', 'hidden.0.bias'):
        assert torch.equal(after[name], before[name])
    assert torch.equal(after['head.weight'], dict(fresh.named_parameters())['head.weight'])
    assert not torch.equal(after['head.weight'], before['head.weight'])


def test_alpha_zero_is_a_fresh_initialisation_with_the_given_seed_everywhere():
    agent = an_agent()
    fresh = agent.fresh_net(seed=99)
    resets.shrink_and_perturb(agent.net, fresh, alpha=0.0)
    for name, value in agent.net.named_parameters():
        assert torch.equal(value, dict(fresh.named_parameters())[name]), name
    # And a fresh net at the same seed is the same net: the seed is what pins the reset.
    again = agent.fresh_net(seed=99)
    assert all(torch.equal(p, dict(again.named_parameters())[n]) for n, p in fresh.named_parameters())


def test_the_trunk_is_interpolated_exactly_and_the_head_is_not():
    agent = an_agent()
    before = params(agent.net)
    fresh = agent.fresh_net(seed=99)
    fresh_params = params(fresh)
    resets.shrink_and_perturb(agent.net, fresh, alpha=0.25)
    after = params(agent.net)
    expected = 0.25 * before['hidden.0.weight'] + 0.75 * fresh_params['hidden.0.weight']
    assert torch.allclose(after['hidden.0.weight'], expected, atol=1e-7)
    assert torch.equal(after['head.weight'], fresh_params['head.weight'])


def test_buffers_are_not_touched():
    agent = an_agent('c51')
    support = agent.net.support.clone()
    resets.shrink_and_perturb(agent.net, agent.fresh_net(seed=5), alpha=0.0)
    assert torch.equal(agent.net.support, support)


# ---------------------------------------------------------------- through the agent

def updates(agent, n, seed=0):
    for i in range(n):
        agent.update(a_batch(seed=seed + i))


@pytest.mark.parametrize('head', [None, 'qrdqn', 'fqf'])
def test_the_agent_resets_on_the_interval_copies_the_target_and_clears_the_optimisers(head):
    agent = an_agent(head, reset_interval=3, reset_alpha=0.5)
    updates(agent, 2)
    assert agent.resets == 0
    assert all(len(opt.state) > 0 for opt in agent.optimizers()), 'the optimisers had run'
    _, metrics = agent.update(a_batch(seed=2))
    assert agent.resets == 1 and metrics['resets'] == 1
    fresh = agent.fresh_net(resets.reset_seed(agent.seed, 1))
    trunk, head_names = resets.partition(agent.net)
    net_params, fresh_params = params(agent.net), params(fresh)
    for name in head_names:
        assert torch.equal(net_params[name], fresh_params[name]), name
    for name in trunk:
        assert not torch.equal(net_params[name], fresh_params[name]), name
    for name, value in agent.target.named_parameters():
        assert torch.equal(value, net_params[name]), 'the target is the reset net'
    assert len(agent.optimizer.state) == 0, 'the moments were cleared'
    if head == 'fqf':
        assert len(agent.fraction_optimizer.state) == 0, "the fraction net's RMSProp was cleared too"
        assert agent.fraction_optimizer in agent.optimizers()


def test_a_reset_step_is_a_different_net_from_an_unreset_one_and_the_same_seed_resets_the_same():
    plain = an_agent(reset_interval=0)
    reset_a = an_agent(reset_interval=3)
    reset_b = an_agent(reset_interval=3)
    for agent in (plain, reset_a, reset_b):
        updates(agent, 3)
    assert not torch.equal(plain.net.head.weight, reset_a.net.head.weight)
    assert torch.equal(reset_a.net.head.weight, reset_b.net.head.weight)


def test_successive_resets_draw_different_fresh_networks():
    agent = an_agent(reset_interval=2)
    updates(agent, 2)
    first = agent.net.head.weight.detach().clone()
    updates(agent, 2, seed=10)
    assert agent.resets == 2
    assert not torch.equal(agent.net.head.weight, first), 'the second reset must not replay the first'
    assert resets.reset_seed(1, 1) != resets.reset_seed(1, 2) and resets.reset_seed(None, 3) is None


def test_no_reset_fires_past_the_stop_and_off_is_off():
    stopped = an_agent(reset_interval=2, reset_stop_after=4)
    updates(stopped, 9)
    assert stopped.resets == 2
    off = an_agent()
    updates(off, 9)
    assert off.resets == 0 and not off.reset_schedule.enabled


def test_the_reset_count_survives_a_resume():
    agent = an_agent(reset_interval=2)
    updates(agent, 4)
    state = agent.state_dict()
    restored = an_agent(reset_interval=2)
    restored.load_state_dict(state)
    assert restored.resets == 2 and restored.train_step == 4
    # An old checkpoint without the field reads as no resets.
    del state['resets']
    restored.load_state_dict(state)
    assert restored.resets == 0


# ---------------------------------------------------------------- the knobs

def test_the_knobs_are_read_and_reach_the_agent(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'qrdqn')
    monkeypatch.setenv('SNEK_RESET_INTERVAL', '40000')
    monkeypatch.setenv('SNEK_RESET_ALPHA', '0.3')
    monkeypatch.setenv('SNEK_RESET_STOP_AFTER', '900000')
    config = train.build_config()
    assert (config['reset_interval'], config['reset_alpha'], config['reset_stop_after']) == (40000, 0.3, 900000)
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40, 'batch_size': 8})
    module = train.ALGOS['qrdqn']
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='qrdqn', **module.arch_fields(config))
    algo = module.build(config, arch)
    assert algo.agent.reset_schedule.interval == 40000
    assert algo.agent.reset_schedule.alpha == 0.3
    assert algo.agent.reset_schedule.stop_after == 900000
    assert 'resets every 40,000 gradient steps at alpha 0.3, none after 900,000' in algo.describe()
    assert any('resets 0' in line for line in algo.log_extra({}))


def test_the_default_is_off_and_says_nothing(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'dqn')
    config = train.build_config()
    assert config['reset_interval'] == 0
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40, 'batch_size': 8})
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='dqn')
    algo = train.ALGOS['dqn'].build(config, arch)
    assert 'resets' not in algo.describe()
    assert algo.log_extra({}) == []


def test_a_bad_alpha_is_refused_at_config_time(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'dqn')
    monkeypatch.setenv('SNEK_RESET_ALPHA', '2')
    with pytest.raises(ValueError, match='reset_alpha'):
        train.build_config()


def test_ppo_refuses_the_reset_knobs(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_RESET_INTERVAL', '5')
    with pytest.raises(ValueError, match='SNEK_RESET_INTERVAL'):
        train.build_config()


# ---------------------------------------------------------------- the within-cycle anneal (BBF)

def a_cycle(steps=10000):
    return resets.CycleSchedule((10, 3), (0.97, 0.997), steps=steps)


def test_the_cycle_runs_from_the_short_horizon_to_the_long_one_and_holds():
    cycle = a_cycle()
    assert [cycle.n_step_at(s) for s in (0, 2500, 5000, 7500, 10000, 50000)] == [10, 7, 5, 4, 3, 3]
    gammas = [cycle.gamma_at(s) for s in (0, 5000, 10000, 50000)]
    assert gammas[0] == pytest.approx(0.97) and gammas[-1] == pytest.approx(0.997) and gammas[2] == pytest.approx(0.997)
    # Exponential in log(1 - gamma): the midpoint is the geometric mean of the two horizons.
    assert 1.0 - gammas[1] == pytest.approx(((1 - 0.97) * (1 - 0.997)) ** 0.5)


def test_an_off_cycle_returns_the_constants_it_was_given():
    off = resets.CycleSchedule(n_step=3, gamma=0.9)
    assert not off.enabled
    assert (off.n_step_at(0), off.gamma_at(0), off.n_step_at(10 ** 6), off.gamma_at(10 ** 6)) == (3, 0.9, 3, 0.9)
    assert off.describe() == ''


def test_a_bad_cycle_names_its_knob():
    with pytest.raises(ValueError, match='RESET_ANNEAL_N_STEP'):
        resets.CycleSchedule((10, 3), None)
    with pytest.raises(ValueError, match='reset_anneal_steps'):
        resets.CycleSchedule((10, 3), (0.97, 0.997), steps=0)
    with pytest.raises(ValueError, match='reset_anneal_gamma'):
        resets.CycleSchedule((10, 3), (0.97, 1.0))
    with pytest.raises(ValueError, match='RESET_ANNEAL_GAMMA'):
        resets.parse_pair('0.97', float, 'RESET_ANNEAL_GAMMA')
    assert resets.parse_pair('', int, 'X') is None and resets.parse_pair('10, 3', int, 'X') == (10, 3)


def test_the_agent_restarts_the_cycle_at_a_reset_and_hands_both_values_to_the_hook():
    seen = []
    agent = an_agent(reset_interval=6, cycle=a_cycle(steps=4), on_cycle=lambda n, g: seen.append((n, round(g, 4))))
    for _ in range(9):
        agent.update(a_batch())
    # Steps 1..5 climb the cycle; step 6 resets and the values jump back; 7..9 climb again.
    assert [n for n, _ in seen[:5]] == [7, 5, 4, 3, 3] and seen[1][1] == 0.9905 and seen[3][1] == 0.997
    assert seen[5] == (10, 0.97), seen
    assert seen[6][0] < 10 and seen[6][1] > 0.97
    assert agent.steps_since_reset == 3 and agent.last_reset_step == 6


def test_the_hook_is_not_called_when_the_cycle_is_off():
    seen = []
    agent = an_agent(reset_interval=3, on_cycle=lambda n, g: seen.append((n, g)))
    for _ in range(4):
        agent.update(a_batch())
    assert seen == [] and agent.resets == 1


def test_the_cycle_position_survives_a_resume():
    agent = an_agent(reset_interval=4, cycle=a_cycle())
    for _ in range(6):
        agent.update(a_batch())
    state = agent.state_dict()
    restored = an_agent(reset_interval=4, cycle=a_cycle())
    restored.load_state_dict(state)
    assert (restored.last_reset_step, restored.steps_since_reset) == (4, 2)
    assert restored.cycle_values() == agent.cycle_values()


def test_the_algo_moves_the_collector_and_the_env_together(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'c51')
    monkeypatch.setenv('SNEK_RESET_INTERVAL', '40')
    monkeypatch.setenv('SNEK_RESET_ANNEAL_N_STEP', '10,3')
    monkeypatch.setenv('SNEK_RESET_ANNEAL_GAMMA', '0.97,0.997')
    monkeypatch.setenv('SNEK_RESET_ANNEAL_STEPS', '20')
    monkeypatch.setenv('SNEK_N_STEP_UPDATE', '3')
    monkeypatch.setenv('SNEK_DISCOUNT', '0.997')
    config = train.build_config()
    assert (config['reset_anneal_n_step'], config['reset_anneal_gamma'], config['reset_anneal_steps']) == ('10,3', '0.97,0.997', 20)
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40, 'batch_size': 8})
    module = train.ALGOS['c51']
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='c51', **module.arch_fields(config))
    algo = module.build(config, arch)
    # The first cycle starts at gradient step 0: the collector opens at the short horizon.
    assert algo.collector.n_step == 10 and algo.collector.discount == pytest.approx(0.97)
    assert algo.collector.vec.shaping_discount == pytest.approx(0.97)
    assert 'n-step 10 -> 3 and gamma 0.97 -> 0.997 over 20 gradient steps after each reset' in algo.describe()
    algo.prefill()
    for _ in range(30):
        algo.advance()
    since = algo.agent.steps_since_reset
    assert 0 < since < 40
    n, g = algo.agent.cycle_values()
    assert algo.collector.n_step == n and algo.collector.discount == pytest.approx(g)
    assert algo.collector.vec.shaping_discount == pytest.approx(g)
    if since >= 20:
        assert (n, g) == (3, pytest.approx(0.997))
    fields = algo.fields()
    assert fields['cycle'] == {'n_step': n, 'gamma': round(g, 5), 'since_reset': since}
    assert any('cycle n-step' in line for line in algo.log_extra(fields))


def test_without_the_anneal_knobs_the_collector_keeps_the_run_constants(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'dqn')
    monkeypatch.setenv('SNEK_RESET_INTERVAL', '40')
    monkeypatch.setenv('SNEK_N_STEP_UPDATE', '3')
    config = train.build_config()
    assert config['reset_anneal_n_step'] == '' and config['reset_anneal_gamma'] == ''
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40, 'batch_size': 8})
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='dqn')
    algo = train.ALGOS['dqn'].build(config, arch)
    assert algo.collector.n_step == 3 and algo.collector.discount == pytest.approx(0.99)
    assert 'cycle' not in algo.fields() and 'gamma' not in algo.describe()


def test_half_an_anneal_pair_is_refused_at_config_time(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'dqn')
    monkeypatch.setenv('SNEK_RESET_ANNEAL_N_STEP', '10,3')
    with pytest.raises(ValueError, match='RESET_ANNEAL_GAMMA'):
        train.build_config()


def test_ppo_and_sac_refuse_the_anneal_knobs(monkeypatch):
    for algo in ('ppo', 'sac2'):
        monkeypatch.setenv('SNEK_ALGO', algo)
        monkeypatch.setenv('SNEK_RESET_ANNEAL_STEPS', '5')
        with pytest.raises(ValueError, match='SNEK_RESET_ANNEAL_STEPS'):
            train.build_config()
