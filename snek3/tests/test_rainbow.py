"""Group C: `algos/rainbow/` -- the noisy layer, the dueling and residual network, the agent's noise
discipline, the two names' defaults, the epsilon floor and the sidecar's trunk. `c-value-stack.md`."""

import numpy as np
import pytest
import torch
from torch import nn

import train
from algos.dqn import schedules
from algos.rainbow import algo as rainbow_algo
from algos.rainbow import net as network
from algos.rainbow.noisy import NoisyLinear, set_noise
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore

C51 = {'type': 'c51', 'atoms': 5, 'v_min': 0.0, 'v_max': 4.0}
QUANT = {'type': 'quantile', 'n': 8}
IQN = {'type': 'iqn', 'embedding': 16, 'n_tau': 8, 'k': 4}
PLAIN = {'dueling': True, 'noisy': True, 'noisy_sigma': 0.5, 'residual': False, 'blocks': 3,
         'spectral_norm': True, 'layer_norm': False}


def trunk(**changes):
    out = dict(PLAIN)
    out.update(changes)
    return out


def an_arch(head, trunk_=None, algo='rainbow', widths=(32,)):
    return arch_tools.build_arch(list(widths), constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                 algo=algo, head=head, trunk=PLAIN if trunk_ is None else trunk_)


def observations(m=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(m, constants.OBS_LEN, generator=generator)


def small_config(monkeypatch, algo, **env):
    monkeypatch.setenv('SNEK_ALGO', algo)
    for key, value in env.items():
        monkeypatch.setenv('SNEK_' + key, str(value))
    config = train.build_config()
    config.update({'seed': 3, 'replay_buffer_max_length': 500, 'initial_collect_steps': 40, 'batch_size': 8})
    return config


def build(monkeypatch, algo, **env):
    config = small_config(monkeypatch, algo, **env)
    module = train.ALGOS[algo]
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo=algo, **module.arch_fields(config))
    return module.build(config, arch), config, arch


# ---------------------------------------------------------------- NoisyLinear

def test_with_noise_off_a_noisy_layer_is_a_linear_over_its_mu_weights():
    layer = NoisyLinear(6, 4, generator=torch.Generator().manual_seed(1))
    plain = nn.Linear(6, 4)
    with torch.no_grad():
        plain.weight.copy_(layer.weight_mu)
        plain.bias.copy_(layer.bias_mu)
    layer.active = False
    x = torch.randn(5, 6)
    assert torch.allclose(layer(x), plain(x))


def test_two_noisy_forwards_differ_and_two_quiet_ones_agree():
    layer = NoisyLinear(6, 4, generator=torch.Generator().manual_seed(1))
    x = torch.randn(5, 6)
    assert not torch.allclose(layer(x), layer(x))
    layer.active = False
    assert torch.allclose(layer(x), layer(x))


def test_the_noisy_initialisation_is_the_papers():
    layer = NoisyLinear(16, 4, sigma_zero=0.5)
    assert torch.allclose(layer.weight_sigma, torch.full_like(layer.weight_sigma, 0.5 / 4.0))
    assert layer.weight_mu.abs().max() <= 1.0 / 4.0


# ---------------------------------------------------------------- the network

def test_the_dueling_combine_has_zero_mean_advantage_over_actions_per_atom():
    net = network.build(an_arch(C51), seed=3)
    set_noise(net, False)
    obs = observations(4)
    logits = net.logits(obs)                                              # (m, actions, atoms)
    assert logits.shape == (4, constants.NUM_ACTIONS, 5)
    value = net.value(net.features(obs)).view(4, 5)
    # mean_a(V + A - mean_a A) = V, for every atom separately.
    assert torch.allclose(logits.mean(dim=1), value, atol=1e-6)


def test_a_single_stream_net_has_no_value_stream_and_matches_a2s_shape():
    net = network.build(an_arch(C51, trunk(dueling=False, noisy=False)), seed=3)
    assert net.value is None and isinstance(net.advantage, nn.Linear)
    assert net.log_probs(observations()).shape == (3, constants.NUM_ACTIONS, 5)
    assert torch.allclose(net.probs(observations()).sum(dim=2), torch.ones(3, constants.NUM_ACTIONS))


def test_the_quantile_and_iqn_heads_read_in_the_dist_agents_shapes():
    quant = network.build(an_arch(QUANT), seed=3)
    assert quant.quantiles(observations()).shape == (3, constants.NUM_ACTIONS, 8)
    assert quant.fractions('cpu')[0] == pytest.approx(1.0 / 16.0)
    iqn = network.build(an_arch(IQN), seed=3)
    features = iqn.features(observations())
    taus = iqn.sample_taus(3, 6)
    assert iqn.quantiles_at(features, taus).shape == (3, 6, constants.NUM_ACTIONS)
    assert iqn.q_values(observations()).shape == (3, constants.NUM_ACTIONS)


def test_spectral_norm_bounds_the_residual_linears_at_one_and_leaves_the_head_alone():
    net = network.build(an_arch(IQN, trunk(residual=True, blocks=2)), seed=3)
    net.train()
    for block in net.trunk.blocks:
        for linear in (block.first, block.second):
            assert torch.linalg.matrix_norm(linear.weight, ord=2).item() == pytest.approx(1.0, abs=0.02)
    assert not hasattr(net.advantage, 'parametrizations') and not hasattr(net.value, 'parametrizations')
    assert not hasattr(net.trunk.stem, 'parametrizations')
    plain = network.build(an_arch(IQN, trunk(residual=True, noisy=False)), seed=3)
    assert isinstance(plain.advantage, nn.Linear) and not hasattr(plain.advantage, 'parametrizations')
    assert not hasattr(plain.value, 'parametrizations')


def test_the_spectral_estimates_are_converged_at_build_and_in_a_fresh_target(monkeypatch):
    """The target stays in eval mode until its first hard copy, so the estimates it is built from are the
    ones it uses: they must be converged before any training forward (they were 5 to 60 until 2026-09-23)."""
    for seed in (0, 1, 2):
        net = network.build(an_arch(IQN, trunk(residual=True, blocks=3)), seed=seed)
        net.eval()
        for block in net.trunk.blocks:
            for linear in (block.first, block.second):
                assert torch.linalg.matrix_norm(linear.weight.detach(), ord=2).item() == pytest.approx(1.0, abs=0.05)
    algo, config, arch = build(monkeypatch, 'btr', FC_LAYERS=32, COLLECT_ENVS=2)
    for block in algo.agent.target.trunk.blocks:
        for linear in (block.first, block.second):
            assert torch.linalg.matrix_norm(linear.weight, ord=2).item() == pytest.approx(1.0, abs=0.05)


def test_stream_hidden_gives_each_stream_its_own_noisy_hidden_layer():
    net = network.build(an_arch(C51, trunk(stream_hidden=True), widths=(24, 32)), seed=3)
    assert [layer.out_features for layer in net.trunk.hidden] == [24]      # the last width moved out
    for stream in (net.advantage, net.value):
        assert isinstance(stream[0], NoisyLinear) and isinstance(stream[2], NoisyLinear)
        assert (stream[0].in_features, stream[0].out_features) == (24, 32)
    assert net.advantage[0].weight_mu.data_ptr() != net.value[0].weight_mu.data_ptr()
    assert net.log_probs(observations()).shape == (3, constants.NUM_ACTIONS, 5)
    residual = network.build(an_arch(IQN, trunk(residual=True, stream_hidden=True)), seed=3)
    assert residual.trunk.width == 32 and residual.advantage[0].in_features == 32
    from algos.dqn import resets
    trunk_names, head_names = resets.partition(net)
    assert not any(name.startswith(('advantage', 'value')) for name in trunk_names)


def test_a_single_stream_net_with_stream_hidden_is_qnet_weight_for_weight():
    from algos.dqn import net as qnet
    net = network.build(an_arch(C51, trunk(dueling=False, noisy=False, stream_hidden=True)), seed=3)
    reference = qnet.QNet(constants.OBS_LEN, [32], 1, seed=3)
    assert torch.equal(net.advantage[0].weight, reference.hidden[0].weight)


def test_a_trunk_without_the_field_is_the_one_linear_layout():
    net = network.build(an_arch(C51, trunk(noisy=False)), seed=3)
    assert isinstance(net.advantage, nn.Linear) and isinstance(net.value, nn.Linear)


def test_a_residual_block_is_the_identity_at_a_zero_final_layer():
    block = network.ResidualBlock(8, spectral_norm=False, layer_norm=False, generator=None)
    with torch.no_grad():
        block.second.weight.zero_()
        block.second.bias.zero_()
    x = torch.randn(5, 8)
    assert torch.allclose(block(x), x)


def test_layer_norm_is_a_flag_on_the_blocks_input():
    with_norm = network.build(an_arch(IQN, trunk(residual=True, layer_norm=True)), seed=3)
    without = network.build(an_arch(IQN, trunk(residual=True, layer_norm=False)), seed=3)
    assert all(isinstance(block.norm, nn.LayerNorm) for block in with_norm.trunk.blocks)
    assert all(block.norm is None for block in without.trunk.blocks)


def test_the_plain_trunk_is_qnets_stack_to_the_initialiser():
    from algos.dqn import net as qnet
    net = network.build(an_arch(C51, trunk(dueling=False, noisy=False)), seed=3)
    reference = qnet.QNet(constants.OBS_LEN, [32], 1, seed=3)
    assert torch.equal(net.trunk.hidden[0].weight, reference.hidden[0].weight)


def test_the_seed_pins_the_weights_and_a_different_seed_moves_them():
    a, b, c = (network.build(an_arch(IQN, trunk(residual=True)), seed=s) for s in (3, 3, 4))
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.advantage.weight_mu, c.advantage.weight_mu)


def test_the_greedy_policy_fn_is_quiet_and_deterministic_across_calls():
    net = network.build(an_arch(C51), seed=3)
    policy = network.greedy_policy_fn(net)
    obs = observations(200).numpy()
    set_noise(net, True)                                                   # as an update would leave it
    first = policy(obs)
    assert all(not layer.active for layer in net.noisy_layers)
    assert np.array_equal(first, policy(obs))
    set_noise(net, True)
    with torch.no_grad():
        assert not torch.allclose(net.q_values(observations(200)), net.q_values(observations(200)))


def test_a_missing_head_or_trunk_is_refused_by_name():
    with pytest.raises(ValueError):
        network.build(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='rainbow',
                                            trunk=PLAIN))
    with pytest.raises(ValueError):
        network.build(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='rainbow',
                                            head=C51))
    with pytest.raises(ValueError):
        network.build(an_arch({'type': 'fqf', 'embedding': 16, 'n': 8}))


# ---------------------------------------------------------------- the agent

def test_the_agent_acts_with_the_noise_on_and_measures_with_it_off(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow')
    agent = algo.agent
    obs = observations(8).numpy()
    agent.greedy_actions(obs)
    assert all(layer.active for layer in agent.net.noisy_layers)
    policy = algo.policy_fn
    policy(obs)
    assert all(not layer.active for layer in agent.net.noisy_layers)
    agent.greedy_actions(obs)
    assert all(layer.active for layer in agent.net.noisy_layers)


def test_with_noisy_off_the_agent_acts_quietly(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow', RAINBOW_NOISY=0, MIN_EPSILON=0.01)
    assert not algo.agent.net.noisy_layers
    assert all(isinstance(layer, (nn.Linear, nn.ReLU)) for layer in algo.agent.net.advantage)


def test_without_double_q_the_target_action_is_the_target_nets_argmax(monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr', RAINBOW_DOUBLE=0, MUNCHAUSEN_ALPHA=0, RAINBOW_NOISY=0,
                               MIN_EPSILON=0.01, RAINBOW_HEAD='quantile')
    agent = algo.agent
    agent.target.load_state_dict(network.build(arch, seed=11).state_dict())   # a target that disagrees
    obs = observations(64)
    with torch.no_grad():
        expected = agent.target.q_values(obs).argmax(dim=1)
        online = agent.net.q_values(obs).argmax(dim=1)
    assert torch.equal(agent._double_q_target_action(obs), expected)
    assert not torch.equal(expected, online)
    weights, shifts, munchausen = agent._next_action_mixture(obs, torch.zeros(64, dtype=torch.long), obs)
    assert torch.equal(weights.argmax(dim=1), expected) and float(shifts.abs().sum()) == 0.0


def test_with_double_q_the_target_action_is_the_online_argmax(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow', RAINBOW_NOISY=0, MIN_EPSILON=0.01, RAINBOW_HEAD='quantile')
    agent = algo.agent
    agent.target.load_state_dict(network.build(arch, seed=11).state_dict())
    obs = observations(64)
    with torch.no_grad():
        assert torch.equal(agent._double_q_target_action(obs), agent.net.q_values(obs).argmax(dim=1))


def test_the_tau_counts_reach_the_agent(monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr', DIST_TAU_SAMPLES=6, DIST_TAU_PRIME_SAMPLES=5, DIST_POLICY_SAMPLES=4)
    assert (algo.agent.n_tau, algo.agent.n_tau_prime, algo.agent.net.k) == (6, 5, 4)
    assert arch['head'] == {'type': 'iqn', 'embedding': 64, 'n_tau': 6, 'k': 4}


@pytest.mark.parametrize('name', rainbow_algo.NAMES)
def test_each_name_prefills_advances_learns_and_checkpoints(name, monkeypatch, tmp_path):
    algo, config, arch = build(monkeypatch, name, COLLECT_ENVS=4, REPLAY_RATIO=0.25)
    assert algo.prefill() >= 40
    steps, transitions = algo.advance()
    # At least one move per lane; more when an episode ends and the n-step window flushes its tail.
    assert steps == 1 and transitions >= 4
    assert algo.agent.train_step >= 1                                      # 4 transitions x 0.25
    assert name in algo.describe()
    checkpoints.save(str(tmp_path), 1, algo.net)
    arch_tools.write_arch(str(tmp_path), arch)
    policy, restored_arch, step = restore.restore(str(tmp_path))
    obs = observations(50).numpy()
    # IQN's mean read draws its fractions from torch's global generator (A4's head does the same), so
    # the two reads are seeded alike to compare the weights and not the draw.
    torch.manual_seed(0)
    restored_actions = policy(obs)
    torch.manual_seed(0)
    assert np.array_equal(restored_actions, algo.policy_fn(obs))
    assert restored_arch['trunk'] == arch['trunk'] and step == 1


def test_a_munchausen_update_with_the_iqn_dueling_head_runs(monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr', COLLECT_ENVS=2, REPLAY_RATIO=0.5)
    algo.prefill()
    algo.advance()
    assert algo.agent.munchausen_alpha == 0.9 and algo.agent.train_step >= 1


# ---------------------------------------------------------------- the two names and their defaults

def test_both_names_are_registered_everywhere():
    for name in rainbow_algo.NAMES:
        assert train.ALGOS[name].NAME == name and name in restore.ALGORITHMS


def test_rainbows_defaults_are_the_papers(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'rainbow')
    config = train.build_config()
    assert (config['learning_rate'], config['adam_epsilon'], config['batch_size']) == (6.25e-5, 1.5e-4, 32)
    assert (config['n_step_update'], config['target_update_period'], config['replay_ratio']) == (3, 2000, 0.25)
    assert (config['priority_exponent'], config['is_beta'], config['is_beta_final']) == (0.5, 0.4, 1.0)
    assert (config['initial_epsilon'], config['min_epsilon'], config['guided_fraction']) == (0.0, 0.0, 0.0)
    assert config['fork'].branches == 1 and config['collect_envs'] == 1 and config['discount'] == 0.99
    assert config['rainbow_head'] == 'c51' and config['rainbow_double'] and not config['btr_residual']
    assert config['munchausen_alpha'] == 0.0 and config['replay_buffer_max_length'] == 1000000
    assert config['beta_anneal_steps'] == 0 and config['rainbow_stream_hidden']
    assert config['rainbow_epsilon_decay'] == 'linear' and config['rainbow_munchausen_logpi'] == 'target'


def test_btrs_defaults_are_the_papers(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'btr')
    config = train.build_config()
    assert (config['learning_rate'], config['adam_epsilon'], config['batch_size']) == (1e-4, 0.005 / 256, 256)
    assert (config['collect_envs'], config['replay_ratio'], config['target_update_period']) == (64, 1.0 / 64.0, 500)
    # The released code's importance exponent: `PER.py` uses alpha (0.2), not the declared beta 0.45.
    assert (config['priority_exponent'], config['is_beta'], config['is_beta_final']) == (0.2, 0.2, 0.2)
    assert config['rainbow_epsilon_decay'] == 'geometric' and config['rainbow_munchausen_logpi'] == 'online'
    assert config['rainbow_stream_hidden']
    assert (config['initial_epsilon'], config['min_epsilon'], config['epsilon_anneal_steps']) == (1.0, 0.01, 2000000)
    assert config['rainbow_epsilon_zero_at'] == 0.5 and config['discount'] == 0.997
    assert config['rainbow_head'] == 'iqn' and not config['rainbow_double'] and config['btr_residual']
    assert config['munchausen_alpha'] == 0.9 and config['gradient_clipping'] == 10.0
    assert config['replay_buffer_max_length'] == 2 ** 20 and config['initial_collect_steps'] == 200000
    assert (config['dist_tau_samples'], config['dist_tau_prime_samples'], config['dist_policy_samples']) == (8, 8, 8)


def test_btr_at_rainbows_flags_is_rainbow_weight_for_weight(monkeypatch):
    """Gate 2 of `c-value-stack.md` §4: the C2 ablations are exact because the two names build one net."""
    rainbow_config = small_config(monkeypatch, 'rainbow')
    rainbow_fields = train.ALGOS['rainbow'].arch_fields(rainbow_config)
    btr_config = small_config(monkeypatch, 'btr', RAINBOW_HEAD='c51', BTR_RESIDUAL=0, RAINBOW_DOUBLE=1,
                              MUNCHAUSEN_ALPHA=0)
    btr_fields = train.ALGOS['btr'].arch_fields(btr_config)
    assert btr_fields == rainbow_fields
    a = network.build(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='rainbow', **rainbow_fields), seed=3)
    b = network.build(arch_tools.build_arch([32], 3, constants.OBS_LEN, constants.OBS_ERA, algo='btr', **btr_fields), seed=3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_the_sidecar_carries_the_trunk_and_it_is_in_the_signature(monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr')
    assert set(arch['trunk']) == set(network.TRUNK_FIELDS + network.OPTIONAL_TRUNK_FIELDS)
    other = dict(arch)
    other['trunk'] = dict(arch['trunk'], residual=False)
    assert arch_tools.signature(other) != arch_tools.signature(arch)


# ---------------------------------------------------------------- the epsilon floor and the zero point

def test_min_epsilon_zero_is_refused_with_noisy_off_and_accepted_with_it_on(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'rainbow')
    monkeypatch.setenv('SNEK_MIN_EPSILON', '0')
    monkeypatch.setenv('SNEK_RAINBOW_NOISY', '0')
    with pytest.raises(ValueError, match='hard floor'):
        train.build_config()
    monkeypatch.setenv('SNEK_RAINBOW_NOISY', '1')
    assert train.build_config()['min_epsilon'] == 0.0
    monkeypatch.setenv('SNEK_MIN_EPSILON', str(schedules.EPSILON_HARD_FLOOR))
    monkeypatch.setenv('SNEK_RAINBOW_NOISY', '0')
    assert train.build_config()['min_epsilon'] == schedules.EPSILON_HARD_FLOOR


def test_epsilon_drops_to_zero_at_the_fraction_of_the_runs_moves(monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr', COLLECT_ENVS=4, MAX_STEPS=100, EPSILON_ANNEAL_STEPS=1000)
    assert algo.epsilon_zero_after == 200                                  # 0.5 x 100 steps x 4 lanes
    algo.moves = 199
    assert algo._linear_epsilon() > 0.0
    algo.moves = 200
    assert algo._linear_epsilon() == 0.0
    assert algo.on_eval([], {'avg_reward': 0.0, 'perfect': 0.0})['epsilon'] == 0.0


def test_a_zero_fraction_never_zeroes_epsilon(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow', RAINBOW_EPSILON_ZERO_AT=0, MIN_EPSILON=0.05,
                               EPSILON_ANNEAL_STEPS=10, MAX_STEPS=100)
    algo.moves = 10 ** 9
    assert algo.epsilon_zero_after == 0 and algo._linear_epsilon() == 0.05


def test_foreign_knobs_are_refused_by_name(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'rainbow')
    monkeypatch.setenv('SNEK_SAC_ALPHA', '0.1')
    with pytest.raises(ValueError, match='SNEK_SAC_ALPHA'):
        train.build_config()


def test_an_unknown_head_is_refused_by_name(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'rainbow')
    monkeypatch.setenv('SNEK_RAINBOW_HEAD', 'fqf')
    with pytest.raises(ValueError, match='SNEK_RAINBOW_HEAD'):
        train.build_config()


# ---------------------------------------------------------------- the update's objective (2026-09-23)

from algos.dist import losses as dist_losses
from algos.rainbow import agent as rainbow_agent


def test_munchausen_quantile_target_averages_actions_inside_each_sample(monkeypatch):
    """M-IQN / BTR: `sum_a' pi(a') (Z_j(a') - tau log pi(a'))` per sample j -- M targets, not A x M."""
    algo, config, arch = build(monkeypatch, 'btr', RAINBOW_HEAD='quantile', RAINBOW_NOISY=0, MIN_EPSILON=0.01)
    agent = algo.agent
    obs, action = observations(1), torch.tensor([1])
    reward, discount = torch.tensor([0.5]), torch.tensor([0.9])
    z = torch.tensor([[[1.0, 3.0], [2.0, 2.5], [0.0, 4.0]]])                 # (B 1, A 3, M 2)
    target = agent._quantile_target(obs, action, reward, discount, obs, z)
    assert target.shape == (1, 2)
    tau = agent.munchausen_tau
    q = z.mean(dim=2)
    pi, log_pi = torch.softmax(q / tau, dim=1), tau * torch.log_softmax(q / tau, dim=1)
    soft = (pi.unsqueeze(2) * (z - log_pi.unsqueeze(2))).sum(dim=1)
    expected = reward + agent._munchausen_reward(obs, action) + discount * soft
    assert torch.allclose(target, expected)


def test_the_quantile_loss_sums_online_and_averages_targets_and_prioritises_by_pairwise_td():
    online, taus = torch.tensor([[0.0, 1.0]]), torch.tensor([[0.25, 0.75]])
    target = torch.tensor([[2.0, 4.0, 6.0]])
    loss, priority = rainbow_agent.quantile_loss(online, taus, target, kappa=1.0)
    assert torch.allclose(loss, 2 * dist_losses.quantile_huber(online, taus, target, kappa=1.0))
    # |u| pairs: online 0 -> 2, 4, 6; online 1 -> 1, 3, 5. Summed over online, averaged over targets.
    assert priority.item() == pytest.approx((2 + 4 + 6 + 1 + 3 + 5) / 3.0)


def test_the_c51_priority_is_the_kl_not_the_cross_entropy():
    matched = torch.full((1, 5), 0.2)
    assert rainbow_agent.categorical_kl(matched.log(), matched).item() == pytest.approx(0.0, abs=1e-6)
    assert dist_losses.categorical_cross_entropy(matched.log(), matched).item() == pytest.approx(np.log(5.0))
    sharp = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    assert rainbow_agent.categorical_kl(matched.log(), sharp).item() == pytest.approx(np.log(5.0))


def test_update_feeds_the_priority_back_not_the_loss(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow', RAINBOW_NOISY=0, MIN_EPSILON=0.01)
    agent = algo.agent
    seen = {}

    def categorical(obs, action, reward, discount, next_obs):
        loss = agent.net.q_values(obs).sum(dim=1) * 0.0 + 7.0
        seen['priority'] = torch.arange(obs.shape[0], dtype=torch.float32)
        return loss, seen['priority']

    monkeypatch.setattr(agent, '_categorical', categorical)
    batch = {'obs': observations(4).numpy(), 'next_obs': observations(4, seed=1).numpy(),
             'action': np.zeros(4, dtype=np.int64), 'reward': np.zeros(4, dtype=np.float32),
             'discount': np.full(4, 0.9, dtype=np.float32)}
    priorities, metrics = agent.update(batch)
    assert np.array_equal(priorities, seen['priority'].numpy()) and metrics['loss'] == pytest.approx(7.0)


@pytest.mark.parametrize('source', ['online', 'target'])
def test_the_munchausen_log_policy_is_read_off_the_named_net(source, monkeypatch):
    algo, config, arch = build(monkeypatch, 'btr', RAINBOW_HEAD='quantile', RAINBOW_NOISY=0, MIN_EPSILON=0.01,
                               RAINBOW_MUNCHAUSEN_LOGPI=source)
    agent = algo.agent
    agent.target.load_state_dict(network.build(arch, seed=11).state_dict())
    obs, action = observations(16), torch.zeros(16, dtype=torch.long)
    agent.net.eval()                                   # no power-iteration step between the two reads
    net, other = (agent.net, agent.target) if source == 'online' else (agent.target, agent.net)
    with torch.no_grad():
        _, other_log_pi = agent._soft_policy(other.q_values(obs))
        assert not torch.allclose(agent._munchausen_reward(obs, action),
                                  torch.clamp(other_log_pi[:, 0], min=agent.munchausen_l0, max=0.0) * agent.munchausen_alpha)
        _, log_pi = agent._soft_policy(net.q_values(obs))
        expected = torch.clamp(log_pi[:, 0], min=agent.munchausen_l0, max=0.0) * agent.munchausen_alpha
        assert torch.allclose(agent._munchausen_reward(obs, action), expected)


def test_an_unknown_log_policy_source_is_refused(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'btr')
    monkeypatch.setenv('SNEK_RAINBOW_MUNCHAUSEN_LOGPI', 'both')
    with pytest.raises(ValueError, match='SNEK_RAINBOW_MUNCHAUSEN_LOGPI'):
        train.build_config()


# ---------------------------------------------------------------- the clocks (2026-09-23)

def test_geometric_epsilon_is_btrs_recurrence():
    eps, steps = 1.0, 10
    for moves in range(1, 40):
        eps = max(eps - (eps - 0.01) / steps, 0.01)
        assert rainbow_algo.geometric_epsilon(moves, 1.0, 0.01, steps) == pytest.approx(eps)
    assert rainbow_algo.geometric_epsilon(2000000, 1.0, 0.01, 2000000) == pytest.approx(0.01 + 0.99 / np.e, abs=1e-6)


def test_btr_anneals_geometrically_and_rainbow_linearly(monkeypatch):
    btr, _, _ = build(monkeypatch, 'btr', COLLECT_ENVS=2, MAX_STEPS=10 ** 6, EPSILON_ANNEAL_STEPS=1000)
    btr.moves = 1000
    assert btr._linear_epsilon() == pytest.approx(0.01 + 0.99 * (1 - 1 / 1000) ** 1000)
    linear, _, _ = build(monkeypatch, 'btr', COLLECT_ENVS=2, MAX_STEPS=10 ** 6, EPSILON_ANNEAL_STEPS=1000,
                         RAINBOW_EPSILON_DECAY='linear')
    linear.moves = 1000
    assert linear._linear_epsilon() == pytest.approx(0.01)


def test_rainbows_beta_anneals_over_the_runs_cap(monkeypatch):
    algo, config, arch = build(monkeypatch, 'rainbow', MAX_STEPS=1000)
    assert algo.buffer.beta_anneal_steps == 250 == config['beta_anneal_steps']   # 1000 moves x 0.25
    assert algo.buffer.beta_for(250) == pytest.approx(1.0) and algo.buffer.beta_for(125) == pytest.approx(0.7)
    pinned, _, _ = build(monkeypatch, 'rainbow', MAX_STEPS=1000, BETA_ANNEAL_STEPS=77)
    assert pinned.buffer.beta_anneal_steps == 77
