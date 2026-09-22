"""BBF (`algos/bbf/`): the sequential replay's sample-time n-step return and SPR sequences, the agent's EMA
target, SPR loss, resets and cycle, the seam, the knobs and the checkpoint. `d-data-efficiency.md` §7."""

import numpy as np
import pytest
import torch

import train
from algos.bbf import agent as bbf_agent
from algos.bbf import algo as bbf_algo
from algos.bbf.agent import BbfAgent, SprHeads, ema_update, spr_loss
from algos.bbf.replay import SequentialReplay
from algos.dqn import resets
from algos.rainbow import net as network
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore

OBS = 4


def row(t):
    return np.full(OBS, float(t), dtype=np.float32)


def a_replay(lanes=1, horizon=5, capacity=64, alpha=0.5, seed=0):
    return SequentialReplay(capacity, OBS, lanes=lanes, horizon=horizon, alpha=alpha, seed=seed)


def fill(replay, rewards, terminals=(), lanes=1):
    """Row t (per lane): obs t, next_obs t+1, action t % 3, reward rewards[t]; a terminal row's discount is 0 and
    the row after it starts a new episode (its obs need not continue)."""
    for t, reward in enumerate(rewards):
        for lane in range(lanes):
            start = 100 * lane
            previous_terminal = (t - 1) in terminals
            obs = row(start + t) if not previous_terminal else row(start + t + 0.5)
            replay.add(obs, t % 3, reward, row(start + t + 1), 0.0 if t in terminals else 0.99)


def small_config(monkeypatch, **env):
    monkeypatch.setenv('SNEK_ALGO', 'bbf')
    for key, value in env.items():
        monkeypatch.setenv('SNEK_' + key, str(value))
    config = train.build_config()
    config.update({'seed': 3, 'replay_buffer_max_length': 2000, 'initial_collect_steps': 60, 'batch_size': 8})
    return config


def build(monkeypatch, **env):
    config = small_config(monkeypatch, **env)
    module = train.ALGOS['bbf']
    arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                 algo='bbf', **module.arch_fields(config))
    return module.build(config, arch), config, arch


def an_arch(widths=(16,), atoms=11):
    return arch_tools.build_arch(list(widths), constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA, algo='bbf',
                                 head={'type': 'c51', 'atoms': atoms, 'v_min': -10.0, 'v_max': 110.0},
                                 trunk={'dueling': True, 'noisy': False, 'noisy_sigma': 0.5, 'residual': False,
                                        'blocks': 1, 'spectral_norm': False, 'layer_norm': False})


def an_agent(**kwargs):
    defaults = dict(seed=1, learning_rate=1e-2, projection_dim=8, spr_steps=2, reset_interval=0)
    defaults.update(kwargs)
    return BbfAgent(an_arch(), **defaults)


def a_batch(m=6, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return {'obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'next_obs': rng.random((m, constants.OBS_LEN)).astype(np.float32),
            'action': rng.integers(0, 3, m).astype(np.int64),
            'reward': rng.choice([0.0, 1.0, 100.0], m).astype(np.float32),
            'discount': rng.choice([0.0, 0.97], m).astype(np.float32),
            'spr_next_obs': rng.random((m, k, constants.OBS_LEN)).astype(np.float32),
            'spr_action': rng.integers(0, 3, (m, k)).astype(np.int64),
            'spr_mask': np.ones((m, k), dtype=np.float32)}


# ---------------------------------------------------------------- the replay

def test_the_replay_insists_that_each_lane_is_a_sequence():
    replay = a_replay(lanes=2)
    fill(replay, [0.0, 1.0, 2.0], lanes=2)
    with pytest.raises(ValueError, match='does not continue'):
        replay.add(row(42), 0, 0.0, row(43), 0.99)
    # After a terminal the next row may start anywhere.
    replay.add(row(3), 0, 0.0, row(4), 0.0)
    replay.add(row(103), 0, 0.0, row(104), 0.0)
    replay.add(row(77), 0, 0.0, row(78), 0.99)


def test_the_n_step_return_is_summed_when_drawn_and_stops_at_the_terminal():
    replay = a_replay(horizon=4, capacity=64)
    fill(replay, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0], terminals=(3,))
    # Force the draw to rows 1 and 2 by priorities.
    replay.tree.set(np.arange(replay.size), np.zeros(replay.size))
    replay.tree.set([1, 2], [1.0, 1.0])
    batch, indexes, weights = replay.sample(16, n_step=3, gamma=0.5, spr_steps=2)
    by_row = {int(i): k for k, i in enumerate(indexes)}
    assert set(by_row) == {1, 2}
    one, two = by_row[1], by_row[2]
    assert batch['reward'][one] == pytest.approx(2.0 + 0.5 * 4.0 + 0.25 * 8.0)        # rows 1, 2, 3 -- 3 is terminal
    assert batch['discount'][one] == 0.0                                                # the window crossed a terminal
    assert batch['reward'][two] == pytest.approx(4.0 + 0.5 * 8.0)                      # row 4 is a new episode: excluded
    assert batch['discount'][two] == 0.0
    assert np.array_equal(batch['next_obs'][one], row(4))                               # next_obs of row 3, n - 1 on
    # SPR: step k reads obs at t + k; masked once the episode has ended.
    assert np.array_equal(batch['spr_next_obs'][one][0], row(2)) and np.array_equal(batch['spr_next_obs'][one][1], row(3))
    assert batch['spr_mask'][one].tolist() == [1.0, 1.0]
    assert batch['spr_mask'][two].tolist() == [1.0, 0.0]                                # row 3 terminal: obs at t+2 is the next game
    assert batch['spr_action'][two].tolist() == [2 % 3, 3 % 3]
    # The same rows at a different n and gamma give a different return: the sum is not stored.
    batch1, indexes1, _ = replay.sample(16, n_step=1, gamma=0.9, spr_steps=0)
    k = [int(i) for i in indexes1].index(2)
    assert batch1['reward'][k] == pytest.approx(4.0) and batch1['discount'][k] == pytest.approx(0.9)
    assert np.array_equal(batch1['next_obs'][k], row(3)) and 'spr_mask' not in batch1


def test_a_full_window_bootstraps_at_gamma_to_the_n():
    replay = a_replay(horizon=3, capacity=64)
    fill(replay, [1.0] * 12)
    replay.tree.set(np.arange(replay.size), np.zeros(replay.size))
    replay.tree.set([0], [1.0])
    batch, indexes, _ = replay.sample(4, n_step=3, gamma=0.5, spr_steps=3)
    assert set(int(i) for i in indexes) == {0}
    assert batch['discount'][0] == pytest.approx(0.125) and batch['reward'][0] == pytest.approx(1.75)
    assert np.array_equal(batch['next_obs'][0], row(3)) and batch['spr_mask'][0].tolist() == [1.0, 1.0, 1.0]


def test_the_newest_rows_are_not_drawable_until_their_successors_exist():
    replay = a_replay(lanes=2, horizon=3, capacity=64)
    assert replay.sample(4, 1, 0.9, 0) is None
    fill(replay, [1.0] * 5, lanes=2)                      # 10 rows; the newest 3 * 2 lack successors
    assert replay.n_drawable == 4
    assert replay.drawable(np.arange(10)).tolist() == [True] * 4 + [False] * 6
    batch, indexes, weights = replay.sample(32, n_step=3, gamma=0.9, spr_steps=3)
    assert set(int(i) for i in indexes) <= {0, 1, 2, 3}
    assert weights.max() == pytest.approx(1.0) and weights.min() > 0.0


def test_priorities_shape_the_draw_and_the_weights_are_normalised_by_the_max():
    replay = a_replay(horizon=1, capacity=64, alpha=1.0)
    fill(replay, [1.0] * 20)
    replay.update_priorities(np.arange(19), np.r_[np.full(18, 0.1), 10.0])
    batch, indexes, weights = replay.sample(64, 1, 0.9, 0)
    assert 0.6 < (indexes == 18).mean() < 0.98 and (indexes != 18).any()
    # Dopamine's form: p ** -0.5 over the batch maximum, so the rarest row drawn is exactly 1 and the heavy row is
    # sqrt(0.1 / 10) of it; a mean-1 normalisation would put the common rows well below 1.
    assert weights[indexes != 18].max() == pytest.approx(1.0)
    assert weights[indexes == 18].max() == pytest.approx((0.1 / 10.0) ** 0.5, rel=1e-3)


def test_the_replay_round_trips_through_disk(tmp_path):
    replay = a_replay(lanes=2, horizon=2, capacity=32)
    fill(replay, [1.0, 2.0, 3.0, 4.0, 5.0], terminals=(2,), lanes=2)
    replay.update_priorities(np.arange(4), [0.1, 0.2, 0.3, 0.4])
    replay.save(str(tmp_path))
    back = a_replay(lanes=2, horizon=2, capacity=32)
    assert back.load(str(tmp_path))
    for name in ('obs', 'next_obs', 'action', 'reward'):
        assert np.array_equal(getattr(back, name)[:replay.size], getattr(replay, name)[:replay.size])
    assert np.array_equal(back.done[:replay.size - 2], replay.done[:replay.size - 2])
    assert (back.size, back.write, back.max_priority) == (replay.size, replay.write, replay.max_priority)
    assert np.array_equal(back.tree.nodes, replay.tree.nodes)
    with pytest.raises(ValueError, match='lane'):
        a_replay(lanes=1, horizon=2, capacity=32).load(str(tmp_path))
    # A resumed buffer ends each lane's saved sequence: the collector's next row is a fresh game.
    assert back.done[replay.size - 2:replay.size].all() and not replay.done[replay.size - 2:replay.size].any()
    back.add(row(999), 0, 0.0, row(1000), 0.99)
    back.add(row(1999), 0, 0.0, row(2000), 0.99)


# ---------------------------------------------------------------- the agent

def test_the_targets_move_by_tau_and_only_by_tau():
    agent = an_agent(target_tau=0.25, spr_weight=1.0)
    before_net = {n: p.clone() for n, p in agent.net.named_parameters()}
    before_target = {n: p.clone() for n, p in agent.target.named_parameters()}
    before_spr_target = {n: p.clone() for n, p in agent.target_spr.named_parameters()}
    agent.update(a_batch())
    for name, parameter in agent.target.named_parameters():
        online = dict(agent.net.named_parameters())[name]
        assert torch.allclose(parameter, 0.75 * before_target[name] + 0.25 * online, atol=1e-6)
        assert not torch.allclose(online, before_net[name])
    for name, parameter in agent.target_spr.named_parameters():
        online = dict(agent.spr.named_parameters())[name]
        assert torch.allclose(parameter, 0.75 * before_spr_target[name] + 0.25 * online, atol=1e-6)


def test_the_spr_loss_is_zero_for_a_perfect_prediction_and_ignores_masked_steps():
    predictions = torch.randn(3, 4, 6)
    assert torch.allclose(spr_loss(predictions, predictions * 3.0, torch.ones(3, 4)), torch.zeros(3), atol=1e-6)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 3)
    other = torch.randn(3, 4, 6)
    full = spr_loss(predictions, other, torch.ones(3, 4))
    masked = spr_loss(predictions, other, mask)
    assert (masked <= full).all() and (masked > 0).all()
    # Each in-episode step is 2 - 2 cos: opposite vectors cost 4 each.
    assert torch.allclose(spr_loss(predictions, -predictions, mask), torch.full((3,), 8.0), atol=1e-5)


def test_the_agents_spr_loss_uses_the_target_encoder_and_the_mask():
    agent = an_agent(spr_weight=1.0)
    with torch.no_grad():                         # the two encoders must differ for the test to see which is read
        for parameter in agent.target.trunk.parameters():
            parameter.add_(0.5)
    batch = a_batch(k=2)
    obs = torch.as_tensor(batch['obs'])
    loss = agent._spr_loss(obs, batch)
    with torch.no_grad():
        latents = agent.spr.rollout(agent.net.features(obs), torch.as_tensor(batch['spr_action']))
        predictions = agent.spr.predict(latents)
        flat = torch.as_tensor(batch['spr_next_obs']).reshape(-1, constants.OBS_LEN)
        targets = agent.target_spr.projection(agent.target.features(flat)).view(6, 2, -1)
        online_targets = agent.target_spr.projection(agent.net.features(flat)).view(6, 2, -1)
    assert torch.allclose(loss, spr_loss(predictions, targets, torch.ones(6, 2)), atol=1e-6)
    assert not torch.allclose(loss, spr_loss(predictions, online_targets, torch.ones(6, 2)), atol=1e-4)
    batch['spr_mask'][:] = 0.0
    assert torch.allclose(agent._spr_loss(obs, batch), torch.zeros(6))


def test_spr_weight_zero_leaves_the_spr_heads_out_of_the_step():
    agent = an_agent(spr_weight=0.0, weight_decay=0.0)
    before = {n: p.clone() for n, p in agent.spr.named_parameters()}
    _, metrics = agent.update(a_batch())
    assert metrics['spr_loss'] == 0.0
    assert all(torch.equal(before[n], p) for n, p in agent.spr.named_parameters())
    trained = an_agent(spr_weight=1.0, weight_decay=0.0)
    before = {n: p.clone() for n, p in trained.spr.named_parameters()}
    trained.update(a_batch())
    assert not all(torch.equal(before[n], p) for n, p in trained.spr.named_parameters())


def test_the_priority_is_the_td_loss_and_the_batch_weights_scale_it():
    agent = an_agent(spr_weight=0.0)
    batch = a_batch()
    td, metrics = agent.update(batch, weights=np.ones(6, dtype=np.float32))
    assert td.shape == (6,) and (td > 0).all() and metrics['td_loss'] == pytest.approx(float(td.mean()), rel=1e-4)


def test_double_q_reads_the_online_argmax_and_off_it_reads_the_targets(monkeypatch):
    for double in (True, False):
        agent = an_agent(double=double, spr_weight=0.0)
        obs = torch.rand(5, constants.OBS_LEN)
        online_best = torch.tensor([0, 1, 2, 0, 1])
        target_best = torch.tensor([2, 2, 0, 1, 0])
        monkeypatch.setattr(agent.net, 'q_values', lambda x: torch.nn.functional.one_hot(online_best, 3).float())
        monkeypatch.setattr(agent.target, 'q_values', lambda x: torch.nn.functional.one_hot(target_best, 3).float())
        chosen = online_best if double else target_best
        # gamma 1, reward 0: the projection is the identity, so the target is the target net's own distribution
        # at the chosen action and the loss is the cross-entropy against exactly that.
        action = torch.tensor([1, 1, 1, 1, 1])
        loss = agent._categorical_loss(obs, action, torch.zeros(5), torch.ones(5), obs)
        with torch.no_grad():
            expected_target = agent.target.probs(obs)[torch.arange(5), chosen]
            log_probs = agent.net.log_probs(obs)[torch.arange(5), action]
            expected = -(expected_target * log_probs).sum(dim=1)
        assert torch.allclose(loss, expected, atol=1e-5)


def test_a_reset_shrinks_the_encoder_and_the_transition_model_replaces_the_rest_and_restarts_the_cycle():
    cycle = resets.CycleSchedule((10, 3), (0.97, 0.997), steps=4)
    agent = an_agent(reset_interval=3, reset_alpha=0.5, cycle=cycle, spr_weight=1.0)
    for _ in range(2):
        agent.update(a_batch())
    assert agent.cycle_values() == (5, pytest.approx(cycle.gamma_at(2)))
    net_before = {n: p.clone() for n, p in agent.net.named_parameters()}
    spr_before = {n: p.clone() for n, p in agent.spr.named_parameters()}
    assert agent.optimizer.state
    agent.update(a_batch())
    assert agent.resets == 1 and agent.last_reset_step == 3 and agent.steps_since_reset == 0
    assert agent.cycle_values() == (10, pytest.approx(0.97))
    assert not agent.optimizer.state
    fresh = network.build(agent.arch, 'cpu', seed=resets.reset_seed(1, 1))
    fresh_spr = agent._fresh_spr(resets.reset_seed(1, 1))
    for name, parameter in agent.net.named_parameters():
        expected = dict(fresh.named_parameters())[name]
        if 'hidden.' in name:
            # The optimiser step at update 3 moved the weights before the reset, so compare to the reset rule
            # applied to the post-step value: halfway between the pre-reset value and the fresh one.
            assert not torch.allclose(parameter, expected) and not torch.allclose(parameter, net_before[name])
        else:
            assert torch.allclose(parameter, expected)
    for name, parameter in agent.spr.named_parameters():
        expected = dict(fresh_spr.named_parameters())[name]
        if name.startswith('transition.'):
            assert not torch.allclose(parameter, expected) and not torch.allclose(parameter, spr_before[name])
        else:
            assert torch.allclose(parameter, expected)
    assert all(torch.equal(a, b) for a, b in zip(agent.target.parameters(), agent.net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.target_spr.parameters(), agent.spr.parameters()))


def test_the_shrink_is_exactly_halfway():
    agent = an_agent(reset_interval=1, reset_alpha=0.5, learning_rate=0.0, weight_decay=0.0)
    before = {n: p.clone() for n, p in agent.net.named_parameters()}
    agent.update(a_batch())                       # lr 0: the step moves nothing, then the reset fires
    fresh = network.build(agent.arch, 'cpu', seed=resets.reset_seed(1, 1))
    for name, parameter in agent.net.named_parameters():
        if 'hidden.' in name:
            assert torch.allclose(parameter, 0.5 * before[name] + 0.5 * dict(fresh.named_parameters())[name], atol=1e-6)


def test_the_agent_round_trips_its_state():
    cycle = resets.CycleSchedule((10, 3), (0.97, 0.997), steps=100)
    agent = an_agent(reset_interval=4, cycle=cycle)
    for _ in range(6):
        agent.update(a_batch())
    back = an_agent(reset_interval=4, cycle=cycle)
    back.load_state_dict(agent.state_dict())
    assert (back.train_step, back.resets, back.last_reset_step) == (6, 1, 4)
    assert back.cycle_values() == agent.cycle_values()
    for a, b in zip(agent.net.parameters(), back.net.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(agent.target_spr.parameters(), back.target_spr.parameters()):
        assert torch.equal(a, b)
    obs = np.random.default_rng(0).random((4, constants.OBS_LEN)).astype(np.float32)
    assert np.array_equal(agent.act(obs, 0.5), back.act(obs, 0.5))


def test_act_is_uniform_epsilon_greedy_and_ignores_the_shield():
    agent = an_agent()
    obs = np.random.default_rng(0).random((200, constants.OBS_LEN)).astype(np.float32)
    greedy = agent.greedy_actions(obs)
    assert np.array_equal(agent.act(obs, 0.0, guided=True), greedy)
    explored = agent.act(obs, 1.0, guided=np.ones(200, dtype=bool))
    assert 0.2 < (explored != greedy).mean() < 0.9 and set(np.unique(explored)) <= {0, 1, 2}


# ---------------------------------------------------------------- the seam

def test_the_defaults_are_the_papers():
    config = train.build_config() if False else None
    import os
    os.environ.pop('SNEK_ALGO', None)
    os.environ['SNEK_ALGO'] = 'bbf'
    try:
        config = train.build_config()
    finally:
        del os.environ['SNEK_ALGO']
    paper = {'learning_rate': 1e-4, 'adam_epsilon': 1.5e-4, 'batch_size': 32, 'discount': 0.997, 'n_step_update': 3,
             'target_update_tau': 0.005, 'gradient_clipping': 10.0, 'initial_epsilon': 1.0, 'min_epsilon': 0.0,
             'epsilon_anneal_steps': 2001, 'collect_envs': 1, 'replay_ratio': 8.0, 'replay_buffer_max_length': 1000000,
             'initial_collect_steps': 2000, 'priority_exponent': 0.5, 'reset_interval': 40000, 'reset_alpha': 0.5,
             'reset_stop_after': 0, 'reset_anneal_n_step': '10,3', 'reset_anneal_gamma': '0.97,0.997',
             'reset_anneal_steps': 10000, 'dist_atoms': 51, 'dist_v_min': -10.0, 'dist_v_max': 110.0,
             'bbf_weight_decay': 0.1, 'bbf_spr_weight': 5.0, 'bbf_spr_steps': 5, 'bbf_projection': 512, 'bbf_transition_width': 256,
             'bbf_dueling': True, 'bbf_double': True}
    assert {key: config[key] for key in paper} == paper
    assert bbf_algo.arch_fields(config) == {
        'head': {'type': 'c51', 'atoms': 51, 'v_min': -10.0, 'v_max': 110.0},
        'trunk': {'dueling': True, 'noisy': False, 'noisy_sigma': 0.5, 'residual': False, 'blocks': 1,
                  'spectral_norm': False, 'layer_norm': False}}


def test_the_algo_trains_through_the_seam_with_the_cycle_moving_everything(monkeypatch):
    algo, config, arch = build(monkeypatch, RESET_INTERVAL=200, RESET_ANNEAL_STEPS=100, REPLAY_RATIO=2)
    assert 'bbf, 1 lane(s), replay ratio 2.0' in algo.describe()
    assert 'n-step 10 -> 3 and gamma 0.97 -> 0.997 over 100 gradient steps' in algo.describe()
    assert algo.collector.n_step == 1 and algo.collector.discount == pytest.approx(0.97)
    assert algo.collector.vec.shaping_discount == pytest.approx(0.97)
    assert algo.prefill() >= 60 and algo.agent.train_step == 0
    asked = []
    original = algo.buffer.sample

    def recording(batch_size, n_step, gamma, spr_steps):
        asked.append((n_step, round(gamma, 5)))
        return original(batch_size, n_step, gamma, spr_steps)
    algo.buffer.sample = recording
    for _ in range(40):
        steps, transitions = algo.advance()
        assert (steps, transitions) == (1, 1)
    assert algo.agent.train_step == 80 and algo.moves == 40
    cycle = resets.CycleSchedule((10, 3), (0.97, 0.997), steps=100)
    assert asked[0] == (10, 0.97) and asked[-1] == (cycle.n_step_at(79), round(cycle.gamma_at(79), 5))
    assert len({n for n, _ in asked}) > 1
    n, gamma = algo.agent.cycle_values()
    assert n == resets.CycleSchedule((10, 3), (0.97, 0.997), steps=100).n_step_at(80) and n < 10
    # The collector's gamma is set before the step, so it is the cycle's value before that step's two updates.
    collected = resets.CycleSchedule((10, 3), (0.97, 0.997), steps=100).gamma_at(78)
    assert algo.collector.discount == pytest.approx(collected) and algo.collector.vec.shaping_discount == pytest.approx(collected)
    assert collected < gamma
    fields = algo.fields()
    assert fields['cycle'] == {'n_step': n, 'gamma': round(gamma, 5), 'since_reset': 80}
    assert fields['bbf']['train_step'] == 80 and 'spr_loss' in fields['bbf'] and 'td_loss' in fields['bbf']
    # Epsilon is set before each step from the moves so far: the 40th step ran at 39 moves.
    assert fields['epsilon'] == pytest.approx(1.0 - 39 / 2001.0, abs=1e-4)
    assert len(algo.log_extra(fields)) == 2 and algo.log_note(fields).startswith('eps ')
    assert algo.on_eval([], {}) == {'epsilon': round(1.0 - 40 / 2001.0, 5)}   # re-read at the moves so far
    for _ in range(70):
        algo.advance()
    assert algo.agent.resets == 1 and algo.agent.last_reset_step == 200


def test_the_anneal_off_is_a_constant_n_and_gamma(monkeypatch):
    algo, config, arch = build(monkeypatch, RESET_ANNEAL_N_STEP='', RESET_ANNEAL_GAMMA='', N_STEP_UPDATE=2, DISCOUNT=0.9)
    assert not algo.agent.cycle.enabled and algo.agent.cycle_values() == (2, 0.9)
    assert 'n-step 2 at gamma 0.9 throughout' in algo.describe() and algo.buffer.horizon == 5
    algo.prefill()
    batch, _, _ = algo.buffer.sample(4, 2, 0.9, 5)
    assert set(np.round(batch['discount'].astype(np.float64), 6).tolist()) <= {0.0, 0.81}


def test_the_algo_round_trips_through_resume_and_the_buffer_through_disk(monkeypatch, tmp_path):
    algo, config, arch = build(monkeypatch, REPLAY_RATIO=1)
    algo.prefill()
    for _ in range(30):
        algo.advance()
    algo.save_side_state(str(tmp_path))
    state = algo.state_dict()
    back, _, _ = build(monkeypatch, REPLAY_RATIO=1)
    back.load_state_dict(state)
    assert back.load_side_state(str(tmp_path))
    assert (back.moves, back.epsilon, back.agent.train_step) == (algo.moves, algo.epsilon, algo.agent.train_step)
    assert back.buffer.size == algo.buffer.size and np.array_equal(back.buffer.obs[:30], algo.buffer.obs[:30])
    assert not build(monkeypatch)[0].load_side_state(str(tmp_path / 'nothing'))


def test_the_checkpoint_is_the_rainbow_network_and_restores_through_the_restore_module(monkeypatch, tmp_path):
    algo, config, arch = build(monkeypatch)
    path = checkpoints.save(str(tmp_path), 7, algo.net)
    assert arch['algo'] == 'bbf' and arch['head']['type'] == 'c51' and arch['trunk']['dueling']
    net = restore.build_net(arch)
    checkpoints.load(path, net)
    obs = np.random.default_rng(0).random((5, constants.OBS_LEN)).astype(np.float32)
    assert np.array_equal(restore.policy_fn_for(arch, net)(obs), algo.policy_fn(obs))
    assert np.array_equal(algo.policy_fn(obs), algo.agent.greedy_actions(obs))
    # init_from: another arm's checkpoint becomes the net and the target.
    other, _, _ = build(monkeypatch)
    assert 'net and target from' in other.init_from(str(tmp_path), 7)
    assert all(torch.equal(a, b) for a, b in zip(other.net.parameters(), algo.net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(other.agent.target.parameters(), algo.net.parameters()))


# ---------------------------------------------------------------- the knobs

def test_bbf_refuses_the_knobs_it_cannot_honour(monkeypatch):
    for knob in ('RAINBOW_NOISY', 'TARGET_UPDATE_PERIOD', 'FORK_BRANCHES', 'MUNCHAUSEN_ALPHA', 'SAC_ALPHA', 'PPO_CLIP',
                 'GUIDED_FRACTION', 'DIST_QUANTILES'):
        monkeypatch.setenv('SNEK_ALGO', 'bbf')
        monkeypatch.setenv('SNEK_' + knob, '1')
        with pytest.raises(ValueError, match='SNEK_' + knob):
            train.build_config()
        monkeypatch.delenv('SNEK_' + knob)


def test_the_other_algorithms_refuse_bbfs_knobs(monkeypatch):
    for algo in ('ppo', 'sac2', 'rainbow'):
        monkeypatch.setenv('SNEK_ALGO', algo)
        monkeypatch.setenv('SNEK_BBF_SPR_WEIGHT', '5')
        with pytest.raises(ValueError, match='SNEK_BBF_SPR_WEIGHT'):
            train.build_config()
        monkeypatch.delenv('SNEK_BBF_SPR_WEIGHT')


def test_epsilon_zero_is_legal_here_and_bad_values_name_their_knob(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'bbf')
    assert train.build_config()['min_epsilon'] == 0.0
    for knob, value, match in (('MIN_EPSILON', '1.5', 'SNEK_MIN_EPSILON'), ('REPLAY_RATIO', '0', 'SNEK_REPLAY_RATIO'),
                               ('TARGET_UPDATE_TAU', '0', 'SNEK_TARGET_UPDATE_TAU'), ('RESET_ANNEAL_N_STEP', '10', 'RESET_ANNEAL_N_STEP'),
                               ('REPLAY_BUFFER_MAX_LENGTH', '10', 'SNEK_REPLAY_BUFFER_MAX_LENGTH'),
                               ('DIST_V_MAX', '-20', 'SNEK_DIST_V_MAX')):
        monkeypatch.setenv('SNEK_' + knob, value)
        with pytest.raises(ValueError, match=match):
            train.build_config()
        monkeypatch.delenv('SNEK_' + knob)


def test_the_horizon_is_the_longest_lookahead_a_batch_can_need(monkeypatch):
    config = small_config(monkeypatch, BBF_SPR_STEPS=2)
    assert bbf_algo.horizon_of(config, resets.cycle_from_config(config)) == 10
    config = small_config(monkeypatch, BBF_SPR_STEPS=12)
    assert bbf_algo.horizon_of(config, resets.cycle_from_config(config)) == 12
    config = small_config(monkeypatch, RESET_ANNEAL_N_STEP='', RESET_ANNEAL_GAMMA='', N_STEP_UPDATE=4, BBF_SPR_STEPS=0)
    assert bbf_algo.horizon_of(config, resets.cycle_from_config(config)) == 4
