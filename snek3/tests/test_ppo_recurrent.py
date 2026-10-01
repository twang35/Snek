"""Recurrent PPO (group E, row E1): the tower, the rollout's sequences, the collector's carried state and
the update's replay, and the one property that protects every batch before it -- **the knob off is the
feed-forward arm bit for bit.**

Silent failures these pin:

- **the replayed cell disagrees with the collecting cell**: the first minibatch's ratio is then not 1 and
  every update is off-policy from the start;
- **`done` inside a chunk does not zero the state**: the next episode opens with the last one's memory;
- **the bootstrap value read advances the carried state**, so the next rollout's first step runs the cell
  twice on one observation;
- **the critic shares the actor's cell**, which the incumbent's two towers do not;
- **the minibatch shuffles transitions**, which breaks the sequence a cell is replayed over.
"""

import copy
import os
import json

import numpy as np
import pytest
import torch

import train
from algos.ppo import algo as ppo_algo
from algos.ppo import net as network
from algos.ppo import rollout as rollout_module
from algos.ppo.agent import PpoAgent
from algos.stateful import StatefulPolicy
from env import constants
from tools import arch as arch_tools
from tools import checkpoints
from tools import restore
from vectorized import engine


def _config(monkeypatch, recurrent='gru', hidden=8, seq=2, **overrides):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.delenv('SNEK_PPO_MINIBATCH', raising=False)
    if recurrent:
        monkeypatch.setenv('SNEK_PPO_RECURRENT', recurrent)
        monkeypatch.setenv('SNEK_PPO_RECURRENT_HIDDEN', str(hidden))
        monkeypatch.setenv('SNEK_PPO_SEQ_MINIBATCH', str(seq))
    else:
        monkeypatch.delenv('SNEK_PPO_RECURRENT', raising=False)
        monkeypatch.setenv('SNEK_PPO_MINIBATCH', '8')
    config = train.build_config()
    config.update({'seed': 5, 'fc_layers': (16,), 'max_steps': 10000})
    config.update(overrides)
    return config


def _arch(config):
    extra = ppo_algo.arch_fields(config)
    return arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                 constants.OBS_ERA, algo='ppo', **extra)


def _built(monkeypatch, **kw):
    config = _config(monkeypatch, **kw)
    return ppo_algo.build(config, _arch(config)), config


# --- the tower --------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind,width', [('gru', 8), ('lstm', 16)])
def test_the_state_is_one_flat_tensor_per_tower(kind, width):
    tower = network.RecurrentTower(26, (16,), 3, kind, 8, seed=1)
    assert tower.state_width == width
    out, state = tower.step(torch.zeros(5, 26), tower.initial_state(5))
    assert out.shape == (5, 3) and state.shape == (5, width)


def test_a_fresh_row_runs_from_the_zero_state_whatever_it_carried():
    tower = network.RecurrentTower(26, (16,), 3, 'lstm', 8, seed=1)
    obs = torch.randn(2, 26)
    carried = torch.randn(2, 16)
    _, from_zero = tower.step(obs, tower.initial_state(2))
    _, from_carried = tower.step(obs, carried, fresh=torch.tensor([True, False]))
    assert torch.allclose(from_carried[0], from_zero[0])
    assert not torch.allclose(from_carried[1], from_zero[1])


def test_unroll_is_step_applied_in_order_with_the_fresh_mask():
    tower = network.RecurrentTower(26, (16,), 3, 'gru', 8, seed=2)
    obs = torch.randn(5, 3, 26)
    fresh = torch.zeros(5, 3, dtype=torch.bool)
    fresh[2, 1] = True
    state = torch.randn(3, 8)
    outputs, final = tower.unroll(obs, state, fresh)
    expected = []
    s = state
    for t in range(5):
        out, s = tower.step(obs[t], s, fresh[t])
        expected.append(out)
    assert torch.allclose(outputs, torch.stack(expected)) and torch.allclose(final, s)


def test_two_towers_at_one_seed_are_the_same_network_and_the_critic_is_a_different_one():
    arch = {'obs_len': 26, 'fc_layer_params': [16], 'num_actions': 3, 'recurrent': {'type': 'gru', 'hidden': 8}}
    a, b = network.build(arch, seed=3), network.build(arch, seed=3)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)
    critic = network.build_critic(arch, seed=3)
    assert critic.head.out_features == 1
    assert not torch.equal(critic.hidden[0].weight, a.hidden[0].weight), 'the critic takes a derived seed'
    assert critic.cell is not a.cell


def test_an_unknown_cell_or_a_zero_width_is_refused_by_name():
    with pytest.raises(ValueError, match='recurrent.type'):
        network.build({'obs_len': 26, 'fc_layer_params': [16], 'num_actions': 3,
                       'recurrent': {'type': 'rnn', 'hidden': 8}})
    with pytest.raises(ValueError, match='recurrent.hidden'):
        network.build({'obs_len': 26, 'fc_layer_params': [16], 'num_actions': 3,
                       'recurrent': {'type': 'gru', 'hidden': 0}})


def test_the_greedy_policy_of_a_recurrent_tower_is_stateful_and_a_feed_forward_ones_is_not():
    arch = {'obs_len': 26, 'fc_layer_params': [16], 'num_actions': 3}
    assert not hasattr(network.greedy_policy_fn(network.build(arch, seed=1)), 'begin')
    recurrent = network.greedy_policy_fn(network.build(dict(arch, recurrent={'type': 'gru', 'hidden': 8}), seed=1))
    assert isinstance(recurrent, StatefulPolicy)
    held = engine.measure(recurrent, 6, lanes=3, seed=1)
    assert len(held['scores']) == 6


# --- the knob off is the feed-forward arm, bit for bit ---------------------------------------------

def test_the_knob_off_collects_and_updates_byte_identically_to_the_code_before_it(monkeypatch):
    """Not against a saved fingerprint, which would pin this build's numbers: against the feed-forward path
    run through the same objects with the recurrent branch unreachable. Both arms share one seed, one env
    stream and one rng, so any divergence is a change in the feed-forward code."""
    algo, _ = _built(monkeypatch, recurrent='')
    assert not algo.agent.recurrent and not algo.rollout.recurrent and not algo.collector.recurrent
    before = copy.deepcopy(algo.agent.state_dict())
    algo.advance()
    first = {k: v.clone() for k, v in algo.agent.actor.state_dict().items()}
    # A second, independent build at the same seed reaches the same weights: the path is deterministic
    # and the recurrent fields do not touch it.
    other, _ = _built(monkeypatch, recurrent='')
    other.advance()
    for key, value in other.agent.actor.state_dict().items():
        assert torch.equal(value, first[key]), key
    assert any(not torch.equal(before['actor'][k], first[k]) for k in first), 'the update moved nothing'


def test_the_feed_forward_sidecar_has_no_recurrent_field(monkeypatch):
    config = _config(monkeypatch, recurrent='')
    assert ppo_algo.arch_fields(config) == {}
    assert 'recurrent' not in _arch(config)


# --- the config -------------------------------------------------------------------------------------

def test_the_recurrent_knobs_reach_the_config_and_the_sidecar(monkeypatch):
    config = _config(monkeypatch, recurrent='lstm', hidden=12, seq=2)
    assert config['ppo_recurrent'] == 'lstm' and config['ppo_recurrent_hidden'] == 12
    assert config['ppo_seq_minibatch'] == 2
    assert config['ppo_minibatch'] == 2 * 8, 'derived: lanes a minibatch x rollout'
    assert _arch(config)['recurrent'] == {'type': 'lstm', 'hidden': 12}


def test_a_minibatch_set_against_the_derived_one_is_refused(monkeypatch):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.setenv('SNEK_PPO_RECURRENT', 'gru')
    monkeypatch.setenv('SNEK_PPO_SEQ_MINIBATCH', '2')
    monkeypatch.setenv('SNEK_PPO_MINIBATCH', '512')
    with pytest.raises(ValueError, match='SNEK_PPO_MINIBATCH=512 conflicts'):
        train.build_config()
    # Set to the derived value it is accepted: a spec may spell it out.
    monkeypatch.setenv('SNEK_PPO_MINIBATCH', '16')
    assert train.build_config()['ppo_minibatch'] == 16


@pytest.mark.parametrize('env,match', [
    ({'SNEK_PPO_RECURRENT': 'rnn'}, 'SNEK_PPO_RECURRENT'),
    ({'SNEK_PPO_RECURRENT': 'gru', 'SNEK_PPO_RECURRENT_HIDDEN': '0'}, 'SNEK_PPO_RECURRENT_HIDDEN'),
    ({'SNEK_PPO_RECURRENT': 'gru', 'SNEK_PPO_SEQ_MINIBATCH': '9'}, 'SNEK_PPO_SEQ_MINIBATCH'),
])
def test_bad_recurrent_knobs_are_refused_by_name(monkeypatch, env, match):
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.delenv('SNEK_PPO_MINIBATCH', raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=match):
        train.build_config()


def test_the_other_algorithms_refuse_the_recurrent_knobs(monkeypatch):
    monkeypatch.setenv('SNEK_PPO_RECURRENT', 'gru')
    for name in ('sac', 'rainbow', 'bbf'):
        monkeypatch.setenv('SNEK_ALGO', name)
        with pytest.raises(ValueError, match='SNEK_PPO_RECURRENT'):
            train.build_config()


# --- the rollout --------------------------------------------------------------------------------------

def test_sequence_minibatches_cover_every_lane_once_in_whole_lanes():
    roll = rollout_module.Rollout(4, 6, 3, actor_state=2, critic_state=2)
    for t in range(4):
        roll.add(t, np.full((6, 3), t, dtype=np.float32), np.arange(6), np.zeros(6), np.zeros(6),
                 np.ones(6), np.zeros(6, bool), actor_state=np.full((6, 2), t), critic_state=np.zeros((6, 2)),
                 fresh=np.zeros(6, bool))
    roll.finish(np.zeros(6), 0.9, 0.9)
    seen = []
    for batch in roll.sequence_minibatches(4, np.random.default_rng(0)):
        assert batch['obs'].shape[0] == 4 and batch['obs'].shape[2] == 3
        # Whole lanes, in step order: the lane's action is constant down its column.
        assert (batch['actions'] == batch['actions'][0]).all()
        assert (batch['obs'][:, :, 0] == np.arange(4)[:, None]).all()
        # The state handed over is the one carried into step 0, not a later one.
        assert (batch['actor_state'] == 0).all()
        seen.extend(batch['actions'][0].tolist())
    assert sorted(seen) == list(range(6))


def test_a_recurrent_rollout_refuses_an_add_without_its_states():
    roll = rollout_module.Rollout(2, 2, 3, actor_state=2, critic_state=2)
    with pytest.raises(ValueError, match='fresh mask'):
        roll.add(0, np.zeros((2, 3)), [0, 0], [0, 0], [0, 0], [0, 0], [False, False])


def test_a_feed_forward_rollout_has_no_sequence_minibatches():
    roll = rollout_module.Rollout(2, 2, 3)
    roll.add(0, np.zeros((2, 3)), [0, 0], [0, 0], [0, 0], [0, 0], [False, False])
    roll.add(1, np.zeros((2, 3)), [0, 0], [0, 0], [0, 0], [0, 0], [False, False])
    roll.finish([0, 0], 0.9, 0.9)
    with pytest.raises(ValueError, match='recurrent rollout'):
        next(roll.sequence_minibatches(1, np.random.default_rng(0)))


# --- the collector and the agent ------------------------------------------------------------------------

def test_the_replayed_cell_reproduces_the_collected_log_probs_so_the_first_ratio_is_one(monkeypatch):
    """**The property the whole design rests on.** The update replays each tower from the stored step-0
    state with the stored fresh mask; if that disagrees with the state the collector ran, the stored
    log-prob and the replayed one differ and the first minibatch's ratio is not 1."""
    algo, _ = _built(monkeypatch, recurrent='lstm', hidden=8, seq=2)
    algo.collector.collect()
    algo.collector.collect()                     # a second rollout: the carried state is now non-zero
    agent = algo.agent
    for batch in algo.rollout.sequence_minibatches(2, np.random.default_rng(1)):
        logits, _ = agent._forward(batch)
        actions = torch.as_tensor(batch['actions'].reshape(-1)).long()
        log_probs, _ = network.evaluate(logits, actions)
        assert log_probs.detach().numpy() == pytest.approx(batch['log_probs'].reshape(-1), abs=1e-5)


def test_a_done_inside_the_rollout_is_stored_as_fresh_on_the_next_step(monkeypatch):
    algo, _ = _built(monkeypatch, recurrent='gru', hidden=8)
    # Drive many rollouts so some lane dies mid-rollout.
    for _ in range(6):
        algo.collector.collect()
        dones = algo.rollout.dones
        fresh = algo.rollout.fresh
        if dones[:-1].any():
            where = np.argwhere(dones[:-1])
            for t, lane in where:
                assert fresh[t + 1, lane], 'the step after a death must be marked fresh'
            return
    pytest.skip('no lane died inside a rollout in six tries')


def test_the_bootstrap_value_read_does_not_advance_the_carried_state(monkeypatch):
    algo, _ = _built(monkeypatch, recurrent='gru', hidden=8)
    agent, coll = algo.agent, algo.collector
    coll.collect()
    before = coll.critic_state.copy()
    agent.values(coll.obs, coll.critic_state, coll.fresh)
    agent.values(coll.obs, coll.critic_state, coll.fresh)
    assert np.array_equal(coll.critic_state, before)
    # And the read is a function of the carried state: a different state gives a different value.
    a = agent.values(coll.obs, coll.critic_state, coll.fresh)
    b = agent.values(coll.obs, coll.critic_state + 1.0, coll.fresh)
    assert not np.allclose(a, b)


def test_the_carried_state_crosses_the_rollout_boundary(monkeypatch):
    algo, _ = _built(monkeypatch, recurrent='gru', hidden=8)
    coll = algo.collector
    coll.collect()
    carried = coll.actor_state.copy()
    assert np.abs(carried).sum() > 0
    coll.collect()
    # The second rollout's step-0 stored state is exactly the state the first left behind.
    assert np.array_equal(algo.rollout.actor_states[0], carried)


def test_an_update_moves_both_cells_and_reports_the_diagnostics(monkeypatch):
    algo, _ = _built(monkeypatch, recurrent='lstm', hidden=8, seq=2)
    before_actor = {k: v.clone() for k, v in algo.agent.actor.cell.state_dict().items()}
    before_critic = {k: v.clone() for k, v in algo.agent.critic.cell.state_dict().items()}
    algo.advance()
    metrics = algo.last_metrics
    assert any(not torch.equal(v, before_actor[k]) for k, v in algo.agent.actor.cell.state_dict().items())
    assert any(not torch.equal(v, before_critic[k]) for k, v in algo.agent.critic.cell.state_dict().items())
    for key in ('entropy', 'approx_kl', 'clip_fraction', 'explained_variance', 'policy_loss', 'value_loss'):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics['train_step'] == 4 * 2, '4 epochs x (4 lanes / 2 a minibatch)'


def test_describe_names_the_cell(monkeypatch):
    algo, _ = _built(monkeypatch, recurrent='gru', hidden=8, seq=2)
    assert 'GRU 8' in algo.describe() and 'whole lane' in algo.describe()


# --- the checkpoint and the restore path ----------------------------------------------------------------

def test_a_recurrent_checkpoint_restores_to_a_stateful_policy_and_a_feed_forward_sidecar_refuses_it(
        monkeypatch, tmp_path):
    algo, config = _built(monkeypatch, recurrent='gru', hidden=8)
    arch = _arch(config)
    policy_dir = str(tmp_path / 'arm')
    arch_tools.write_arch(policy_dir, arch)
    checkpoints.save(policy_dir, 1000, algo.net)
    policy_fn, read_arch, step = restore.restore(policy_dir, 1000)
    assert isinstance(policy_fn, StatefulPolicy) and step == 1000
    assert read_arch['recurrent'] == {'type': 'gru', 'hidden': 8}
    held = engine.measure(policy_fn, 4, lanes=2, seed=1)
    assert len(held['scores']) == 4
    # The same weights into a feed-forward net: refused by the signature, not half-loaded.
    plain = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN,
                                  constants.OBS_ERA, algo='ppo')
    with pytest.raises(arch_tools.ArchMismatch, match='recurrent'):
        arch_tools.assert_same_network(plain, read_arch)


def test_a_resume_restores_both_cells(monkeypatch):
    algo, config = _built(monkeypatch, recurrent='lstm', hidden=8)
    algo.advance()
    state = algo.state_dict()
    other, _ = _built(monkeypatch, recurrent='lstm', hidden=8)
    other.load_state_dict(state)
    for k, v in algo.agent.actor.state_dict().items():
        assert torch.equal(other.agent.actor.state_dict()[k], v)
    for k, v in algo.agent.critic.state_dict().items():
        assert torch.equal(other.agent.critic.state_dict()[k], v)


def test_the_trainer_runs_a_recurrent_arm_end_to_end(tmp_path, monkeypatch):
    """A whole smoke through `train.py`: the sidecar carries the cell, checkpoints land, stage A measures
    them through the stateful policy, and a resume continues."""
    monkeypatch.setattr(constants, 'POLICY_DIR', str(tmp_path / 'policies'))
    monkeypatch.setenv('SNEK_ALGO', 'ppo')
    monkeypatch.setenv('SNEK_COLLECT_ENVS', '4')
    monkeypatch.setenv('SNEK_PPO_ROLLOUT', '8')
    monkeypatch.delenv('SNEK_PPO_MINIBATCH', raising=False)
    monkeypatch.setenv('SNEK_PPO_RECURRENT', 'gru')
    monkeypatch.setenv('SNEK_PPO_RECURRENT_HIDDEN', '8')
    monkeypatch.setenv('SNEK_PPO_SEQ_MINIBATCH', '2')
    monkeypatch.setattr(train, 'EVAL_INTERVAL', 32)
    monkeypatch.setattr(train, 'EVAL_EPISODES', 4)
    monkeypatch.setattr(train, 'RESUME_INTERVAL', 64)
    monkeypatch.setattr(train, 'REPORT_INTERVAL', 64)
    config = train.build_config()
    config.update({'max_steps': 96, 'fc_layers': (16,), 'eval_queue': False, 'min_checkpoint_score': 0,
                   'eval_workers': 0})
    trainer = train.Trainer('rec-smoke', config)
    trainer.run()
    sidecar = json.load(open(arch_tools.arch_path(trainer.policy_dir)))
    assert sidecar['recurrent'] == {'type': 'gru', 'hidden': 8}
    assert checkpoints.steps(trainer.policy_dir) == [32, 64, 96]
    assert [row['step'] for row in trainer.eval_rows] == [32, 64, 96]
    config['max_steps'] = 128
    resumed = train.Trainer('rec-smoke', config)
    assert resumed.step == 96
    resumed.run()
    assert resumed.step == 128


@pytest.mark.parametrize('kind', ['lstm', 'gru'])
def test_the_fused_unroll_is_step_applied_t_times(kind):
    """`unroll` cuts the sequence at every fresh step and runs each piece through the fused kernel; the
    outputs and the final state must equal the step loop's, resets and all."""
    torch.manual_seed(0)
    tower = network.RecurrentTower(constants.OBS_LEN, (16,), 3, kind, 8, seed=5)
    T, n = 23, 4
    obs = torch.randn(T, n, constants.OBS_LEN)
    fresh = torch.rand(T, n) < 0.15
    fresh[0, 1] = True
    state = torch.randn(n, tower.state_width)
    fused_out, fused_state = tower.unroll(obs, state.clone(), fresh)
    outs, s = [], state.clone()
    for t in range(T):
        out, s = tower.step(obs[t], s, fresh[t])
        outs.append(out)
    assert torch.allclose(fused_out, torch.stack(outs), atol=1e-5)
    assert torch.allclose(fused_state, s, atol=1e-5)


def test_the_bootstrap_value_read_honours_the_fresh_flags():
    """A lane marked fresh at the bootstrap read is valued from the zero state, whatever state is handed over."""
    monkey = {'SNEK_ALGO': 'ppo', 'SNEK_PPO_RECURRENT': 'lstm', 'SNEK_PPO_RECURRENT_HIDDEN': '8', 'SNEK_COLLECT_ENVS': '4',
              'SNEK_PPO_ROLLOUT': '8', 'SNEK_FC_LAYERS': '16', 'SNEK_CHART_WINDOW': '0'}
    old = {k: os.environ.get(k) for k in monkey}
    os.environ.update(monkey)
    try:
        config = train.build_config()
        extra = ppo_algo.arch_fields(config)
        arch = arch_tools.build_arch(config['fc_layers'], constants.NUM_ACTIONS, constants.OBS_LEN, constants.OBS_ERA,
                                     algo='ppo', **extra)
        agent = ppo_algo.build(config, arch).agent
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    obs = np.random.default_rng(0).standard_normal((4, constants.OBS_LEN)).astype(np.float32)
    state = np.full((4, agent.critic_state_width), 0.7, dtype=np.float32)
    fresh = np.array([True, False, True, False])
    values = agent.values(obs, state, fresh)
    from_zero = agent.values(obs, np.zeros_like(state), np.zeros(4, dtype=bool))
    carried = agent.values(obs, state, np.zeros(4, dtype=bool))
    assert values[0] == pytest.approx(from_zero[0]) and values[2] == pytest.approx(from_zero[2])
    assert values[1] == pytest.approx(carried[1]) and values[3] == pytest.approx(carried[3])
    assert values[0] != pytest.approx(carried[0])
