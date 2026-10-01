"""R2D2 (`algos/r2d2/`): the sequence replay's grid, the collector's loss-block rule, the learner's burn-in
and targets, the ladder, and the seam (`plans/algoExploration/e-memory.md` §0 and §2).

What would be silent without these:

- **a window placed off the grid** -- the loss taken at the wrong slots -- still produces a finite loss;
- **a block whose burn-in comes from a different episode**, or whose stored state is the state *after*
  its first row rather than before it, trains on a mismatched memory and only reads as a weaker curve;
- **a feed-forward control that takes its loss at different positions** than the LSTM arm is not a control;
- **a target that bootstraps past a terminal**, or past the rows the window holds, is a biased target that
  still goes down;
- **a ladder that puts every lane at the base** is epsilon-greedy with a longer name.
"""

import copy

import numpy as np
import pytest
import torch

from algos.dqn import schedules
from algos.r2d2 import agent as r2d2_agent
from algos.r2d2 import algo as r2d2_algo
from algos.r2d2 import net as network
from algos.r2d2.collect import Collector
from algos.r2d2.replay import SequenceReplay
from algos.stateful import StatefulPolicy
from env import constants
from tools import arch as arch_tools
from tools import restore

OBS = constants.OBS_LEN
ACTIONS = constants.NUM_ACTIONS


# --- fakes -----------------------------------------------------------------------------------------

class ScriptedVec(object):
    """N lanes whose episode lengths are scripted per lane; the observation's first value is the step within
    the episode, the reward is that step, `done` on the last step. Auto-resets like `VecSnake`."""

    def __init__(self, lengths):
        self.lengths = [list(l) for l in lengths]
        self.n = len(lengths)
        self.position = np.zeros(self.n, dtype=np.int64)
        self.episode = np.zeros(self.n, dtype=np.int64)

    def _obs(self):
        obs = np.zeros((self.n, OBS), dtype=np.float32)
        obs[:, 0] = self.position
        obs[:, 1] = self.episode
        return obs

    def reset_all(self):
        return self._obs()

    def step(self, actions):
        reward = self.position.astype(np.float32) + 1.0
        done = np.zeros(self.n, dtype=bool)
        for lane in range(self.n):
            length = self.lengths[lane][self.episode[lane] % len(self.lengths[lane])]
            self.position[lane] += 1
            if self.position[lane] >= length:
                done[lane] = True
                self.position[lane] = 0
                self.episode[lane] += 1
        return self._obs(), reward, done, {'perfect': np.zeros(self.n, dtype=bool)}


class CountingAgent(object):
    """A stateless stand-in for the learner's `act`: the state counts the steps since the episode began
    (zero on a fresh lane, then +1 per step), the action is the lane index."""

    def __init__(self, width=2):
        self.width = width

    def act(self, obs, prev_action, prev_reward, state, fresh, epsilons):
        state = np.asarray(state, dtype=np.float32) * (~np.asarray(fresh, dtype=bool))[:, None]
        return np.arange(obs.shape[0], dtype=np.int64) % ACTIONS, state + 1.0


def replay(lanes, windows=64, seq=80, burn=40, n=5, width=2, **kw):
    return SequenceReplay(windows, OBS, lanes, width, seq_length=seq, burn_in=burn, lookahead=n, seed=0, **kw)


def collect(lengths, steps, **kw):
    vec = ScriptedVec(lengths)
    buffer = replay(vec.n, **kw)
    collector = Collector(vec, CountingAgent(), buffer, np.full(vec.n, 0.1), seed=0)
    for _ in range(steps):
        collector.step()
    return collector, buffer


def small_arch(kind='lstm', head='scalar', hidden=8, prev_input=True):
    extra = {'recurrent': {'type': kind, 'hidden': hidden, 'stream_width': 8, 'prev_input': prev_input},
             'head': {'type': head, 'atoms': 11, 'v_min': -10.0, 'v_max': 110.0} if head == 'c51' else {'type': 'scalar'}}
    return arch_tools.build_arch((16,), ACTIONS, OBS, constants.OBS_ERA, algo='r2d2', **extra)


# --- the rescaling and the ladder -----------------------------------------------------------------------

def test_rescaling_inverts_itself_over_the_return_range():
    x = torch.linspace(-20.0, 120.0, 2001, dtype=torch.float64)
    back = r2d2_agent.unscale(r2d2_agent.rescale(x))
    assert torch.allclose(back, x, atol=1e-6)
    # And it compresses: h(100) is well under 100, and h is odd.
    assert r2d2_agent.rescale(torch.tensor(100.0)).item() < 12.0
    assert r2d2_agent.rescale(torch.tensor(-3.0)).item() == pytest.approx(-r2d2_agent.rescale(torch.tensor(3.0)).item())


def test_the_apex_ladder_spans_base_to_base_to_the_one_plus_alpha_and_falls_monotonically():
    ladder = schedules.apex_epsilons(32, 0.4, 7.0)
    assert ladder.shape == (32,)
    assert ladder[0] == pytest.approx(0.4)
    assert ladder[-1] == pytest.approx(0.4 ** 8)
    assert np.all(np.diff(ladder) < 0)
    # One lane: the base alone. Alpha 0: every lane at the base.
    assert schedules.apex_epsilons(1, 0.4, 7.0).tolist() == [0.4]
    assert np.allclose(schedules.apex_epsilons(4, 0.4, 0.0), 0.4)


def test_the_ladder_refuses_a_base_outside_zero_one_and_a_negative_alpha():
    with pytest.raises(ValueError):
        schedules.apex_epsilons(4, 0.0, 7.0)
    with pytest.raises(ValueError):
        schedules.apex_epsilons(4, 0.4, -1.0)


# --- the collector's loss-block rule --------------------------------------------------------------------

def windows_of(buffer):
    return [(int(buffer.burn[i]), int(buffer.length[i]), int(buffer.ahead[i])) for i in range(buffer.n_windows)]


def test_a_twenty_step_episode_is_one_window_with_no_burn_in_and_no_lookahead():
    _, buffer = collect([[20]], steps=20)
    assert windows_of(buffer) == [(0, 20, 0)]
    assert buffer.state[0].tolist() == [0.0, 0.0]


def test_a_hundred_step_episode_cuts_into_the_papers_geometry_mid_episode():
    """Blocks at 0, 40 and 80: the first has no burn-in, the second a full 40 (the paper's 80-step window),
    the third 40 burn-in and the 20 steps the episode had left, with no lookahead past its end."""
    _, buffer = collect([[100]], steps=100)
    assert windows_of(buffer) == [(0, 40, 5), (40, 40, 5), (40, 20, 0)]


def test_the_stored_state_is_the_one_carried_into_the_windows_first_row():
    """`CountingAgent`'s state is the step within the episode, so the state stored for a window that starts
    at episode step p must read p -- the state *before* that row was stepped, which is what a replay from
    it reproduces."""
    _, buffer = collect([[100]], steps=100)
    # Window 2 starts at step 0 (burn 40 rows before loss block at 40 -> start 0); window 3 at step 40.
    assert buffer.state[0, 0] == 0.0
    assert buffer.state[1, 0] == 0.0
    assert buffer.state[2, 0] == 40.0


def test_a_window_waits_for_its_lookahead_rows_and_is_banked_the_moment_the_episode_ends():
    collector, buffer = collect([[100]], steps=44)
    assert buffer.n_windows == 0                                      # block full at 40, lookahead needs 45
    collector.step()
    assert windows_of(buffer) == [(0, 40, 5)]
    collector, buffer = collect([[43]], steps=43)
    assert windows_of(buffer) == [(0, 40, 3), (40, 3, 0)]            # death at 43: 3 lookahead rows, then the stub


def test_the_previous_action_is_none_and_the_previous_reward_zero_at_an_episode_start():
    collector, buffer = collect([[3, 3]], steps=6)
    # Rows 0 and 3 open episodes.
    assert buffer.prev_action[0] == -1 and buffer.prev_action[3] == -1
    assert buffer.prev_reward[0] == 0.0 and buffer.prev_reward[3] == 0.0
    assert buffer.prev_action[1] == 0 and buffer.prev_reward[1] == 1.0   # lane 0's action is 0, step reward 1


def test_lanes_write_rows_in_lane_order_so_a_window_strides_by_the_lane_count():
    _, buffer = collect([[50], [50]], steps=50)
    rows = buffer.window_rows(0)
    assert rows.tolist() == list(range(0, 90, 2))                      # lane 0: every other row, 40 + 40 + 5
    assert buffer.window_rows(1).tolist() == list(range(1, 90, 2))
    assert np.all(buffer.obs[rows, 0] == np.arange(45))                # the episode step runs 0..44


def test_a_window_never_spans_two_episodes():
    _, buffer = collect([[30, 30, 30]], steps=90)
    for slot in range(buffer.n_windows):
        rows = buffer.window_rows(slot)
        assert len(set(buffer.obs[rows, 1].tolist())) == 1               # one episode id per window


# --- the replay's grid --------------------------------------------------------------------------------

def test_a_sampled_window_sits_on_the_grid_with_its_loss_block_at_the_burn_in_slot():
    _, buffer = collect([[100]], steps=100)
    batch, slots, weights = buffer.sample(3)
    valid, fresh = batch['valid'], batch['fresh']
    for b, slot in enumerate(slots):
        burn, length, ahead = int(buffer.burn[slot]), int(buffer.length[slot]), int(buffer.ahead[slot])
        first = 40 - burn
        assert valid[b, first:first + burn + length + ahead].all()
        assert not valid[b, :first].any() and not valid[b, first + burn + length + ahead:].any()
        # The loss block always begins at slot 40, and the stored state applies to slot `first`.
        assert valid[b, 40]
        assert fresh[b].sum() == (1 if burn < 40 else 0)
        if burn < 40:
            assert fresh[b, first]


def test_the_grid_is_burn_in_plus_block_plus_lookahead_wide():
    buffer = replay(1)
    assert buffer.slots == 85 and buffer.block == 40


def test_importance_weights_are_max_normalised_and_follow_the_priorities():
    _, buffer = collect([[100], [100]], steps=400)
    buffer.update_priorities(np.arange(buffer.n_windows), np.linspace(1.0, 10.0, buffer.n_windows))
    _, slots, weights = buffer.sample(buffer.n_windows)
    assert weights.max() == pytest.approx(1.0)
    assert np.all(weights <= 1.0 + 1e-6)


def test_the_sequence_priority_is_the_eta_mix_of_max_and_mean():
    td = torch.tensor([[1.0], [3.0], [2.0], [100.0]])
    mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    assert r2d2_agent.mix_priority(td, mask, 1.0)[0].item() == pytest.approx(3.0)
    assert r2d2_agent.mix_priority(td, mask, 0.0)[0].item() == pytest.approx(2.0)
    assert r2d2_agent.mix_priority(td, mask, 0.9)[0].item() == pytest.approx(0.9 * 3.0 + 0.1 * 2.0)


def test_an_updates_priorities_are_the_mix_over_its_own_td_errors():
    agent, buffer, batch, slots, weights = real_batch()
    agent.priority_eta = 1.0
    priorities, _ = agent.update(batch, weights)
    valid = torch.as_tensor(batch['valid'][:, 40:80]).transpose(0, 1).float()
    assert np.allclose(priorities, (agent.last_abs_td * valid).max(dim=0).values.numpy(), atol=1e-5)


def test_a_new_window_enters_at_the_running_maximum_priority():
    _, buffer = collect([[100]], steps=100)
    buffer.update_priorities(np.array([0]), np.array([7.0]))
    buffer.add_window(0, 0, 10, 0, np.zeros(2, dtype=np.float32))
    leaves = buffer.tree.nodes[buffer.tree.size:buffer.tree.size + buffer.n_windows]
    assert leaves[-1] == pytest.approx(leaves.max())


def test_the_ring_never_overwrites_a_live_windows_rows():
    """Tiny capacity, long episodes, many windows: every sample stays readable."""
    _, buffer = collect([[300], [300], [300]], steps=3000, windows=16)
    for _ in range(50):
        batch, _, _ = buffer.sample(8)
        assert batch['valid'][:, 40].all()


def test_the_buffer_survives_a_save_and_a_load(tmp_path):
    _, buffer = collect([[100], [37]], steps=300)
    buffer.update_priorities(np.arange(buffer.n_windows), np.linspace(1.0, 5.0, buffer.n_windows))
    buffer.save(str(tmp_path))
    again = replay(2)
    assert again.load(str(tmp_path))
    assert again.n_windows == buffer.n_windows and again.rows == buffer.rows
    assert again.tree.total == pytest.approx(buffer.tree.total)
    assert np.array_equal(again.state[:again.n_windows], buffer.state[:buffer.n_windows])
    assert np.array_equal(again.obs[:again.rows], buffer.obs[:buffer.rows])


def test_a_load_into_a_differently_shaped_buffer_is_refused(tmp_path):
    _, buffer = collect([[100], [37]], steps=100)
    buffer.save(str(tmp_path))
    with pytest.raises(ValueError, match='lane'):
        replay(3).load(str(tmp_path))


# --- the learner --------------------------------------------------------------------------------------

def test_n_step_targets_stop_at_a_terminal_and_bootstrap_only_past_a_full_lookahead():
    """Three windows of 4 reward rows then 2 lookahead rows (n = 2):
    (a) no terminal: every loss slot bootstraps from slot t+2;
    (b) terminal at slot 1: slot 0 sums r0 + g r1 and stops, slot 1 is r1 alone, slots 2 and 3 hold no rows;
    (c) the window is short: slot 3's lookahead is missing, so it does not bootstrap."""
    g = 0.5
    T = 6
    reward = torch.ones(T, 3)
    done = torch.zeros(T, 3, dtype=torch.bool)
    valid = torch.ones(T, 3, dtype=torch.bool)
    boot = torch.full((T, 3), 10.0)
    done[1, 1] = True
    valid[2:, 1] = False
    valid[5, 2] = False
    returns, mask = r2d2_agent.n_step_targets(reward, done, valid, boot, g, 2, 0, 4)
    # (a)
    assert torch.allclose(returns[:, 0], torch.full((4,), 1.0 + g + g * g * 10.0))
    # (b)
    assert returns[0, 1] == pytest.approx(1.0 + g)
    assert returns[1, 1] == pytest.approx(1.0)
    assert mask[:, 1].tolist() == [True, True, False, False]
    # (c): slot 3 needs rows 4 and 5; row 5 is missing, so r3 + g r4 and no bootstrap
    assert returns[3, 2] == pytest.approx(1.0 + g)
    assert returns[2, 2] == pytest.approx(1.0 + g + g * g * 10.0)


def make_agent(kind='lstm', head='scalar', rescale=None, **kw):
    arch = small_arch(kind, head)
    if rescale is None:
        rescale = head != 'c51'
    return r2d2_agent.R2d2Agent(arch, burn_in=40, block=40, n_step=5, rescale_targets=rescale, seed=3, **kw)


def real_batch(kind='lstm', head='scalar', steps=400):
    """A batch collected by the real net on the scripted env, so stored states are the net's own."""
    agent = make_agent(kind, head)
    vec = ScriptedVec([[100], [60], [23], [100]])
    buffer = SequenceReplay(128, OBS, 4, agent.state_width, seq_length=80, burn_in=40, lookahead=5, seed=0)
    collector = Collector(vec, agent, buffer, np.full(4, 0.2), seed=0)
    for _ in range(steps):
        collector.step()
    batch, slots, weights = buffer.sample(16)
    return agent, buffer, batch, slots, weights


def test_an_update_runs_a_backward_pass_and_returns_one_priority_per_window():
    agent, buffer, batch, slots, weights = real_batch()
    priorities, metrics = agent.update(batch, weights)
    assert priorities.shape == (16,) and np.all(np.isfinite(priorities)) and np.all(priorities >= 0)
    assert np.isfinite(metrics['loss']) and metrics['train_step'] == 1 and 'grad_norm' in metrics


def test_the_burn_in_carries_memory_into_the_loss_for_the_lstm_and_nothing_for_the_dense_control():
    """Perturbing the observations in the burn-in slots moves the LSTM arm's loss (the state carried into
    the loss block changed) and leaves the dense arm's untouched (no state, and no loss there)."""
    for kind, moves in (('lstm', True), ('dense', False)):
        agent, buffer, batch, slots, weights = real_batch(kind)
        full = [b for b in range(16) if batch['burn'][b] == 40]
        if not full:
            pytest.skip('no full burn-in window drawn')
        torch.manual_seed(0)
        agent_state = copy.deepcopy(agent.state_dict())        # the live dict shares the parameters' storage
        _, before = agent.update(batch, weights)
        agent.load_state_dict(agent_state)
        agent.train_step = 0
        shaken = dict(batch)
        shaken['obs'] = batch['obs'].copy()
        shaken['obs'][:, :40, :] += 3.0
        _, after = agent.update(shaken, weights)
        assert (abs(after['loss'] - before['loss']) > 1e-7) is moves, kind


def test_the_dense_control_takes_its_loss_at_exactly_the_lstm_arms_positions():
    """The mask of loss positions is a property of the window, not the cell."""
    agent_l, _, batch, _, _ = real_batch('lstm')
    agent_d = make_agent('dense')
    b_l = agent_l._tensors(batch)
    b_d = agent_d._tensors({**batch, 'state': np.zeros((16, 0), dtype=np.float32)})
    on_l, tg_l = agent_l._unroll_both(b_l)
    on_d, tg_d = agent_d._unroll_both(b_d)
    tail = slice(40, 85)
    _, mask_l = agent_l._scalar(on_l, tg_l, b_l['action'][tail], b_l['reward'][tail], b_l['done'][tail], b_l['valid'][tail], 40)
    _, mask_d = agent_d._scalar(on_d, tg_d, b_d['action'][tail], b_d['reward'][tail], b_d['done'][tail], b_d['valid'][tail], 40)
    assert torch.equal(mask_l, mask_d)
    assert torch.equal(mask_l, b_l['valid'][40:80])


def test_a_stored_state_replayed_through_the_net_reproduces_the_collectors_state():
    """For a window with a full burn-in, unrolling the net from the stored state over the burn-in rows lands
    on the state the collector carried into the loss block's first row -- the next window's stored state."""
    agent, buffer, batch, slots, weights = real_batch('lstm')
    # Consecutive windows of one lane: window k+1's start row is window k's loss-block start row.
    by_start = {int(buffer.start[s]): s for s in range(buffer.n_windows)}
    checked = 0
    for s in range(buffer.n_windows):
        if buffer.burn[s] != 40:
            continue
        rows = buffer.window_rows(s)
        loss_first_row = int(rows[40])
        if loss_first_row not in by_start:
            continue
        other = by_start[loss_first_row]
        burn_rows = rows[:40]
        with torch.no_grad():
            state = torch.as_tensor(buffer.state[s]).unsqueeze(0)
            for row in burn_rows:
                _, state = agent.net.step(torch.as_tensor(buffer.obs[row]).unsqueeze(0),
                                          torch.as_tensor(buffer.prev_action[row]).unsqueeze(0),
                                          torch.as_tensor(buffer.prev_reward[row]).unsqueeze(0), state)
        assert np.allclose(state.squeeze(0).numpy(), buffer.state[other], atol=1e-5)
        checked += 1
    assert checked > 0


def test_the_c51_recipe_updates_and_refuses_rescaling():
    agent, buffer, batch, slots, weights = real_batch('lstm', 'c51')
    priorities, metrics = agent.update(batch, weights)
    assert np.all(np.isfinite(priorities)) and np.isfinite(metrics['loss'])
    with pytest.raises(ValueError, match='rescaling'):
        make_agent('lstm', 'c51', rescale=True)


def test_the_target_is_copied_on_the_period_and_not_before():
    agent, buffer, batch, slots, weights = real_batch()
    agent.target_update_period = 3
    before = [p.clone() for p in agent.target.parameters()]
    agent.update(batch, weights)
    agent.update(batch, weights)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.target.parameters()))
    agent.update(batch, weights)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.target.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.net.parameters(), agent.target.parameters()))


def test_acting_carries_the_state_and_explores_on_the_lanes_own_epsilon():
    agent = make_agent()
    obs = np.zeros((4, OBS), dtype=np.float32)
    state = np.zeros((4, agent.state_width), dtype=np.float32)
    actions, next_state = agent.act(obs, np.full(4, -1), np.zeros(4, np.float32), state, np.ones(4, bool), np.zeros(4))
    assert actions.shape == (4,) and next_state.shape == (4, 16)
    assert np.any(next_state != 0.0)
    # Epsilon 1 on lane 0 and 0 elsewhere: over many draws lane 0 varies, the others never leave greedy.
    greedy = actions.copy()
    seen = set()
    for _ in range(50):
        a, _ = agent.act(obs, np.full(4, -1), np.zeros(4, np.float32), state, np.ones(4, bool), np.array([1.0, 0, 0, 0]))
        seen.add(int(a[0]))
        assert a[1:].tolist() == greedy[1:].tolist()
    assert len(seen) > 1


# --- the net and the restore seam --------------------------------------------------------------------

def test_the_r2d2_sidecar_rebuilds_the_net_and_its_policy_is_stateful():
    arch = small_arch('lstm', 'c51')
    net = restore.build_net(arch)
    assert isinstance(net, network.R2d2Net) and net.recurrent and net.head_type == 'c51'
    policy = restore.policy_fn_for(arch, net)
    assert isinstance(policy, StatefulPolicy)
    out = policy(np.zeros((3, OBS), dtype=np.float32))
    assert out.shape == (3,)
    dense = restore.build_net(small_arch('dense'))
    assert not dense.recurrent and dense.state_width == 0


def test_a_fresh_row_starts_from_the_zero_state_in_step():
    net = network.build(small_arch())
    obs = torch.zeros(2, OBS)
    state = torch.ones(2, net.state_width)
    _, out = net.step(obs, torch.tensor([-1, -1]), torch.zeros(2), state, fresh=torch.tensor([True, False]))
    _, from_zero = net.step(obs[:1], torch.tensor([-1]), torch.zeros(1), torch.zeros(1, net.state_width))
    assert torch.allclose(out[0], from_zero[0])
    assert not torch.allclose(out[1], from_zero[0])


# --- the config -----------------------------------------------------------------------------------------

def tuned_from(values):
    def tuned(name, default, cast=float):
        return cast(values[name]) if name in values else default
    return tuned


def test_the_defaults_are_the_papers():
    c = r2d2_algo.build_config(tuned_from({}))
    assert (c['learning_rate'], c['adam_epsilon'], c['batch_size'], c['discount'], c['n_step_update']) == (1e-4, 1e-3, 64, 0.997, 5)
    assert (c['target_update_period'], c['gradient_clipping'], c['r2d2_seq_length'], c['r2d2_burn_in'], c['r2d2_stride']) == (2500, 40.0, 120, 40, 40)
    assert (c['priority_exponent'], c['r2d2_is_beta'], c['r2d2_priority_eta'], c['r2d2_hidden']) == (0.9, 0.6, 0.9, 512)
    assert c['epsilon_schedule'] == 'apex' and c['initial_epsilon'] == 0.4 and c['apex_alpha'] == 7.0


def test_c51_with_rescaling_on_and_unknown_cells_are_refused_by_name():
    with pytest.raises(ValueError, match='RESCALE'):
        r2d2_algo.build_config(tuned_from({'R2D2_HEAD': 'c51'}))
    with pytest.raises(ValueError, match='R2D2_RECURRENT'):
        r2d2_algo.build_config(tuned_from({'R2D2_RECURRENT': 'gru'}))
    with pytest.raises(ValueError, match='BURN_IN'):
        r2d2_algo.build_config(tuned_from({'R2D2_BURN_IN': 120}))
    with pytest.raises(ValueError, match='STRIDE'):
        r2d2_algo.build_config(tuned_from({'R2D2_STRIDE': 81}))


def test_a_foreign_knob_is_refused_by_name(monkeypatch):
    monkeypatch.setenv('SNEK_FORK_BRANCHES', '4')
    with pytest.raises(ValueError, match='SNEK_FORK_BRANCHES'):
        r2d2_algo.build_config(tuned_from({}))


def test_arch_fields_carry_the_cell_and_the_head():
    c = r2d2_algo.build_config(tuned_from({'R2D2_HEAD': 'c51', 'R2D2_RESCALE': 0, 'R2D2_RECURRENT': 'dense'}))
    fields = r2d2_algo.arch_fields(c)
    assert fields['recurrent'] == {'type': 'dense', 'hidden': 512, 'stream_width': 512, 'prev_input': True}
    assert fields['head']['type'] == 'c51' and fields['head']['atoms'] == 51


# --- added after the mutation run of 2026-09-30 (three survivors) and the paper's geometry ------------------------

def test_a_stride_below_the_block_overlaps_consecutive_windows_the_papers_way():
    """Block 80 every 40 steps: a 200-step episode gives loss blocks at 0, 40, 80, 120, 160; the first with no
    burn-in, the rest with 40; the last two cut short by the episode's end."""
    vec = ScriptedVec([[200]])
    buffer = SequenceReplay(64, OBS, 1, 2, seq_length=120, burn_in=40, lookahead=5, seed=0)
    collector = Collector(vec, CountingAgent(), buffer, np.full(1, 0.1), stride=40, seed=0)
    for _ in range(200):
        collector.step()
    assert windows_of(buffer) == [(0, 80, 5), (40, 80, 5), (40, 80, 5), (40, 80, 0), (40, 40, 0)]
    # Window 2 starts at episode step 0 (block at 40, 40 burn-in rows before it); window 3 at 40.
    assert buffer.state[1, 0] == 0.0 and buffer.state[2, 0] == 40.0
    with pytest.raises(ValueError, match='stride'):
        Collector(ScriptedVec([[10]]), CountingAgent(), buffer, np.full(1, 0.1), stride=81)


def test_the_n_step_sum_stops_at_a_terminal_even_when_rows_follow_it():
    """Synthetic: a terminal at slot 1 with valid rows after it (a real window never has them, which is why the
    `done` factor and the `valid` factor are both kept). The sum for slot 0 is r0 + g r1 and nothing bootstraps."""
    g = 0.5
    reward = torch.ones(6, 1)
    done = torch.zeros(6, 1, dtype=torch.bool)
    done[1, 0] = True
    valid = torch.ones(6, 1, dtype=torch.bool)
    boot = torch.full((6, 1), 10.0)
    returns, mask = r2d2_agent.n_step_targets(reward, done, valid, boot, g, 3, 0, 2)
    assert returns[0, 0] == pytest.approx(1.0 + g)
    assert returns[1, 0] == pytest.approx(1.0)


def test_the_burn_in_starts_from_the_stored_state_not_from_zero():
    """Windows with a full burn-in: the loss-block outputs differ when the stored state is replaced by zeros,
    so the learner is reading the state the collector stored (the paper's stored-state strategy)."""
    agent, buffer, batch, slots, weights = real_batch('lstm')
    full = [b for b in range(16) if batch['burn'][b] == 40]
    if not full:
        pytest.skip('no full burn-in window drawn')
    b = agent._tensors(batch)
    with torch.no_grad():
        stored, _ = agent._unroll_both(b)
        zeroed = dict(b); zeroed['state'] = torch.zeros_like(b['state'])
        from_zero, _ = agent._unroll_both(zeroed)
    assert not torch.allclose(stored[:, full], from_zero[:, full], atol=1e-6)
