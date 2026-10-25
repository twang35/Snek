"""NGU (`algos/ngu/`): the intrinsic reward's two terms, the per-lane memory and snapshot, the Retrace target, the
ladders and their permutation, the board the modules read, the replay's new columns, the policy through the seam
(`plans/algoExploration/f-ngu-agent57.md` §0-2).

What would be silent without these:

- **a memory not emptied at an episode start** scores the first boards of every episode as familiar, and r^i
  falls for a reason that has nothing to do with the game;
- **a lifelong factor without its clip** hands a high-β lane a reward in the hundreds on a novel board;
- **a trace coefficient above 1** turns Retrace into an off-policy estimator that does not converge;
- **an embedding snapshot that moves mid-episode** compares keys taken under two different f's;
- **the mixture one-hot at the wrong index** trains Q_j on another mixture's data with no error raised;
- **an eval policy whose r^i differs from the trainer's** is measured on an input distribution it never saw.
"""

import numpy as np
import pytest
import torch

from algos.ngu import agent as ngu_agent
from algos.ngu import algo as ngu_algo
from algos.ngu import board as board_codes
from algos.ngu import intrinsic
from algos.ngu import net as network
from algos.ngu.collect import NguCollector
from algos.ngu.replay import NguReplay
from algos.r2d2.agent import rescale, unscale
from env import constants
from tools import arch as arch_tools
from tools import restore
from vectorized import vec_env
from vectorized.vec_env import VecSnake

OBS = constants.OBS_LEN
ACTIONS = constants.NUM_ACTIONS


# --- fixtures --------------------------------------------------------------------------------------

def small_arch(mixtures=4, beta_max=0.3, inp='board', hidden=8, kernel_eps=1e-4, memory=64, embed=(8, 16)):
    return arch_tools.build_arch(
        (16,), ACTIONS, OBS, constants.OBS_ERA, algo='ngu',
        recurrent={'type': 'lstm', 'hidden': hidden, 'stream_width': 8, 'prev_input': True}, head={'type': 'scalar'},
        conditioning={'mixtures': mixtures, 'beta_max': beta_max, 'gamma_top': 0.997, 'gamma_min': 0.99, 'split': 'none'},
        intrinsic={'input': inp, 'embed_dim': embed[0], 'embed_hidden': embed[1], 'rnd_hidden': 8, 'rnd_out': 8, 'knn_k': 3,
                   'kernel_epsilon': kernel_eps, 'cluster_distance': 8e-3, 'pseudo_count': 1e-3, 'max_similarity': 8.0,
                   'memory': memory, 'lifelong_max': 5.0})


def make_agent(arch=None, **kw):
    arch = arch or small_arch()
    defaults = dict(burn_in=4, block=8, seed=1)
    defaults.update(kw)
    return ngu_agent.NguAgent(arch, **defaults)


def make_collector(lanes=4, beta_max=0.3, seed=1, windows=64, assign='fixed', agent=None):
    agent = agent or make_agent(small_arch(mixtures=lanes, beta_max=beta_max))
    buffer = NguReplay(windows, OBS, lanes, agent.state_width, board_len=vec_env.BOARD_CELLS, seq_length=12, burn_in=4,
                       lookahead=2, beta=0.0, eta=1.0, seed=seed)
    novelty = intrinsic.LaneNovelty(agent.net, lanes, training=True)
    vec = VecSnake(lanes, seed=seed)
    mixtures = np.arange(lanes) if assign == 'fixed' else np.zeros(lanes, dtype=np.int64)
    collector = NguCollector(vec, agent, buffer, [0.1] * lanes, novelty, mixtures, assign=assign, stride=4, seed=seed)
    return collector, buffer, agent, vec


def tuned_from(values):
    def tuned(name, default, cast=float):
        return cast(values[name]) if name in values else default
    return tuned


# --- the board ------------------------------------------------------------------------------------------

def test_the_board_codes_place_the_head_the_body_order_and_the_food():
    vec = VecSnake(2, seed=3)
    for _ in range(5):
        vec.step(np.array([1, 1]))
    codes = vec.board_codes()
    for i in range(2):
        body = vec.body_list(i)
        xy = [vec_env.unflat(cell) for cell in body]
        food = vec_env.unflat(int(vec.food[i])) if vec.food[i] >= 0 else None
        assert np.array_equal(codes[i], vec_env.board_codes_of(xy, food))
        head_x, head_y = xy[0]
        assert codes[i, head_y * 10 + head_x] == vec_env.HEAD_CODE
        tail_x, tail_y = xy[-1]
        assert codes[i, tail_y * 10 + tail_x] == vec_env.HEAD_CODE - (len(body) - 1)
        if food is not None:
            assert codes[i, food[1] * 10 + food[0]] == vec_env.FOOD_CODE
        assert np.count_nonzero(codes[i]) == len(body) + (food is not None)


def test_the_decoded_board_has_four_channels_and_the_head_is_the_oldest_age():
    codes = np.zeros((1, 100), dtype=np.uint8)
    codes[0, 5] = vec_env.HEAD_CODE
    codes[0, 6] = vec_env.HEAD_CODE - 1
    codes[0, 7] = vec_env.FOOD_CODE
    out = board_codes.decode(torch.as_tensor(codes))
    assert out.shape == (1, 400)
    body, head, food, age = out[0, :100], out[0, 100:200], out[0, 200:300], out[0, 300:]
    assert body[5] == 1 and body[6] == 1 and body[7] == 0
    assert head[5] == 1 and head[6] == 0
    assert food[7] == 1 and food.sum() == 1
    assert age[5] == 1.0 and 0.99 < age[6] < 1.0 and age[7] == 0


# --- Alg. 1 and the lifelong term ------------------------------------------------------------------------

def test_alg_1_by_hand_on_three_keys():
    k, eps, xi, c, sm = 3, 1e-4, 8e-3, 1e-3, 8.0
    dist = torch.tensor([[0.5, 1.0, 2.0]])
    mean = torch.tensor([(0.5 + 1.0 + 2.0) / 3.0])
    dn = dist / mean
    dn = (dn - xi).clamp(min=0.0)
    kernel = eps / (dn + eps)
    s = float(torch.sqrt(kernel.sum() + c))
    expected = 0.0 if s > sm else 1.0 / s
    got = intrinsic.episodic_reward(dist, mean, k, eps, xi, c, sm)
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(expected)


def test_a_repeated_board_earns_less_each_time_and_the_memory_resets_with_the_episode():
    memory = intrinsic.EpisodicMemory(1, 64, 4, 3)
    key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    other = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    memory.insert([0], other)
    rewards = []
    for _ in range(4):
        nearest, mean = memory.query([0], key)
        rewards.append(float(intrinsic.episodic_reward(nearest, mean, 3, 1e-4, 8e-3, 1e-3, 8.0)[0]))
        memory.insert([0], key)
    assert rewards[0] > rewards[1] > rewards[2]
    assert rewards[-1] == 0.0 or rewards[-1] < rewards[0] / 2
    memory.reset([0])
    assert memory.sizes()[0] == 0
    nearest, mean = memory.query([0], key)
    assert not torch.isfinite(nearest).any()
    assert float(intrinsic.episodic_reward(nearest, mean, 3, 1e-4, 8e-3, 1e-3, 8.0)[0]) == 0.0


def test_begin_episode_empties_the_lanes_memory_and_stores_the_first_board():
    net = network.build(small_arch(), seed=0)
    novelty = intrinsic.LaneNovelty(net, 2, training=True)
    inputs = torch.as_tensor(np.random.default_rng(0).random((2, 400)).astype(np.float32))
    novelty.begin_episode([0, 1], inputs)
    for _ in range(5):
        novelty.reward([0, 1], inputs + 0.1)
    assert list(novelty.memory.sizes()) == [6, 6]
    novelty.begin_episode([1], inputs[1:])
    assert list(novelty.memory.sizes()) == [6, 1]
    assert float(novelty.memory.dist_n[1]) == 0.0


def test_the_memory_grows_in_steps_without_changing_what_a_query_returns():
    grow = intrinsic.EpisodicMemory.GROW
    small, big = intrinsic.EpisodicMemory(2, 4 * grow, 3, 2), intrinsic.EpisodicMemory(2, 4 * grow, 3, 2)
    big._ensure(4 * grow)
    rng = np.random.default_rng(0)
    for _ in range(grow + 5):
        keys = torch.as_tensor(rng.normal(size=(2, 3)).astype(np.float32))
        a = small.query([0, 1], keys)
        b = big.query([0, 1], keys)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        small.insert([0, 1], keys)
        big.insert([0, 1], keys)
    assert small.allocated == 2 * grow and big.allocated == 4 * grow


def test_the_lifelong_factor_is_one_before_statistics_and_clipped_to_the_cap():
    err = torch.tensor([0.0, 1.0, 100.0])
    assert torch.equal(intrinsic.lifelong_factor(err, torch.tensor(0.0), torch.tensor(0.0), 5.0), torch.ones(3))
    got = intrinsic.lifelong_factor(err, torch.tensor(1.0), torch.tensor(1.0), 5.0)
    assert got.tolist() == [1.0, 1.0, 5.0]


def test_the_rnd_error_falls_on_boards_the_predictor_trains_on_and_stays_on_novel_ones():
    net = network.build(small_arch(), seed=0)
    agent = make_agent()
    seen = torch.as_tensor(np.random.default_rng(1).random((16, 400)).astype(np.float32))
    novel = torch.as_tensor(np.random.default_rng(2).random((16, 400)).astype(np.float32))
    with torch.no_grad():
        before_seen, before_novel = float(agent.net.rnd_error(seen).mean()), float(agent.net.rnd_error(novel).mean())
    for _ in range(300):
        loss = agent.net.rnd_error(seen).mean()
        agent.rnd_optimizer.zero_grad()
        loss.backward()
        agent.rnd_optimizer.step()
    with torch.no_grad():
        after_seen, after_novel = float(agent.net.rnd_error(seen).mean()), float(agent.net.rnd_error(novel).mean())
    assert after_seen < before_seen * 0.5
    assert after_novel > after_seen
    del net


def test_welford_matches_numpy_over_batches():
    net = network.build(small_arch())
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 3.0, size=50).astype(np.float32)
    for chunk in np.array_split(data, 5):
        net.update_rnd_stats(torch.as_tensor(chunk))
    assert float(net.rnd_mean) == pytest.approx(float(data.mean()), rel=1e-4)
    assert float(net.rnd_std()) == pytest.approx(float(data.std(ddof=1)), rel=1e-4)


# --- the embedding ----------------------------------------------------------------------------------------

def test_inverse_dynamics_on_the_board_beats_chance_and_shuffled_boards_do_not():
    """Held-out accuracy, so the classifier cannot pass by memorising the training boards."""
    vec = VecSnake(8, seed=5)
    rng = np.random.default_rng(5)
    boards, nexts, actions = [], [], []
    for _ in range(300):
        before = vec.board_codes()
        a = rng.integers(0, ACTIONS, size=8)
        _, _, done, _ = vec.step(a)
        after = vec.board_codes()
        keep = ~done
        boards.append(before[keep]); nexts.append(after[keep]); actions.append(a[keep])
    now = board_codes.decode(torch.as_tensor(np.concatenate(boards)))
    nxt = board_codes.decode(torch.as_tensor(np.concatenate(nexts)))
    taken = torch.as_tensor(np.concatenate(actions))
    split = int(0.7 * taken.shape[0])

    def held_out_accuracy(now, nxt):
        agent = make_agent(small_arch(embed=(16, 64)), seed=0)
        for _ in range(800):
            loss = torch.nn.functional.cross_entropy(agent.net.classify(now[:split], nxt[:split]), taken[:split])
            agent.embed_optimizer.zero_grad()
            loss.backward()
            agent.embed_optimizer.step()
        with torch.no_grad():
            return float((agent.net.classify(now[split:], nxt[split:]).argmax(-1) == taken[split:]).float().mean())

    assert held_out_accuracy(now, nxt) > 0.55                      # chance is 1/3
    perm = torch.as_tensor(rng.permutation(taken.shape[0]))
    assert held_out_accuracy(now, nxt[perm]) < 0.45                 # the successor from another step: nothing to read


def test_the_lanes_snapshot_of_the_embedding_does_not_move_mid_episode():
    net = network.build(small_arch(), seed=0)
    novelty = intrinsic.LaneNovelty(net, 2, training=True)
    inputs = torch.as_tensor(np.random.default_rng(0).random((2, 400)).astype(np.float32))
    novelty.begin_episode([0, 1], inputs)
    before = novelty.snapshots.embed([0, 1], inputs)
    with torch.no_grad():
        for p in net.embed_first.parameters():
            p.add_(1.0)
    novelty.reward([0, 1], inputs + 0.1)
    assert torch.equal(novelty.snapshots.embed([0, 1], inputs), before)
    assert not torch.allclose(net.embed(inputs), before)
    novelty.begin_episode([1], inputs[1:])
    assert torch.allclose(novelty.snapshots.embed([1], inputs[1:]), net.embed(inputs[1:]))
    assert torch.equal(novelty.snapshots.embed([0], inputs[:1]), before[:1])


def test_an_eval_policy_computes_the_same_episodic_term_as_the_trainer():
    net = network.build(small_arch(), seed=0)
    trainer = intrinsic.LaneNovelty(net, 1, training=True)
    evaluator = intrinsic.LaneNovelty(net, 1, training=False)
    vec = VecSnake(1, seed=2)
    codes = [vec.board_codes()]
    for _ in range(30):
        _, _, done, _ = vec.step(np.array([1]))
        if done[0]:
            break
        codes.append(vec.board_codes())
    inputs = [board_codes.decode(torch.as_tensor(c)) for c in codes]
    trainer.begin_episode([0], inputs[0])
    evaluator.begin_episode([0], inputs[0])
    for x in inputs[1:]:
        trainer.reward([0], x)
        evaluator.reward([0], x)
        assert trainer.last['episodic'] == evaluator.last['episodic']


# --- the ladders -------------------------------------------------------------------------------------------

def test_the_beta_ladder_has_the_papers_endpoints_and_rises():
    betas = network.beta_ladder(32, 0.3)
    assert betas[0] == 0.0 and betas[-1] == 0.3
    assert np.all(np.diff(betas) > 0)
    assert np.all(network.beta_ladder(32, 0.0) == 0.0)


def test_the_gamma_ladder_falls_strictly_between_the_papers_endpoints():
    gammas = network.gamma_ladder(32, 0.997, 0.99)
    assert gammas[0] == pytest.approx(0.997) and gammas[-1] == pytest.approx(0.99)
    assert np.all(np.diff(gammas) < 0)


def test_the_epsilon_permutation_is_fixed_across_seeds_and_uncorrelated_with_the_index():
    a = ngu_algo.permuted_rungs(32, True)
    b = ngu_algo.permuted_rungs(32, True)
    assert np.array_equal(a, b) and sorted(a) == list(range(32))
    assert not np.array_equal(a, np.arange(32))
    ranks = np.argsort(np.argsort(a))
    spearman = np.corrcoef(ranks, np.arange(32))[0, 1]
    assert abs(spearman) < 0.3
    assert np.array_equal(ngu_algo.permuted_rungs(32, False), np.arange(32))


def test_the_cell_input_carries_the_one_hot_at_the_lanes_own_index():
    net = network.build(small_arch(mixtures=4))
    features = torch.zeros(3, 16)
    mixture = torch.tensor([0, 2, 3])
    out = net._cell_input(features, torch.full((3,), -1), torch.zeros(3), torch.zeros(3), mixture)
    one_hot = out[:, 16:20]
    assert torch.equal(one_hot, torch.nn.functional.one_hot(mixture, 4).float())
    assert out.shape[1] == 16 + 4 + ACTIONS + 2


# --- Retrace ------------------------------------------------------------------------------------------------

def _retrace_inputs(mu_equals_pi, lam=0.9, gamma=0.9):
    T, B, A = 3, 1, 2
    q_online = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])      # greedy action 0 everywhere
    q_target = torch.tensor([[[2.0, 1.0]], [[1.5, 3.0]], [[4.0, 2.0]]])      # the target net prefers action 1 at slot 1
    action = torch.zeros(T, B, dtype=torch.int64)
    reward = torch.tensor([[1.0], [1.0], [1.0]])
    done = torch.zeros(T, B, dtype=torch.bool)
    valid = torch.ones(T, B, dtype=torch.bool)
    eps = 0.0
    mu = torch.ones(T, B) if mu_equals_pi else torch.full((T, B), 0.25)
    return q_online, q_target, action, reward, done, valid, mu, torch.tensor([gamma]), lam, eps


def test_retrace_with_mu_equal_to_pi_is_the_lambda_return_truncated_at_the_window():
    q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps = _retrace_inputs(True)
    target, mask = ngu_agent.retrace_targets(q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps, 2,
                                             False, 1e-3)
    g = 0.9
    # pi greedy on the ONLINE net (action 0 everywhere): V = Q_target(., 0) = 2, 1.5, 4 -- not the target net's own
    # argmax at slot 1. delta_0 = 1 + g*1.5 - 2 = 0.35; delta_1 = 1 + g*4 - 1.5 = 3.1; delta_2 = 0 (no successor stored).
    # G_1 = 1.5 + 3.1; G_0 = 2 + 0.35 + g*lam*3.1.
    assert float(target[1, 0]) == pytest.approx(1.5 + 3.1)
    assert float(target[0, 0]) == pytest.approx(2 + 0.35 + g * lam * 3.1)
    assert mask.all()


def test_retrace_caps_the_trace_at_one_when_mu_is_below_pi():
    q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps = _retrace_inputs(False)
    capped, _ = ngu_agent.retrace_targets(q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps, 2, False, 1e-3)
    same, _ = ngu_agent.retrace_targets(*_retrace_inputs(True)[:6], torch.ones(3, 1), gamma, lam, eps, 2, False, 1e-3)
    # pi/mu = 4 > 1: the trace is min(1, 4) = 1, the same as mu = pi.
    assert torch.allclose(capped, same)


def test_retrace_stops_at_a_terminal_and_the_rescaled_form_inverts_the_target_net():
    q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps = _retrace_inputs(True)
    done[0, 0] = True
    target, _ = ngu_agent.retrace_targets(q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps, 1, False, 1e-3)
    assert float(target[0, 0]) == pytest.approx(1.0)                   # r alone: Q + (r - Q)
    q_h = rescale(q_target, 1e-3)
    target_h, _ = ngu_agent.retrace_targets(q_online, q_h, action, reward, done, valid, mu, gamma, lam, eps, 1, True, 1e-3)
    assert float(unscale(target_h, 1e-3)[0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_the_target_policy_is_epsilon_soft_on_the_online_argmax():
    q_online, q_target, action, reward, done, valid, mu, gamma, lam, _ = _retrace_inputs(True)
    eps = 0.5
    target, _ = ngu_agent.retrace_targets(q_online, q_target, action, reward, done, valid, mu, gamma, lam, eps, 1, False, 1e-3)
    # V(x_1) = 0.75 * 1.5 + 0.25 * 3.0 = 1.875; delta_0 = 1 + 0.9 * 1.875 - 2; c_1 = lam * min(1, 0.75 / 1) = 0.675;
    # delta_1 = 1 + 0.9 * (0.75 * 4 + 0.25 * 2) - 1.5 = 2.65; G_0 = 2 + delta_0 + 0.9 * 0.675 * 2.65.
    delta_0 = 1 + 0.9 * 1.875 - 2
    assert float(target[0, 0]) == pytest.approx(2 + delta_0 + 0.9 * 0.675 * 2.65)


def test_the_aux_pairs_are_the_last_n_valid_non_terminal_ones():
    valid = torch.ones(8, 1, dtype=torch.bool)
    valid[6:, 0] = False
    done = torch.zeros(8, 1, dtype=torch.bool)
    done[3, 0] = True
    picked = ngu_agent._last_pairs(valid, done, 3)
    # pairs (s, s+1) with both valid and s not terminal: s in {0, 1, 2, 4}; the last three are 1, 2, 4.
    assert picked[:, 0].nonzero().flatten().tolist() == [1, 2, 4]


# --- the collector and the replay ------------------------------------------------------------------------------

def test_acting_reports_the_behaviour_probability_of_the_action_taken():
    agent = make_agent()
    n = 200
    obs = np.zeros((n, OBS), dtype=np.float32)
    actions, _, mu = agent.act(obs, np.full(n, -1), np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64),
                               np.zeros((n, agent.state_width), dtype=np.float32), np.ones(n, dtype=bool), np.full(n, 0.3))
    greedy_mu, explore_mu = 1 - 0.3 + 0.3 / ACTIONS, 0.3 / ACTIONS
    is_greedy, is_explore = np.isclose(mu, greedy_mu), np.isclose(mu, explore_mu)
    assert np.all(is_greedy | is_explore)
    assert is_explore.sum() > 0 and is_greedy.sum() > n // 2


def test_at_beta_zero_the_stored_reward_is_the_extrinsic_one_to_the_bit():
    collector, buffer, agent, vec = make_collector(beta_max=0.0)
    seen = []
    real_step = vec.step

    def spy(actions):
        out = real_step(actions)
        seen.append(out[1].astype(np.float32).copy())
        return out
    vec.step = spy
    for _ in range(30):
        collector.step()
    stored = buffer.reward[:buffer.rows].reshape(-1, vec.n)
    assert np.array_equal(stored, np.stack(seen))
    assert np.any(buffer.reward_int[:buffer.rows] > 0)


def test_the_intrinsic_term_enters_the_reward_at_the_lanes_beta_and_a_terminal_row_carries_none():
    collector, buffer, agent, vec = make_collector(beta_max=0.3)
    seen = []
    real_step = vec.step

    def spy(actions):
        out = real_step(actions)
        seen.append(out[1].astype(np.float32).copy())
        return out
    vec.step = spy
    steps = 0
    while steps < 200 and not buffer.done[:buffer.rows].any():
        collector.step()
        steps += 1
    rows = buffer.rows
    ext = np.stack(seen).reshape(-1)
    betas = agent.net.betas.numpy()[buffer.mixture[0] * 0 + np.tile(np.arange(vec.n), rows // vec.n)]
    assert np.array_equal(buffer.reward[:rows], ext + betas * buffer.reward_int[:rows])
    assert np.all(buffer.reward_int[:rows][buffer.done[:rows]] == 0.0)
    assert np.all(buffer.reward_int[:rows][buffer.mixture[0] * 0 + np.tile(np.arange(vec.n), rows // vec.n) == 1] >= 0.0)


def test_a_fresh_lane_draws_a_random_previous_action_and_zero_previous_rewards():
    collector, buffer, agent, vec = make_collector()
    assert set(collector.prev_action.tolist()) <= set(range(ACTIONS))
    steps = 0
    while steps < 300 and not buffer.done[:buffer.rows].any():
        collector.step()
        steps += 1
    rows = buffer.rows
    finished = np.flatnonzero(buffer.done[:rows])
    assert finished.size
    after = finished + vec.n                                  # the same lane's next row: the next episode's first
    after = after[after < rows]
    assert np.all(buffer.prev_action[after] >= 0)
    assert np.all(buffer.prev_reward[after] == 0.0) and np.all(buffer.prev_reward_int[after] == 0.0)


def test_a_window_carries_its_lanes_mixture_and_the_batch_carries_the_new_columns():
    collector, buffer, agent, vec = make_collector()
    for _ in range(40):
        collector.step()
    assert buffer.n_windows >= 4
    starts = buffer.start[:buffer.n_windows]
    assert np.array_equal(buffer.mixture[:buffer.n_windows], starts % vec.n)
    batch, slots, weights = buffer.sample(4)
    for key in ('reward_int', 'prev_reward_int', 'mu', 'board', 'mixture'):
        assert key in batch
    assert batch['board'].shape == (4, buffer.slots, vec_env.BOARD_CELLS)
    assert batch['board'].dtype == np.uint8
    assert np.all(weights == 1.0)
    assert np.all((batch['mu'] > 0.0) & (batch['mu'] <= 1.0))


def test_the_replay_survives_a_save_and_a_load_with_its_columns(tmp_path):
    collector, buffer, agent, vec = make_collector()
    for _ in range(40):
        collector.step()
    buffer.save(str(tmp_path))
    other = NguReplay(64, OBS, vec.n, agent.state_width, board_len=vec_env.BOARD_CELLS, seq_length=12, burn_in=4, lookahead=2)
    assert other.load(str(tmp_path))
    rows = buffer.rows
    assert np.array_equal(other.board[:rows], buffer.board[:rows])
    assert np.array_equal(other.mu[:rows], buffer.mu[:rows])
    assert np.array_equal(other.mixture[:buffer.n_windows], buffer.mixture[:buffer.n_windows])


# --- the learner -------------------------------------------------------------------------------------------------

def test_an_update_runs_the_three_optimisers_and_returns_one_priority_per_window():
    collector, buffer, agent, vec = make_collector()
    for _ in range(60):
        collector.step()
    batch, slots, weights = buffer.sample(8)
    before = {name: p.detach().clone() for name, p in agent.net.named_parameters()}
    priorities, metrics = agent.update(batch, weights)
    assert priorities.shape == (8,) and np.all(np.isfinite(priorities))
    moved = {name for name, p in agent.net.named_parameters() if not torch.equal(p, before[name])}
    assert any(name.startswith('cell') for name in moved)
    assert any(name.startswith('embed_') for name in moved)
    assert any(name.startswith('rnd_predictor') for name in moved)
    assert not any(name.startswith('rnd_target') for name in moved)
    for key in ('td_loss', 'embed_loss', 'embed_acc', 'rnd_loss', 'aux_pairs'):
        assert key in metrics


def test_the_q_optimiser_holds_no_intrinsic_parameter():
    agent = make_agent()
    core = {id(p) for group in agent.optimizer.param_groups for p in group['params']}
    for name, p in agent.net.named_parameters():
        if name.startswith(('embed_', 'classifier', 'rnd_')):
            assert id(p) not in core, name
        else:
            assert id(p) in core, name


# --- the seam -------------------------------------------------------------------------------------------------------

def test_the_sidecar_rebuilds_the_net_and_the_policy_reads_the_board():
    arch = small_arch()
    net = restore.build_net(arch)
    assert isinstance(net, network.NguNet)
    policy = restore.policy_fn_for(arch, net)
    assert policy.needs_board and policy.mixture == network.EXPLOIT_INDEX
    with pytest.raises(ValueError):
        policy.begin(np.array([0]), np.array([True]), np.array([0.0], dtype=np.float32))
    variant = restore.policy_fn_for(arch, net, variant='mixture:3')
    assert variant.mixture == 3
    with pytest.raises(ValueError):
        restore.policy_fn_for(arch, net, variant='mixture:9')
    with pytest.raises(ValueError):
        restore.policy_fn_for(arch, net, variant='cvar:0.5')


def test_the_greedy_read_is_the_exploitative_mixture_and_draws_a_random_first_action():
    net = network.build(small_arch(), seed=0)
    policy = network.greedy_policy_fn(net)
    assert float(net.betas[policy.mixture]) == 0.0
    assert float(net.gammas[policy.mixture]) == float(net.gammas.max())
    seen = []
    real_step = net.step

    def spy(obs, prev_action, *rest):
        seen.append(prev_action.clone())
        return real_step(obs, prev_action, *rest)
    net.step = spy
    board = np.zeros((40, 100), dtype=np.uint8)
    policy.begin(np.arange(40), np.ones(40, dtype=bool), np.zeros(40, dtype=np.float32), board=board)
    policy(np.zeros((40, OBS), dtype=np.float32))
    first = seen[0]
    assert (first >= 0).all() and len(set(first.tolist())) > 1      # drawn, not "none" and not one constant


def test_the_policy_plays_through_the_engine_and_its_memory_follows_the_lanes():
    from vectorized import engine
    net = network.build(small_arch(), seed=0)
    policy = network.greedy_policy_fn(net)
    result = engine.measure(policy, 6, lanes=3, seed=1)
    assert len(result['scores']) == 6
    assert policy.novelty.lanes >= 3
    assert float(policy.prev_reward_int.abs().sum()) >= 0.0


def test_an_obs_input_policy_needs_no_board():
    arch = small_arch(inp='obs')
    net = restore.build_net(arch)
    assert net.intrinsic_len == OBS
    policy = restore.policy_fn_for(arch, net)
    assert not policy.needs_board
    policy.begin(np.array([0]), np.array([True]), np.array([0.0], dtype=np.float32))
    assert policy(np.zeros((1, OBS), dtype=np.float32)).shape == (1,)


def test_arch_fields_carry_the_conditioning_and_the_intrinsic_constants():
    config = ngu_algo.build_config(tuned_from({}))
    extra = ngu_algo.arch_fields(config)
    assert extra['conditioning'] == {'mixtures': 32, 'beta_max': 0.3, 'gamma_top': 0.997, 'gamma_min': 0.99, 'split': 'none'}
    assert extra['intrinsic']['input'] == 'board' and extra['intrinsic']['memory'] == 30000
    assert set(extra['intrinsic']) == set(network.INTRINSIC_FIELDS)
    assert extra['head'] == {'type': 'scalar'}


# --- the config -----------------------------------------------------------------------------------------------------

def test_the_defaults_are_the_papers():
    c = ngu_algo.build_config(tuned_from({}))
    assert (c['adam_epsilon'], c['target_update_period'], c['r2d2_is_beta'], c['r2d2_priority_eta'], c['apex_alpha']) == (1e-4, 1500, 0.0, 1.0, 8.0)
    assert (c['ngu_mixtures'], c['ngu_beta_max'], c['ngu_gamma_min'], c['ngu_retrace_lambda'], c['ngu_target_epsilon']) == (32, 0.3, 0.99, 0.95, 0.01)
    assert (c['ngu_knn_k'], c['ngu_kernel_epsilon'], c['ngu_cluster_distance'], c['ngu_pseudo_count'], c['ngu_max_similarity'], c['ngu_memory'], c['ngu_lifelong_max']) == (10, 1e-4, 8e-3, 1e-3, 8.0, 30000, 5.0)
    assert c['ngu_min_windows'] == 6250 and c['ngu_assign'] == 'fixed' and c['ngu_epsilon_permute'] and c['ngu_input'] == 'board'
    assert (c['r2d2_seq_length'], c['r2d2_burn_in'], c['r2d2_stride']) == (120, 40, 40)


@pytest.mark.parametrize('values', [
    {'NGU_ASSIGN': 'bandit'}, {'NGU_RETRACE_LAMBDA': '0'}, {'NGU_INPUT': 'frame'}, {'COLLECT_ENVS': '16'},
    {'NGU_GAMMA_MIN': '0.999'}, {'NGU_MIN_WINDOWS': '200000'}, {'NGU_BETA_MAX': '-1'}])
def test_bad_cells_are_refused_by_name(values):
    with pytest.raises(ValueError):
        ngu_algo.build_config(tuned_from(values))


def test_a_foreign_knob_is_refused_by_name(monkeypatch):
    monkeypatch.setenv('SNEK_FORK_BRANCHES', '4')
    with pytest.raises(ValueError, match='SNEK_FORK_BRANCHES'):
        ngu_algo.build_config(tuned_from({}))


def test_r2d2_refuses_the_ngu_knobs(monkeypatch):
    from algos.r2d2 import algo as r2d2_algo
    monkeypatch.setenv('SNEK_NGU_BETA_MAX', '0')
    with pytest.raises(ValueError, match='SNEK_NGU_BETA_MAX'):
        r2d2_algo.build_config(tuned_from({}))


def test_the_prefill_waits_for_the_papers_window_count():
    from algos.dqn import schedules  # noqa: F401  (import check only)
    config = ngu_algo.build_config(tuned_from({'COLLECT_ENVS': '4', 'NGU_MIXTURES': '4', 'R2D2_WINDOWS': '300',
                                               'NGU_MIN_WINDOWS': '70', 'INITIAL_COLLECT_STEPS': '8', 'R2D2_HIDDEN': '8',
                                               'R2D2_STREAM_WIDTH': '8', 'NGU_EMBED_HIDDEN': '8', 'NGU_RND_HIDDEN': '8',
                                               'NGU_RND_OUT': '8', 'NGU_EMBED_DIM': '4', 'NGU_MEMORY': '64',
                                               'R2D2_SEQ_LENGTH': '12', 'R2D2_BURN_IN': '4', 'R2D2_STRIDE': '4', 'N_STEP_UPDATE': '2'}))
    config.update({'seed': 1, 'fc_layers': (16,)})
    arch = arch_tools.build_arch((16,), ACTIONS, OBS, constants.OBS_ERA, algo='ngu', **ngu_algo.arch_fields(config))
    algo = ngu_algo.build(config, arch)
    algo.prefill()
    assert algo.buffer.n_windows >= 70
    fields = algo.fields()
    for key in ('intrinsic_mean', 'memory_mean', 'updates_per_s', 'rss_gb', 'windows'):
        assert key in fields['ngu']
