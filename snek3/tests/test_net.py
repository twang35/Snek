"""The Q-network, its initialisers, the `policy_fn` seam, and the TF weight transpose.

Three separate things are pinned here and each has a distinct failure mode.

**The transpose.** Keras kernels are `(in, out)`; torch `Linear` weights are `(out, in)`. Get it
backwards on a square layer and nothing raises — the network computes a different function with no
shape error anywhere. `state_dict_from` is tested against a hand-written numpy forward pass rather
than against itself.

**The initialiser correction.** snek2's hidden layers used Keras `VarianceScaling(scale=2.0,
mode='fan_in', distribution='truncated_normal')`, which divides σ by 0.8796 to compensate for
truncation pulling the realised variance down. Drop that and every snek3 arm starts ~12% narrower
than every snek2 arm did — a difference no test of the forward pass would ever see.

**The seam.** `policy_fn` is `(m, obs_len) float32 -> (m,) int64` and nothing else, which is what
keeps torch out of `vectorized/`.
"""

import math

import numpy as np
import pytest
import torch

from dqn import net as network
from tools import import_tf_checkpoint as importer


def test_the_shape_comes_from_the_arch():
    net = network.build({'obs_len': 30, 'fc_layer_params': [64, 32], 'num_actions': 3,
                         'algo': 'dqn', 'obs_era': 'x'})
    assert [layer.weight.shape for layer in net.hidden] == [(64, 30), (32, 64)]
    assert net.head.weight.shape == (3, 32)


def test_the_head_has_no_activation():
    """Q-values are unbounded, so a relu on the head would clamp every negative one to zero.

    That is not a small error: the death penalty is -5 and the discount is near 1, so most of the
    Q-values a mid-game state produces are negative, and clamping them makes the argmax over three
    actions a tie at 0. Asserted by driving a *known* negative through the head.
    """
    net = network.QNet(4, [4], 2)
    with torch.no_grad():
        for layer in net.hidden:
            layer.weight.fill_(0.0)
            layer.bias.fill_(1.0)
        net.head.weight.fill_(0.0)
        net.head.bias.copy_(torch.tensor([-3.0, 2.0]))
    values = net(torch.zeros(1, 4))
    assert values[0, 0].item() == pytest.approx(-3.0)
    assert values[0, 1].item() == pytest.approx(2.0)


def test_the_hidden_layers_are_relu_activated():
    net = network.QNet(1, [1], 1)
    with torch.no_grad():
        net.hidden[0].weight.fill_(1.0)
        net.hidden[0].bias.fill_(0.0)
        net.head.weight.fill_(1.0)
        net.head.bias.fill_(0.0)
    # A relu clips the negative input to 0; a linear layer would pass -2 straight through.
    assert net(torch.tensor([[-2.0]]))[0, 0].item() == pytest.approx(0.0)
    assert net(torch.tensor([[2.0]]))[0, 0].item() == pytest.approx(2.0)


def test_the_hidden_initialiser_reproduces_keras_variance_scaling():
    """The realised standard deviation must land on He's `sqrt(2 / fan_in)`, not below it.

    This is the assertion the 0.8796 divisor exists for: a plain `trunc_normal_` at He's σ realises
    0.8796 of it. The window is tight enough (±3%) that dropping the divisor fails, and wide enough
    that a 320-wide layer's sampling noise does not.
    """
    fan_in = 512
    net = network.QNet(fan_in, [2048], 3)
    realised = net.hidden[0].weight.std().item()
    target = math.sqrt(2.0 / fan_in)
    assert abs(realised - target) / target < 0.03, (realised, target)
    # And truncated, not merely normal: nothing may exceed 2σ of the *pre*-correction sigma.
    bound = 2 * target / 0.87962566103423978
    assert net.hidden[0].weight.abs().max().item() <= bound * 1.001


def test_the_head_starts_near_zero_so_the_opening_policy_is_near_uniform():
    net = network.QNet(30, [320], 3)
    assert net.head.weight.abs().max().item() <= 0.03
    assert net.head.bias.abs().max().item() == 0.0


def test_the_policy_fn_contract():
    net = network.QNet(30, [16], 3)
    policy_fn = network.greedy_policy_fn(net)
    actions = policy_fn(np.zeros((5, 30), dtype=np.float32))
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.int64 and actions.shape == (5,)
    assert actions.min() >= 0 and actions.max() < 3


def test_the_policy_fn_is_the_argmax_over_q():
    net = network.QNet(3, [3], 3)
    with torch.no_grad():
        # Identity through the hidden layer, so the head sees the observation itself and the Q for
        # action i is the input's ith component — making the expected argmax obvious.
        net.hidden[0].weight.copy_(torch.eye(3))
        net.hidden[0].bias.fill_(0.0)
        net.head.weight.copy_(torch.eye(3))
        net.head.bias.fill_(0.0)
    observations = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert list(network.greedy_policy_fn(net)(observations)) == [0, 2, 1]


def test_the_policy_fn_builds_no_autograd_graph():
    # Not a speed point. A 500-episode measurement under a live graph holds every intermediate
    # activation of every step alive, which is a leak measured in gigabytes.
    net = network.QNet(30, [16], 3)
    net.hidden[0].weight.requires_grad_(True)
    network.greedy_policy_fn(net)(np.zeros((4, 30), dtype=np.float32))
    assert net.hidden[0].weight.grad is None


def test_the_policy_fn_leaves_the_net_in_eval_mode():
    net = network.QNet(30, [16], 3)
    net.train()
    network.greedy_policy_fn(net)
    assert not net.training


# ------------------------------------------------- the TF conversion, without TensorFlow

def tf_style_arrays(rng, widths):
    """Kernels and biases in Keras's `(in, out)` orientation, as `tf_export.py` writes them."""
    arrays = {'layers': np.asarray(len(widths) - 1)}
    for index in range(len(widths) - 1):
        arrays['kernel{0}'.format(index)] = rng.standard_normal(
            (widths[index], widths[index + 1])).astype(np.float32)
        arrays['bias{0}'.format(index)] = rng.standard_normal(
            widths[index + 1]).astype(np.float32)
    return arrays


def keras_forward(arrays, observations):
    """The network Keras computes from those arrays: relu-activated dense layers, linear head."""
    layers = int(arrays['layers'])
    values = observations
    for index in range(layers):
        values = values @ arrays['kernel{0}'.format(index)] + arrays['bias{0}'.format(index)]
        if index < layers - 1:
            values = np.maximum(values, 0.0)
    return values


def test_a_converted_network_computes_the_same_function():
    """The whole point of the importer, checked against an independent forward pass.

    Independent matters: comparing `state_dict_from`'s output against a transpose of itself would
    pass with the orientation backwards. This multiplies the raw `(in, out)` kernels in numpy the way
    Keras does and requires the torch net to agree.
    """
    rng = np.random.default_rng(7)
    arrays = tf_style_arrays(rng, [30, 320, 3])
    state, fc_layer_params, obs_len, num_actions = importer.state_dict_from(arrays)
    assert (fc_layer_params, obs_len, num_actions) == ([320], 30, 3)

    net = network.QNet(obs_len, fc_layer_params, num_actions)
    net.load_state_dict(state, strict=True)
    observations = rng.random((64, 30)).astype(np.float32)
    with torch.no_grad():
        got = net(torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(got, keras_forward(arrays, observations), rtol=1e-5, atol=1e-5)


def test_the_conversion_would_notice_a_missing_transpose():
    """A fixture whose subject cannot violate it is not a fixture.

    On a square layer a wrong orientation raises nothing and changes the function, so this proves
    the comparison above is capable of failing — by transposing the kernels and asserting it does.
    """
    rng = np.random.default_rng(8)
    arrays = tf_style_arrays(rng, [16, 16, 16])
    state, fc, obs_len, num_actions = importer.state_dict_from(arrays)
    for key in list(state):
        state[key] = state[key].T.contiguous() if state[key].ndim == 2 else state[key]

    net = network.QNet(obs_len, fc, num_actions)
    net.load_state_dict(state, strict=True)
    observations = rng.random((8, 16)).astype(np.float32)
    with torch.no_grad():
        got = net(torch.from_numpy(observations)).numpy()
    assert not np.allclose(got, keras_forward(arrays, observations), rtol=1e-3, atol=1e-3)


def test_a_deeper_network_converts_layer_by_layer_in_order():
    # Distinct widths, so a transposition of two *layers* — as opposed to of one kernel — cannot
    # compose and would raise. The point is that the order is checked, not assumed.
    rng = np.random.default_rng(9)
    arrays = tf_style_arrays(rng, [30, 100, 50, 3])
    state, fc_layer_params, obs_len, num_actions = importer.state_dict_from(arrays)
    assert fc_layer_params == [100, 50]
    net = network.QNet(obs_len, fc_layer_params, num_actions)
    net.load_state_dict(state, strict=True)
    observations = rng.random((16, 30)).astype(np.float32)
    with torch.no_grad():
        got = net(torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(got, keras_forward(arrays, observations), rtol=1e-4, atol=1e-4)


def test_layers_that_do_not_compose_are_refused():
    arrays = {'layers': np.asarray(2),
              'kernel0': np.zeros((30, 320), dtype=np.float32),
              'bias0': np.zeros(320, dtype=np.float32),
              'kernel1': np.zeros((64, 3), dtype=np.float32),
              'bias1': np.zeros(3, dtype=np.float32)}
    with pytest.raises(SystemExit):
        importer.state_dict_from(arrays)
