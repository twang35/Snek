"""Fixtures for `hyperparamTuning/perDiagnostics/plasticity.py`.

Every metric here is a single number that will be read as evidence about a training run, and each one
has a plausible-looking wrong version: a dormancy score that tracks Q magnitude instead of dormancy, a
rank that sits at 1.1 whatever the features do, a weight-norm ratio measured against one constant so
the Q head reads 10x below its own initialisation forever. None of those fail loudly. So each is
pinned against a case whose answer is known independently of the implementation.

The layer-walk test is the one that protects the rest: if `layer_stack` returned the Q head as a
hidden layer, dormancy would be computed on three linear outputs and the feature rank would be the
rank of a 3-wide matrix — both would still produce numbers.

**Ten wrong versions were written and each one confirmed to fail a named test here.** Two fixtures
asserted nothing on the first pass, both on `dormant_fraction`, which is the metric the whole
investigation turns on:

- the scale-free fixture was an all-healthy layer, which reads 0.0 dormant under a layer-relative
  score *and* an absolute one, at every scale — so it passed with the normalisation deleted. It now
  straddles tau in absolute terms and pins the value, not just its invariance.
- the boundary fixture used tau = 0.025, which is not dyadic: the boundary unit's ratio lands 3.5e-18
  below tau, so `<` and `<=` agree and the comparison direction was never pinned. It now uses 2^-5.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'hyperparamTuning', 'perDiagnostics'))

import numpy as np
import tensorflow as tf

import plasticity
from under_the_hood import build_q_net


def built_net(fc=(50, 100, 50), obs_len=30):
    net = build_q_net(3, fc)
    net(tf.zeros([2, obs_len], tf.float32), step_type=tf.fill([2], 1))
    return net


def test_layer_stack_separates_the_q_head_from_the_hidden_layers():
    net = built_net((50, 100, 50))
    hidden, head = plasticity.layer_stack(net)
    assert [layer.units for layer in hidden] == [50, 100, 50]
    assert head.units == 3
    # The head must be linear and the hidden layers relu — the whole reason they are treated
    # differently by dormancy, rank and the weight-norm baseline.
    assert head.activation.__name__ == 'linear'
    assert all(layer.activation.__name__ == 'relu' for layer in hidden)


def test_layer_stack_handles_a_single_hidden_layer():
    # `fc 320` is one hidden layer, and it is the shape batch 24's record came from. An
    # off-by-one in the split would leave no hidden layers at all here.
    hidden, head = plasticity.layer_stack(built_net((320,)))
    assert [layer.units for layer in hidden] == [320]
    assert head.units == 3


def test_dormant_fraction_is_scale_free():
    """The property that makes the score mean anything: activations grow ~2x over training.

    The fixture has to straddle tau in *absolute* terms or it asserts nothing — an all-healthy layer
    reads 0.0 dormant under a layer-relative score and under an absolute one alike, at every scale.
    So the units here sit near 0.025 itself: an absolute threshold calls all four dormant at scale 1
    and none of them at scale 1000, while the real score reports the same one quiet unit at both.
    """
    acts = np.zeros((40, 4))
    acts[:, :3] = 0.02                       # healthy: 1.33x the layer mean, but below tau absolutely
    acts[:, 3] = 0.0001                      # 0.7% of the layer mean: dormant at any scale
    before = plasticity.dormant_fraction(acts)
    after = plasticity.dormant_fraction(acts * 1000.0)
    assert before == after, (before, after)
    assert before == (0.25, 0.0), before     # pinned, so "unchanged" cannot mean "always zero"


def test_dormant_fraction_counts_the_quiet_and_the_dead_separately():
    acts = np.zeros((100, 4))
    acts[:, 0] = 1.0                     # healthy
    acts[:, 1] = 1.0                     # healthy
    acts[:, 2] = 0.001                   # quiet but never zero: dormant, not dead
    acts[:, 3] = 0.0                     # never fires: dormant AND dead
    dormant, dead = plasticity.dormant_fraction(acts, tau=0.025)
    assert dormant == 0.5, dormant       # units 2 and 3, both under 2.5% of the layer mean
    assert dead == 0.25, dead            # only unit 3


def test_dormant_threshold_is_the_documented_boundary():
    """Pins the comparison direction, which a `<` would silently flip.

    **tau has to be a dyadic fraction here.** With the real 0.025 the boundary unit's ratio lands
    3.5e-18 *below* tau, so `<` and `<=` agree and the direction is not pinned at all. 0.03125 is
    2^-5, so `3a + tau = 5` is exact, the layer mean is exactly 1.0, and the ratio is exactly tau.
    """
    tau = 0.03125
    acts = np.zeros((10, 5))
    acts[:, :3] = (5.0 - tau) / 3.0
    acts[:, 3] = tau
    scores = np.abs(acts).mean(axis=0)
    assert scores.mean() == 1.0, scores
    assert scores[3] / scores.mean() == tau, 'the boundary must be exactly representable'
    dormant, _ = plasticity.dormant_fraction(acts, tau=tau)
    # unit 3 sits exactly at the threshold and unit 4 below it; units 0-2 are above
    assert dormant == 0.4, (dormant, scores)
    # and just under the threshold only the silent unit is dormant, so the boundary is the boundary
    assert plasticity.dormant_fraction(acts, tau=np.nextafter(tau, 0.0))[0] == 0.2


def test_a_fully_silent_layer_reads_as_entirely_dormant():
    dormant, dead = plasticity.dormant_fraction(np.zeros((50, 8)))
    assert (dormant, dead) == (1.0, 1.0)


def test_effective_rank_on_matrices_of_known_rank():
    # A rank-1 matrix: one singular value carries the whole spectrum.
    rank_one = np.outer(np.arange(1, 101, dtype=float), np.ones(10))
    srank, stable = plasticity.effective_rank(rank_one)
    assert srank == 1
    assert abs(stable - 1.0) < 1e-6
    # An orthogonal basis: every direction carries equal mass, so srank must reach full width and
    # stable rank must equal the width.
    identity = np.eye(12)
    srank, stable = plasticity.effective_rank(identity)
    assert srank == 12
    assert abs(stable - 12.0) < 1e-6


def test_effective_rank_of_an_all_zero_matrix_is_zero_not_a_crash():
    assert plasticity.effective_rank(np.zeros((20, 6))) == (0, 0.0)


def test_centring_is_what_makes_rank_sensitive_to_collapse():
    """The measured reason the centred ranks exist — the raw ones pin near 1 for ReLU features.

    A large common offset plus variation in exactly two orthogonal directions of equal magnitude, so
    both ranks have an exact known answer: 2. The raw spectrum is dominated by the mean vector and
    reports 1; only the centred version sees that the *variation* is two-dimensional.
    """
    t = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    left = np.tile([1.0, -1.0], 8) / 4.0        # two orthonormal directions over the 16 units
    right = np.concatenate([np.ones(8), -np.ones(8)]) / 4.0
    assert abs(left @ right) < 1e-12            # equal singular values need orthogonality
    varying = np.outer(np.sin(t), left) + np.outer(np.cos(t), right)
    feats = varying + 50.0                      # the DC component a ReLU layer carries
    raw_srank, raw_stable = plasticity.effective_rank(feats)
    cen_srank, cen_stable = plasticity.centred_rank(feats)
    assert raw_srank == 1, raw_srank            # raw sees one direction: the offset
    assert raw_stable < 1.01, raw_stable
    assert cen_srank == 2, cen_srank            # centred sees the real dimensionality
    assert abs(cen_stable - 2.0) < 1e-6, cen_stable


def test_constant_fraction_finds_units_that_say_nothing_however_loud():
    acts = np.zeros((200, 4))
    rng = np.random.default_rng(2)
    acts[:, 0] = rng.normal(size=200)
    acts[:, 1] = rng.normal(size=200)
    acts[:, 2] = 7.0                            # large but constant: no information
    acts[:, 3] = 0.0
    assert plasticity.constant_fraction(acts) == 0.5


def test_weight_growth_reads_one_on_a_fresh_network_for_every_layer():
    """The baseline the growth factor is measured against, checked rather than derived on paper.

    Hidden layers initialise with VarianceScaling(2, fan_in) and the head with RandomUniform(0.03),
    so a single constant baseline would put the head an order of magnitude below 1.0 forever and a
    real 30x head growth would read as 3x. Averaged over draws because one draw is noisy.
    """
    for fc in ((50, 100, 50), (320,), (100, 200, 100)):
        factors = {}
        for _ in range(5):
            hidden, head = plasticity.layer_stack(built_net(fc))
            for row in plasticity.weight_stats(hidden, head):
                factors.setdefault((row['layer'], row['is_head']), []).append(row['growth'])
        for (index, is_head), values in factors.items():
            mean = float(np.mean(values))
            assert 0.9 < mean < 1.1, (fc, index, is_head, mean)


def test_growth_factor_scales_with_the_weights():
    hidden, head = plasticity.layer_stack(built_net((50, 100, 50)))
    before = plasticity.weight_stats(hidden, head)
    for layer in list(hidden) + [head]:
        weights = layer.get_weights()
        weights[0] = weights[0] * 3.0
        layer.set_weights(weights)
    after = plasticity.weight_stats(hidden, head)
    for old, new in zip(before, after):
        assert abs(new['growth'] / old['growth'] - 3.0) < 1e-4


def test_hidden_kernels_excludes_the_head_and_matches_the_parameter_count():
    net = built_net((50, 100, 50))
    hidden, head = plasticity.layer_stack(net)
    flat = plasticity.hidden_kernels(hidden)
    assert flat.shape == (30 * 50 + 50 * 100 + 100 * 50,)
    # Moving only the head must leave the movement measure at exactly zero, or "the network stopped
    # changing" would be reported whenever the head happened to be quiet.
    weights = head.get_weights()
    weights[0] = weights[0] * 10.0
    head.set_weights(weights)
    assert np.array_equal(flat, plasticity.hidden_kernels(hidden))


def test_build_ladder_snaps_to_real_checkpoints_and_reports_the_rest(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        for step in (12000, 100000, 152000, 900000):
            open(os.path.join(directory, 'ckpt-%d.index' % step), 'w').close()
        ladder, skipped = plasticity.build_ladder(directory, 50000, [])
        # 0 has nothing within SNAP=5000 (the first checkpoint is at 12000), so it is skipped, not
        # silently answered with the 12000 checkpoint. Silent snapping is the failure that matters:
        # the analysis aligns each row against a step, so a row 38000 steps from where it claims to
        # be would put a metric on the wrong side of a peak.
        assert 0 in skipped
        assert 50000 in skipped                 # 12000 and 100000 are both further than SNAP
        assert 200000 in skipped                # 152000 is 48000 away
        # The 150000 rung is 2000 from the 152000 checkpoint, so it snaps.
        assert ladder == [100000, 152000, 900000], ladder
        assert all(os.path.exists(os.path.join(directory, 'ckpt-%d.index' % s)) for s in ladder)


class _FakeAgent(object):
    """`fresh_baseline` only ever touches `_q_network`, so the real agent is not needed."""

    def __init__(self, net):
        self._q_network = net


def test_the_fresh_control_is_reproducible_and_reports_a_standard_error():
    """The control is the reference every trained row is read against, so it must not wobble.

    Unseeded and at 5 draws it moved its own mean by 0.174 -> 0.143 dormant between two runs of the
    same arm, against a trained-vs-fresh difference of 0.08. Two calls must now agree exactly, and
    the reported `stderr` must be the sd over draws divided by sqrt(draws) — the printed row is the
    bar a departure has to clear, and the sd is ~4x looser than that.
    """
    rng = np.random.default_rng(3)
    obs = rng.random((64, 30)).astype(np.float32)
    fc = [16, 8]
    first = plasticity.fresh_baseline(_FakeAgent(built_net(tuple(fc))), obs, fc, draws=3)
    second = plasticity.fresh_baseline(_FakeAgent(built_net(tuple(fc))), obs, fc, draws=3)
    # To 1e-9, not bit-exact: TF's threaded matmul reorders its reduction, so the same weights give
    # the same activations to ~1e-16. That is eight orders of magnitude below the ~0.08 effect.
    for key in ('dormant_all', 'srank_c', 'stable_rank_c', 'growth_hidden', 'growth_head'):
        assert abs(first[key] - second[key]) <= 1e-9 * max(1.0, abs(first[key])), \
            (key, first[key], second[key])
    assert first['draws'] == 3 and first['seed'] == plasticity.FRESH_SEED
    # The redraw must still be the *real* initialisation, or the control is a different network from
    # the one training starts at: a mis-cloned initialiser would move these off 1.0 silently.
    assert 0.9 < first['growth_hidden'] < 1.1, first['growth_hidden']
    assert 0.9 < first['growth_head'] < 1.1, first['growth_head']
    for key, spread in first['spread'].items():
        assert abs(first['stderr'][key] - spread / np.sqrt(3)) < 1e-12, key
    # A different seed must give a different draw, or the seeding is not reaching the initialiser and
    # "reproducible" would be true for the uninteresting reason that nothing is random.
    other = plasticity.fresh_baseline(_FakeAgent(built_net(tuple(fc))), obs, fc, draws=3, seed=7)
    assert other['growth_hidden'] != first['growth_hidden']


def test_the_control_leaves_the_agents_own_network_in_place():
    # fresh_baseline swaps a random net onto the agent to measure it. If it failed to put the trained
    # one back, every row after the control would be measuring noise.
    net = built_net((16, 8))
    agent = _FakeAgent(net)
    rng = np.random.default_rng(4)
    plasticity.fresh_baseline(agent, rng.random((32, 30)).astype(np.float32), [16, 8], draws=2)
    assert agent._q_network is net


def test_parse_steps_round_trip():
    # Imported from input_sensitivity_over_time, so this pins the contract this script relies on.
    assert plasticity.parse_steps('1000,3000-5000:1000') == [1000, 3000, 4000, 5000]
