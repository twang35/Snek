"""The seam for the four distributional rungs: `DqnAlgo` with a `DistAgent` and a `head` in the sidecar.

One class, parameterised by the rung, because the rungs share every seam answer with DQN -- the step
is one `collector.step()`, the replay is prioritised (or not, by knob), the schedules are DQN's --
and differ only in the agent they build and the head they record. The four modules `c51.py`,
`qrdqn.py`, `iqn.py`, `fqf.py` each register one `NAME` for `train.ALGOS` and forward to this.

Knobs, all `SNEK_DIST_*` and each its config key lowercased, over DQN's own (which keep their names):

| knob | rungs | default | paper |
|---|---|---|---|
| `SNEK_DIST_ATOMS`, `SNEK_DIST_V_MIN`, `SNEK_DIST_V_MAX` | c51 | 51, -10, 110 | 51 atoms; the support is this game's return range, `a-return-tail.md` §1 |
| `SNEK_DIST_QUANTILES` | qrdqn, fqf | 200 (qrdqn), 32 (fqf) | N |
| `SNEK_DIST_TAU_SAMPLES`, `SNEK_DIST_TAU_PRIME_SAMPLES`, `SNEK_DIST_POLICY_SAMPLES` | iqn | 64, 64, 32 | N, N', K |
| `SNEK_DIST_EMBEDDING` | iqn, fqf | 64 | the cosine embedding's width |
| `SNEK_DIST_KAPPA` | qrdqn, iqn, fqf | 1.0 | the Huber threshold |
| `SNEK_DIST_FRACTION_LR`, `SNEK_DIST_FRACTION_ENTROPY` | fqf | 2.5e-9, 0.001 | the proposal net's RMSProp rate and entropy bonus |
| `SNEK_DIST_RISK_ALPHA`, `SNEK_DIST_RISK_TRAIN` | iqn, fqf (and the CVaR read of every rung) | 1.0, 0 | the CVaR level; 1 trains the paper's risk-sensitive agent (acting and target argmax under the distortion) |
"""

from algos.dist import net as network
from algos.dist.agent import DistAgent
from algos.dqn import algo as dqn_algo

RUNGS = ('c51', 'qrdqn', 'iqn', 'fqf')
HEAD_TYPE = {'c51': 'c51', 'qrdqn': 'quantile', 'iqn': 'iqn', 'fqf': 'fqf'}


def build_config(tuned, rung):
    """DQN's knobs plus the rung's. Every key is its `SNEK_` variable lowercased."""
    config = dqn_algo.build_config(tuned)
    config.update({
        'dist_atoms': int(tuned('DIST_ATOMS', 51, int)),
        'dist_v_min': tuned('DIST_V_MIN', -10.0),
        'dist_v_max': tuned('DIST_V_MAX', 110.0),
        'dist_quantiles': int(tuned('DIST_QUANTILES', 200 if rung == 'qrdqn' else 32, int)),
        'dist_tau_samples': int(tuned('DIST_TAU_SAMPLES', 64, int)),
        'dist_tau_prime_samples': int(tuned('DIST_TAU_PRIME_SAMPLES', 64, int)),
        'dist_policy_samples': int(tuned('DIST_POLICY_SAMPLES', 32, int)),
        'dist_embedding': int(tuned('DIST_EMBEDDING', 64, int)),
        'dist_kappa': tuned('DIST_KAPPA', 1.0),
        'dist_fraction_lr': tuned('DIST_FRACTION_LR', 2.5e-9),
        'dist_fraction_entropy': tuned('DIST_FRACTION_ENTROPY', 0.001),
        'dist_risk_alpha': tuned('DIST_RISK_ALPHA', 1.0),
        'dist_risk_train': bool(int(tuned('DIST_RISK_TRAIN', 0, int))),
    })
    if not 0.0 < config['dist_risk_alpha'] <= 1.0:
        raise ValueError('SNEK_DIST_RISK_ALPHA must be in (0, 1], got {0}'.format(config['dist_risk_alpha']))
    if rung == 'c51' and config['dist_v_max'] <= config['dist_v_min']:
        raise ValueError('SNEK_DIST_V_MAX must exceed SNEK_DIST_V_MIN')
    return config


def arch_fields(config, rung):
    """The `head` the sidecar records for this rung, from the config. `train.py` calls this hook."""
    kind = HEAD_TYPE[rung]
    if kind == 'c51':
        head = {'type': 'c51', 'atoms': config['dist_atoms'], 'v_min': float(config['dist_v_min']),
                'v_max': float(config['dist_v_max'])}
    elif kind == 'quantile':
        head = {'type': 'quantile', 'n': config['dist_quantiles']}
    elif kind == 'iqn':
        head = {'type': 'iqn', 'embedding': config['dist_embedding'],
                'n_tau': config['dist_tau_samples'], 'k': config['dist_policy_samples']}
    else:
        head = {'type': 'fqf', 'embedding': config['dist_embedding'], 'n': config['dist_quantiles']}
    return {'head': head}


def reportable(config):
    return dqn_algo.reportable(config)


class DistAlgo(dqn_algo.DqnAlgo):
    """`DqnAlgo` whose agent is a `DistAgent`. The parent builds a `DdqnAgent` first; it is replaced
    before the collector is built, so the collector holds the right one."""

    def __init__(self, config, arch, device='cpu', rung='c51'):
        self.rung = rung
        super().__init__(config, arch, device=device)
        if network.head_of(arch)['type'] != HEAD_TYPE[rung]:
            raise ValueError('arch head {0!r} is not {1}\'s ({2})'.format(
                arch['head'], rung, HEAD_TYPE[rung]))
        self.agent = DistAgent(arch,
                               learning_rate=config['learning_rate'],
                               adam_epsilon=config['adam_epsilon'],
                               target_update_period=config['target_update_period'],
                               target_update_tau=config['target_update_tau'],
                               gradient_clipping=config['gradient_clipping'],
                               use_is_weights=config['is_weights'],
                               seed=config['seed'], device=device,
                               munchausen_alpha=config['munchausen_alpha'],
                               munchausen_tau=config['munchausen_tau'],
                               munchausen_l0=config['munchausen_l0'],
                               kappa=config['dist_kappa'],
                               n_tau=config['dist_tau_samples'],
                               n_tau_prime=config['dist_tau_prime_samples'],
                               fraction_lr=config['dist_fraction_lr'],
                               fraction_entropy=config['dist_fraction_entropy'],
                               risk_alpha=config['dist_risk_alpha'],
                               risk_train=config['dist_risk_train'])
        self.collector.agent = self.agent

    def describe(self):
        head = self.arch['head']
        return '{0}, {1} head {2}'.format(super().describe(), self.rung,
                                          ' '.join('{0}={1}'.format(k, v) for k, v in sorted(head.items())
                                                   if k != 'type'))


def build(config, arch, device='cpu', rung='c51'):
    return DistAlgo(config, arch, device=device, rung=rung)
