"""The `c51` rung of group A, as one entry in `train.ALGOS`. Everything is `algos/dist/algo.py`'s."""

from algos.dist import algo as dist_algo

NAME = 'c51'


def build_config(tuned):
    return dist_algo.build_config(tuned, NAME)


def reportable(config):
    return dist_algo.reportable(config)


def arch_fields(config):
    return dist_algo.arch_fields(config, NAME)


def build(config, arch, device='cpu'):
    return dist_algo.build(config, arch, device=device, rung=NAME)
