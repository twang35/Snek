"""The `rainbow` name of group C -- Rainbow (Hessel et al. 2018), row C1 -- as one entry in `train.ALGOS`.
Everything is `algos/rainbow/algo.py`'s; this module fixes the name and so the paper defaults."""

from algos.rainbow import algo as rainbow_algo

NAME = 'rainbow'


def build_config(tuned):
    return rainbow_algo.build_config(tuned, NAME)


def reportable(config):
    return rainbow_algo.reportable(config)


def arch_fields(config):
    return rainbow_algo.arch_fields(config)


def build(config, arch, device='cpu'):
    return rainbow_algo.build(config, arch, device=device, name=NAME)
