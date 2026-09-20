"""`sac2`: discrete SAC with Zhou et al. 2022's fixes as its defaults (`algos/sac/algo.py`, `DEFAULTS`)."""

from algos.sac import algo as sac_algo

NAME = 'sac2'


def build_config(tuned):
    return sac_algo.build_config(tuned, NAME)


reportable = sac_algo.reportable
build = sac_algo.build
