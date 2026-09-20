"""Discrete soft actor-critic (Group B of the algorithm series): `sac`, Christodoulou 2019, and `sac2`,
the same agent with Zhou et al. 2022's three fixes as its defaults. One package because B2 *is* B1 plus
fixes, and the fixes are knobs (`plans/algoExploration/b-entropy.md`).

`net.py` builds the actor `tools/restore.py` loads (DQN's `QNet`, read as logits, exactly as PPO's) and the
two critics; `agent.py` is the soft target, the twin-critic loss, the actor loss and the temperature;
`algo.py` is the seam `train.py` drives, on DQN's replay and collector with the fork and the shield off.
"""
