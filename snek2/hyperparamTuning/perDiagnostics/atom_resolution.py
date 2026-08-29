"""Is the atom spacing coarse relative to the decisions the policy actually has to make?

`c51_stability.py` answers whether the support's *range* is right (it is — boundary mass is ~0). This
answers the separate question of whether its *resolution* is, which is what decides two proposals that
look reasonable on paper: shrinking `PERFECT_GAME_REWARD` so the support can be narrower, and raising
`num_atoms`.

The greedy action turns on the gap between the top two expected Q values. Where that gap is smaller than
one atom spacing, the grid cannot represent the distinction the argmax depends on. Where it is many atoms
wide, resolution is not the binding constraint and changing it buys nothing.

**Report the median and the length breakdown, never the mean.** The gap distribution here is violently
bimodal — median 0.28 reward units against a 75th percentile of 42.7 — so the mean lands in neither mode
and reads as a comfortable middle that does not exist. A mean-based reading of this quantity is what
produced the claim in `findings.md` that "the actions are not near-tied", which the median reverses.

**The length breakdown is the part that settles anything.** On the shipped config (51 atoms over
`[-5, 120]`, spacing 2.5, `FOOD_REWARD` 1.0 = 0.40 atoms) 59-67% of states have a sub-atom gap, but they
are concentrated in early open-board play where several moves genuinely are interchangeable:

    length  1-49   median 0.11 atoms   79.0% under one atom
    length 50-84   median 0.40 atoms   57.0%
    length 85-94   median 25.1 atoms   17.7%

So the endgame this project has established decides games is resolved ~25x over, and a flip in the lower
mode is the grid correctly reporting that the choice does not matter. That is the measurement that makes
shrinking the win reward a bad trade: the 62.8-unit endgame gap *is* the +100 being in or out of reach, so
a smaller win compresses exactly the gaps that are currently well resolved.

Reads any arm through `eval_agent.build_eval_agent`, so the support comes from the policy's own
`arch.json`. Writes nothing. A scalar arm has no atom grid and is skipped rather than guessed at.

    PYTHONPATH=. python hyperparamTuning/perDiagnostics/atom_resolution.py <policy> [<policy> ...]
"""
import os, sys
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import numpy as np, tensorflow as tf
from tf_agents.environments import tf_py_environment
sys.path.insert(0,'.')
import policy_arch, under_the_hood
from eval_agent import build_eval_agent
from snake_environment import SnakeEnvironment
from snake_constants import FOOD_REWARD

for policy in sys.argv[1:]:
    d = os.path.join('savedPolicies', policy)
    steps = sorted(int(f[5:].split('.')[0]) for f in os.listdir(d)
                   if f.startswith('ckpt-') and f.endswith('.index'))
    env = SnakeEnvironment(discount=0.99, display=False, policy_name=policy)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    agent, ckpt, gs = build_eval_agent(tf_env, env, d)
    arch = policy_arch.read_arch(d); support = policy_arch.support_from_arch(arch)
    if support is None:
        print('%s: scalar arm, no atom grid to measure' % policy)
        continue
    spacing = float(support[1] - support[0])
    net = agent._q_network
    ckpt.restore(os.path.join(d, 'ckpt-%d' % steps[-1])).expect_partial()

    gaps, lens = [], []
    ts = env.reset()
    while len(gaps) < 3000:
        if ts.is_last(): ts = env.reset()
        o = np.asarray(ts.observation, np.float32)
        q = np.asarray(under_the_hood.expected_q(net, tf.constant(o[None]), support=support))[0]
        srt = np.sort(q)
        gaps.append(srt[-1] - srt[-2])
        lens.append(len(env._game.snapshot().body))
        ts = env.step(np.int32(np.argmax(q)))
    g = np.array(gaps); L = np.array(lens)
    print('%s   spacing %.2f   food reward %.2f (= %.2f atoms)' % (policy, spacing, FOOD_REWARD,
                                                                  FOOD_REWARD/spacing))
    pct = [1,5,10,25,50,75,90]
    print('   action-gap percentiles (reward units): ' + '  '.join(
        '%d%%=%.2f' % (p, np.percentile(g,p)) for p in pct))
    print('   in atoms:                             ' + '  '.join(
        '%d%%=%.2f' % (p, np.percentile(g,p)/spacing) for p in pct))
    print('   share of states with gap < 1 atom: %.1f%%   < 0.5 atom: %.1f%%' % (
        100*np.mean(g < spacing), 100*np.mean(g < spacing/2)))
    for lo,hi in [(1,49),(50,84),(85,94),(95,99)]:
        m = (L>=lo)&(L<=hi)
        if m.sum() > 30:
            print('     length %2d-%2d  n=%-5d median gap %.2f (%.2f atoms)  share <1 atom %.1f%%' % (
                lo,hi,m.sum(), np.median(g[m]), np.median(g[m])/spacing, 100*np.mean(g[m] < spacing)))
