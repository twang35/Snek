"""Feature audit for HOF checkpoints: occupancy, saliency, flip sensitivity, real-play ablation."""
import json, os, sys, time
import multiprocessing as mp
import numpy as np

CKPTS = {
    'b10ck': ('hallOfFame/b10ck-g100-seed3-ckpt30523392', 30523392),
    'b17cl': ('hallOfFame/b17cl-clipanneal001hold80-seed4-ckpt11386880', 11386880),
    'b9ch':  ('hallOfFame/b9ch-lam999-seed4-ckpt47251456', 47251456),
}
NAMES = {0:'food_closer_L',1:'food_dist_L',2:'food_closer_R',3:'food_dist_R',4:'food_closer_F',5:'food_dist_F',
 6:'safe_L',7:'safe_R',8:'safe_F',9:'tail_reach_L',10:'lg_groups_L',11:'tail_reach_R',12:'lg_groups_R',
 13:'tail_reach_F',14:'lg_groups_F',15:'chase_safe_L',16:'chase_safe_R',17:'chase_safe_F',
 18:'wins_L',19:'wins_R',20:'wins_F',21:'starve_left',22:'board_fill',23:'hug_L',24:'hug_R',25:'hug_F',
 26:'not_tail_L',27:'not_tail_R',28:'not_tail_F',29:'food_room'}
GROUPS = {
 'food(0-5)': list(range(0,6)), 'food_closer(0,2,4)': [0,2,4], 'food_dist(1,3,5)': [1,3,5],
 'safe(6-8)': [6,7,8], 'tail_reach(9,11,13)': [9,11,13], 'lg_groups(10,12,14)': [10,12,14],
 'chase_safe(15-17)': [15,16,17], 'wins(18-20)': [18,19,20], 'starve(21)': [21], 'fill(22)': [22],
 'hug(23-25)': [23,24,25], 'not_tail(26-28)': [26,27,28], 'food_room(29)': [29],
}
PHASES = [('early <50', 0.0, 0.5), ('mid 50-80', 0.5, 0.8), ('late 80-95', 0.8, 0.95), ('endgame >=95', 0.95, 1.01)]
OUT = os.path.dirname(os.path.abspath(__file__))


def load(key):
    import torch
    from tools import restore
    d, step = CKPTS[key]
    policy_fn, arch, _ = restore.restore(d, step)
    net = restore.build_net(arch)
    from tools import checkpoints
    checkpoints.load(checkpoints.path(d, step), net)
    net.eval()
    return net, policy_fn


def collect(key, episodes):
    from vectorized import engine
    net, policy_fn = load(key)
    obs_buf, act_buf = [], []
    def rec(obs):
        a = policy_fn(obs)
        obs_buf.append(obs.copy()); act_buf.append(a.copy())
        return a
    held = engine.measure(rec, episodes, seed=0)
    obs = np.concatenate(obs_buf); act = np.concatenate(act_buf)
    return net, obs, act, held


def analyse(key, episodes=400):
    import torch
    net, obs, act, held = collect(key, episodes)
    n = len(obs)
    res = {'ckpt': key, 'states': int(n), 'episodes': episodes,
           'perfect_pct': 100.0*float(np.mean(held['perfect']))}
    fill = obs[:, 22]
    X = torch.tensor(obs).requires_grad_(True)
    logits = net(X)
    # saliency of the decision margin (chosen minus runner-up)
    top2 = logits.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).sum()
    margin.backward()
    grad = X.grad.abs().numpy()
    with torch.no_grad():
        base_act = logits.argmax(1).numpy()
    per_feature = {}
    for i in range(30):
        f = {'name': NAMES[i], 'mean': float(obs[:, i].mean()), 'std': float(obs[:, i].std()),
             'nonzero_pct': 100.0*float((obs[:, i] != 0).mean()),
             'mode_pct': 100.0*float((obs[:, i] == np.round(obs[:, i].mean())).mean()) }
        f['saliency'] = float(grad[:, i].mean())
        f['saliency_share'] = float(grad[:, i].sum() / grad.sum())
        f['phase'] = {}
        for pname, lo, hi in PHASES:
            m = (fill >= lo) & (fill < hi)
            if m.sum() == 0: continue
            f['phase'][pname] = {'n': int(m.sum()), 'std': float(obs[m, i].std()),
                                  'nonzero_pct': 100.0*float((obs[m, i] != 0).mean()),
                                  'saliency_share': float(grad[m, i].sum() / grad[m].sum())}
        per_feature[i] = f
    # flip sensitivity: replace feature by its shuffled value (within all states) -> fraction of argmax changes
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(30):
            o2 = obs.copy(); o2[:, i] = o2[rng.permutation(n), i]
            a2 = net(torch.as_tensor(o2)).argmax(1).numpy()
            changed = a2 != base_act
            per_feature[i]['shuffle_flip_pct'] = 100.0*float(changed.mean())
            per_feature[i]['shuffle_flip_phase'] = {}
            for pname, lo, hi in PHASES:
                m = (fill >= lo) & (fill < hi)
                if m.sum(): per_feature[i]['shuffle_flip_phase'][pname] = 100.0*float(changed[m].mean())
            # mean-substitution flips
            o3 = obs.copy(); o3[:, i] = obs[:, i].mean()
            a3 = net(torch.as_tensor(o3)).argmax(1).numpy()
            per_feature[i]['meansub_flip_pct'] = 100.0*float((a3 != base_act).mean())
        # win-flag probe: at states with length==99 (fill 0.99), what does forcing wins flag do?
        m99 = np.isclose(fill, 0.99)
        res['n_fill99'] = int(m99.sum())
        if m99.sum():
            o = obs[m99]
            base = net(torch.as_tensor(o))
            deltas = []
            for a in range(3):
                o4 = o.copy(); o4[:, 18+a] = 1.0
                d = (net(torch.as_tensor(o4)) - base)[:, a].numpy()
                deltas.append(float(d.mean()))
            res['force_win_flag_logit_delta'] = deltas
            # margin at those states
            t2 = base.topk(2, dim=1).values
            res['margin_at_99'] = float((t2[:, 0]-t2[:, 1]).mean())
    # first-layer weight norms
    W = net.hidden[0].weight.detach().numpy()
    res['w1_l2_by_input'] = [float(x) for x in np.linalg.norm(W, axis=0)]
    res['features'] = per_feature
    res['means'] = [float(x) for x in obs.mean(0)]
    with open(os.path.join(OUT, 'audit_{0}.json'.format(key)), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('analysed', key, n, 'states', res['perfect_pct'], flush=True)
    return res


def ablate_task(args):
    key, label, indices, mode, means, episodes, seed = args
    import torch
    torch.set_num_threads(1)
    from vectorized import engine
    net, policy_fn = load(key)
    idx = np.array(indices, dtype=int)
    sub = np.array([means[i] for i in indices], dtype=np.float32)
    def pf(obs):
        o = obs.copy()
        if mode == 'mean': o[:, idx] = sub
        elif mode == 'mean_late':
            late = o[:, 22] >= 0.8
            o[np.ix_(late, idx)] = sub
        elif mode == 'mean_early':
            early = o[:, 22] < 0.8
            o[np.ix_(early, idx)] = sub
        elif mode == 'zero': o[:, idx] = 0.0
        elif mode == 'one': o[:, idx] = 1.0
        return policy_fn(o)
    t = time.time()
    held = engine.measure(pf, episodes, seed=seed)
    p = 100.0*float(np.mean(held['perfect']))
    row = {'ckpt': key, 'label': label, 'indices': indices, 'mode': mode, 'episodes': episodes,
           'perfect_pct': p, 'avg_score': float(np.mean(held['scores'])), 'secs': time.time()-t}
    print(json.dumps(row), flush=True)
    return row


def main():
    phase = sys.argv[1]
    keys = sys.argv[2].split(',') if len(sys.argv) > 2 else list(CKPTS)
    if phase == 'analyse':
        for k in keys: analyse(k)
    elif phase == 'ablate_phase':
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        tasks = []
        for k in keys:
            means = json.load(open(os.path.join(OUT, 'audit_{0}.json'.format(k))))['means']
            for g, idx in GROUPS.items():
                for mode in ('mean_late', 'mean_early'):
                    tasks.append((k, g, idx, mode, means, episodes, 1))
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            rows = pool.map(ablate_task, tasks, chunksize=1)
        with open(os.path.join(OUT, 'ablation_phase_{0}.json'.format('_'.join(keys))), 'w') as fh:
            json.dump(rows, fh, indent=1)
    elif phase == 'ablate':
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        tasks = []
        for k in keys:
            means = json.load(open(os.path.join(OUT, 'audit_{0}.json'.format(k))))['means']
            tasks.append((k, 'baseline', [], 'mean', means, episodes, 1))
            for g, idx in GROUPS.items():
                tasks.append((k, g, idx, 'mean', means, episodes, 1))
                tasks.append((k, g, idx, 'zero', means, episodes, 1))
            for i in range(30):
                tasks.append((k, NAMES[i], [i], 'mean', means, episodes, 1))
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            rows = pool.map(ablate_task, tasks, chunksize=1)
        with open(os.path.join(OUT, 'ablation_{0}.json'.format('_'.join(keys))), 'w') as fh:
            json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()
