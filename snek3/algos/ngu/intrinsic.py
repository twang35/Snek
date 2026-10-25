"""NGU's intrinsic reward (Badia et al. 2020, §2 and Alg. 1; `plans/algoExploration/f-ngu-agent57.md` §0-2):

    r^i_t = r^episodic_t x clip(alpha_t, 1, L)

**The episodic term** counts how new the current board is *within this episode*: a k-nearest-neighbour kernel
over the embeddings of the boards seen so far, in a per-lane memory reset with the lane. Alg. 1 line for line
(`episodic_reward`): the k nearest squared distances, normalised by a running mean of all neighbour distances
seen this episode, clustered by `max(d - xi, 0)`, kernel `eps / (d + eps)`, similarity `sqrt(sum K + c)`, reward
`1 / s` or 0 when `s > s_max`.

**The lifelong term** is random network distillation: a fixed random network and a predictor trained to match it
on the boards the learner samples; the prediction error, normalised by its running mean and std, is
`alpha = 1 + (err - mu) / sigma`. Novel boards score high, boards the predictor has seen score ~1.

**The embedding** is the inverse-dynamics one: `f(board)` trained so a classifier on `(f(b_t), f(b_{t+1}))`
predicts `a_t`, which keeps f on what the agent controls. The modules live on the net (`net.py`) so the checkpoint
carries them; this file holds the per-lane state the collector and the restored policy both need -- the episodic
memories, the embedding snapshot each lane acts on (refreshed at its episode start, the paper's "embeddings target
update: once per episode") -- and the pure functions.
"""

import numpy as np
import torch


# ------------------------------------------------------------------ Alg. 1

def episodic_reward(dist_sq, mean_sq, k, kernel_eps, cluster, pseudo_count, max_similarity):
    """Alg. 1 on a batch of lanes. `dist_sq (n, k)` the k smallest squared distances (+inf where the memory
    holds fewer keys), `mean_sq (n,)` each lane's running mean of neighbour distances **including this query's**;
    -> `(n,)` rewards. A lane whose memory holds nothing gets 0 (no neighbours, nothing to compare)."""
    present = torch.isfinite(dist_sq)
    d = torch.where(present, dist_sq, torch.zeros_like(dist_sq))
    denom = mean_sq.unsqueeze(-1)
    # d^2_m of zero (every key identical) makes every normalised distance 0: the most-visited state.
    dn = torch.where(denom > 0.0, d / denom.clamp(min=1e-12), torch.zeros_like(d))
    dn = (dn - cluster).clamp(min=0.0)
    kernel = kernel_eps / (dn + kernel_eps)
    kernel = kernel * present.to(kernel.dtype)
    s = torch.sqrt(kernel.sum(dim=-1) + pseudo_count)
    reward = torch.where(s > max_similarity, torch.zeros_like(s), 1.0 / s)
    return torch.where(present.any(dim=-1), reward, torch.zeros_like(reward))


def lifelong_factor(error, mean, std, cap):
    """`alpha = 1 + (err - mu) / sigma`, clipped to [1, cap]; 1 where sigma is 0 (nothing normalised yet)."""
    alpha = torch.where(std > 0.0, 1.0 + (error - mean) / std.clamp(min=1e-12), torch.ones_like(error))
    return alpha.clamp(min=1.0, max=float(cap))


# ------------------------------------------------------------------ per-lane state

class EpisodicMemory(object):
    """One ring of up to `capacity` keys per lane, with the running mean of neighbour distances Alg. 1 keeps per
    episode. `reset(rows)` empties a lane; `query(rows, keys)` returns the k nearest squared distances to the
    lane's stored keys and updates the running mean; `insert(rows, keys)` stores.

    **The store grows with the longest live episode**, in steps of `GROW`, up to `capacity`, and **holds its keys in
    float16**: the eval engine runs 1,024 lanes, and 1,024 x 30,000 x 32 floats is 3.9 GB where 1,024 lanes x a Snake
    game's ~2,300 steps x 32 halves is ~150 MB -- the memory a stage-A worker or a stage-B shard adds at competence.
    Distances are computed in float32 from the stored halves, on the trainer's 32 lanes and a shard's 1,024 alike,
    so the two compute the same r^i; the quantisation (2^-11 relative) is far under the clustering distance ξ."""

    GROW = 256

    def __init__(self, lanes, capacity, dim, k, device='cpu'):
        self.lanes, self.capacity, self.dim, self.k = int(lanes), int(capacity), int(dim), int(k)
        self.device = device
        self.keys = torch.zeros((self.lanes, 0, self.dim), dtype=torch.float16, device=device)
        self.count = torch.zeros(self.lanes, dtype=torch.int64, device=device)
        self.write = torch.zeros(self.lanes, dtype=torch.int64, device=device)
        self.dist_sum = torch.zeros(self.lanes, dtype=torch.float32, device=device)
        self.dist_n = torch.zeros(self.lanes, dtype=torch.float32, device=device)

    @property
    def allocated(self):
        return int(self.keys.shape[1])

    def _ensure(self, slots):
        """Room for slot index `slots - 1` on every lane."""
        want = min(self.capacity, int(np.ceil(slots / float(self.GROW))) * self.GROW)
        if want > self.allocated:
            extra = torch.zeros((self.lanes, want - self.allocated, self.dim), dtype=torch.float16, device=self.device)
            self.keys = torch.cat([self.keys, extra], dim=1)

    def reset(self, rows):
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        self.count[rows] = 0
        self.write[rows] = 0
        self.dist_sum[rows] = 0.0
        self.dist_n[rows] = 0.0

    def sizes(self):
        return self.count.cpu().numpy()

    def query(self, rows, keys):
        """`(k nearest squared distances (m, k) with +inf padding, running mean (m,))` for `keys (m, dim)` against the
        lanes in `rows`. The running mean is updated with this query's distances first, as Alg. 1 line 6 does."""
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        m = rows.shape[0]
        count = self.count[rows]
        span = int(count.max()) if m else 0
        if span == 0:
            inf = torch.full((m, self.k), float('inf'), device=self.device)
            return inf, torch.zeros(m, device=self.device)
        stored = self.keys[rows, :span].to(torch.float32)                      # (m, span, dim)
        query = keys.to(torch.float16).to(torch.float32)                         # as it will be stored
        d = ((stored - query.unsqueeze(1)) ** 2).sum(dim=-1)                   # (m, span)
        slot = torch.arange(span, device=self.device).unsqueeze(0)
        d = torch.where(slot < count.unsqueeze(1), d, torch.full_like(d, float('inf')))
        kk = min(self.k, span)
        nearest = torch.topk(d, kk, dim=1, largest=False).values
        if kk < self.k:
            nearest = torch.cat([nearest, torch.full((m, self.k - kk), float('inf'), device=self.device)], dim=1)
        present = torch.isfinite(nearest)
        self.dist_sum[rows] += torch.where(present, nearest, torch.zeros_like(nearest)).sum(dim=1)
        self.dist_n[rows] += present.sum(dim=1).to(torch.float32)
        n = self.dist_n[rows]
        mean = torch.where(n > 0.0, self.dist_sum[rows] / n.clamp(min=1.0), torch.zeros_like(n))
        return nearest, mean

    def insert(self, rows, keys):
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        slot = self.write[rows]
        self._ensure(int(slot.max()) + 1)
        self.keys[rows, slot] = keys.detach().to(torch.float16)
        self.write[rows] = (slot + 1) % self.capacity
        self.count[rows] = torch.clamp(self.count[rows] + 1, max=self.capacity)

    def state_dict(self):
        return {'keys': self.keys.cpu(), 'count': self.count.cpu(), 'write': self.write.cpu(),
                'dist_sum': self.dist_sum.cpu(), 'dist_n': self.dist_n.cpu()}

    def load_state_dict(self, state):
        self.keys = state['keys'].to(self.device).clone()
        for name in ('count', 'write', 'dist_sum', 'dist_n'):
            getattr(self, name).copy_(state[name].to(self.device))


class LaneEmbeddings(object):
    """Each lane's own copy of the embedding MLP `f` (two linear layers, ReLU between), refreshed from the live
    module at the lane's episode start and fixed through the episode, so a lane's stored keys all come from one
    f (NGU Table 6: the actors' embedding is updated once per episode). Batched weights, applied with `bmm`."""

    def __init__(self, lanes, first, second, device='cpu'):
        self.lanes = int(lanes)
        self.device = device
        self.w1 = torch.zeros((self.lanes,) + tuple(first.weight.shape), device=device)
        self.b1 = torch.zeros((self.lanes,) + tuple(first.bias.shape), device=device)
        self.w2 = torch.zeros((self.lanes,) + tuple(second.weight.shape), device=device)
        self.b2 = torch.zeros((self.lanes,) + tuple(second.bias.shape), device=device)
        self.refresh(np.arange(self.lanes), first, second)

    def refresh(self, rows, first, second):
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        if rows.numel() == 0:
            return
        with torch.no_grad():
            self.w1[rows] = first.weight.detach()
            self.b1[rows] = first.bias.detach()
            self.w2[rows] = second.weight.detach()
            self.b2[rows] = second.bias.detach()

    def embed(self, rows, inputs):
        """`inputs (m, in)` through lane `rows[i]`'s snapshot -> `(m, dim)`."""
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)
        with torch.no_grad():
            h = torch.relu(torch.bmm(inputs.unsqueeze(1), self.w1[rows].transpose(1, 2)).squeeze(1) + self.b1[rows])
            return torch.bmm(h.unsqueeze(1), self.w2[rows].transpose(1, 2)).squeeze(1) + self.b2[rows]

    def state_dict(self):
        return {'w1': self.w1.cpu(), 'b1': self.b1.cpu(), 'w2': self.w2.cpu(), 'b2': self.b2.cpu()}

    def load_state_dict(self, state):
        for name in ('w1', 'b1', 'w2', 'b2'):
            getattr(self, name).copy_(state[name].to(self.device))


class LaneNovelty(object):
    """The per-lane half of the intrinsic reward, for the collector and the restored policy alike: the memories,
    the embedding each lane acts on, and the step that turns a board into `r^i`. `net` is the `NguNet` (its
    `embed_first`, `embed_second`, `rnd_target`, `rnd_predictor`, the RND statistics and the constants).

    `training=True` is the collector's mode: each lane acts on a **snapshot** of f taken at its episode start (the
    paper's once-per-episode embedding update) and every query feeds the RND error's running normalisation. An eval
    (`training=False`) reads the checkpoint's f directly -- nothing trains, so the snapshot would equal it -- and
    leaves the statistics as the checkpoint saved them. `grow(lanes)` adds lanes, for a policy whose rows arrive
    as the engine assigns them."""

    def __init__(self, net, lanes, device='cpu', training=False):
        self.net = net
        self.device = device
        self.training = bool(training)
        c = net.intrinsic
        self.memory = EpisodicMemory(lanes, c['memory'], c['embed_dim'], c['knn_k'], device=device)
        self.snapshots = LaneEmbeddings(lanes, net.embed_first, net.embed_second, device=device) if self.training else None
        self.last = {}

    @property
    def lanes(self):
        return self.memory.lanes

    def grow(self, lanes):
        if lanes <= self.lanes:
            return
        if self.training:
            raise ValueError('the collector\'s lane count is fixed')
        old, c = self.memory, self.net.intrinsic
        new = EpisodicMemory(lanes, c['memory'], c['embed_dim'], c['knn_k'], device=self.device)
        new._ensure(old.allocated)
        new.keys[:old.lanes, :old.allocated] = old.keys
        for name in ('count', 'write', 'dist_sum', 'dist_n'):
            getattr(new, name)[:old.lanes] = getattr(old, name)
        self.memory = new

    def _embed(self, rows, inputs):
        if self.training:
            return self.snapshots.embed(rows, inputs)
        with torch.no_grad():
            return self.net.embed(inputs)

    def begin_episode(self, rows, inputs):
        """A lane's first board: empty its memory, take its snapshot of f, store the board's key. No reward --
        `r^i_{-1}` is 0 by the paper's convention."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            return
        self.memory.reset(rows)
        if self.training:
            self.snapshots.refresh(rows, self.net.embed_first, self.net.embed_second)
        self.memory.insert(rows, self._embed(rows, inputs))

    def reward(self, rows, inputs):
        """`r^i` for arriving at `inputs` on lanes `rows`, each key then stored. `(m,) float32` numpy."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            return np.zeros(0, dtype=np.float32)
        c = self.net.intrinsic
        with torch.no_grad():
            keys = self._embed(rows, inputs)
            nearest, mean = self.memory.query(rows, keys)
            episodic = episodic_reward(nearest, mean, c['knn_k'], c['kernel_epsilon'], c['cluster_distance'],
                                       c['pseudo_count'], c['max_similarity'])
            self.memory.insert(rows, keys)
            error = self.net.rnd_error(inputs)
            if self.training:
                self.net.update_rnd_stats(error)
            alpha = lifelong_factor(error, self.net.rnd_mean, self.net.rnd_std(), c['lifelong_max'])
            reward = episodic * alpha
        self.last = {'episodic': float(episodic.mean()), 'alpha': float(alpha.mean())}
        return reward.cpu().numpy().astype(np.float32)

    def state_dict(self):
        out = {'memory': self.memory.state_dict()}
        if self.training:
            out['snapshots'] = self.snapshots.state_dict()
        return out

    def load_state_dict(self, state):
        self.memory.load_state_dict(state['memory'])
        if self.training and 'snapshots' in state:
            self.snapshots.load_state_dict(state['snapshots'])
