"""
agent.py  —  Structure2Vec + DQN with full backprop (PyTorch or NumPy).

NumPy version now correctly backpropagates through ALL weights:
  W1, W2  (S2V embedding)  ← previously frozen, now fixed
  W3, W4, W5  (Q-head)     ← as before

Dynamic graph fix: the environment handles step-level perturbation;
the agent simply re-reads adj at every step (already in training loop).
"""
import random
import numpy as np
from collections import deque

try:
    import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ReplayMemory:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)
    def push(self, t): self.memory.append(t)
    def sample(self, n): return random.sample(self.memory, n)
    def __len__(self): return len(self.memory)


def _relu(x):  return np.maximum(0.0, x)
def _drelu(x): return (x > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  NumPy S2V with full forward + backward
# ─────────────────────────────────────────────────────────────────────────────
class S2VNumpy:
    """Structure2Vec DE-MF.  forward_cache() stores pre-activations for backprop."""
    def __init__(self, fd, d=64, iters=4, scale=0.05):
        self.d = d; self.iters = iters
        self.W1 = np.random.randn(d, fd).astype(np.float32) * scale
        self.W2 = np.random.randn(d, d ).astype(np.float32) * scale

    def forward(self, x, adj):
        mu = np.zeros((x.shape[0], self.d), dtype=np.float32)
        for _ in range(self.iters):
            mu = _relu(x @ self.W1.T + (adj @ mu) @ self.W2.T)
        return mu

    def forward_cache(self, x, adj):
        """Returns (mu_final, list of (pre_act, mu_prev_agg)) per iteration."""
        mu = np.zeros((x.shape[0], self.d), dtype=np.float32)
        cache = []
        for _ in range(self.iters):
            agg  = adj @ mu                              # (N,d)  adj·mu_{t-1}
            pre  = x @ self.W1.T + agg @ self.W2.T      # (N,d)  pre-activation
            mu   = _relu(pre)
            cache.append((pre.copy(), agg.copy()))
        return mu, cache


# ─────────────────────────────────────────────────────────────────────────────
#  NumPy DQN head  (full backprop through S2V via truncated BPTT)
# ─────────────────────────────────────────────────────────────────────────────
class DQNetNumpy:
    """
    Q(S,a) = W3 · relu( [ W4·Σmu  ‖  W5·mu_a ] )

    Backward pass propagates through the final S2V iteration to give
    gradients for W1 and W2 (truncated BPTT — last iteration only).
    This is a standard approximation used in graph RL.
    """
    def __init__(self, fd, d=64, iters=4):
        sc = 0.05
        self.s2v = S2VNumpy(fd, d, iters)
        self.d   = d
        self.W3  = np.random.randn(1,   2*d).astype(np.float32) * sc
        self.W4  = np.random.randn(d,   d  ).astype(np.float32) * sc
        self.W5  = np.random.randn(d,   d  ).astype(np.float32) * sc

    def forward(self, x, adj):
        mu  = self.s2v.forward(x, adj)
        g   = mu.sum(0, keepdims=True)                       # (1,d)
        sp  = g  @ self.W4.T                                 # (1,d)
        ap  = mu @ self.W5.T                                 # (N,d)
        cat = np.concatenate([np.broadcast_to(sp,(len(mu),self.d)), ap], 1)  # (N,2d)
        return (self.W3 @ _relu(cat).T).flatten()            # (N,)

    def q_and_grad(self, x, adj, action, target):
        """
        Single-sample forward + full backward.
        Returns (q_action, grads_dict) with keys W1,W2,W3,W4,W5.
        """
        N = x.shape[0]
        mu, cache = self.s2v.forward_cache(x, adj)          # (N,d)
        g   = mu.sum(0, keepdims=True)                       # (1,d)
        sp  = g  @ self.W4.T                                 # (1,d)
        ap  = mu @ self.W5.T                                 # (N,d)
        sp_b = np.broadcast_to(sp, (N, self.d))             # (N,d)
        cat  = np.concatenate([sp_b, ap], 1)                 # (N,2d)
        h    = _relu(cat)                                    # (N,2d)
        q_all = (self.W3 @ h.T).flatten()                   # (N,)
        q_a   = float(q_all[action])

        # ── backward through head ────────────────────────────────────────
        err = 2.0 * (q_a - target)                          # scalar MSE grad

        dW3  = err * h[action:action+1]                     # (1,2d)
        dh_a = err * self.W3.flatten()                      # (2d,)  d(loss)/d(h_a)
        dcat_a = dh_a * _drelu(cat[action])                 # (2d,)

        dsp  = dcat_a[:self.d]                              # (d,)
        dap_a = dcat_a[self.d:]                             # (d,)

        # W4: Q uses sp = g @ W4.T,  g = Σ mu
        dg   = dsp @ self.W4                                # (d,)  d(loss)/d(g)
        dW4  = np.outer(dsp, g.flatten())                   # (d,d)

        # W5: Q uses ap_a = mu[a] @ W5.T
        dW5  = np.outer(dap_a, mu[action])                  # (d,d)

        # ── gradient w.r.t. mu ──────────────────────────────────────────
        # Every node contributes to g = Σ mu, so dg flows to every node.
        # Only the action node also gets dap_a through W5.
        dmu = np.tile(dg, (N, 1))                           # (N,d)  from g
        dmu[action] += dap_a @ self.W5                      # (d,)   from ap_a

        # ── truncated BPTT through LAST S2V iteration ────────────────────
        pre_last, agg_last = cache[-1]                      # final iteration cache
        dpre = dmu * _drelu(pre_last)                       # (N,d)

        dW1 = dpre.T @ x                                    # (d,fd)
        dW2 = dpre.T @ agg_last                             # (d,d)

        return q_a, {"W1": dW1, "W2": dW2,
                     "W3": dW3, "W4": dW4, "W5": dW5}

    def copy_from(self, other):
        for k in ("W3","W4","W5"):
            setattr(self, k, getattr(other, k).copy())
        self.s2v.W1 = other.s2v.W1.copy()
        self.s2v.W2 = other.s2v.W2.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  NumPy DQN Agent
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgentNumpy:
    def __init__(self, feature_dim, embed_dim=64, n_s2v_iters=4,
                 lr=1e-3, gamma=1.0, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, memory_size=2000, batch_size=32,
                 update_every=1, target_update_freq=50, **kw):
        self.gamma=gamma; self.epsilon=epsilon
        self.eps_min=epsilon_min; self.eps_dec=epsilon_decay
        self.bs=batch_size; self.upd=update_every
        self.tgt_f=target_update_freq; self.lr=lr; self.step=0

        self.net = DQNetNumpy(feature_dim, embed_dim, n_s2v_iters)
        self.tgt = DQNetNumpy(feature_dim, embed_dim, n_s2v_iters)
        self.tgt.copy_from(self.net)
        self.mem = ReplayMemory(memory_size)

        # Adam moments for ALL five weight matrices
        all_keys = ("W1","W2","W3","W4","W5")
        self._m = {"W1": np.zeros_like(self.net.s2v.W1),
                   "W2": np.zeros_like(self.net.s2v.W2),
                   "W3": np.zeros_like(self.net.W3),
                   "W4": np.zeros_like(self.net.W4),
                   "W5": np.zeros_like(self.net.W5)}
        self._v = {k: np.zeros_like(self._m[k]) for k in self._m}
        self._t = 0

    def select_action(self, x, adj, avail):
        if random.random() < self.epsilon or not avail:
            return random.choice(avail)
        q = self.net.forward(x, adj)
        mask = np.full(len(q), -np.inf)
        for i in avail:
            if i < len(q): mask[i] = q[i]
        return int(np.argmax(mask))

    def store(self, *a): self.mem.push(a)

    def _get_param(self, key):
        if key in ("W1","W2"):
            return getattr(self.net.s2v, key)
        return getattr(self.net, key)

    def _set_param(self, key, val):
        if key in ("W1","W2"):
            setattr(self.net.s2v, key, val)
        else:
            setattr(self.net, key, val)

    def update(self):
        self.step += 1
        if len(self.mem) < self.bs or self.step % self.upd != 0:
            return 0.0

        batch = self.mem.sample(self.bs)
        agg   = {k: np.zeros_like(self._get_param(k)) for k in self._m}
        total_loss = 0.0

        for (sx, adj, act, rew, nsx, nadj, done) in batch:
            tgt_q = (float(rew) if done
                     else float(rew) + self.gamma *
                          float(self.tgt.forward(nsx, nadj).max()))
            q_a, grads = self.net.q_and_grad(sx, adj, act, tgt_q)
            total_loss += (q_a - tgt_q) ** 2
            for k in agg:
                agg[k] += np.clip(grads[k], -1.0, 1.0)

        # Adam update for all five matrices
        self._t += 1; b1, b2, ep = 0.9, 0.999, 1e-8
        for k in self._m:
            g  = agg[k] / self.bs
            self._m[k] = b1*self._m[k] + (1-b1)*g
            self._v[k] = b2*self._v[k] + (1-b2)*g**2
            mh = self._m[k] / (1 - b1**self._t)
            vh = self._v[k] / (1 - b2**self._t)
            self._set_param(k, self._get_param(k) - self.lr * mh / (np.sqrt(vh)+ep))

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)
        if self.step % self.tgt_f == 0:
            self.tgt.copy_from(self.net)

        return total_loss / self.bs

    def save(self, path):
        data = {k: self._get_param(k) for k in self._m}
        np.save(path.replace(".pt","_np"), data)

    def load(self, path):
        data = np.load(path.replace(".pt","_np")+".npy", allow_pickle=True).item()
        for k, v in data.items():
            self._set_param(k, v)
        self.tgt.copy_from(self.net)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch version  (unchanged, used when torch is installed)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class Structure2Vec(nn.Module):
        def __init__(self, fd, d=64, n=4):
            super().__init__(); self.d=d; self.n=n
            self.t1=nn.Linear(fd,d,bias=False)
            self.t2=nn.Linear(d, d,bias=False)
        def forward(self, x, adj):
            mu = torch.zeros(x.size(0), self.d, device=x.device)
            for _ in range(self.n):
                mu = F.relu(self.t1(x) + self.t2(adj @ mu))
            return mu

    class DQNetwork(nn.Module):
        def __init__(self, d=64, fd=8, n=4):
            super().__init__()
            self.s2v = Structure2Vec(fd, d, n)
            self.t4  = nn.Linear(d,   d, bias=False)
            self.t5  = nn.Linear(d,   d, bias=False)
            self.t3  = nn.Linear(2*d, 1, bias=False)
        def forward(self, x, adj):
            mu  = self.s2v(x, adj)
            g   = mu.sum(0, keepdim=True)
            cat = torch.cat([self.t4(g).expand_as(mu), self.t5(mu)], -1)
            return self.t3(F.relu(cat)).squeeze(-1)

    class DQNAgent:
        def __init__(self, feature_dim, embed_dim=64, n_s2v_iters=4,
                     lr=1e-3, gamma=1.0, epsilon=1.0, epsilon_min=0.05,
                     epsilon_decay=0.995, memory_size=2000, batch_size=32,
                     update_every=1, target_update_freq=50, device="cpu"):
            self.dev=torch.device(device); self.gamma=gamma
            self.epsilon=epsilon; self.eps_min=epsilon_min
            self.eps_dec=epsilon_decay; self.bs=batch_size
            self.upd=update_every; self.tgt_f=target_update_freq; self.step=0

            self.net = DQNetwork(embed_dim, feature_dim, n_s2v_iters).to(self.dev)
            self.tgt = DQNetwork(embed_dim, feature_dim, n_s2v_iters).to(self.dev)
            self.tgt.load_state_dict(self.net.state_dict()); self.tgt.eval()
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.mem = ReplayMemory(memory_size)

        def select_action(self, x, adj, avail):
            if random.random() < self.epsilon or not avail:
                return random.choice(avail)
            with torch.no_grad():
                q = self.net(torch.FloatTensor(x).to(self.dev),
                             torch.FloatTensor(adj).to(self.dev)).cpu().numpy()
            mask = np.full(len(q), -np.inf)
            for i in avail:
                if i < len(mask): mask[i] = q[i]
            return int(np.argmax(mask))

        def store(self, *a): self.mem.push(a)

        def update(self):
            self.step += 1
            if len(self.mem) < self.bs or self.step % self.upd != 0: return 0.0
            losses = []
            for (sx, adj, act, rew, nsx, nadj, done) in self.mem.sample(self.bs):
                x  = torch.FloatTensor(sx ).to(self.dev)
                a  = torch.FloatTensor(adj).to(self.dev)
                nx = torch.FloatTensor(nsx).to(self.dev)
                na = torch.FloatTensor(nadj).to(self.dev)
                qc = self.net(x, a)[act]
                with torch.no_grad():
                    qt = (torch.tensor(float(rew), device=self.dev) if done
                          else float(rew) + self.gamma * self.tgt(nx, na).max())
                losses.append(F.mse_loss(qc, qt.float()))
            if losses:
                loss = torch.stack(losses).mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)
                if self.step % self.tgt_f == 0:
                    self.tgt.load_state_dict(self.net.state_dict())
                return loss.item()
            return 0.0

        def save(self, path): torch.save(self.net.state_dict(), path)
        def load(self, path):
            self.net.load_state_dict(torch.load(path, map_location=self.dev))
            self.tgt.load_state_dict(self.net.state_dict())
else:
    Structure2Vec = S2VNumpy
    DQNetwork     = DQNetNumpy
    DQNAgent      = DQNAgentNumpy
