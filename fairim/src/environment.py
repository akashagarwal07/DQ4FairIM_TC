"""
environment.py
--------------
Dynamic Fair-IM environment wrapping the graph and diffusion model.
The environment evolves over time (temporal changes, PDTF-IM style)
and supports a time-critical deadline (FAIRTCIM style).

State  : (graph snapshot, selected seeds so far, node features)
Action : pick one unselected node to add to seed set
Reward : marginal gain in (outreach + phi * maximin_fairness - gamma * latency_penalty)
"""

import random
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Optional

from .diffusion import simulate_ic_communities, simulate_tc_ic, marginal_reward


class FairIMEnvironment:
    """
    A time-evolving graph environment for fair influence maximization.

    Improvements over the base proposal
    ------------------------------------
    1. Temporal graph dynamics: edges added/removed each episode (PDTF-IM §III-C).
    2. Time-critical deadline: reward uses TC-IC when `deadline` is set (FAIRTCIM).
    3. Multi-objective reward: Reach + phi*Fairness - gamma*LatencyPenalty (Eq. 1, proposal).
    4. Node features include community membership alongside centrality.
    """

    def __init__(
        self,
        base_graph: nx.Graph,
        communities: dict,              # {node -> community_label}
        budget: int = 10,
        deadline: Optional[int] = None, # None = no time constraint
        temporal_change_rate: float = 0.02,  # fraction of edges changed at reset()
        step_change_rate: float = 0.0,       # fraction of edges changed every step() (continuous dynamics)
        ic_prob: float = 0.1,
        num_sim: int = 50,
        phi: float = 1.0,               # fairness weight  (DQ4FairIM Eq. 5)
        gamma_latency: float = 0.5,     # latency-penalty weight (proposal Eq. 1)
        seed: int = 42,
    ):
        self.base_graph = base_graph.copy()
        self.communities = communities
        self.budget = budget
        self.deadline = deadline
        self.temporal_change_rate = temporal_change_rate
        self.step_change_rate = step_change_rate
        self.ic_prob = ic_prob
        self.num_sim = num_sim
        self.phi = phi
        self.gamma_latency = gamma_latency
        random.seed(seed)
        np.random.seed(seed)

        # community metadata
        self._comm_nodes = defaultdict(set)
        for node, comm in communities.items():
            self._comm_nodes[comm].add(node)

        # working graph (updated each episode)
        self.graph: nx.Graph = None
        self.seed_set: set = None
        self.step_count: int = 0
        self._last_stats: dict = {}

    # ──────────────────────────────────────────────
    #  Episode Management
    # ──────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Apply temporal perturbation, reset seed set, return initial state."""
        self.graph = self._apply_temporal_change(self.base_graph.copy())
        self.seed_set = set()
        self.step_count = 0
        self._last_stats = {
            "outreach": 0.0,
            "fairness": 0.0,
            "per_comm": {c: 0.0 for c in self._comm_nodes},
        }
        return self._build_state()

    def step(self, action: int):
        """
        Add `action` (node ID) to seed set.

        Returns
        -------
        state  : np.ndarray
        reward : float
        done   : bool
        info   : dict
        """
        assert action not in self.seed_set, "Node already in seed set"

        self.seed_set.add(action)
        self.step_count += 1

        # ── Continuous dynamic graph evolution ───────────────────────────
        # Apply per-step perturbation so the graph evolves while the agent
        # is mid-way through selecting its seed set.  This implements
        # true continuous dynamics vs the episode-level change in reset().
        if self.step_change_rate > 0.0:
            self.graph = self._apply_temporal_change(
                self.graph, rate=self.step_change_rate)

        reward, info = self._compute_reward(action)
        self._last_stats = info

        done = (self.step_count >= self.budget)
        return self._build_state(), reward, done, info

    # ──────────────────────────────────────────────
    #  Reward (Eq. 1 of proposal + DQ4FairIM Eq. 6)
    # ──────────────────────────────────────────────

    def _compute_reward(self, new_node: int):
        sim_fn = simulate_tc_ic if self.deadline else simulate_ic_communities
        tc_kwargs = {"deadline": self.deadline} if self.deadline else {}

        prev_set = self.seed_set - {new_node}

        if prev_set:
            before = sim_fn(
                self.graph, prev_set, self.communities,
                prob=self.ic_prob, num_simulations=self.num_sim, **tc_kwargs
            )
        else:
            before = {"outreach": 0.0, "fairness": 0.0, "per_comm": {}}

        after = sim_fn(
            self.graph, self.seed_set, self.communities,
            prob=self.ic_prob, num_simulations=self.num_sim, **tc_kwargs
        )

        delta_reach    = after["outreach"] - before["outreach"]
        delta_fairness = after["fairness"] - before["fairness"]

        # latency penalty: penalise if minority community lags majority
        latency_penalty = 0.0
        if self.deadline and after["per_comm"]:
            vals = list(after["per_comm"].values())
            if vals:
                latency_penalty = max(vals) - min(vals)   # = disparity

        reward = (delta_reach
                  + self.phi * delta_fairness
                  - self.gamma_latency * latency_penalty)

        info = {**after, "latency_penalty": latency_penalty}
        return reward, info

    # ──────────────────────────────────────────────
    #  State Representation
    # ──────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """
        Per-node feature vector of shape (N, F).
        Features:
          0 : selected (binary)
          1 : degree (normalised)
          2 : pagerank
          3 : community one-hot encoded (up to 8 communities)
          ...
        """
        nodes = sorted(self.graph.nodes())
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}

        comm_labels = sorted(self._comm_nodes.keys())
        C = len(comm_labels)
        comm_idx = {c: i for i, c in enumerate(comm_labels)}

        F = 3 + C          # selected + degree + pagerank + community onehot
        state = np.zeros((N, F), dtype=np.float32)

        # degree
        degrees = dict(self.graph.degree())
        max_deg = max(degrees.values()) if degrees else 1

        # pagerank
        try:
            pr = nx.pagerank(self.graph, max_iter=50, tol=1e-3)
        except Exception:
            pr = {n: 1.0 / N for n in nodes}
        max_pr = max(pr.values()) if pr else 1

        for n in nodes:
            i = node_idx[n]
            state[i, 0] = 1.0 if n in self.seed_set else 0.0
            state[i, 1] = degrees.get(n, 0) / max_deg
            state[i, 2] = pr.get(n, 0) / max_pr
            comm = self.communities.get(n)
            if comm in comm_idx:
                state[i, 3 + comm_idx[comm]] = 1.0

        return state   # shape (N, F)

    # ──────────────────────────────────────────────
    #  Temporal Graph Evolution (PDTF-IM §III-C)
    # ──────────────────────────────────────────────

    def _apply_temporal_change(self, g: nx.Graph, rate: float = None) -> nx.Graph:
        """
        Randomly add/remove `temporal_change_rate` fraction of edges.
        Preserves node set and community assignments.
        """
        edges = list(g.edges())
        _rate = rate if rate is not None else self.temporal_change_rate
        n_change = max(1, int(len(edges) * _rate))

        # remove random edges
        n_remove = n_change // 2
        remove_edges = random.sample(edges, min(n_remove, len(edges)))
        g.remove_edges_from(remove_edges)

        # add random edges between existing nodes
        nodes = list(g.nodes())
        n_add = n_change - n_remove
        added = 0
        attempts = 0
        while added < n_add and attempts < n_add * 10:
            u, v = random.sample(nodes, 2)
            if not g.has_edge(u, v):
                weight = round(random.uniform(0.05, 0.3), 3)
                g.add_edge(u, v, weight=weight)
                added += 1
            attempts += 1

        return g

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def available_actions(self):
        return [n for n in self.graph.nodes() if n not in self.seed_set]

    def evaluate_seed_set(self, seed_set=None) -> dict:
        """Full evaluation (more simulations) of a given or current seed set."""
        s = seed_set if seed_set is not None else self.seed_set
        sim_fn = simulate_tc_ic if self.deadline else simulate_ic_communities
        tc_kwargs = {"deadline": self.deadline} if self.deadline else {}
        return sim_fn(
            self.graph, s, self.communities,
            prob=self.ic_prob, num_simulations=500, **tc_kwargs
        )

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def feature_dim(self):
        return 3 + len(self._comm_nodes)
