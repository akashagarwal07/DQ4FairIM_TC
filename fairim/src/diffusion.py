"""
diffusion.py
------------
Implements Independent Cascade (IC) model, Time-Critical IC (TCIM),
and fairness metrics derived from:
  - DQ4FairIM (Saxena et al., 2025)
  - FAIRTCIM   (Ali et al., 2023)
  - PDTF-IM    (Meena et al., 2025)
"""

import random
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────
#  Core IC Simulation
# ─────────────────────────────────────────────────

def simulate_ic(graph, seed_set, prob: float = 0.1,
                num_simulations: int = 100) -> float:
    """
    Standard Independent Cascade model.

    Parameters
    ----------
    graph          : networkx.Graph
    seed_set       : iterable of seed node IDs
    prob           : uniform edge activation probability
    num_simulations: number of Monte Carlo runs

    Returns
    -------
    float : expected fraction of activated nodes
    """
    seed_set = set(seed_set)
    total_nodes = graph.number_of_nodes()
    total_activated = 0

    for _ in range(num_simulations):
        activated = set(seed_set)
        frontier = set(seed_set)

        while frontier:
            new_frontier = set()
            for u in frontier:
                for v in graph.neighbors(u):
                    if v not in activated:
                        edge_prob = graph[u][v].get("weight", prob)
                        if random.random() < edge_prob:
                            activated.add(v)
                            new_frontier.add(v)
            frontier = new_frontier

        total_activated += len(activated)

    return total_activated / (total_nodes * num_simulations)


# ─────────────────────────────────────────────────
#  Community-aware IC (returns per-community stats)
# ─────────────────────────────────────────────────

def simulate_ic_communities(graph, seed_set, communities: dict,
                             prob: float = 0.1,
                             num_simulations: int = 100) -> dict:
    """
    IC model that tracks influence per community.

    Parameters
    ----------
    communities : dict  {node_id -> community_label}

    Returns
    -------
    dict : {
        "outreach"  : float  (fraction of all nodes activated),
        "per_comm"  : {community_label -> fraction of community activated},
        "fairness"  : float  (maximin fairness score),
        "disparity" : float  (max pairwise difference),
    }
    """
    seed_set = set(seed_set)
    nodes = list(graph.nodes())
    total_nodes = len(nodes)

    # community -> set of nodes
    comm_nodes = defaultdict(set)
    for node, comm in communities.items():
        comm_nodes[comm].add(node)

    comm_labels = list(comm_nodes.keys())

    # accumulators
    total_act = 0
    comm_act = defaultdict(float)

    for _ in range(num_simulations):
        activated = set(seed_set)
        frontier = set(seed_set)

        while frontier:
            new_frontier = set()
            for u in frontier:
                for v in graph.neighbors(u):
                    if v not in activated:
                        edge_prob = graph[u][v].get("weight", prob)
                        if random.random() < edge_prob:
                            activated.add(v)
                            new_frontier.add(v)
            frontier = new_frontier

        total_act += len(activated)
        for comm in comm_labels:
            comm_act[comm] += len(activated & comm_nodes[comm])

    # normalise
    outreach = total_act / (total_nodes * num_simulations)
    per_comm = {}
    for comm in comm_labels:
        size = len(comm_nodes[comm])
        per_comm[comm] = comm_act[comm] / (size * num_simulations) if size > 0 else 0.0

    fairness = min(per_comm.values()) if per_comm else 0.0

    values = list(per_comm.values())
    disparity = (max(values) - min(values)) if len(values) > 1 else 0.0

    return {
        "outreach": outreach,
        "per_comm": per_comm,
        "fairness": fairness,        # maximin (Eq. 2, DQ4FairIM)
        "disparity": disparity,      # Eq. 2, FAIRTCIM
    }


# ─────────────────────────────────────────────────
#  Time-Critical IC  (FAIRTCIM, Ali et al. 2023)
# ─────────────────────────────────────────────────

def simulate_tc_ic(graph, seed_set, communities: dict,
                   deadline: int = 5,
                   prob: float = 0.1,
                   num_simulations: int = 100) -> dict:
    """
    Time-Critical Independent Cascade.
    A node earns utility 1 only if activated BEFORE `deadline`.

    Returns same structure as simulate_ic_communities.
    """
    seed_set = set(seed_set)
    total_nodes = graph.number_of_nodes()

    comm_nodes = defaultdict(set)
    for node, comm in communities.items():
        comm_nodes[comm].add(node)
    comm_labels = list(comm_nodes.keys())

    total_act = 0
    comm_act = defaultdict(float)

    for _ in range(num_simulations):
        # track activation time for each node
        activation_time = {n: float("inf") for n in graph.nodes()}
        for s in seed_set:
            activation_time[s] = 0

        activated = set(seed_set)
        frontier = set(seed_set)
        t = 0

        while frontier and t < deadline:
            t += 1
            new_frontier = set()
            for u in frontier:
                for v in graph.neighbors(u):
                    if v not in activated:
                        edge_prob = graph[u][v].get("weight", prob)
                        if random.random() < edge_prob:
                            activated.add(v)
                            activation_time[v] = t
                            new_frontier.add(v)
            frontier = new_frontier

        # only count nodes activated within deadline
        timely = {n for n, t_act in activation_time.items()
                  if t_act <= deadline and t_act != float("inf")}

        total_act += len(timely)
        for comm in comm_labels:
            comm_act[comm] += len(timely & comm_nodes[comm])

    outreach = total_act / (total_nodes * num_simulations)
    per_comm = {}
    for comm in comm_labels:
        size = len(comm_nodes[comm])
        per_comm[comm] = comm_act[comm] / (size * num_simulations) if size > 0 else 0.0

    fairness = min(per_comm.values()) if per_comm else 0.0
    values = list(per_comm.values())
    disparity = (max(values) - min(values)) if len(values) > 1 else 0.0

    return {
        "outreach": outreach,
        "per_comm": per_comm,
        "fairness": fairness,
        "disparity": disparity,
    }


# ─────────────────────────────────────────────────
#  Marginal Reward  (for RL agent step)
# ─────────────────────────────────────────────────

def marginal_reward(graph, current_seed, new_node, communities: dict,
                    phi: float = 1.0,
                    deadline: int = None,
                    prob: float = 0.1,
                    num_simulations: int = 50) -> float:
    """
    Computes the marginal gain in (outreach + phi * fairness)
    from adding `new_node` to `current_seed`.

    This matches Eq. (6) from DQ4FairIM.
    When deadline is set, uses TC-IC (FAIRTCIM setting).
    """
    sim_fn = simulate_tc_ic if deadline is not None else simulate_ic_communities
    kwargs = {"deadline": deadline} if deadline is not None else {}

    before = sim_fn(graph, current_seed, communities,
                    prob=prob, num_simulations=num_simulations, **kwargs)
    after  = sim_fn(graph, current_seed | {new_node}, communities,
                    prob=prob, num_simulations=num_simulations, **kwargs)

    delta_outreach  = after["outreach"]  - before["outreach"]
    delta_fairness  = after["fairness"]  - before["fairness"]

    return delta_outreach + phi * delta_fairness


# ─────────────────────────────────────────────────
#  Fairness Evaluation Helpers
# ─────────────────────────────────────────────────

def compute_welfare(per_comm: dict, alpha: float = -1.0) -> float:
    """
    Welfare function from Rahmattalabi et al. (cited in FairSNA):
      W_alpha(U) = sum_i U_i^alpha / alpha   for alpha != 0
      W_0(U)    = sum_i log(U_i)
    alpha -> -inf corresponds to maximin.
    """
    values = [v for v in per_comm.values() if v > 0]
    if not values:
        return 0.0
    if alpha == 0:
        return sum(np.log(v) for v in values)
    return sum(v ** alpha for v in values) / alpha


def gini_coefficient(per_comm: dict) -> float:
    """
    Gini coefficient over community outreach fractions.
    0 = perfect equality, 1 = maximum inequality.
    Used in FIMMAGA (Gong & Guo, 2023).
    """
    values = sorted(per_comm.values())
    n = len(values)
    if n == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(values))
    total = sum(values)
    if total == 0:
        return 0.0
    return (2 * cumulative) / (n * total) - (n + 1) / n
