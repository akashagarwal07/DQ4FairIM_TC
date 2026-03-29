"""
baselines.py
------------
Fairness-agnostic and fairness-aware baseline seed selection methods.
Used to benchmark DQ4FairIM-TC (our agent).

Baselines implemented:
  Agnostic : CELF, Degree, PageRank
  Fair     : Parity (degree), Fair-PageRank, Greedy-Maximin
"""

import heapq
import random
import numpy as np
import networkx as nx
from collections import defaultdict

from .diffusion import simulate_ic_communities, simulate_tc_ic


# ──────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────

def _sim(graph, seed_set, communities, deadline, ic_prob, num_sim):
    if deadline:
        return simulate_tc_ic(graph, seed_set, communities,
                              deadline=deadline, prob=ic_prob,
                              num_simulations=num_sim)
    return simulate_ic_communities(graph, seed_set, communities,
                                   prob=ic_prob, num_simulations=num_sim)


# ──────────────────────────────────────────────────────────
#  1. Degree Heuristic
# ──────────────────────────────────────────────────────────

def degree_seeding(graph: nx.Graph, k: int, **kwargs) -> set:
    """Top-k highest degree nodes (fairness-agnostic)."""
    return set(sorted(graph.nodes(), key=lambda n: graph.degree(n),
                      reverse=True)[:k])


# ──────────────────────────────────────────────────────────
#  2. PageRank Heuristic
# ──────────────────────────────────────────────────────────

def pagerank_seeding(graph: nx.Graph, k: int, **kwargs) -> set:
    """Top-k PageRank nodes."""
    pr = nx.pagerank(graph, max_iter=100)
    return set(sorted(pr, key=pr.get, reverse=True)[:k])


# ──────────────────────────────────────────────────────────
#  3. CELF  (Leskovec et al., 2007)
# ──────────────────────────────────────────────────────────

def celf_seeding(graph: nx.Graph, k: int, communities: dict,
                 deadline=None, ic_prob=0.1, num_sim=100) -> set:
    """
    Cost-Effective Lazy Forward greedy for IM.
    Uses submodularity to skip re-evaluations.
    """
    seed_set = set()
    gains    = {}

    # initialise marginal gains
    for node in graph.nodes():
        res = _sim(graph, {node}, communities, deadline, ic_prob, num_sim // 2)
        gains[node] = res["outreach"]

    for _ in range(k):
        if not gains:
            break
        best = max(gains, key=gains.get)
        seed_set.add(best)
        del gains[best]

        # lazy re-evaluation
        updated = {}
        for node in gains:
            new_set = seed_set | {node}
            res = _sim(graph, new_set, communities, deadline, ic_prob, num_sim // 2)
            updated[node] = res["outreach"] - _sim(graph, seed_set, communities,
                                                   deadline, ic_prob, num_sim // 2)["outreach"]
        gains = updated

    return seed_set


# ──────────────────────────────────────────────────────────
#  4. Parity Seeding  (Stoica et al., 2020)
# ──────────────────────────────────────────────────────────

def parity_seeding(graph: nx.Graph, k: int, communities: dict,
                   **kwargs) -> set:
    """
    Degree-based parity: seed set mirrors community size fractions.
    """
    comm_nodes = defaultdict(list)
    for node, comm in communities.items():
        comm_nodes[comm].append(node)

    total = graph.number_of_nodes()
    seed_set = set()

    # allocate budget proportionally
    allocations = {}
    remaining   = k
    comm_list   = sorted(comm_nodes.keys())

    for i, comm in enumerate(comm_list):
        if i == len(comm_list) - 1:
            allocations[comm] = remaining
        else:
            frac = len(comm_nodes[comm]) / total
            alloc = max(1, round(frac * k))
            allocations[comm] = min(alloc, remaining)
            remaining -= allocations[comm]

    for comm, alloc in allocations.items():
        nodes_sorted = sorted(comm_nodes[comm],
                              key=lambda n: graph.degree(n), reverse=True)
        seed_set.update(nodes_sorted[:alloc])

    return seed_set


# ──────────────────────────────────────────────────────────
#  5. Fair PageRank Seeding
# ──────────────────────────────────────────────────────────

def fair_pagerank_seeding(graph: nx.Graph, k: int, communities: dict,
                           **kwargs) -> set:
    """Parity allocation applied to PageRank scores."""
    comm_nodes = defaultdict(list)
    for node, comm in communities.items():
        comm_nodes[comm].append(node)

    pr    = nx.pagerank(graph, max_iter=100)
    total = graph.number_of_nodes()
    seed_set = set()

    comm_list = sorted(comm_nodes.keys())
    remaining = k
    for i, comm in enumerate(comm_list):
        if i == len(comm_list) - 1:
            alloc = remaining
        else:
            frac  = len(comm_nodes[comm]) / total
            alloc = max(1, round(frac * k))
            remaining -= alloc

        nodes_sorted = sorted(comm_nodes[comm], key=lambda n: pr.get(n, 0), reverse=True)
        seed_set.update(nodes_sorted[:alloc])

    return seed_set


# ──────────────────────────────────────────────────────────
#  6. Greedy Maximin  (Farnadi et al., 2020)
# ──────────────────────────────────────────────────────────

def greedy_maximin_seeding(graph: nx.Graph, k: int, communities: dict,
                            deadline=None, ic_prob=0.1, num_sim=50) -> set:
    """
    Greedy seed selection maximising maximin fairness.
    At each step, adds the node with greatest marginal gain in fairness.
    """
    seed_set = set()

    for _ in range(k):
        best_node, best_gain = None, -np.inf
        for node in graph.nodes():
            if node in seed_set:
                continue
            res = _sim(graph, seed_set | {node}, communities,
                       deadline, ic_prob, num_sim)
            gain = res["fairness"]
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is not None:
            seed_set.add(best_node)

    return seed_set


# ──────────────────────────────────────────────────────────
#  Evaluation Helper
# ──────────────────────────────────────────────────────────

def evaluate_all_baselines(graph, k, communities, deadline=None,
                            ic_prob=0.1, num_sim=200,
                            include_slow=False) -> dict:
    """
    Run all baselines and return a results dict.
    Set include_slow=True to include CELF and Greedy-Maximin.
    """
    results = {}

    print("  Running Degree...")
    s = degree_seeding(graph, k)
    results["Degree"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
    results["Degree"]["seed_set"] = s

    print("  Running PageRank...")
    s = pagerank_seeding(graph, k)
    results["PageRank"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
    results["PageRank"]["seed_set"] = s

    print("  Running Parity...")
    s = parity_seeding(graph, k, communities)
    results["Parity"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
    results["Parity"]["seed_set"] = s

    print("  Running Fair-PageRank...")
    s = fair_pagerank_seeding(graph, k, communities)
    results["Fair-PageRank"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
    results["Fair-PageRank"]["seed_set"] = s

    if include_slow:
        print("  Running CELF (slow)...")
        s = celf_seeding(graph, k, communities, deadline, ic_prob, num_sim // 2)
        results["CELF"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
        results["CELF"]["seed_set"] = s

        print("  Running Greedy-Maximin (slow)...")
        s = greedy_maximin_seeding(graph, k, communities, deadline, ic_prob, num_sim // 4)
        results["Greedy-Maximin"] = _sim(graph, s, communities, deadline, ic_prob, num_sim)
        results["Greedy-Maximin"]["seed_set"] = s

    return results
