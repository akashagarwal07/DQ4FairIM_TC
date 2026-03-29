"""
graph_utils.py
--------------
Graph construction, dataset loading, and community assignment utilities.

Supports:
  • Rice University Facebook dataset  (socfb-Rice31)
  • Synthetic Homophily-BA networks   (DQ4FairIM baseline)
  • Stochastic Block Model            (FAIRTCIM synthetic experiments)
"""

import os
import random
import numpy as np
import networkx as nx
from collections import defaultdict


# ──────────────────────────────────────────────────────────
#  Synthetic: Stochastic Block Model  (FAIRTCIM §6.1)
# ──────────────────────────────────────────────────────────

def build_sbm_graph(
    n: int = 300,
    group_ratio: float = 0.7,    # fraction in majority group
    p_within: float = 0.05,      # edge prob within group
    p_across: float = 0.005,     # edge prob across groups
    weight_range: tuple = (0.05, 0.3),
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """
    Stochastic Block Model with 2 communities.
    Returns (graph, communities) where communities = {node -> group_label}.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    sizes = [int(n * group_ratio), n - int(n * group_ratio)]
    probs = [[p_within, p_across], [p_across, p_within]]

    G_sbm = nx.stochastic_block_model(sizes, probs, seed=seed)
    G     = nx.Graph()
    G.add_nodes_from(G_sbm.nodes())

    for u, v in G_sbm.edges():
        w = round(rng.uniform(*weight_range), 3)
        G.add_edge(u, v, weight=w)

    # community label: 0 = majority, 1 = minority
    communities = {}
    for node in G.nodes():
        communities[node] = 0 if node < sizes[0] else 1

    return G, communities


# ──────────────────────────────────────────────────────────
#  Synthetic: Homophily-BA  (DQ4FairIM §VI-A)
# ──────────────────────────────────────────────────────────

def build_hba_graph(
    n: int = 500,
    minority_ratio: float = 0.2,
    homophily: float = 0.8,
    m: int = 4,                  # edges per new node
    weight_range: tuple = (0.05, 0.3),
    seed: int = 42,
) -> tuple[nx.Graph, dict]:
    """
    Homophily-Barabási-Albert network.
    Minority nodes: label 1 (20%),  Majority: label 0 (80%).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    G = nx.Graph()
    communities = {}

    def assign_group(node_id):
        return 1 if rng.random() < minority_ratio else 0

    # seed graph
    for i in range(m + 1):
        G.add_node(i)
        communities[i] = assign_group(i)

    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=round(rng.uniform(*weight_range), 3))

    # grow
    for new_node in range(m + 1, n):
        new_group = assign_group(new_node)
        G.add_node(new_node)
        communities[new_node] = new_group

        existing = list(G.nodes())[:-1]
        degrees  = dict(G.degree())
        targets  = set()

        attempts = 0
        while len(targets) < m and attempts < m * 20:
            attempts += 1
            # preferential attachment weighted by homophily
            weights = []
            for nd in existing:
                if nd in targets:
                    weights.append(0)
                    continue
                h = homophily if communities[nd] == new_group else (1 - homophily)
                weights.append(h * (degrees.get(nd, 1)))

            total = sum(weights)
            if total == 0:
                break
            probs  = [w / total for w in weights]
            chosen = rng.choices(existing, weights=probs)[0]
            targets.add(chosen)

        for t in targets:
            G.add_edge(new_node, t,
                       weight=round(rng.uniform(*weight_range), 3))

    return G, communities


# ──────────────────────────────────────────────────────────
#  Real Dataset: Rice University Facebook  (socfb-Rice31)
# ──────────────────────────────────────────────────────────

def load_rice_facebook(path: str = None,
                       group_attr: str = "age") -> tuple[nx.Graph, dict]:
    """
    Load or simulate the Rice University Facebook dataset.
    Groups based on student age (18-19 vs 20-22) as in FAIRTCIM §7.1.

    If `path` is provided and the file exists, loads from edge list.
    Otherwise generates a synthetic approximation for demo purposes.
    """
    if path and os.path.exists(path):
        G = nx.read_edgelist(path, nodetype=int)
        # assign random ages as proxy if attribute file unavailable
        nodes = list(G.nodes())
        ages  = {n: random.choice([18, 19, 20, 21, 22]) for n in nodes}
        communities = {n: 0 if ages[n] <= 19 else 1 for n in nodes}
        # add random weights
        for u, v in G.edges():
            G[u][v]["weight"] = round(random.uniform(0.01, 0.05), 4)
        return G, communities

    # ── Synthetic approximation ──────────────────────────
    # 1205 nodes, ~42 000 edges, 4 age groups → 2 super-groups
    # Group V1 (age 18-19): ~97 nodes, Group V2 (age 20): ~344 nodes
    # We approximate with SBM
    print("[graph_utils] Rice dataset not found; using SBM approximation.")
    G, communities = build_sbm_graph(
        n=500,
        group_ratio=0.70,   # V1 larger
        p_within=0.08,
        p_across=0.005,
        seed=42,
    )
    return G, communities


# ──────────────────────────────────────────────────────────
#  Adjacency Matrix Helper
# ──────────────────────────────────────────────────────────

def get_adjacency_matrix(graph: nx.Graph,
                         weighted: bool = True) -> np.ndarray:
    """Return the adjacency matrix as a numpy array."""
    nodes = sorted(graph.nodes())
    n     = len(nodes)
    idx   = {nd: i for i, nd in enumerate(nodes)}
    adj   = np.zeros((n, n), dtype=np.float32)

    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0) if weighted else 1.0
        adj[idx[u], idx[v]] = w
        adj[idx[v], idx[u]] = w

    return adj


# ──────────────────────────────────────────────────────────
#  Graph Pool (for generalisation experiments, DQ4FairIM §VII-A)
# ──────────────────────────────────────────────────────────

def build_graph_pool(n_graphs: int = 10,
                     graph_type: str = "hba",
                     n_nodes: int = 300,
                     seed_start: int = 0,
                     **kwargs) -> list[tuple]:
    """
    Build a pool of graphs for training the RL agent.
    Each element is (graph, communities).
    """
    pool = []
    for i in range(n_graphs):
        s = seed_start + i
        if graph_type == "hba":
            g, c = build_hba_graph(n=n_nodes, seed=s, **kwargs)
        elif graph_type == "sbm":
            g, c = build_sbm_graph(n=n_nodes, seed=s, **kwargs)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        pool.append((g, c))
    return pool


# ──────────────────────────────────────────────────────────
#  Community Statistics
# ──────────────────────────────────────────────────────────

def community_stats(graph: nx.Graph, communities: dict) -> dict:
    """Print and return basic community statistics."""
    comm_counts = defaultdict(int)
    for node in graph.nodes():
        comm_counts[communities.get(node, "unknown")] += 1

    degrees = dict(graph.degree())
    comm_degree = defaultdict(list)
    for node, deg in degrees.items():
        comm_degree[communities.get(node, "unknown")].append(deg)

    stats = {}
    for comm, count in comm_counts.items():
        avg_deg = np.mean(comm_degree[comm]) if comm_degree[comm] else 0
        stats[comm] = {
            "size": count,
            "fraction": count / graph.number_of_nodes(),
            "avg_degree": round(avg_deg, 2),
        }

    print("\n── Community Statistics ──────────────")
    for comm, s in stats.items():
        print(f"  Group {comm}: size={s['size']} "
              f"({s['fraction']:.1%}), avg_degree={s['avg_degree']}")
    # set community as node attribute for assortativity calculation
    nx.set_node_attributes(graph, communities, name="community")
    try:
        h_idx = nx.attribute_assortativity_coefficient(graph, "community")
        print(f"  Homophily index: {h_idx:.3f}")
    except Exception:
        print("  Homophily index: N/A")
    print("──────────────────────────────────────\n")
    return stats
