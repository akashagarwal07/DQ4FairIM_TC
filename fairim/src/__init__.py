"""
Dynamic Fairness-Aware Influence Maximization (DQ4FairIM-TC)
------------------------------------------------------------
An RL-based framework combining:
  • DQ4FairIM   (Saxena et al., 2025)   - Deep Q-learning for Fair IM
  • FAIRTCIM    (Ali et al., 2023)      - Time-critical fairness
  • PDTF-IM     (Meena et al., 2025)    - Privacy, diversity, temporal fairness
"""

from .diffusion    import simulate_ic, simulate_ic_communities, simulate_tc_ic, marginal_reward
from .environment  import FairIMEnvironment
from .agent        import DQNAgent, DQNetwork, Structure2Vec
from .graph_utils  import (build_sbm_graph, build_hba_graph, load_rice_facebook,
                            get_adjacency_matrix, build_graph_pool, community_stats)
from .baselines    import (degree_seeding, pagerank_seeding, parity_seeding,
                            fair_pagerank_seeding, greedy_maximin_seeding,
                            evaluate_all_baselines)

__all__ = [
    "simulate_ic", "simulate_ic_communities", "simulate_tc_ic", "marginal_reward",
    "FairIMEnvironment",
    "DQNAgent", "DQNetwork", "Structure2Vec",
    "build_sbm_graph", "build_hba_graph", "load_rice_facebook",
    "get_adjacency_matrix", "build_graph_pool", "community_stats",
    "degree_seeding", "pagerank_seeding", "parity_seeding",
    "fair_pagerank_seeding", "greedy_maximin_seeding", "evaluate_all_baselines",
]
