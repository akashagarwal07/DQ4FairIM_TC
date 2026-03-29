import os, time, torch, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, 'fairim')
from src import build_graph_pool, simulate_ic_communities, simulate_tc_ic, FairIMEnvironment, DQNAgent, get_adjacency_matrix, community_stats

def main():
    for d in ['fairim/src', 'fairim/checkpoints', 'fairim/results']: os.makedirs(d, exist_ok=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    pool = build_graph_pool(n_graphs=10, graph_type='hba', n_nodes=300, seed_start=0, homophily=0.75, minority_ratio=0.15)
    g0, c0 = pool[0]
    agent = DQNAgent(feature_dim=3 + len(set(c0.values())), embed_dim=64, n_s2v_iters=4, device=DEVICE)
    print('Training agent...')
    for ep in range(1, 401):
        g, c = pool[(ep - 1) % len(pool)]
        env = FairIMEnvironment(base_graph=g, communities=c, budget=10, temporal_change_rate=0.02, step_change_rate=0.005, ic_prob=0.1, num_sim=30, phi=1.0)
        state, done = env.reset(), False
        adj = get_adjacency_matrix(env.graph)
        while not done:
            action = agent.select_action(state, adj, env.available_actions())
            nstate, reward, done, _ = env.step(action)
            nadj = get_adjacency_matrix(env.graph)
            agent.store(state, adj, action, float(reward), nstate, nadj, done)
            agent.update()
            state, adj = nstate, nadj
        if ep % 50 == 0: print(f"  Episode {ep}/400 completed.")
    agent.save('fairim/checkpoints/agent_v3.pt')
    print('Saved to fairim/checkpoints/agent_v3.pt')

if __name__ == '__main__': main()
