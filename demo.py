import sys, torch
sys.path.insert(0, 'fairim')
from src import build_hba_graph, DQNAgent, get_adjacency_matrix, FairIMEnvironment

def main():
    print("--- Quick Demo ---")
    g, c = build_hba_graph(n=100, m=3, homophily=0.75, minority_ratio=0.15)
    agent = DQNAgent(feature_dim=3 + len(set(c.values())), embed_dim=64, n_s2v_iters=4)
    try: agent.load('fairim/checkpoints/agent_v3.pt'); print("Loaded agent.")
    except: print("Using untrained agent.")
    env = FairIMEnvironment(base_graph=g, communities=c, budget=5, temporal_change_rate=0.02, step_change_rate=0.005, ic_prob=0.1, num_sim=50, phi=1.0)
    state, done = env.reset(), False
    adj = get_adjacency_matrix(env.graph)
    while not done:
        action = agent.select_action(state, adj, env.available_actions(), evaluate=True)
        state, reward, done, _ = env.step(action)
        adj = get_adjacency_matrix(env.graph)
        print(f"Selected: {action} | Marginal Reward: {reward:.4f}")

if __name__ == '__main__': main()
