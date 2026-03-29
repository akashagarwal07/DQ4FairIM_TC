import os, json, torch, sys
sys.path.insert(0, 'fairim')
from src import build_hba_graph, DQNAgent, get_adjacency_matrix, evaluate_all_baselines, FairIMEnvironment
from src.diffusion import gini_coefficient

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    g_test, c_test = build_hba_graph(n=300, m=3, homophily=0.75, minority_ratio=0.15)
    print("Evaluating baselines...")
    results = evaluate_all_baselines(g_test, k=10, communities=c_test, ic_prob=0.1, num_sim=300)
    agent = DQNAgent(feature_dim=3 + len(set(c_test.values())), embed_dim=64, n_s2v_iters=4, device=DEVICE)
    try: agent.load('fairim/checkpoints/agent_v3.pt')
    except: print("No agent found. Run train.py first."); return
    env = FairIMEnvironment(base_graph=g_test, communities=c_test, budget=10, temporal_change_rate=0.0, step_change_rate=0.0, ic_prob=0.1, num_sim=300, phi=1.0)
    state, done = env.reset(), False
    adj = get_adjacency_matrix(env.graph)
    while not done:
        action = agent.select_action(state, adj, env.available_actions(), evaluate=True)
        state, _, done, _ = env.step(action)
        adj = get_adjacency_matrix(env.graph)
    res_agent = env.evaluate_seed_set()
    results['DQ4FairIM-TC'] = res_agent
    summary = {m: {'outreach': r['outreach'], 'fairness': r['fairness'], 'disparity': r['disparity']} for m, r in results.items()}
    os.makedirs('fairim/results', exist_ok=True)
    with open('fairim/results/summary_v3.json', 'w') as f: json.dump(summary, f, indent=2)
    print("Results saved to fairim/results/summary_v3.json")

if __name__ == '__main__': main()
