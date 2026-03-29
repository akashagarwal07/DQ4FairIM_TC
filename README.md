# DQ4FairIM-TC

Dynamic Fairness-Aware Seed Selection for Real-Time Information Diffusion. 
This project implements a Deep Q-Network (DQN) with a fully trainable Structure2Vec (S2V) embedding to solve the Influence Maximization problem on highly homophilic, dynamic networks.

## Setup

***Method - 1***
    Use the self-contained Google Colab notebook. Needs uploading of files in 'fairim/src' folder.

***Method - 2***

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/akashagarwal07/DQ4FairIM_TC
   cd DQ4FairIM_TC
   
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt

## Usage

1. **Train the Agent**
Train the DQ4FairIM-TC agent using the continuous dynamic graph settings. The trained weights will be saved to fairim/checkpoints/agent_v3.pt.
    ```bash
    python train.py

2. **Evaluate Baselines and Agent**
Run a full evaluation on a test graph, comparing the trained agent against heuristics like Degree, PageRank, Parity, and Fair-PageRank. Results will be saved to fairim/results/summary_v3.json.
    ```bash
    python evaluate.py

3. **Quick Demo**
Run a quick, lightweight demonstration of seed selection on a smaller graph.
    ```bash
    python demo.py
