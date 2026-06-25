# T-MAPPO: Multi-UAV Trajectory Optimization with Transformer-Enhanced MAPPO

This repository implements the T-MAPPO algorithm from the ICCT 2023 paper:

> **"Multi-UAV Searching Trajectory Optimization Algorithm based on Deep Reinforcement Learning"**
> Bo Zhang, Kunhao Yang, et al.
> 2023 IEEE 23rd International Conference on Communication Technology (ICCT)

## Overview

T-MAPPO is a variant of MAPPO (Multi-Agent Proximal Policy Optimization) that introduces a **Transformer encoder with self-attention mechanism** into the critic network. The key idea is that as the number of agents increases, the environment state dimension grows, making it difficult for a standard fully-connected MLP critic to converge. The Transformer enables the critic to differentiate attention across agents and improve training.

## T-MAPPO vs MAPPO: Key Differences

| Aspect | MAPPO | T-MAPPO (This Work) |
|--------|-------|---------------------|
| Critic Network | MLP (fully connected) | MLP + Transformer Encoder + Self-Attention |
| Agent Interaction | Agents implicitly interact via concatenated state | Agents explicitly attend to each other's features |
| Scalability | Performance degrades as #agents grows | Attention mechanism helps maintain convergence |
| Training Speed | Faster per-iteration | Slightly slower per-iteration due to attention computation |
| Convergence | Slower with many agents | Faster convergence, better rewards |

### Why Transformer in the Critic?

In MAPPO's CTDE (Centralized Training Decentralized Execution) framework, the critic has access to global information. With simple MLP, this global information is concatenated as a flat vector, which becomes prohibitively large as #agents grows. The Transformer solves this by:

1. Each agent's observation is encoded into a feature vector eₙ via an MLP
2. All feature vectors are fed into a Transformer with self-attention (Q, K, V projections)
3. The attention output is passed through another MLP to produce the value estimate
4. This allows the critic to selectively focus on the most relevant agents' information

## File Structure

```
code/t-mappo/
├── env_uav_search.py    # Custom multi-UAV search environment
├── tmappo.py            # T-MAPPO algorithm implementation
└── README.md            # This file
```

## Environment: UAV Search Grid

The environment simulates a post-disaster search scenario:

- **Grid**: 10×10 discrete grid (representing 10km × 10km area)
- **Agents**: 4 UAVs searching for ground users
- **Users**: 10 mobile ground users (random walk)
- **Observations**: Positions of all UAVs, their directions, visited cells, and found users
- **Actions**: Choose direction offset (-1, 0, +1) relative to current heading (3 choices, per Fig. 2 in paper)
- **Rewards**: β × discovered_users + α × visited_cells − connectivity_penalty − collision_penalty − energy_cost − boundary_penalty

### Constraints
- UAV communication range: d_max = 7.5 km (connectivity must be maintained)
- Minimum collision distance: d_min = 0.15 km
- Each UAV can only turn to 3 adjacent directions per step (curvature constraint)

## Quick Start

```bash
# Run training (20 episodes)
python -c "
from env_uav_search import UAVSearchEnv
from tmappo import TMAPPO, train_tmappo

env = UAVSearchEnv(grid_size=(10,10), num_uavs=4, num_users=10, max_steps=200)
agent = TMAPPO(obs_dim=20, num_agents=4, n_actions=3, use_transformer=True)

rewards, found = train_tmappo(env, agent, num_episodes=20)
print(f'Avg reward: {sum(rewards)/len(rewards):.2f}')
print(f'Avg found: {sum(found)/len(found):.1f}/{env.num_users}')
"
```

## Hyperparameters (from paper Table I)

| Parameter | Value |
|-----------|-------|
| Grid size | 10×10 (10km × 10km) |
| Number of UAVs (N) | 4 |
| Number of users (M) | 10 |
| UAV altitude (H) | 100m |
| Max speed (Vmax) | 50m/s |
| Mission time (T) | 30 min |
| Max communication distance (dmax) | 7500m |
| Min collision distance (dmin) | 150m |
| Episode length | 200 steps |
| Total training steps | 2.5k |
| Actor learning rate | 0.001 |
| Critic learning rate | 0.0003 |
| Transformer heads | 4 |
| Hidden dimension | 128 |

## Notes

- The environment is CPU-only (no GPU required)
- The implementation uses PyTorch and runs on standard Python 3.9+
- For best results, train with 100+ episodes
- Results are logged every 50 episodes during training
