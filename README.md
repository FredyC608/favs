# To create a local branch
git clone --recurse-submodules https://github.com/FredyC608/favs.git

VON: Hybrid RL/Heuristic Agent for "Case Closed"

VON is a cutting-edge, hybrid AI agent designed for the "Case Closed" (Tron Light Cycles) challenge. It leverages a unique two-phase architecture, transitioning from a deep reinforcement learning model for early-game strategy to a high-precision heuristic search for endgame tactical dominance.

🏆 Key Features & Hybrid Strategy

VON adapts its playstyle based on the game phase to maximize its win rate on the $18 \times 20$ torus grid.

**Phase 1: Early Game (Turn 30+) - PPO Reinforcement Learning**
## Reinforcement Learning Strategy

Our agent uses **self-play PPO** (Proximal Policy Optimization) with a ResNet-based policy network trained on 512 parallel game environments over 50M timesteps. The training pipeline implements **adaptive curriculum learning** that progressively shifts reward emphasis from survival (early) to territory control (mid) to strategic winning (late) based on achieved game lengths. We employ **adaptive batch sizing** based on policy loss variance and **learning rate annealing** to ensure stable convergence, with value loss clipping to prevent divergence. The opponent pool stores checkpoints every 5 updates, forcing the agent to continuously adapt to stronger versions of itself rather than exploiting fixed strategies. Our final deployment combines the trained RL policy with action masking (preventing illegal moves) and boost timing logic, achieving sub-4-second inference on CPU through optimized PyTorch operations.

**Phase 2: Mid/Late Game (Turn 30+) - Tactical Heuristics**

- Transition: At turn 30, control seamlessly passes to a Minimax w/ Alpha-Beta Pruning algorithm.

- Objective: As the board becomes crowded, precision is paramount. The heuristic engine aims to execute ruthless tactical maneuvers.

- Core Heuristic: Torus-Aware Voronoi Partitioning

- Instead of simple pathfinding, DeepBlueJay uses a Voronoi-based heuristic to evaluate board states.

- It calculates the "ownership" of every empty cell on the board based on which agent can reach it first (shortest path distance).

- This provides a fast and aggressive estimate of future territory control, encouraging the agent to move towards open space that it can claim faster than the opponent.

**Technical Deep Dive: The Voronoi Approach**

- We chose a Voronoi-based heuristic for our endgame search for several key reasons:

- Speed: Calculating BFS distances from two points is computationally faster than running full flood-fill operations for every reachable area. This allows our Minimax search to reach greater depths within the strict 4-second time limit.

- Aggression: Voronoi naturally rewards moving towards contested space to "claim" it, rather than just passively maximizing currently safe territory.

- Torus Awareness: Our implementation explicitly accounts for the wrap-around grid, correctly identifying when a cell on the far side of the board is actually just a few steps away.
