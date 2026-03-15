# Learning-Path-Algorithms

A Sokoban puzzle game with four AI agents — two classical, one reinforcement learning, one quantum — built to understand how different algorithms approach the same problem.

## Agents

| Agent      | Algorithm                |       Win Rate     | Notes                          |
|------------|--------------------------|--------------------|--------------------------------|
| A*         | Heuristic search         | 100% Easy / Medium | Optimal solution, no training  |
| MCTS       | Monte Carlo Tree Search  | 95%+ Easy          | 500 simulations per move       |
| Q-Learning | Tabular RL               | 87–90% Easy        | Trains in ~6s, saves to JSON   |
| Grover     | Quantum search           | 90%+ Easy          | Hybrid quantum-classical       |

## Setup

```bash
git clone https://github.com/SaiAbhiRam9496/Learning-Path-Algorithms.git
cd Learning-Path-Algorithms
pip install -r requirements.txt
python main.py
```

**Dependencies:** `pygame` `numpy` `tqdm` `qiskit` `qiskit-aer`

> Grover runs in simulated mode if Qiskit is not installed — behaviour is identical, clearly labelled in the UI.

## Controls

| Key        | Action       |
|------------|--------------|
| Arrow keys | Move         |
| R          | Reset level  |
| ESC        | Menu         |

During gameplay, the agent dropdown (top-right) lets you switch agents mid-game without resetting the board.

## Levels

| Level       | Boxes | Min Moves | Agent Support |
|-------------|-------|-----------|---------------|
| Easy        | 2     | 10        | All agents    |
| Medium      | 3     | 20        | All agents    |
| Hard        | 6     | ~50       | A*, MCTS only |
| Impossible  | 10    | ~100      | A*, MCTS only |



## Training Q-Learning

Navigate to **Training** from the home screen. Recommended settings:

| Parameter         | Value |
|-------------------|-------|
| Episodes          | 10,000 |
| Epsilon           | 1.0 |
| Learning Rate     | 0.1 |
| Gamma             | 0.95 |

Training runs in the terminal with a live progress bar. The model saves automatically to `models/` on completion.



## Project Structure

```
├── core/
│   ├── maps.py          4 levels, BFS-verified
│   └── sokoban.py       Game logic, no pygame dependency
│
├── agents/
│   ├── base_agent.py    Shared interface
│   ├── astar.py         A* search
│   ├── mcts.py          Monte Carlo Tree Search
│   ├── qlearning.py     Tabular Q-Learning
│   └── human.py         Keyboard input
│
├── quantum/
│   └── grover.py        Grover's search via Qiskit Aer
│
├── ui/
│   ├── renderer.py      Grid and HUD rendering
│   ├── screens.py       Menu and completion screens
│   ├── training_panel.py  Training setup and results
│   └── game_panel.py    Game select, overlay, post-game
│
├── models/              Saved Q-tables (JSON)
├── results/             Training run history
└── tests/
    ├── test_sokoban.py
    └── test_agents.py   43 tests
```
## Tests
```bash
python tests/test_agents.py
```
## Findings
Quantum does not outperform classical on this problem. With only 4 possible moves, Grover's quadratic speedup provides a 4× amplification — demonstrable but not decisive. Classical A* remains optimal.
The value here is not performance. It is understanding amplitude amplification as a mechanism, and seeing exactly where the boundary between quantum advantage and classical efficiency sits.
