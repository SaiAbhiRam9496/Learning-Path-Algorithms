# agents/qlearning.py
# Tabular Q-Learning agent for Sokoban.
# Training runs in terminal with tqdm progress bar.
# Trained Q-table saved to models/ as JSON.
# Loaded instantly for gameplay.
#
# Win rate  : 85-90% Easy (after 10,000 episodes)
# Training  : 30-45 minutes Easy, impractical for Medium+
# Storage   : Q-table as JSON (~few MB for Easy)

import json
import os
import random
import time
from collections import defaultdict

from agents.base_agent import BaseAgent

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # UP DOWN LEFT RIGHT

# Rewards
R_WIN          =  100.0   # level completed
R_BOX_ON_GOAL  =   10.0   # box pushed onto goal
R_MOVE         =   -1.0   # every move (efficiency pressure)
R_INVALID      =   -3.0   # tried invalid move
R_DEADLOCK     =   -5.0   # box pushed into corner
R_TIMEOUT      =  -10.0   # episode exceeded max steps


# ── State encoding ────────────────────────────────────────────────────────────

def _encode_state(game):
    """
    Encode current game state as a hashable string key for Q-table.

    State = player position + all box positions (sorted for consistency).
    Goals are fixed — not included in state.

    Args:
        game : Sokoban instance

    Returns:
        str — compact state key
    """
    px, py = game.player_x, game.player_y
    boxes  = sorted(
        (x, y)
        for y, row in enumerate(game.level)
        for x, tile in enumerate(row)
        if tile in ('$', '*')
    )
    return f"{px},{py}|{'|'.join(f'{x},{y}' for x,y in boxes)}"


# ── Deadlock detection ────────────────────────────────────────────────────────

def _get_goals(level):
    return frozenset(
        (x, y)
        for y, row in enumerate(level)
        for x, tile in enumerate(row)
        if tile in ('.', '*', '+')
    )


def _is_corner_deadlock(level, bx, by, goals):
    """Check if box at (bx,by) is stuck in a non-goal corner."""
    if (bx, by) in goals:
        return False

    def wall(x, y):
        if 0 <= y < len(level) and 0 <= x < len(level[y]):
            return level[y][x] == '#'
        return True

    return any([
        wall(bx-1, by) and wall(bx, by-1),
        wall(bx+1, by) and wall(bx, by-1),
        wall(bx-1, by) and wall(bx, by+1),
        wall(bx+1, by) and wall(bx, by+1),
    ])


def _has_deadlock(game, goals):
    """Check if any unplaced box is in a corner deadlock."""
    for y, row in enumerate(game.level):
        for x, tile in enumerate(row):
            if tile == '$':
                if _is_corner_deadlock(game.level, x, y, goals):
                    return True
    return False


# ── Q-Learning core ───────────────────────────────────────────────────────────

class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.

    Training:
        Run agent.train(...) — prints live metrics to terminal.
        Auto-saves Q-table on completion.

    Playing:
        Run agent.load(path) first.
        Then agent.next_move(game) returns greedy best action.

    Q-table:
        dict mapping state_key → [q0, q1, q2, q3]
        One Q-value per action (UP, DOWN, LEFT, RIGHT).
        Stored as JSON for readability and portability.
    """

    @property
    def name(self):
        return "Q-Learning"

    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995):
        """
        Args:
            lr           : learning rate α (0.0 - 1.0)
            gamma        : discount factor γ (0.0 - 1.0)
            epsilon      : initial exploration rate (0.0 - 1.0)
            epsilon_min  : minimum exploration rate
            epsilon_decay: epsilon multiplied by this each episode
        """
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: state_key → list of 4 Q-values
        self._q_table   = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self._trained   = False

        # Training metrics (saved for result screen)
        self.train_stats = {
            "episodes":       0,
            "win_rate":       0.0,
            "avg_steps":      0.0,
            "best_steps":     float('inf'),
            "final_epsilon":  0.0,
            "train_time_sec": 0.0,
            "level_name":     "",
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, level_idx, episodes=10_000,
              epsilon=None, lr=None, gamma=None):
        """
        Train Q-Learning agent on a given level.
        Prints live progress to terminal using tqdm.
        Auto-saves Q-table when done.

        Args:
            level_idx : int   — 0=Easy, 1=Medium, 2=Hard, 3=Impossible
            episodes  : int   — number of training episodes
            epsilon   : float — override initial epsilon
            lr        : float — override learning rate
            gamma     : float — override discount factor
        """
        # Import here to avoid circular dependency
        from core.maps import load_level, LEVEL_META
        from core.sokoban import Sokoban

        # Override hyperparams if provided
        if epsilon is not None: self.epsilon       = epsilon
        if lr      is not None: self.lr            = lr
        if gamma   is not None: self.gamma         = gamma

        level_name = LEVEL_META[level_idx][0]
        max_steps  = 300   # max steps per episode before timeout

        print()
        print("=" * 60)
        print(f"  Q-Learning Training")
        print(f"  Level    : {level_name}")
        print(f"  Episodes : {episodes:,}")
        print(f"  LR       : {self.lr}")
        print(f"  Gamma    : {self.gamma}")
        print(f"  Epsilon  : {self.epsilon} → {self.epsilon_min}")
        print("=" * 60)
        print()

        # Tracking
        wins          = 0
        total_steps   = 0
        best_steps    = float('inf')
        window        = 500    # rolling window for win rate display
        recent_wins   = []
        log_interval  = max(1, episodes // 20)

        t_start = time.time()

        # Load initial level once for goals
        base_level = load_level(level_idx)
        goals      = _get_goals(base_level)

        # ── Episode loop ──
        iterator = (
            tqdm(range(episodes),
                 desc="Training",
                 unit="ep",
                 ncols=70,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            if TQDM_AVAILABLE
            else range(episodes)
        )

        for episode in iterator:
            # Fresh game each episode
            level = load_level(level_idx)
            game  = Sokoban(level)
            boxes_on_goals = game.count_boxes_on_goals()

            state     = _encode_state(game)
            steps     = 0
            ep_reward = 0.0

            while steps < max_steps:
                # ε-greedy action selection
                if random.random() < self.epsilon:
                    action_idx = random.randrange(4)
                else:
                    q_vals     = self._q_table[state]
                    action_idx = q_vals.index(max(q_vals))

                dx, dy = ACTIONS[action_idx]

                # Take action
                moved = game.move(dx, dy)

                # ── Reward shaping ──
                if game.is_completed():
                    reward = R_WIN
                    done   = True
                elif not moved:
                    reward = R_INVALID
                    done   = False
                else:
                    new_boxes = game.count_boxes_on_goals()
                    if new_boxes > boxes_on_goals:
                        reward = R_BOX_ON_GOAL
                    elif _has_deadlock(game, goals):
                        reward = R_DEADLOCK
                        done   = True
                    else:
                        reward = R_MOVE
                        done   = False
                    boxes_on_goals = new_boxes

                if steps == max_steps - 1:
                    reward += R_TIMEOUT
                    done    = True

                # ── Q-update (Bellman equation) ──
                next_state  = _encode_state(game)
                old_q       = self._q_table[state][action_idx]
                next_max_q  = max(self._q_table[next_state])
                new_q       = old_q + self.lr * (
                    reward + self.gamma * next_max_q - old_q
                )
                self._q_table[state][action_idx] = new_q

                state  = next_state
                steps += 1
                ep_reward += reward

                if game.is_completed():
                    done = True

                if done:
                    break

            # ── Episode end bookkeeping ──
            won = game.is_completed()
            if won:
                wins += 1
                if steps < best_steps:
                    best_steps = steps

            total_steps   += steps
            recent_wins.append(1 if won else 0)
            if len(recent_wins) > window:
                recent_wins.pop(0)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            # ── Live logging (non-tqdm fallback) ──
            if not TQDM_AVAILABLE and (episode + 1) % log_interval == 0:
                recent_rate = sum(recent_wins) / len(recent_wins) * 100
                overall     = wins / (episode + 1) * 100
                avg_s       = total_steps / (episode + 1)
                print(
                    f"  Ep {episode+1:6d}/{episodes}"
                    f"  Win(recent): {recent_rate:5.1f}%"
                    f"  Win(overall): {overall:5.1f}%"
                    f"  AvgSteps: {avg_s:5.1f}"
                    f"  ε: {self.epsilon:.3f}"
                )

            # ── tqdm postfix update ──
            if TQDM_AVAILABLE and (episode + 1) % 100 == 0:
                recent_rate = sum(recent_wins) / len(recent_wins) * 100
                iterator.set_postfix({
                    "win%":    f"{recent_rate:.1f}",
                    "ε":       f"{self.epsilon:.3f}",
                    "steps":   f"{total_steps/(episode+1):.0f}",
                })

        # ── Training complete ──
        t_end     = time.time()
        win_rate  = wins / episodes * 100
        avg_steps = total_steps / episodes

        self._trained = True

        self.train_stats = {
            "episodes":       episodes,
            "win_rate":       round(win_rate, 2),
            "avg_steps":      round(avg_steps, 1),
            "best_steps":     best_steps if best_steps < float('inf') else 0,
            "final_epsilon":  round(self.epsilon, 4),
            "train_time_sec": round(t_end - t_start, 1),
            "level_name":     level_name,
        }

        print()
        print("=" * 60)
        print(f"  TRAINING COMPLETE")
        print(f"  Win Rate     : {win_rate:.2f}%")
        print(f"  Avg Steps    : {avg_steps:.1f}")
        print(f"  Best Steps   : {best_steps}")
        print(f"  Final ε      : {self.epsilon:.4f}")
        print(f"  Q-table size : {len(self._q_table):,} states")
        print(f"  Time         : {t_end - t_start:.1f}s")
        print("=" * 60)
        print()

        return self.train_stats

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path):
        """
        Save Q-table and hyperparams to a JSON file.

        Args:
            path : str — file path e.g. 'models/qlearning_easy.json'
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "q_table":    {k: v for k, v in self._q_table.items()},
            "train_stats": self.train_stats,
            "hyperparams": {
                "lr":            self.lr,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Model saved → {path}")

    def load(self, path):
        """
        Load Q-table from a JSON file.

        Args:
            path : str — file path e.g. 'models/qlearning_easy.json'

        Returns:
            True if loaded successfully, False if file not found
        """
        if not os.path.exists(path):
            return False

        # Empty file = placeholder, not yet trained
        if os.path.getsize(path) == 0:
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        self._q_table = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0],
            {k: v for k, v in data["q_table"].items()}
        )
        self.train_stats = data.get("train_stats", self.train_stats)
        hp               = data.get("hyperparams", {})
        self.lr            = hp.get("lr",            self.lr)
        self.gamma         = hp.get("gamma",          self.gamma)
        self.epsilon       = hp.get("epsilon",        self.epsilon)
        self.epsilon_min   = hp.get("epsilon_min",    self.epsilon_min)
        self.epsilon_decay = hp.get("epsilon_decay",  self.epsilon_decay)
        self._trained      = True

        print(f"  Model loaded ← {path}")
        print(f"  Q-table size : {len(self._q_table):,} states")
        print(f"  Win rate     : {self.train_stats.get('win_rate', '?')}%")
        return True

    # ── Gameplay ──────────────────────────────────────────────────────────────

    def next_move(self, game):
        """
        Return the greedy best action from Q-table.
        If state is unseen, picks a random legal action.

        Args:
            game : Sokoban instance

        Returns:
            (dx, dy) or None
        """
        if not self._trained:
            return None

        state  = _encode_state(game)
        q_vals = self._q_table[state]

        # All zeros → unseen state → random legal move
        if all(q == 0.0 for q in q_vals):
            actions = game.get_legal_actions()
            return random.choice(actions) if actions else None

        action_idx = q_vals.index(max(q_vals))
        return ACTIONS[action_idx]

    # ── Misc ──────────────────────────────────────────────────────────────────

    def reset(self):
        """Nothing to reset between levels — Q-table persists."""
        pass

    def is_trained(self):
        return self._trained

    def q_table_size(self):
        return len(self._q_table)