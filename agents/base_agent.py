# agents/base_agent.py
# Common interface that every agent must follow.
# main.py and game_panel.py never need to know which
# agent is running — they always call the same methods.
#
# Every agent inherits from BaseAgent and implements:
#   next_move(game) → (dx, dy)
#   is_trained()    → bool
#   reset()         → None


class BaseAgent:
    """
    Abstract base class for all Sokoban agents.

    Subclasses must implement:
        next_move(game)  → (dx, dy) tuple
        name             → string property
        is_trained()     → bool

    Subclasses may override:
        reset()          → called when level resets
        on_level_start() → called once before agent plays
    """

    # ── Identity ─────────────────────────────────────────────────────────

    @property
    def name(self):
        """
        Human-readable name shown in the UI dropdown.
        Every subclass must override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must define a 'name' property."
        )

    # ── Core method ───────────────────────────────────────────────────────

    def next_move(self, game):
        """
        Given the current game state, return the next move.

        Args:
            game : Sokoban instance (from core/sokoban.py)

        Returns:
            (dx, dy) tuple where dx, dy ∈ {-1, 0, 1}
            e.g. (0, -1) = UP, (0, 1) = DOWN,
                 (-1, 0) = LEFT, (1, 0) = RIGHT

            Return None if the agent has no move
            (level unsolvable, agent stuck, training needed).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement next_move()."
        )

    # ── Training ──────────────────────────────────────────────────────────

    def is_trained(self):
        """
        Returns True if the agent is ready to play.

        For agents that need no training (A*, MCTS, Grover,  Human):
            always returns True.

        For agents that need training (Q-Learning):
            returns True only after training or loading a saved model.
        """
        return True   # default: no training needed

    def train(self, level_idx, episodes, epsilon, lr, gamma):
        """
        Train the agent on a given level.

        Only Q-Learning implements this.
        All other agents raise NotImplementedError if called.

        Args:
            level_idx : int   — 0=Easy, 1=Medium, 2=Hard, 3=Impossible
            episodes  : int   — number of training episodes
            epsilon   : float — initial exploration rate (0.0 - 1.0)
            lr        : float — learning rate (Q-Learning only)
            gamma     : float — discount factor (Q-Learning only)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training."
        )

    def save(self, path):
        """
        Save the trained model to a file.
        Only Q-Learning implements this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving."
        )

    def load(self, path):
        """
        Load a previously saved model from a file.
        Only Q-Learning implements this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading."
        )

    # ── Lifecycle hooks ───────────────────────────────────────────────────

    def reset(self):
        """
        Called when the level is reset (player presses R).
        Agents that build a solution path (A*) clear it here.
        Stateless agents (MCTS, Grover) do nothing.
        """
        pass

    def on_level_start(self, game):
        """
        Called once before the agent starts playing a level.
        A* and Grover compute their full solution path here.
        MCTS and Q-Learning do nothing (they decide move by move).

        Args:
            game : Sokoban instance at initial state
        """
        pass

    # ── String representation ─────────────────────────────────────────────

    def __str__(self):
        trained = "trained" if self.is_trained() else "not trained"
        return f"{self.name} ({trained})"

    def __repr__(self):
        return self.__str__()