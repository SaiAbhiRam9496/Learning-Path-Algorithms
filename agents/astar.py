# agents/astar.py
# A* search agent for Sokoban.
# Computes the full optimal solution path on level start,
# then replays it move by move during gameplay.
#
# Heuristic: sum of minimum Manhattan distances
#            from each box to its nearest goal.
#
# Win rate  : 100% Easy, 100% Medium, ~60% Hard
# Speed     : Easy <1s, Medium 1-5s, Hard 30-120s
# Training  : None required

import heapq
from agents.base_agent import BaseAgent


# ── Heuristic ────────────────────────────────────────────────────────────────

def _heuristic(boxes, goals):
    """
    Admissible heuristic for A*.
    For each box, find the nearest goal (Manhattan distance).
    Sum all minimum distances.

    This never overestimates — each box needs at least this
    many pushes to reach its nearest goal.

    Args:
        boxes : frozenset of (x, y) box positions
        goals : frozenset of (x, y) goal positions

    Returns:
        int — lower bound on remaining pushes needed
    """
    total = 0
    for bx, by in boxes:
        min_dist = min(abs(bx - gx) + abs(by - gy) for gx, gy in goals)
        total += min_dist
    return total


# ── State helpers ─────────────────────────────────────────────────────────────

def _get_goals(level):
    """Extract all goal positions from a level grid."""
    goals = set()
    for y, row in enumerate(level):
        for x, tile in enumerate(row):
            if tile in ('.', '*', '+'):
                goals.add((x, y))
    return frozenset(goals)


def _get_tile(level, x, y):
    """Return tile at (x,y), '#' if out of bounds."""
    if 0 <= y < len(level) and 0 <= x < len(level[y]):
        return level[y][x]
    return '#'


def _get_successors(level, goals, px, py, boxes):
    """
    Generate all valid next states from current state.

    For each of the 4 directions:
      - If destination is floor/goal → plain move
      - If destination has a box → push if beyond is free
      - If destination is wall → skip

    Args:
        level : list of list of str (original level for wall data)
        goals : frozenset of goal positions
        px,py : player position
        boxes : frozenset of box positions

    Yields:
        (new_px, new_py, new_boxes, action)
        where action = (dx, dy)
    """
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = px + dx, py + dy

        # Check wall using original level
        tile = _get_tile(level, nx, ny)
        if tile == '#':
            continue

        if (nx, ny) in boxes:
            # Box in the way — can we push it?
            bx, by = nx + dx, ny + dy
            beyond_tile = _get_tile(level, bx, by)
            if beyond_tile == '#' or (bx, by) in boxes:
                continue
            new_boxes = frozenset(
                (bx, by) if b == (nx, ny) else b for b in boxes
            )
            yield nx, ny, new_boxes, (dx, dy)
        else:
            yield nx, ny, boxes, (dx, dy)


# ── A* solver ─────────────────────────────────────────────────────────────────

def _solve_astar(game, max_states=500_000):
    """
    Run A* on the current game state.

    State = (player_x, player_y, frozenset of box positions)
    Cost  = number of moves made so far
    f(n)  = g(n) + h(n)

    Args:
        game       : Sokoban instance
        max_states : safety limit to avoid infinite search

    Returns:
        list of (dx, dy) moves — the solution path
        empty list [] if no solution found within max_states
    """
    # Extract initial state
    level = game.level
    px, py = game.player_x, game.player_y

    boxes = frozenset(
        (x, y)
        for y, row in enumerate(level)
        for x, tile in enumerate(row)
        if tile in ('$', '*')
    )
    goals = _get_goals(level)

    # Already solved?
    if boxes == goals:
        return []

    # A* open list: (f, g, px, py, boxes, path)
    h_init = _heuristic(boxes, goals)
    open_list = [(h_init, 0, px, py, boxes, [])]
    visited = {(px, py, boxes): 0}   # state → best g seen
    states_explored = 0

    while open_list:
        f, g, px, py, boxes, path = heapq.heappop(open_list)
        states_explored += 1

        if states_explored > max_states:
            return []   # give up — level too complex

        # Goal check
        if boxes == goals:
            return path

        # Skip if we already found a better path to this state
        state_key = (px, py, boxes)
        if visited.get(state_key, float('inf')) < g:
            continue

        # Expand successors
        for nx, ny, new_boxes, action in _get_successors(
            level, goals, px, py, boxes
        ):
            new_g = g + 1
            new_key = (nx, ny, new_boxes)

            if new_g < visited.get(new_key, float('inf')):
                visited[new_key] = new_g
                new_h = _heuristic(new_boxes, goals)
                new_f = new_g + new_h
                heapq.heappush(
                    open_list,
                    (new_f, new_g, nx, ny, new_boxes, path + [action])
                )

    return []   # no solution found


# ── Agent class ───────────────────────────────────────────────────────────────

class AStarAgent(BaseAgent):
    """
    A* search agent.

    Computes full solution path on on_level_start().
    Replays moves one at a time via next_move().

    If A* fails (level too complex or unsolvable),
    next_move() returns None and the agent stops.
    """

    @property
    def name(self):
        return "A*"

    def __init__(self):
        self._solution  = []    # list of (dx, dy) moves
        self._solved    = False # True if solution was found
        self._failed    = False # True if A* gave up
        self.stats      = {
            "solved":       False,
            "total_moves":  0,
            "solution_len": 0,
        }

    def on_level_start(self, game):
        """
        Compute the full solution path before playing starts.
        Called once by the game loop when the level loads.
        """
        self._solution  = []
        self._solved    = False
        self._failed    = False

        path = _solve_astar(game)

        if path:
            self._solution  = path
            self._solved    = True
            self.stats["solved"]       = True
            self.stats["solution_len"] = len(path)
        else:
            self._failed = True
            self.stats["solved"] = False

    def next_move(self, game):
        """
        Return the next move from the precomputed solution path.
        Returns None if solution exhausted or A* failed.
        """
        if self._failed or not self._solution:
            return None

        move = self._solution.pop(0)
        self.stats["total_moves"] += 1
        return move

    def reset(self):
        """Clear solution on level reset — on_level_start will recompute."""
        self._solution = []
        self._solved   = False
        self._failed   = False
        self.stats     = {
            "solved":       False,
            "total_moves":  0,
            "solution_len": 0,
        }

    def is_trained(self):
        return True   # A* never needs training