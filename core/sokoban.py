# core/sokoban.py
# Pure game logic — zero pygame dependencies.
# Safe for use by renderer (main.py) and RL agents.
#
# Tile legend:
#   ' ' = empty floor   '#' = wall
#   '@' = player        '+' = player on goal
#   '$' = box           '*' = box on goal
#   '.' = goal


class Sokoban:

    def __init__(self, level):
        """
        level: list of equal-width strings (from maps.load_level).
        Stores original for reset().
        """
        self.original_level = [list(row) for row in level]
        self.reset()

    # ── Setup ────────────────────────────────────────────────────────────

    def reset(self):
        """Restore level to initial state, clear all counters and history."""
        self.level  = [row[:] for row in self.original_level]
        self.height = len(self.level)
        self.width  = max(len(row) for row in self.level)

        # Pad all rows to equal width (safety — load_level already pads)
        for row in self.level:
            while len(row) < self.width:
                row.append(' ')

        # Locate player
        self.player_x = 0
        self.player_y = 0
        for y, row in enumerate(self.level):
            for x, tile in enumerate(row):
                if tile in ('@', '+'):
                    self.player_x = x
                    self.player_y = y

        self.history    = []  # list of (level_snapshot, px, py, push_count)
        self.move_count = 0
        self.push_count = 0

    # ── Private helpers ──────────────────────────────────────────────────

    def _get(self, x, y):
        """Return tile at (x,y). Returns '#' if out of bounds."""
        if 0 <= y < self.height and 0 <= x < len(self.level[y]):
            return self.level[y][x]
        return '#'

    def _set(self, x, y, tile):
        self.level[y][x] = tile

    def _snapshot(self):
        """
        Save full game state to history.
        Includes push_count so undo restores it correctly.
        FIX: original code did NOT save push_count — this caused
             push_count to become wrong after undo operations.
        """
        self.history.append((
            [row[:] for row in self.level],
            self.player_x,
            self.player_y,
            self.push_count,        # ← FIX: save push_count
        ))

    # ── Public API ───────────────────────────────────────────────────────

    def move(self, dx, dy):
        """
        Attempt to move player by (dx, dy).
        Returns True if the move was made, False if blocked.

        Move logic:
          1. If destination is wall/OOB → blocked
          2. If destination has a box:
               check cell beyond box — if not free → blocked
               otherwise push the box, snapshot first
          3. Plain move — snapshot, move player
        """
        nx, ny   = self.player_x + dx, self.player_y + dy
        target   = self._get(nx, ny)

        # Wall or out of bounds
        if target == '#':
            return False

        pushed = False

        if target in ('$', '*'):
            # Box in the way — can we push it?
            bx, by = nx + dx, ny + dy
            beyond = self._get(bx, by)

            if beyond not in (' ', '.'):
                return False        # box blocked

            self._snapshot()

            # Place box at new position
            # If landing on goal → '*', else plain box '$'
            self._set(bx, by, '*' if beyond == '.' else '$')

            # Restore tile under the box that just moved
            # If box was on goal ('*') → reveal goal ('.')
            # If box was on floor ('$') → leave floor (' ')
            self._set(nx, ny, '.' if target == '*' else ' ')

            pushed = True

        else:
            self._snapshot()        # plain move — still snapshot for undo

        # Vacate player's current tile
        cur = self._get(self.player_x, self.player_y)
        self._set(self.player_x, self.player_y,
                  '.' if cur == '+' else ' ')

        # Place player at destination
        # Read tile NOW — step above may have updated it (box pushed off goal)
        dest_now = self._get(nx, ny)
        self._set(nx, ny, '+' if dest_now == '.' else '@')

        self.player_x   = nx
        self.player_y   = ny
        self.move_count += 1
        if pushed:
            self.push_count += 1

        return True

    def undo(self):
        """
        Undo the last move.
        Returns True if successful, False if nothing to undo.

        FIX: restores push_count from snapshot (original code only
             decremented move_count, leaving push_count wrong).
        """
        if not self.history:
            return False

        snapshot, px, py, saved_push = self.history.pop()

        self.level      = snapshot
        self.player_x   = px
        self.player_y   = py
        self.move_count = max(0, self.move_count - 1)
        self.push_count = saved_push    # ← FIX: restore instead of decrement

        return True

    def is_completed(self):
        """
        True when every box is on a goal.
        No bare '$' tiles remain → all boxes placed on goals.
        This is correct because validate_level ensures boxes == goals,
        so if no '$' exists then all boxes must be on goals ('*').
        """
        return not any('$' in row for row in self.level)

    # ── Stats ─────────────────────────────────────────────────────────────

    def count_boxes_on_goals(self):
        """Number of boxes currently sitting on goals."""
        return sum(row.count('*') for row in self.level)

    def count_remaining_boxes(self):
        """Number of boxes NOT yet on a goal."""
        return sum(row.count('$') for row in self.level)

    # ── RL interface ──────────────────────────────────────────────────────

    def get_state(self):
        """
        Hashable state tuple for tabular RL or visited-set checks.
        Returns: ((player_x, player_y), frozenset_of_box_positions)
        """
        boxes = frozenset(
            (x, y)
            for y, row in enumerate(self.level)
            for x, tile in enumerate(row)
            if tile in ('$', '*')
        )
        return (self.player_x, self.player_y), boxes

    def get_legal_actions(self):
        """
        Returns list of (dx, dy) moves that are not immediately blocked.
        Used by RL agents to avoid wasting steps on walls.
        Does NOT check deadlocks — only immediate legality.
        """
        actions = []
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = self.player_x + dx, self.player_y + dy
            target = self._get(nx, ny)
            if target == '#':
                continue
            if target in ('$', '*'):
                bx, by = nx + dx, ny + dy
                beyond = self._get(bx, by)
                if beyond not in (' ', '.'):
                    continue
            actions.append((dx, dy))
        return actions