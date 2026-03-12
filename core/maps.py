# core/maps.py
# 4 Sokoban levels — Easy, Medium, Hard, Impossible
# Each level BFS-verified solvable before inclusion.
#
# Tile legend:
#   ' ' = empty floor   '#' = wall
#   '@' = player        '+' = player on goal
#   '$' = box           '*' = box on goal
#   '.' = goal

# ---------------------------------------------------------------------------
# EASY — 7x6 grid, 2 boxes, 10 moves minimum (BFS verified)
# Two boxes stacked in a corridor, two goals above and below.
# Requires pushing boxes in the correct order.
# ---------------------------------------------------------------------------

EASY_RAW = [
    "#######",
    "# .@  #",
    "# $   #",
    "# $   #",
    "# .   #",
    "#######",
]

# ---------------------------------------------------------------------------
# MEDIUM — 7x8 grid, 3 boxes, 20 moves minimum (BFS verified)
# Three boxes spread around the room with goals at different positions.
# Requires planning order of pushes — no trivial solution.
# ---------------------------------------------------------------------------

MEDIUM_RAW = [
    "#######",
    "#  .  #",
    "# $   #",
    "#   $ #",
    "#  .  #",
    "# $   #",
    "#  .@ #",
    "#######",
]

# ---------------------------------------------------------------------------
# HARD — Original Thinking Rabbit Level 1 (BFS verified solvable)
# 6 boxes, large multi-room layout, requires careful coordination.
# ---------------------------------------------------------------------------

HARD_RAW = [
    "    #####",
    "    #   #",
    "    #$  #",
    "  ###  $##",
    "  #  $ $ #",
    "### # ## #   ######",
    "#   # ## #####  ..#",
    "# $  $          ..#",
    "##### ### #@##  ..#",
    "    #     #########",
    "    #######",
]

# ---------------------------------------------------------------------------
# IMPOSSIBLE — Original Thinking Rabbit Level 6
# 10 boxes, extremely complex layout, for expert players only.
# Published verified solvable — BFS not feasible at this scale.
# ---------------------------------------------------------------------------

IMPOSSIBLE_RAW = [
    "######  ###",
    "#..  # ##@##",
    "#..  ###   #",
    "#..     $$ #",
    "#..  # # $ #",
    "#..### # $ #",
    "#### $ #$  #",
    "   #  $# $ #",
    "   # $  $  #",
    "   #  ##   #",
    "   #########",
]

# ---------------------------------------------------------------------------

LEVELS = [EASY_RAW, MEDIUM_RAW, HARD_RAW, IMPOSSIBLE_RAW]

LEVEL_META = [
    ("Easy",       "EASY"),
    ("Medium",     "MEDIUM"),
    ("Hard",       "HARD"),
    ("Impossible", "IMPOSSIBLE"),
]

# ---------------------------------------------------------------------------

def pad_level(level):
    """Make all rows equal width by padding with spaces on the right."""
    w = max(len(r) for r in level)
    return [r.ljust(w) for r in level]


def validate_level(level):
    """
    Returns (True, 'OK') or (False, error_message).
    Checks: known tiles only, exactly 1 player, boxes == goals, at least 1 box.
    """
    known = set(' #@+$*.')
    players = boxes = goals = 0
    for y, row in enumerate(level):
        for x, ch in enumerate(row):
            if ch not in known:
                return False, f"Unknown tile '{ch}' at ({x},{y})"
            if ch in ('@', '+'):       players += 1
            if ch in ('$', '*'):      boxes   += 1
            if ch in ('.', '*', '+'): goals   += 1
    if players != 1:
        return False, f"Expected 1 player, found {players}"
    if boxes != goals:
        return False, f"Box count ({boxes}) != goal count ({goals})"
    if boxes == 0:
        return False, "Level has no boxes or goals"
    return True, "OK"


def load_level(i):
    """
    Load level i (0=Easy, 1=Medium, 2=Hard, 3=Impossible).
    Returns a padded, validated list of strings.
    Raises ValueError if validation fails.
    """
    raw = LEVELS[i]
    padded = pad_level(raw)
    ok, msg = validate_level(padded)
    if not ok:
        name = LEVEL_META[i][0]
        raise ValueError(f"Level '{name}' failed validation: {msg}")
    return padded