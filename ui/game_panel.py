# ui/game_panel.py
# Game selection screen and in-game agent overlay.
# No game logic. Pure pygame drawing.
#
# Contains:
#   draw_game_select()   — level + agent selection before playing
#   draw_in_game_hud()   — agent info bar + dropdown during gameplay
#   draw_post_game()     — win/loss result screen after game ends
#
# Matches existing screens.py / training_panel.py style exactly.

import pygame


# ── Constants ─────────────────────────────────────────────────────────────────

LEVEL_NAMES  = ["Easy", "Medium", "Hard", "Impossible"]
AGENT_NAMES  = ["Human", "A*", "Q-Learning", "MCTS", "Grover"]

DIFF_KEY = {
    0: "EASY",
    1: "MEDIUM",
    2: "HARD",
    3: "IMPOSSIBLE",
}


# ── Game selection screen ─────────────────────────────────────────────────────

def draw_game_select(screen, selected_level, selected_agent,
                     qlearning_trained, levels,
                     fonts, colors, window_width, window_height):
    """
    Draw the game selection screen.

    Left panel  : level selector (4 buttons)
    Right panel : agent selector (5 buttons) + minimap preview

    Q-Learning button is greyed out if no model saved yet.

    Args:
        screen           : pygame.Surface
        selected_level   : int 0-3
        selected_agent   : int 0-4  (index into AGENT_NAMES)
        qlearning_trained: bool — True if models/qlearning_*.json exists
        levels           : list of raw level grids (for minimap)
        fonts, colors, window_width, window_height

    Returns:
        dict of rects:
          "level_btns" : list of 4 Rects
          "agent_btns" : list of 5 Rects
          "play_btn"   : Rect
    """
    screen.fill(colors["bg"])
    cx = window_width  // 2

    diff_color_map = {
        0: colors["easy"],
        1: colors["medium"],
        2: colors["hard"],
        3: colors["impossible"],
    }

    # ── Title ──────────────────────────────────────────────────────────────
    title = fonts["big"].render("SELECT LEVEL & AGENT", True, colors["accent"])
    screen.blit(title, (cx - title.get_width() // 2, 28))

    # ── Layout: two columns ────────────────────────────────────────────────
    left_x  = 40
    left_w  = window_width // 2 - 60
    right_x = window_width // 2 + 20
    right_w = window_width // 2 - 60

    # ── Left: Level selector ───────────────────────────────────────────────
    lbl = fonts["small"].render("LEVEL", True, colors["grey"])
    screen.blit(lbl, (left_x, 85))

    lbtn_w = left_w
    lbtn_h = 58
    lbtn_gap = 12
    level_rects = []

    for i, name in enumerate(LEVEL_NAMES):
        by   = 115 + i * (lbtn_h + lbtn_gap)
        rect = pygame.Rect(left_x, by, lbtn_w, lbtn_h)
        level_rects.append(rect)

        dcol = diff_color_map[i]
        bg   = colors["btn_hov"] if i == selected_level else colors["btn"]

        pygame.draw.rect(screen, bg,   rect, border_radius=8)
        pygame.draw.rect(screen, dcol, rect, 2, border_radius=8)

        ns = fonts["med"].render(name, True, colors["white"])
        ds = fonts["small"].render(DIFF_KEY[i], True, dcol)

        screen.blit(ns, (rect.x + 16,
                         rect.y + lbtn_h // 2 - ns.get_height() // 2 - 8))
        screen.blit(ds, (rect.x + 16,
                         rect.y + lbtn_h // 2 + 4))

    # ── Right: Agent selector ──────────────────────────────────────────────
    albl = fonts["small"].render("AGENT", True, colors["grey"])
    screen.blit(albl, (right_x, 85))

    abtn_w   = right_w
    abtn_h   = 48
    abtn_gap = 10
    agent_rects = []

    agent_colors = {
        "Human":     colors["white"],
        "A*":        colors["easy"],
        "Q-Learning":colors["medium"],
        "MCTS":      colors["hard"],
        "Grover":    (180, 100, 255),   # quantum purple
    }

    agent_descs = {
        "Human":     "You play with arrow keys",
        "A*":        "Optimal path search · 100% win rate",
        "Q-Learning":"Trained RL agent · needs model file",
        "MCTS":      "Monte Carlo tree search · no training",
        "Grover":    "Quantum search · Grover's algorithm",
    }

    for i, name in enumerate(AGENT_NAMES):
        by      = 115 + i * (abtn_h + abtn_gap)
        rect    = pygame.Rect(right_x, by, abtn_w, abtn_h)
        agent_rects.append(rect)

        # Q-Learning greyed out if not trained
        is_ql_unavail = (name == "Q-Learning" and not qlearning_trained)
        acol  = agent_colors.get(name, colors["white"])
        if is_ql_unavail:
            acol = (60, 60, 60)

        bg = colors["btn_hov"] if i == selected_agent else colors["btn"]
        pygame.draw.rect(screen, bg,   rect, border_radius=8)
        pygame.draw.rect(screen, acol, rect, 2, border_radius=8)

        ns = fonts["small"].render(name, True,
                                   acol if i != selected_agent
                                   else colors["white"])
        screen.blit(ns, (rect.x + 14,
                         rect.y + abtn_h // 2 - ns.get_height() // 2 - 7))

        desc_text = agent_descs.get(name, "")
        if is_ql_unavail:
            desc_text = "No model found · train first"
        ds = fonts["small"].render(desc_text, True, (100, 100, 120))
        screen.blit(ds, (rect.x + 14,
                         rect.y + abtn_h // 2 + 5))

        # Lock icon for unavailable
        if is_ql_unavail:
            lock = fonts["small"].render("🔒", True, (80, 80, 80))
            screen.blit(lock, (rect.x + rect.width - 30, rect.y + 14))

    # ── PLAY button ────────────────────────────────────────────────────────
    can_play  = not (AGENT_NAMES[selected_agent] == "Q-Learning"
                     and not qlearning_trained)
    play_w    = 200
    play_h    = 52
    play_rect = pygame.Rect(cx - play_w // 2,
                            window_height - 80,
                            play_w, play_h)

    play_col = colors["accent"] if can_play else (50, 50, 60)
    pygame.draw.rect(screen, play_col, play_rect, border_radius=10)

    pt = fonts["med"].render("PLAY", True,
                             colors["bg"] if can_play else (80, 80, 90))
    screen.blit(pt, (
        play_rect.x + play_rect.width  // 2 - pt.get_width()  // 2,
        play_rect.y + play_rect.height // 2 - pt.get_height() // 2,
    ))

    # ── Hints ──────────────────────────────────────────────────────────────
    hint = fonts["small"].render("ESC = menu", True, colors["grey"])
    screen.blit(hint, (cx - hint.get_width() // 2, window_height - 28))

    return {
        "level_btns": level_rects,
        "agent_btns": agent_rects,
        "play_btn":   play_rect,
    }


# ── Dropdown widget ───────────────────────────────────────────────────────────

class AgentDropdown:
    """
    A simple dropdown for switching agents mid-game.
    Rendered on top of the game grid.

    Usage:
        dd = AgentDropdown(x, y, width)
        dd.draw(screen, current_agent_idx, qlearning_trained, fonts, colors)
        clicked = dd.handle_click(pos, current_agent_idx, qlearning_trained)
        # clicked returns new agent index or None
    """

    def __init__(self, x, y, width=160):
        self.x      = x
        self.y      = y
        self.width  = width
        self.open   = False
        self._item_h = 30

    def handle_click(self, pos, current_idx, qlearning_trained):
        """
        Returns new agent index if user clicked an option,
        None if click was outside or toggled closed.
        """
        header_rect = pygame.Rect(self.x, self.y, self.width, 30)

        if header_rect.collidepoint(pos):
            self.open = not self.open
            return None

        if self.open:
            for i, name in enumerate(AGENT_NAMES):
                item_rect = pygame.Rect(
                    self.x,
                    self.y + 30 + i * self._item_h,
                    self.width,
                    self._item_h,
                )
                if item_rect.collidepoint(pos):
                    self.open = False
                    if name == "Q-Learning" and not qlearning_trained:
                        return None   # locked
                    if i == current_idx:
                        return None   # already selected
                    return i

            # Click outside dropdown closes it
            self.open = False

        return None

    def draw(self, screen, current_idx, qlearning_trained, fonts, colors):
        agent_name = AGENT_NAMES[current_idx]

        agent_col_map = {
            "Human":     colors["white"],
            "A*":        colors["easy"],
            "Q-Learning":colors["medium"],
            "MCTS":      colors["hard"],
            "Grover":    (180, 100, 255),
        }
        cur_col = agent_col_map.get(agent_name, colors["white"])

        # ── Header (always visible) ────────────────────────────────────────
        header = pygame.Rect(self.x, self.y, self.width, 30)
        pygame.draw.rect(screen, (25, 25, 35), header, border_radius=6)
        pygame.draw.rect(screen, cur_col, header, 1, border_radius=6)

        name_s = fonts["small"].render(agent_name, True, cur_col)
        arrow  = fonts["small"].render("▼" if not self.open else "▲",
                                       True, colors["grey"])
        screen.blit(name_s, (header.x + 8, header.y + 7))
        screen.blit(arrow,  (header.x + self.width - 20, header.y + 7))

        # ── Dropdown list ──────────────────────────────────────────────────
        if self.open:
            for i, name in enumerate(AGENT_NAMES):
                item_rect = pygame.Rect(
                    self.x,
                    self.y + 30 + i * self._item_h,
                    self.width,
                    self._item_h,
                )
                is_locked = (name == "Q-Learning" and not qlearning_trained)
                is_cur    = (i == current_idx)

                bg = (35, 35, 50) if is_cur else (20, 20, 30)
                pygame.draw.rect(screen, bg, item_rect)
                pygame.draw.rect(screen, (50, 50, 70), item_rect, 1)

                col = (60, 60, 60) if is_locked else \
                      agent_col_map.get(name, colors["white"])
                ns  = fonts["small"].render(name, True, col)
                screen.blit(ns, (item_rect.x + 8, item_rect.y + 7))

                if is_locked:
                    lk = fonts["small"].render("🔒", True, (60, 60, 60))
                    screen.blit(lk, (item_rect.x + self.width - 22,
                                     item_rect.y + 6))


# ── In-game HUD overlay ───────────────────────────────────────────────────────

def draw_in_game_hud(screen, game, level_idx, agent_name,
                     dropdown, qlearning_trained,
                     fonts, colors, window_width, hud_height=48):
    """
    Draw the in-game HUD bar at the top of the screen.

    Shows: level name | agent name | moves | pushes | boxes left
    Also renders the agent dropdown widget.

    Args:
        screen            : pygame.Surface
        game              : Sokoban instance
        level_idx         : int
        agent_name        : str (current agent name)
        dropdown          : AgentDropdown instance
        qlearning_trained : bool
        fonts, colors, window_width, hud_height
    """
    diff_color_map = {
        0: colors["easy"],
        1: colors["medium"],
        2: colors["hard"],
        3: colors["impossible"],
    }
    dcol = diff_color_map.get(level_idx, colors["white"])

    # HUD background strip
    hud_rect = pygame.Rect(0, 0, window_width, hud_height)
    pygame.draw.rect(screen, (18, 18, 26), hud_rect)
    pygame.draw.line(screen, (40, 40, 60),
                     (0, hud_height - 1), (window_width, hud_height - 1))

    level_name = LEVEL_NAMES[level_idx]

    # Level badge
    lbadge = fonts["small"].render(
        f"{level_name}  [{DIFF_KEY[level_idx]}]", True, dcol
    )
    screen.blit(lbadge, (12, hud_height // 2 - lbadge.get_height() // 2))

    # Stats: moves | pushes | boxes left
    remaining = game.count_remaining_boxes()
    stats_str = (
        f"Moves: {game.move_count}"
        f"   Pushes: {game.push_count}"
        f"   Boxes left: {remaining}"
    )
    stats_s = fonts["small"].render(stats_str, True, colors["white"])
    screen.blit(stats_s, (
        window_width // 2 - stats_s.get_width() // 2,
        hud_height // 2 - stats_s.get_height() // 2,
    ))

    # Agent dropdown (top-right)
    dd_x = window_width - dropdown.width - 10

    # "Switch:" label
    sw = fonts["small"].render("Switch:", True, colors["grey"])
    screen.blit(sw, (dd_x - sw.get_width() - 6,
                     hud_height // 2 - sw.get_height() // 2))

    dropdown.x = dd_x
    dropdown.y = hud_height // 2 - 15

    # Find current agent index
    agent_idx = AGENT_NAMES.index(agent_name) if agent_name in AGENT_NAMES else 0
    dropdown.draw(screen, agent_idx, qlearning_trained, fonts, colors)


# ── Post-game result screen ───────────────────────────────────────────────────

def draw_post_game(screen, game, level_idx, agent_name,
                   time_elapsed, won,
                   fonts, colors, window_width, window_height,
                   grover_explanation=""):
    """
    Draw the post-game result screen (win or loss).

    Shows: result header | agent | level | stats | optional quantum info
    Buttons: MENU | RETRY | NEXT LEVEL (if won)

    Args:
        screen            : pygame.Surface
        game              : Sokoban instance
        level_idx         : int
        agent_name        : str
        time_elapsed      : float seconds
        won               : bool
        fonts, colors, window_width, window_height
        grover_explanation: str (shown if agent was Grover)

    Returns:
        dict of rects:
          "menu_btn"  : Rect
          "retry_btn" : Rect
          "next_btn"  : Rect (None if last level or loss)
    """
    screen.fill((10, 10, 15))
    cx = window_width  // 2
    cy = window_height // 2

    diff_color_map = {
        0: colors["easy"],
        1: colors["medium"],
        2: colors["hard"],
        3: colors["impossible"],
    }
    dcol = diff_color_map.get(level_idx, colors["white"])

    # ── Header ─────────────────────────────────────────────────────────────
    if won:
        hdr_text = "✓  LEVEL COMPLETE!"
        hdr_col  = colors["green"]
    else:
        hdr_text = "✗  LEVEL FAILED"
        hdr_col  = colors["hard"]

    hdr = fonts["big"].render(hdr_text, True, hdr_col)
    screen.blit(hdr, (cx - hdr.get_width() // 2, 36))

    # Agent + level subtitle
    sub = fonts["med"].render(
        f"{agent_name}  ·  {LEVEL_NAMES[level_idx]}", True, colors["accent"]
    )
    screen.blit(sub, (cx - sub.get_width() // 2, 84))

    # ── Stats panel ────────────────────────────────────────────────────────
    panel_w = 380
    panel_h = 200
    panel   = pygame.Rect(cx - panel_w // 2, 124, panel_w, panel_h)
    pygame.draw.rect(screen, colors["panel"], panel, border_radius=10)
    pygame.draw.rect(screen, dcol, panel, 1, border_radius=10)

    mins = int(time_elapsed) // 60
    secs = int(time_elapsed)  % 60
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    rows = [
        ("Moves",       str(game.move_count),   colors["white"]),
        ("Pushes",      str(game.push_count),   colors["white"]),
        ("Boxes left",  str(game.count_remaining_boxes()),
                        colors["green"] if won else colors["hard"]),
        ("Time",        time_str,               colors["white"]),
    ]

    row_h = panel_h // (len(rows) + 1)
    for i, (label, value, vcol) in enumerate(rows):
        y = panel.y + (i + 1) * row_h - row_h // 2

        if i > 0:
            pygame.draw.line(screen, (40, 40, 60),
                             (panel.x + 16, y - row_h // 2 + 4),
                             (panel.x + panel_w - 16, y - row_h // 2 + 4))

        ls = fonts["small"].render(label, True, colors["grey"])
        vs = fonts["small"].render(value, True, vcol)
        screen.blit(ls, (panel.x + 20,  y))
        screen.blit(vs, (panel.x + panel_w - 20 - vs.get_width(), y))

    # ── Grover quantum info ────────────────────────────────────────────────
    if agent_name == "Grover" and grover_explanation:
        qy      = panel.y + panel_h + 14
        q_label = fonts["small"].render(
            "Quantum details (last move):", True, (180, 100, 255)
        )
        screen.blit(q_label, (cx - q_label.get_width() // 2, qy))

        lines = grover_explanation.split('\n')[:4]   # show max 4 lines
        for j, line in enumerate(lines):
            ls = fonts["small"].render(line.strip(), True, (140, 140, 180))
            screen.blit(ls, (cx - ls.get_width() // 2, qy + 22 + j * 20))

    # ── Buttons ────────────────────────────────────────────────────────────
    btn_w = 150
    btn_h = 46
    gap   = 16
    has_next = won and level_idx < len(LEVEL_NAMES) - 1

    if has_next:
        total_btn_w = 3 * btn_w + 2 * gap
        bx_start    = cx - total_btn_w // 2
    else:
        total_btn_w = 2 * btn_w + gap
        bx_start    = cx - total_btn_w // 2

    by = window_height - 80

    menu_rect  = pygame.Rect(bx_start, by, btn_w, btn_h)
    retry_rect = pygame.Rect(bx_start + btn_w + gap, by, btn_w, btn_h)
    next_rect  = None

    if has_next:
        next_rect = pygame.Rect(bx_start + 2 * (btn_w + gap), by, btn_w, btn_h)

    # Draw buttons
    for rect, label, bg, fg in [
        (menu_rect,  "MENU",       colors["btn"],    colors["white"]),
        (retry_rect, "RETRY",      colors["btn"],    colors["white"]),
    ]:
        pygame.draw.rect(screen, bg, rect, border_radius=8)
        pygame.draw.rect(screen, colors["grey"], rect, 1, border_radius=8)
        s = fonts["med"].render(label, True, fg)
        screen.blit(s, (rect.x + rect.width  // 2 - s.get_width()  // 2,
                        rect.y + rect.height // 2 - s.get_height() // 2))

    if next_rect:
        pygame.draw.rect(screen, colors["accent"], next_rect, border_radius=8)
        s = fonts["med"].render("NEXT", True, colors["bg"])
        screen.blit(s, (next_rect.x + next_rect.width  // 2 - s.get_width()  // 2,
                        next_rect.y + next_rect.height // 2 - s.get_height() // 2))

    # ── Keyboard hints ─────────────────────────────────────────────────────
    hint = fonts["small"].render(
        "ESC = menu   R = retry" + ("   N = next" if has_next else ""),
        True, colors["grey"]
    )
    screen.blit(hint, (cx - hint.get_width() // 2, window_height - 28))

    return {
        "menu_btn":  menu_rect,
        "retry_btn": retry_rect,
        "next_btn":  next_rect,
    }