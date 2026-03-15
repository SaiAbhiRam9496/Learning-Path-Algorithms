# ui/training_panel.py
# Training setup screen and training result screen.
# No game logic. Pure pygame drawing.
# Training itself runs in terminal — this file handles:
#   1. draw_training_setup() — parameter selection UI before training
#   2. draw_training_result() — metrics display after training completes
#
# Matches existing screens.py style exactly.

import pygame


# ── Input box helper ──────────────────────────────────────────────────────────

class InputBox:
    """
    A simple single-line text input box for pygame.
    Supports typing, backspace, and active/inactive visual state.
    """

    def __init__(self, rect, default="", label="", numeric=True):
        self.rect    = pygame.Rect(rect)
        self.text    = str(default)
        self.label   = label
        self.numeric = numeric
        self.active  = False

    def handle_event(self, event, colors):
        """
        Process a pygame event.
        Returns True if ENTER was pressed (user confirmed input).
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                return True
            else:
                ch = event.unicode
                if self.numeric:
                    # Allow digits and one decimal point
                    if ch.isdigit() or (ch == '.' and '.' not in self.text):
                        if len(self.text) < 10:
                            self.text += ch
                else:
                    if len(self.text) < 20:
                        self.text += ch
        return False

    def get_float(self, default=0.0):
        try:
            return float(self.text)
        except ValueError:
            return default

    def get_int(self, default=0):
        try:
            return int(float(self.text))
        except ValueError:
            return default

    def draw(self, screen, fonts, colors):
        # Label above box
        if self.label:
            lbl = fonts["small"].render(self.label, True, colors["grey"])
            screen.blit(lbl, (self.rect.x, self.rect.y - 22))

        # Box background
        bg = colors["btn_hov"] if self.active else colors["btn"]
        pygame.draw.rect(screen, bg, self.rect, border_radius=6)

        border_col = colors["accent"] if self.active else (60, 60, 80)
        pygame.draw.rect(screen, border_col, self.rect, 2, border_radius=6)

        # Text
        txt_surf = fonts["small"].render(self.text, True, colors["white"])
        screen.blit(txt_surf, (
            self.rect.x + 10,
            self.rect.y + self.rect.height // 2 - txt_surf.get_height() // 2
        ))

        # Cursor blink (simple: always shown when active)
        if self.active:
            cx = self.rect.x + 10 + txt_surf.get_width() + 2
            cy = self.rect.y + 8
            pygame.draw.line(screen, colors["white"],
                             (cx, cy), (cx, self.rect.y + self.rect.height - 8), 2)


# ── Training setup screen ─────────────────────────────────────────────────────

# Algorithms that require training (others always available)
TRAINABLE_AGENTS = ["Q-Learning"]

# All agents in display order
ALL_AGENTS = ["Q-Learning"]   # only Q-Learning needs training panel

# Level options
LEVEL_NAMES = ["Easy", "Medium", "Hard", "Impossible"]


def make_training_inputs(window_width, window_height):
    """
    Create all InputBox instances for the training panel.
    Called once by main.py when entering training_setup state.

    Returns:
        dict of InputBox objects keyed by param name
    """
    cx = window_width  // 2
    cy = window_height // 2
    w  = 200
    h  = 40

    return {
        "episodes": InputBox((cx - w // 2, cy - 80, w, h),
                             default="10000", label="Episodes"),
        "epsilon":  InputBox((cx - w // 2, cy,      w, h),
                             default="1.0",   label="Epsilon (exploration)"),
        "lr":       InputBox((cx - w // 2, cy + 80, w, h),
                             default="0.1",   label="Learning Rate"),
        "gamma":    InputBox((cx - w // 2, cy + 160, w, h),
                             default="0.95",  label="Discount Factor (gamma)"),
    }


def draw_training_setup(screen, inputs, selected_level,
                        fonts, colors, window_width, window_height):
    """
    Draw the training setup screen.

    Shows:
      - Title
      - Level selector (4 buttons: Easy / Medium / Hard / Impossible)
      - 4 input boxes: episodes, epsilon, lr, gamma
      - TRAIN button
      - Note: Hard/Impossible will run but never converge

    Args:
        screen         : pygame.Surface
        inputs         : dict of InputBox from make_training_inputs()
        selected_level : int (0-3)
        fonts, colors, window_width, window_height

    Returns:
        dict of rects for hit-testing:
          "level_btns" : list of 4 Rects
          "train_btn"  : Rect
    """
    screen.fill(colors["bg"])
    cx = window_width  // 2
    cy = window_height // 2

    diff_color_map = {
        0: colors["easy"],
        1: colors["medium"],
        2: colors["hard"],
        3: colors["impossible"],
    }

    # ── Title ──────────────────────────────────────────────────────────────
    title = fonts["big"].render("TRAINING SETUP", True, colors["accent"])
    screen.blit(title, (cx - title.get_width() // 2, 30))

    sub = fonts["small"].render(
        "Q-Learning  ·  terminal shows live progress",
        True, colors["grey"]
    )
    screen.blit(sub, (cx - sub.get_width() // 2, 78))

    # ── Level selector ─────────────────────────────────────────────────────
    lbl = fonts["small"].render("Select Level:", True, colors["grey"])
    screen.blit(lbl, (cx - 210, cy - 175))

    btn_w   = 100
    btn_h   = 38
    btn_gap = 10
    total_w = 4 * btn_w + 3 * btn_gap
    bx_start = cx - total_w // 2
    by       = cy - 150

    level_rects = []
    for i, name in enumerate(LEVEL_NAMES):
        bx       = bx_start + i * (btn_w + btn_gap)
        rect     = pygame.Rect(bx, by, btn_w, btn_h)
        level_rects.append(rect)
        dcol     = diff_color_map[i]
        bg       = colors["btn_hov"] if i == selected_level else colors["btn"]

        pygame.draw.rect(screen, bg,   rect, border_radius=6)
        pygame.draw.rect(screen, dcol, rect, 2, border_radius=6)

        ns = fonts["small"].render(name, True,
                                   colors["white"] if i == selected_level
                                   else colors["grey"])
        screen.blit(ns, (
            rect.x + rect.width  // 2 - ns.get_width()  // 2,
            rect.y + rect.height // 2 - ns.get_height() // 2,
        ))

    # Hard / Impossible warning
    if selected_level >= 2:
        warn = fonts["small"].render(
            "⚠  Hard/Impossible will run but never converge — for demo only",
            True, colors["hard"]
        )
        screen.blit(warn, (cx - warn.get_width() // 2, by + btn_h + 8))

    # ── Input boxes ────────────────────────────────────────────────────────
    for box in inputs.values():
        box.draw(screen, fonts, colors)

    # ── Hyperparameter descriptions ────────────────────────────────────────
    descs = [
        ("Episodes",     "How many games the agent plays while learning"),
        ("Epsilon",      "1.0 = full explore → 0.0 = full exploit"),
        ("Learn Rate",   "How fast Q-values update (0.01 - 0.5)"),
        ("Gamma",        "How much future rewards matter (0.9 - 0.99)"),
    ]
    desc_x = cx + 120
    desc_y = cy - 80
    for i, (param, desc) in enumerate(descs):
        p_surf = fonts["small"].render(param + ":", True, colors["accent"])
        d_surf = fonts["small"].render(desc,        True, colors["grey"])
        screen.blit(p_surf, (desc_x, desc_y + i * 80))
        screen.blit(d_surf, (desc_x, desc_y + i * 80 + 22))

    # ── TRAIN button ───────────────────────────────────────────────────────
    train_w    = 200
    train_h    = 50
    train_rect = pygame.Rect(cx - train_w // 2,
                             window_height - 90,
                             train_w, train_h)

    pygame.draw.rect(screen, colors["accent"], train_rect, border_radius=8)
    t_surf = fonts["med"].render("START TRAINING", True, colors["bg"])
    screen.blit(t_surf, (
        train_rect.x + train_rect.width  // 2 - t_surf.get_width()  // 2,
        train_rect.y + train_rect.height // 2 - t_surf.get_height() // 2,
    ))

    # ── Back hint ──────────────────────────────────────────────────────────
    back = fonts["small"].render("ESC = back to menu", True, colors["grey"])
    screen.blit(back, (cx - back.get_width() // 2, window_height - 30))

    return {
        "level_btns": level_rects,
        "train_btn":  train_rect,
    }


# ── Training result screen ────────────────────────────────────────────────────

def draw_training_result(screen, stats, fonts, colors,
                         window_width, window_height):
    """
    Draw the training result screen shown after training completes.
    Displays all key metrics from the training run.

    Args:
        screen : pygame.Surface
        stats  : dict from QLearningAgent.train_stats
                 keys: episodes, win_rate, avg_steps, best_steps,
                       final_epsilon, train_time_sec, level_name
        fonts, colors, window_width, window_height

    Returns:
        dict of rects:
          "play_btn" : Rect — play now with trained agent
          "menu_btn" : Rect — back to main menu
    """
    screen.fill((10, 10, 15))
    cx = window_width  // 2
    cy = window_height // 2

    # ── Header ─────────────────────────────────────────────────────────────
    hdr = fonts["big"].render("TRAINING COMPLETE", True, colors["green"])
    screen.blit(hdr, (cx - hdr.get_width() // 2, 40))

    level_name = stats.get("level_name", "?")
    sub = fonts["med"].render(
        f"Q-Learning  ·  {level_name}", True, colors["accent"]
    )
    screen.blit(sub, (cx - sub.get_width() // 2, 90))

    # ── Metrics panel ──────────────────────────────────────────────────────
    panel_w = 420
    panel_h = 280
    panel   = pygame.Rect(cx - panel_w // 2, 140, panel_w, panel_h)
    pygame.draw.rect(screen, colors["panel"], panel, border_radius=10)
    pygame.draw.rect(screen, colors["accent"], panel, 1, border_radius=10)

    win_rate   = stats.get("win_rate",       0.0)
    episodes   = stats.get("episodes",       0)
    avg_steps  = stats.get("avg_steps",      0.0)
    best_steps = stats.get("best_steps",     0)
    epsilon    = stats.get("final_epsilon",  0.0)
    t_sec      = stats.get("train_time_sec", 0.0)

    # Win rate colour: green if good, orange if ok, red if poor
    if win_rate >= 70:
        wr_col = colors["green"]
    elif win_rate >= 40:
        wr_col = colors["medium"]
    else:
        wr_col = colors["hard"]

    # Format time
    mins = int(t_sec) // 60
    secs = int(t_sec)  % 60
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    rows = [
        ("Episodes Trained",  f"{episodes:,}",          colors["white"]),
        ("Win Rate",          f"{win_rate:.2f}%",        wr_col),
        ("Avg Steps/Episode", f"{avg_steps:.1f}",        colors["white"]),
        ("Best Run (steps)",  f"{best_steps}",           colors["green"]),
        ("Final Epsilon",     f"{epsilon:.4f}",          colors["grey"]),
        ("Training Time",     time_str,                  colors["white"]),
    ]

    row_h  = panel_h // (len(rows) + 1)
    lbl_x  = panel.x + 24
    val_x  = panel.x + panel_w - 24

    for i, (label, value, vcol) in enumerate(rows):
        y = panel.y + (i + 1) * row_h - row_h // 2

        # Separator line (except last)
        if i > 0:
            pygame.draw.line(screen, (40, 40, 60),
                             (panel.x + 16, y - row_h // 2 + 4),
                             (panel.x + panel_w - 16, y - row_h // 2 + 4))

        lbl_s = fonts["small"].render(label, True, colors["grey"])
        val_s = fonts["small"].render(value, True, vcol)

        screen.blit(lbl_s, (lbl_x, y))
        screen.blit(val_s, (val_x - val_s.get_width(), y))

    # ── Buttons ────────────────────────────────────────────────────────────
    btn_w = 180
    btn_h = 48
    gap   = 20
    by    = panel.y + panel_h + 30

    play_rect = pygame.Rect(cx - btn_w - gap // 2, by, btn_w, btn_h)
    menu_rect = pygame.Rect(cx + gap // 2,          by, btn_w, btn_h)

    # Play Now
    pygame.draw.rect(screen, colors["accent"], play_rect, border_radius=8)
    ps = fonts["med"].render("PLAY NOW", True, colors["bg"])
    screen.blit(ps, (
        play_rect.x + play_rect.width  // 2 - ps.get_width()  // 2,
        play_rect.y + play_rect.height // 2 - ps.get_height() // 2,
    ))

    # Menu
    pygame.draw.rect(screen, colors["btn"], menu_rect, border_radius=8)
    pygame.draw.rect(screen, colors["grey"], menu_rect, 1, border_radius=8)
    ms = fonts["med"].render("MENU", True, colors["white"])
    screen.blit(ms, (
        menu_rect.x + menu_rect.width  // 2 - ms.get_width()  // 2,
        menu_rect.y + menu_rect.height // 2 - ms.get_height() // 2,
    ))

    # ── Hint ───────────────────────────────────────────────────────────────
    hint = fonts["small"].render(
        "Model auto-saved to models/", True, colors["grey"]
    )
    screen.blit(hint, (cx - hint.get_width() // 2, window_height - 30))

    return {
        "play_btn": play_rect,
        "menu_btn": menu_rect,
    }