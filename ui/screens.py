# ui/screens.py
# Draws the menu screen and the level-complete screen.
# No game logic. Everything passed in as arguments.
# Depends on: pygame only.

import pygame

# ── Mini-map preview ─────────────────────────────────────────────────────────

# Colours used to render the tiny level preview grid
_PREVIEW_COLOURS = {
    '#': (80,  80,  80),   # wall      — dark grey
    '.': (40,  160, 40),   # goal      — green
    '$': (180, 120, 40),   # box       — orange-brown
    '*': (220, 180, 40),   # box+goal  — gold
    '@': (60,  120, 220),  # player    — blue
    '+': (60,  180, 220),  # player+goal — cyan
    ' ': (28,  28,  35),   # empty     — matches bg
}


def _draw_minimap(screen, level, rect):
    """
    Render a tiny pixel-art preview of `level` inside `rect`.
    Each tile is drawn as a coloured square, sized to fit the rect.

    Args:
        screen : pygame.Surface
        level  : list of strings  (from maps.load_level)
        rect   : pygame.Rect      where to draw the preview
    """
    if not level:
        return

    rows = len(level)
    cols = max(len(r) for r in level)
    if rows == 0 or cols == 0:
        return

    ts = min(rect.width // cols, rect.height // rows)
    ts = max(ts, 2)

    # Centre the minimap inside rect
    grid_w = cols * ts
    grid_h = rows * ts
    ox = rect.x + (rect.width  - grid_w) // 2
    oy = rect.y + (rect.height - grid_h) // 2

    for y, row in enumerate(level):
        for x, tile in enumerate(row):
            colour = _PREVIEW_COLOURS.get(tile, (28, 28, 35))
            pygame.draw.rect(
                screen, colour,
                (ox + x * ts, oy + y * ts, ts, ts)
            )


# ── Menu screen ───────────────────────────────────────────────────────────────

def draw_menu(screen, level_meta, levels, selected, fonts, colors,
              window_width, window_height):
    """
    Draw the main menu.

    Layout:
      Left panel  (~40% width) — title + 3 level buttons stacked vertically
      Right panel (~60% width) — mini-map preview of the selected level

    Args:
        screen      : pygame.Surface
        level_meta  : list of (name, difficulty) tuples
        levels      : list of raw level grids (for preview)
        selected    : index of currently highlighted level (0/1/2)
        fonts       : dict of pygame.font.Font  (keys: 'big','med','small')
        colors      : dict of colour tuples
        window_width, window_height : int
    """
    screen.fill(colors["bg"])

    left_w  = window_width * 2 // 5
    right_x = left_w
    right_w = window_width - right_x

    diff_color_map = {
        "EASY":       colors["easy"],
        "MEDIUM":     colors["medium"],
        "HARD":       colors["hard"],
        "IMPOSSIBLE": colors["impossible"],
    }

    # ── Left panel ───────────────────────────────────────────────────────

    # Title
    title_surf = fonts["big"].render("Sai AbhiRam's SOKOBAN", True, colors["accent"])
    screen.blit(title_surf, ((left_w - title_surf.get_width()) // 2, 40))

    # Subtitle
    sub_surf = fonts["small"].render("Select a level", True, colors["grey"])
    screen.blit(sub_surf, ((left_w - sub_surf.get_width()) // 2, 88))

    # Level buttons — stacked vertically, centred in left panel
    btn_w   = left_w - 60
    btn_h   = 56
    btn_gap = 20
    total_h = len(level_meta) * btn_h + (len(level_meta) - 1) * btn_gap
    btn_start_y = (window_height - total_h) // 2

    button_rects = []
    for i, (name, diff) in enumerate(level_meta):
        bx = (left_w - btn_w) // 2
        by = btn_start_y + i * (btn_h + btn_gap)
        btn_rect = pygame.Rect(bx, by, btn_w, btn_h)
        button_rects.append(btn_rect)

        dcol = diff_color_map.get(diff, colors["white"])
        bg   = colors["btn_hov"] if i == selected else colors["btn"]

        pygame.draw.rect(screen, bg,   btn_rect, border_radius=8)
        pygame.draw.rect(screen, dcol, btn_rect, 2, border_radius=8)

        # Level name (larger)
        name_surf = fonts["med"].render(name, True, colors["white"])
        screen.blit(name_surf, (
            btn_rect.x + 16,
            btn_rect.y + btn_h // 2 - name_surf.get_height() // 2 - 8,
        ))

        # Difficulty label (smaller, coloured)
        diff_surf = fonts["small"].render(diff, True, dcol)
        screen.blit(diff_surf, (
            btn_rect.x + 16,
            btn_rect.y + btn_h // 2 + 4,
        ))

    # ── Right panel — preview ─────────────────────────────────────────────

    # Vertical divider
    pygame.draw.line(screen, (50, 50, 65),
                     (right_x, 20), (right_x, window_height - 20))

    if 0 <= selected < len(levels) and levels[selected]:
        level = levels[selected]
        name, diff = level_meta[selected]
        dcol = diff_color_map.get(diff, colors["white"])

        # Panel label
        label_surf = fonts["med"].render(
            f"Preview — {name}  [{diff}]", True, dcol
        )
        screen.blit(label_surf, (
            right_x + (right_w - label_surf.get_width()) // 2, 40
        ))

        # Mini-map
        padding  = 60
        map_rect = pygame.Rect(
            right_x + padding,
            90,
            right_w - padding * 2,
            window_height - 90 - padding,
        )

        # Background for the preview area
        pygame.draw.rect(screen, colors["panel"], map_rect, border_radius=6)
        pygame.draw.rect(screen, (50, 50, 65),   map_rect, 1, border_radius=6)

        _draw_minimap(screen, level, map_rect)

        # Controls hint at bottom of right panel
        hint = fonts["small"].render(
            "Click to select   ·   ENTER to play", True, colors["grey"]
        )
        screen.blit(hint, (
            right_x + (right_w - hint.get_width()) // 2,
            window_height - 36,
        ))

    return button_rects   # returned so main.py can do hit-testing


# ── Level-complete screen ─────────────────────────────────────────────────────

def draw_complete(screen, game, level_idx, level_meta, fonts, colors,
                  window_width, window_height):
    """
    Draw the level-complete screen.

    Shows: big tick + 'LEVEL COMPLETE', level name + difficulty,
           move/push stats, and key hints.

    Args:
        screen      : pygame.Surface
        game        : Sokoban instance (for move_count, push_count)
        level_idx   : int
        level_meta  : list of (name, difficulty)
        fonts, colors, window_width, window_height
    """
    screen.fill((10, 10, 15))

    name, diff = level_meta[level_idx]

    diff_color_map = {
        "EASY":       colors["easy"],
        "MEDIUM":     colors["medium"],
        "HARD":       colors["hard"],
        "IMPOSSIBLE": colors["impossible"],
    }
    dcol = diff_color_map.get(diff, colors["white"])

    cx = window_width  // 2
    cy = window_height // 2

    lines = [
        ("✓  LEVEL COMPLETE!",                               colors["green"], -90, "big"),
        (f"{name}",                                          dcol,            -30, "med"),
        (f"[{diff}]",                                        dcol,            +10, "small"),
        (f"Moves: {game.move_count}   Pushes: {game.push_count}",
                                                             colors["white"],  60, "med"),
        ("ENTER = Menu",                                     colors["grey"],  120, "small"),
        ("R     = Play again",                               colors["grey"],  150, "small"),
    ]

    for text, color, dy, font_key in lines:
        surf = fonts[font_key].render(text, True, color)
        screen.blit(surf, (cx - surf.get_width() // 2, cy + dy))