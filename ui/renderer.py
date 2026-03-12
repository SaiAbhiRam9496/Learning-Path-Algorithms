# ui/renderer.py
# Handles all in-game drawing: the tile grid and the HUD bar.
# No game logic here — everything is passed in as arguments.
# Depends on: pygame only. Receives game object but never calls logic methods.

import pygame

# Tile characters → asset key mapping
_TILE_ASSET = {
    '#': 'wall',
    '.': 'goal',
    '$': 'box',
    '*': 'box_on_goal',
    '@': 'player',
    '+': 'player_on_goal',
}

HUD_HEIGHT = 48


def tile_size(game, window_width, window_height, max_tile=64):
    """
    Calculate the largest square tile size that fits the level
    inside the available window space (above the HUD).
    Never smaller than 8px to avoid invisible tiles.
    """
    available_h = window_height - HUD_HEIGHT
    ts = min(
        window_width  // game.width,
        available_h   // game.height,
        max_tile,
    )
    return max(ts, 8)


def draw_grid(screen, game, assets, ts, window_width, window_height):
    """
    Draw the full tile grid centred in the window above the HUD.

    Rendering order per tile:
      1. Always draw floor first (covers any gaps).
      2. Overlay the tile-specific sprite on top.
      3. For composite tiles (* and +), draw two sprites.
    """
    # Centre the grid horizontally and vertically (above HUD)
    ox = (window_width  - game.width  * ts) // 2
    oy = (window_height - HUD_HEIGHT  - game.height * ts) // 2

    floor_surf = assets["floor"]

    for y, row in enumerate(game.level):
        for x, tile in enumerate(row):
            px = ox + x * ts
            py = oy + y * ts

            # Layer 1 — floor always underneath
            screen.blit(floor_surf, (px, py))

            # Layer 2 — tile overlay
            if tile == '#':
                screen.blit(assets["wall"], (px, py))

            elif tile == '.':
                screen.blit(assets["goal"], (px, py))

            elif tile == '$':
                screen.blit(assets["box"], (px, py))

            elif tile == '*':
                # Box sitting on a goal — draw goal then box_on_goal
                screen.blit(assets["goal"],        (px, py))
                screen.blit(assets["box_on_goal"], (px, py))

            elif tile == '@':
                screen.blit(assets["player"], (px, py))

            elif tile == '+':
                # Player standing on a goal — draw goal then player_on_goal
                screen.blit(assets["goal"],           (px, py))
                screen.blit(assets["player_on_goal"], (px, py))

            # ' ' (empty space) — floor already drawn, nothing more needed


def draw_hud(screen, game, level_idx, level_meta, fonts, colors,
             window_width, window_height):
    """
    Draw the status bar at the bottom of the window.

    Shows: level name + difficulty | Moves | Pushes | Boxes left | Controls hint
    """
    hud_y = window_height - HUD_HEIGHT

    # Background panel
    pygame.draw.rect(screen, colors["panel"],
                     (0, hud_y, window_width, HUD_HEIGHT))

    # Top border line
    pygame.draw.line(screen, (50, 50, 60),
                     (0, hud_y), (window_width, hud_y))

    name, diff = level_meta[level_idx]

    diff_color = {
        "EASY":       colors["easy"],
        "MEDIUM":     colors["medium"],
        "HARD":       colors["hard"],
        "IMPOSSIBLE": colors["impossible"],
    }.get(diff, colors["white"])

    items = [
        (f"{level_idx + 1}. {name}",                    diff_color),
        (f"Moves : {game.move_count}",                  colors["white"]),
        (f"Pushes: {game.push_count}",                  colors["white"]),
        (f"Left  : {game.count_remaining_boxes()}",     colors["grey"]),
        ("↑↓←→   R=Reset   Z=Undo   ESC=Menu",         colors["grey"]),
    ]

    font  = fonts["small"]
    gap   = 28
    x     = 16
    cy    = hud_y + HUD_HEIGHT // 2

    for text, color in items:
        surface = font.render(text, True, color)
        screen.blit(surface, (x, cy - surface.get_height() // 2))
        x += surface.get_width() + gap