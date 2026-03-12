# main.py
# Entry point. Game loop, state machine, input handling, save/load progress.
# Imports from core/ and ui/ — no game logic or drawing happens here directly.

import pygame
import sys
import json
import os

from core.maps    import LEVELS, LEVEL_META, load_level
from core.sokoban import Sokoban
from ui.assets    import load_assets
from ui.renderer  import draw_grid, draw_hud, tile_size, HUD_HEIGHT
from ui.screens   import draw_menu, draw_complete

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 650
FPS           = 60

SAVES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saves", "progress.json"
)

# ── Colour palette ────────────────────────────────────────────────────────────
# Defined once here, passed to all draw functions.

COLORS = {
    "bg":         (28,  28,  35),
    "panel":      (18,  18,  24),
    "white":      (230, 230, 230),
    "grey":       (110, 110, 120),
    "accent":     (0,   190, 220),
    "green":      (80,  200,  80),
    "easy":       (70,  170, 100),
    "medium":     (200, 160,  40),
    "hard":       (200,  70,  70),
    "impossible": (180,   0, 255),
    "btn":        (45,   55,  75),
    "btn_hov":    (65,   85, 115),
}

# ── Save / Load ───────────────────────────────────────────────────────────────

def load_progress():
    """
    Load saved progress from saves/progress.json.
    Returns a dict: { "0": {"moves": int, "pushes": int}, ... }
    Returns empty dict if file missing or corrupt.
    """
    try:
        with open(SAVES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(progress, level_idx, moves, pushes):
    """
    Save completion data for a level.
    Only updates if this is a new best (fewer moves).
    """
    key = str(level_idx)
    existing = progress.get(key)

    # Save if first time completing OR new best move count
    if existing is None or moves < existing.get("moves", 9999):
        progress[key] = {"moves": moves, "pushes": pushes}
        try:
            os.makedirs(os.path.dirname(SAVES_PATH), exist_ok=True)
            with open(SAVES_PATH, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception:
            pass  # saving is nice-to-have, never crash the game


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sai AbhiRam's Sokoban")
    clock = pygame.time.Clock()

    # Fonts — defined after pygame.init()
    fonts = {
        "big":   pygame.font.SysFont("arial", 36),
        "med":   pygame.font.SysFont("arial", 22),
        "small": pygame.font.SysFont("arial", 15),
    }

    # Game state
    state    = "menu"      # "menu" | "playing" | "completed"
    selected = 0           # currently highlighted level in menu
    game     = None
    assets   = {}
    progress = load_progress()

    # Pre-load raw levels for minimap previews (already parsed by maps.py)
    preview_levels = LEVELS   # list of raw string grids

    # Button rects returned by draw_menu for click detection
    button_rects = []

    while True:
        clock.tick(FPS)

        # ── Events ───────────────────────────────────────────────────────

        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # ── Menu ─────────────────────────────────────────────────────
            if state == "menu":

                if ev.type == pygame.MOUSEMOTION:
                    mx, my = ev.pos
                    for i, rect in enumerate(button_rects):
                        if rect.collidepoint(mx, my):
                            selected = i
                            break

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    for i, rect in enumerate(button_rects):
                        if rect.collidepoint(mx, my):
                            selected = i
                            # Launch the level
                            game   = Sokoban(load_level(selected))
                            assets = load_assets(tile_size(
                                game, WINDOW_WIDTH, WINDOW_HEIGHT
                            ))
                            state  = "playing"
                            break

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_UP:
                        selected = (selected - 1) % len(LEVEL_META)
                    elif ev.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(LEVEL_META)
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        game   = Sokoban(load_level(selected))
                        assets = load_assets(tile_size(
                            game, WINDOW_WIDTH, WINDOW_HEIGHT
                        ))
                        state  = "playing"

            # ── Playing ──────────────────────────────────────────────────
            elif state == "playing":

                if ev.type == pygame.KEYDOWN:
                    if   ev.key == pygame.K_UP:     game.move( 0, -1)
                    elif ev.key == pygame.K_DOWN:   game.move( 0,  1)
                    elif ev.key == pygame.K_LEFT:   game.move(-1,  0)
                    elif ev.key == pygame.K_RIGHT:  game.move( 1,  0)
                    elif ev.key == pygame.K_r:      game.reset()
                    elif ev.key == pygame.K_z:      game.undo()
                    elif ev.key == pygame.K_ESCAPE: state = "menu"

                    # Check win after every key press
                    if game and game.is_completed():
                        save_progress(
                            progress, selected,
                            game.move_count, game.push_count
                        )
                        state = "completed"

            # ── Completed ─────────────────────────────────────────────────
            elif state == "completed":

                if ev.type == pygame.KEYDOWN:
                    if   ev.key == pygame.K_RETURN: state = "menu"
                    elif ev.key == pygame.K_r:
                        game.reset()
                        state = "playing"

        # ── Draw ─────────────────────────────────────────────────────────

        screen.fill(COLORS["bg"])

        if state == "menu":
            button_rects = draw_menu(
                screen, LEVEL_META, preview_levels, selected,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "playing":
            draw_grid(
                screen, game, assets,
                tile_size(game, WINDOW_WIDTH, WINDOW_HEIGHT),
                WINDOW_WIDTH, WINDOW_HEIGHT
            )
            draw_hud(
                screen, game, selected, LEVEL_META,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "completed":
            draw_complete(
                screen, game, selected, LEVEL_META,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        pygame.display.flip()


if __name__ == "__main__":
    main()