# ui/assets.py
# Handles all image loading for the game.
# Falls back to solid-colour surfaces if image files are missing.
# Zero game-logic dependencies — only pygame and os.

import pygame
import os

# Path to the images folder, relative to this file's location
IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "assets", "images"
)

# Fallback colours used when an image file cannot be loaded
_FALLBACK_COLOURS = {
    "wall.png":            (80,  80,  80),
    "floor.png":           (45,  45,  45),
    "box.png":             (180, 120, 40),
    "goal.png":            (40,  160, 40),
    "player.png":          (60,  120, 220),
    "box_on_goal.png":     (220, 180, 40),
    "player_on_goal.png":  (60,  180, 220),
}

# Maps asset key → image filename
_ASSET_FILES = {
    "wall":            "wall.png",
    "floor":           "floor.png",
    "goal":            "goal.png",
    "box":             "box.png",
    "box_on_goal":     "box_on_goal.png",
    "player":          "player.png",
    "player_on_goal":  "player_on_goal.png",
}


def _make_fallback(colour, size):
    """Create a solid-colour square surface as a fallback tile."""
    surface = pygame.Surface((size, size))
    surface.fill(colour)
    return surface


def _load_img(filename, size):
    """
    Load a single image from IMAGES_DIR, scaled to (size x size).
    Returns a fallback coloured surface if the file is missing or corrupt.
    """
    path = os.path.join(IMAGES_DIR, filename)
    try:
        img = pygame.image.load(path)
        return pygame.transform.scale(img, (size, size))
    except Exception:
        colour = _FALLBACK_COLOURS.get(filename, (200, 0, 200))
        return _make_fallback(colour, size)


def load_assets(tile_size):
    """
    Load all tile images at the given tile_size.
    Returns a dict: { asset_key: pygame.Surface }

    Called once when a level is loaded (or when tile size changes).
    Re-call if tile_size changes (e.g. window resize in the future).
    """
    return {
        key: _load_img(filename, tile_size)
        for key, filename in _ASSET_FILES.items()
    }