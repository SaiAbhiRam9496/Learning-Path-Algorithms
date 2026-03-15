# agents/human.py
# Human keyboard input agent.
# Wraps pygame keyboard events into the BaseAgent interface
# so the game loop treats human input exactly like any other agent.
#
# next_move() returns None until a key is pressed,
# then returns the corresponding (dx, dy) move.

import pygame
from agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """
    Human player — reads arrow key input from pygame events.

    Unlike AI agents, HumanAgent does not compute a move
    automatically. It waits for the player to press a key,
    then returns the corresponding direction.

    Usage in game loop:
        agent = HumanAgent()
        move  = agent.next_move(game)   # returns None if no key pressed
        if move:
            game.move(*move)
    """

    @property
    def name(self):
        return "Human"

    # ── Key state ─────────────────────────────────────────────────────────

    def __init__(self):
        self._pending_move = None   # set by feed_event(), consumed by next_move()

    def feed_event(self, event):
        """
        Feed a pygame KEYDOWN event to the agent.
        Called by the game loop for every event in pygame.event.get().

        Only arrow keys are accepted — all other keys are ignored here
        (R, Z, ESC are handled separately by the game loop).

        Args:
            event : pygame.event.Event
        """
        if event.type != pygame.KEYDOWN:
            return

        key_map = {
            pygame.K_UP:    ( 0, -1),
            pygame.K_DOWN:  ( 0,  1),
            pygame.K_LEFT:  (-1,  0),
            pygame.K_RIGHT: ( 1,  0),
        }

        if event.key in key_map:
            self._pending_move = key_map[event.key]

    def next_move(self, game):
        """
        Return the pending move if a key was pressed, else None.

        The pending move is consumed on read — calling next_move()
        twice without a new key press returns None the second time.

        Args:
            game : Sokoban instance (not used for human — input is external)

        Returns:
            (dx, dy) tuple or None
        """
        move = self._pending_move
        self._pending_move = None   # consume
        return move

    def reset(self):
        """Clear any buffered keypress on level reset."""
        self._pending_move = None

    def is_trained(self):
        return True   # human is always ready