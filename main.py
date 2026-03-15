# main.py
# Entry point. Game loop, state machine, input handling.
# Wires together core/, agents/, quantum/, and ui/.
#
# States:
#   "home"            → Training / Game buttons
#   "training_setup"  → parameter selection (pygame)
#   "training_done"   → result screen after training (pygame)
#   "game_select"     → level + agent selection
#   "playing"         → active gameplay (human or agent)
#   "post_game"       → win/loss result screen

import pygame
import sys
import json
import os
import time


from core.maps    import LEVELS, LEVEL_META, load_level
from core.sokoban import Sokoban
from ui.assets    import load_assets
from ui.renderer  import draw_grid, draw_hud, tile_size, HUD_HEIGHT
from ui.screens   import draw_complete
from ui.training_panel import (
    make_training_inputs, draw_training_setup, draw_training_result
)
from ui.game_panel import (
    draw_game_select, draw_in_game_hud, draw_post_game,
    AgentDropdown, AGENT_NAMES
)

from agents.human     import HumanAgent
from agents.astar     import AStarAgent
from agents.mcts      import MCTSAgent
from agents.qlearning import QLearningAgent
from quantum.grover   import GroverAgent

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 650
FPS           = 60

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVES_PATH = os.path.join(BASE_DIR, "saves",  "progress.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR= os.path.join(BASE_DIR, "results")

# Q-Learning model paths per level index
QL_MODEL_PATH = {
    0: os.path.join(MODELS_DIR, "qlearning_easy.json"),
    1: os.path.join(MODELS_DIR, "qlearning_medium.json"),
    2: os.path.join(MODELS_DIR, "qlearning_hard.json"),
    3: os.path.join(MODELS_DIR, "qlearning_impossible.json"),
}

# ── Colour palette ────────────────────────────────────────────────────────────

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

# ── Agent speed control ────────────────────────────────────────────────────────
# Delay between agent moves in milliseconds.
# Human = 0 (instant, driven by keypress).
AGENT_MOVE_DELAY = {
    "Human":     0,
    "A*":        150,    # fast replay
    "Q-Learning":200,
    "MCTS":      0,      # MCTS is slow to compute — no extra delay
    "Grover":    300,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_progress():
    try:
        with open(SAVES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(progress, level_idx, moves, pushes):
    key = str(level_idx)
    existing = progress.get(key)
    if existing is None or moves < existing.get("moves", 9999):
        progress[key] = {"moves": moves, "pushes": pushes}
        try:
            os.makedirs(os.path.dirname(SAVES_PATH), exist_ok=True)
            with open(SAVES_PATH, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception:
            pass


def qlearning_trained_for(level_idx):
    """Check if a Q-Learning model exists for this level."""
    path = QL_MODEL_PATH.get(level_idx)
    return path is not None and os.path.exists(path)


def any_qlearning_trained():
    """True if any Q-Learning model exists."""
    return any(os.path.exists(p) for p in QL_MODEL_PATH.values())


def save_training_log(stats):
    """Append training run stats to results/training_log.json."""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "training_log.json")
        logs = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                logs = json.load(f)
        logs.append(stats)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass


def make_agent(name, level_idx):
    """
    Instantiate the correct agent by name.
    For Q-Learning: loads saved model automatically.
    Returns agent instance.
    """
    if name == "Human":
        return HumanAgent()
    elif name == "A*":
        return AStarAgent()
    elif name == "MCTS":
        return MCTSAgent()
    elif name == "Q-Learning":
        agent = QLearningAgent()
        path  = QL_MODEL_PATH.get(level_idx, "")
        agent.load(path)
        return agent
    elif name == "Grover":
        return GroverAgent()
    return HumanAgent()


# ── Home screen ───────────────────────────────────────────────────────────────

def draw_home(screen, fonts, colors, window_width, window_height):
    """
    Draw the home screen with TRAINING and GAME buttons.
    Returns dict of rects for hit-testing.
    """
    screen.fill(colors["bg"])
    cx = window_width  // 2
    cy = window_height // 2

    # Title
    t1 = fonts["big"].render("SAI ABHIRAM'S SOKOBAN", True, colors["accent"])
    screen.blit(t1, (cx - t1.get_width() // 2, cy - 180))

    # Subtitle
    t2 = fonts["small"].render(
        "Classical & Quantum Pathfinding Agents", True, colors["grey"]
    )
    screen.blit(t2, (cx - t2.get_width() // 2, cy - 130))

    # Divider
    pygame.draw.line(screen, (50, 50, 70),
                     (cx - 200, cy - 105), (cx + 200, cy - 105))

    # Buttons
    btn_w = 260
    btn_h = 64
    gap   = 24

    train_rect = pygame.Rect(cx - btn_w // 2, cy - 60, btn_w, btn_h)
    game_rect  = pygame.Rect(cx - btn_w // 2, cy - 60 + btn_h + gap,
                             btn_w, btn_h)

    for rect, label, sub, col in [
        (train_rect, "TRAINING",
         "Configure & train agents", colors["medium"]),
        (game_rect,  "GAME",
         "Play or watch agents solve", colors["accent"]),
    ]:
        pygame.draw.rect(screen, colors["btn"], rect, border_radius=10)
        pygame.draw.rect(screen, col, rect, 2, border_radius=10)

        ls = fonts["med"].render(label, True, colors["white"])
        ss = fonts["small"].render(sub,  True, col)

        screen.blit(ls, (rect.x + rect.width // 2 - ls.get_width() // 2,
                         rect.y + 10))
        screen.blit(ss, (rect.x + rect.width // 2 - ss.get_width() // 2,
                         rect.y + 36))

    # Agent legend at bottom
    agents = [
        ("Human",     colors["white"]),
        ("A*",        colors["easy"]),
        ("Q-Learning",colors["medium"]),
        ("MCTS",      colors["hard"]),
        ("Grover",    (180, 100, 255)),
    ]
    legend_y = cy + 130
    ls_total = sum(
        fonts["small"].size(n)[0] + 20 for n, _ in agents
    )
    lx = cx - ls_total // 2
    for name, col in agents:
        s = fonts["small"].render(f"● {name}", True, col)
        screen.blit(s, (lx, legend_y))
        lx += s.get_width() + 20

    hint = fonts["small"].render("ESC to quit", True, colors["grey"])
    screen.blit(hint, (cx - hint.get_width() // 2, window_height - 28))

    return {
        "train_btn": train_rect,
        "game_btn":  game_rect,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sai Abhiram's Sokoban")
    clock = pygame.time.Clock()

    fonts = {
        "big":   pygame.font.SysFont("segoeui", 36, bold=True),
        "med":   pygame.font.SysFont("segoeui", 22),
        "small": pygame.font.SysFont("segoeui", 15),
    }

    # ── State machine variables ────────────────────────────────────────────
    state          = "home"
    home_rects     = {}
    progress       = load_progress()

    # Training state
    training_inputs       = None
    training_level_idx    = 0
    training_rects        = {}
    last_train_stats      = {}

    # Game select state
    sel_level      = 0
    sel_agent_idx  = 0   # index into AGENT_NAMES
    game_sel_rects = {}

    # Playing state
    game           = None
    assets         = {}
    agent          = None
    agent_name     = "Human"
    level_idx      = 0
    game_start_t   = 0.0
    last_agent_t   = 0.0
    dropdown       = AgentDropdown(x=0, y=0, width=160)

    # Post-game state
    post_rects     = {}
    game_won       = False
    time_elapsed   = 0.0

    # ── Game loop ──────────────────────────────────────────────────────────
    while True:
        clock.tick(FPS)
        now = time.time()

        # ── Events ────────────────────────────────────────────────────────
        for ev in pygame.event.get():

            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # ── HOME ──────────────────────────────────────────────────────
            if state == "home":
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if home_rects.get("train_btn", pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        training_inputs    = make_training_inputs(
                            WINDOW_WIDTH, WINDOW_HEIGHT
                        )
                        training_level_idx = 0
                        state = "training_setup"

                    elif home_rects.get("game_btn", pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        sel_level     = 0
                        sel_agent_idx = 0
                        state = "game_select"

            # ── TRAINING SETUP ────────────────────────────────────────────
            elif state == "training_setup":
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        state = "home"

                # Feed events to input boxes
                if training_inputs:
                    for box in training_inputs.values():
                        box.handle_event(ev, COLORS)

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    # Level selector
                    for i, rect in enumerate(
                        training_rects.get("level_btns", [])
                    ):
                        if rect.collidepoint(ev.pos):
                            training_level_idx = i

                    # Train button
                    if training_rects.get("train_btn",
                                          pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):

                        # Read hyperparams
                        episodes = training_inputs["episodes"].get_int(10000)
                        epsilon  = training_inputs["epsilon"].get_float(1.0)
                        lr       = training_inputs["lr"].get_float(0.1)
                        gamma    = training_inputs["gamma"].get_float(0.95)

                        # Clamp values
                        episodes = max(100, min(episodes, 500_000))
                        epsilon  = max(0.01, min(epsilon, 1.0))
                        lr       = max(0.001, min(lr, 1.0))
                        gamma    = max(0.1, min(gamma, 0.999))

                        # ── Run training in terminal ──────────────────────
                        print()
                        print("Pygame window is idle — switch to terminal")
                        print("Training will appear there...")
                        print()

                        ql = QLearningAgent(
                            lr=lr, gamma=gamma, epsilon=epsilon
                        )
                        stats = ql.train(
                            level_idx=training_level_idx,
                            episodes=episodes,
                        )

                        # Auto-save model
                        save_path = QL_MODEL_PATH[training_level_idx]
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        ql.save(save_path)

                        # Save log
                        save_training_log(stats)

                        last_train_stats = stats
                        state = "training_done"

            # ── TRAINING DONE ─────────────────────────────────────────────
            elif state == "training_done":
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        state = "home"

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if post_rects.get("play_btn",
                                      pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        # Jump straight into game_select
                        sel_level     = training_level_idx
                        sel_agent_idx = AGENT_NAMES.index("Q-Learning")
                        state = "game_select"

                    elif post_rects.get("menu_btn",
                                        pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        state = "home"

            # ── GAME SELECT ───────────────────────────────────────────────
            elif state == "game_select":
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        state = "home"

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    # Level buttons
                    for i, rect in enumerate(
                        game_sel_rects.get("level_btns", [])
                    ):
                        if rect.collidepoint(ev.pos):
                            sel_level = i

                    # Agent buttons
                    for i, rect in enumerate(
                        game_sel_rects.get("agent_btns", [])
                    ):
                        if rect.collidepoint(ev.pos):
                            name = AGENT_NAMES[i]
                            # Block locked Q-Learning
                            if name == "Q-Learning" and \
                                    not qlearning_trained_for(sel_level):
                                pass
                            else:
                                sel_agent_idx = i

                    # Play button
                    play_rect = game_sel_rects.get(
                        "play_btn", pygame.Rect(0,0,0,0)
                    )
                    if play_rect.collidepoint(ev.pos):
                        chosen_name = AGENT_NAMES[sel_agent_idx]
                        if chosen_name == "Q-Learning" and \
                                not qlearning_trained_for(sel_level):
                            pass   # locked — do nothing
                        else:
                            # ── Launch game ───────────────────────────────
                            level_idx  = sel_level
                            agent_name = chosen_name
                            agent      = make_agent(agent_name, level_idx)
                            game       = Sokoban(load_level(level_idx))
                            ts         = tile_size(
                                game, WINDOW_WIDTH,
                                WINDOW_HEIGHT - HUD_HEIGHT
                            )
                            assets     = load_assets(ts)
                            dropdown   = AgentDropdown(x=0, y=0, width=160)
                            game_start_t  = now
                            last_agent_t  = now

                            # Let agent precompute path if needed
                            agent.on_level_start(game)

                            state = "playing"

            # ── PLAYING ───────────────────────────────────────────────────
            elif state == "playing":

                # Dropdown click handling (agent switch)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    new_idx = dropdown.handle_click(
                        ev.pos,
                        AGENT_NAMES.index(agent_name),
                        any_qlearning_trained()
                    )
                    if new_idx is not None:
                        agent_name = AGENT_NAMES[new_idx]
                        agent = make_agent(agent_name, level_idx)
                        agent.on_level_start(game)
                        last_agent_t = now

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        state = "game_select"

                    elif ev.key == pygame.K_r:
                        game.reset()
                        agent.reset()
                        agent.on_level_start(game)
                        game_start_t = now
                        last_agent_t = now

                    elif ev.key == pygame.K_z and agent_name == "Human":
                        game.undo()

                    # Human movement
                    elif agent_name == "Human":
                        if isinstance(agent, HumanAgent):
                            agent.feed_event(ev)

                # Check completion
                if game and game.is_completed():
                    game_won    = True
                    time_elapsed= now - game_start_t
                    save_progress(
                        progress, level_idx,
                        game.move_count, game.push_count
                    )
                    state = "post_game"

            # ── POST GAME ─────────────────────────────────────────────────
            elif state == "post_game":
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_m):
                        state = "game_select"
                    elif ev.key == pygame.K_r:
                        game.reset()
                        agent.reset()
                        agent.on_level_start(game)
                        game_start_t = now
                        last_agent_t = now
                        game_won     = False
                        state = "playing"
                    elif ev.key == pygame.K_n:
                        next_idx = level_idx + 1
                        if next_idx < len(LEVEL_META):
                            level_idx  = next_idx
                            game       = Sokoban(load_level(level_idx))
                            ts         = tile_size(
                                game, WINDOW_WIDTH,
                                WINDOW_HEIGHT - HUD_HEIGHT
                            )
                            assets     = load_assets(ts)
                            agent      = make_agent(agent_name, level_idx)
                            agent.on_level_start(game)
                            game_start_t = now
                            last_agent_t = now
                            game_won     = False
                            state = "playing"

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if post_rects.get("menu_btn",
                                      pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        state = "game_select"

                    elif post_rects.get("retry_btn",
                                        pygame.Rect(0,0,0,0))\
                            .collidepoint(ev.pos):
                        game.reset()
                        agent.reset()
                        agent.on_level_start(game)
                        game_start_t = now
                        last_agent_t = now
                        game_won     = False
                        state = "playing"

                    elif post_rects.get("next_btn") and \
                            post_rects["next_btn"].collidepoint(ev.pos):
                        next_idx = level_idx + 1
                        if next_idx < len(LEVEL_META):
                            level_idx  = next_idx
                            game       = Sokoban(load_level(level_idx))
                            ts         = tile_size(
                                game, WINDOW_WIDTH,
                                WINDOW_HEIGHT - HUD_HEIGHT
                            )
                            assets     = load_assets(ts)
                            agent      = make_agent(agent_name, level_idx)
                            agent.on_level_start(game)
                            game_start_t = now
                            last_agent_t = now
                            game_won     = False
                            state = "playing"

        # ── Agent move tick (outside event loop) ──────────────────────────
        if state == "playing" and agent and game and not game.is_completed():
            delay = AGENT_MOVE_DELAY.get(agent_name, 200) / 1000.0

            if agent_name != "Human":
                if now - last_agent_t >= delay:
                    move = agent.next_move(game)
                    if move is not None:
                        game.move(*move)
                    last_agent_t = now

                    # Check completion after agent move
                    if game.is_completed():
                        game_won    = True
                        time_elapsed= now - game_start_t
                        save_progress(
                            progress, level_idx,
                            game.move_count, game.push_count
                        )
                        state = "post_game"

            else:
                # Human: process buffered keypress
                move = agent.next_move(game)
                if move is not None:
                    game.move(*move)
                    if game.is_completed():
                        game_won    = True
                        time_elapsed= now - game_start_t
                        save_progress(
                            progress, level_idx,
                            game.move_count, game.push_count
                        )
                        state = "post_game"

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(COLORS["bg"])

        if state == "home":
            home_rects = draw_home(
                screen, fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "training_setup":
            training_rects = draw_training_setup(
                screen, training_inputs or {}, training_level_idx,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "training_done":
            post_rects = draw_training_result(
                screen, last_train_stats,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "game_select":
            game_sel_rects = draw_game_select(
                screen,
                sel_level, sel_agent_idx,
                any_qlearning_trained(),
                LEVELS,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT
            )

        elif state == "playing":
            # Draw game grid below HUD
            ts = tile_size(game, WINDOW_WIDTH, WINDOW_HEIGHT - HUD_HEIGHT)
            draw_grid(
                screen, game, assets, ts,
                WINDOW_WIDTH, WINDOW_HEIGHT
            )
            # In-game HUD with dropdown
            draw_in_game_hud(
                screen, game, level_idx, agent_name,
                dropdown, any_qlearning_trained(),
                fonts, COLORS, WINDOW_WIDTH, HUD_HEIGHT
            )

        elif state == "post_game":
            grover_exp = ""
            if agent_name == "Grover" and isinstance(agent, GroverAgent):
                grover_exp = agent.get_explanation()

            post_rects = draw_post_game(
                screen, game, level_idx, agent_name,
                time_elapsed, game_won,
                fonts, COLORS, WINDOW_WIDTH, WINDOW_HEIGHT,
                grover_explanation=grover_exp
            )

        pygame.display.flip()


if __name__ == "__main__":
    main()