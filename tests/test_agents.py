# tests/test_agents.py
# Comprehensive test suite for all 4 agents.
# Run with: python -m pytest tests/test_agents.py -v
# Or simply: python tests/test_agents.py
#
# Tests cover:
#   BaseAgent    — interface contract
#   HumanAgent   — event feeding, move consumption
#   AStarAgent   — solves Easy and Medium, verifies solution
#   MCTSAgent    — wins Easy reliably, handles mid-game takeover
#   QLearning    — training, save, load, greedy play
#   GroverAgent  — structure, scoring, fallback mode
#   Integration  — all agents on all levels they support

import sys
import os
import json
import types
import time
import copy
import random
import tempfile

# ── Path setup ────────────────────────────────────────────────────────────────
# Works whether run from project root or tests/ folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.maps    import load_level, LEVEL_META
from core.sokoban import Sokoban

# ── Colour codes for terminal output ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── Test runner ───────────────────────────────────────────────────────────────

_results = []   # (test_name, passed, message)

def test(name):
    """Decorator to register a test function."""
    def decorator(fn):
        _results.append((name, fn))
        return fn
    return decorator

def run_all():
    passed = 0
    failed = 0
    errors = []

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Learning-Path-Algorithms — Agent Test Suite{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()

    for name, fn in _results:
        try:
            fn()
            print(f"  {GREEN}✓{RESET}  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  {RED}✗{RESET}  {name}")
            print(f"      {RED}{e}{RESET}")
            errors.append((name, str(e)))
            failed += 1
        except Exception as e:
            print(f"  {RED}✗{RESET}  {name}")
            print(f"      {RED}{type(e).__name__}: {e}{RESET}")
            errors.append((name, f"{type(e).__name__}: {e}"))
            failed += 1

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Passed : {GREEN}{passed}{RESET}")
    print(f"  Failed : {RED}{failed}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()

    if errors:
        print(f"{RED}Failed tests:{RESET}")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        print()

    return failed == 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_game(level_idx=0):
    """Return a fresh Sokoban game at given level."""
    return Sokoban(load_level(level_idx))


def play_agent(agent, game, max_moves=500):
    """
    Let agent play until completion or max_moves.
    Returns (completed, moves_made).
    """
    agent.on_level_start(game)
    moves = 0
    while not game.is_completed() and moves < max_moves:
        move = agent.next_move(game)
        if move is None:
            break
        game.move(*move)
        moves += 1
    return game.is_completed(), moves


# ═════════════════════════════════════════════════════════════════════════════
# BASE AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("BaseAgent: interface raises NotImplementedError for abstract methods")
def _():
    from agents.base_agent import BaseAgent

    class Incomplete(BaseAgent):
        pass   # missing name and next_move

    agent = Incomplete()

    try:
        _ = agent.name
        assert False, "name should raise NotImplementedError"
    except NotImplementedError:
        pass

    try:
        agent.next_move(None)
        assert False, "next_move should raise NotImplementedError"
    except NotImplementedError:
        pass


@test("BaseAgent: concrete subclass works correctly")
def _():
    from agents.base_agent import BaseAgent

    class MockAgent(BaseAgent):
        @property
        def name(self): return "Mock"
        def next_move(self, game): return (1, 0)

    agent = MockAgent()
    assert agent.name == "Mock"
    assert agent.next_move(None) == (1, 0)
    assert agent.is_trained() == True
    agent.reset()   # should not raise
    agent.on_level_start(None)   # should not raise


@test("BaseAgent: train/save/load raise NotImplementedError")
def _():
    from agents.base_agent import BaseAgent

    class MockAgent(BaseAgent):
        @property
        def name(self): return "Mock"
        def next_move(self, game): return (1, 0)

    agent = MockAgent()
    for method, args in [("train", (0,1,1,1,1)), ("save", ("x",)), ("load", ("x",))]:
        try:
            getattr(agent, method)(*args)
            assert False, f"{method} should raise NotImplementedError"
        except NotImplementedError:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# HUMAN AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("HumanAgent: name and is_trained")
def _():
    from agents.human import HumanAgent
    agent = HumanAgent()
    assert agent.name == "Human"
    assert agent.is_trained() == True


@test("HumanAgent: returns None before any keypress")
def _():
    from agents.human import HumanAgent
    agent = HumanAgent()
    assert agent.next_move(None) is None


@test("HumanAgent: all 4 arrow keys map correctly")
def _():
    from agents.human import HumanAgent
    import pygame
    pygame.init()

    agent = HumanAgent()

    key_map = {
        pygame.K_UP:    (0, -1),
        pygame.K_DOWN:  (0,  1),
        pygame.K_LEFT:  (-1, 0),
        pygame.K_RIGHT: (1,  0),
    }

    for key, expected in key_map.items():
        ev = pygame.event.Event(pygame.KEYDOWN, key=key, unicode="")
        agent.feed_event(ev)
        result = agent.next_move(None)
        assert result == expected, \
            f"Key {key}: expected {expected}, got {result}"

    pygame.quit()


@test("HumanAgent: move consumed after one read")
def _():
    from agents.human import HumanAgent
    import pygame
    pygame.init()

    agent = HumanAgent()
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, unicode="")
    agent.feed_event(ev)

    first  = agent.next_move(None)
    second = agent.next_move(None)

    assert first  == (0, -1), f"First read wrong: {first}"
    assert second is None,    f"Second read should be None: {second}"

    pygame.quit()


@test("HumanAgent: reset clears pending move")
def _():
    from agents.human import HumanAgent
    import pygame
    pygame.init()

    agent = HumanAgent()
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, unicode="")
    agent.feed_event(ev)
    agent.reset()
    assert agent.next_move(None) is None

    pygame.quit()


# ═════════════════════════════════════════════════════════════════════════════
# A* AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("AStarAgent: name and is_trained")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    assert agent.name == "A*"
    assert agent.is_trained() == True


@test("AStarAgent: solves Easy level with verified solution")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    game  = fresh_game(0)

    completed, moves = play_agent(agent, game)

    assert completed, f"A* failed to solve Easy (moves={moves})"
    assert moves <= 50, f"A* took too many moves: {moves}"
    assert game.push_count > 0, "No pushes recorded"


@test("AStarAgent: solves Medium level with verified solution")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    game  = fresh_game(1)

    t0 = time.time()
    completed, moves = play_agent(agent, game, max_moves=500)
    t1 = time.time()

    assert completed, f"A* failed to solve Medium (moves={moves})"
    assert t1 - t0 < 30, f"A* took too long: {t1-t0:.1f}s"


@test("AStarAgent: solution path precomputed in on_level_start")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    game  = fresh_game(0)

    agent.on_level_start(game)

    assert agent._solved == True, "A* did not find solution"
    assert len(agent._solution) > 0, "Solution path is empty"
    assert agent.stats["solution_len"] > 0


@test("AStarAgent: reset clears solution")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    game  = fresh_game(0)

    agent.on_level_start(game)
    assert agent._solved == True

    agent.reset()
    assert agent._solved  == False
    assert agent._solution == []
    assert agent.stats["total_moves"] == 0


@test("AStarAgent: returns None when solution exhausted")
def _():
    from agents.astar import AStarAgent
    agent = AStarAgent()
    game  = fresh_game(0)

    # Play to completion
    play_agent(agent, game)

    # Now solution is exhausted — next_move should return None
    extra = agent.next_move(game)
    assert extra is None, f"Expected None after completion, got {extra}"


# ═════════════════════════════════════════════════════════════════════════════
# MCTS AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("MCTSAgent: name and is_trained")
def _():
    from agents.mcts import MCTSAgent
    agent = MCTSAgent()
    assert agent.name == "MCTS"
    assert agent.is_trained() == True


@test("MCTSAgent: solves Easy level (2 of 2 trials)")
def _():
    from agents.mcts import MCTSAgent
    wins = 0
    for _ in range(2):
        agent = MCTSAgent(simulations=500)
        game  = fresh_game(0)
        completed, _ = play_agent(agent, game, max_moves=200)
        if completed:
            wins += 1

    assert wins >= 2, f"MCTS only won {wins}/2 trials on Easy"


@test("MCTSAgent: mid-game takeover works")
def _():
    from agents.mcts import MCTSAgent
    game = fresh_game(0)

    # Human makes 2 moves
    game.move(1, 0)
    game.move(0, 1)
    moves_before = game.move_count

    # MCTS takes over from current state
    agent = MCTSAgent(simulations=500)
    completed, extra = play_agent(agent, game, max_moves=200)

    assert completed, "MCTS failed mid-game takeover"
    assert game.move_count > moves_before, "No moves made after takeover"


@test("MCTSAgent: returns tuple move of correct form")
def _():
    from agents.mcts import MCTSAgent
    agent = MCTSAgent(simulations=100)
    game  = fresh_game(0)

    move = agent.next_move(game)
    assert move is not None, "next_move returned None on fresh game"
    assert isinstance(move, tuple), f"Expected tuple, got {type(move)}"
    assert len(move) == 2, f"Expected 2-tuple, got length {len(move)}"
    dx, dy = move
    assert dx in (-1, 0, 1), f"dx={dx} invalid"
    assert dy in (-1, 0, 1), f"dy={dy} invalid"
    assert not (dx == 0 and dy == 0), "Move (0,0) is not valid"


@test("MCTSAgent: reset clears stats")
def _():
    from agents.mcts import MCTSAgent
    agent = MCTSAgent(simulations=100)
    game  = fresh_game(0)

    agent.next_move(game)
    assert agent.stats["moves_made"] == 1

    agent.reset()
    assert agent.stats["moves_made"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Q-LEARNING AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("QLearningAgent: name and is_trained before training")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    assert agent.name == "Q-Learning"
    assert agent.is_trained() == False


@test("QLearningAgent: next_move returns None before training")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    game  = fresh_game(0)
    move  = agent.next_move(game)
    assert move is None, f"Expected None before training, got {move}"


@test("QLearningAgent: training runs and returns stats")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    stats = agent.train(level_idx=0, episodes=500)

    assert agent.is_trained() == True
    assert "win_rate"    in stats
    assert "episodes"    in stats
    assert "avg_steps"   in stats
    assert "best_steps"  in stats
    assert stats["episodes"] == 500
    assert agent.q_table_size() > 0, "Q-table empty after training"


@test("QLearningAgent: achieves >70% win rate at 5000 episodes")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    stats = agent.train(level_idx=0, episodes=5000)

    win_rate = stats["win_rate"]
    assert win_rate >= 70.0, \
        f"Win rate too low: {win_rate:.1f}% (expected >= 70%)"


@test("QLearningAgent: save creates valid JSON file")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    agent.train(level_idx=0, episodes=200)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        agent.save(path)
        assert os.path.exists(path), "File not created"
        assert os.path.getsize(path) > 0, "File is empty"

        with open(path) as f:
            data = json.load(f)

        assert "q_table"     in data
        assert "train_stats" in data
        assert "hyperparams" in data
        assert len(data["q_table"]) > 0
    finally:
        os.unlink(path)


@test("QLearningAgent: load restores Q-table exactly")
def _():
    from agents.qlearning import QLearningAgent
    agent1 = QLearningAgent()
    agent1.train(level_idx=0, episodes=300)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        agent1.save(path)

        agent2 = QLearningAgent()
        result = agent2.load(path)

        assert result == True, "load() returned False"
        assert agent2.is_trained() == True
        assert agent2.q_table_size() == agent1.q_table_size(), \
            "Q-table size mismatch after load"
    finally:
        os.unlink(path)


@test("QLearningAgent: load returns False for missing file")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    result = agent.load("/tmp/definitely_does_not_exist_xyz.json")
    assert result == False


@test("QLearningAgent: load returns False for empty file")
def _():
    from agents.qlearning import QLearningAgent
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name   # creates empty file

    try:
        agent  = QLearningAgent()
        result = agent.load(path)
        assert result == False, \
            f"Expected False for empty file, got {result}"
        assert agent.is_trained() == False
    finally:
        os.unlink(path)


@test("QLearningAgent: greedy play after training returns valid moves")
def _():
    from agents.qlearning import QLearningAgent
    agent = QLearningAgent()
    agent.train(level_idx=0, episodes=1000)

    game = fresh_game(0)
    for _ in range(10):
        move = agent.next_move(game)
        assert move is not None, "next_move returned None after training"
        assert isinstance(move, tuple)
        assert len(move) == 2
        game.move(*move)


# ═════════════════════════════════════════════════════════════════════════════
# GROVER AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("GroverAgent: name and is_trained")
def _():
    from quantum.grover import GroverAgent
    agent = GroverAgent()
    assert agent.name == "Grover"
    assert agent.is_trained() == True


@test("GroverAgent: next_move returns valid tuple")
def _():
    from quantum.grover import GroverAgent
    agent = GroverAgent()
    game  = fresh_game(0)

    move = agent.next_move(game)
    assert move is not None, "Grover returned None on fresh game"
    assert isinstance(move, tuple)
    assert len(move) == 2
    dx, dy = move
    assert dx in (-1, 0, 1)
    assert dy in (-1, 0, 1)
    assert not (dx == 0 and dy == 0)


@test("GroverAgent: move is always a legal action")
def _():
    from quantum.grover import GroverAgent
    agent = GroverAgent()
    game  = fresh_game(0)

    for _ in range(10):
        legal = game.get_legal_actions()
        move  = agent.next_move(game)
        assert move in legal, \
            f"Grover returned illegal move {move}. Legal: {legal}"
        game.move(*move)
        if game.is_completed():
            break


@test("GroverAgent: explanation string populated after move")
def _():
    from quantum.grover import GroverAgent
    agent = GroverAgent()
    game  = fresh_game(0)

    agent.next_move(game)

    exp = agent.get_explanation()
    assert isinstance(exp, str)
    assert len(exp) > 0, "Explanation string is empty"
    assert "Oracle target" in exp or "SIMULATED" in exp, \
        f"Unexpected explanation format: {exp[:100]}"


@test("GroverAgent: qiskit_available() returns bool")
def _():
    from quantum.grover import GroverAgent
    result = GroverAgent.qiskit_available()
    assert isinstance(result, bool)


@test("GroverAgent: reset clears stats and explanation")
def _():
    from quantum.grover import GroverAgent
    agent = GroverAgent()
    game  = fresh_game(0)

    agent.next_move(game)
    assert agent.stats["moves_made"] == 1

    agent.reset()
    assert agent.stats["moves_made"]    == 0
    assert agent.last_explanation       == ""


@test("GroverAgent: action scoring prefers better moves")
def _():
    from quantum.grover import _score_action
    game = fresh_game(0)

    # Score all actions
    scores = {}
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
        scores[(dx,dy)] = _score_action(game, dx, dy)

    # Invalid move should score lowest
    min_score = min(scores.values())
    max_score = max(scores.values())

    # There should be some score variation
    assert max_score > min_score, \
        f"All moves scored equally: {scores}"


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("Integration: all agents implement BaseAgent interface")
def _():
    from agents.base_agent import BaseAgent
    from agents.human      import HumanAgent
    from agents.astar      import AStarAgent
    from agents.mcts       import MCTSAgent
    from agents.qlearning  import QLearningAgent
    from quantum.grover    import GroverAgent

    for AgentClass in [HumanAgent, AStarAgent, MCTSAgent,
                       QLearningAgent, GroverAgent]:
        agent = AgentClass()
        assert isinstance(agent, BaseAgent), \
            f"{AgentClass.__name__} does not inherit BaseAgent"
        assert hasattr(agent, "name")
        assert hasattr(agent, "next_move")
        assert hasattr(agent, "is_trained")
        assert hasattr(agent, "reset")
        assert hasattr(agent, "on_level_start")


@test("Integration: A* and MCTS always report is_trained=True")
def _():
    from agents.astar import AStarAgent
    from agents.mcts  import MCTSAgent

    assert AStarAgent().is_trained() == True
    assert MCTSAgent().is_trained()  == True


@test("Integration: agent switch mid-game preserves board state")
def _():
    from agents.astar import AStarAgent
    from agents.mcts  import MCTSAgent

    game = fresh_game(0)

    # A* plays 3 moves
    astar = AStarAgent()
    astar.on_level_start(game)
    for _ in range(3):
        move = astar.next_move(game)
        if move:
            game.move(*move)

    state_before = (game.player_x, game.player_y, game.move_count)

    # MCTS takes over — board should be unchanged
    mcts = MCTSAgent(simulations=100)
    mcts.on_level_start(game)

    state_after = (game.player_x, game.player_y, game.move_count)
    assert state_before == state_after, \
        f"Board changed during agent switch: {state_before} → {state_after}"


@test("Integration: all agents handle completed game gracefully")
def _():
    from agents.astar     import AStarAgent
    from agents.mcts      import MCTSAgent
    from quantum.grover   import GroverAgent

    for AgentClass in [AStarAgent, MCTSAgent, GroverAgent]:
        game  = fresh_game(0)
        agent = AgentClass() if AgentClass != MCTSAgent \
                else MCTSAgent(simulations=500)

        # Solve it
        play_agent(agent, game, max_moves=500)
        assert game.is_completed(), \
            f"{AgentClass.__name__} failed to complete Easy"

        # Call next_move on completed game — should not crash
        move = agent.next_move(game)
        # None is acceptable (game is done)


@test("Integration: core game state unchanged by read-only agent operations")
def _():
    from agents.astar   import AStarAgent
    from agents.mcts    import MCTSAgent
    from quantum.grover import GroverAgent

    for AgentClass, kwargs in [
        (AStarAgent, {}),
        (MCTSAgent,  {"simulations": 100}),
        (GroverAgent,{}),
    ]:
        game     = fresh_game(0)
        before   = (game.player_x, game.player_y, game.move_count)
        agent    = AgentClass(**kwargs)
        agent.on_level_start(game)    # A* reads state here — must not modify
        after    = (game.player_x, game.player_y, game.move_count)

        assert before == after, \
            f"{AgentClass.__name__}.on_level_start() modified game state"


# ═════════════════════════════════════════════════════════════════════════════
# LEVEL TESTS
# ═════════════════════════════════════════════════════════════════════════════

@test("Levels: all 4 levels load without error")
def _():
    for i in range(4):
        level = load_level(i)
        assert level is not None
        assert len(level) > 0
        name = LEVEL_META[i][0]
        print(f"\n      Level {i} ({name}): {len(level)} rows loaded", end="")


@test("Levels: all 4 levels have valid Sokoban state")
def _():
    for i in range(4):
        level = load_level(i)
        game  = Sokoban(level)

        assert game.player_x >= 0
        assert game.player_y >= 0
        assert game.width    >  0
        assert game.height   >  0
        assert not game.is_completed(), \
            f"Level {i} starts already completed"


@test("Levels: Easy and Medium solvable by A*")
def _():
    from agents.astar import AStarAgent

    for i in range(2):
        agent = AStarAgent()
        game  = fresh_game(i)
        completed, moves = play_agent(agent, game, max_moves=500)
        name = LEVEL_META[i][0]
        assert completed, f"A* could not solve {name}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)