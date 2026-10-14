"""Arena matches between a candidate model and the current best model."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import (
        BEST_CHECKPOINT_PATH,
        CANDIDATE_CHECKPOINT_PATH,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
    )
    from Go_rel.go_env import BLACK, WHITE, GoEnv
    from Go_rel.mcts import MCTS, NetworkEvaluator
    from Go_rel.model import PolicyValueNet, load_checkpoint, save_checkpoint
else:
    from .config import (
        BEST_CHECKPOINT_PATH,
        CANDIDATE_CHECKPOINT_PATH,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
    )
    from .go_env import BLACK, WHITE, GoEnv
    from .mcts import MCTS, NetworkEvaluator
    from .model import PolicyValueNet, load_checkpoint, save_checkpoint


def ensure_checkpoint(path, size: int = 9):
    """Create a random checkpoint if ``path`` does not exist."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNet(board_size=size).eval()
    save_checkpoint(model, path, extra={"created_by": "arena.ensure_checkpoint"})
    print(f"created initial random checkpoint at {path}")
    return path


def load_model(path, size: int = 9):
    path = ensure_checkpoint(path, size=size)
    return load_checkpoint(path)


def play_match_game(
    candidate_model,
    best_model,
    candidate_color: int,
    simulations: int,
    max_moves: int,
    size: int = 9,
    komi: float = KOMI,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    """Play one candidate-vs-best game and return winner details."""
    env = GoEnv(size=size, komi=komi)
    candidate_mcts = MCTS(
        NetworkEvaluator(candidate_model),
        num_simulations=simulations,
        suppress_pass_until=suppress_pass_until,
    )
    best_mcts = MCTS(
        NetworkEvaluator(best_model),
        num_simulations=simulations,
        suppress_pass_until=suppress_pass_until,
    )

    while not env.done and env.move_count < max_moves:
        if env.current_player == candidate_color:
            action, _ = candidate_mcts.select_action(env, temperature=1e-6)
        else:
            action, _ = best_mcts.select_action(env, temperature=1e-6)
        env.step(action)

    winner = env.winner()
    score = env.score()
    if winner == 0:
        result = "draw"
    elif winner == candidate_color:
        result = "candidate_win"
    else:
        result = "best_win"
    return result, winner, score, env.move_count


def arena(
    candidate_path=CANDIDATE_CHECKPOINT_PATH,
    best_path=BEST_CHECKPOINT_PATH,
    games: int = 20,
    simulations: int = DEFAULT_SIMULATIONS,
    threshold: float = 0.55,
    max_moves: int = DEFAULT_MAX_MOVES,
    size: int = 9,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
    promote: bool = True,
):
    """Evaluate candidate against best and optionally promote candidate."""
    candidate_path = Path(candidate_path)
    best_path = Path(best_path)
    ensure_checkpoint(candidate_path, size=size)
    ensure_checkpoint(best_path, size=size)

    candidate_model = load_checkpoint(candidate_path)
    best_model = load_checkpoint(best_path)

    candidate_wins = 0
    best_wins = 0
    draws = 0

    for game in range(1, games + 1):
        candidate_color = BLACK if game % 2 == 1 else WHITE
        result, winner, score, moves = play_match_game(
            candidate_model,
            best_model,
            candidate_color=candidate_color,
            simulations=simulations,
            max_moves=max_moves,
            size=size,
            suppress_pass_until=suppress_pass_until,
        )
        if result == "candidate_win":
            candidate_wins += 1
        elif result == "best_win":
            best_wins += 1
        else:
            draws += 1
        color_name = "black" if candidate_color == BLACK else "white"
        print(
            f"game {game:3d}/{games} | candidate {color_name:5s} | {result:13s} "
            f"| moves {moves:3d} | B {score['black']:.1f} W {score['white']:.1f}"
        )

    decisive_games = candidate_wins + best_wins
    win_rate = candidate_wins / decisive_games if decisive_games else 0.0
    promoted = promote and win_rate >= threshold

    print(
        f"summary | candidate_wins {candidate_wins} | best_wins {best_wins} "
        f"| draws {draws} | candidate_win_rate {win_rate:.2%} | threshold {threshold:.2%}"
    )

    if promoted:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, best_path)
        print(f"promoted candidate to best: {best_path}")
    else:
        print("candidate not promoted")

    return {
        "candidate_wins": candidate_wins,
        "best_wins": best_wins,
        "draws": draws,
        "candidate_win_rate": win_rate,
        "promoted": promoted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(CANDIDATE_CHECKPOINT_PATH))
    parser.add_argument("--best", default=str(BEST_CHECKPOINT_PATH))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--suppress-pass-until", type=int, default=DEFAULT_SUPPRESS_PASS_UNTIL)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    arena(
        candidate_path=args.candidate,
        best_path=args.best,
        games=args.games,
        simulations=args.simulations,
        threshold=args.threshold,
        max_moves=args.max_moves,
        size=args.size,
        suppress_pass_until=args.suppress_pass_until,
        promote=not args.no_promote,
    )


if __name__ == "__main__":
    main()
