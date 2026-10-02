"""Evaluate a Go agent against a random legal-move baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import DEFAULT_MAX_MOVES, DEFAULT_SIMULATIONS, KOMI, LATEST_CHECKPOINT_PATH
    from Go_rel.go_env import BLACK, WHITE, GoEnv
    from Go_rel.mcts import MCTS, NetworkEvaluator
    from Go_rel.model import PolicyValueNet, load_checkpoint
else:
    from .config import DEFAULT_MAX_MOVES, DEFAULT_SIMULATIONS, KOMI, LATEST_CHECKPOINT_PATH
    from .go_env import BLACK, WHITE, GoEnv
    from .mcts import MCTS, NetworkEvaluator
    from .model import PolicyValueNet, load_checkpoint


def load_or_random_model(path, size: int = 9):
    path = Path(path)
    if path.exists():
        print(f"loaded checkpoint {path}")
        return load_checkpoint(path)
    print("checkpoint not found; evaluating a random initialized network")
    return PolicyValueNet(board_size=size).eval()


def random_action(env):
    legal = env.legal_actions()
    non_pass = [action for action in legal if action != env.pass_action]
    choices = non_pass if non_pass else legal
    return int(np.random.choice(choices))


def play_eval_game(model, model_color: int, simulations: int, max_moves: int, size: int = 9, komi: float = KOMI):
    env = GoEnv(size=size, komi=komi)
    mcts = MCTS(NetworkEvaluator(model), num_simulations=simulations)

    while not env.done and env.move_count < max_moves:
        if env.current_player == model_color:
            action, _ = mcts.select_action(env, temperature=1e-6)
        else:
            action = random_action(env)
        env.step(action)

    return env.winner(), env.score(), env.move_count


def evaluate(games: int = 10, simulations: int = DEFAULT_SIMULATIONS, checkpoint=LATEST_CHECKPOINT_PATH, max_moves: int = DEFAULT_MAX_MOVES):
    model = load_or_random_model(checkpoint)
    wins = 0
    losses = 0
    draws = 0

    for game in range(1, games + 1):
        model_color = BLACK if game % 2 == 1 else WHITE
        winner, score, moves = play_eval_game(model, model_color, simulations, max_moves)
        if winner == model_color:
            wins += 1
            result = "win"
        elif winner == 0:
            draws += 1
            result = "draw"
        else:
            losses += 1
            result = "loss"
        color_name = "black" if model_color == BLACK else "white"
        print(f"game {game:3d}/{games} | model {color_name:5s} | {result:4s} | moves {moves:3d} | B {score['black']:.1f} W {score['white']:.1f}")

    win_rate = wins / games if games else 0.0
    print(f"summary | wins {wins} | losses {losses} | draws {draws} | win_rate {win_rate:.2%}")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    args = parser.parse_args()
    evaluate(games=args.games, simulations=args.simulations, checkpoint=args.checkpoint, max_moves=args.max_moves)


if __name__ == "__main__":
    main()
