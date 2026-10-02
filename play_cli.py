"""Play 9x9 Go against the MCTS agent in a terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def load_or_random_model(path):
    path = Path(path)
    if path.exists():
        print(f"loaded checkpoint {path}")
        return load_checkpoint(path)
    print("checkpoint not found; using a random initialized network")
    return PolicyValueNet().eval()


def parse_human_action(text: str, env: GoEnv) -> int:
    text = text.strip().lower()
    if text in {"p", "pass"}:
        return env.pass_action
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("enter row col, or pass")
    row, col = int(parts[0]), int(parts[1])
    return env.coord_to_action(row, col)


def play(simulations: int = DEFAULT_SIMULATIONS, checkpoint=LATEST_CHECKPOINT_PATH, human_color: str = "black", max_moves: int = DEFAULT_MAX_MOVES):
    env = GoEnv(size=9, komi=KOMI)
    model = load_or_random_model(checkpoint)
    mcts = MCTS(NetworkEvaluator(model), num_simulations=simulations)
    human = BLACK if human_color.lower() == "black" else WHITE

    while not env.done and env.move_count < max_moves:
        print()
        print(env.render())
        if env.current_player == human:
            while True:
                raw = input("your move (row col or pass): ")
                try:
                    action = parse_human_action(raw, env)
                    if env.is_legal(action):
                        break
                    print("illegal move")
                except (ValueError, IndexError) as exc:
                    print(exc)
        else:
            action, _ = mcts.select_action(env, temperature=1e-6)
            if action == env.pass_action:
                print("agent plays: pass")
            else:
                row, col = env.action_to_coord(action)
                print(f"agent plays: {row} {col}")
        env.step(action)

    print()
    print(env.render())
    score = env.score()
    winner = env.winner()
    winner_name = "black" if winner == BLACK else "white" if winner == WHITE else "tie"
    print(f"final score | black {score['black']:.1f} | white {score['white']:.1f} | winner {winner_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--human-color", choices=["black", "white"], default="black")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    args = parser.parse_args()
    play(simulations=args.simulations, checkpoint=args.checkpoint, human_color=args.human_color, max_moves=args.max_moves)


if __name__ == "__main__":
    main()
