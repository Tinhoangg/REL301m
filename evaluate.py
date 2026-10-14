"""Evaluate a Go agent against KataGo or a checkpoint baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import (
        BEST_CHECKPOINT_PATH,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
        LATEST_CHECKPOINT_PATH,
    )
    from Go_rel.go_env import BLACK, WHITE, GoEnv
    from Go_rel.katago import KataGoEngine, KataGoError
    from Go_rel.mcts import MCTS, NetworkEvaluator
    from Go_rel.model import PolicyValueNet, load_checkpoint
else:
    from .config import (
        BEST_CHECKPOINT_PATH,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
        LATEST_CHECKPOINT_PATH,
    )
    from .go_env import BLACK, WHITE, GoEnv
    from .katago import KataGoEngine, KataGoError
    from .mcts import MCTS, NetworkEvaluator
    from .model import PolicyValueNet, load_checkpoint


def load_or_random_model(path, size: int = 9, fallback_path=BEST_CHECKPOINT_PATH):
    path = Path(path)
    if path.exists():
        print(f"loaded checkpoint {path}")
        return load_checkpoint(path)
    fallback_path = Path(fallback_path)
    if fallback_path.exists():
        print(f"checkpoint {path} not found; using fallback checkpoint {fallback_path}")
        return load_checkpoint(fallback_path)
    print("checkpoint not found; evaluating a random initialized network")
    return PolicyValueNet(board_size=size).eval()


def load_required_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"opponent checkpoint not found: {path}")
    print(f"loaded opponent checkpoint {path}")
    return load_checkpoint(path)


def select_model_action(model, env, simulations: int, suppress_pass_until: int) -> int:
    mcts = MCTS(
        NetworkEvaluator(model),
        num_simulations=simulations,
        suppress_pass_until=suppress_pass_until,
    )
    action, _ = mcts.select_action(env, temperature=1e-6)
    return int(action)


def play_checkpoint_game(
    model,
    opponent_model,
    model_color: int,
    simulations: int,
    max_moves: int,
    size: int = 9,
    komi: float = KOMI,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    env = GoEnv(size=size, komi=komi)

    while not env.done and env.move_count < max_moves:
        current_model = model if env.current_player == model_color else opponent_model
        action = select_model_action(current_model, env, simulations, suppress_pass_until)
        env.step(action)

    return env.winner(), env.score(), env.move_count, False


def play_katago_game(
    model,
    katago: KataGoEngine,
    model_color: int,
    simulations: int,
    max_moves: int,
    size: int = 9,
    komi: float = KOMI,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    env = GoEnv(size=size, komi=komi)
    katago.new_game()
    resigned = False
    resignation_winner = 0

    while not env.done and env.move_count < max_moves:
        player = env.current_player
        if player == model_color:
            action = select_model_action(model, env, simulations, suppress_pass_until)
            env.step(action)
            katago.play_action(player, action)
        else:
            action = katago.genmove_action(player)
            if action is None:
                resigned = True
                resignation_winner = model_color
                break
            if not env.is_legal(action):
                raise KataGoError(f"KataGo returned illegal action {action} at move {env.move_count}")
            env.step(action)

    winner = resignation_winner if resigned else env.winner()
    return winner, env.score(), env.move_count, resigned


def print_game_result(game: int, games: int, model_color: int, result: str, moves: int, score, resigned: bool):
    color_name = "black" if model_color == BLACK else "white"
    suffix = " | katago resigned" if resigned else ""
    print(
        f"game {game:3d}/{games} | model {color_name:5s} | {result:4s} "
        f"| moves {moves:3d} | B {score['black']:.1f} W {score['white']:.1f}{suffix}"
    )


def evaluate(
    games: int = 10,
    simulations: int = DEFAULT_SIMULATIONS,
    checkpoint=LATEST_CHECKPOINT_PATH,
    opponent: str = "katago",
    opponent_checkpoint=BEST_CHECKPOINT_PATH,
    katago_path=None,
    katago_model=None,
    katago_config=None,
    katago_extra_args: list[str] | None = None,
    max_moves: int = DEFAULT_MAX_MOVES,
    size: int = 9,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    model = load_or_random_model(checkpoint, size=size)
    wins = 0
    losses = 0
    draws = 0

    if opponent == "katago":
        if not katago_path:
            raise ValueError(
                "KataGo evaluation needs --katago-path. You can also set KATAGO_PATH, "
                "KATAGO_MODEL, and KATAGO_CONFIG environment variables."
            )
        print(f"opponent: KataGo ({katago_path})")
        game_runner = None
        engine_context = KataGoEngine(
            katago_path,
            model=katago_model,
            config=katago_config,
            extra_args=katago_extra_args,
            size=size,
            komi=KOMI,
        )
        engine_context.start()
    elif opponent == "checkpoint":
        opponent_model = load_required_model(opponent_checkpoint)
        print(f"opponent: checkpoint ({opponent_checkpoint})")
        game_runner = opponent_model
        engine_context = None
    else:
        raise ValueError(f"unknown opponent: {opponent}")

    try:
        for game in range(1, games + 1):
            model_color = BLACK if game % 2 == 1 else WHITE
            if opponent == "katago":
                winner, score, moves, resigned = play_katago_game(
                    model,
                    engine_context,
                    model_color=model_color,
                    simulations=simulations,
                    max_moves=max_moves,
                    size=size,
                    suppress_pass_until=suppress_pass_until,
                )
            else:
                winner, score, moves, resigned = play_checkpoint_game(
                    model,
                    game_runner,
                    model_color=model_color,
                    simulations=simulations,
                    max_moves=max_moves,
                    size=size,
                    suppress_pass_until=suppress_pass_until,
                )

            if winner == model_color:
                wins += 1
                result = "win"
            elif winner == 0:
                draws += 1
                result = "draw"
            else:
                losses += 1
                result = "loss"
            print_game_result(game, games, model_color, result, moves, score, resigned)
    finally:
        if engine_context is not None:
            engine_context.close()

    win_rate = wins / games if games else 0.0
    print(f"summary | wins {wins} | losses {losses} | draws {draws} | win_rate {win_rate:.2%}")
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--opponent", choices=("katago", "checkpoint"), default="katago")
    parser.add_argument("--opponent-checkpoint", default=str(BEST_CHECKPOINT_PATH))
    parser.add_argument("--katago-path", default=os.environ.get("KATAGO_PATH"))
    parser.add_argument("--katago-model", default=os.environ.get("KATAGO_MODEL"))
    parser.add_argument("--katago-config", default=os.environ.get("KATAGO_CONFIG"))
    parser.add_argument("--katago-arg", action="append", default=[], help="extra argument passed to KataGo")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--suppress-pass-until", type=int, default=DEFAULT_SUPPRESS_PASS_UNTIL)
    args = parser.parse_args()
    evaluate(
        games=args.games,
        simulations=args.simulations,
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        opponent_checkpoint=args.opponent_checkpoint,
        katago_path=args.katago_path,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        katago_extra_args=args.katago_arg,
        max_moves=args.max_moves,
        size=args.size,
        suppress_pass_until=args.suppress_pass_until,
    )


if __name__ == "__main__":
    main()
