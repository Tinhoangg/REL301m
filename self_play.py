"""Generate self-play data for the Go policy-value network."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import (
        DEFAULT_C_PUCT,
        DEFAULT_GAMES,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_TEMPERATURE_MOVES,
        KOMI,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
    )
    from Go_rel.go_env import GoEnv
    from Go_rel.mcts import MCTS, NetworkEvaluator
    from Go_rel.model import load_checkpoint
    from Go_rel.replay import replay_dataset
else:
    from .config import (
        DEFAULT_C_PUCT,
        DEFAULT_GAMES,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_TEMPERATURE_MOVES,
        KOMI,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
    )
    from .go_env import GoEnv
    from .mcts import MCTS, NetworkEvaluator
    from .model import load_checkpoint
    from .replay import replay_dataset


def load_model_if_available(path):
    path = Path(path)
    if path.exists():
        return load_checkpoint(path)
    return None


def play_game(model=None, size: int = 9, komi: float = KOMI, simulations: int = DEFAULT_SIMULATIONS, max_moves: int = DEFAULT_MAX_MOVES):
    env = GoEnv(size=size, komi=komi)
    mcts = MCTS(NetworkEvaluator(model), num_simulations=simulations, c_puct=DEFAULT_C_PUCT)
    states = []
    policies = []
    players = []

    while not env.done and env.move_count < max_moves:
        states.append(env.encode())
        players.append(env.current_player)
        temperature = 1.0 if env.move_count < DEFAULT_TEMPERATURE_MOVES else 1e-6
        action, policy = mcts.select_action(env, temperature=temperature, add_noise=True)
        policies.append(policy)
        env.step(action)

    winner = env.winner()
    values = [0.0 if winner == 0 else (1.0 if player == winner else -1.0) for player in players]
    return np.asarray(states, dtype=np.float32), np.asarray(policies, dtype=np.float32), np.asarray(values, dtype=np.float32), winner


def generate_self_play(
    games: int = DEFAULT_GAMES,
    simulations: int = DEFAULT_SIMULATIONS,
    output_path=LATEST_DATA_PATH,
    checkpoint_path=LATEST_CHECKPOINT_PATH,
    size: int = 9,
    komi: float = KOMI,
    max_moves: int = DEFAULT_MAX_MOVES,
):
    model = load_model_if_available(checkpoint_path)
    all_states = []
    all_policies = []
    all_values = []
    winners = []
    game_lengths = []

    for game in range(1, games + 1):
        states, policies, values, winner = play_game(
            model=model,
            size=size,
            komi=komi,
            simulations=simulations,
            max_moves=max_moves,
        )
        all_states.append(states)
        all_policies.append(policies)
        all_values.append(values)
        winners.append(winner)
        game_lengths.append(len(values))
        print(f"game {game:3d}/{games} | moves {len(values):3d} | winner {winner:+d}")

    states = np.concatenate(all_states, axis=0)
    policies = np.concatenate(all_policies, axis=0)
    values = np.concatenate(all_values, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        policies=policies,
        values=values,
        winners=np.asarray(winners, dtype=np.int8),
        game_lengths=np.asarray(game_lengths, dtype=np.int32),
    )
    print(f"saved {len(values)} positions to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--output", default=str(LATEST_DATA_PATH))
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--replay", action="store_true", help="quickly replay generated games in the terminal")
    parser.add_argument("--replay-delay", type=float, default=0.05)
    args = parser.parse_args()
    output_path = generate_self_play(
        games=args.games,
        simulations=args.simulations,
        max_moves=args.max_moves,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    if args.replay:
        replay_dataset(output_path, delay=args.replay_delay)


if __name__ == "__main__":
    main()
