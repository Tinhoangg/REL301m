"""Run repeated self-play and training until stopped."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import (
        AUTO_CHECKPOINT_DIR,
        AUTO_DATA_DIR,
        AUTO_RESULTS_DIR,
        AUTO_TRAIN_LOG_PATH,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_GAMES,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
        TRAINING_HISTORY_PATH,
        TRAINING_PLOT_PATH,
    )
    from Go_rel.self_play import load_model_if_available, play_game
    from Go_rel.train import plot_training_history, train
else:
    from .config import (
        AUTO_CHECKPOINT_DIR,
        AUTO_DATA_DIR,
        AUTO_RESULTS_DIR,
        AUTO_TRAIN_LOG_PATH,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_GAMES,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_MOVES,
        DEFAULT_SIMULATIONS,
        DEFAULT_SUPPRESS_PASS_UNTIL,
        KOMI,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
        TRAINING_HISTORY_PATH,
        TRAINING_PLOT_PATH,
    )
    from .self_play import load_model_if_available, play_game
    from .train import plot_training_history, train


LOG_FIELDS = [
    "iteration",
    "started_at",
    "finished_at",
    "games",
    "positions",
    "avg_game_length",
    "min_game_length",
    "max_game_length",
    "black_wins",
    "white_wins",
    "draws",
    "pass_rate",
    "avg_pass_policy",
    "final_loss",
    "final_policy_loss",
    "final_value_loss",
    "checkpoint_path",
    "data_path",
    "plot_path",
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iteration_name(iteration: int) -> str:
    return f"iter_{iteration:04d}"


def save_self_play_dataset(games_data, output_path):
    """Save completed self-play games to a compressed dataset."""
    if not games_data:
        return None

    states = np.concatenate([item["states"] for item in games_data], axis=0)
    policies = np.concatenate([item["policies"] for item in games_data], axis=0)
    values = np.concatenate([item["values"] for item in games_data], axis=0)
    actions = np.concatenate([item["actions"] for item in games_data], axis=0)
    winners = np.asarray([item["winner"] for item in games_data], dtype=np.int8)
    game_lengths = np.asarray([len(item["values"]) for item in games_data], dtype=np.int32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states.astype(np.float32),
        policies=policies.astype(np.float32),
        values=values.astype(np.float32),
        actions=actions.astype(np.int32),
        winners=winners,
        game_lengths=game_lengths,
    )
    return output_path


def self_play_iteration(
    iteration: int,
    games: int,
    simulations: int,
    checkpoint_path,
    output_path,
    latest_output_path=LATEST_DATA_PATH,
    size: int = 9,
    komi: float = KOMI,
    max_moves: int = DEFAULT_MAX_MOVES,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    """Generate one self-play dataset and return stats."""
    model = load_model_if_available(checkpoint_path, size=size)
    completed = []

    try:
        for game in range(1, games + 1):
            states, policies, values, winner, actions = play_game(
                model=model,
                size=size,
                komi=komi,
                simulations=simulations,
                max_moves=max_moves,
                suppress_pass_until=suppress_pass_until,
                return_actions=True,
            )
            completed.append(
                {
                    "states": states,
                    "policies": policies,
                    "values": values,
                    "winner": winner,
                    "actions": actions,
                }
            )
            print(f"iteration {iteration:4d} | game {game:3d}/{games} | moves {len(values):3d} | winner {winner:+d}")
    except KeyboardInterrupt:
        interrupted_path = Path(output_path).parent / "interrupted_self_play.npz"
        saved = save_self_play_dataset(completed, interrupted_path)
        if saved is not None:
            print(f"saved interrupted self-play dataset to {saved}")
        raise

    saved_path = save_self_play_dataset(completed, output_path)
    if latest_output_path is not None and saved_path is not None:
        Path(latest_output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_path, latest_output_path)
        print(f"updated latest self-play dataset: {latest_output_path}")
    return saved_path, self_play_stats(completed, size * size)


def self_play_stats(games_data, pass_action: int):
    """Compute self-play summary values for logging and training curves."""
    if not games_data:
        return {
            "games": 0,
            "positions": 0,
            "avg_game_length": 0.0,
            "min_game_length": 0,
            "max_game_length": 0,
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "pass_rate": 0.0,
            "avg_pass_policy": 0.0,
        }

    lengths = np.asarray([len(item["values"]) for item in games_data], dtype=np.float32)
    winners = np.asarray([item["winner"] for item in games_data], dtype=np.int8)
    actions = np.concatenate([item["actions"] for item in games_data], axis=0)
    policies = np.concatenate([item["policies"] for item in games_data], axis=0)
    return {
        "games": len(games_data),
        "positions": int(lengths.sum()),
        "avg_game_length": float(lengths.mean()),
        "min_game_length": int(lengths.min()),
        "max_game_length": int(lengths.max()),
        "black_wins": int(np.sum(winners == 1)),
        "white_wins": int(np.sum(winners == -1)),
        "draws": int(np.sum(winners == 0)),
        "pass_rate": float(np.mean(actions == pass_action)) if len(actions) else 0.0,
        "avg_pass_policy": float(np.mean(policies[:, pass_action])) if len(policies) else 0.0,
    }


def save_iteration_history(history, path, iteration: int, lr: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "epoch", "loss", "policy_loss", "value_loss", "learning_rate"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "iteration": iteration,
                    "epoch": row["epoch"],
                    "loss": row["loss"],
                    "policy_loss": row["policy_loss"],
                    "value_loss": row["value_loss"],
                    "learning_rate": lr,
                }
            )


def append_auto_train_log(row, log_path=AUTO_TRAIN_LOG_PATH):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def copy_if_exists(source, destination):
    source = Path(source)
    if not source.exists():
        return
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def auto_train(
    games: int = DEFAULT_GAMES,
    simulations: int = DEFAULT_SIMULATIONS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    max_moves: int = DEFAULT_MAX_MOVES,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
    iterations: int = 0,
    checkpoint_in=LATEST_CHECKPOINT_PATH,
    checkpoint_out=LATEST_CHECKPOINT_PATH,
    data_dir=AUTO_DATA_DIR,
    checkpoint_dir=AUTO_CHECKPOINT_DIR,
    results_dir=AUTO_RESULTS_DIR,
    latest_data_path=LATEST_DATA_PATH,
    latest_history_path=TRAINING_HISTORY_PATH,
    latest_plot_path=TRAINING_PLOT_PATH,
    log_path=AUTO_TRAIN_LOG_PATH,
    size: int = 9,
    komi: float = KOMI,
):
    """Run self-play and training repeatedly.

    ``iterations=0`` means run forever until interrupted.
    """
    iteration = 1
    current_checkpoint = Path(checkpoint_in)

    while iterations <= 0 or iteration <= iterations:
        started_at = timestamp()
        stem = iteration_name(iteration)
        data_path = Path(data_dir) / f"self_play_{stem}.npz"
        iter_checkpoint = Path(checkpoint_dir) / f"latest_{stem}.pt"
        iter_history_path = Path(results_dir) / f"training_history_{stem}.csv"
        iter_plot_path = Path(results_dir) / f"training_losses_{stem}.png"
        interrupted_checkpoint = Path(checkpoint_out).parent / "interrupted_latest.pt"

        print(f"\n=== auto-train iteration {iteration} started at {started_at} ===")
        dataset_path, stats = self_play_iteration(
            iteration=iteration,
            games=games,
            simulations=simulations,
            checkpoint_path=current_checkpoint,
            output_path=data_path,
            latest_output_path=latest_data_path,
            size=size,
            komi=komi,
            max_moves=max_moves,
            suppress_pass_until=suppress_pass_until,
        )

        model, history = train(
            data_path=dataset_path,
            save_path=iter_checkpoint,
            init_checkpoint=current_checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            history_path=latest_history_path,
            plot_path=latest_plot_path,
            interrupted_checkpoint_path=interrupted_checkpoint,
            return_history=True,
        )
        del model

        save_iteration_history(history, iter_history_path, iteration=iteration, lr=lr)
        plot_training_history(history, iter_plot_path)
        copy_if_exists(iter_checkpoint, checkpoint_out)

        final = history[-1] if history else {"loss": "", "policy_loss": "", "value_loss": ""}
        finished_at = timestamp()
        append_auto_train_log(
            {
                "iteration": iteration,
                "started_at": started_at,
                "finished_at": finished_at,
                "games": stats["games"],
                "positions": stats["positions"],
                "avg_game_length": stats["avg_game_length"],
                "min_game_length": stats["min_game_length"],
                "max_game_length": stats["max_game_length"],
                "black_wins": stats["black_wins"],
                "white_wins": stats["white_wins"],
                "draws": stats["draws"],
                "pass_rate": stats["pass_rate"],
                "avg_pass_policy": stats["avg_pass_policy"],
                "final_loss": final["loss"],
                "final_policy_loss": final["policy_loss"],
                "final_value_loss": final["value_loss"],
                "checkpoint_path": str(iter_checkpoint),
                "data_path": str(dataset_path),
                "plot_path": str(iter_plot_path),
            },
            log_path=log_path,
        )
        print(f"iteration {iteration} complete | latest checkpoint: {checkpoint_out}")
        current_checkpoint = Path(checkpoint_out)
        iteration += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--suppress-pass-until", type=int, default=DEFAULT_SUPPRESS_PASS_UNTIL)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--checkpoint-in", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--checkpoint-out", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--size", type=int, default=9)
    args = parser.parse_args()

    try:
        auto_train(
            games=args.games,
            simulations=args.simulations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_moves=args.max_moves,
            suppress_pass_until=args.suppress_pass_until,
            iterations=args.iterations,
            checkpoint_in=args.checkpoint_in,
            checkpoint_out=args.checkpoint_out,
            size=args.size,
        )
    except KeyboardInterrupt:
        print("\nauto-train stopped by user")


if __name__ == "__main__":
    main()
