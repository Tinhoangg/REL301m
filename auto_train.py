"""Run repeated self-play and training until stopped."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
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
        AUTO_TRAIN_STATE_PATH,
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
        AUTO_TRAIN_STATE_PATH,
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


def atomic_write_json(payload, path):
    """Write resume metadata without leaving a half-written state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    temporary.replace(path)


def load_auto_train_state(path=AUTO_TRAIN_STATE_PATH):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid auto-train state file {path}: {exc}") from exc
    if not isinstance(state, dict) or "iteration" not in state or "phase" not in state:
        raise ValueError(f"invalid auto-train state file {path}: missing iteration or phase")
    if state["phase"] not in {"self_play", "training"}:
        raise ValueError(f"invalid auto-train phase {state['phase']!r} in {path}")
    state["iteration"] = int(state["iteration"])
    return state


def save_auto_train_state(state, path=AUTO_TRAIN_STATE_PATH):
    atomic_write_json(state, path)


def remove_if_exists(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def _iteration_numbers(directory, pattern):
    directory = Path(directory)
    if not directory.exists():
        return set()
    regex = re.compile(pattern)
    numbers = set()
    for path in directory.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def completed_log_iterations(log_path=AUTO_TRAIN_LOG_PATH):
    """Return completed iteration ids and the newest logged completion time."""
    log_path = Path(log_path)
    if not log_path.exists():
        return set(), None
    completed = set()
    newest_finished_at = None
    with log_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                completed.add(int(row["iteration"]))
            except (KeyError, TypeError, ValueError):
                continue
            value = row.get("finished_at")
            if value:
                try:
                    finished_at = datetime.fromisoformat(value)
                except ValueError:
                    continue
                if newest_finished_at is None or finished_at > newest_finished_at:
                    newest_finished_at = finished_at
    return completed, newest_finished_at


def discover_last_completed_iteration(
    log_path=AUTO_TRAIN_LOG_PATH,
    data_dir=AUTO_DATA_DIR,
    checkpoint_dir=AUTO_CHECKPOINT_DIR,
    results_dir=AUTO_RESULTS_DIR,
):
    """Find the highest completed iteration without treating partial output as complete."""
    logged, _ = completed_log_iterations(log_path)
    datasets = _iteration_numbers(data_dir, r"self_play_iter_(\d+)\.npz")
    checkpoints = _iteration_numbers(checkpoint_dir, r"latest_iter_(\d+)\.pt")
    histories = _iteration_numbers(results_dir, r"training_history_iter_(\d+)\.csv")
    plots = _iteration_numbers(results_dir, r"training_losses_iter_(\d+)\.png")
    complete_artifacts = datasets & checkpoints & histories & plots
    completed = logged | complete_artifacts
    return max(completed, default=0)


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
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    np.savez_compressed(
        temporary,
        states=states.astype(np.float32),
        policies=policies.astype(np.float32),
        values=values.astype(np.float32),
        actions=actions.astype(np.int32),
        winners=winners,
        game_lengths=game_lengths,
    )
    temporary.replace(output_path)
    return output_path


def load_self_play_games(path):
    """Load a saved self-play dataset and split it back into completed games."""
    path = Path(path)
    try:
        with np.load(path) as data:
            required = {"states", "policies", "values", "actions", "winners", "game_lengths"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"missing arrays: {', '.join(sorted(missing))}")
            states = data["states"].astype(np.float32)
            policies = data["policies"].astype(np.float32)
            values = data["values"].astype(np.float32)
            actions = data["actions"].astype(np.int32)
            winners = data["winners"].astype(np.int8)
            game_lengths = data["game_lengths"].astype(np.int64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid self-play resume dataset {path}: {exc}") from exc

    positions = len(states)
    if len(policies) != positions or len(values) != positions or len(actions) != positions:
        raise ValueError(f"invalid self-play resume dataset {path}: position arrays have different lengths")
    if len(winners) != len(game_lengths):
        raise ValueError(f"invalid self-play resume dataset {path}: winners and game_lengths differ")
    if np.any(game_lengths < 0) or int(game_lengths.sum()) != positions:
        raise ValueError(f"invalid self-play resume dataset {path}: game_lengths do not match positions")

    games_data = []
    offset = 0
    for winner, length in zip(winners, game_lengths):
        end = offset + int(length)
        games_data.append(
            {
                "states": states[offset:end],
                "policies": policies[offset:end],
                "values": values[offset:end],
                "actions": actions[offset:end],
                "winner": int(winner),
            }
        )
        offset = end
    return games_data


def self_play_iteration(
    iteration: int,
    games: int,
    simulations: int,
    checkpoint_path,
    output_path,
    latest_output_path=LATEST_DATA_PATH,
    partial_path=None,
    resume_path=None,
    progress_callback=None,
    size: int = 9,
    komi: float = KOMI,
    max_moves: int = DEFAULT_MAX_MOVES,
    suppress_pass_until: int = DEFAULT_SUPPRESS_PASS_UNTIL,
):
    """Generate one self-play dataset and return stats."""
    model = load_model_if_available(checkpoint_path, size=size)
    partial_path = Path(partial_path) if partial_path is not None else Path(output_path).with_suffix(".partial.npz")
    resume_path = Path(resume_path) if resume_path is not None else partial_path
    completed = load_self_play_games(resume_path) if resume_path.exists() else []
    if len(completed) > games:
        raise ValueError(
            f"resume dataset {resume_path} contains {len(completed)} games, "
            f"but this iteration is configured for {games}"
        )
    if completed:
        sample_state = completed[0]["states"]
        sample_policy = completed[0]["policies"]
        expected_action_size = size * size + 1
        if sample_state.ndim != 4 or sample_state.shape[-2:] != (size, size):
            raise ValueError(f"resume dataset {resume_path} does not match board size {size}")
        if sample_policy.ndim != 2 or sample_policy.shape[-1] != expected_action_size:
            raise ValueError(f"resume dataset {resume_path} does not match action size {expected_action_size}")
        print(f"resuming iteration {iteration} with {len(completed)}/{games} self-play games complete")

    try:
        for game in range(len(completed) + 1, games + 1):
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
            save_self_play_dataset(completed, partial_path)
            if progress_callback is not None:
                progress_callback(len(completed))
            print(f"iteration {iteration:4d} | game {game:3d}/{games} | moves {len(values):3d} | winner {winner:+d}")
    except KeyboardInterrupt:
        saved = save_self_play_dataset(completed, partial_path)
        if saved is not None:
            print(f"saved interrupted self-play dataset to {saved}")
        raise

    saved_path = save_self_play_dataset(completed, output_path)
    if latest_output_path is not None and saved_path is not None:
        Path(latest_output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_path, latest_output_path)
        print(f"updated latest self-play dataset: {latest_output_path}")
    remove_if_exists(partial_path)
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
    """Insert or replace one iteration row, keeping the log ordered and duplicate-free."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    iteration = int(row["iteration"])
    valid_rows = []
    for item in rows:
        try:
            existing_iteration = int(item.get("iteration", -1))
        except (TypeError, ValueError):
            continue
        if existing_iteration != iteration:
            valid_rows.append(item)
    rows = valid_rows
    rows.append({field: row.get(field, "") for field in LOG_FIELDS})
    rows.sort(key=lambda item: int(item["iteration"]))

    temporary = log_path.with_name(f"{log_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(log_path)


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
    state_path=None,
    size: int = 9,
    komi: float = KOMI,
):
    """Run self-play and training repeatedly.

    ``iterations=0`` means run forever until interrupted.
    """
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    checkpoint_out = Path(checkpoint_out)
    log_path = Path(log_path)
    state_path = Path(state_path) if state_path is not None else results_dir / AUTO_TRAIN_STATE_PATH.name
    requested_config = {
        "games": int(games),
        "simulations": int(simulations),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "max_moves": int(max_moves),
        "suppress_pass_until": int(suppress_pass_until),
        "size": int(size),
        "komi": float(komi),
    }

    last_completed = discover_last_completed_iteration(
        log_path=log_path,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )
    state = load_auto_train_state(state_path)
    if state is not None and state["iteration"] <= last_completed:
        print(f"discarding stale auto-train state for completed iteration {state['iteration']}")
        remove_if_exists(state_path)
        state = None

    iteration = state["iteration"] if state is not None else last_completed + 1
    if state is not None:
        print(f"found unfinished auto-train iteration {iteration} ({state['phase']})")

    if last_completed > 0 and checkpoint_out.exists():
        current_checkpoint = checkpoint_out
    elif last_completed > 0 and (checkpoint_dir / f"latest_{iteration_name(last_completed)}.pt").exists():
        current_checkpoint = checkpoint_dir / f"latest_{iteration_name(last_completed)}.pt"
    else:
        current_checkpoint = Path(checkpoint_in)
    completed_this_run = 0

    while iterations <= 0 or completed_this_run < iterations:
        stem = iteration_name(iteration)
        data_path = data_dir / f"self_play_{stem}.npz"
        partial_path = data_dir / f"self_play_{stem}.partial.npz"
        legacy_partial_path = data_dir / "interrupted_self_play.npz"
        iter_checkpoint = checkpoint_dir / f"latest_{stem}.pt"
        iter_history_path = results_dir / f"training_history_{stem}.csv"
        iter_plot_path = results_dir / f"training_losses_{stem}.png"
        interrupted_checkpoint = checkpoint_dir / f"interrupted_{stem}.pt"
        legacy_interrupted_checkpoint = checkpoint_out.parent / "interrupted_latest.pt"

        state_created_now = state is None or state["iteration"] != iteration
        if state_created_now:
            state = {
                "version": 1,
                "iteration": iteration,
                "phase": "training" if data_path.exists() else "self_play",
                "started_at": timestamp(),
                "checkpoint_in": str(current_checkpoint.resolve()),
                "config": requested_config,
                "games_completed": 0,
                "epochs_completed": 0,
            }
            save_auto_train_state(state, state_path)
        elif data_path.exists() and state["phase"] == "self_play":
            state["phase"] = "training"
            save_auto_train_state(state, state_path)

        iteration_config = dict(requested_config)
        iteration_config.update(state.get("config", {}))
        started_at = state.get("started_at", timestamp())
        iteration_checkpoint_in = Path(state.get("checkpoint_in", current_checkpoint))

        print(f"\n=== auto-train iteration {iteration} started at {started_at} ===")
        if state["phase"] == "self_play":
            resume_path = partial_path
            if not resume_path.exists() and legacy_partial_path.exists():
                _, newest_finished_at = completed_log_iterations(log_path)
                legacy_modified_at = datetime.fromtimestamp(legacy_partial_path.stat().st_mtime)
                if newest_finished_at is None or legacy_modified_at > newest_finished_at:
                    resume_path = legacy_partial_path
                    print(f"migrating legacy interrupted self-play data from {legacy_partial_path}")

            def record_game_progress(completed_games):
                state["games_completed"] = int(completed_games)
                save_auto_train_state(state, state_path)

            try:
                dataset_path, stats = self_play_iteration(
                    iteration=iteration,
                    games=int(iteration_config["games"]),
                    simulations=int(iteration_config["simulations"]),
                    checkpoint_path=iteration_checkpoint_in,
                    output_path=data_path,
                    latest_output_path=latest_data_path,
                    partial_path=partial_path,
                    resume_path=resume_path,
                    progress_callback=record_game_progress,
                    size=int(iteration_config["size"]),
                    komi=float(iteration_config["komi"]),
                    max_moves=int(iteration_config["max_moves"]),
                    suppress_pass_until=int(iteration_config["suppress_pass_until"]),
                )
            except ValueError:
                if state_created_now and not partial_path.exists():
                    remove_if_exists(state_path)
                raise
            if resume_path == legacy_partial_path:
                remove_if_exists(legacy_partial_path)
            state["phase"] = "training"
            state["games_completed"] = int(stats["games"])
            save_auto_train_state(state, state_path)
        else:
            if not data_path.exists():
                raise ValueError(f"cannot resume training iteration {iteration}: missing dataset {data_path}")
            dataset_path = data_path
            resumed_games = load_self_play_games(dataset_path)
            expected_games = int(iteration_config["games"])
            if len(resumed_games) != expected_games:
                raise ValueError(
                    f"iteration {iteration} dataset contains {len(resumed_games)} games, "
                    f"but its saved configuration expects {expected_games}"
                )
            stats = self_play_stats(resumed_games, int(iteration_config["size"]) ** 2)

        resume_checkpoint = None
        if interrupted_checkpoint.exists():
            resume_checkpoint = interrupted_checkpoint
        elif (
            legacy_interrupted_checkpoint.exists()
            and legacy_interrupted_checkpoint.stat().st_mtime >= Path(dataset_path).stat().st_mtime
        ):
            resume_checkpoint = legacy_interrupted_checkpoint

        def record_epoch_progress(completed_epoch, _history):
            state["epochs_completed"] = int(completed_epoch)
            save_auto_train_state(state, state_path)

        model, history = train(
            data_path=dataset_path,
            save_path=iter_checkpoint,
            init_checkpoint=iteration_checkpoint_in,
            epochs=int(iteration_config["epochs"]),
            batch_size=int(iteration_config["batch_size"]),
            lr=float(iteration_config["lr"]),
            history_path=latest_history_path,
            plot_path=latest_plot_path,
            interrupted_checkpoint_path=interrupted_checkpoint,
            resume_checkpoint=resume_checkpoint,
            epoch_callback=record_epoch_progress,
            return_history=True,
        )
        del model

        save_iteration_history(history, iter_history_path, iteration=iteration, lr=float(iteration_config["lr"]))
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
        remove_if_exists(interrupted_checkpoint)
        if resume_checkpoint == legacy_interrupted_checkpoint:
            remove_if_exists(legacy_interrupted_checkpoint)
        remove_if_exists(partial_path)
        remove_if_exists(state_path)
        current_checkpoint = checkpoint_out
        state = None
        iteration += 1
        completed_this_run += 1


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
