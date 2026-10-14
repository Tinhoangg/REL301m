"""Train the Go policy-value network from self-play data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_WEIGHT_DECAY,
        BEST_CHECKPOINT_PATH,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
        TRAINING_HISTORY_PATH,
        TRAINING_PLOT_PATH,
    )
    from Go_rel.model import PolicyValueNet, load_checkpoint, save_checkpoint
    from Go_rel.replay import replay_dataset
else:
    from .config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_WEIGHT_DECAY,
        BEST_CHECKPOINT_PATH,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
        TRAINING_HISTORY_PATH,
        TRAINING_PLOT_PATH,
    )
    from .model import PolicyValueNet, load_checkpoint, save_checkpoint
    from .replay import replay_dataset


def load_dataset(path):
    with np.load(path) as data:
        states = torch.from_numpy(data["states"].astype(np.float32))
        policies = torch.from_numpy(data["policies"].astype(np.float32))
        values = torch.from_numpy(data["values"].astype(np.float32))
    return TensorDataset(states, policies, values)


def train(
    data_path=LATEST_DATA_PATH,
    save_path=LATEST_CHECKPOINT_PATH,
    init_checkpoint=None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    replay_games: bool = False,
    replay_delay: float = 0.03,
    replay_max_games: int | None = None,
    history_path=TRAINING_HISTORY_PATH,
    plot_path=TRAINING_PLOT_PATH,
    save_plot: bool = True,
    interrupted_checkpoint_path=None,
    return_history: bool = False,
    device=None,
):
    if replay_games:
        replay_dataset(data_path, delay=replay_delay, max_games=replay_max_games)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = load_dataset(data_path)
    sample_state, sample_policy, _ = dataset[0]
    board_size = int(sample_state.shape[-1])
    action_size = int(sample_policy.shape[-1])

    init_path = choose_init_checkpoint(init_checkpoint, save_path)
    model = None
    if init_path is not None:
        candidate_model = load_checkpoint(init_path, device=device)
        if candidate_model.action_size == action_size and candidate_model.board_size == board_size:
            model = candidate_model
            print(f"loaded initial checkpoint from {init_path}")
        else:
            print(
                f"skipped checkpoint {init_path}: model action size {candidate_model.action_size} "
                f"does not match dataset action size {action_size}"
            )
    if model is None:
        model = PolicyValueNet(board_size=board_size)
        print("training from a random initialized model")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            for states, target_policies, target_values in loader:
                states = states.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)

                logits, values = model(states)
                policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(values, target_values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * states.size(0)
                total_policy_loss += float(policy_loss.item()) * states.size(0)
                total_value_loss += float(value_loss.item()) * states.size(0)

            n = len(dataset)
            print(
                f"epoch {epoch:3d}/{epochs} | loss {total_loss / n:.4f} "
                f"| policy {total_policy_loss / n:.4f} | value {total_value_loss / n:.4f}"
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / n,
                    "policy_loss": total_policy_loss / n,
                    "value_loss": total_value_loss / n,
                }
            )
    except KeyboardInterrupt:
        if interrupted_checkpoint_path is not None:
            save_checkpoint(
                model,
                interrupted_checkpoint_path,
                optimizer=optimizer,
                extra={"data_path": str(data_path), "epochs_completed": len(history), "interrupted": True},
            )
            print(f"saved interrupted checkpoint to {interrupted_checkpoint_path}")
        if history:
            save_training_history(history, history_path)
            print(f"saved interrupted training history to {history_path}")
            if save_plot:
                plot_training_history(history, plot_path)
                print(f"saved interrupted training plot to {plot_path}")
        raise

    save_checkpoint(model, save_path, optimizer=optimizer, extra={"data_path": str(data_path), "epochs": epochs})
    print(f"saved checkpoint to {save_path}")
    save_training_history(history, history_path)
    print(f"saved training history to {history_path}")
    if save_plot:
        plot_training_history(history, plot_path)
        print(f"saved training plot to {plot_path}")
    if return_history:
        return model, history
    return model


def save_training_history(history, path):
    """Save epoch loss values as CSV so they can be reused in reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "policy_loss", "value_loss"])
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history, path):
    """Create a PNG chart for loss, policy loss, and value loss."""
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    policy_losses = [row["policy_loss"] for row in history]
    value_losses = [row["value_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(epochs, losses, marker="o", linewidth=2.4, label="Total loss")
    ax.plot(epochs, policy_losses, marker="o", linewidth=2.0, label="Policy loss")
    ax.plot(epochs, value_losses, marker="o", linewidth=2.0, label="Value loss")
    ax.set_title("Training Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def choose_init_checkpoint(init_checkpoint, save_path):
    """Pick a checkpoint to continue training from, if one exists."""
    candidates = []
    if init_checkpoint:
        candidates.append(Path(init_checkpoint))
    candidates.append(Path(save_path))
    candidates.append(Path(BEST_CHECKPOINT_PATH))
    candidates.append(Path(LATEST_CHECKPOINT_PATH))
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(LATEST_DATA_PATH))
    parser.add_argument("--save-path", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--replay-games", action="store_true", help="quickly replay saved self-play games before training")
    parser.add_argument("--replay-delay", type=float, default=0.03)
    parser.add_argument("--replay-max-games", type=int, default=None)
    parser.add_argument("--history-path", default=str(TRAINING_HISTORY_PATH))
    parser.add_argument("--plot-path", default=str(TRAINING_PLOT_PATH))
    parser.add_argument("--no-plot", action="store_true", help="skip saving the training loss PNG")
    parser.add_argument("--interrupted-checkpoint-path", default=None)
    args = parser.parse_args()
    train(
        data_path=args.data,
        save_path=args.save_path,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        replay_games=args.replay_games,
        replay_delay=args.replay_delay,
        replay_max_games=args.replay_max_games,
        history_path=args.history_path,
        plot_path=args.plot_path,
        save_plot=not args.no_plot,
        interrupted_checkpoint_path=args.interrupted_checkpoint_path,
    )


if __name__ == "__main__":
    main()
