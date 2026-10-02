"""Train the Go policy-value network from self-play data."""

from __future__ import annotations

import argparse
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
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
    )
    from Go_rel.model import PolicyValueNet, save_checkpoint
    from Go_rel.replay import replay_dataset
else:
    from .config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_WEIGHT_DECAY,
        LATEST_CHECKPOINT_PATH,
        LATEST_DATA_PATH,
    )
    from .model import PolicyValueNet, save_checkpoint
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
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    replay_games: bool = False,
    replay_delay: float = 0.03,
    replay_max_games: int | None = None,
    device=None,
):
    if replay_games:
        replay_dataset(data_path, delay=replay_delay, max_games=replay_max_games)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = load_dataset(data_path)
    sample_state, sample_policy, _ = dataset[0]
    board_size = int(sample_state.shape[-1])
    action_size = int(sample_policy.shape[-1])

    model = PolicyValueNet(board_size=board_size)
    if model.action_size != action_size:
        raise ValueError(f"dataset action size {action_size} does not match model action size {model.action_size}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    save_checkpoint(model, save_path, optimizer=optimizer, extra={"data_path": str(data_path), "epochs": epochs})
    print(f"saved checkpoint to {save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(LATEST_DATA_PATH))
    parser.add_argument("--save-path", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--replay-games", action="store_true", help="quickly replay saved self-play games before training")
    parser.add_argument("--replay-delay", type=float, default=0.03)
    parser.add_argument("--replay-max-games", type=int, default=None)
    args = parser.parse_args()
    train(
        data_path=args.data,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        replay_games=args.replay_games,
        replay_delay=args.replay_delay,
        replay_max_games=args.replay_max_games,
    )


if __name__ == "__main__":
    main()
