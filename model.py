"""Policy-value neural network for 9x9 Go."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    """Small CNN with AlphaZero-style policy and value heads."""

    def __init__(self, board_size: int = 9, channels: int = 64, blocks: int = 3):
        super().__init__()
        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size + 1
        self.channels = int(channels)
        self.blocks = int(blocks)

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.residual(self.stem(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value


def save_checkpoint(model: PolicyValueNet, path, optimizer=None, extra=None):
    """Save model weights and minimal architecture metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "board_size": model.board_size,
        "channels": model.channels,
        "blocks": model.blocks,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, device=None) -> PolicyValueNet:
    """Load a ``PolicyValueNet`` checkpoint."""
    checkpoint = torch.load(path, map_location=device or "cpu")
    model = PolicyValueNet(
        board_size=int(checkpoint.get("board_size", 9)),
        channels=int(checkpoint.get("channels", 64)),
        blocks=int(checkpoint.get("blocks", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model
