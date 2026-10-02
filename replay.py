"""Fast console replay helpers for self-play datasets."""

from __future__ import annotations

import os
import time

import numpy as np

from .go_env import BLACK, EMPTY, WHITE


def board_from_encoded_state(state: np.ndarray) -> tuple[np.ndarray, int]:
    """Reconstruct an absolute-color board from one encoded state."""
    state = np.asarray(state)
    own = state[0] > 0.5
    opponent = state[1] > 0.5
    current_player = BLACK if float(np.mean(state[2])) > 0.5 else WHITE

    board = np.zeros(own.shape, dtype=np.int8)
    if current_player == BLACK:
        board[own] = BLACK
        board[opponent] = WHITE
    else:
        board[own] = WHITE
        board[opponent] = BLACK
    return board, current_player


def render_board(board: np.ndarray, current_player: int | None = None, move_number: int | None = None) -> str:
    """Render a board reconstructed from dataset states."""
    stones = {BLACK: "X", WHITE: "O", EMPTY: "."}
    size = int(board.shape[0])
    rows = ["   " + " ".join(str(i) for i in range(size))]
    for row in range(size):
        rows.append(f"{row:2d} " + " ".join(stones[int(board[row, col])] for col in range(size)))
    meta = []
    if move_number is not None:
        meta.append(f"move {move_number}")
    if current_player is not None:
        meta.append("to play: black" if current_player == BLACK else "to play: white")
    if meta:
        rows.append(" | ".join(meta))
    return "\n".join(rows)


def game_slices(total_positions: int, game_lengths=None):
    """Yield ``(start, end)`` slices for each game in a dataset."""
    if game_lengths is None:
        yield 0, total_positions
        return

    start = 0
    for length in game_lengths:
        end = start + int(length)
        yield start, min(end, total_positions)
        start = end
        if start >= total_positions:
            break


def replay_dataset(data_path, delay: float = 0.05, max_games: int | None = None, clear_screen: bool = True):
    """Replay all saved self-play games quickly in the terminal."""
    with np.load(data_path) as data:
        states = data["states"].astype(np.float32)
        values = data["values"].astype(np.float32) if "values" in data else None
        winners = data["winners"].astype(np.int8) if "winners" in data else None
        game_lengths = data["game_lengths"].astype(np.int32) if "game_lengths" in data else None

    slices = list(game_slices(len(states), game_lengths))
    if max_games is not None:
        slices = slices[:max_games]

    for game_index, (start, end) in enumerate(slices, start=1):
        winner_text = ""
        if winners is not None and game_index - 1 < len(winners):
            winner = int(winners[game_index - 1])
            winner_text = "black" if winner == BLACK else "white" if winner == WHITE else "tie"
        for offset, state_index in enumerate(range(start, end), start=1):
            board, current_player = board_from_encoded_state(states[state_index])
            if clear_screen:
                os.system("cls" if os.name == "nt" else "clear")
            print(f"Self-play game {game_index}/{len(slices)}")
            if winner_text:
                print(f"winner: {winner_text}")
            if values is not None:
                print(f"value target: {values[state_index]:+.1f}")
            print(render_board(board, current_player=current_player, move_number=offset))
            if delay > 0:
                time.sleep(delay)
