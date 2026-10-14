"""Tromp-Taylor Go environment used by self-play and MCTS."""

from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0
PASS_ACTION = 81


class GoEnv:
    """Small Go environment with area scoring and positional superko.

    For a board of size ``N``, actions ``0`` through ``N*N - 1`` map to board
    coordinates and action ``N*N`` means pass.
    """

    def __init__(self, size: int = 9, komi: float = 7.5):
        if size < 2:
            raise ValueError("size must be at least 2")
        self.size = int(size)
        self.komi = float(komi)
        self.pass_action = self.size * self.size
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = BLACK
        self.done = False
        self.consecutive_passes = 0
        self.move_count = 0
        self._history = set()
        self.reset()

    def reset(self, seed=None) -> np.ndarray:
        """Reset the game and return the encoded observation."""
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = BLACK
        self.done = False
        self.consecutive_passes = 0
        self.move_count = 0
        self._history = {self._board_key(self.board)}
        return self.encode()

    def step(self, action: int):
        """Apply one move and return ``(observation, reward, done, info)``."""
        if self.done:
            raise ValueError("cannot step after the game is done")
        if not self.is_legal(action):
            raise ValueError(f"illegal action: {action}")

        player = self.current_player
        info = {"player": player, "action": int(action), "captured": 0, "pass": False}

        if int(action) == self.pass_action:
            self.consecutive_passes += 1
            self.done = self.consecutive_passes >= 2
            info["pass"] = True
        else:
            row, col = self.action_to_coord(action)
            self.board, captured = self._play_on_board(self.board, player, row, col)
            self._history.add(self._board_key(self.board))
            self.consecutive_passes = 0
            info["captured"] = captured

        self.move_count += 1

        reward = 0.0
        if self.done:
            win_color = self.winner()
            reward = 0.0 if win_color == 0 else (1.0 if win_color == player else -1.0)
            info["score"] = self.score()
            info["winner"] = win_color
        else:
            self.current_player = -self.current_player

        return self.encode(), reward, self.done, info

    def legal_actions(self) -> list[int]:
        """Return all legal moves for the current player."""
        if self.done:
            return []
        actions = [action for action in range(self.pass_action) if self.is_legal(action)]
        actions.append(self.pass_action)
        return actions

    def is_legal(self, action: int) -> bool:
        """Return whether ``action`` is legal in the current position."""
        if self.done or not isinstance(action, (int, np.integer)):
            return False
        action = int(action)
        if action == self.pass_action:
            return True
        if action < 0 or action >= self.pass_action:
            return False

        row, col = self.action_to_coord(action)
        if self.board[row, col] != EMPTY:
            return False

        try:
            next_board, _ = self._play_on_board(self.board, self.current_player, row, col)
        except ValueError:
            return False
        return self._board_key(next_board) not in self._history

    def clone(self) -> "GoEnv":
        """Return an independent copy of the current game state."""
        clone = GoEnv(self.size, self.komi)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.done = self.done
        clone.consecutive_passes = self.consecutive_passes
        clone.move_count = self.move_count
        clone._history = set(self._history)
        return clone

    def score(self) -> dict[str, float]:
        """Compute Tromp-Taylor area score."""
        visited = np.zeros_like(self.board, dtype=bool)
        black_score = float(np.sum(self.board == BLACK))
        white_score = float(np.sum(self.board == WHITE)) + self.komi

        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] != EMPTY or visited[row, col]:
                    continue
                region, adjacent = self._empty_region(row, col, visited)
                if adjacent == {BLACK}:
                    black_score += len(region)
                elif adjacent == {WHITE}:
                    white_score += len(region)

        return {"black": black_score, "white": white_score, "komi": self.komi}

    def winner(self) -> int:
        """Return ``BLACK``, ``WHITE``, or ``0`` for a tie."""
        scores = self.score()
        if scores["black"] > scores["white"]:
            return BLACK
        if scores["white"] > scores["black"]:
            return WHITE
        return 0

    def render(self) -> str:
        """Return an ASCII board."""
        stones = {BLACK: "X", WHITE: "O", EMPTY: "."}
        rows = ["   " + " ".join(str(i) for i in range(self.size))]
        for row in range(self.size):
            cells = " ".join(stones[int(self.board[row, col])] for col in range(self.size))
            rows.append(f"{row:2d} {cells}")
        turn = "black" if self.current_player == BLACK else "white"
        rows.append(f"turn: {turn}")
        return "\n".join(rows)

    def encode(self) -> np.ndarray:
        """Encode state from the current player's perspective as ``(3, N, N)``."""
        own = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        color = np.full(
            (self.size, self.size),
            1.0 if self.current_player == BLACK else 0.0,
            dtype=np.float32,
        )
        return np.stack([own, opponent, color])

    def action_to_coord(self, action: int) -> tuple[int, int]:
        action = int(action)
        if action < 0 or action >= self.pass_action:
            raise ValueError(f"action must be in [0, {self.pass_action - 1}]")
        return divmod(action, self.size)

    def coord_to_action(self, row: int, col: int) -> int:
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("coordinate is outside the board")
        return row * self.size + col

    def _play_on_board(self, board: np.ndarray, player: int, row: int, col: int):
        if board[row, col] != EMPTY:
            raise ValueError("point is occupied")

        next_board = board.copy()
        next_board[row, col] = player
        opponent = -player
        captured = 0
        checked = set()

        for nr, nc in self._neighbors(row, col):
            if next_board[nr, nc] != opponent or (nr, nc) in checked:
                continue
            group, liberties = self._group_and_liberties(next_board, nr, nc)
            checked.update(group)
            if not liberties:
                captured += len(group)
                for gr, gc in group:
                    next_board[gr, gc] = EMPTY

        _, own_liberties = self._group_and_liberties(next_board, row, col)
        if not own_liberties:
            raise ValueError("suicide move")
        return next_board, captured

    def _group_and_liberties(self, board: np.ndarray, row: int, col: int):
        color = board[row, col]
        if color == EMPTY:
            return set(), set()

        group = {(row, col)}
        liberties = set()
        queue = deque([(row, col)])

        while queue:
            cr, cc = queue.popleft()
            for nr, nc in self._neighbors(cr, cc):
                value = board[nr, nc]
                if value == EMPTY:
                    liberties.add((nr, nc))
                elif value == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    queue.append((nr, nc))

        return group, liberties

    def _empty_region(self, row: int, col: int, visited: np.ndarray):
        region = {(row, col)}
        adjacent = set()
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
            cr, cc = queue.popleft()
            for nr, nc in self._neighbors(cr, cc):
                value = int(self.board[nr, nc])
                if value == EMPTY and not visited[nr, nc]:
                    visited[nr, nc] = True
                    region.add((nr, nc))
                    queue.append((nr, nc))
                elif value in (BLACK, WHITE):
                    adjacent.add(value)

        return region, adjacent

    def _neighbors(self, row: int, col: int):
        if row > 0:
            yield row - 1, col
        if row + 1 < self.size:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.size:
            yield row, col + 1

    @staticmethod
    def _board_key(board: np.ndarray) -> bytes:
        return board.tobytes()
