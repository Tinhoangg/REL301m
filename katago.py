"""Small GTP wrapper for evaluating against KataGo."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .go_env import BLACK, WHITE


GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


class KataGoError(RuntimeError):
    """Raised when the KataGo GTP process fails or rejects a command."""


def color_to_gtp(color: int) -> str:
    if color == BLACK:
        return "B"
    if color == WHITE:
        return "W"
    raise ValueError(f"unknown Go color: {color}")


def action_to_gtp(action: int, size: int) -> str:
    """Convert internal action index to a GTP coordinate."""
    pass_action = size * size
    action = int(action)
    if action == pass_action:
        return "pass"
    if action < 0 or action >= pass_action:
        raise ValueError(f"action must be in [0, {pass_action}]")
    row, col = divmod(action, size)
    return f"{GTP_COLUMNS[col]}{size - row}"


def gtp_to_action(move: str, size: int) -> int | None:
    """Convert a GTP move to an internal action.

    Returns ``None`` for resign because the local environment has no resign
    action. The caller should score the game as a resignation result.
    """
    move = move.strip()
    if move.lower() == "pass":
        return size * size
    if move.lower() == "resign":
        return None
    if len(move) < 2:
        raise ValueError(f"invalid GTP move: {move!r}")

    col_letter = move[0].upper()
    if col_letter not in GTP_COLUMNS[:size]:
        raise ValueError(f"GTP column outside {size}x{size} board: {move!r}")
    col = GTP_COLUMNS.index(col_letter)
    row_number = int(move[1:])
    if row_number < 1 or row_number > size:
        raise ValueError(f"GTP row outside {size}x{size} board: {move!r}")
    row = size - row_number
    return row * size + col


class KataGoEngine:
    """Manage a KataGo GTP subprocess."""

    def __init__(
        self,
        executable,
        model=None,
        config=None,
        extra_args: list[str] | None = None,
        size: int = 9,
        komi: float = 7.5,
    ):
        self.executable = Path(executable)
        self.model = Path(model) if model else None
        self.config = Path(config) if config else None
        self.extra_args = list(extra_args or [])
        self.size = int(size)
        self.komi = float(komi)
        self._command_id = 0
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def start(self):
        if self.process is not None:
            return
        if not self.executable.exists():
            raise FileNotFoundError(f"KataGo executable not found: {self.executable}")

        args = [str(self.executable), "gtp"]
        if self.model is not None:
            args.extend(["-model", str(self.model)])
        if self.config is not None:
            args.extend(["-config", str(self.config)])
        args.extend(self.extra_args)

        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.command("protocol_version")

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.command("quit")
        except (BrokenPipeError, KataGoError):
            pass
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.process = None

    def new_game(self):
        self.command(f"boardsize {self.size}")
        self.command(f"komi {self.komi}")
        self.command("clear_board")

    def play_action(self, color: int, action: int):
        self.command(f"play {color_to_gtp(color)} {action_to_gtp(action, self.size)}")

    def genmove_action(self, color: int) -> int | None:
        move = self.command(f"genmove {color_to_gtp(color)}")
        return gtp_to_action(move.splitlines()[0], self.size)

    def command(self, command: str) -> str:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise KataGoError("KataGo process is not running")

        self._command_id += 1
        command_id = self._command_id
        self.process.stdin.write(f"{command_id} {command}\n")
        self.process.stdin.flush()

        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise KataGoError(f"KataGo stopped while running command: {command}")
            line = line.rstrip("\r\n")
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)

        success_prefix = f"={command_id}"
        error_prefix = f"?{command_id}"
        first = lines[0]
        if first.startswith(error_prefix):
            message = first[len(error_prefix) :].strip()
            raise KataGoError(f"KataGo rejected command {command!r}: {message}")
        if not first.startswith(success_prefix):
            raise KataGoError(f"unexpected KataGo response to {command!r}: {first!r}")

        payload = [first[len(success_prefix) :].strip()]
        payload.extend(lines[1:])
        return "\n".join(part for part in payload if part).strip()
