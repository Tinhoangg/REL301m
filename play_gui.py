"""Tkinter window for playing 9x9 Go against the MCTS agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Go_rel.config import DEFAULT_MAX_MOVES, KOMI, LATEST_CHECKPOINT_PATH
    from Go_rel.go_env import BLACK, EMPTY, WHITE, GoEnv
    from Go_rel.mcts import MCTS, NetworkEvaluator
    from Go_rel.model import PolicyValueNet, load_checkpoint
else:
    from .config import DEFAULT_MAX_MOVES, KOMI, LATEST_CHECKPOINT_PATH
    from .go_env import BLACK, EMPTY, WHITE, GoEnv
    from .mcts import MCTS, NetworkEvaluator
    from .model import PolicyValueNet, load_checkpoint


DEFAULT_GUI_SIMULATIONS = 25
BOARD_MARGIN = 38
CELL_SIZE = 54
STONE_RADIUS = 20


def canvas_point_to_action(x, y, size: int = 9, margin: int = BOARD_MARGIN, cell: int = CELL_SIZE, tolerance: int | None = None):
    """Map a canvas click to a board action, or ``None`` if outside points."""
    tolerance = tolerance if tolerance is not None else cell // 3
    col = round((x - margin) / cell)
    row = round((y - margin) / cell)
    if row < 0 or row >= size or col < 0 or col >= size:
        return None

    point_x = margin + col * cell
    point_y = margin + row * cell
    if abs(x - point_x) > tolerance or abs(y - point_y) > tolerance:
        return None
    return row * size + col


def action_to_canvas_point(action: int, size: int = 9, margin: int = BOARD_MARGIN, cell: int = CELL_SIZE):
    """Map a non-pass action to canvas coordinates."""
    row, col = divmod(int(action), size)
    return margin + col * cell, margin + row * cell


def color_name(color: int) -> str:
    if color == BLACK:
        return "Black"
    if color == WHITE:
        return "White"
    return "Tie"


def format_status(env: GoEnv, human_color: int, thinking: bool = False, agent_vs_agent: bool = False) -> str:
    """Build the status text shown in the GUI."""
    score = env.score()
    human = color_name(human_color)
    turn = color_name(env.current_player)
    mode = "Agent vs Agent" if agent_vs_agent else f"Human: {human}"
    if env.done:
        winner = color_name(env.winner())
        return (
            f"Game over | Winner: {winner}\n"
            f"Score: Black {score['black']:.1f} - White {score['white']:.1f}\n"
            f"{mode} | Moves: {env.move_count}"
        )
    if thinking:
        turn_line = "Agent is thinking..."
    else:
        turn_line = f"Turn: {turn}"
    return (
        f"{turn_line}\n"
        f"Score: Black {score['black']:.1f} - White {score['white']:.1f}\n"
        f"{mode} | Moves: {env.move_count}"
    )


def load_or_random_model(path):
    """Load a checkpoint if present, otherwise return a random network."""
    path = Path(path)
    if path.exists():
        return load_checkpoint(path), f"Loaded checkpoint: {path}"
    return PolicyValueNet().eval(), "No checkpoint found; using random network"


class GoWindow:
    """Interactive Tkinter form for Go human-vs-agent play."""

    def __init__(
        self,
        root: tk.Tk,
        simulations: int = DEFAULT_GUI_SIMULATIONS,
        checkpoint=LATEST_CHECKPOINT_PATH,
        human_color: int = BLACK,
        max_moves: int = DEFAULT_MAX_MOVES,
    ):
        self.root = root
        self.root.title("Go 9x9 RL Agent")
        self.env = GoEnv(size=9, komi=KOMI)
        self.human_color = human_color
        self.max_moves = int(max_moves)
        self.last_action = None
        self.thinking = False
        self.agent_vs_agent = False
        self.position_token = 0
        self.agent_queue: queue.Queue = queue.Queue()

        self.model, load_message = load_or_random_model(checkpoint)
        self.checkpoint = checkpoint

        self.simulations_var = tk.IntVar(value=max(1, int(simulations)))
        self.auto_delay_var = tk.IntVar(value=350)
        self.human_color_var = tk.StringVar(value="black" if human_color == BLACK else "white")
        self.mode_var = tk.StringVar(value="human")
        self.status_var = tk.StringVar()
        self.message_var = tk.StringVar(value=load_message)

        self.board_pixels = BOARD_MARGIN * 2 + CELL_SIZE * (self.env.size - 1)
        self._build_layout()
        self._draw_board()
        self._update_status()
        if self.env.current_player != self.human_color:
            self._start_agent_turn()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container,
            width=self.board_pixels,
            height=self.board_pixels,
            background="#D7A94B",
            highlightthickness=1,
            highlightbackground="#8B5A1F",
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        panel = ttk.Frame(container, padding=(16, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="new")

        ttk.Label(panel, text="Go 9x9 Agent", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(panel, textvariable=self.status_var, justify="left").grid(row=1, column=0, columnspan=2, pady=(12, 10), sticky="w")
        ttk.Label(panel, textvariable=self.message_var, wraplength=260, foreground="#555555").grid(row=2, column=0, columnspan=2, pady=(0, 12), sticky="w")

        ttk.Label(panel, text="Mode").grid(row=3, column=0, sticky="w")
        mode_frame = ttk.Frame(panel)
        mode_frame.grid(row=3, column=1, sticky="w")
        self.human_mode_radio = ttk.Radiobutton(mode_frame, text="Human", variable=self.mode_var, value="human", command=self.set_human_mode)
        self.human_mode_radio.grid(row=0, column=0, sticky="w")
        self.auto_mode_radio = ttk.Radiobutton(mode_frame, text="Auto", variable=self.mode_var, value="auto", command=self.set_auto_mode)
        self.auto_mode_radio.grid(row=0, column=1, sticky="w")

        ttk.Label(panel, text="Human color").grid(row=4, column=0, sticky="w")
        color_frame = ttk.Frame(panel)
        color_frame.grid(row=4, column=1, sticky="w")
        self.black_radio = ttk.Radiobutton(color_frame, text="Black", variable=self.human_color_var, value="black", command=self.reset_game)
        self.black_radio.grid(row=0, column=0, sticky="w")
        self.white_radio = ttk.Radiobutton(color_frame, text="White", variable=self.human_color_var, value="white", command=self.reset_game)
        self.white_radio.grid(row=0, column=1, sticky="w")

        ttk.Label(panel, text="Simulations").grid(row=5, column=0, pady=(10, 0), sticky="w")
        self.sim_spinbox = ttk.Spinbox(panel, from_=1, to=500, increment=5, textvariable=self.simulations_var, width=8)
        self.sim_spinbox.grid(row=5, column=1, pady=(10, 0), sticky="w")

        ttk.Label(panel, text="Auto delay ms").grid(row=6, column=0, pady=(10, 0), sticky="w")
        self.delay_spinbox = ttk.Spinbox(panel, from_=0, to=3000, increment=50, textvariable=self.auto_delay_var, width=8)
        self.delay_spinbox.grid(row=6, column=1, pady=(10, 0), sticky="w")

        self.pass_button = ttk.Button(panel, text="Pass", command=self.pass_turn)
        self.pass_button.grid(row=7, column=0, pady=(16, 0), sticky="ew")
        self.reset_button = ttk.Button(panel, text="Reset", command=self.reset_game)
        self.reset_button.grid(row=7, column=1, pady=(16, 0), padx=(8, 0), sticky="ew")

        self.auto_button = ttk.Button(panel, text="Start Auto", command=self.toggle_auto_play)
        self.auto_button.grid(row=8, column=0, columnspan=2, pady=(8, 0), sticky="ew")

        ttk.Label(
            panel,
            text="Human mode: click an intersection. Auto mode: two agents play each other.",
            wraplength=260,
            foreground="#555555",
        ).grid(row=9, column=0, columnspan=2, pady=(16, 0), sticky="w")

    def _draw_board(self):
        self.canvas.delete("all")
        size = self.env.size
        start = BOARD_MARGIN
        end = BOARD_MARGIN + CELL_SIZE * (size - 1)

        for i in range(size):
            pos = BOARD_MARGIN + i * CELL_SIZE
            self.canvas.create_line(start, pos, end, pos, fill="#3C2413", width=2)
            self.canvas.create_line(pos, start, pos, end, fill="#3C2413", width=2)

        for row, col in [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]:
            x, y = action_to_canvas_point(row * size + col, size)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#3C2413", outline="")

        if self.last_action is not None and self.last_action != self.env.pass_action:
            x, y = action_to_canvas_point(self.last_action, size)
            self.canvas.create_rectangle(x - 25, y - 25, x + 25, y + 25, outline="#E14B2D", width=3)

        for row in range(size):
            for col in range(size):
                stone = int(self.env.board[row, col])
                if stone == EMPTY:
                    continue
                action = row * size + col
                x, y = action_to_canvas_point(action, size)
                fill = "#111111" if stone == BLACK else "#F4F4F4"
                outline = "#000000" if stone == BLACK else "#888888"
                self.canvas.create_oval(
                    x - STONE_RADIUS,
                    y - STONE_RADIUS,
                    x + STONE_RADIUS,
                    y + STONE_RADIUS,
                    fill=fill,
                    outline=outline,
                    width=2,
                )

    def _on_canvas_click(self, event):
        if self.agent_vs_agent or self.thinking or self.env.done or self.env.current_player != self.human_color:
            return
        action = canvas_point_to_action(event.x, event.y, self.env.size)
        if action is None:
            return
        self._play_human_action(action)

    def pass_turn(self):
        if self.agent_vs_agent or self.thinking or self.env.done or self.env.current_player != self.human_color:
            return
        self._play_human_action(self.env.pass_action)

    def reset_game(self):
        self.position_token += 1
        self.thinking = False
        self.agent_vs_agent = self.mode_var.get() == "auto"
        self.env = GoEnv(size=9, komi=KOMI)
        self.human_color = BLACK if self.human_color_var.get() == "black" else WHITE
        self.last_action = None
        self.message_var.set("New game")
        self._set_controls_enabled(True)
        self._sync_auto_button()
        self._draw_board()
        self._update_status()
        if self.agent_vs_agent:
            self._schedule_next_auto_turn()
        elif self.env.current_player != self.human_color:
            self._start_agent_turn()

    def set_human_mode(self):
        self.agent_vs_agent = False
        self.mode_var.set("human")
        self.reset_game()

    def set_auto_mode(self):
        self.agent_vs_agent = True
        self.mode_var.set("auto")
        self.reset_game()

    def toggle_auto_play(self):
        if self.agent_vs_agent:
            self.agent_vs_agent = False
            self.mode_var.set("human")
            self.position_token += 1
            self.thinking = False
            self.message_var.set("Auto play stopped")
            self._set_controls_enabled(True)
            self._sync_auto_button()
            self._update_status()
            return

        self.agent_vs_agent = True
        self.mode_var.set("auto")
        self.message_var.set("Auto play started")
        self._sync_auto_button()
        if self.env.done or self.env.move_count >= self.max_moves:
            self.reset_game()
        else:
            self._schedule_next_auto_turn()

    def _play_human_action(self, action: int):
        try:
            self.env.step(action)
        except ValueError as exc:
            self.message_var.set(str(exc))
            return
        self.last_action = action
        self.message_var.set(self._describe_action("Human", action))
        self._draw_board()
        self._update_status()
        self._finish_or_start_agent()

    def _finish_or_start_agent(self):
        if self.env.done:
            self._show_game_over()
        elif self.env.move_count >= self.max_moves:
            self.message_var.set(f"Reached max moves ({self.max_moves}). Current score shown.")
            self.agent_vs_agent = False
            self.mode_var.set("human")
            self._sync_auto_button()
        elif self.agent_vs_agent:
            self._schedule_next_auto_turn()
        elif self.env.current_player != self.human_color:
            self._start_agent_turn()

    def _schedule_next_auto_turn(self):
        if not self.agent_vs_agent or self.thinking or self.env.done:
            return
        delay = max(0, int(self.auto_delay_var.get()))
        self.root.after(delay, self._start_agent_turn)

    def _start_agent_turn(self):
        if self.env.done:
            return
        self.thinking = True
        self.position_token += 1
        token = self.position_token
        env_snapshot = self.env.clone()
        simulations = max(1, int(self.simulations_var.get()))
        self._set_controls_enabled(False)
        self._update_status()

        def worker():
            try:
                mcts = MCTS(NetworkEvaluator(self.model), num_simulations=simulations)
                action, _ = mcts.select_action(env_snapshot, temperature=1e-6)
                self.agent_queue.put((token, action, None))
            except Exception as exc:  # pragma: no cover - defensive GUI path
                self.agent_queue.put((token, None, exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_agent_result)

    def _poll_agent_result(self):
        try:
            token, action, error = self.agent_queue.get_nowait()
        except queue.Empty:
            if self.thinking:
                self.root.after(100, self._poll_agent_result)
            return

        if token != self.position_token:
            return
        self.thinking = False
        self._set_controls_enabled(True)
        if error is not None:
            self.message_var.set(f"Agent error: {error}")
            self._update_status()
            return

        try:
            self.env.step(action)
            self.last_action = action
            self.message_var.set(self._describe_action("Agent", action))
        except ValueError as exc:
            self.message_var.set(f"Agent illegal move: {exc}")
        self._draw_board()
        self._update_status()
        self._finish_or_start_agent()

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        pass_state = state if not self.agent_vs_agent else "disabled"
        self.pass_button.configure(state=pass_state)
        self.sim_spinbox.configure(state=state)
        self.delay_spinbox.configure(state=state)
        self.black_radio.configure(state=state if not self.agent_vs_agent else "disabled")
        self.white_radio.configure(state=state if not self.agent_vs_agent else "disabled")

    def _update_status(self):
        self.status_var.set(format_status(self.env, self.human_color, self.thinking, self.agent_vs_agent))

    def _sync_auto_button(self):
        self.auto_button.configure(text="Stop Auto" if self.agent_vs_agent else "Start Auto")

    def _show_game_over(self):
        self._update_status()
        score = self.env.score()
        winner = color_name(self.env.winner())
        if self.agent_vs_agent:
            self.agent_vs_agent = False
            self.mode_var.set("human")
            self._sync_auto_button()
            self.message_var.set(f"Auto game over. Winner: {winner}")
            self._set_controls_enabled(True)
        else:
            messagebox.showinfo("Game Over", f"Winner: {winner}\nBlack {score['black']:.1f} - White {score['white']:.1f}")

    def _describe_action(self, actor: str, action: int) -> str:
        if action == self.env.pass_action:
            return f"{actor} passed"
        row, col = self.env.action_to_coord(action)
        return f"{actor} played row {row}, col {col}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=DEFAULT_GUI_SIMULATIONS)
    parser.add_argument("--checkpoint", default=str(LATEST_CHECKPOINT_PATH))
    parser.add_argument("--human-color", choices=["black", "white"], default="black")
    parser.add_argument("--auto-play", action="store_true", help="start with two agents playing each other")
    parser.add_argument("--auto-delay", type=int, default=350)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    args = parser.parse_args()

    human_color = BLACK if args.human_color == "black" else WHITE
    root = tk.Tk()
    app = GoWindow(
        root,
        simulations=args.simulations,
        checkpoint=args.checkpoint,
        human_color=human_color,
        max_moves=args.max_moves,
    )
    app.auto_delay_var.set(max(0, args.auto_delay))
    if args.auto_play:
        app.set_auto_mode()
    root.mainloop()


if __name__ == "__main__":
    main()
