from __future__ import annotations

import curses
import re
import textwrap
import threading
from pathlib import Path
from typing import Callable

from config import ConfigReader
from schemas.ids import UserId
from utils.concurrency.message_queue import AgentMessageQueue, TaskQueue, UserMessageQueue
from schemas.types import UserMessage, UserMsgType
from utils.log.log import Logger, zap
from utils.concurrency.thread_event import ThreadEvent

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_LOGO = (
    "\033[3m\n"
    "  ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗\n"
    "  ╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝\n"
    "     ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗\n"
    "     ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║\n"
    "     ██║   ██║  ██║██║██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝\n"
    "     ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝\n"
    "\n"
    "        ██╗███████╗    ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗\n"
    "        ██║██╔════╝    ╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝\n"
    "        ██║███████╗       ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗\n"
    "        ██║╚════██║       ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║\n"
    "        ██║███████║       ██║   ██║  ██║██║██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝\n"
    "        ╚═╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝\n"
    "\033[0m"
)

_MENU = (
    "\n"
    "  ┌─────────────────────────────────────────┐\n"
    "  │           COMMAND INTERFACE             │\n"
    "  ├─────────────────────────────────────────┤\n"
    "  │  [1]  New Task    — start a new task    │\n"
    "  │  [2]  Cancel      — cancel current task │\n"
    "  │  [3]  Suggest     — send a suggestion   │\n"
    "  │  [4]  Clarify     — send clarification  │\n"
    "  │  [5]  Resume      — resume paused task  │\n"
    "  │  [q]  Quit        — exit the program    │\n"
    "  └─────────────────────────────────────────┘\n"
)

_ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


# ──────────────────────────────────────────────────────────────────────────────
# Split-pane TUI (curses)
# ──────────────────────────────────────────────────────────────────────────────

class _SplitPane:
    """Left = user input history, Right = agent output stream."""

    def __init__(self, stdscr: "curses.window") -> None:
        self._scr = stdscr
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)

        self._left_lines: list[str] = []
        self._right_lines: list[str] = []
        self._input_buf = ""
        self._prompt = "You> "
        self._lock = threading.Lock()
        self._redraw()

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt
            self._draw_input_bar()
            self._scr.refresh()

    def add_user_line(self, text: str) -> None:
        with self._lock:
            self._left_lines.append(text)
            self._redraw()

    def add_agent_line(self, text: str) -> None:
        with self._lock:
            self._right_lines.append(text)
            self._redraw()

    def read_input(self, prompt: str | None = None, check_done: Callable[[], bool] | None = None) -> str:
        if prompt is not None:
            with self._lock:
                self._prompt = prompt
                self._draw_input_bar()
                self._scr.refresh()
        self._input_buf = ""
        self._scr.timeout(200)
        while True:
            if check_done and check_done():
                return ""
            with self._lock:
                self._draw_input_bar()
            try:
                ch = self._scr.get_wch()
            except curses.error:
                continue
            if ch in ("\n", "\r", curses.KEY_ENTER):
                result = self._input_buf
                self._input_buf = ""
                return result
            elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                self._input_buf = self._input_buf[:-1]
            elif isinstance(ch, str) and ch.isprintable():
                self._input_buf += ch
            elif ch == curses.KEY_RESIZE:
                with self._lock:
                    self._redraw()

    def _redraw(self) -> None:
        self._scr.erase()
        h, w = self._scr.getmaxyx()
        mid = w // 2

        for row in range(h - 3):
            try:
                self._scr.addch(row, mid, "|", curses.color_pair(1))
            except curses.error:
                pass

        self._scr.addstr(0, 1, "[ USER INPUT ]", curses.color_pair(1) | curses.A_BOLD)
        self._scr.addstr(0, mid + 2, "[ AGENT OUTPUT ]", curses.color_pair(1) | curses.A_BOLD)

        left_w = mid - 2
        max_rows = h - 4  # rows available between header (row 1) and separator (h-3)
        self._render_pane(
            self._left_lines, left_w, max_rows, 1, curses.color_pair(2), h
        )

        right_w = w - mid - 3
        self._render_pane(
            self._right_lines, right_w, max_rows, mid + 2, curses.color_pair(3), h
        )

        try:
            self._scr.addstr(h - 3, 0, "-" * w, curses.color_pair(1))
        except curses.error:
            pass

        self._draw_input_bar()
        self._scr.refresh()

    def _render_pane(
        self,
        lines: list[str],
        width: int,
        max_rows: int,
        col: int,
        attr: int,
        h: int,
    ) -> None:
        # Expand all lines into display segments, then show the last max_rows of them
        segments: list[str] = []
        for line in lines:
            segments.extend(textwrap.wrap(line, width) or [""])
        visible = segments[-max_rows:]
        for i, seg in enumerate(visible):
            row = 2 + i
            if row >= h - 2:
                break
            try:
                self._scr.addstr(row, col, seg[:width], attr)
            except curses.error:
                pass

    def _draw_input_bar(self) -> None:
        h, w = self._scr.getmaxyx()
        line = (self._prompt + self._input_buf)[: w - 1]
        try:
            self._scr.addstr(h - 2, 0, " " * (w - 1))
            self._scr.addstr(h - 2, 0, line, curses.color_pair(4) | curses.A_BOLD)
            self._scr.move(h - 2, min(len(line), w - 2))
        except curses.error:
            pass
        self._scr.refresh()


# ──────────────────────────────────────────────────────────────────────────────
# UserThread
# ──────────────────────────────────────────────────────────────────────────────

class UserThread(threading.Thread):
    def __init__(
        self,
        task_queue: TaskQueue,
        agent_msg_queue: AgentMessageQueue,
        user_msg_queue: UserMessageQueue,
        config: ConfigReader,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="UserThread", daemon=False)
        self._task_queue = task_queue
        self._agent_msg_queue = agent_msg_queue
        self._user_msg_queue = user_msg_queue
        self._config = config
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger

        self._agent_poll_timeout = self._config.positive_float(
            "agent.latency.agent_message_poll_timeout_seconds", 1.0
        )
        self._user_id: UserId = "1944515138"
        self._task_id = ""
        self._task_started = False
        self._task_completed = threading.Event()
        self._pane: _SplitPane | None = None
        self._pane_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_callback(self.name)

    def release_resources(self) -> None:
        return None

    def reset(self) -> None:
        self._task_id = ""
        self._task_started = False
        self._task_completed.clear()

    def run(self) -> None:
        try:
            self._run_split_pane()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self._logger.error("UserThread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _run_split_pane(self) -> None:
        drain = threading.Thread(
            target=self._agent_drain_loop,
            name="AgentDrainLoop",
            daemon=True,
        )
        drain.start()
        try:
            curses.wrapper(self._curses_main)
        finally:
            drain.join(timeout=2.0)

    def _curses_main(self, stdscr: "curses.window") -> None:
        pane = _SplitPane(stdscr)
        with self._pane_lock:
            self._pane = pane

        # Show LOGO and menu in left pane (strip ANSI codes for curses)
        for line in _strip_ansi(_LOGO).splitlines():
            pane.add_user_line(line)
        for line in _MENU.splitlines():
            pane.add_user_line(line)

        while self._is_running():
            choice = pane.read_input("Command> ")
            if not choice:
                continue
            choice = choice.strip()

            if choice == "q":
                break
            elif choice == "1":
                pane.add_user_line("  Task description (or file path with @prefix):")
                task_input = pane.read_input("Task> ").strip()
                if not task_input:
                    continue
                if task_input.startswith("@"):
                    task_content = self._load_from_file(task_input[1:], pane)
                else:
                    task_content = task_input
                if not task_content:
                    continue
                self._dispatch_task(task_content)
                pane.add_user_line(f"  [Submitted] {task_content[:60]}")
                pane.add_user_line("  (type guidance while task runs, or wait for completion)")
                # Task mode: accept guidance until task completes
                while self._is_running() and not self._task_completed.is_set():
                    raw = pane.read_input(
                        "Guidance> ",
                        check_done=self._task_completed.is_set,
                    )
                    if raw:
                        pane.add_user_line(f"  You: {raw}")
                        self._dispatch_guidance(raw)
                pane.add_user_line("  [Task finished — returning to menu]")
                self.reset()
                for line in _MENU.splitlines():
                    pane.add_user_line(line)
            elif choice == "2":
                self._dispatch_cancel()
                pane.add_user_line("  Cancel sent.")
            elif choice == "3":
                content = pane.read_input("Suggest> ").strip()
                if content:
                    self._dispatch_guidance(content)
                    pane.add_user_line(f"  [Suggestion sent] {content[:60]}")
            elif choice == "4":
                content = pane.read_input("Clarify> ").strip()
                if content:
                    self._dispatch_clarification(content)
                    pane.add_user_line(f"  [Clarification sent] {content[:60]}")
            elif choice == "5":
                self._dispatch_resume()
                pane.add_user_line("  Resume sent.")
            else:
                pane.add_user_line(f"  Unknown command: {choice!r}. Enter 1-5 or q.")

        with self._pane_lock:
            self._pane = None

    def _load_from_file(self, path_str: str, pane: _SplitPane) -> str | None:
        path = Path(path_str).expanduser()
        if not path.exists():
            pane.add_user_line(f"  File not found: {path}")
            return None
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            pane.add_user_line(f"  Failed to read file: {exc}")
            return None
        if not content:
            pane.add_user_line("  File is empty.")
            return None
        pane.add_user_line(f"  Loaded {len(content)} chars from {path.name}")
        return content

    def _dispatch_task(self, content: str) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.NEW_TASK,
            user_id=self._user_id,
            content=content,
        )
        self._task_queue.send_message(msg)
        self._task_started = True
        self._task_completed.clear()

    def _dispatch_cancel(self) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.CANCEL,
            user_id=self._user_id,
        )
        self._agent_msg_queue.send_message(msg)

    def _dispatch_guidance(self, content: str) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.GUIDANCE,
            user_id=self._user_id,
            content=content,
        )
        self._agent_msg_queue.send_message(msg)

    def _dispatch_clarification(self, content: str) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.CLARIFICATION,
            user_id=self._user_id,
            content=content,
        )
        self._agent_msg_queue.send_message(msg)

    def _dispatch_resume(self) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.RESUME,
            user_id=self._user_id,
        )
        self._agent_msg_queue.send_message(msg)

    def _agent_drain_loop(self) -> None:
        while self._is_running():
            msg = self._user_msg_queue.get_message(timeout=self._agent_poll_timeout)
            if msg is None:
                continue
            is_last_msg = msg.metadata.get("is_last_message", False)
            formatted = self._format_message(msg)
            with self._pane_lock:
                if self._pane is not None:
                    self._pane.add_agent_line(formatted)
            if is_last_msg:
                self._task_completed.set()

    def _format_message(self, msg: UserMessage) -> str:
        return msg.content

    def _is_running(self) -> bool:
        return (
            not self._stop_event.is_set()
            and not self._task_queue.is_closed()
            and not self._agent_msg_queue.is_closed()
            and not self._user_msg_queue.is_closed()
        )
