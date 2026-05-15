from __future__ import annotations

import curses
import json
import sys
import textwrap
import threading
from pathlib import Path
from typing import Callable
from uuid import uuid4

from config import ConfigReader
from schemas.ids import TaskId, UserId
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

# ──────────────────────────────────────────────────────────────────────────────
# Split-pane TUI (curses)
# ──────────────────────────────────────────────────────────────────────────────

class _SplitPane:
    """Left = user input history, Right = agent output stream."""

    def __init__(self, stdscr: "curses.window") -> None:
        self._scr = stdscr
        curses.curs_set(1)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)

        self._left_lines: list[str] = []
        self._right_lines: list[str] = []
        self._input_buf = ""
        self._lock = threading.Lock()
        self._user_id: UserId = "1944515138"
        self._redraw()

    def add_user_line(self, text: str) -> None:
        with self._lock:
            self._left_lines.append(text)
            self._redraw()

    def add_agent_line(self, text: str) -> None:
        with self._lock:
            self._right_lines.append(text)
            self._redraw()

    def read_input(self, check_done: Callable[[], bool] | None = None) -> str:
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
        left_rows = h - 4
        left_visible = self._left_lines[-left_rows:]
        for i, line in enumerate(left_visible):
            wrapped = textwrap.wrap(line, left_w) or [""]
            for j, seg in enumerate(wrapped):
                row = 2 + i + j
                if row >= h - 2:
                    break
                try:
                    self._scr.addstr(row, 1, seg[:left_w], curses.color_pair(2))
                except curses.error:
                    pass

        right_w = w - mid - 3
        right_rows = h - 4
        right_visible = self._right_lines[-right_rows:]
        for i, line in enumerate(right_visible):
            wrapped = textwrap.wrap(line, right_w) or [""]
            for j, seg in enumerate(wrapped):
                row = 2 + i + j
                if row >= h - 2:
                    break
                try:
                    self._scr.addstr(row, mid + 2, seg[:right_w], curses.color_pair(3))
                except curses.error:
                    pass

        try:
            self._scr.addstr(h - 3, 0, "-" * w, curses.color_pair(1))
        except curses.error:
            pass

        self._draw_input_bar()
        self._scr.refresh()

    def _draw_input_bar(self) -> None:
        h, w = self._scr.getmaxyx()
        prompt = "You> "
        line = (prompt + self._input_buf)[: w - 1]
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
        self._task_id = ""
        self._task_started = False
        self._task_completed = False
        self._pane: _SplitPane | None = None
        self._pane_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_callback(self.name)

    def release_resources(self) -> None:
        return None

    def reset(self) -> None:
        self._task_id = ""
        self._task_started = False
        self._task_completed = False

    def run(self) -> None:
        try:
            self._show_splash()
            self._run_menu_loop()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self._logger.error("UserThread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _show_splash(self) -> None:
        # ANSI: clear screen + move cursor to top-left, then flush immediately.
        # Avoids os.system("clear") subprocess overhead and race with other threads.
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        print(_LOGO)
        print(_MENU)
        sys.stdout.flush()

    def _run_menu_loop(self) -> None:
        while self._is_running():
            try:
                choice = input("  Enter command number: ").strip()
            except EOFError:
                break

            if choice == "q":
                break
            elif choice == "1":
                task_content = self._handle_new_task()
                if task_content:
                    self._dispatch_task(task_content)
                    self._run_split_pane()
                    if not self._is_running():
                        break
                    self._show_splash()
            elif choice == "2":
                self._dispatch_cancel()
            elif choice == "3":
                content = self._prompt_content("Suggest")
                if content:
                    self._dispatch_guidance(content)
            elif choice == "4":
                content = self._prompt_content("Clarify")
                if content:
                    self._dispatch_clarification(content)
            elif choice == "5":
                self._dispatch_resume()
            else:
                print(f"  Unknown command: {choice!r}. Enter 1-5 or q.")

    def _handle_new_task(self) -> str | None:
        print()
        print("  ┌─────────────────────────────────┐")
        print("  │  New Task                       │")
        print("  │  [1]  Input task manually       │")
        print("  │  [2]  Upload from file          │")
        print("  └─────────────────────────────────┘")
        try:
            sub = input("  Choose (1/2): ").strip()
        except EOFError:
            return None

        if sub == "1":
            try:
                content = input("  Task description: ").strip()
            except EOFError:
                return None
            return content or None
        elif sub == "2":
            try:
                path_str = input("  File path: ").strip()
            except EOFError:
                return None
            return self._load_from_file(path_str)
        else:
            print(f"  Unknown option: {sub!r}")
            return None

    def _load_from_file(self, path_str: str) -> str | None:
        path = Path(path_str).expanduser()
        if not path.exists():
            print(f"  File not found: {path}")
            return None
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"  Failed to read file: {exc}")
            return None
        if not content:
            print("  File is empty.")
            return None
        print(f"  Loaded {len(content)} characters from {path.name}")
        return content

    def _prompt_content(self, label: str) -> str | None:
        try:
            content = input(f"  {label} content: ").strip()
        except EOFError:
            return None
        return content or None

    def _dispatch_task(self, content: str) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.NEW_TASK,
            user_id=self._user_id,
            content=content,
        )
        self._task_queue.send_message(msg)
        self._task_started = True
        self._task_completed = False

    def _dispatch_cancel(self) -> None:
        msg = UserMessage(
            msg_type=UserMsgType.CANCEL,
            user_id=self._user_id,
        )
        self._agent_msg_queue.send_message(msg)
        print("  Cancel sent.")

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
        print("  Resume sent.")

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

        pane.add_agent_line("Agent is processing your task...")

        while self._is_running() and not self._task_completed:
            raw = pane.read_input(check_done=lambda: self._task_completed)
            if not raw:
                continue
            stripped = raw.strip()
            if stripped.lower() in {"exit", "quit", "q"}:
                break
            pane.add_user_line(f"You: {stripped}")
            self._dispatch_guidance(stripped)

        if self._task_completed:
            pane.add_agent_line("--- Task finished. Press Enter to return to menu. ---")
            pane.read_input()

        with self._pane_lock:
            self._pane = None

    def _agent_drain_loop(self) -> None:
        while self._is_running() and not self._task_completed:
            msg = self._user_msg_queue.get_message(timeout=self._agent_poll_timeout)
            if msg is None:
                continue
            is_last_msg = msg.metadata.get("is_last_message", False)
            if is_last_msg:
                self._task_completed = True
            formatted = self._format_message(msg)
            with self._pane_lock:
                if self._pane is not None:
                    self._pane.add_agent_line(formatted)

    def _format_message(self, msg: UserMessage) -> str:
        return msg.content

    def _is_running(self) -> bool:
        return (
            not self._stop_event.is_set()
            and not self._task_queue.is_closed()
            and not self._agent_msg_queue.is_closed()
            and not self._user_msg_queue.is_closed()
        )
