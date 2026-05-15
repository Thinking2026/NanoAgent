from __future__ import annotations

import curses
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

_LOGO_LINES = [
    r" _____ _  _ ___ _  _ _  ___ _  _  ___",
    r"|_   _| || |_ _| \| | |/ __| || ||_ _|",
    r"  |_|  \__/|___|_|\_|_|\___|\__/|_|   ",
    r" ___ ___   _____ _  _ ___ _  _ _  ___ _  _  ___",
    r"|_ _/ __|  |_   _| || |_ _| \| | |/ __| || ||_ _|",
    r"|_|\__ \    |_|  \__/|___|_|\_|_|\___|\__/|_|    ",
]

_MENU_TITLE = "COMMAND MENU"
_MENU_MAIN = [
    "[1] New Task    - start a new task",
    "[2] Cancel      - cancel current task",
    "[3] Guidance    - submit guidance",
    "[4] Clarify     - submit clarification",
    "[5] Resume      - resume current task",
    "[q] Quit        - exit the program",
]
_MENU_NEW_TASK_TITLE = "NEW TASK"
_MENU_NEW_TASK = [
    "[1] Input       - enter task description",
    "[2] Upload File - load task from file",
    "[b] Back        - return to main menu",
    "[q] Quit        - exit the program",
]
_MENU_GUIDANCE_TITLE = "GUIDANCE"
_MENU_GUIDANCE = [
    "Enter guidance for the running task",
    "[Enter] Submit  [blank] Cancel",
]
_MENU_CLARIFY_TITLE = "CLARIFY"
_MENU_CLARIFY = [
    "Enter clarification for the agent",
    "[Enter] Submit  [blank] Cancel",
]


# ──────────────────────────────────────────────────────────────────────────────
# Split-pane TUI (curses)
# ──────────────────────────────────────────────────────────────────────────────

class _SplitPane:
    """Left = fixed LOGO + menu + scrolling user input. Right = scrolling agent output."""

    def __init__(self, stdscr: "curses.window") -> None:
        self._scr = stdscr
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # headers, divider
        curses.init_pair(2, curses.COLOR_GREEN, -1)   # right pane (agent)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # input bar prompt
        curses.init_pair(4, curses.COLOR_WHITE, -1)   # left pane (user)

        self._logo_lines: list[str] = []
        self._menu_title: str = ""
        self._menu_options: list[str] = []
        self._user_input_lines: list[str] = []
        self._right_lines: list[str] = []
        self._input_buf = ""
        self._prompt = "Command> "
        self._lock = threading.Lock()
        self._redraw()

    def set_logo(self, lines: list[str]) -> None:
        with self._lock:
            self._logo_lines = list(lines)
            self._redraw()

    def set_menu(self, title: str, options: list[str]) -> None:
        with self._lock:
            self._menu_title = title
            self._menu_options = list(options)
            self._redraw()

    def add_user_line(self, text: str) -> None:
        with self._lock:
            self._user_input_lines.append(text)
            self._redraw()

    def add_agent_line(self, text: str) -> None:
        with self._lock:
            if self._right_lines:
                self._right_lines.append("")  # blank separator between messages
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
        left_w = mid - 2
        right_w = w - mid - 3

        # Vertical divider
        for row in range(h - 3):
            try:
                self._scr.addch(row, mid, "|", curses.color_pair(1))
            except curses.error:
                pass

        # Column headers
        try:
            self._scr.addstr(0, 1, "[ USER INPUT ]", curses.color_pair(1) | curses.A_BOLD)
            self._scr.addstr(0, mid + 2, "[ AGENT OUTPUT ]", curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

        self._render_left(left_w, h)
        self._render_right(right_w, mid + 2, h)

        # Horizontal separator
        try:
            self._scr.addstr(h - 3, 0, "-" * w, curses.color_pair(1))
        except curses.error:
            pass

        self._draw_input_bar()
        self._scr.refresh()

    def _render_left(self, left_w: int, h: int) -> None:
        white = curses.color_pair(4)
        bold = curses.A_BOLD
        row = 1

        # Block A: LOGO (fixed, centered)
        for line in self._logo_lines:
            if row >= h - 3:
                break
            try:
                self._scr.addstr(row, 1, line.center(left_w)[:left_w], white)
            except curses.error:
                pass
            row += 1

        row += 2  # two blank lines after logo

        # Block B: Menu title (fixed, centered, bold)
        if row < h - 3 and self._menu_title:
            try:
                self._scr.addstr(row, 1, self._menu_title.center(left_w)[:left_w], white | bold)
            except curses.error:
                pass
            row += 1

        # Block C: Menu options (fixed, centered)
        for opt in self._menu_options:
            if row >= h - 3:
                break
            try:
                self._scr.addstr(row, 1, opt.center(left_w)[:left_w], white)
            except curses.error:
                pass
            row += 1

        row += 2  # two blank lines after menu

        # Block D: User input lines (scrolling — oldest removed from top)
        available = (h - 3) - row
        if available > 0:
            visible = self._user_input_lines[-available:]
            for line in visible:
                if row >= h - 3:
                    break
                try:
                    self._scr.addstr(row, 1, line[:left_w], white)
                except curses.error:
                    pass
                row += 1

    def _render_right(self, right_w: int, col: int, h: int) -> None:
        green = curses.color_pair(2)
        segments: list[str] = []
        for raw in self._right_lines:
            expanded = raw.expandtabs(4)
            for subline in expanded.split("\n"):
                if subline.strip():
                    wrapped = textwrap.wrap(subline, right_w)
                    segments.extend(wrapped if wrapped else [""])
                else:
                    segments.append("")
        max_rows = h - 4
        visible = segments[-max_rows:]
        for i, seg in enumerate(visible):
            row = 1 + i
            if row >= h - 3:
                break
            try:
                self._scr.addstr(row, col, seg[:right_w], green)
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

        pane.set_logo(_LOGO_LINES)
        pane.set_menu(_MENU_TITLE, _MENU_MAIN)

        menu_level = 1  # 1 = main menu, 2 = new-task submenu

        while self._is_running():
            raw = pane.read_input("Thinking> ")
            if not raw:
                continue
            cmd = raw.strip()

            # #q exits at any menu level
            if cmd.startswith("#q"):
                break

            if menu_level == 1:
                if cmd.startswith("#1"):
                    menu_level = 2
                    pane.set_menu(_MENU_NEW_TASK_TITLE, _MENU_NEW_TASK)
                elif cmd.startswith("#2"):
                    self._dispatch_cancel()
                    pane.add_user_line("User: [cancel sent]")
                elif cmd.startswith("#3"):
                    pane.set_menu(_MENU_GUIDANCE_TITLE, _MENU_GUIDANCE)
                    content = pane.read_input("Thinking> ").strip()
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                    if content:
                        self._dispatch_guidance(content)
                        pane.add_user_line(f"User: {content}")
                elif cmd.startswith("#4"):
                    pane.set_menu(_MENU_CLARIFY_TITLE, _MENU_CLARIFY)
                    content = pane.read_input("Thinking> ").strip()
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                    if content:
                        self._dispatch_clarification(content)
                        pane.add_user_line(f"User: {content}")
                elif cmd.startswith("#5"):
                    self._dispatch_resume()
                    pane.add_user_line("User: [resume sent]")
                else:
                    pane.add_user_line(f"User: unknown command {cmd!r}")

            elif menu_level == 2:
                if cmd.startswith("#b"):
                    menu_level = 1
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                elif cmd.startswith("#1"):
                    content = pane.read_input("Thinking> ").strip()
                    if content:
                        self.reset()
                        self._dispatch_task(content)
                        pane.add_user_line(f"User: {content}")
                        menu_level = 1
                        pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                elif cmd.startswith("#2"):
                    file_path = pane.read_input("Thinking> ").strip()
                    if file_path:
                        task_content = self._load_from_file(file_path, pane)
                        if task_content:
                            self.reset()
                            self._dispatch_task(task_content)
                            pane.add_user_line(f"User: [loaded from {file_path}]")
                            menu_level = 1
                            pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                else:
                    pane.add_user_line(f"User: unknown command {cmd!r}")

        with self._pane_lock:
            self._pane = None

    def _load_from_file(self, path_str: str, pane: _SplitPane) -> str | None:
        path = Path(path_str).expanduser()
        if not path.exists():
            pane.add_user_line(f"User: file not found: {path}")
            return None
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            pane.add_user_line(f"User: failed to read file: {exc}")
            return None
        if not content:
            pane.add_user_line("User: file is empty")
            return None
        pane.add_user_line(f"User: loaded {len(content)} chars from {path.name}")
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
        return f"Argus: {msg.content}" if msg.content else "Argus: "

    def _is_running(self) -> bool:
        return (
            not self._stop_event.is_set()
            and not self._task_queue.is_closed()
            and not self._agent_msg_queue.is_closed()
            and not self._user_msg_queue.is_closed()
        )
