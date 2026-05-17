from __future__ import annotations

import curses
import textwrap
import threading
import time
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
    r" _____ _   _ _____ _   _  _   _______ _   _ _____",
    r"|_   _| | | |_   _| \ | || | / /_   _| \ | |  __ " + "\\",
    r"  | | | |_| | | | |  \| || |/ /  | | |  \| | |  \/",
    r"  | | |  _  | | | | . ` ||    \  | | | . ` | | __",
    r"  | | | | | |_| |_| |\  || |\  \_| |_| |\  | |_\ " + "\\",
    r"  \_/ \_| |_/\___/\_| \_/\_| \_/\___/\_| \_/\____/",
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
_MENU_INPUT_TITLE = "INPUT TASK"
_MENU_INPUT = [
    "Type your task description below",
    "[Enter] Submit  [#b] Back",
]
_MENU_UPLOAD_TITLE = "UPLOAD FILE"
_MENU_UPLOAD = [
    "Type the file path below",
    "[Enter] Submit  [#b] Back",
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
        curses.init_pair(1, curses.COLOR_CYAN, -1)    # divider, separators
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # input bar prompt
        curses.init_pair(4, curses.COLOR_WHITE, -1)   # left pane (user)
        if curses.COLORS >= 256:
            curses.init_pair(2, 71, -1)   # muted green for agent output
            curses.init_pair(5, 220, -1)  # duck-egg yellow for [ USER INPUT ] / [ AGENT OUTPUT ]
        else:
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(5, curses.COLOR_YELLOW, -1)

        self._logo_lines: list[str] = []
        self._menu_title: str = ""
        self._menu_options: list[str] = []
        self._user_input_lines: list[str] = []
        self._right_lines: list[str] = []
        self._waiting_overlay: str | None = None
        self._waiting_frame: int = 0
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
            if self._user_input_lines:
                self._user_input_lines.append("")  # blank separator between messages
            self._user_input_lines.append(text)
            self._redraw()

    def add_agent_line(self, text: str) -> None:
        with self._lock:
            self._waiting_overlay = None
            if self._right_lines:
                self._right_lines.append("")  # blank separator between messages
            self._right_lines.append(text)
            self._redraw()

    def set_waiting_overlay(self, text: str | None, frame: int = 0) -> None:
        with self._lock:
            self._waiting_overlay = text
            self._waiting_frame = frame
            self._redraw()

    def read_input(self, prompt: str | None = None, check_done: Callable[[], bool] | None = None) -> str:
        if prompt is not None:
            with self._lock:
                self._prompt = prompt
                self._draw_input_bar()
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
        # right panel: starts at col mid+2, width = w - (mid+2) - 1 (leave last col)
        right_col = mid + 2
        right_w = w - right_col - 1
        right_h = h - 3  # rows 0 .. h-4 (above the horizontal separator)

        # Vertical divider
        for row in range(h - 3):
            try:
                self._scr.addch(row, mid, "|", curses.color_pair(1))
            except curses.error:
                pass

        # Column headers — [ AGENT OUTPUT ] is drawn inside right_win to stay pinned
        try:
            self._scr.addstr(0, 1, "[ USER INPUT ]", curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass

        self._render_left(left_w, h)

        # Horizontal separator
        try:
            self._scr.addstr(h - 3, 0, "-" * (w - 1), curses.color_pair(1))
        except curses.error:
            pass

        # stdscr must be noutrefresh'd BEFORE subwindows so subwindow content
        # is painted on top of the base layer, not overwritten by it.
        self._scr.noutrefresh()

        if right_w > 0 and right_h > 1:
            try:
                right_win = curses.newwin(right_h, right_w, 0, right_col)
                self._render_right_win(right_win, right_w, right_h)
                right_win.noutrefresh()
            except curses.error:
                pass

        self._draw_input_bar()

    def _render_left(self, left_w: int, h: int) -> None:
        white = curses.color_pair(4)
        cyan = curses.color_pair(1)
        bold = curses.A_BOLD
        row = 1

        # Block A: LOGO — cyan bold, centered
        for line in self._logo_lines:
            if row >= h - 3:
                break
            try:
                self._scr.addstr(row, 1, line.center(left_w)[:left_w], cyan | bold)
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

        # Block C: Menu options — block is centered as a whole, items left-aligned within it
        if self._menu_options:
            block_w = min(max(len(o) for o in self._menu_options), left_w)
            block_col = max(1, 1 + (left_w - block_w) // 2)
            for opt in self._menu_options:
                if row >= h - 3:
                    break
                try:
                    self._scr.addstr(row, block_col, opt[:block_w], white)
                except curses.error:
                    pass
                row += 1

        row += 2  # two blank lines after menu

        # Block D: User input lines (scrolling — wrap each line to left_w, oldest removed from top)
        available = (h - 3) - row
        if available > 0:
            segments: list[str] = []
            for line in self._user_input_lines:
                wrapped = textwrap.wrap(line, left_w)
                segments.extend(wrapped if wrapped else [""])
            visible = segments[-available:]
            for seg in visible:
                if row >= h - 3:
                    break
                try:
                    self._scr.addstr(row, 1, seg[:left_w], white)
                except curses.error:
                    pass
                row += 1

    def _render_right_win(self, win: "curses.window", right_w: int, right_h: int) -> None:
        green = curses.color_pair(2)
        white = curses.color_pair(4)
        duck_yellow_bold = curses.color_pair(5) | curses.A_BOLD

        # Always pin "[ AGENT OUTPUT ]" at row 0 of the subwindow
        try:
            win.addstr(0, 0, "[ AGENT OUTPUT ]"[:right_w], duck_yellow_bold)
        except curses.error:
            pass

        # Content starts at row 2 (one blank line gap below header)
        content_start_row = 2
        max_rows = right_h - content_start_row
        if max_rows <= 0:
            return

        # Reserve last row for waiting overlay if active
        overlay_row = content_start_row + max_rows - 1
        content_rows = max_rows - 1 if self._waiting_overlay is not None else max_rows

        segments: list[str] = []
        for raw in self._right_lines:
            expanded = raw.expandtabs(4)
            for subline in expanded.split("\n"):
                if subline.strip():
                    wrapped = textwrap.wrap(
                        subline, right_w,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    segments.extend(wrapped if wrapped else [""])
                else:
                    segments.append("")
        visible = segments[-content_rows:] if content_rows > 0 else []
        for i, seg in enumerate(visible):
            row = content_start_row + i
            if row >= right_h:
                break
            try:
                if seg.startswith("Argus:"):
                    prefix = "Argus:"
                    rest = seg[len(prefix):]
                    win.addstr(row, 0, prefix[:right_w], white | curses.A_BOLD)
                    if rest and len(prefix) < right_w:
                        win.addstr(row, len(prefix), rest[:right_w - len(prefix)], green)
                else:
                    win.addstr(row, 0, seg[:right_w], green)
            except curses.error:
                pass

        # Render animated waiting overlay on the last row
        if self._waiting_overlay is not None and overlay_row < right_h:
            overlay = self._waiting_overlay
            prefix = "Argus:"
            rest = overlay[len(prefix):] if overlay.startswith(prefix) else overlay
            frame = self._waiting_frame
            try:
                win.addstr(overlay_row, 0, prefix[:right_w], white | curses.A_BOLD)
            except curses.error:
                pass
            col = len(prefix)
            wave_len = len(rest)
            for ci, ch in enumerate(rest):
                if col >= right_w:
                    break
                # wave: one character is white, rest are green; highlight travels left-to-right
                if wave_len > 0 and ci == frame % wave_len:
                    attr = white | curses.A_BOLD
                else:
                    attr = green
                try:
                    win.addstr(overlay_row, col, ch, attr)
                except curses.error:
                    pass
                col += 1

    def _draw_input_bar(self) -> None:
        h, w = self._scr.getmaxyx()
        line = (self._prompt + self._input_buf)[: w - 1]
        try:
            self._scr.addstr(h - 2, 0, " " * (w - 1))
            self._scr.addstr(h - 2, 0, line, curses.color_pair(4) | curses.A_BOLD)
            self._scr.move(h - 2, min(len(line), w - 2))
        except curses.error:
            pass
        self._scr.noutrefresh()
        curses.doupdate()


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
        self._show_waiting = threading.Event()
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
        anim = threading.Thread(
            target=self._waiting_anim_loop,
            name="WaitingAnimLoop",
            daemon=True,
        )
        drain.start()
        anim.start()
        try:
            curses.wrapper(self._curses_main)
        finally:
            drain.join(timeout=2.0)
            anim.join(timeout=2.0)

    def _curses_main(self, stdscr: "curses.window") -> None:
        pane = _SplitPane(stdscr)
        with self._pane_lock:
            self._pane = pane

        pane.set_logo(_LOGO_LINES)
        pane.set_menu(_MENU_TITLE, _MENU_MAIN)

        menu_level = 1  # 1 = main menu, 2 = new-task submenu
        current_mode: str | None = None  # "input", "upload", "guidance", "clarify"

        _PROMPTS: dict[str | None, str] = {
            None: "Thinking> ",
            "input": "Please input your task> ",
            "upload": "Please input task file path> ",
            "guidance": "Please input your guidance> ",
            "clarify": "Please input your clarification> ",
        }

        def reset_to_main() -> None:
            nonlocal menu_level, current_mode
            menu_level = 1
            current_mode = None
            pane.set_menu(_MENU_TITLE, _MENU_MAIN)

        while self._is_running():
            raw = pane.read_input(_PROMPTS.get(current_mode, "Thinking> "))
            if not raw:
                continue
            cmd = raw.strip()

            if cmd.startswith("#q"):
                break

            # Non-command input: treat as content for the current mode
            if not cmd.startswith("#") and current_mode is not None:
                if current_mode == "input":
                    self.reset()
                    self._dispatch_task(cmd)
                    pane.add_user_line(f"User: {cmd}")
                    menu_level = 2
                    current_mode = None
                    pane.set_menu(_MENU_NEW_TASK_TITLE, _MENU_NEW_TASK)
                elif current_mode == "upload":
                    task_content = self._load_from_file(cmd, pane)
                    if task_content:
                        self.reset()
                        self._dispatch_task(task_content)
                        pane.add_user_line(f"User: [loaded from {cmd}]")
                        menu_level = 2
                        current_mode = None
                        pane.set_menu(_MENU_NEW_TASK_TITLE, _MENU_NEW_TASK)
                elif current_mode == "guidance":
                    self._dispatch_guidance(cmd)
                    pane.add_user_line(f"User: {cmd}")
                    reset_to_main()
                elif current_mode == "clarify":
                    self._dispatch_clarification(cmd)
                    pane.add_user_line(f"User: {cmd}")
                    reset_to_main()
                continue

            # Command input — process regardless of current_mode
            if menu_level == 1:
                if cmd.startswith("#1"):
                    menu_level = 2
                    current_mode = None
                    pane.set_menu(_MENU_NEW_TASK_TITLE, _MENU_NEW_TASK)
                elif cmd.startswith("#2"):
                    self._dispatch_cancel()
                    pane.add_user_line("User: [cancel sent]")
                    current_mode = None
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                elif cmd.startswith("#3"):
                    content = cmd[2:].strip()
                    if content:
                        self._dispatch_guidance(content)
                        pane.add_user_line(f"User: {content}")
                        current_mode = None
                        pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                    else:
                        current_mode = "guidance"
                        pane.set_menu(_MENU_GUIDANCE_TITLE, _MENU_GUIDANCE)
                elif cmd.startswith("#4"):
                    content = cmd[2:].strip()
                    if content:
                        self._dispatch_clarification(content)
                        pane.add_user_line(f"User: {content}")
                        current_mode = None
                        pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                    else:
                        current_mode = "clarify"
                        pane.set_menu(_MENU_CLARIFY_TITLE, _MENU_CLARIFY)
                elif cmd.startswith("#5"):
                    self._dispatch_resume()
                    pane.add_user_line("User: [resume sent]")
                    current_mode = None
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                else:
                    pane.add_user_line(f"User: unknown command {cmd!r}")

            elif menu_level == 2:
                if cmd.startswith("#b"):
                    menu_level = 1
                    current_mode = None
                    pane.set_menu(_MENU_TITLE, _MENU_MAIN)
                elif cmd.startswith("#1"):
                    content = cmd[2:].strip()
                    if content:
                        self.reset()
                        self._dispatch_task(content)
                        pane.add_user_line(f"User: {content}")
                        reset_to_main()
                    else:
                        current_mode = "input"
                        pane.set_menu(_MENU_INPUT_TITLE, _MENU_INPUT)
                elif cmd.startswith("#2"):
                    file_path = cmd[2:].strip()
                    if file_path:
                        task_content = self._load_from_file(file_path, pane)
                        if task_content:
                            self.reset()
                            self._dispatch_task(task_content)
                            pane.add_user_line(f"User: [loaded from {file_path}]")
                            reset_to_main()
                    else:
                        current_mode = "upload"
                        pane.set_menu(_MENU_UPLOAD_TITLE, _MENU_UPLOAD)
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
        _WAITING_DELAY = 60.0
        task_silence_start: float | None = None

        while self._is_running():
            msg = self._user_msg_queue.get_message(timeout=self._agent_poll_timeout)
            if msg is None:
                if self._task_started and not self._task_completed.is_set():
                    if task_silence_start is None:
                        task_silence_start = time.monotonic()
                    if time.monotonic() - task_silence_start >= _WAITING_DELAY:
                        self._show_waiting.set()
                continue
            # Real message received
            task_silence_start = None
            self._show_waiting.clear()
            is_last_msg = msg.metadata.get("is_last_message", False)
            formatted = self._format_message(msg)
            with self._pane_lock:
                if self._pane is not None:
                    self._pane.add_agent_line(formatted)
            if is_last_msg:
                self._task_completed.set()
                task_silence_start = None

    def _waiting_anim_loop(self) -> None:
        _ANIM_INTERVAL = 0.08
        frame = 0
        while self._is_running():
            if self._show_waiting.wait(timeout=0.5):
                frame += 1
                with self._pane_lock:
                    if self._pane is not None:
                        self._pane.set_waiting_overlay("Argus: I am Working...", frame)
                time.sleep(_ANIM_INTERVAL)
            else:
                # Waiting was cleared — remove overlay if it's still showing
                with self._pane_lock:
                    if self._pane is not None:
                        self._pane.set_waiting_overlay(None)
                frame = 0

    def _format_message(self, msg: UserMessage) -> str:
        return f"Argus: {msg.content}" if msg.content else "Argus: "

    def _is_running(self) -> bool:
        return (
            not self._stop_event.is_set()
            and not self._task_queue.is_closed()
            and not self._agent_msg_queue.is_closed()
            and not self._user_msg_queue.is_closed()
        )
