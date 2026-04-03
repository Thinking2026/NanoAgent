from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid4().hex


@dataclass(slots=True)
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: str
    end_time: str | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


class SpanHandle:
    def __init__(
        self,
        tracer: "Tracer | None",
        record: SpanRecord | None = None,
        is_root: bool = False,
    ) -> None:
        self._tracer = tracer
        self._record = record
        self._is_root = is_root
        self._finished = False

    @property
    def trace_id(self) -> str | None:
        return None if self._record is None else self._record.trace_id

    @property
    def span_id(self) -> str | None:
        return None if self._record is None else self._record.span_id

    def add_attributes(self, attributes: dict[str, Any] | None) -> None:
        if self._record is None or not attributes:
            return
        self._record.attributes.update(attributes)

    def finish(
        self,
        status: str = "ok",
        error: BaseException | dict[str, Any] | None = None,
    ) -> None:
        if self._finished or self._record is None or self._tracer is None:
            self._finished = True
            return
        self._finished = True
        self._tracer._finish_span(self, status=status, error=error)

    def __enter__(self) -> "SpanHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.finish(status="error", error=exc)
            return
        self.finish(status="ok")


class Tracer:
    def __init__(
        self,
        enabled: bool = True,
        output_path: str | Path = "runtime",
        capture_payloads: bool = False,
        max_content_length: int = 1000,
    ) -> None:
        self._enabled = enabled
        self._output_dir = self._resolve_output_dir(output_path)
        self._capture_payloads = capture_payloads
        self._max_content_length = max(64, int(max_content_length))
        self._lock = threading.Lock()
        self._local = threading.local()
        if self._enabled:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanHandle:
        if not self._enabled:
            return SpanHandle(None)
        trace_id = _new_id()
        self._local.trace_id = trace_id
        self._local.span_stack = []
        return self._start_span(
            name=name,
            kind="session",
            attributes=attributes,
            trace_id=trace_id,
            parent_span_id=None,
            is_root=True,
        )

    def start_span(
        self,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanHandle:
        if not self._enabled:
            return SpanHandle(None)
        trace_id = self.current_trace_id()
        if trace_id is None:
            return SpanHandle(None)
        return self._start_span(
            name=name,
            kind=kind,
            attributes=attributes,
            trace_id=trace_id,
            parent_span_id=self.current_span_id(),
        )

    def current_trace_id(self) -> str | None:
        return getattr(self._local, "trace_id", None)

    def current_span_id(self) -> str | None:
        stack = getattr(self._local, "span_stack", None) or []
        if not stack:
            return None
        return stack[-1]

    def _start_span(
        self,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None,
        trace_id: str,
        parent_span_id: str | None,
        is_root: bool = False,
    ) -> SpanHandle:
        record = SpanRecord(
            trace_id=trace_id,
            span_id=_new_id(),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=utc_now_iso(),
            attributes=self._normalize_attributes(attributes or {}),
        )
        stack = list(getattr(self._local, "span_stack", []))
        stack.append(record.span_id)
        self._local.span_stack = stack
        return SpanHandle(self, record=record, is_root=is_root)

    def _finish_span(
        self,
        handle: SpanHandle,
        status: str,
        error: BaseException | dict[str, Any] | None,
    ) -> None:
        record = handle._record
        if record is None:
            return
        record.status = status
        record.end_time = utc_now_iso()
        if error is not None:
            record.error = self._normalize_error(error)
        self._write_record(record)
        stack = list(getattr(self._local, "span_stack", []))
        if record.span_id in stack:
            stack.remove(record.span_id)
        self._local.span_stack = stack
        if handle._is_root:
            self._local.trace_id = None
            self._local.span_stack = []

    def _write_record(self, record: SpanRecord) -> None:
        if not self._enabled:
            return
        serialized = json.dumps(
            {
                "trace_id": record.trace_id,
                "span_id": record.span_id,
                "parent_span_id": record.parent_span_id,
                "name": record.name,
                "kind": record.kind,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "status": record.status,
                "attributes": record.attributes,
                "error": record.error,
            },
            ensure_ascii=False,
        )
        with self._lock:
            with self._build_log_path().open("a", encoding="utf-8") as file_handle:
                file_handle.write(serialized + "\n")

    def _build_log_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d%H")
        return self._output_dir / f"{timestamp}_trace.jsonl"

    @staticmethod
    def _resolve_output_dir(output_path: str | Path) -> Path:
        path = Path(output_path)
        if path.suffix:
            return path.parent if str(path.parent) != "." else Path(".")
        return path

    def _normalize_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_value(value)
            for key, value in attributes.items()
        }

    def _normalize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if self._capture_payloads:
                return value[:self._max_content_length]
            return f"<str len={len(value)}>"
        if isinstance(value, dict):
            return {
                str(key): self._normalize_value(nested_value)
                for key, nested_value in list(value.items())[:20]
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in list(value)[:20]]
        return repr(value)[:self._max_content_length]

    def _normalize_error(self, error: BaseException | dict[str, Any]) -> dict[str, Any]:
        if isinstance(error, dict):
            return self._normalize_attributes(error)
        return {
            "type": error.__class__.__name__,
            "message": str(error),
        }
