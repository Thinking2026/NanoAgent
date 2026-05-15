from __future__ import annotations

import logging

from agent.events.events import DomainEvent
from schemas.event_bus import EventBus,EventHandler,TypeEvent
from utils.log.log import zap

logger = logging.getLogger(__name__)

def _resolve_key(event_type: TypeEvent) -> str:
    """Normalise an event type reference to its string key."""
    if isinstance(event_type, str):
        return event_type
    return event_type.__name__


class InMemoryEventBus(EventBus):
    """Synchronous in-process event bus.

    Handlers are invoked in subscription order.  A failing handler is logged
    and skipped so that remaining handlers always run.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}

    # ------------------------------------------------------------------
    # EventBus interface
    # ------------------------------------------------------------------

    def publish(self, event: DomainEvent) -> None:
        handlers = list(self._handlers.get(type(event).__name__, []))
        if not handlers:
            logger.info("no handler can process this event", zap.any("event", event.content))
            return
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event handler raised an exception",
                    extra={
                        "event": event.content,
                        "handler": getattr(handler, "__qualname__", repr(handler)),
                    },
                )

    def subscribe(self, event_type: TypeEvent, handler: EventHandler) -> None:
        self._handlers.setdefault(_resolve_key(event_type), []).append(handler)

    def unsubscribe(self, event_type: TypeEvent, handler: EventHandler) -> None:
        key = _resolve_key(event_type)
        handlers = self._handlers.get(key, [])
        if handler in handlers:
            handlers.remove(handler)