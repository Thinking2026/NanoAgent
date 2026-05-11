from __future__ import annotations

from abc import ABC, abstractmethod


class PromptRenderer(ABC):
    """Abstract interface for prompt template rendering."""

    @abstractmethod
    def render(self, template_name: str, context: dict) -> str:
        """Render *template_name* with *context* and return the result string.

        template_name uses slash-separated paths relative to the templates root,
        e.g. "planner/task_context.j2" or "analyzer/system.j2".
        """
        ...

    @abstractmethod
    def render_string(self, source: str, context: dict) -> str:
        """Render an inline template string."""
        ...
