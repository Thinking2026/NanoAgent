from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from infra.rendering_engine.renderer import PromptRenderer


class TemplateNotFoundError(RuntimeError):
    pass


class TemplateRenderError(RuntimeError):
    pass


class Jinja2PromptRenderer(PromptRenderer):
    """Jinja2-backed prompt renderer.

    Uses FileSystemLoader pointed at the templates/ directory.
    StrictUndefined raises on any missing variable — prevents silent
    prompt corruption from typos in context keys.
    """

    def __init__(self, templates_dir: Path | str | None = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._register_filters()
        self._env.globals["json"] = json

    def render(self, template_name: str, context: dict) -> str:
        try:
            tmpl = self._env.get_template(template_name)
        except TemplateNotFound as exc:
            raise TemplateNotFoundError(f"Prompt template not found: {template_name}") from exc
        try:
            return tmpl.render(**context)
        except Exception as exc:
            raise TemplateRenderError(f"Failed to render template '{template_name}': {exc}") from exc

    def render_string(self, source: str, context: dict) -> str:
        try:
            return self._env.from_string(source).render(**context)
        except Exception as exc:
            raise TemplateRenderError(f"Failed to render inline template: {exc}") from exc

    def _register_filters(self) -> None:
        env = self._env

        def tojson_filter(value, indent: int = 2, ensure_ascii: bool = False) -> str:
            return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)

        def format_step_inputs(inputs: list) -> str:
            if not inputs:
                return "  (none)"
            lines = []
            for inp in inputs:
                line = f"  - [{inp.source}] {inp.value}"
                if getattr(inp, "step_ref", None) is not None:
                    line += f" (from step {inp.step_ref})"
                if getattr(inp, "constraint_note", ""):
                    line += f" [constraint: {inp.constraint_note}]"
                lines.append(line)
            return "\n".join(lines)

        def format_dependencies(deps: list) -> str:
            if not deps:
                return "  (none)"
            lines = []
            for dep in deps:
                dep_detail = ", ".join(dep.depends_on) if dep.depends_on else "output"
                lines.append(f"  - step {dep.step_order} must complete first, providing: {dep.depends_on}")
            return "\n".join(lines)

        def format_constraints(constraints: list) -> str:
            if not constraints:
                return "  (none)"
            lines = []
            for c in constraints:
                label = "STRICT" if c.strict else "soft"
                lines.append(f"  - [{label}/{c.source}] {c.description}")
            return "\n".join(lines)

        def format_entities(entities: list) -> str:
            if not entities:
                return "  (none)"
            lines = []
            for e in entities:
                line = f"  - [{e.type}] {e.value}"
                if getattr(e, "normalized", False) and e.raw != e.value:
                    line += f" (raw: '{e.raw}')"
                lines.append(line)
            return "\n".join(lines)

        def format_risks(risks: list) -> str:
            if not risks:
                return "  (none)"
            lines = []
            for r in risks:
                if isinstance(r, str):
                    lines.append(f"  - {r}")
                else:
                    lines.append(f"  - [{r.severity.upper()}/{r.category}] {r.description}")
            return "\n".join(lines)

        def or_none(value) -> str:
            return value if value else "(none)"

        env.filters["tojson"] = tojson_filter
        env.filters["format_step_inputs"] = format_step_inputs
        env.filters["format_dependencies"] = format_dependencies
        env.filters["format_constraints"] = format_constraints
        env.filters["format_entities"] = format_entities
        env.filters["format_risks"] = format_risks
        env.filters["or_none"] = or_none
