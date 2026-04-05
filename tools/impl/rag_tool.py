from __future__ import annotations

from schemas import AgentError, ToolResult, build_error
from rag.rag_service import RAGService
from tools.tools import BaseTool, build_tool_output


class RAGTool(BaseTool):
    name = "rag_search"
    description = (
        "Search the external knowledge base for relevant background information. "
        "Use this tool when the answer depends on facts that may exist in the configured RAG storage."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query used to retrieve relevant knowledge snippets.",
            },
            "top_k": {
                "type": "integer",
                "description": "The maximum number of matches to return.",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            error = build_error("TOOL_ARGUMENT_ERROR", "RAG tool requires a non-empty query.")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        top_k = self._normalize_top_k(arguments.get("top_k", 3))
        if isinstance(top_k, AgentError):
            return ToolResult(
                output=build_tool_output(success=False, error=top_k),
                success=False,
                error=top_k,
            )

        try:
            matches = self._rag_service.retrieve(query, top_k=top_k)
        except AgentError as exc:
            return ToolResult(
                output=build_tool_output(success=False, error=exc),
                success=False,
                error=exc,
            )
        except Exception as exc:
            error = build_error("RAG_TOOL_ERROR", f"RAG tool failed unexpectedly: {exc}")
            return ToolResult(
                output=build_tool_output(success=False, error=error),
                success=False,
                error=error,
            )

        return ToolResult(
            output=build_tool_output(
                success=True,
                data={
                    "query": query,
                    "top_k": top_k,
                    "match_count": len(matches),
                    "source": self._rag_service.get_source_name(),
                    "matches": matches,
                },
            ),
            success=True,
        )

    @staticmethod
    def _normalize_top_k(value: object) -> int | AgentError:
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            return build_error("TOOL_ARGUMENT_ERROR", "RAG tool top_k must be an integer.")

        if top_k < 1 or top_k > 10:
            return build_error("TOOL_ARGUMENT_ERROR", "RAG tool top_k must be between 1 and 10.")
        return top_k
