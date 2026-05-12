from __future__ import annotations

import ast
import math
from typing import Any

from schemas import CALCULATION_ERROR, TOOL_ARGUMENT_ERROR, ToolResult, build_pipeline_error
from tools.tool_base import BaseTool, build_tool_output


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Safely evaluate an arithmetic expression. "
        "Supports +, -, *, /, //, %, **, parentheses, and math functions: "
        "sqrt, sin, cos, tan, log, log10, exp, floor, ceil, abs, pow, round. "
        "Constants: pi, e, tau."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "The arithmetic expression to evaluate. "
                    "Example: '(1250.5 * 18 + 320 * 6) / 1.13'"
                ),
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    _ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], Any] = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left ** right,
    }
    _ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], Any] = {
        ast.UAdd: lambda value: +value,
        ast.USub: lambda value: -value,
    }
    _ALLOWED_FUNCTIONS: dict[str, Any] = {
        "abs": abs,
        "ceil": math.ceil,
        "exp": math.exp,
        "floor": math.floor,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
    }
    _ALLOWED_CONSTANTS: dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            error = build_pipeline_error(
                TOOL_ARGUMENT_ERROR,
                "Calculator tool requires a non-empty expression.",
            )
            return self._error_result(error)

        try:
            parsed = ast.parse(expression, mode="eval")
            value = self._evaluate(parsed.body)
        except ZeroDivisionError:
            error = build_pipeline_error(CALCULATION_ERROR, "Division by zero is not allowed.")
            return self._error_result(error)
        except ValueError as exc:
            error = build_pipeline_error(CALCULATION_ERROR, f"Invalid calculation: {exc}")
            return self._error_result(error)
        except SyntaxError as exc:
            error = build_pipeline_error(CALCULATION_ERROR, f"Invalid expression syntax: {exc.msg}")
            return self._error_result(error)
        except TypeError as exc:
            error = build_pipeline_error(CALCULATION_ERROR, f"Unsupported expression: {exc}")
            return self._error_result(error)
        except Exception as exc:
            error = build_pipeline_error(CALCULATION_ERROR, f"Failed to evaluate expression: {exc}")
            return self._error_result(error)

        return ToolResult(
            output=build_tool_output(
                success=True,
                data={
                    "expression": expression,
                    "result": value,
                },
            ),
            success=True,
        )

    def _evaluate(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError("Only numeric constants are allowed.")

        if isinstance(node, ast.Name):
            if node.id in self._ALLOWED_CONSTANTS:
                return self._ALLOWED_CONSTANTS[node.id]
            raise TypeError(f"Unknown identifier: {node.id}")

        if isinstance(node, ast.BinOp):
            operator = self._ALLOWED_BINARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            return operator(self._evaluate(node.left), self._evaluate(node.right))

        if isinstance(node, ast.UnaryOp):
            operator = self._ALLOWED_UNARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            return operator(self._evaluate(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("Only direct function calls are allowed.")
            function = self._ALLOWED_FUNCTIONS.get(node.func.id)
            if function is None:
                raise TypeError(f"Unsupported function: {node.func.id}")
            if node.keywords:
                raise TypeError("Keyword arguments are not allowed.")
            evaluated_args = [self._evaluate(argument) for argument in node.args]
            return function(*evaluated_args)

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    @staticmethod
    def _error_result(error) -> ToolResult:
        return ToolResult(
            output=build_tool_output(success=False, error=error),
            success=False,
            error=error,
        )
