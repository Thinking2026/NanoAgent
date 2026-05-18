from __future__ import annotations

from enum import Enum
# ---------------------------------------------------------------------------
# Structured LLM error hierarchy
#
# Three layers of information on every LLMError:
#   1. ErrorCategory  – coarse grouping that drives retry/fallback routing
#   2. LLMErrorCode   – fine-grained code for logging, metrics, and alerting
#   3. CallerAction   – machine-readable instruction to the caller:
#        RETRY          – safe to retry the same provider (transient)
#        RETRY_BACKOFF  – retry same provider but honour retry_after / backoff
#        SWITCH_MODEL   – this model cannot serve the request; try next provider
#        DEGRADE        – content/context/quota issue; caller may downgrade
#                         (shorter prompt, no tools, simpler model, cached reply)
#        IGNORE         – benign empty/partial response; caller may skip silently
#        FATAL          – misconfiguration or auth; abort without retry
# ---------------------------------------------------------------------------


class ErrorCategory(str, Enum):
    TRANSIENT      = "TRANSIENT"                  # network / timeout / 5xx → retry same provider
    RATE_LIMIT     = "RATE_LIMIT"                 # 429 / quota → retry with backoff
    CONTEXT        = "CONTEXT"                    # context too long → trim / degrade
    AUTH           = "AUTH"                       # 401/403 / invalid key → switch provider
    CONTENT_POLICY = "CONTENT_POLICY"             # content filtered / safety block → degrade
    RESPONSE       = "RESPONSE"                   # bad / unparseable response → self-repair / skip
    CONFIG         = "CONFIG"                     # missing key / bad config → fatal
    RETRY_EXCEED   = "EXCEED_MAX_RETRY_ATTEMPTS"  # exceed max retry times for one provider → switch provider


class CallerAction(str, Enum):
    RETRY         = "RETRY"          # immediate retry, same provider
    RETRY_BACKOFF = "RETRY_BACKOFF"  # retry after delay (honour retry_after)
    SWITCH_MODEL  = "SWITCH_MODEL"   # must move to a different provider/model
    DEGRADE       = "DEGRADE"        # reduce scope (trim ctx / drop tools / use cache)
    IGNORE        = "IGNORE"         # benign; caller may silently skip
    FATAL         = "FATAL"          # abort; no recovery possible


class LLMNormalizedErrorCode(str, Enum):
    # ── Transient ────────────────────────────────────────────────────────────
    NETWORK_ERROR          = "NETWORK_ERROR"           # socket / DNS / connection refused
    TIMEOUT                = "TIMEOUT"                 # read or connect timeout
    HTTP_5XX               = "HTTP_5XX"                # 500/502/503/504 from provider
    PROVIDER_OVERLOADED    = "PROVIDER_OVERLOADED"     # 529 (Claude) / 503 overloaded body

    # ── Rate limit / quota ───────────────────────────────────────────────────
    RATE_LIMITED           = "RATE_LIMITED"            # 429 RPM/TPM limit
    QUOTA_EXCEEDED         = "QUOTA_EXCEEDED"          # monthly/daily hard quota hit

    # ── Context / length ─────────────────────────────────────────────────────
    CONTEXT_TOO_LONG       = "CONTEXT_TOO_LONG"        # prompt exceeds model context window
    OUTPUT_TOO_LONG        = "OUTPUT_TOO_LONG"         # max_tokens too large for model
    OUTPUT_TRUNCATED       = "OUTPUT_TRUNCATED"        # max_tokens too small and model truncate response
    INVALID_REQUEST        = "INVALID_REQUEST"         # 400 not covered by other codes

    # ── Auth / permission ────────────────────────────────────────────────────
    AUTH_FAILED            = "AUTH_FAILED"             # 401 / invalid API key
    PERMISSION_DENIED      = "PERMISSION_DENIED"       # 403 / model not enabled for key

    # ── Content policy ───────────────────────────────────────────────────────
    CONTENT_FILTERED       = "CONTENT_FILTERED"        # output blocked by safety filter
    INPUT_CONTENT_POLICY   = "INPUT_CONTENT_POLICY"    # input rejected by safety filter

    # ── Response quality ─────────────────────────────────────────────────────
    RESPONSE_ERROR         = "RESPONSE_ERROR"          # missing / structurally invalid response
    RESPONSE_PARSE_ERROR   = "RESPONSE_PARSE_ERROR"    # tool-call / JSON-mode parse failure
    EMPTY_RESPONSE         = "EMPTY_RESPONSE"          # choices present but all content empty
    EMPTY_CHOICES          = "EMPTY_CHOICES"           # choices array absent or empty
    TOOL_CALL_PARSE_ERROR  = "TOOL_CALL_PARSE_ERROR"   # tool-call arguments not valid JSON
    JSON_MODE_PARSE_ERROR  = "JSON_MODE_PARSE_ERROR"   # JSON-mode output not valid JSON
    JSON_REQUIRED_KEY_MISSING = "JSON_REQUIRED_KEY_MISSING"  # required top-level key absent after repair
    FINISH_REASON_LENGTH   = "FINISH_REASON_LENGTH"    # finish_reason == "length" (truncated)

    # ── Config ───────────────────────────────────────────────────────────────
    CONFIG_ERROR           = "CONFIG_ERROR"            # missing key / bad provider config

    # ── max retry ───────────────────────────────────────────────────────────────
    EXCEED_MAX_RETRY_TIMES = "EXCEED_MAX_RETRY_TIMES"  # exceed max retry attempts for single provider model(include fallback model)


# ---------------------------------------------------------------------------
# Canonical mapping: code → (category, caller_action)
# ---------------------------------------------------------------------------

_CODE_META: dict[LLMNormalizedErrorCode, tuple[ErrorCategory, CallerAction]] = {
    # Transient
    LLMNormalizedErrorCode.NETWORK_ERROR:         (ErrorCategory.TRANSIENT,      CallerAction.RETRY),
    LLMNormalizedErrorCode.TIMEOUT:               (ErrorCategory.TRANSIENT,      CallerAction.RETRY),
    LLMNormalizedErrorCode.HTTP_5XX:              (ErrorCategory.TRANSIENT,      CallerAction.RETRY_BACKOFF),
    LLMNormalizedErrorCode.PROVIDER_OVERLOADED:   (ErrorCategory.TRANSIENT,      CallerAction.RETRY_BACKOFF),

    # Rate limit / quota
    LLMNormalizedErrorCode.RATE_LIMITED:          (ErrorCategory.RATE_LIMIT,     CallerAction.RETRY_BACKOFF),
    LLMNormalizedErrorCode.QUOTA_EXCEEDED:        (ErrorCategory.RATE_LIMIT,     CallerAction.DEGRADE),

    # Context / length
    LLMNormalizedErrorCode.CONTEXT_TOO_LONG:      (ErrorCategory.CONTEXT,        CallerAction.SWITCH_MODEL),
    LLMNormalizedErrorCode.OUTPUT_TOO_LONG:       (ErrorCategory.CONTEXT,        CallerAction.SWITCH_MODEL),
    LLMNormalizedErrorCode.OUTPUT_TRUNCATED:      (ErrorCategory.CONTEXT,        CallerAction.SWITCH_MODEL),
    LLMNormalizedErrorCode.INVALID_REQUEST:       (ErrorCategory.CONTEXT,        CallerAction.SWITCH_MODEL),

    # Auth / permission
    LLMNormalizedErrorCode.AUTH_FAILED:           (ErrorCategory.AUTH,           CallerAction.SWITCH_MODEL),
    LLMNormalizedErrorCode.PERMISSION_DENIED:     (ErrorCategory.AUTH,           CallerAction.SWITCH_MODEL),

    # Content policy
    LLMNormalizedErrorCode.CONTENT_FILTERED:      (ErrorCategory.CONTENT_POLICY, CallerAction.DEGRADE),
    LLMNormalizedErrorCode.INPUT_CONTENT_POLICY:  (ErrorCategory.CONTENT_POLICY, CallerAction.DEGRADE),

    # Response quality
    LLMNormalizedErrorCode.RESPONSE_ERROR:        (ErrorCategory.RESPONSE,       CallerAction.SWITCH_MODEL),
    LLMNormalizedErrorCode.RESPONSE_PARSE_ERROR:  (ErrorCategory.RESPONSE,       CallerAction.DEGRADE),
    LLMNormalizedErrorCode.EMPTY_RESPONSE:        (ErrorCategory.RESPONSE,       CallerAction.DEGRADE),
    LLMNormalizedErrorCode.EMPTY_CHOICES:         (ErrorCategory.RESPONSE,       CallerAction.IGNORE),
    LLMNormalizedErrorCode.TOOL_CALL_PARSE_ERROR: (ErrorCategory.RESPONSE,       CallerAction.DEGRADE),
    LLMNormalizedErrorCode.JSON_MODE_PARSE_ERROR: (ErrorCategory.RESPONSE,       CallerAction.DEGRADE),
    LLMNormalizedErrorCode.JSON_REQUIRED_KEY_MISSING: (ErrorCategory.RESPONSE,   CallerAction.FATAL),
    LLMNormalizedErrorCode.FINISH_REASON_LENGTH:  (ErrorCategory.RESPONSE,       CallerAction.DEGRADE),

    # Config
    LLMNormalizedErrorCode.CONFIG_ERROR:          (ErrorCategory.CONFIG,         CallerAction.FATAL),
    LLMNormalizedErrorCode.EXCEED_MAX_RETRY_TIMES:(ErrorCategory.RETRY_EXCEED,   CallerAction.SWITCH_MODEL),
}


class LLMNormalizedError(Exception):
    """Structured LLM error carrying three layers of information.

    Attributes
    ----------
    code:          Fine-grained LLMErrorCode for logging / metrics.
    category:      Coarse ErrorCategory for routing decisions.
    caller_action: Machine-readable CallerAction that tells the caller exactly
                   what to do next (retry / switch / degrade / ignore / fatal).
    message:       Human-readable description including provider context.
    retry_after:   Seconds to wait before retrying (from Retry-After header).
    provider:      Name of the provider that raised this error (optional).
    raw_status:    Original HTTP status code, if applicable (optional).
    """

    def __init__(
        self,
        code: LLMNormalizedErrorCode,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str | None = None,
        raw_status: int | None = None,
    ) -> None:
        category, caller_action = _CODE_META[code]
        self.code = code
        self.category: ErrorCategory = category
        self.caller_action: CallerAction = caller_action
        self.message = message
        self.retry_after = retry_after
        self.provider = provider
        self.raw_status = raw_status
        super().__init__(
            f"[{category.value}/{code.value}/{caller_action.value}] {message}"
        )

    @property
    def is_retryable(self) -> bool:
        return self.caller_action in (CallerAction.RETRY, CallerAction.RETRY_BACKOFF)

    @property
    def is_degradable(self) -> bool:
        return self.caller_action == CallerAction.DEGRADE

    @property
    def is_fatal(self) -> bool:
        return self.caller_action == CallerAction.FATAL


# ---------------------------------------------------------------------------
# Legacy flat error codes (kept for tools, storage, and agent-level code)
# ---------------------------------------------------------------------------

AGENT_MAX_ITERATIONS_EXCEEDED = "AGENT_MAX_ITERATIONS_EXCEEDED"
CALCULATION_ERROR = "CALCULATION_ERROR"
CONFIG_ERROR = "CONFIG_ERROR"
EXCEL_TOOL_ERROR = "EXCEL_TOOL_ERROR"
EXCEL_TOOL_DEPENDENCY_ERROR = "EXCEL_TOOL_DEPENDENCY_ERROR"
EXCEL_TOOL_FILE_NOT_FOUND = "EXCEL_TOOL_FILE_NOT_FOUND"
EXCEL_TOOL_SHEET_EXISTS = "EXCEL_TOOL_SHEET_EXISTS"
EXCEL_TOOL_SHEET_NOT_FOUND = "EXCEL_TOOL_SHEET_NOT_FOUND"
FILE_TOOL_ERROR = "FILE_TOOL_ERROR"
LLM_ALL_PROVIDERS_FAILED = "LLM_ALL_PROVIDERS_FAILED"
LLM_NETWORK_ERROR = "LLM_NETWORK_ERROR"
LLM_PROVIDER_NOT_FOUND = "LLM_PROVIDER_NOT_FOUND"
LLM_RESPONSE_ERROR = "LLM_RESPONSE_ERROR"
LLM_RESPONSE_PARSE_ERROR = "LLM_RESPONSE_PARSE_ERROR"
LLM_TIMEOUT = "LLM_TIMEOUT"
SHELL_COMMAND_FAILED = "SHELL_COMMAND_FAILED"
SHELL_EXECUTION_ERROR = "SHELL_EXECUTION_ERROR"
SHELL_TIMEOUT = "SHELL_TIMEOUT"
SQL_QUERY_TOOL_ERROR = "SQL_QUERY_TOOL_ERROR"
SQL_SCHEMA_TOOL_ERROR = "SQL_SCHEMA_TOOL_ERROR"
STORAGE_CONFIG_ERROR = "STORAGE_CONFIG_ERROR"
STORAGE_DEPENDENCY_ERROR = "STORAGE_DEPENDENCY_ERROR"
STORAGE_QUERY_ERROR = "STORAGE_QUERY_ERROR"
STORAGE_RESOURCE_NOT_FOUND = "STORAGE_RESOURCE_NOT_FOUND"
STORAGE_RESOURCE_REQUIRED = "STORAGE_RESOURCE_REQUIRED"
TOOL_ARGUMENT_ERROR = "TOOL_ARGUMENT_ERROR"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
TOOL_TIMEOUT = "TOOL_TIMEOUT"
PYTHON_TOOL_ERROR = "PYTHON_TOOL_ERROR"
PYTHON_TOOL_FORBIDDEN_IMPORT = "PYTHON_TOOL_FORBIDDEN_IMPORT"
PYTHON_TOOL_TIMEOUT = "PYTHON_TOOL_TIMEOUT"
PYTHON_TOOL_RESOURCE_LIMIT = "PYTHON_TOOL_RESOURCE_LIMIT"
SEARCH_TOOL_ERROR = "SEARCH_TOOL_ERROR"
SEARCH_TOOL_TIMEOUT = "SEARCH_TOOL_TIMEOUT"
SEARCH_TOOL_PROVIDER_ERROR = "SEARCH_TOOL_PROVIDER_ERROR"
VECTOR_SCHEMA_TOOL_ERROR = "VECTOR_SCHEMA_TOOL_ERROR"
VECTOR_SEARCH_TOOL_ERROR = "VECTOR_SEARCH_TOOL_ERROR"
FLIGHT_SEARCH_TOOL_ERROR = "FLIGHT_SEARCH_TOOL_ERROR"
FLIGHT_SEARCH_TOOL_TIMEOUT = "FLIGHT_SEARCH_TOOL_TIMEOUT"
FLIGHT_SEARCH_TOOL_PROVIDER_ERROR = "FLIGHT_SEARCH_TOOL_PROVIDER_ERROR"
HOTEL_SEARCH_TOOL_ERROR = "HOTEL_SEARCH_TOOL_ERROR"
HOTEL_SEARCH_TOOL_TIMEOUT = "HOTEL_SEARCH_TOOL_TIMEOUT"
HOTEL_SEARCH_TOOL_PROVIDER_ERROR = "HOTEL_SEARCH_TOOL_PROVIDER_ERROR"
TRAVEL_GUIDE_TOOL_ERROR = "TRAVEL_GUIDE_TOOL_ERROR"
TRAVEL_GUIDE_TOOL_TIMEOUT = "TRAVEL_GUIDE_TOOL_TIMEOUT"
TRAVEL_GUIDE_TOOL_PROVIDER_ERROR = "TRAVEL_GUIDE_TOOL_PROVIDER_ERROR"
RESTAURANT_SEARCH_TOOL_ERROR = "RESTAURANT_SEARCH_TOOL_ERROR"
RESTAURANT_SEARCH_TOOL_TIMEOUT = "RESTAURANT_SEARCH_TOOL_TIMEOUT"
RESTAURANT_SEARCH_TOOL_PROVIDER_ERROR = "RESTAURANT_SEARCH_TOOL_PROVIDER_ERROR"
PARAMETER_FORGET_SET = "PARAMETER_FORGET_SET"
JSON_LOAD_ERROR = "JSON_LOAD_ERROR"
UNKNOWN_LOGIC_ERROR = "UNKNOWN_LOGIC_ERROR"
TRUNCATION_FAILED = "TRUNCATION_FAILED"
TASK_ANALYSIS_LOW_CONFIDENCE = "TASK_ANALYSIS_LOW_CONFIDENCE"

class PipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_pipeline_error(code: str, message: str) -> PipelineError:
    return PipelineError(code=code, message=message)


class ConfigError(Exception):
    def __init__(self, message: str = "", code: str = CONFIG_ERROR) -> None:
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_config_error(code: str, message: str) -> ConfigError:
    return ConfigError(message=message, code=code)

class ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_tool_error(code: str, message: str) -> ToolError:
    return ToolError(code=code, message=message)

class LogicError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_logic_error(code: str, message: str) -> LogicError:
    return LogicError(code=code, message=message)

class JsonError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_json_error(code: str, message: str) -> JsonError:
    return JsonError(code=code, message=message)

class HttpError(Exception):
    """Raised for HTTP error responses; carries status code and Retry-After."""

    def __init__(self, status: int, body: str, retry_after: float | None = None) -> None:
        self.status = status
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"HTTP {status}: {body}")

def build_http_error(status: int, body: str, retry_after: float | None = None) -> HttpError:
    return HttpError(status=status, body=body, retry_after=retry_after)
