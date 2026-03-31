from __future__ import annotations

import json
from typing import Any

from agent.agent import Agent, AgentExecutionResult
from schemas import AgentError, ChatMessage, LLMRequest, LLMResponse, SessionStatus, ToolResult, build_error


class ReActAgent(Agent):
    def run(
        self,
        session_status: SessionStatus,
        user_message: ChatMessage | None,
    ) -> AgentExecutionResult:
        request, request_result = self._build_llm_request(session_status, user_message)
        if request_result is not None:
            return request_result

        llm_response, error_result = self._call_llm_with_timeout_handling(request)
        if error_result is not None:
            return error_result

        parsed_response, parse_result = self._parse_llm_api_response(llm_response)
        if parse_result is not None:
            return parse_result

        return self._route_llm_response(parsed_response)

    def _build_llm_request(
        self,
        session_status: SessionStatus,
        user_message: ChatMessage | None,
    ) -> tuple[LLMRequest | None, AgentExecutionResult | None]:
        rag_context = []
        '''
        if user_message is not None and user_message.content.strip():
            rag_context, rag_result = self._retrieve_rag_context(user_message.content) #TODO 缺少容错fallback
            if rag_result is not None:
                return None, rag_result
        '''

        conversation = self._shared_context.get_conversation_history()
        next_message, request_result = self._build_next_message(session_status, user_message)
        if request_result is not None:
            return None, request_result

        if next_message is not None:
            conversation.append(next_message)
        return (
            self._message_formatter.build_request(
                system_prompt=self._shared_context.get_system_prompt(),
                conversation=conversation,
                tools=self._tool_registry.get_tool_schemas(),
                context=rag_context,
            ),
            None,
        )

    def _build_next_message(
        self,
        session_status: SessionStatus,
        user_message: ChatMessage | None,
    ) -> tuple[ChatMessage | None, AgentExecutionResult | None]:
        if session_status == SessionStatus.NEW_TASK:
            if user_message is None:#按照现在的设计不可能走到这里
                raise build_error("MISSING_USER_MESSAGE", "A new task requires a user message.")

        if user_message is not None and user_message.content.strip():
            message = ChatMessage(role="user", content=user_message.content.strip())
            self._shared_context.append_conversation_message(message)
            return message, None

        return None, None

    def _call_llm_with_timeout_handling(
        self,
        request: LLMRequest,
    ) -> tuple[LLMResponse | None, AgentExecutionResult | None]:
        try:
            self._cur_react_attempt_iterations += 1
            return self._llm_client.generate(request), None
        except TimeoutError as exc: #TODO 如何fallback需要想方案
            return None, self._build_error_result(
                f"LLM call timed out. Temporary timeout strategy applied: {exc}"
            )
        except AgentError as exc:
            if exc.code == "LLM_ALL_PROVIDERS_FAILED":
                return None, self._build_error_result("Can not get response from llm")
            if exc.code == "LLM_TIMEOUT":
                return None, self._build_error_result(
                    f"LLM call timed out. Temporary timeout strategy applied: {exc}"
                )
            if exc.code in {"LLM_RESPONSE_PARSE_ERROR", "LLM_RESPONSE_ERROR"}:
                return None, self._build_error_result(
                    f"LLM returned a response that could not be parsed: {exc}"
                )
            raise

    def _parse_llm_api_response(
        self,
        response: Any,
    ) -> tuple[LLMResponse | None, AgentExecutionResult | None]:
        try:
            if response is None:
                return None, self._build_error_result("LLM returned an empty response.")
            if not isinstance(response, LLMResponse):
                return None, self._build_error_result(
                    f"LLM returned an unexpected response format: {response}"
                )
            return self._message_formatter.parse_response(response), None
        except Exception as exc:
            return None, self._build_error_result(
                f"LLM returned an unexpected response format: {exc}"
            )

    def _route_llm_response(self, response: LLMResponse) -> AgentExecutionResult:
        self._shared_context.append_conversation_message(response.assistant_message)
        llm_messages: list[ChatMessage] = []
        llm_content = response.assistant_message.content.strip()
        if llm_content:
            llm_messages.append(
                ChatMessage(
                    role="assistant",
                    content=llm_content,
                    metadata={"source": "llm"},
                )
            )

        if response.tool_calls:
            tool_result = self._handle_tool_calls(response)
            return AgentExecutionResult(
                user_messages=[*llm_messages, *tool_result.user_messages],
                should_reset=tool_result.should_reset,
            )

        external_query = self._extract_external_query(response)
        if external_query:
            rag_result = self._handle_external_lookup(external_query)
            return AgentExecutionResult(
                user_messages=[*llm_messages, *rag_result.user_messages],
                should_reset=rag_result.should_reset,
            )

        return AgentExecutionResult(
            user_messages=[self._format_final_conclusion(response)],
            should_reset=True,
        )

    @staticmethod
    def _format_final_conclusion(response: LLMResponse) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=response.assistant_message.content,
            metadata=response.assistant_message.metadata,
        )

    def _handle_tool_calls(self, response: LLMResponse) -> AgentExecutionResult:
        intermediate_messages: list[ChatMessage] = []
        for tool_call in response.tool_calls:
            result = self._tool_registry.execute(
                tool_call.name,
                tool_call.arguments,
                tool_call.call_id,
            )
            if not result.success:
                return self._build_tool_error_result(tool_call.name, result)

            observation = self._message_formatter.format_tool_observation(
                tool_name=tool_call.name,
                output=result.output,
                call_id=tool_call.call_id,
            )
            self._shared_context.append_conversation_message(observation)
            intermediate_messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"[tool:{tool_call.name}] {result.output}",
                    metadata={"source": "tool", "tool_name": tool_call.name},
                )
            )
        return AgentExecutionResult(user_messages=intermediate_messages)

    @staticmethod
    def _extract_external_query(response: LLMResponse) -> str | None:
        metadata_query = response.assistant_message.metadata.get("external_query")
        if isinstance(metadata_query, str) and metadata_query.strip():
            return metadata_query.strip()

        content = response.assistant_message.content.strip()
        prefix = "RAG_QUERY:"
        if content.startswith(prefix):
            query = content[len(prefix):].strip()
            return query or None
        return None

    def _handle_external_lookup(self, query: str) -> AgentExecutionResult:
        rag_context, rag_result = self._retrieve_rag_context(query)
        if rag_result is not None:
            return rag_result

        observation = self._message_formatter.format_rag_observation(
            query=query,
            context=rag_context,
        )
        self._shared_context.append_conversation_message(observation)
        return AgentExecutionResult(
            user_messages=[
                ChatMessage(
                    role="assistant",
                    content=f"[rag:{query}] {json.dumps(rag_context, ensure_ascii=False)}",
                    metadata={"source": "rag", "query": query},
                )
            ]
        )

    def _retrieve_rag_context(
        self,
        query: str,
    ) -> tuple[list[dict], AgentExecutionResult | None]:
        try:
            return self._rag_service.retrieve(query), None
        except TimeoutError as exc:
            return [], self._build_error_result(f"External knowledge lookup timed out: {exc}")
        except AgentError as exc:
            if exc.code == "RAG_TIMEOUT":
                return [], self._build_error_result(f"External knowledge lookup timed out: {exc}")
            return [], self._build_error_result(f"External knowledge lookup failed: {exc}")
        except Exception as exc:
            return [], self._build_error_result(f"External knowledge lookup failed: {exc}")

    def _build_tool_error_result(self, tool_name: str, result: ToolResult) -> AgentExecutionResult:
        error = result.error
        if error is None:
            error = build_error("TOOL_EXECUTION_ERROR", f"Tool `{tool_name}` failed with an unknown error.")

        if error.code == "TOOL_NOT_FOUND":
            content = f"Requested tool `{tool_name}` was not found."
        elif "TIMEOUT" in error.code:
            content = f"Tool `{tool_name}` timed out: {error.message}"
        else:
            content = f"Tool `{tool_name}` returned an error: {error.message}"

        return AgentExecutionResult(
            user_messages=[
                ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={"error_code": error.code, "tool_name": tool_name},
                )
            ],
            should_reset=True,
        )

    @staticmethod
    def _build_error_result(content: str) -> AgentExecutionResult:
        return AgentExecutionResult(
            user_messages=[ChatMessage(role="assistant", content=content)],
            should_reset=True,
        )
