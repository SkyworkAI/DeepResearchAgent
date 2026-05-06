"""
LiteLLM chat provider — route to 100+ LLM providers via a unified interface.

Model strings use the ``provider/model`` format, e.g.
``anthropic/claude-sonnet-4-20250514``, ``azure/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``,
``openai/gpt-4o``, ``ollama/llama3``, etc.
See https://docs.litellm.ai/docs/providers for the full list.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from src.message.types import Message
from src.model.openai.serializer import OpenAIChatSerializer
from src.model.types import LLMExtra, LLMResponse
from src.logger import logger

try:
    from src.tool.types import Tool
except ImportError:
    Tool = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool


class ChatLiteLLM(BaseModel):
    """
    A wrapper around the LiteLLM SDK that provides a unified interface
    for calling 100+ LLM providers (OpenAI, Anthropic, Google, Azure,
    Bedrock, Ollama, etc.) through a single API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    model: str
    temperature: Optional[float] = 0.7
    max_completion_tokens: Optional[int] = 16384
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 2
    reasoning: Optional[Dict[str, Any]] = None

    @property
    def provider(self) -> str:
        return "litellm"

    @property
    def name(self) -> str:
        return str(self.model)

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for ChatLiteLLM. Install it with: pip install litellm"
            )

        try:
            openai_messages = OpenAIChatSerializer.serialize_messages(messages)

            params: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "drop_params": True,
            }

            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.max_completion_tokens is not None:
                params["max_completion_tokens"] = self.max_completion_tokens
            if self.api_key:
                params["api_key"] = self.api_key
            if self.api_base:
                params["api_base"] = self.api_base
            if self.reasoning:
                params.update(self.reasoning)

            if tools:
                formatted_tools = OpenAIChatSerializer.serialize_tools(tools)
                if formatted_tools:
                    params["tools"] = formatted_tools

            if response_format:
                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    params["response_format"] = OpenAIChatSerializer.serialize_response_format(
                        response_format
                    )
                elif isinstance(response_format, BaseModel):
                    params["response_format"] = OpenAIChatSerializer.serialize_response_format(
                        response_format
                    )
                elif isinstance(response_format, dict):
                    params["response_format"] = response_format

            if stream:
                params["stream"] = True

            params.update(kwargs)

            response = await litellm.acompletion(**params)
            return await self._format_response(response, tools, response_format)

        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            return LLMResponse(
                success=False,
                message=f"LiteLLM error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        try:
            resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

            if not resp_dict.get("choices"):
                return LLMResponse(
                    success=False,
                    message="No choices in response",
                    extra=LLMExtra(data={"raw_response": resp_dict}),
                )

            choice = resp_dict["choices"][0]
            message = choice.get("message", {})
            usage = resp_dict.get("usage")
            finish_reason = choice.get("finish_reason")

            if tools and message.get("tool_calls"):
                formatted_lines = []
                functions = []
                for tool_call in message["tool_calls"]:
                    fn = tool_call.get("function", {})
                    name = fn.get("name", "")
                    try:
                        arguments = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    if arguments:
                        args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")
                    functions.append({"name": name, "args": arguments})

                return LLMResponse(
                    success=True,
                    message="\n".join(formatted_lines),
                    extra=LLMExtra(
                        data={
                            "raw_response": resp_dict,
                            "functions": functions,
                            "usage": usage,
                            "finish_reason": finish_reason,
                        }
                    ),
                )

            elif response_format and isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                content = message.get("content", "")
                if not content:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(data={"raw_response": resp_dict}),
                    )
                try:
                    data = json.loads(content)
                    parsed_model = response_format.model_validate(data)
                    model_dict = parsed_model.model_dump()
                    field_lines = [f"{k}={v!r}" for k, v in model_dict.items()]
                    formatted_message = (
                        f"Response result:\n\n{response_format.__name__}(\n"
                        + ",\n".join(f"    {line}" for line in field_lines)
                        + "\n)"
                    )
                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=LLMExtra(
                            parsed_model=parsed_model,
                            data={
                                "raw_response": resp_dict,
                                "usage": usage,
                                "finish_reason": finish_reason,
                            },
                        ),
                    )
                except (json.JSONDecodeError, Exception) as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse structured response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": content}),
                    )

            else:
                content = message.get("content", "")
                return LLMResponse(
                    success=True,
                    message=content,
                    extra=LLMExtra(
                        data={
                            "raw_response": resp_dict,
                            "usage": usage,
                            "finish_reason": finish_reason,
                        }
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to format LiteLLM response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)}),
            )
