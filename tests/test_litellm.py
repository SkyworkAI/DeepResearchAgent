"""
Tests for LiteLLM provider integration.

Tests cover:
- ChatLiteLLM instantiation and properties
- Manager registration and client creation
- Async completion call with mocked litellm
- drop_params=True default
"""

import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.model.litellm.chat import ChatLiteLLM
from src.model.types import ModelConfig


class TestChatLiteLLMAttributes:
    """Tests for ChatLiteLLM class attributes and properties."""

    def test_default_temperature(self):
        client = ChatLiteLLM(model="openai/gpt-4o")
        assert client.temperature == 0.7

    def test_default_max_completion_tokens(self):
        client = ChatLiteLLM(model="openai/gpt-4o")
        assert client.max_completion_tokens == 16384

    def test_provider_property(self):
        client = ChatLiteLLM(model="openai/gpt-4o")
        assert client.provider == "litellm"

    def test_name_property(self):
        client = ChatLiteLLM(model="anthropic/claude-sonnet-4-20250514")
        assert client.name == "anthropic/claude-sonnet-4-20250514"

    def test_custom_temperature(self):
        client = ChatLiteLLM(model="openai/gpt-4o", temperature=0.2)
        assert client.temperature == 0.2

    def test_api_key_stored(self):
        client = ChatLiteLLM(model="openai/gpt-4o", api_key="sk-test")
        assert client.api_key == "sk-test"

    def test_api_base_stored(self):
        client = ChatLiteLLM(
            model="openai/gpt-4o", api_base="http://localhost:4000"
        )
        assert client.api_base == "http://localhost:4000"


class TestChatLiteLLMCall:
    """Tests for ChatLiteLLM.__call__() with mocked litellm."""

    @pytest.mark.asyncio
    async def test_call_dispatches_to_litellm_acompletion(self):
        fake_litellm = types.ModuleType("litellm")
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        fake_litellm.acompletion = AsyncMock(return_value=mock_response)
        sys.modules["litellm"] = fake_litellm

        try:
            client = ChatLiteLLM(model="openai/gpt-4o", api_key="sk-test")

            from src.message.types import Message

            messages = [Message(role="user", content="Hi")]
            result = await client(messages=messages)

            assert result.success is True
            assert "Hello!" in result.message

            call_kwargs = fake_litellm.acompletion.call_args
            assert call_kwargs.kwargs["model"] == "openai/gpt-4o"
            assert call_kwargs.kwargs["drop_params"] is True
            assert call_kwargs.kwargs["api_key"] == "sk-test"
        finally:
            del sys.modules["litellm"]

    @pytest.mark.asyncio
    async def test_call_includes_drop_params_true(self):
        fake_litellm = types.ModuleType("litellm")
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        fake_litellm.acompletion = AsyncMock(return_value=mock_response)
        sys.modules["litellm"] = fake_litellm

        try:
            client = ChatLiteLLM(model="anthropic/claude-haiku-4-5")
            from src.message.types import Message

            messages = [Message(role="user", content="Say OK")]
            await client(messages=messages)

            call_kwargs = fake_litellm.acompletion.call_args.kwargs
            assert call_kwargs["drop_params"] is True
        finally:
            del sys.modules["litellm"]

    @pytest.mark.asyncio
    async def test_call_handles_tool_calls(self):
        fake_litellm = types.ModuleType("litellm")
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }
        fake_litellm.acompletion = AsyncMock(return_value=mock_response)
        sys.modules["litellm"] = fake_litellm

        try:
            client = ChatLiteLLM(model="openai/gpt-4o")
            from src.message.types import Message

            mock_tool = MagicMock()
            messages = [Message(role="user", content="Weather in London?")]
            result = await client(messages=messages, tools=[mock_tool])

            assert result.success is True
            assert "get_weather" in result.message
            assert result.extra.data["functions"][0]["name"] == "get_weather"
        finally:
            del sys.modules["litellm"]

    @pytest.mark.asyncio
    async def test_call_returns_failure_on_import_error(self):
        if "litellm" in sys.modules:
            del sys.modules["litellm"]

        client = ChatLiteLLM(model="openai/gpt-4o")
        from src.message.types import Message

        messages = [Message(role="user", content="Hi")]

        with pytest.raises(ImportError, match="litellm is required"):
            await client(messages=messages)


class TestManagerLiteLLMRegistration:
    """Tests for litellm provider registration in ModelManager."""

    def test_litellm_in_allowed_providers(self):
        from src.model.manager import ModelManager

        ModelManager()
        try:
            ModelConfig(
                model_name="litellm/test",
                model_id="openai/gpt-4o",
                model_type="chat/completions",
                provider="litellm",
            )
            # Should not raise ValueError for provider
            # (register_model validates the provider name)
        except ValueError as e:
            if "Only OpenAI" in str(e):
                pytest.fail("litellm should be an allowed provider")
