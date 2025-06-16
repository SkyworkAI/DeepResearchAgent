import warnings
from typing import Any, Dict, List, Optional

from src.models.base import (
    ApiModel,
    ChatMessage,
    tool_role_conversions,
)
from src.models.message_manager import MessageManager


class LiteLLMModel(ApiModel):
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base=None,
        api_key=None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        http_client=None,
        **kwargs,
    ):
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        self.http_client = http_client

        self.message_manager = MessageManager(model_id=model_id)

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """Create the LiteLLM client."""
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        http_client=None,
        timeout: Optional[int] = 300,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }
        # Handle timeout
        if timeout is not None:
            completion_kwargs["timeout"] = timeout

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [
                        self.message_manager.get_tool_json_schema(
                            tool, model_id=self.model_id
                        )
                        for tool in tools_to_call_from
                    ],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        if http_client:
            completion_kwargs["client"] = http_client

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(
            completion_kwargs
        )

        return completion_kwargs

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            http_client=self.http_client,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        response = self.client.completion(**completion_kwargs)

        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}
            ),
            raw=response,
        )
        return self.postprocess_message(first_message, tools_to_call_from)
