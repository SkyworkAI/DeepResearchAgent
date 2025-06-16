from typing import Any, Dict, List, Optional

import requests

from src.models.base import ApiModel, ChatMessage, tool_role_conversions
from src.models.message_manager import MessageManager
from src.proxy.local_proxy import PROXY_URL


class RestfulClient:
    def __init__(self, api_base: str, api_key: str, http_client=None):
        self.api_base = api_base
        self.api_key = api_key
        self.http_client = http_client

        self.url = "restful_api_url"

    def completion(self, model, messages, tools, tool_choice, **kwargs):
        proxies = {
            "http": PROXY_URL,
            "https": PROXY_URL,
        }

        headers = {"app_key": self.api_key, "Content-Type": "application/json"}

        model = model.split("/")[-1]
        data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        response = requests.post(self.url, json=data, headers=headers, proxies=proxies)

        return response.json()


class RestfulModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        http_client=None,
        **kwargs,
    ):
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
        return RestfulClient(api_base=self.api_base, api_key=self.api_key)

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
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
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
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
