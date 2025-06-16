import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.models.base import (
    ApiModel,
    ChatMessage,
    tool_role_conversions,
)
from src.models.message_manager import MessageManager

STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]


class InferenceClientModel(ApiModel):
    """A class to interact with Hugging Face's Inference Providers for language model interaction.

    This model allows you to communicate with Hugging Face's models using Inference Providers. It can be used in both serverless mode, with a dedicated endpoint, or even with a local URL, supporting features like stop sequences and grammar customization.

    Providers include Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.

    Parameters:
        model_id (`str`, *optional*, default `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference.
            This can be a model identifier from the Hugging Face model hub or a URL to a deployed Inference Endpoint.
            Currently, it defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`, but this may change in the future.
        provider (`str`, *optional*):
            Name of the provider to use for inference. A list of supported providers can be found in the [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
            Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order [here](https://hf.co/settings/inference-providers).
            If `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference Providers'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the Hugging Face InferenceClient.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        api_key (`str`, *optional*):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
        bill_to (`str`, *optional*):
            The billing account to use for the requests. By default the requests are billed on the user’s account. Requests can only be billed to
            an organization the user is a member of, and which has subscribed to Enterprise Hub.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the Hugging Face InferenceClient.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = InferenceClientModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     provider="nebius",
    ...     token="your_hf_token_here",
    ...     max_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        provider: str | None = None,
        token: str | None = None,
        timeout: int = 120,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        api_key: str | None = None,
        bill_to: str | None = None,
        base_url: str | None = None,
        http_client=None,
        **kwargs,
    ):
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )

        self.http_client = http_client
        self.message_manager = MessageManager(model_id=model_id)

        token = token if token is not None else api_key
        if token is None:
            token = os.getenv("HUGGINEFACE_API_KEY")
        self.client_kwargs = {
            **(client_kwargs or {}),
            "model": model_id,
            "provider": provider,
            "token": token,
            "timeout": timeout,
            "bill_to": bill_to,
            "base_url": base_url,
        }
        super().__init__(
            model_id=model_id, custom_role_conversions=custom_role_conversions, **kwargs
        )

    def create_client(self):
        """Create the Hugging Face client."""
        from huggingface_hub import InferenceClient

        return InferenceClient(**self.client_kwargs)

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        http_client=None,
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

        if http_client:
            completion_kwargs["client"] = http_client

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(
            completion_kwargs
        )

        return completion_kwargs

    async def __call__(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if (
            response_format is not None
            and self.client_kwargs["provider"] not in STRUCTURED_GENERATION_PROVIDERS
        ):
            raise ValueError(
                "InferenceClientModel only supports structured outputs with these providers:"
                + ", ".join(STRUCTURED_GENERATION_PROVIDERS)
            )
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            # response_format=response_format,
            http_client=self.http_client,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        response = self.client.chat_completion(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        first_message = ChatMessage.from_dict(
            asdict(response.choices[0].message),
            raw=response,
        )
        return self.postprocess_message(first_message, tools_to_call_from)
