from typing import Any

from src.models.base import (
    ApiModel,
    ChatMessage,
    MessageRole,
)


class AmazonBedrockServerModel(ApiModel):
    """
    A model class for interacting with Amazon Bedrock Server models through the Bedrock API.

    This class provides an interface to interact with various Bedrock language models,
    allowing for customized model inference, guardrail configuration, message handling,
    and other parameters allowed by boto3 API.

    Parameters:
        model_id (`str`):
            The model identifier to use on Bedrock (e.g. "us.amazon.nova-pro-v1:0").
        client (`boto3.client`, *optional*):
            A custom boto3 client for AWS interactions. If not provided, a default client will be created.
        client_kwargs (dict[str, Any], *optional*):
            Keyword arguments used to configure the boto3 client if it needs to be created internally.
            Examples include `region_name`, `config`, or `endpoint_url`.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
            Defaults to converting all roles to "user" role to enable using all the Bedrock models.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs
            Additional keyword arguments passed directly to the underlying API calls.

    Example:
        Creating a model instance with default settings:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0'
        ... )

        Creating a model instance with a custom boto3 client:
        >>> import boto3
        >>> client = boto3.client('bedrock-runtime', region_name='us-west-2')
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client=client
        ... )

        Creating a model instance with client_kwargs for internal client creation:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client_kwargs={'region_name': 'us-west-2', 'endpoint_url': 'https://custom-endpoint.com'}
        ... )

        Creating a model instance with inference and guardrail configurations:
        >>> additional_api_config = {
        ...     "inferenceConfig": {
        ...         "maxTokens": 3000
        ...     },
        ...     "guardrailConfig": {
        ...         "guardrailIdentifier": "identify1",
        ...         "guardrailVersion": 'v1'
        ...     },
        ... }
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='anthropic.claude-3-haiku-20240307-v1:0',
        ...     **additional_api_config
        ... )
    """

    def __init__(
        self,
        model_id: str,
        client=None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        self.client_kwargs = client_kwargs or {}

        # Bedrock only supports `assistant` and `user` roles.
        # Many Bedrock models do not allow conversations to start with the `assistant` role, so the default is set to `user/user`.
        # This parameter is retained for future model implementations and extended support.
        custom_role_conversions = custom_role_conversions or {
            MessageRole.SYSTEM: MessageRole.USER,
            MessageRole.ASSISTANT: MessageRole.USER,
            MessageRole.TOOL_CALL: MessageRole.USER,
            MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API doesn't support flatten messages, must be a list of messages
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> dict:
        """
        Overrides the base method to handle Bedrock-specific configurations.

        This implementation adapts the completion keyword arguments to align with
        Bedrock's requirements, ensuring compatibility with its unique setup and
        constraints.
        """
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Bedrock support stop_sequence using Inference Config
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            **kwargs,
        )

        # Not all models in Bedrock support `toolConfig`. Also, smolagents already include the tool call in the prompt,
        # so adding `toolConfig` could cause conflicts. We remove it to avoid issues.
        completion_kwargs.pop("toolConfig", None)

        # The Bedrock API does not support the `type` key in requests.
        # This block of code modifies the object to meet Bedrock's requirements.
        for message in completion_kwargs.get("messages", []):
            for content in message.get("content", []):
                if "type" in content:
                    del content["type"]

        return {
            "modelId": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        try:
            import boto3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'bedrock' extra to use AmazonBedrockServerModel: `pip install 'smolagents[bedrock]'`"
            ) from e

        return boto3.client("bedrock-runtime", **self.client_kwargs)

    async def __call__(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # self.client is created in ApiModel class
        response = self.client.converse(**completion_kwargs)

        # Get first message
        response["output"]["message"]["content"] = response["output"]["message"][
            "content"
        ][0]["text"]

        self._last_input_token_count = response["usage"]["inputTokens"]
        self._last_output_token_count = response["usage"]["outputTokens"]

        return ChatMessage.from_dict(
            response["output"]["message"],
            raw=response,
        )
