from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.models.base import MessageRole
from src.utils import encode_image_base64, make_image_url

DEFAULT_ANTHROPIC_MODELS = [
    "claude37-sonnet",
    "claude37-sonnet-thinking",
]
UNSUPPORTED_STOP_MODELS = ["claude37-sonnet", "o4-mini", "o3", "langchain-o3"]
UNSUPPORTED_TOOL_CHOICE_MODELS = [
    "claude37-sonnet",
]


class MessageManager:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_clean_message_list(
        self,
        message_list: List[Dict[str, str]],
        role_conversions: Dict[MessageRole, MessageRole] = {},
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Subsequent messages with the same role will be concatenated to a single message.
        output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

        Args:
            message_list (`list[dict[str, str]]`): List of chat messages.
            role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
            convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
            flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
        """
        output_message_list = []
        message_list = deepcopy(message_list)  # Avoid modifying the original list
        for message in message_list:
            role = message["role"]
            if role not in MessageRole.roles():
                raise ValueError(
                    f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
                )

            if role in role_conversions:
                message["role"] = role_conversions[role]
            # encode images if needed
            if isinstance(message["content"], list):
                for element in message["content"]:
                    if element["type"] == "image":
                        assert (
                            not flatten_messages_as_text
                        ), f"Cannot use images with {flatten_messages_as_text=}"
                        if convert_images_to_image_urls:
                            model_id = self.model_id.split("/")[-1]

                            if model_id in DEFAULT_ANTHROPIC_MODELS:
                                element.update(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": encode_image_base64(
                                                element.pop("image")
                                            ),
                                        },
                                    }
                                )
                            else:
                                element.update(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": make_image_url(
                                                encode_image_base64(
                                                    element.pop("image")
                                                )
                                            )
                                        },
                                    }
                                )
                        else:
                            element["image"] = encode_image_base64(element["image"])

            if (
                len(output_message_list) > 0
                and message["role"] == output_message_list[-1]["role"]
            ):
                assert isinstance(message["content"], list), (
                    "Error: wrong content:" + str(message["content"])
                )
                if flatten_messages_as_text:
                    output_message_list[-1]["content"] += (
                        "\n" + message["content"][0]["text"]
                    )
                else:
                    for el in message["content"]:
                        if (
                            el["type"] == "text"
                            and output_message_list[-1]["content"][-1]["type"] == "text"
                        ):
                            # Merge consecutive text messages rather than creating new ones
                            output_message_list[-1]["content"][-1]["text"] += (
                                "\n" + el["text"]
                            )
                        else:
                            output_message_list[-1]["content"].append(el)
            else:
                if flatten_messages_as_text:
                    content = message["content"][0]["text"]
                else:
                    content = message["content"]
                output_message_list.append(
                    {"role": message["role"], "content": content}
                )
        return output_message_list

    def get_tool_json_schema(self, tool: Any, model_id: Optional[str] = None) -> Dict:
        properties = deepcopy(tool.parameters["properties"])

        required = []
        for key, value in properties.items():
            if value["type"] == "any":
                value["type"] = "string"
            if not ("nullable" in value and value["nullable"]):
                required.append(key)

        model_id = model_id.split("/")[-1]

        if model_id in DEFAULT_ANTHROPIC_MODELS:
            return {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

    def get_clean_completion_kwargs(self, completion_kwargs: Dict[str, Any]):
        model_id = self.model_id.split("/")[-1]

        if model_id in UNSUPPORTED_TOOL_CHOICE_MODELS:
            completion_kwargs.pop("tool_choice", None)
        if model_id in UNSUPPORTED_STOP_MODELS:
            completion_kwargs.pop("stop", None)
        return completion_kwargs
