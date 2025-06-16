import tiktoken


def get_token_count(prompt: str, model: str = "gpt-4o") -> int:
    """
    Get the number of tokens in a prompt.
    :param prompt: The prompt to count tokens for.
    :param model: The model to use for tokenization. Default is "gpt-4o".
    :return: The number of tokens in the prompt.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))
