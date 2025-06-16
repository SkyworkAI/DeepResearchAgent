import copy

from src.logger import logger
from src.models import MessageRole, Model


async def prepare_response(
    original_task: str, inner_messages, reformulation_model: Model
) -> str:
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]

    # The first message just repeats the question, so remove it
    # if len(inner_messages) > 1:
    #    del inner_messages[0]

    # copy them to this context
    try:
        for message in inner_messages:
            if not message.get("content"):
                continue
            message = copy.deepcopy(message)
            message["role"] = MessageRole.USER
            messages.append(message)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": str(inner_messages)}]

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
* You MUST pay attention to the required units of the calculation result. For example, if the question asks "how many thousand hours...", then the answer `1000 hours` should be `1`, not `1000`.
* You MUST pay attention to extracting key stage names, personal names, and location names when the task required.
* If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
* If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
* If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
* If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        }
    )

    response = await reformulation_model(messages)
    response = response.content

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    logger.info(f"> Reformulated answer: {final_answer}")

    return final_answer
