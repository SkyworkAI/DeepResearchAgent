import os
from typing import Any, Dict, Optional

from PIL import Image

from src.config import config
from src.logger import logger
from src.models import model_manager
from src.models.base import MessageRole
from src.registry import register_tool
from src.tools import AsyncTool, ToolResult
from src.tools.markdown.mdconvert import MarkitdownConverter

_DEEP_ANALYZER_DESCRIPTION = """A tool that performs systematic, step-by-step analysis or calculation of a given task, optionally leveraging information from external resources such as attached file or uri to provide comprehensive reasoning and answers.
* At least one of `task` or `source` must be provided. When both are available, the tool will analyze and solve the task in the context of the provided source.
* The `source` can be a local file path or an uri. Support file extensions and uri are as follows:
 - Text: txt, doc, docx, ppt, pptx, csv, pdf, json, jsonl, jsonld, py, pdb, xml...
 - Image: png, jpg, jpeg...
 - Audio: mp3, m4a, wav...
 - Video: mp4, mov...
 - Archive: zip, rar... (NOTE: DO NOT need to unpack the archive, this tool will automatically handle it.)
 - Uri: https://xx.html, http://xx.htm, http://xx.pdf, file://xx, data://...
"""

_DEEP_ANALYZER_INSTRUCTION = """You should step-by-step analyze the task and/or the attached content.
* When the task involves playing a game or performing calculations. Please consider the conditions imposed by the game or calculation rules. You may take extreme conditions into account.
* When the task involves spelling words, you must ensure that the spelling rules are followed and that the resulting word is meaningful.
* When the task involves compute the area in a specific polygon. You should separate the polygon into sub-polygons and ensure that the area of each sub-polygon is computable (e.g, rectangle, circle, triangle, etc.). Step-by-step to compute the area of each sub-polygon and sum them up to get the final area.
* When the task involves calculation and statistics, it is essential to consider all constraints. Failing to account for these constraints can easily lead to statistical errors.

Here is the task:
"""

_DEEP_ANALYZER_SUMMARY_DESCRIPTION = """Please conduct a step-by-step analysis of the outputs from different models. Compare their results, identify discrepancies, extract the accurate components, eliminate the incorrect ones, and synthesize a coherent summary."""


@register_tool("deep_analyzer")
class DeepAnalyzerTool(AsyncTool):
    name: str = "deep_analyzer"
    description: str = _DEEP_ANALYZER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {
                "description": "The task to be analyzed and should be solved. If not provided, the tool will focus solely on captioning the attached files or linked URLs.",
                "type": "string",
                "nullable": True,
            },
            "source": {
                "description": "The attached file or uri to be analyzed. The tool will process and interpret the content of the file or webpage.",
                "type": "string",
                "nullable": True,
            },
        },
        "required": [],
        "additionalProperties": False,
    }
    output_type = "any"

    def __init__(self):
        self.analyzer_config = config.deep_analyzer_tool

        self.analyzer_models = {
            model_id: model_manager.registed_models[model_id]
            for model_id in self.analyzer_config.analyzer_model_ids
        }
        self.summary_model = model_manager.registed_models[
            self.analyzer_config.summarizer_model_id
        ]

        self.converter: MarkitdownConverter = MarkitdownConverter(
            use_llm=False,
            model_id="gpt-4.1",
            timeout=30,
        )

        super(DeepAnalyzerTool, self).__init__()

    async def _analyze(
        self, model, task: Optional[str] = None, source: Optional[str] = None
    ) -> str:
        add_note = False
        if not task:
            add_note = True
            task = "Please write a detailed caption for the attached file or uri."

        task = _DEEP_ANALYZER_INSTRUCTION + task
        content = [
            {"type": "text", "text": task},
        ]

        if source:
            ext = os.path.splitext(source)[-1].lower()

            if ext in [".png", ".jpg", ".jpeg"]:
                content.append(
                    {
                        "type": "image",
                        "image": Image.open(source),
                    }
                )
            else:
                extracted_content = self.converter.convert(source).text_content
                content.append(
                    {
                        "type": "text",
                        "text": " - Attached file content: \n\n" + extracted_content,
                    }
                )

        messages = [
            {
                "role": MessageRole.USER,
                "content": content,
            }
        ]

        response = await model(
            messages=messages,
        )

        try:
            output = response.content
        except Exception:
            raise Exception(f"Response format unexpected: {response}")

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output

    async def _summarize(
        self,
        model,
        analysis: Dict[str, Any],
    ) -> str:
        """
        Summarize the analysis and provide a final answer.
        """
        prompt = _DEEP_ANALYZER_SUMMARY_DESCRIPTION

        prompt += "Analysis: \n"
        for model_name, analysis in analysis.items():
            prompt += f"{model_name}:\n{analysis}\n\n"

        content = [
            {"type": "text", "text": prompt},
        ]

        messages = [
            {
                "role": MessageRole.USER,
                "content": content,
            }
        ]

        response = await model(
            messages=messages,
        )

        try:
            output = response.content
        except Exception:
            raise Exception(f"Response format unexpected: {response}")

        return output

    async def forward(
        self, task: Optional[str] = None, source: Optional[str] = None
    ) -> ToolResult:
        """
        Forward the task and/or source to the analyzer model and get the analysis.
        """
        if not task and not source:
            raise ValueError("At least one of task or source should be provided.")

        analysis = {}
        for model_name, model in self.analyzer_models.items():
            analysis[model_name] = await self._analyze(model, task, source)
            logger.info(f"{model_name}:\n{analysis[model_name]}\n")

        summary = await self._summarize(self.summary_model, analysis)

        logger.info(f"Summary:\n{summary}\n")

        # Construct the output
        output = "Analysis of models:\n"
        for model_name, analysis in analysis.items():
            output += f"{model_name}:\n{analysis}\n\n"
        output += f"Summary:\n{summary}\n"

        result = ToolResult(
            output=output,
            error=None,
        )

        return result
