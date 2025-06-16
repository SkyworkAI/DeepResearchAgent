from src.models import Model
from src.tools import AsyncTool, ToolResult
from src.tools.markdown.mdconvert import MarkitdownConverter

_FILE_READER_DESCRIPTION = """Call this tool to read a file as markdown.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx", ".pdb", '.zip'], and all other types of files.
* If the file is an image, use the `deep_analyzer` tool instead!
"""


class FileReaderTool(AsyncTool):
    name: str = "read_file"
    description: str = _FILE_READER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file you want to read as text. Must be a '.obj' file, like '.pdf'.",
            },
        },
        "required": ["file_path"],
        "additionalProperties": False,
    }
    output_type = "any"

    def __init__(self, model: Model, text_limit: int = 50000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

        self.converter: MarkitdownConverter = MarkitdownConverter(
            use_llm=False, model_id="gpt-4.1", timeout=30
        )

    async def forward(self, file_path: str) -> ToolResult:
        """Read a file and return its content as text."""

        result = self.converter.convert(file_path)

        result = ToolResult(output=result.text_content, error=None)
        return result
