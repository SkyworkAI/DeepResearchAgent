from src.registry import register_tool
from src.tools import AsyncTool, ToolResult
from src.tools.executor.local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)


@register_tool("python_interpreter")
class PythonInterpreterTool(AsyncTool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The python code to run in interpreter.",
            },
        },
        "required": ["code"],
    }
    output_type = "any"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(
                set(BASE_BUILTIN_MODULES) | set(authorized_imports)
            )
        self.parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                        f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                    ),
                }
            },
            "required": ["code"],
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        super().__init__(*args, **kwargs)

    async def forward(self, code: str) -> ToolResult:
        try:
            state = {}
            output = str(
                self.python_evaluator(
                    code,
                    state=state,
                    static_tools=self.base_python_tools,
                    authorized_imports=self.authorized_imports,
                )[0]  # The second element is boolean is_final_answer
            )

            output = f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"

            result = ToolResult(output=output, error=None)
        except Exception as e:
            result = ToolResult(
                output=None,
                error=str(e),
            )
        return result
