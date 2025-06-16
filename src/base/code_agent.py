# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from typing import (
    TYPE_CHECKING,
    Any,
)

import yaml
from rich.console import Group
from rich.text import Text

if TYPE_CHECKING:
    pass

from src.base.multistep_agent import MultiStepAgent, PromptTemplates, populate_template
from src.exception import (
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
)
from src.logger import (
    YELLOW_HEX,
    LogLevel,
)
from src.memory import ActionStep, ToolCall
from src.models import (
    ChatMessage,
    Model,
)
from src.tools import Tool
from src.tools.executor.local_python_executor import (
    LocalPythonExecutor,
    PythonExecutor,
    fix_final_answer_code,
)
from src.tools.executor.remote_executors import DockerExecutor, E2BExecutor
from src.utils import BASE_BUILTIN_MODULES, parse_code_blobs, truncate_content


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor_type (`str`, default `"local"`): Which executor type to use between `"local"`, `"e2b"`, or `"docker"`.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        grammar: dict[str, str] | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor_type: str | None = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        **kwargs,
    ):
        self.additional_authorized_imports = (
            additional_authorized_imports if additional_authorized_imports else []
        )
        self.authorized_imports = sorted(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports)
        )
        self.max_print_outputs_length = max_print_outputs_length
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("src.base.prompts")
            .joinpath("code_agent.yaml")
            .read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        self.executor_type = executor_type or "local"
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor = self.create_python_executor()

    def create_python_executor(self) -> PythonExecutor:
        match self.executor_type:
            case "e2b" | "docker":
                if self.managed_agents:
                    raise Exception(
                        "Managed agents are not yet supported with remote code execution."
                    )
                if self.executor_type == "e2b":
                    return E2BExecutor(
                        self.additional_authorized_imports,
                        self.logger,
                        **self.executor_kwargs,
                    )
                else:
                    return DockerExecutor(
                        self.additional_authorized_imports,
                        self.logger,
                        **self.executor_kwargs,
                    )
            case "local":
                return LocalPythonExecutor(
                    self.additional_authorized_imports,
                    max_print_outputs_length=self.max_print_outputs_length,
                )
            case _:  # if applicable
                raise ValueError(f"Unsupported executor type: {self.executor_type}")

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> None | Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        try:
            additional_args = (
                {"grammar": self.grammar} if self.grammar is not None else {}
            )
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                output_text = ""
                with Live(
                    "", console=self.logger.console, vertical_overflow="visible"
                ) as live:
                    for event in output_stream:
                        if event.content is not None:
                            output_text += event.content
                            live.update(Markdown(output_text))

                model_output = output_text
                chat_message = ChatMessage(role="assistant", content=model_output)
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
            else:
                chat_message: ChatMessage = self.model(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                model_output = chat_message.content
                self.logger.log_markdown(
                    content=model_output,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # This adds <end_code> sequence to the history.
            # This will nudge ulterior LLM calls to finish with <end_code>, thus efficiently stopping generation.
            if model_output and model_output.strip().endswith("```"):
                model_output += "<end_code>"
                memory_step.model_output_message.content = model_output

            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(
                f"Error in generating model output:\n{e}", self.logger
            ) from e

        ### Parse output ###
        try:
            code_action = fix_final_answer_code(parse_code_blobs(model_output))
        except Exception as e:
            error_msg = (
                f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            )
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        ### Execute action ###
        self.logger.log_code(
            title="Executing parsed code:", content=code_action, level=LogLevel.INFO
        )
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if (
                hasattr(self.python_executor, "state")
                and "_print_outputs" in self.python_executor.state
            ):
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(
                        Group(*execution_outputs_console), level=LogLevel.INFO
                    )
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        return output if is_final_answer else None

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        agent_dict = super().to_dict()
        agent_dict["authorized_imports"] = self.authorized_imports
        agent_dict["executor_type"] = self.executor_type
        agent_dict["executor_kwargs"] = self.executor_kwargs
        agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
        }
        # Filter out None values
        code_agent_kwargs = {
            k: v for k, v in code_agent_kwargs.items() if v is not None
        }
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)
