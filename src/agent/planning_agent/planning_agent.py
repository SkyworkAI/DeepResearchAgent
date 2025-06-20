from typing import (
    Any,
    Callable,
    Optional
)
# Remove imports that are now in BaseAgent or not directly needed
# import json
# import yaml
# from rich.panel import Panel
# from rich.text import Text

from src.tools import AsyncTool
from src.exception import (
    AgentGenerationError,
    AgentParsingError,
    AgentToolExecutionError,
    AgentToolCallError
)
from src.base.async_multistep_agent import (PromptTemplates,
                                            populate_template,
                                            AsyncMultiStepAgent
                                            )
from src.memory import (ActionStep,
                        ToolCall,
                        AgentMemory)
from src.logger import (LogLevel, 
                        YELLOW_HEX, 
                        logger)
from src.models import Model, parse_json_if_needed, ChatMessage
from src.utils.agent_types import (
    AgentAudio,
    AgentImage,
)
from src.registry import register_agent
from src.utils import assemble_project_path

# Import the new base class
from src.agent.base_agent import BaseAgent


@register_agent("planning_agent") # 使用字符串常量而不是类属性，避免前向引用问题
class PlanningAgent(BaseAgent):
    AGENT_NAME = "planning_agent"  # Define a specific agent name

    def __init__(
        self,
        config,  # Specific configuration for this agent
        tools: list[AsyncTool],
        model: Model,
        # prompt_templates_path is now a required parameter for BaseAgent
        # It will be extracted from config
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        # name: str | None = None, # Handled by BaseAgent using AGENT_NAME
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        **kwargs
    ):
        # Extract prompt_templates_path from the agent's configuration
        # Original code used: self.config.template_path
        prompt_templates_path = assemble_project_path(config.template_path)

        super().__init__(
            config=config,  # Pass the full configuration to the base class
            tools=tools,
            model=model,
            prompt_templates_path=prompt_templates_path,  # Pass the path
            max_steps=max_steps,
            add_base_tools=add_base_tools,
            verbosity_level=verbosity_level,
            grammar=grammar,
            managed_agents=managed_agents,
            step_callbacks=step_callbacks,
            planning_interval=planning_interval,
            # name is handled by BaseAgent
            description=description,
            provide_run_summary=provide_run_summary,
            final_answer_checks=final_answer_checks,
            **kwargs
        )
        # All other initialization (prompts, memory) is now handled by BaseAgent
        # Any PlanningAgent-specific initialization can be added here

    # All other methods (initialize_system_prompt, initialize_user_prompt, 
    # initialize_task_instruction, _substitute_state_variables, 
    # execute_tool_call, step) are now inherited from BaseAgent

