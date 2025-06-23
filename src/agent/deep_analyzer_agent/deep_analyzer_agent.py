from typing import (
    Any,
    Callable,
    Optional
)

from src.tools import AsyncTool
from src.logger import LogLevel
from src.models import Model
from src.registry import register_agent
from src.agent.base_agent import BaseAgent
from src.utils import assemble_project_path


@register_agent("deep_analyzer_agent")
class DeepAnalyzerAgent(BaseAgent):
    AGENT_NAME = "deep_analyzer_agent"

    def __init__(
        self,
        config,
        tools: list[AsyncTool],
        model: Model,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        **kwargs
    ):
        prompt_templates_path = assemble_project_path(config.template_path)
        super().__init__(
            config=config,
            tools=tools,
            model=model,
            prompt_templates_path=prompt_templates_path,
            max_steps=max_steps,
            add_base_tools=add_base_tools,
            verbosity_level=verbosity_level,
            grammar=grammar,
            managed_agents=managed_agents,
            step_callbacks=step_callbacks,
            planning_interval=planning_interval,
            description=description,
            provide_run_summary=provide_run_summary,
            final_answer_checks=final_answer_checks,
            **kwargs
        )

