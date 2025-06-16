from src.config import config
from src.models import model_manager
from src.registry import REGISTED_AGENTS, REGISTED_TOOLS
from src.tools import make_tool_instance

AUTHORIZED_IMPORTS = ["pandas", "requests", "numpy"]


def create_agent():
    if config.agent.use_hierarchical_agent:
        planning_agent_config = getattr(config.agent, "planning_agent_config")

        sub_agents_ids = planning_agent_config.managed_agents
        sub_agents = []
        for sub_agent_id in sub_agents_ids:
            if sub_agent_id not in REGISTED_AGENTS:
                raise ValueError(f"Agent ID '{sub_agent_id}' is not registered.")
            sub_agent_config = getattr(config.agent, f"{sub_agent_id}_config")

            tool_ids = sub_agent_config.tools
            tools = []
            for tool_id in tool_ids:
                if tool_id not in REGISTED_TOOLS:
                    raise ValueError(f"Tool ID '{tool_id}' is not registered.")
                tools.append(REGISTED_TOOLS[tool_id]())

            sub_agent = REGISTED_AGENTS[sub_agent_id](
                config=sub_agent_config,
                model=model_manager.registed_models[sub_agent_config.model_id],
                tools=tools,
                max_steps=sub_agent_config.max_steps,
                name=sub_agent_config.name,
                description=sub_agent_config.description,
                provide_run_summary=True,
            )

            sub_agents.append(sub_agent)

        sub_agent_tools = [make_tool_instance(agent) for agent in sub_agents]

        tool_ids = planning_agent_config.tools
        tools = []
        for tool_id in tool_ids:
            if tool_id not in REGISTED_TOOLS:
                raise ValueError(f"Tool ID '{tool_id}' is not registered.")
            tools.append(REGISTED_TOOLS[tool_id]())

        tools = tools + sub_agent_tools

        agent = REGISTED_AGENTS["planning_agent"](
            config=planning_agent_config,
            model=model_manager.registed_models[planning_agent_config.model_id],
            tools=tools,
            max_steps=planning_agent_config.max_steps,
            description=planning_agent_config.description,
            name=planning_agent_config.name,
            provide_run_summary=True,
        )

        return agent

    else:
        deep_analyzer_agent_config = getattr(config.agent, "deep_analyzer_agent_config")
        tools = []
        for tool_id in deep_analyzer_agent_config.tools:
            if tool_id not in REGISTED_TOOLS:
                raise ValueError(f"Tool ID '{tool_id}' is not registered.")
            tools.append(REGISTED_TOOLS[tool_id]())

        agent = REGISTED_AGENTS["deep_analyzer_agent"](
            config=deep_analyzer_agent_config,
            model=model_manager.registed_models[deep_analyzer_agent_config.model_id],
            tools=tools,
            max_steps=deep_analyzer_agent_config.max_steps,
            name=deep_analyzer_agent_config.name,
            description=deep_analyzer_agent_config.description,
            provide_run_summary=True,
        )

        return agent
