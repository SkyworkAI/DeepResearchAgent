from typing import Any, Dict

REGISTED_AGENTS: Dict[str, Any] = {}
REGISTED_TOOLS: Dict[str, Any] = {}


def register_agent(agent_id_or_cls=None):
    """
    Decorator to register an agent class with a unique ID.

    Usage:
        @register_agent
        class MyAgent: ...

        @register_agent("custom_id")
        class MyOtherAgent: ...
    """

    def decorator(cls):
        # Determine the registration key: use custom ID or class name
        agent_id = agent_id_or_cls if isinstance(agent_id_or_cls, str) else cls.__name__
        # Check for duplicate registration
        if agent_id in REGISTED_AGENTS:
            raise ValueError(f"Agent ID '{agent_id}' is already registered.")
        # Register the class (not instance)
        REGISTED_AGENTS[agent_id] = cls
        return cls

    # Support both @register_agent and @register_agent("custom_id") usages
    if callable(agent_id_or_cls):
        # Used as @register_agent
        return decorator(agent_id_or_cls)
    else:
        # Used as @register_agent("custom_id")
        return decorator


def register_tool(tool_id_or_cls=None):
    """
    Decorator to register a tool class with a unique ID.

    Usage:
        @register_tool
        class MyTool: ...

        @register_tool("custom_id")
        class MyOtherTool: ...
    """

    def decorator(cls):
        # Determine the registration key: use custom ID or class name
        tool_id = tool_id_or_cls if isinstance(tool_id_or_cls, str) else cls.__name__

        # Check for duplicate registration
        if tool_id in REGISTED_TOOLS:
            raise ValueError(f"Tool ID '{tool_id}' is already registered.")

        # Register the class (not instance)
        REGISTED_TOOLS[tool_id] = cls
        return cls

    # Support both @register_tool and @register_tool("custom_id") usages
    if callable(tool_id_or_cls):
        # Used as @register_tool
        return decorator(tool_id_or_cls)
    else:
        # Used as @register_tool("custom_id")
        return decorator
