# Deep Research Agent

English | [中文说明](README_zh.md)

Deep Research Agent is a self-evolution protocol and runtime for LLM-based agent systems.

Recent agent protocols often under-specify cross-entity **lifecycle/context management**, **version tracking**, and **safe evolution update interfaces**, which encourages monolithic compositions and brittle glue code. Deep Research Agent addresses this by decoupling **what evolves** from **how evolution occurs**:

- **RSPL (Resource Substrate Protocol Layer)**: models *prompts, agents, tools, environments, and memory* as protocol-registered resources with explicit **state**, **lifecycle**, and **versioned** interfaces.
- **SEPL (Self Evolution Protocol Layer)**: specifies a closed-loop operator interface to **propose**, **assess**, and **commit** improvements with auditable lineage and **rollback**.

Built on Deep Research Agent, the system includes a **Deep Research Agent** style tool-calling agent that can dynamically instantiate/retrieve/refine resources and improve during execution.

## Self-evolution at a glance

At a high level, Deep Research Agent supports an iterative loop:

- **Act**: an agent produces actions/outputs using an LLM and the available tools/environments.
- **Observe**: capture outcomes, traces, intermediate reasoning, and environment feedback.
- **Optimize**: update prompts/solutions/variables using an optimizer (e.g., reflection or RL-style methods).
- **Remember**: persist summaries/insights/records to memory for later steps and sessions.

## Core building blocks

- **Agents (`src/agent/`)**: runtime logic that decides *what to do next* (planning, tool-calling, domain agents, etc.).
- **Tools (`src/tool/`)**: callable capabilities exposed to agents (workflow tools + default tools).
- **Environments (`src/environment/`)**: stateful interfaces that tools/agents can interact with (filesystem, trading backtest envs, browser/mobile envs, etc.).
- **Memory (`src/memory/`)**: session/event memory systems for summarization, insights, and long-term state.
- **Optimizers (`src/optimizer/`)**: self-improvement algorithms that turn feedback into updated prompts/solutions/variables (reflection, GRPO, Reinforce++, etc.).
- **Tracing & versioning (`src/tracer/`, `src/version/`)**: record trajectories and manage iterative artifacts across runs.
- **Config system (`configs/`, `src/config/`)**: MMEngine-style configs to compose agents/tools/envs/memory/models consistently.

## Design goals

- **Composable**: add/replace agents, tools, environments, memory systems, and optimizers without rewriting the whole stack.
- **Inspectable**: structured traces and memory events make it easier to analyze failures and improvement steps.
- **Evolvable**: explicit optimizers + persistent memory enable iterative refinement rather than one-shot inference.

## Repository layout

```
DeepResearchAgent/
  configs/                 # config composition (agents/tools/envs/memory/models)
  src/
    agent/                 # agents
    environment/           # environments
    tool/                  # tools
    memory/                # memory systems
    optimizer/             # self-evolution optimizers
    model/                 # model manager + provider backends
    prompt/                # prompt templates / prompt manager
    tracer/                # tracing
    version/               # versioning
  libs/                    # vendored libraries
  workdir/                 # runtime artifacts (logs, traces, results, etc.)
```

## Optional: run a Tool-Calling Agent

For a standalone, keyless web search and page fetching example using the project's
MCP adapter, see [Parallel Search MCP](examples/parallel_search_mcp.md).

Prerequisites:
- Install dependencies in your environment
- Copy `.env.template` to `.env` and set a model API key (e.g. `OPENROUTER_API_KEY=...`)

Example:

```bash
python examples/run_tool_calling_agent.py --config configs/tool_calling_agent.py
```

Override model/workdir:

```bash
python examples/run_tool_calling_agent.py \
  --config configs/tool_calling_agent.py \
  --cfg-options model_name=openrouter/gpt-4o workdir=workdir/demo tag=demo
```
