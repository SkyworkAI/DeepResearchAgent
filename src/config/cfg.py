import os
from typing import Any, Dict, List, Optional

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(verbose=True)

from src.utils import assemble_project_path


class SearcherToolConfig(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=10,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(
        default="en", description="Language code for search results (e.g., en, zh, fr)"
    )
    country: str = Field(
        default="us", description="Country code for search results (e.g., us, cn, uk)"
    )
    filter_year: int = Field(
        default=None, description="Filter results by year (0 for no filter)"
    )
    num_results: int = Field(
        default=5, description="Number of search results to return"
    )
    fetch_content: bool = Field(
        default=False, description="Whether to fetch content from the search results"
    )
    max_length: int = Field(
        default=50000,
        description="Maximum character length for the content to be fetched",
    )


class DeepResearcherToolConfig(BaseModel):
    model_id: str = Field(default="gpt-4.1", description="Model ID for the LLM to use")
    max_depth: int = Field(default=2, description="Maximum depth for the search")
    max_insights: int = Field(
        default=20, description="Maximum number of insights to extract"
    )
    time_limit_seconds: int = Field(
        default=60, description="Time limit for the search in seconds"
    )
    max_follow_ups: int = Field(
        default=3, description="Maximum number of follow-up questions to ask"
    )


class BrowserToolConfig(BaseModel):
    model_id: str = Field(default="gpt-4.1", description="Model ID for the LLM to use")
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    use_proxy: bool = Field(False, description="Whether to use a proxy")
    proxy: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "server": "xxxx",
            "username": "xxxx",
            "password": "xxxx",
        },
        description="Proxy settings for the browser",
    )
    max_length: int = Field(
        default=50000,
        description="Maximum character length for the content to be fetched",
    )


class DeepAnalyzerToolConfig(BaseModel):
    analyzer_model_ids: List[str] = Field(
        default_factory=lambda: ["gemini-2.5-pro", "o3"],
        description="Model IDs for the LLMs to use",
    )
    summarizer_model_id: str = Field(
        default="gemini-2.5-pro", description="Model ID for the LLM to use"
    )


class AgentConfig(BaseModel):
    model_id: str = Field(default="gpt-4.1", description="Model ID for the LLM to use")
    name: str = Field(default="agent", description="Name of the agent")
    description: str = Field(
        default="A multi-step agent that can perform various tasks.",
        description="Description of the agent",
    )
    max_steps: int = Field(
        default=20, description="Maximum number of steps the agent can take"
    )
    template_path: str = Field(
        default="", description="Path to the template file for the agent"
    )
    tools: List[str] = Field(
        default_factory=lambda: [], description="List of tools the agent can use"
    )
    managed_agents: List[str] = Field(
        default_factory=lambda: [], description="List of agents the agent can manage"
    )


class HierarchicalAgentConfig(BaseModel):
    name: str = Field(default="dra", description="Name of the hierarchical agent")
    use_hierarchical_agent: bool = Field(
        default=True, description="Whether to use hierarchical agent"
    )
    planning_agent_config: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_id="gpt-4.1",
            name="planning_agent",
            description="A planning agent that can plan the steps to complete the task.",
            max_steps=20,
            template_path=assemble_project_path(
                "src/agent/planning_agent/prompts/planning_agent.yaml"
            ),
            tools=["planning"],
            managed_agents=[
                "deep_analyzer_agent",
                "browser_use_agent",
                "deep_researcher_agent",
            ],
        )
    )
    deep_analyzer_agent_config: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_id="gpt-4.1",
            name="deep_analyzer_agent",
            description="A team member that that performs systematic, step-by-step analysis of a given task, optionally leveraging information from external resources such as attached file or uri to provide comprehensive reasoning and answers. For any tasks that require in-depth analysis, particularly those involving attached file or uri, game, chess, computational tasks, or any other complex tasks. Please ask him for the reasoning and the final answer.",
            max_steps=3,
            template_path=assemble_project_path(
                "src/agent/deep_analyzer_agent/prompts/deep_analyzer_agent.yaml"
            ),
            tools=["deep_analyzer", "python_interpreter"],
        )
    )
    browser_use_agent_config: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_id="gpt-4.1",
            name="browser_use_agent",
            description="A team member that can search the most relevant web pages and interact with them to find answers to tasks, specializing in precise information retrieval and accurate page-level interactions. Please ask this member to get the answers from the web when high accuracy and detailed extraction are required.",
            max_steps=5,
            template_path=assemble_project_path(
                "src/agent/browser_use_agent/prompts/browser_use_agent.yaml"
            ),
            tools=["auto_browser_use", "python_interpreter"],
        )
    )
    deep_researcher_agent_config: AgentConfig = Field(
        default_factory=lambda: AgentConfig(
            model_id="gpt-4.1",
            name="deep_researcher_agent",
            description="A team member capable of conducting extensive web searches to complete tasks, primarily focused on retrieving broad and preliminary information for quickly understanding a topic or obtaining rough answers. For tasks that require precise, structured, or interactive page-level information retrieval, please use the `browser_use_agent`.",
            max_steps=3,
            template_path=assemble_project_path(
                "src/agent/deep_researcher_agent/prompts/deep_researcher_agent.yaml"
            ),
            tools=["deep_researcher", "python_interpreter"],
        )
    )


class DatasetConfig(BaseModel):
    name: str = Field(default="2023_all", description="Dataset name")
    path: str = Field(
        default=assemble_project_path("data/GAIA"), description="Path to the dataset"
    )


class Config(BaseModel):
    # General Config
    workdir: str = "workdir"
    tag: str = "agentscope"
    concurrency: int = 4
    log_path: str = "log.txt"
    download_path: str = "downloads_folder"
    use_local_proxy: bool = Field(
        default=True, description="Whether to use local proxy"
    )
    split: str = Field(default="validation", description="Set name")
    save_path: str = Field(
        default="agentscope.jsonl", description="Path to save the answers"
    )

    # Tool Config
    searcher_tool: SearcherToolConfig = Field(default_factory=SearcherToolConfig)
    deep_researcher_tool: DeepResearcherToolConfig = Field(
        default_factory=DeepResearcherToolConfig
    )
    browser_tool: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    deep_analyzer_tool: DeepAnalyzerToolConfig = Field(
        default_factory=DeepAnalyzerToolConfig
    )

    # Agent Config
    agent: HierarchicalAgentConfig = Field(default_factory=HierarchicalAgentConfig)

    # Dataset Config
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    def init_config(self, config_path: "config.toml"):
        with open(config_path, "r") as f:
            config = toml.load(f)

        # General Config
        self.workdir = config["workdir"]
        self.tag = config["tag"]
        self.concurrency = config["concurrency"]
        self.log_path = config["log_path"]
        self.download_path = config["download_path"]
        self.use_local_proxy = config["use_local_proxy"]
        self.split = config["split"]
        self.save_path = config["save_path"]

        # Create Workdir
        self.workdir = assemble_project_path(os.path.join("workdir", self.tag))
        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "log.txt")
        self.download_path = os.path.join(self.workdir, "downloads_folder")
        os.makedirs(self.download_path, exist_ok=True)
        self.save_path = os.path.join(self.workdir, self.save_path)

        # Tool Config
        self.searcher_tool = SearcherToolConfig(**config["searcher_tool"])
        self.deep_researcher_tool = DeepResearcherToolConfig(
            **config["deep_researcher_tool"]
        )
        self.browser_tool = BrowserToolConfig(**config["browser_tool"])
        self.deep_analyzer_tool = DeepAnalyzerToolConfig(**config["deep_analyzer_tool"])

        # Agent Config
        planning_agent_config = AgentConfig(**config["agent"]["planning_agent_config"])
        planning_agent_config.template_path = assemble_project_path(
            config["agent"]["planning_agent_config"]["template_path"]
        )
        deep_analyzer_agent_config = AgentConfig(
            **config["agent"]["deep_analyzer_agent_config"]
        )
        deep_analyzer_agent_config.template_path = assemble_project_path(
            config["agent"]["deep_analyzer_agent_config"]["template_path"]
        )
        browser_use_agent_config = AgentConfig(
            **config["agent"]["browser_use_agent_config"]
        )
        browser_use_agent_config.template_path = assemble_project_path(
            config["agent"]["browser_use_agent_config"]["template_path"]
        )
        deep_researcher_agent_config = AgentConfig(
            **config["agent"]["deep_researcher_agent_config"]
        )
        deep_researcher_agent_config.template_path = assemble_project_path(
            config["agent"]["deep_researcher_agent_config"]["template_path"]
        )
        self.agent = HierarchicalAgentConfig(
            name=config["agent"]["name"],
            use_hierarchical_agent=config["agent"]["use_hierarchical_agent"],
            planning_agent_config=planning_agent_config,
            deep_analyzer_agent_config=deep_analyzer_agent_config,
            browser_use_agent_config=browser_use_agent_config,
            deep_researcher_agent_config=deep_researcher_agent_config,
        )

        # Dataset Config
        self.dataset = DatasetConfig(**config["dataset"])

    def __str__(self):
        return self.model_dump_json(indent=4)


config = Config()
