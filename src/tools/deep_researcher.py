import json
import re
import time
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import model_manager
from src.tools.web_searcher import WebSearcherTool, SearchResult
from src.tools import AsyncTool, ToolResult
from src.config import config
from src.logger import logger
from src.registry import register_tool


_DEEP_RESEARCHER_DESCRIPTION = """Performs comprehensive research on a topic through multi-level web searches and content analysis. 
Returns a structured summary of findings with source attribution and relevance ratings."""

# Prompts for LLM interactions
OPTIMIZE_QUERY_PROMPT = """
You are a research assistant helping to optimize a search query for web research.
Your task is to reformulate the given query to be more effective for web searches.
Make it specific, use relevant keywords, and ensure it's clear and concise.

Original query: {query}

Provide only the optimized query text without any explanation or additional formatting.
"""

EXTRACT_INSIGHTS_PROMPT = """
Analyze the following content and extract key insights related to the research query.
For each insight, assess its relevance to the query on a scale of 0.0 to 1.0.

Research query: {query}
Content to analyze:
{content}

Extract up to 3 most important insights from this content. For each insight:
1. Provide the insight content
2. Provide relevance score (0.0-1.0)
"""

GENERATE_FOLLOW_UPS_PROMPT = """
Based on the insights discovered so far, generate follow-up research queries to explore gaps or related areas.
These should help deepen our understanding of the topic.

Original query: {original_query}
Current query: {current_query}
Key insights so far:
{insights}

Generate up to 3 specific follow-up queries that would help address gaps in our current knowledge.
Each query should be concise and focused on a specific aspect of the research topic.
"""

# Constants for insight parsing
DEFAULT_RELEVANCE_SCORE = 1.0
FALLBACK_RELEVANCE_SCORE = 0.7
FALLBACK_CONTENT_LIMIT = 500
# Pattern to detect start of an insight (number., -, *, •) and capture content
INSIGHT_MARKER_PATTERN = re.compile(r"^\s*(?:\d+\.|-|\*|•)\s*(.*)")
# Pattern to detect relevance score, capturing the number (case-insensitive)
RELEVANCE_SCORE_PATTERN = re.compile(r"relevance.*?:.*?(\d\.?\d*)", re.IGNORECASE)

class ResearchInsight(BaseModel):
    """A single insight discovered during research."""

    model_config = ConfigDict(frozen=True)  # Make insights immutable

    content: str = Field(description="The insight content")
    source_url: str = Field(description="URL where this insight was found")
    source_title: Optional[str] = Field(default=None, description="Title of the source")
    relevance_score: float = Field(
        default=1.0, description="Relevance score (0.0-1.0)", ge=0.0, le=1.0
    )

    def __str__(self) -> str:
        """Format insight as string with source attribution."""
        source = self.source_title or self.source_url
        return f"{self.content} [Source: {source}]"

class ResearchContext(BaseModel):
    """Research context for tracking research progress."""
    query: str = Field(description="The original research query")
    insights: List[ResearchInsight] = Field(default_factory=list, description="Key insights discovered")
    follow_up_queries: List[str] = Field(default_factory=list, description="Generated follow-up queries")
    visited_urls: Set[str] = Field(default_factory=set, description="URLs visited during research")
    current_depth: int = Field(default=0, description="Current depth of research exploration", ge=0)
    max_depth: int = Field(default=2, description="Maximum depth of research to reach", ge=1)

class ResearchSummary(BaseModel):
    """Comprehensive summary of deep research results."""

    output: str = Field(default="", description="Formatted research summary")
    query: str = Field(description="The original research query")
    insights: List[ResearchInsight] = Field(default_factory=list, description="Key insights discovered")
    visited_urls: Set[str] = Field(default_factory=set, description="URLs visited during research")
    depth_reached: int = Field(default=0, description="Maximum depth of research reached", ge=0)

    @model_validator(mode="after")
    def populate_output(self) -> "ResearchSummary":
        """Populate the output field after validation."""
        # Group and sort insights by relevance
        grouped_insights = {
            "Key Findings": [i for i in self.insights if i.relevance_score >= 0.8],
            "Additional Information": [
                i for i in self.insights if 0.5 <= i.relevance_score < 0.8
            ],
            "Supplementary Information": [
                i for i in self.insights if i.relevance_score < 0.5
            ],
        }

        sections = [
            f"# Research: {self.query}\n",
            f"**Sources**: {len(self.visited_urls)} | **Depth**: {self.depth_reached + 1}\n",
        ]

        for section_title, insights in grouped_insights.items():
            if insights:
                sections.append(f"## {section_title}")
                for i, insight in enumerate(insights, 1):
                    sections.extend(
                        [
                            insight.content,
                            f"> Source: [{insight.source_title or 'Link'}]({insight.source_url})\n",
                        ]
                    )

        # Assign the formatted string to the 'output' field inherited from ToolResult
        self.output = "\n".join(sections)
        return self


class OptimizedQueryTool(AsyncTool):
    """Tool for generating optimized search queries."""

    name: str = "optimize_query"
    description: str = """Generates an optimized search query based on the original query. This tool reformulates the query to improve search effectiveness."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The original query to optimize.",
            },
            "filter_year": {
                "type": "integer",
                "description": "(optional) Filter results by year (e.g., 2025).",
                "nullable": True,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    output_type = "any"

    async def forward(self, query: str, filter_year: Optional[int] = None):
        """Generate an optimized search query."""
        # Placeholder for actual optimization logic
        # In a real implementation, this would involve LLM interactions
        return query, filter_year

class GenerateFollowUpsTool(AsyncTool):
    """Tool for generating follow-up queries based on insights."""

    name: str = "generate_follow_ups"
    description: str = """Generates follow-up queries based on the insights discovered during research. This tool helps to explore gaps or related areas in the research topic."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "follow_up_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of follow-up queries (max 3) that would help address gaps in current knowledge",
                "maxItems": 3,
            },
        },
        "required": ["follow_up_queries"],
        "additionalProperties": False,
    }
    output_type = "any"

    async def forward(self, follow_up_queries: List[str]) -> List[str]:
        """Generate follow-up queries based on insights."""
        # Placeholder for actual generation logic
        # In a real implementation, this would involve LLM interactions
        return follow_up_queries

class ExtractInsightsTool(AsyncTool):
    """Tool for extracting insights from content."""

    name: str = "extract_insights"
    description: str = """Extracts key insights from content based on relevance to the research query. This tool assesses the relevance of each insight on a scale of 0.0 to 1.0. """
    parameters: dict = {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The insight content",
                        },
                        "relevance_score": {
                            "type": "number",
                            "description": "Relevance score between 0.0 and 1.0",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["content", "relevance_score"],
                },
                "description": "List of key insights extracted from the content",
                "maxItems": 3,
            }
        },
    }
    output_type = "any"
    async def forward(self, insights: any) -> any:
        """Extract insights from content based on relevance to query."""
        # Placeholder for actual extraction logic
        # In a real implementation, this would involve LLM interactions
        return insights

@register_tool("deep_researcher")
class DeepResearcherTool(AsyncTool):
    """Advanced research tool that explores a topic through iterative web searches."""

    name: str = "deep_researcher"
    description: str = _DEEP_RESEARCHER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research query to explore.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    output_type = "any"

    def __init__(self):
        self.deep_researcher_tool_config = config.deep_researcher_tool

        self.max_depth = (
            getattr(self.deep_researcher_tool_config, "max_depth", 2)
            if self.deep_researcher_tool_config
            else 5
        )
        self.max_insights = (
            getattr(self.deep_researcher_tool_config, "max_insights", 20)
            if self.deep_researcher_tool_config
            else 20
        )
        self.time_limit_seconds = (
            getattr(self.deep_researcher_tool_config, "time_limit_seconds", 120)
            if self.deep_researcher_tool_config
            else 120
        )
        self.max_follow_ups = (
            getattr(self.deep_researcher_tool_config, "max_follow_ups", 3)
            if self.deep_researcher_tool_config
            else 3
        )

        self.model = model_manager.registed_models[self.deep_researcher_tool_config.model_id]
        self.web_searcher = WebSearcherTool()
        self.web_searcher.fetch_content = True # Enable content fetching
        super().__init__()

    async def forward(
        self,
        query: str,
    ) -> ToolResult:
        """Execute deep research on the given query."""
        # Normalize parameters
        max_depth = max(1, min(self.max_depth, 5))

        # Initialize research context and set deadline
        context = ResearchContext(query=query, max_depth=max_depth)
        deadline = time.time() + self.time_limit_seconds

        try:
            optimized_query, filter_year = await self._generate_optimized_query(query)
            await self._research_graph(context=context,
                                 query=optimized_query,
                                 filter_year=filter_year,
                                 deadline=deadline
                                 )
        except Exception as e:
            res_str = f"DeepResearchTool failed to complete the research cycle: {str(e)}"
            logger.error(res_str)
            return ToolResult(
                output=None,
                error=res_str,
            )

        # Prepare final summary reference
        reference = ResearchSummary(
            query=query,
            insights=sorted(context.insights, key=lambda x: x.relevance_score, reverse=True)[:self.max_insights],
            visited_urls=context.visited_urls,
            depth_reached=context.current_depth,
        )

        output = await self._summary(query, reference.output)

        result = ToolResult(
            output=output,
            error=None,
        )

        return result

    async def _generate_optimized_query(self, query: str) -> Tuple[str, Optional[int]]:
        """Generate an optimized search query using LLM."""
        try:
            prompt = OPTIMIZE_QUERY_PROMPT.format(query=query)

            messages = [
                {"role": "user", "content": prompt}
            ]
            tools = [
                OptimizedQueryTool()
            ]

            response = await self.model(
                messages = messages,
                tools_to_call_from=tools
            )

            logger.info(f"DeepResearchTool Optimized query - Input tokens: {self.model.last_input_token_count}, Output tokens: {self.model.last_output_token_count}")

            # Extract the query from the tool_call response
            if response and response.tool_calls and len(response.tool_calls) > 0:
                arguments = response.tool_calls[0].function.arguments
                optimized_query = arguments.get("query", "")
                filter_year = arguments.get("filter_year", None)
            else:
                res = f"DeepResearchTool failed to generate optimized query: {response}"
                logger.warning(res)
                return query, None

            if not optimized_query:
                res = f"DeepResearchTool generated an empty optimized query: {optimized_query}"
                logger.warning(res)
                return query, None

            logger.info(f"DeepResearchTool generated optimized query: {optimized_query}")

            return optimized_query, filter_year
        except Exception as e:
            res = f"DeepResearchTool failed to generate optimized query: {str(e)}"
            logger.error(res)
            return query, None

    async def _research_graph(
        self,
        context: ResearchContext,
        query: str,
        filter_year: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> None:
        """Run a complete research cycle (search, analyze, generate follow-ups)."""
        # Check termination conditions
        if time.time() >= deadline or context.current_depth >= context.max_depth:
            return

        logger.info(f"DeepResearchTool Research cycle at depth {context.current_depth + 1} - Query: {query}")

        # 1. Web search
        search_results = await self._search_web(query, filter_year)

        if not search_results:
            return

        # 2. Extract insights
        new_insights = await self._extract_insights(
            context,
            search_results,
            context.query,
            deadline
        )

        if not new_insights:
            return

        # 3. Generate follow-up queries
        follow_up_queries = await self._generate_follow_ups(
            new_insights,
            query,
            context.query
        )
        context.follow_up_queries.extend(follow_up_queries)

        # Update depth and proceed to next level
        context.current_depth += 1

        # 4. Continue research with follow-up queries
        if follow_up_queries and context.current_depth < context.max_depth:
            tasks = []  # Create a list to hold the tasks
            for follow_up in follow_up_queries[:2]:  # Limit branching factor
                if time.time() >= deadline:
                    break

                # Create a coroutine for the recursive research call
                task = await self._research_graph(
                    context=context,
                    query=follow_up,
                    filter_year=filter_year,
                    deadline=deadline,
                )
                tasks.append(task)  # Add the task to the list

    async def _search_web(self,
                    query: str,
                    filter_year: Optional[int] = None) -> List[SearchResult]:
        """Perform web search for the given query."""
        search_response = await self.web_searcher.forward(
            query=query,
            filter_year=filter_year,
        )
        return [] if search_response.error else search_response.results

    async def _extract_insights(
        self,
        context: ResearchContext,
        results: List[SearchResult],
        original_query: str,
        deadline: float,
    ) -> List[ResearchInsight]:
        """Extract insights from search results."""
        all_insights = []

        for rst in results:
            # Skip if URL already visited or time exceeded
            if rst.url in context.visited_urls or time.time() >= deadline:
                continue

            context.visited_urls.add(rst.url)

            # Skip if no content available
            if not rst.raw_content:
                continue

            # Extract insights using LLM
            insights = await self._analyze_content(
                content=rst.raw_content,  # Limit content size
                url=rst.url,
                title=rst.title,
                query=original_query,
            )

            all_insights.extend(insights)
            context.insights.extend(insights)

            # Log discovered insights
            logger.info(f"DeepResearchTool found {len(insights)} insights in {rst.title or rst.url}.")

        return all_insights

    async def _generate_follow_ups(
        self,
        insights: List[ResearchInsight],
        current_query: str,
        original_query: str
    ) -> List[str]:
        """Generate follow-up queries based on insights."""
        if not insights:
            return []

        # Format insights for the prompt
        insights_text = "\n".join([f"- {insight.content}" for insight in insights[:5]])

        # Create prompt for generating follow-up queries
        prompt = GENERATE_FOLLOW_UPS_PROMPT.format(
            original_query=original_query,
            current_query=current_query,
            insights=insights_text,
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        tools = [
            GenerateFollowUpsTool()
        ]

        # Get follow-up queries from LLM using structured output
        response = await self.model(
            messages=messages,
            tools_to_call_from=tools
        )

        logger.info(f"DeepResearchTool Generate follow-ups - Input tokens: {self.model.last_input_token_count}, Output tokens: {self.model.last_output_token_count}")

        # Extract queries from the tool response
        queries = []
        if response and response.tool_calls and len(response.tool_calls) > 0:
            arguments = response.tool_calls[0].function.arguments
            queries = arguments.get("follow_up_queries", [])

        return queries[:min(len(queries), self.max_follow_ups)]

    async def _analyze_content(
        self, content: str, url: str, title: str, query: str
    ) -> List[ResearchInsight]:
        """Extract insights from content based on relevance to query."""
        prompt = EXTRACT_INSIGHTS_PROMPT.format(
            query=query, content=content  # Limit content size
        )

        messages = [
            {"role": "user", "content": prompt}
        ]
        tools = [
            ExtractInsightsTool()
        ]

        response = await self.model(
            messages=messages,
            tools_to_call_from=tools
        )

        logger.info(f"DeepResearchTool Extract insights - Input tokens: {self.model.last_input_token_count}, Output tokens: {self.model.last_output_token_count}")

        insights = []

        # Process structured JSON response
        if response and response.tool_calls and len(response.tool_calls) > 0:
            arguments = response.tool_calls[0].function.arguments
            extracted_insights = arguments.get("insights", [])

            for insight_data in extracted_insights:
                insights.append(
                    ResearchInsight(
                        content=insight_data.get("content", ""),
                        source_url=url,
                        source_title=title,
                        relevance_score=insight_data.get(
                            "relevance_score", FALLBACK_RELEVANCE_SCORE
                        ),
                    )
                )

        # Fallback: if no structured insights found, use fallback approach
        if not insights:
            logger.info(f"Could not parse structured insights from LLM response for {url}. Using fallback.")
            insights.append(
                ResearchInsight(
                    content=f"Failed to extract structured insights from content about {title or url}."[
                        :FALLBACK_CONTENT_LIMIT
                    ],
                    source_url=url,
                    source_title=title,
                    relevance_score=FALLBACK_RELEVANCE_SCORE,
                )
            )

        return insights

    async def _summary(self, query: str, reference_materials: str) -> str:
        model = model_manager.registed_models["Qwen"]

        messages = [
            {"role": "user", "content": query}
        ]
        response = await model(
            messages=messages,
        )
        content = response.content

        output = reference_materials + "\n" + content

        return output
