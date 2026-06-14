from typing import Any, Dict, List, Optional
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import asyncio

from src.tool.default_tools.web_fetcher import WebFetcherTool
from src.tool.default_tools.search import (
    FirecrawlSearch,
    CrwSearch,
    SearchItem,
    BraveSearch,
    BingSearch,
    GoogleSearch,
    DDGSSearch
)
from src.logger import logger
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.model import model_manager
from src.message.types import HumanMessage, SystemMessage
from src.utils import dedent
from src.registry import TOOL

_WEB_SEARCHER_DESCRIPTION = """Search the web for real-time information about any topic.
This tool performs deep research by:
1. Using multiple search engines in parallel to find relevant websites
2. Fetching and analyzing content from each webpage
3. Summarizing each page based on the query
4. Merging all summaries into a comprehensive report with citations.
If the primary search engine fails, it automatically falls back to alternative engines.


Args:
- query (str): The search query to submit to the search engine.
- num_results (Optional[int]): The number of search results to return.
- lang (Optional[str]): Language code for search results.
- country (Optional[str]): Country code for search results.
- filter_year (Optional[int]): Filter results by year.

Example: {"name": "web_searcher", "args": {"query": "What is the capital of France?", "num_results": 5, "lang": "en", "country": "us", "filter_year": 2025}}.
"""

@TOOL.register_module(force=True)
class WebSearcherTool(Tool):
    """Search the web for information using various search engines."""

    name: str = "web_searcher"
    description: str = _WEB_SEARCHER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    max_length: int = Field(default=4096, description="The maximum length of the search results")
    retry_delay: int = Field(default=10, description="The delay between retries")
    max_retries: int = Field(default=3, description="The maximum number of retries")
    lang: str = Field(default="en", description="The language to use for the search")
    country: str = Field(default="us", description="The country to use for the search")
    num_results: int = Field(default=5, description="The number of search results to return")
    fetch_content: bool = Field(default=True, description="Whether to fetch content from the search results")
    summarize_pages: bool = Field(default=True, description="Whether to summarize each page based on the query")
    merge_summaries: bool = Field(default=True, description="Whether to merge all summaries into a final report")
    model_name: str = Field(default="openrouter/gemini-3-flash-preview", description="The model to use for summarization")
    search_tools: Dict[str, Any] = Field(default=None, description="The search tools to use")
    content_fetcher: WebFetcherTool = Field(default=None, description="The content fetcher to use")

    def __init__(self, model_name: Optional[str] = None, require_grad: bool = False, **kwargs):
        """Search the web for information using various search engines."""
        super().__init__(require_grad=require_grad, **kwargs)
        if model_name is not None:
            self.model_name = model_name
        # Initialize search engines and content fetcher
        self.search_tools = {
            # "firecrawl_search": FirecrawlSearch(),
            # "crw_search": CrwSearch(),
            # "bing_search": BingSearch(),
            "ddgs_search": DDGSSearch(),
        }
        self.content_fetcher = WebFetcherTool()

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 5,
        lang: Optional[str] = "en",
        country: Optional[str] = "us",
        filter_year: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """
        Execute a Web search and return detailed search results with deep research.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (Optional[int]): The number of search results to return.
            lang (Optional[str]): Language code for search results.
            country (Optional[str]): Country code for search results.
            filter_year (Optional[int]): Filter results by year.
        """
        try:
            logger.info(f"| 🔍 Starting deep web search for query: {query}")
            
            search_params = {"lang": lang, "country": country}
            if filter_year is not None:
                search_params["filter_year"] = filter_year

            # Step 1: Try searching with retries when all engines fail
            results = None
            for retry_count in range(self.max_retries + 1):
                results = await self._try_all_engines(query, num_results, search_params)
                if results:
                    break
                
                if retry_count < self.max_retries:
                    logger.warning(f"| ❌ All search tools failed. Waiting {self.retry_delay} seconds before retry {retry_count + 1}/{self.max_retries}...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"| ❌ All search tools failed after {self.max_retries} retries. Giving up.")
                    return ToolResponse(
                        success=False,
                        message=f"Error: All search tools failed to return results after multiple retries.",
                        extra=ToolExtra(
                            data={
                                "query": query,
                                "status": "failed",
                                "results": [],
                                "total_results": 0,
                                "language": lang,
                                "country": country,
                                "search_tools_used": [],
                                "retries": self.max_retries
                            }
                        )
                    )
            
            if not results:
                return ToolResponse(
                    success=False,
                    message="No search results found.",
                    extra=ToolExtra(
                        data={
                            "query": query,
                            "status": "failed",
                            "results": [],
                            "total_results": 0,
                            "language": lang,
                            "country": country
                        }
                    )
                )

            logger.info(f"| ✅ Found {len(results)} search results from multiple engines")

            # Step 2: Fetch content from all pages
            if self.fetch_content:
                logger.info("| 📥 Fetching content from web pages...")
                results = await self._fetch_content_for_results(results)
                logger.info(f"| ✅ Fetched content from {len([r for r in results if r.get('raw_content')])} pages")

            # Step 3: Summarize each page based on the query
            if self.summarize_pages:
                logger.info("| 📝 Summarizing each page based on the query...")
                results = await self._summarize_results(results, query)
                logger.info(f"| ✅ Summarized {len([r for r in results if r.get('summary')])} pages")

            # Step 4: Merge all summaries into a final report
            merged_summary = None
            if self.merge_summaries and any(r.get("summary") for r in results):
                logger.info("| 🔗 Merging all summaries into final report...")
                merged_summary = await self._merge_summaries(results, query)
                logger.info("| ✅ Generated merged summary with citations")

            # Get unique search engines that contributed results
            search_engines_used = list(set(r.get("source", "") for r in results))

            # Format the output message
            if merged_summary:
                output_message = merged_summary
            else:
                # Fallback format
                output_lines = [f"Search results for '{query}':"]
                for i, result in enumerate(results, 1):
                    title = result.get("title", "").strip() or "No title"
                    url = result.get("url", "")
                    output_lines.append(f"\n{i}. {title}")
                    output_lines.append(f"   URL: {url}")
                    if result.get("description", "").strip():
                        output_lines.append(f"   Description: {result['description']}")
                    if result.get("summary"):
                        output_lines.append(f"   Summary: {result['summary']}")
                output_message = "\n".join(output_lines)
                
            message = f"Web search results for query: {query}\n\n{output_message}"

            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    data={
                        "query": query,
                        "num_results": len(results),
                        "search_engines_used": search_engines_used,
                        "merged_summary": merged_summary,
                    }
                )
            )

        except Exception as e:
            logger.error(f"| ❌ Error in web search: {e}")
            return ToolResponse(
                success=False,
                message=f"Error during web search: {e}"
            )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Try all search tools in parallel and merge results."""
        tool_order = self._get_tool_order()
        
        # Create tasks for all search engines to run in parallel
        async def search_with_tool(tool_name: str):
            """Search with a single tool and return results or None on failure."""
            tool = self.search_tools[tool_name]
            logger.info(f"| 🔎 Attempting search with {tool_name.capitalize()}...")
            
            try:
                search_items = await self._perform_search_with_tool(
                    tool, query, num_results, search_params
                )

                if not search_items:
                    logger.warning(f"| ❌ Search tool {tool_name} returned no results")
                    return None

                # Transform search items into dictionaries
                results = [
                    {
                        "position": i + 1,
                        "url": item.url,
                        "title": item.title or f"Result {i+1}",
                        "description": item.description or "",
                        "source": tool_name,
                        "raw_content": None,
                        "summary": None,
                    }
                    for i, item in enumerate(search_items)
                ]
                logger.info(f"| ✅ {tool_name.capitalize()} returned {len(results)} results")
                return results
            except (ValueError, RetryError, Exception) as e:
                error_msg = str(e)
                if isinstance(e, RetryError):
                    # Extract the last exception from RetryError
                    if e.last_attempt and e.last_attempt.exception():
                        error_msg = str(e.last_attempt.exception())
                logger.warning(f"| ❌ Search tool {tool_name} failed: {error_msg}")
                return None

        # Execute all searches in parallel
        search_tasks = [search_with_tool(tool_name) for tool_name in tool_order]
        results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results: filter out None and exceptions, then merge
        all_results = []
        successful_tools = []
        failed_tools = []
        
        for i, result in enumerate(results_list):
            tool_name = tool_order[i]
            if isinstance(result, Exception):
                failed_tools.append(tool_name)
                logger.error(f"| ❌ Search tool {tool_name} raised exception: {str(result)}")
            elif result is None:
                failed_tools.append(tool_name)
            else:
                successful_tools.append(tool_name)
                all_results.extend(result)
        
        # Log summary
        if successful_tools:
            logger.info(f"| ✅ Successfully retrieved results from: {', '.join(successful_tools)}")
        if failed_tools:
            logger.warning(f"| ⚠️ Failed search tools: {', '.join(failed_tools)}")
        
        if not all_results:
            logger.error(f"| ❌ All search tools failed: {', '.join(failed_tools)}")
            return []
        
        # Deduplicate results by URL (keep first occurrence)
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append(result)
        
        # Reassign positions based on merged order
        for i, result in enumerate(unique_results, 1):
            result["position"] = i
        
        logger.info(f"| 📊 Merged {len(unique_results)} unique results from {len(successful_tools)} search engine(s)")
        return unique_results

    async def _fetch_content_for_results(
            self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fetch and add web content to search results."""
        if not results:
            return []

        # Create tasks for each result with timeout protection
        # Use return_exceptions=True to handle individual failures gracefully
        fetched_results = await asyncio.gather(
            *[self._fetch_single_result_content(result) for result in results],
            return_exceptions=True
        )

        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(fetched_results):
            if isinstance(result, Exception):
                logger.warning(f"| ❌ Exception fetching content for result {i+1}: {result}")
                # Keep original result without content
                if i < len(results):
                    results[i]["raw_content"] = None
                    final_results.append(results[i])
            else:
                final_results.append(result)

        return final_results

    async def _fetch_single_result_content(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch content for a single search result with timeout protection."""
        url = result.get("url")
        if url:
            try:
                # Add timeout protection (20 seconds per URL)
                # This ensures slow websites don't block the entire process
                response = await asyncio.wait_for(
                    self.content_fetcher(url=url),
                    timeout=20.0
                )
                if response.success and response.message:
                    content = response.message
                    if len(content) > self.max_length:
                        content = content[: self.max_length] + "..."
                    result["raw_content"] = content
                else:
                    result["raw_content"] = None
            except asyncio.TimeoutError:
                logger.warning(f"| ❌ Timeout fetching content from {url} (exceeded 20s)")
                result["raw_content"] = None
            except Exception as e:
                logger.warning(f"| ❌ Error fetching content from {url}: {e}")
                result["raw_content"] = None
        return result

    async def _summarize_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Summarize each search result based on the query using LLM."""
        if not results:
            return results

        async def summarize_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
            """Summarize a single search result."""
            raw_content = result.get("raw_content")
            if not raw_content:
                result["summary"] = "No content available to summarize."
                return result

            title = result.get("title", "Untitled")
            url = result.get("url", "")

            prompt = dedent(f"""Given this search query: "{query}"

            And this webpage content:
            Title: {title}
            URL: {url}
            Content: {raw_content[:self.max_length]}

            Provide a concise summary (2-4 sentences) that:
            1. Directly addresses the search query
            2. Highlights the most relevant information from this page
            3. Focuses on factual information and key insights
            4. Is clear and actionable

            Return only the summary, nothing else.""")

            try:
                message = HumanMessage(content=prompt)
                response = await model_manager(model=self.model_name, messages=[message])
                if response and response.message.strip():
                    result["summary"] = response.message.strip()
                else:
                    result["summary"] = "Failed to generate summary."
            except Exception as e:
                logger.warning(f"| ❌ Failed to summarize {url}: {e}")
                result["summary"] = f"Summary unavailable: {str(e)}"

            return result

        # Process all results in parallel
        tasks = [summarize_single_result(result) for result in results]
        summarized_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(summarized_results):
            if isinstance(result, Exception):
                logger.error(f"| ❌ Exception while summarizing result {i}: {result}")
                # Keep original result without summary
                if i < len(results):
                    results[i]["summary"] = f"Summary failed: {str(result)}"
                    final_results.append(results[i])
            else:
                final_results.append(result)

        return final_results

    async def _merge_summaries(self, results: List[Dict[str, Any]], query: str) -> str:
        """Merge all summaries into a comprehensive report with citations."""
        # Filter results that have summaries
        summarized_results = [r for r in results if r.get("summary")]
        
        if not summarized_results:
            return "No summaries available to merge."

        # Build the prompt with all summaries and their citations
        summaries_text = []
        for i, result in enumerate(summarized_results, 1):
            title = result.get("title", "").strip() or "Untitled"
            summary = result.get("summary", "")
            url = result.get("url", "")
            summaries_text.append(f"[{i}] {title} ({url})\nSummary: {summary}\n")

        prompt = dedent(f"""You are creating a comprehensive research report based on multiple web sources.

        Search Query: "{query}"

        Source Summaries:
        {''.join(summaries_text)}

        Please create a well-structured, comprehensive report that:
        1. Directly answers the search query
        2. Synthesizes information from all sources
        3. Organizes information logically
        4. Includes inline citations using [1], [2], etc. format when referencing specific sources
        5. Highlights key findings and insights
        6. Is clear, professional, and easy to read

        Format the report with:
        - A clear introduction that addresses the query
        - Main findings organized by topic or theme
        - Inline citations [1], [2], etc. when referencing specific sources
        - A conclusion that summarizes the key points

        Return only the report content, nothing else.""")

        try:
            system_message = SystemMessage(
                content="You are an expert at synthesizing information from multiple sources into comprehensive research reports."
            )
            user_message = HumanMessage(content=prompt)
            response = await model_manager(
                model=self.model_name,
                messages=[system_message, user_message]
            )

            if response and response.message.strip():
                # Append reference list at the end
                report = response.message.strip()
                report += "\n\n## References:\n"
                for i, result in enumerate(summarized_results, 1):
                    title = result.get("title", "").strip() or "Untitled"
                    url = result.get("url", "")
                    report += f"[{i}] [{title}]({url})\n"
                return report
            else:
                return self._fallback_merge_summaries(summarized_results, query)

        except Exception as e:
            logger.error(f"| ❌ Failed to merge summaries: {e}")
            return self._fallback_merge_summaries(summarized_results, query)

    def _fallback_merge_summaries(self, results: List[Dict[str, Any]], query: str) -> str:
        """Fallback method to merge summaries when LLM fails."""
        report = f"# Research Report: {query}\n\n"
        report += "## Summary\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "").strip() or "Untitled"
            summary = result.get("summary", "")
            report += f"### Source {i}: {title}\n"
            report += f"{summary}\n\n"
        
        report += "## References\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "").strip() or "Untitled"
            url = result.get("url", "")
            report += f"{i}. [{title}]({url})\n"
        
        return report

    def _get_tool_order(self) -> List[str]:
        """Determines the order in which to try search engines."""
        preferred = getattr(self, 'tool', None) or "firecrawl_search"
        fallbacks = list(getattr(self, 'fallback_tools', []))

        # Start with preferred engine, then fallbacks, then remaining engines
        tool_order = [preferred] if preferred in self.search_tools else []
        tool_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self.search_tools and fb not in tool_order
            ]
        )
        tool_order.extend([t for t in self.search_tools if t not in tool_order])

        return tool_order

    async def _perform_search_with_tool(
        self,
        tool: Any,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Execute search with the given tool and parameters."""
        
        @retry(
            stop=stop_after_attempt(3), 
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        async def _do_search():
            result = await tool(
                query=query,
                num_results=num_results,
                lang=search_params.get("lang"),
                country=search_params.get("country"),
                filter_year=search_params.get("filter_year"),
            )
            
            # Handle ToolResponse
            if isinstance(result, ToolResponse):
                if not result.success:
                    raise ValueError(f"Search tool failed: {result.message}")
                if result.extra:
                    search_items = result.extra.data.get("search_items", [])
                    return search_items
                else:
                    raise ValueError(f"Search tool returned invalid response: missing extra")
            else:
                raise ValueError(f"Search tool returned invalid response: result is not a ToolResponse")
        
        try:
            return await _do_search()
        except RetryError as e:
            # Extract the last exception from RetryError
            last_exception = e.last_attempt.exception() if e.last_attempt else None
            if last_exception:
                error_msg = str(last_exception)
            else:
                error_msg = f"Search failed after 3 retries: {str(e)}"
            logger.error(f"| ❌ Search tool {tool.name if hasattr(tool, 'name') else 'unknown'} failed after retries: {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            logger.error(f"| ❌ Unexpected error in search tool {tool.name if hasattr(tool, 'name') else 'unknown'}: {str(e)}")
            raise
