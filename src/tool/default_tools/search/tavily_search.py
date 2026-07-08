from __future__ import annotations
from typing import Any, Optional, Dict, List
import json
import os
from pydantic import ConfigDict, Field
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
load_dotenv()

from src.tool.default_tools.search.types import SearchItem
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL


@TOOL.register_module(force=True)
class TavilySearch(Tool):
    """Tool that queries the Tavily search engine.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = TavilySearch()

    .. code-block:: python
        # with custom search kwargs
        tool = TavilySearch.from_search_kwargs({"search_depth": "advanced"})
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "tavily_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    api_key: Optional[str] = Field(default=None, description="Tavily API key")

    def __init__(self, **kwargs):
        """Initialize the TavilySearch tool."""
        # Set api_key from environment if not provided
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("TAVILY_API_KEY")

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> TavilySearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_tavily(self,
                             query: str,
                             num_results: int = 10,
                             country: str = "us",
                             search_depth: str = "basic",
                             topic: str = "general",
                             filter_year: Optional[int] = None) -> List[SearchItem]:
        """
        Perform a Tavily search using the provided parameters.
        Returns a list of SearchItem objects.
        """
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")

        results: List[SearchItem] = []

        client = AsyncTavilyClient(api_key=self.api_key)

        # Tavily accepts 1-100 results; clamp to be safe.
        max_results = max(1, min(int(num_results or 10), 100))
        search_kwargs: Dict[str, Any] = {
            "query": query,
            # "basic" keeps cost (1 credit) and latency low; callers can opt into
            # "advanced" when they need deeper extraction.
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
        }

        # NOTE: `country` is intentionally NOT forwarded. Tavily expects a full
        # lowercase country *name* (e.g. "united states") and only honors it for
        # topic="general" — not the ISO codes ("us") this shared search interface
        # uses. Passing the interface value verbatim would send an invalid enum
        # and 400. Treated like `lang`: accepted for interface parity, not sent.

        # Optional single-year publish-date window. Unlike the hardcoded 2025
        # default copied across the sibling providers, we treat None as "no date
        # filter" so current-year results are not silently excluded (the repo's
        # WebSearcherTool passes filter_year=None by default).
        if filter_year is not None:
            if 1900 <= filter_year <= 2100:
                search_kwargs["start_date"] = f"{filter_year}-01-01"
                search_kwargs["end_date"] = f"{filter_year}-12-31"
            else:
                logger.warning(f"Invalid filter_year: {filter_year}. Expected 1900-2100. Ignoring date filter.")

        try:
            response = await client.search(**search_kwargs)
        except Exception as e:
            logger.error(f"Tavily API call failed: {e}")
            return results
        finally:
            # Close the client if the installed SDK version supports it
            # (older tavily-python releases create an httpx client per request
            # and expose no close()).
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass

        if response is None:
            logger.warning("Tavily search returned None response")
            return results

        # Response is a dict: {"results": [...], "answer": ..., "query": ...}
        if isinstance(response, dict):
            web_results = response.get("results") or []
        else:
            web_results = getattr(response, "results", None) or []

        if not web_results:
            logger.warning(
                f"Tavily search response has no results. "
                f"Response type: {type(response)}, Response: {str(response)[:200]}"
            )
            return results

        try:
            for item in web_results:
                if item is None:
                    continue

                if isinstance(item, dict):
                    title = item.get("title", "") or ""
                    url = item.get("url", "") or ""
                    description = item.get("content", "") or ""
                    date = item.get("published_date") or None
                else:
                    title = getattr(item, "title", None) or ""
                    url = getattr(item, "url", None) or ""
                    description = getattr(item, "content", None) or ""
                    date = getattr(item, "published_date", None) or None

                if url:  # Only add items with valid URLs
                    results.append(SearchItem(
                        title=title,
                        url=url,
                        description=description,
                        date=date,
                        source="tavily",
                    ))
        except (TypeError, AttributeError) as e:
            logger.error(f"Error iterating over Tavily search results: {e}, web_results type: {type(web_results)}")
            return results

        return results

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 5,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = None,
        search_depth: Optional[str] = "basic",
        topic: Optional[str] = "general",
        **kwargs
    ) -> ToolResponse:
        """
        Tavily search tool.

        Args:
            query (str): The query to search for.
            num_results (Optional[int]): The number of search results to return.
            country (Optional[str]): The country to geo-target the search in.
            lang (Optional[str]): The language to search in (kept for interface
                compatibility; Tavily filters by country rather than language).
            filter_year (Optional[int]): Restrict results to this publish year.
                Defaults to None (no date restriction).
            search_depth (Optional[str]): "basic" (default, 1 credit) or "advanced".
            topic (Optional[str]): "general" (default), "news", or "finance".
        """

        try:
            # Perform search
            search_items = await self._search_tavily(
                query,
                num_results=num_results,
                country=country,
                search_depth=search_depth or "basic",
                topic=topic or "general",
                filter_year=filter_year,
            )

            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)

            message = f"Tavily search results for query: {query}\n\n{results_json}"

            return ToolResponse(success=True, message=message, extra=ToolExtra(
                data={
                    "query": query,
                    "num_results": len(search_items),
                    "search_items": search_items,
                    "engine": "tavily"
                }
            ))

        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return ToolResponse(success=False, message=f"Error in Tavily search: {str(e)}")
