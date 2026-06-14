from __future__ import annotations
from typing import Any, Optional, Dict, List, Type
import json
import os
from pydantic import ConfigDict, Field
from firecrawl import AsyncFirecrawlApp
from dotenv import load_dotenv
load_dotenv()

from src.tool.default_tools.search.types import SearchItem, SearchToolArgs
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL

# Default fastCRW cloud base URL. fastCRW is a Firecrawl-compatible web scraper
# (single binary; self-host or cloud), so the official Firecrawl client can be
# pointed at it by overriding the base URL. Set CRW_API_URL to use a self-hosted
# server (e.g. http://localhost:3000).
DEFAULT_CRW_API_URL = "https://fastcrw.com/api"

@TOOL.register_module(force=True)
class CrwSearch(Tool):
    """Tool that queries the fastCRW search engine.

    fastCRW is a Firecrawl-compatible web data engine (single binary; self-host
    or cloud). It exposes the same /v1/search API as Firecrawl, so the official
    Firecrawl client is reused with the fastCRW base URL.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = CrwSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = CrwSearch.from_search_kwargs({"limit": 5})
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "crw_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    api_key: Optional[str] = Field(default=None, description="fastCRW API key")
    api_url: Optional[str] = Field(default=None, description="fastCRW base URL (override for self-host)")

    def __init__(self, **kwargs):
        """Initialize the CrwSearch tool."""
        # Set api_key from environment if not provided
        super().__init__(**kwargs)
        self.api_key = os.getenv("CRW_API_KEY")
        self.api_url = os.getenv("CRW_API_URL", DEFAULT_CRW_API_URL)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> CrwSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)

    async def _search_crw(self,
                          query: str,
                          num_results: int = 10,
                          filter_year: Optional[int] = 2025) -> List[SearchItem]:
        """
        Perform a fastCRW search using the provided parameters.
        Returns a list of SearchItem objects.
        """
        if not self.api_key:
            raise ValueError("CRW_API_KEY environment variable is required")

        results = []

        app = AsyncFirecrawlApp(api_key=self.api_key, api_url=self.api_url)
        search_kwargs = {
            "query": query,
            "limit": num_results,
        }

        # Add date filter if year is valid (1900-2100)
        # Handle None case (when explicitly passed from caller)
        if filter_year is None:
            filter_year = 2025  # Use default if None

        if 1900 <= filter_year <= 2100:
            search_kwargs["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
        else:
            logger.warning(f"Invalid filter_year: {filter_year}. Expected 1900-2100. Ignoring date filter.")

        try:
            response = await app.search(**search_kwargs)
        except Exception as e:
            logger.error(f"fastCRW API call failed: {e}")
            return results

        # Check if response and response.web exist and are not None
        if response is None:
            logger.warning("fastCRW search returned None response")
            return results

        # Log response structure for debugging
        logger.debug(f"fastCRW response type: {type(response)}")
        logger.debug(f"fastCRW response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")

        # Check for different possible response formats
        web_results = None

        # Try to access web results from response object
        if hasattr(response, 'web') and response.web is not None:
            web_results = response.web
        elif hasattr(response, 'data') and response.data is not None:
            # Some API versions might use 'data' instead of 'web'
            web_results = response.data
        elif hasattr(response, 'results') and response.results is not None:
            # Try 'results' attribute
            web_results = response.results
        elif isinstance(response, dict):
            # Response might be a dict
            web_results = response.get('web') or response.get('data') or response.get('results')
        elif isinstance(response, list):
            # Response might be a list directly
            web_results = response
        else:
            # Try to convert response to dict if it's a Pydantic model
            try:
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    web_results = response_dict.get('web') or response_dict.get('data') or response_dict.get('results')
                elif hasattr(response, 'dict'):
                    response_dict = response.dict()
                    web_results = response_dict.get('web') or response_dict.get('data') or response_dict.get('results')
            except Exception:
                pass

        if web_results is None:
            # Log full response structure for debugging
            logger.warning(
                f"fastCRW search response has no accessible results. "
                f"Response type: {type(response)}, Response: {str(response)[:200]}"
            )
            # Try to log all attributes
            if hasattr(response, '__dict__'):
                logger.debug(f"Response attributes: {list(response.__dict__.keys())}")
            elif hasattr(response, '__fields__'):
                logger.debug(f"Response fields: {list(response.__fields__.keys())}")
            return results

        # Safely iterate over web_results
        try:
            for item in web_results:
                if item is None:
                    continue

                # Handle both object and dict formats
                if isinstance(item, dict):
                    title = item.get('title', '') or ""
                    url = item.get('url', '') or ""
                    description = item.get('description', '') or item.get('snippet', '') or ""
                else:
                    title = getattr(item, 'title', None) or ""
                    url = getattr(item, 'url', None) or ""
                    description = getattr(item, 'description', None) or getattr(item, 'snippet', None) or ""

                if url:  # Only add items with valid URLs
                    results.append(SearchItem(
                        title=title,
                        url=url,
                        description=description
                    ))
        except (TypeError, AttributeError) as e:
            logger.error(f"Error iterating over fastCRW search results: {e}, web_results type: {type(web_results)}")
            return results

        return results

    async def __call__(
        self,
        query: str,
        num_results: Optional[int] = 5,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = 2025,
        **kwargs
    ) -> ToolResponse:
        """
        fastCRW search tool.

        Args:
            query (str): The query to search for.
            num_results (Optional[int]): The number of search results to return.
            country (Optional[str]): The country to search in.
            lang (Optional[str]): The language to search in.
            filter_year (int): The year to filter results by. Defaults to 2025.
        """

        try:

            # Perform search
            search_items = await self._search_crw(query, num_results=num_results, filter_year=filter_year)

            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=4)

            message = f"fastCRW search results for query: {query}\n\n{results_json}"

            return ToolResponse(success=True, message=message, extra=ToolExtra(
                data={
                    "query": query,
                    "num_results": len(search_items),
                    "search_items": search_items,
                    "engine": "crw"
                }
            ))

        except Exception as e:
            logger.error(f"Error in fastCRW search: {e}")
            return ToolResponse(success=False, message=f"Error in fastCRW search: {str(e)}")
