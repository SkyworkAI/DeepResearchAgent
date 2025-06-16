from typing import List, Optional

from pydantic import BaseModel, Field


class SearchItem(BaseModel):
    """Represents a single search result item"""

    title: str = Field(description="The title of the search result")
    url: str = Field(description="The URL of the search result")
    date: Optional[str] = Field(
        default=None, description="The date of the search result"
    )
    position: Optional[int] = Field(
        default=None, description="The position of the search result in the list"
    )
    source: Optional[str] = Field(
        default=None, description="The source of the search result"
    )
    description: Optional[str] = Field(
        default=None, description="A description or snippet of the search result"
    )

    def __str__(self) -> str:
        """String representation of a search result item."""
        return f"{self.title} - {self.url}"


class WebSearchEngine(BaseModel):
    """Base class for web search engines."""

    model_config = {"arbitrary_types_allowed": True}

    async def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Perform a web search and return a list of search items.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            List[SearchItem]: A list of SearchItem objects matching the search query.
        """
        raise NotImplementedError
