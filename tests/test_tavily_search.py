"""Integration test for the Tavily search provider.

Mirrors the style of tests/test_report_tool.py: runnable directly with a
TAVILY_API_KEY set. Skips (rather than fails) when the key is absent, so it
never breaks a keyless environment.

    export TAVILY_API_KEY=tvly-...
    python tests/test_tavily_search.py
"""
import asyncio
import os

from src.tool.default_tools.search import TavilySearch
from src.tool.default_tools.search.types import SearchItem


async def test_tavily_search_returns_mapped_results():
    """Live search returns SearchItem-shaped results mapped from the Tavily API."""
    if not os.getenv("TAVILY_API_KEY"):
        print("SKIP: TAVILY_API_KEY not set")
        return

    tool = TavilySearch()
    assert tool.name == "tavily_search"

    response = await tool(query="what is retrieval augmented generation", num_results=5)
    assert response.success, f"search failed: {response.message}"

    items = response.extra.data["search_items"]
    assert isinstance(items, list) and len(items) > 0, "expected at least one result"
    for item in items:
        assert isinstance(item, SearchItem)
        assert item.url, "each result must have a url"
    assert response.extra.data["engine"] == "tavily"
    print(f"OK: {len(items)} results; first: {items[0].title} - {items[0].url}")


async def test_tavily_search_handles_missing_key():
    """With no API key, the tool fails gracefully (success=False, no exception)."""
    saved = os.environ.pop("TAVILY_API_KEY", None)
    try:
        response = await TavilySearch()(query="anything", num_results=3)
        assert response.success is False
        print("OK: missing-key path returns success=False without raising")
    finally:
        if saved is not None:
            os.environ["TAVILY_API_KEY"] = saved


async def main():
    await test_tavily_search_returns_mapped_results()
    await test_tavily_search_handles_missing_key()


if __name__ == "__main__":
    asyncio.run(main())
