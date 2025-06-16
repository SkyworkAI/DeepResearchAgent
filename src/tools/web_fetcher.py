from typing import Optional

from crawl4ai import AsyncWebCrawler
from markitdown._base_converter import DocumentConverterResult

from src.logger import logger
from src.tools import AsyncTool
from src.tools.markdown.mdconvert import MarkitdownConverter

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text. """


async def fetch_url(
    url: str, converter: Optional[MarkitdownConverter] = None
) -> Optional[DocumentConverterResult]:
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
            )

            if result:
                markdown = result.markdown

                res = DocumentConverterResult(
                    markdown=markdown,
                    title=f"Fetched content from {url}",
                )
                return res
            else:
                if converter:
                    res = converter.convert(result.html)
                    return res
    except Exception as e:
        logger.error(f"Error fetching URL: {url}, Error: {e}")
        return None


class WebFetcherTool(AsyncTool):
    name = "web_fetcher"
    description = _WEB_FETCHER_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The relative or absolute url of the webpage to visit.",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    output_type = "any"

    def __init__(self):
        super(WebFetcherTool, self).__init__()
        self.converter = MarkitdownConverter(
            use_llm=False,
            model_id="gpt-4.1",
            timeout=30,
        )

    async def forward(self, url: str) -> Optional[DocumentConverterResult]:
        """Fetch content from a given URL."""

        # try to use asyncio to fetch the URL content
        try:
            res = await fetch_url(url, self.converter)
            if not res:
                logger.error(f"Failed to fetch content from {url}")
                res = DocumentConverterResult(
                    markdown=f"Failed to fetch content from {url}",
                    title="Error",
                )
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            res = DocumentConverterResult(
                markdown=f"Failed to fetch content: {e}",
                title="Error",
            )
        return res
