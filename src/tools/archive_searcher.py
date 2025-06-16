import requests

from src.logger import logger
from src.tools import AsyncTool, ToolResult
from src.tools.web_fetcher import WebFetcherTool


class ArchiveSearcherTool(AsyncTool):
    name: str = "archive_searcher"
    description: str = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."

    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The url you need the archive for.",
            },
            "date": {
                "type": "string",
                "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
            },
        },
        "required": ["url", "date"],
    }
    output_type = "any"

    content_fetcher: WebFetcherTool = WebFetcherTool()

    async def forward(self, url, date) -> ToolResult:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"

        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()

        if (
            "archived_snapshots" in response
            and "closest" in response["archived_snapshots"]
        ):
            closest = response["archived_snapshots"]["closest"]
            logger.info(f"Archive found! {closest}")

        elif (
            "archived_snapshots" in response_notimestamp
            and "closest" in response_notimestamp["archived_snapshots"]
        ):
            closest = response_notimestamp["archived_snapshots"]["closest"]
            logger.info(f"Archive found! {closest}")
        else:
            return ToolResult(
                res=None,
                error=f"Your {url=} was not archived on Wayback Machine, try a different url.",
            )

        target_url = closest["url"]

        res = self.content_fetcher.forward(target_url)

        output = f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n\n"
        output += f"Title: {res.title.strip() if res.title else 'NO Title'} \n\n"
        output += f"Content: {res.markdown.strip() if res.markdown else 'NO Content'}"

        result = ToolResult(
            output=output,
            error=None,
        )
        return result
