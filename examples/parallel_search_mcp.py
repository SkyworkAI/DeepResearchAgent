"""Opt-in Parallel Search MCP example using the project's MCP adapter."""

import argparse
import asyncio
import json
import sys
from uuid import uuid4

from langchain_mcp_adapters.client import MultiServerMCPClient


# This connection dict also follows MCPImportTool's `connections` format.
CONNECTIONS = {
    "parallel": {
        "transport": "streamable_http",
        "url": "https://search.parallel.ai/mcp",
        "headers": {"User-Agent": "DeepResearchAgent/parallel-search-mcp"},
        "timeout": 30,
        "sse_read_timeout": 60,
    }
}


async def invoke(action, arguments):
    """Discover the remote tools and invoke one with its advertised schema."""
    client = MultiServerMCPClient(CONNECTIONS, handle_tool_errors=False)
    tools = await client.get_tools(server_name="parallel")
    tool = next((tool for tool in tools if tool.name == action), None)
    if tool is None:
        raise RuntimeError(f"Parallel MCP did not advertise {action}")
    return await tool.ainvoke(arguments)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Search the web with excerpts")
    search.add_argument("objective", help="A focused description of what to find")
    search.add_argument("--query", action="append", required=True,
                        help="A keyword query; repeat for related queries")
    fetch = commands.add_parser("fetch", help="Fetch excerpts from known URLs")
    fetch.add_argument("urls", nargs="+", help="Up to 20 HTTP or HTTPS URLs")
    fetch.add_argument("--objective", help="What to find in the pages (max 200 characters)")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if len(args.urls) > 20:
            parser.error("fetch accepts at most 20 URLs")
        if args.objective and len(args.objective) > 200:
            parser.error("fetch objective must be at most 200 characters")
    return args


def main(argv=None):
    args = parse_args(argv)
    payload = {"session_id": str(uuid4())}
    if args.command == "search":
        action = "web_search"
        payload.update(objective=args.objective, search_queries=args.query)
    else:
        action = "web_fetch"
        payload.update(urls=args.urls)
        if args.objective:
            payload["objective"] = args.objective
    try:
        result = asyncio.run(asyncio.wait_for(invoke(action, payload), timeout=120))
    except Exception as error:
        print(f"Parallel MCP request failed: {error!r}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
