"""Exercise the example through the real adapter with an HTTP MCP fixture."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


spec = importlib.util.spec_from_file_location(
    "parallel_example", Path(__file__).parents[1] / "examples/parallel_search_mcp.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def requests(monkeypatch):
    observed = []

    def respond(request):
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["User-Agent"] == "DeepResearchAgent/parallel-search-mcp"
        assert "Authorization" not in request.headers
        message = json.loads(request.content)
        observed.append(message)
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [
                {"name": "web_search", "description": "Search",
                 "inputSchema": {"type": "object", "properties": {
                     "objective": {"type": "string"},
                     "search_queries": {"type": "array", "items": {"type": "string"}},
                     "session_id": {"type": "string"}},
                     "required": ["objective", "search_queries"]}},
                {"name": "web_fetch", "description": "Fetch",
                 "inputSchema": {"type": "object", "properties": {
                     "urls": {"type": "array", "items": {"type": "string"}},
                     "objective": {"type": "string"},
                     "session_id": {"type": "string"}}, "required": ["urls"]}},
            ]}
        else:
            assert method == "tools/call"
            if message["params"]["arguments"].get("objective") == "error":
                result = {"isError": True, "content": [{"type": "text", "text": "rate limited"}]}
            else:
                result = {"content": [{"type": "text", "text": "https://example.org: useful excerpt"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    def client(connections, **kwargs):
        # Preserve and verify the advertised connection, replacing only HTTP I/O.
        connection = dict(connections["parallel"], httpx_client_factory=factory)
        return MultiServerMCPClient({"parallel": connection}, **kwargs)

    monkeypatch.setattr(example, "MultiServerMCPClient", client)
    return observed


@pytest.mark.parametrize("argv,action,expected", [
    (["search", "Find asyncio cancellation", "--query", "Python asyncio cancellation",
      "--query", "Python cancelled task"], "web_search",
     {"objective": "Find asyncio cancellation",
      "search_queries": ["Python asyncio cancellation", "Python cancelled task"]}),
    (["fetch", "https://example.org", "--objective", "page contents"], "web_fetch",
     {"urls": ["https://example.org"], "objective": "page contents"}),
])
def test_cli_calls_discovered_tools(argv, action, expected, requests, capsys):
    assert example.main(argv) == 0
    assert "useful excerpt" in capsys.readouterr().out
    call = next(r["params"] for r in requests if r["method"] == "tools/call")
    assert call["name"] == action
    payload = call["arguments"]
    assert len(payload.pop("session_id")) == 36
    assert payload == expected
    assert any(r["method"] == "tools/list" for r in requests)


def test_remote_error_is_failure(requests, capsys):
    assert example.main(["search", "error", "--query", "query"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "Parallel MCP request failed" in output.err
    assert "rate limited" in output.err


@pytest.mark.parametrize("argv", [
    ["search", "objective"],
    ["fetch", *["https://example.org"] * 21],
    ["fetch", "https://example.org", "--objective", "x" * 201],
])
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit) as error:
        example.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.asyncio
async def test_missing_tool_fails(requests):
    with pytest.raises(RuntimeError, match="did not advertise unknown"):
        await example.invoke("unknown", {})
