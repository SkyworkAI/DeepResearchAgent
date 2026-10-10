# Parallel Search MCP example

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search and page excerpts without an account or API key. Anonymous
search uses Fast mode and is rate limited. This opt-in example uses the same
`MultiServerMCPClient` connection format as the project's `MCPImportTool`.
It calls discovered tools directly, without running a model or agent loop.

From the repository root, install in a separate Python 3.10+ environment:

```bash
python -m venv .venv-parallel
.venv-parallel/bin/python -m pip install -r examples/requirements-parallel-mcp.txt
```

Search with a focused objective and one or more keyword queries:

```bash
.venv-parallel/bin/python examples/parallel_search_mcp.py search \
  "Find the Python asyncio documentation on task cancellation" \
  --query "Python asyncio task cancellation"
```

Fetch excerpts from a known page:

```bash
.venv-parallel/bin/python examples/parallel_search_mcp.py fetch \
  https://docs.python.org/3/library/asyncio-task.html \
  --objective "How task cancellation works"
```

The commands print MCP content blocks, including source URLs and excerpts, as JSON.
Each command creates one session identifier for its request. HTTP operations time
out after 30 seconds, stream reads after 60 seconds, and the complete command after
120 seconds. Errors exit with a nonzero status. Requests send a project User-Agent
and no authorization header; the script does not load `.env` or saved credentials.

For an existing MCP setup, the importable `CONNECTIONS` dictionary in
`parallel_search_mcp.py` is the optional Parallel server connection. Existing
provider selections and agent configurations are unchanged. The standalone
example does not require the full repository dependencies.
