"""MCP tool access for the graph, with a per-node allowlist.

Every tool call names the graph node making it, and is refused unless that
node is allowed that tool. The LLM only ever sees the tools of the node it's
running in (in practice: search_clauses, in policy_lookup), so a prompt
injection in a claim description can't steer it to any other tool. There is
no tool that approves or pays a claim: that decision only enters the graph
from a human, through the approval interrupt.
"""

import os

SERVERS = {
    "policy": os.getenv("POLICY_MCP_URL", "http://policy:8000/mcp"),
    "pricing": os.getenv("PRICING_MCP_URL", "http://pricing:8000/mcp"),
    "claims_history": os.getenv(
        "CLAIMS_HISTORY_MCP_URL", "http://claims-history:8000/mcp"
    ),
}

ALLOWED = {
    "policy_lookup": {
        ("policy", "get_policy"),
        ("policy", "check_coverage"),
        ("policy", "search_clauses"),
    },
    "cost_estimate": {("pricing", "estimate_repair_cost")},
    "fraud_checks": {
        ("claims_history", "get_claim_history"),
        ("claims_history", "find_duplicate_photos"),
    },
}


class ToolNotAllowed(PermissionError):
    pass


class ToolFailed(RuntimeError):
    """The tool ran and reported an error (e.g. unknown policy)."""


def check_allowed(node, server, tool):
    if (server, tool) not in ALLOWED.get(node, set()):
        raise ToolNotAllowed(f"node {node!r} may not call {server}.{tool}")


class McpTools:
    """Calls MCP servers over streamable HTTP. The servers are stateless, so a
    short-lived client per call is fine and keeps no session to go stale."""

    def __init__(self, servers=None):
        self.servers = servers or SERVERS
        self._schemas = {}

    async def call(self, node, server, tool, args):
        from mcp import Client

        check_allowed(node, server, tool)
        async with Client(self.servers[server]) as client:
            result = await client.call_tool(tool, args)
        if result.is_error:
            raise ToolFailed(result.content[0].text if result.content else tool)
        return result.structured_content

    async def schema(self, node, server, tool):
        """The tool's description and input schema, to offer it to the LLM."""
        from mcp import Client

        check_allowed(node, server, tool)
        if (server, tool) not in self._schemas:
            async with Client(self.servers[server]) as client:
                for t in (await client.list_tools()).tools:
                    self._schemas[(server, t.name)] = {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    }
        return self._schemas[(server, tool)]
