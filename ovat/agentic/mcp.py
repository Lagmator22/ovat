"""Session-only MCP selections and public connection pictures."""


_MCP_TYPES = ("mcp_stdio", "mcp_http", "mcp_sse")


class MCPSelection:
    def __init__(self):
        self.disabled = set()
        self.disabled_tools = set()
        self.known_tools = {}

    def config_for_build(self, config):
        return config.model_copy(update={"tools": [
            tool for tool in config.tools
            if tool.type not in _MCP_TYPES or tool.name not in self.disabled]})

    def select_tools(self, tools):
        return {name: spec for name, spec in tools.items()
                if (spec.get("mcp_source"), name) not in self.disabled_tools
                and (spec.get("mcp_source"),
                     spec.get("mcp_tool_name", name)) not in self.disabled_tools}

    def catalog(self, config, agent):
        servers = {getattr(server, "mcp_name", None): server
                   for server in getattr(agent, "mcp_servers", ())}
        sources = []
        for tool in config.tools:
            if tool.type not in _MCP_TYPES:
                continue
            server = servers.get(tool.name)
            status = server.status() if server is not None else {}
            if server is not None:
                self.known_tools[tool.name] = [
                    {"name": t.name, "description": t.description or ""}
                    for t in server.tools]
            sources.append({
                "name": tool.name,
                "status": ("disabled" if tool.name in self.disabled else
                           "connected" if status.get("connected") else "unavailable"),
                "error": status.get("error"),
                "tools": [{**t, "enabled": (
                    tool.name not in self.disabled and
                    (tool.name, t["name"]) not in self.disabled_tools)}
                    for t in self.known_tools.get(tool.name, ())]})
        return sources

    def toggle(self, name, disabled, tool_name=None):
        selected = self.disabled if tool_name is None else self.disabled_tools
        key = name if tool_name is None else (name, tool_name)
        if disabled:
            selected.add(key)
        else:
            selected.discard(key)
