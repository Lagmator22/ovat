# ovat/tools/mcp_serve.py
"""Start a built-in tool as a standalone MCP server (`python -m ovat.tools.X`)."""
import os

import fastmcp


def serve(mcp) -> None:
    """Run `mcp` with FastMCP's update check switched off.

    FastMCP's start-up banner asks pypi.org whether a newer FastMCP exists.
    A local tool server has no reason to call out, so OVAT turns that off
    unless the person set FASTMCP_CHECK_FOR_UPDATES themselves. The setting
    is changed on the object: FastMCP reads the environment once, at import,
    so setting the variable here would come too late.
    """
    if "FASTMCP_CHECK_FOR_UPDATES" not in os.environ:
        fastmcp.settings.check_for_updates = "off"
    mcp.run()
