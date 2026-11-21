# ovat/tools/search_docs.py
"""The ``search_docs`` document retrieval tool and standalone FastMCP server.

Exposes semantic chunk retrieval via a plain ``search_docs_impl`` function
(unit-testable without MCP overhead) wrapped in a FastMCP tool server.
Returns an explicit stub result when no retriever is configured.
"""
from fastmcp import FastMCP

from ovat.providers.base import RetrieverProvider

mcp = FastMCP("search_docs")

# Module-level retriever slot populated by configure() for standalone MCP mode.
_retriever: RetrieverProvider | None = None


def configure(retriever: RetrieverProvider) -> None:
    """Bind the active retriever instance for standalone MCP mode."""
    global _retriever
    _retriever = retriever


def search_docs_impl(query: str, top_k: int = 5,
                     retriever: RetrieverProvider | None = None) -> list[dict]:
    """Run document similarity search, returning stub output if no retriever is configured."""
    if retriever is None:
        return [{
            "text": f"[stub] search_docs has no retriever wired yet. Query was: {query}",
            "distance": 0.0,
        }]
    try:
        hits = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        # The error is reported IN the declared shape, not as a bare string.
        # Returning a str here read fine on the native path, because
        # AgentLoop._execute calls str() on whatever a tool hands back. Over
        # MCP it was a crash: FastMCP validates the return against this
        # function's `-> list[dict]` annotation and rejected the string with
        # "is not of type 'array'", so a locked database surfaced as a
        # client-side exception instead of a sentence the model could read
        # and recover from on its next turn.
        return [{
            "text": f"Error retrieving documents: {exc}",
            "source": None,
            "distance": 0.0,
        }]
    # Forward slashes on every OS. The model sees these hits printed as a
    # Python list, which doubles every backslash of a Windows path
    # (examples\\rag\\docs\\x.md), and it copied that into its citation.
    return [{**hit, "source": hit["source"].replace("\\", "/")}
            if isinstance(hit.get("source"), str) else hit for hit in hits]


# Co-located OpenAI-style tool schema imported by the agent factory so tool
# descriptions and implementation signatures never drift apart.
SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the user's indexed local documents and return the "
                       "most relevant text chunks. Use when the user asks about "
                       "the contents of their own files or notes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural language search query"},
                # "default" makes this schema the COMPLETE contract: the
                # LangChain path derives its argument model from here.
                "top_k": {"type": "integer", "description": "max chunks to return",
                          "default": 5},
            },
            "required": ["query"],
        },
    },
}


@mcp.tool
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search the user's indexed local documents and return the most relevant chunks.

    Use when the user asks about the contents of their own files or notes.
    Takes a natural language query and returns up to ``top_k`` text chunks, each
    with a distance where smaller means a closer match.
    """
    return search_docs_impl(query, top_k, _retriever)


def configure_from_config(config_path: str) -> None:
    """Build this process's OWN retriever from a workflow file.

    Why this has to exist: configure() above is called by the factory, which
    runs in the PARENT process. A tool declared `type: mcp_stdio` is launched
    as a SEPARATE Python process, and objects do not cross a process boundary,
    so that call never reached the server -- _retriever stayed None for its
    whole life and every request took the stub branch.

    Measured on the AI PC: the MCP path answered with the 104-character
    "[stub] search_docs has no retriever wired yet" while the builtin path
    retrieved 1932 real characters from the same index in the same session.
    The agent then told the user it could not find anything.

    The child therefore needs the config, not the object. Imported lazily so
    running this module standalone stays cheap and the stub path pulls in no
    OpenVINO.
    """
    from ovat.agent.factory import build_rag
    from ovat.config.workflow import load_workflow

    config = load_workflow(config_path)
    if config.rag is None:
        return          # no rag: section, so stub mode is the correct answer
    configure(build_rag(config))


if __name__ == "__main__":
    # Standalone MCP server mode. A workflow launches this with:
    #
    #   command: ["python", "-m", "ovat.tools.search_docs", "--config", "workflow.yml"]
    #
    # --config is what makes real retrieval possible over the wire; without it
    # this process has no rag: section to build from and answers in the
    # documented stub mode, which is still how the tool runs by itself.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="Workflow YAML whose rag: section this server "
                             "should serve. Omit for stub mode.")
    args = parser.parse_args()
    if args.config:
        configure_from_config(args.config)
    from ovat.tools.mcp_serve import serve
    serve(mcp)
