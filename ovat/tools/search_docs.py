# ovat/tools/search_docs.py
"""Deliverable 3: the search_docs MCP tool.

Note to myself: this exposes my document retrieval as a tool any agent can
call over MCP. MCP is just a standard plug, and FastMCP turns a plain Python
function into a server that advertises that plug. The model reads the function
docstring to decide when to call it, so I write the docstring for the model,
not only for humans.

Build order from my plan: stub first, real retrieval after the plumbing works.
Right now if no retriever is wired in, I return an obvious stub result so I can
prove the agent can reach the tool before I add the heavy vector search.
"""
from fastmcp import FastMCP

from ovat.providers.base import RetrieverProvider

# The MCP server object. Its name is what shows up to any MCP client.
mcp = FastMCP("search_docs")

# I keep the real retriever in a module level slot. It starts empty (stub
# mode) and I fill it with configure() once my vector search is ready.
_retriever: RetrieverProvider | None = None


def configure(retriever: RetrieverProvider) -> None:
    """I call this once at startup to swap the stub for real retrieval."""
    global _retriever
    _retriever = retriever


def search_docs_impl(query: str, top_k: int = 5,
                     retriever: RetrieverProvider | None = None) -> list[dict]:
    """The real logic, kept separate from the MCP wrapper so I can unit test it.

    Note to myself: keeping the logic in a plain function means my tests can
    call it directly with a fake retriever, without spinning up an MCP server.
    """
    if retriever is None:
        # Stub mode. No retriever wired yet, so I echo the query back in an
        # obviously fake result. This proves the agent can reach me.
        return [{
            "text": f"[stub] search_docs has no retriever wired yet. Query was: {query}",
            "distance": 0.0,
        }]
    try:
        return retriever.retrieve(query, top_k=top_k)
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


# The OpenAI-style tool schema my agent loop shows the model. I keep it next to
# the tool itself so the description the model reads and the function that runs
# can never drift apart. The factory imports this to build the agent's menu.
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
    """Search my indexed local documents and return the most relevant chunks.

    Use me when the user asks about the contents of their own files or notes.
    I take a natural language query and return up to top_k text chunks, each
    with a distance where smaller means a closer match.
    """
    return search_docs_impl(query, top_k, _retriever)


if __name__ == "__main__":
    # Note to myself: this runs the tool as a standalone MCP server so an agent
    # can connect to it. My workflow YAML launches me with: python search_docs.py
    mcp.run()
