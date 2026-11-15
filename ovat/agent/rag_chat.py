# ovat/agent/rag_chat.py
"""Local retrieval-augmented chat: retrieve context, then generate once via local LLM.

Designed for direct ``openvino_genai`` execution where an OVMS tool-calling
server is unavailable (including macOS development environments). Instead of
running a tool-calling loop, each question embeds once, retrieves top chunks,
injects them into the prompt with source labels, and generates a single answer.
Kept free of CLI and model-loading code so unit tests run quickly with fakes.
"""
from ovat.providers.base import LLMProvider, RetrieverProvider

_DEFAULT_SYSTEM = (
    "You answer the user's question using ONLY the provided context. "
    "Cite the source file(s) you used. If the context does not contain the "
    "answer, say you don't know rather than inventing one."
)


#: Reconciles agentic workflow prompts that instruct the model to call
#: search_docs with this tool-less path where retrieval already ran, preventing
#: small/reasoning models from wasting tokens trying to invoke missing tools.
_NO_TOOLS_NOTE = (
    "In this chat the documents have already been searched for you: the "
    "Context in the user's message is the search result. There are no tools "
    "here, so do not try to call search_docs or any other tool. Answer from "
    "the Context and cite its source path."
)


def build_context(hits: list) -> str:
    """Turn retrieved chunks into a context block that names each source."""
    if not hits:
        return "(no relevant documents were found in the index)"
    blocks = []
    for h in hits:
        source = h.get("source") or "unknown"
        blocks.append(f"[source: {source}]\n{h['text']}")
    return "\n\n".join(blocks)


def retrieval_query(question: str, history: list | None) -> str:
    """Build the retrieval query by prepending the most recent user turn.

    Prepending only the single preceding user question keeps follow-ups like
    "How is that measured?" anchored to the active topic without letting stale
    earlier topics dominate retrieval.
    """
    previous = next((m.get("content") for m in reversed(history or [])
                     if m.get("role") == "user" and m.get("content")), None)
    return f"{previous}\n{question}" if previous else question


def rag_chat(retriever: RetrieverProvider, llm: LLMProvider, question: str,
             top_k: int = 4, system_prompt: str | None = None,
             history: list | None = None, on_token=None) -> tuple:
    """Answer `question` from the index. Returns (answer_text, source_list).

    Steps: retrieve the top_k closest chunks, build a context block that labels
    each chunk with its source, ask the local model to answer from that context,
    and return the answer plus the de-duplicated list of sources it was given.

    history: optional prior turns ([{role, content}, ...]) slotted between the
    system prompt and this question, so a chat UI gets real conversation
    memory, and the previous question joins the retrieval query (see
    retrieval_query). Capped to the last 8 messages
    (four question-and-answer turns) so a long chat cannot blow the model's
    context window.
    on_token: optional streaming callback, forwarded to providers that
    support it (GenAILLMProvider); None keeps the old single-shot call.
    """
    hits = retriever.retrieve(retrieval_query(question, history), top_k=top_k)
    context = build_context(hits)
    # A workflow's prompt is kept (it may carry a persona or rules), with the
    # note that reconciles it with this tool-less path; see _NO_TOOLS_NOTE.
    system = (f"{system_prompt}\n\n{_NO_TOOLS_NOTE}" if system_prompt
              else _DEFAULT_SYSTEM)
    messages = [
        {"role": "system", "content": system},
        *(history or [])[-8:],
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    if on_token is not None:
        reply = llm.chat(messages, on_token=on_token)
    else:
        reply = llm.chat(messages)

    sources = []
    for h in hits:
        source = h.get("source")
        if source and source not in sources:    # keep order, drop duplicates
            sources.append(source)
    # `or ""` for the same reason loop.py does it: a server can finish with
    # content=None, and returning that straight through puts the literal
    # string "None" on screen or into a saved session.
    return reply["content"] or "", sources
