# ovat/agent/rag_chat.py
"""Local retrieval-augmented chat: retrieve, then answer with a local LLM.

This is the macOS-friendly path the proposal calls the "GenAI fallback for
macOS dev". OVMS (and therefore tool-calling) is not available on Apple Silicon,
but openvino_genai runs natively, so I can still do real RAG here: embed the
question, pull the closest chunks from the index, put them in the prompt, and
let a local model answer with citations.

It is deliberately NOT the agent loop. There is no tool-calling; I always
retrieve first and then generate once. That is the classic RAG shape, and it is
all a non-tool-calling local model can do. The agentic version (the model
deciding to call search_docs) stays on the OVMS path.

I keep the logic here, free of any model or CLI, so a fake retriever and a fake
LLM can test the whole flow in milliseconds.
"""
from ovat.providers.base import LLMProvider, RetrieverProvider

_DEFAULT_SYSTEM = (
    "You answer the user's question using ONLY the provided context. "
    "Cite the source file(s) you used. If the context does not contain the "
    "answer, say you don't know rather than inventing one."
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
    """What to search the index with: the new question, plus the previous one.

    A follow-up like "And how is that number measured?" has no topic of its
    own; "that number" lives in the turn before. Searched alone, it ranked the
    token-count chunk first and the memory-budget chunk third, and the model
    answered about token counts (measured on the AI PC, Qwen3-8B). Searched
    with the previous question in front, the subject is back in the query.
    Only the ONE previous question, so an old topic cannot keep steering
    retrieval after the conversation has moved on.
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
    messages = [
        {"role": "system", "content": system_prompt or _DEFAULT_SYSTEM},
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
