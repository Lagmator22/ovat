# tests/test_rag_chat.py
"""Tests for local RAG chat (retrieve -> answer), with fakes.

Note to myself: no model and no OVMS here. A fake retriever returns canned hits
and a fake LLM records what prompt it was handed, so I can prove the context and
sources are wired correctly in milliseconds.
"""
from ovat.agent.rag_chat import build_context, rag_chat


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, top_k=5):
        return self._hits[:top_k]


class FakeLLM:
    def __init__(self):
        self.seen = None

    def chat(self, messages, tools=None, on_token=None):
        self.seen = messages
        if on_token is not None:                 # stream like the real GenAI plug
            for token in ("AN", "SW", "ER"):
                on_token(token)
        return {"finish_reason": "stop", "content": "ANSWER", "tool_calls": None}


def test_rag_chat_puts_retrieved_text_in_the_prompt_and_returns_sources():
    hits = [
        {"text": "Q3 revenue grew 12 percent", "source": "finance.md", "distance": 0.1},
        {"text": "data-center costs fell", "source": "finance.md", "distance": 0.2},
    ]
    llm = FakeLLM()
    answer, sources = rag_chat(FakeRetriever(hits), llm, "how did revenue do?", top_k=2)

    assert answer == "ANSWER"
    assert sources == ["finance.md"]                 # de-duplicated, order kept
    user_prompt = llm.seen[-1]["content"]
    assert "Q3 revenue grew 12 percent" in user_prompt   # context was injected
    assert "finance.md" in user_prompt                   # with its source label


def test_rag_chat_handles_an_empty_index():
    llm = FakeLLM()
    answer, sources = rag_chat(FakeRetriever([]), llm, "anything")
    assert answer == "ANSWER" and sources == []
    assert "no relevant" in llm.seen[-1]["content"].lower()


def test_build_context_labels_each_source():
    ctx = build_context([{"text": "hello", "source": "a.md"}])
    assert "[source: a.md]" in ctx and "hello" in ctx


def test_rag_chat_streams_tokens_when_asked():
    tokens = []
    answer, _ = rag_chat(FakeRetriever([]), FakeLLM(), "q", on_token=tokens.append)
    assert "".join(tokens) == "ANSWER"           # every token reached the callback
    assert answer == "ANSWER"                    # final text unchanged


def test_rag_chat_threads_history_between_system_and_question():
    llm = FakeLLM()
    history = [{"role": "user", "content": "earlier question"},
               {"role": "assistant", "content": "earlier answer"}]
    rag_chat(FakeRetriever([]), llm, "follow-up", history=history)
    roles = [m["role"] for m in llm.seen]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.seen[1]["content"] == "earlier question"
    assert "follow-up" in llm.seen[-1]["content"]


def test_rag_chat_caps_history_to_the_last_8_turns():
    llm = FakeLLM()
    history = [{"role": "user", "content": f"turn {i}"} for i in range(20)]
    rag_chat(FakeRetriever([]), llm, "q", history=history)
    # system + capped history (8) + the new question
    assert len(llm.seen) == 1 + 8 + 1
    assert llm.seen[1]["content"] == "turn 12"   # only the most recent survive


def test_a_none_answer_becomes_an_empty_string():
    """A server can finish with content=None. loop.py guards this with `or ""`;
    rag_chat returned it straight through, so the literal string "None" landed
    on screen and in saved sessions."""
    from ovat.agent.rag_chat import rag_chat

    class NoneLLM:
        def chat(self, messages, tools=None, on_token=None):
            return {"finish_reason": "stop", "content": None,
                    "tool_calls": None}

    class EmptyRetriever:
        def retrieve(self, query, top_k=5):
            return []

    answer, sources = rag_chat(EmptyRetriever(), NoneLLM(), "hi")
    assert answer == ""
    assert sources == []
