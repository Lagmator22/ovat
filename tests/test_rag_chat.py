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

    def chat(self, messages, tools=None):
        self.seen = messages
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
