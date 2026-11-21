# ovat/agent/session.py
"""In-memory and serializable conversation history (``Session``).

Stores an ordered list of OpenAI-compatible chat messages (``system``,
``user``, ``assistant``, ``tool``) so stateless inference requests receive the
full prior context, and supports thread-safe JSON persistence.
"""
import json
import threading


class Session:
    """Holds the full ordered message history for one conversation."""

    def __init__(self, system_prompt: str | None = None):
        # Ordered message sequence passed verbatim on each model turn.
        self.messages: list[dict] = []
        # Synchronizes appends during streaming against concurrent UI saves,
        # loads, or clears so snapshots stay consistent.
        self._lock = threading.Lock()
        # Prepend optional system prompt at index 0.
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user(self, content: str) -> None:
        """Append a user message to the history."""
        with self._lock:
            self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None = None,
                      tool_calls: list[dict] | None = None) -> None:
        """Record an assistant reply and its tool calls, if any.

        Attaches ``tool_calls`` only when non-empty so text replies stay compact.
        """
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        with self._lock:
            self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool execution output correlated by ``tool_call_id``."""
        with self._lock:
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            })

    def snapshot(self) -> list[dict]:
        """A copy of the history, taken while no one is appending to it."""
        with self._lock:
            return list(self.messages)

    def truncate(self, length: int) -> None:
        """Rewind conversation history from index ``length`` onward.

        ``length`` must sit on a user message boundary (or ``len(messages)``)
        so an assistant ``tool_calls`` turn is never severed from its tool results.
        """
        with self._lock:
            if not 0 <= length <= len(self.messages):
                raise ValueError(f"No message {length} to rewind to")
            if (length < len(self.messages)
                    and self.messages[length].get("role") != "user"):
                raise ValueError(f"Message {length} is not a user message")
            del self.messages[length:]

    def save(self, path: str) -> None:
        """Serialize message history atomically to a JSON file."""
        # Snapshot under the lock and perform JSON encoding/I/O outside so
        # streaming appends are not blocked.
        from pathlib import Path
        from ovat.agentic.atomic import write_text
        snapshot = self.snapshot()
        write_text(Path(path), json.dumps(snapshot, indent=2))

    @classmethod
    def load(cls, path: str) -> "Session":
        """Restore a ``Session`` from a previously saved JSON file."""
        session = cls()
        with open(path, "r", encoding="utf-8") as f:
            session.messages = json.load(f)
        return session
