# ovat/agent/session.py
"""Layer 3: Session, the agent's memory of one conversation.

Note to myself: an LLM is stateless. Every time I call OVMS it forgets the
previous turn completely. So if I want turn 5 to remember turn 1, I have to
resend the whole conversation every single time. This class is just that
growing list of messages, plus save and load so a chat can survive a restart.

The message shapes follow the OpenAI chat format, because OVMS speaks that
format. The four roles I will ever store are: system, user, assistant, tool.
"""
import json


class Session:
    """Holds the full ordered message history for one conversation."""

    def __init__(self, system_prompt: str | None = None):
        # I keep every message in order in this one list. Order matters,
        # the model reads it top to bottom like a script.
        self.messages: list[dict] = []
        # The system prompt is optional. If I pass one, it always goes first.
        if system_prompt is not None:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user(self, content: str) -> None:
        """I call this when the human says something."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None = None,
                      tool_calls: list[dict] | None = None) -> None:
        """I call this to record what the model replied.

        Note to myself: when the model wants a tool, content is usually None
        and tool_calls holds the request. I only attach tool_calls if there
        are any, so a plain text reply stays a clean two key message.
        """
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """I call this after I run a tool, to feed its output back to the model.

        Note to myself: tool_call_id is the join key. It must match the id of
        the exact tool_call the model asked for, otherwise the model cannot
        tell which answer belongs to which question.
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def save(self, path: str) -> None:
        """I dump the whole history to a JSON file so I can resume later."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Session":
        """I rebuild a Session from a JSON file I saved earlier."""
        session = cls()
        with open(path, "r", encoding="utf-8") as f:
            session.messages = json.load(f)
        return session
