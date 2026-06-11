# ovat/agent/loop.py
"""Layer 3: the agent loop, the part that gives the LLM hands.

Note to myself: the model on its own can only write text. It cannot run my
Python functions. The loop is the cycle that turns its written request
(please call search_docs) into a real function call, feeds the result back,
and asks again, until the model is happy and answers in plain words.

The four beats every turn:
  1. ASK    I send the whole history plus my tool menu to the model.
  2. READ   I look at finish_reason in the reply.
  3. ACT    If it wants tools, I run those Python functions myself.
  4. REPORT I append each result as a tool message, then loop back to 1.

Exit when finish_reason is "stop". A max_iterations guard makes sure I can
never spin forever.
"""
import json

from ovat.agent.session import Session
from ovat.providers.base import LLMProvider


def _parse_args(arguments: str) -> dict:
    """The model sends tool arguments as a JSON string. I turn it into a dict.

    Note to myself: a no argument tool can send an empty string, and a small
    model can occasionally send broken JSON. I never want that to crash the
    whole agent, so I fall back to an empty dict.
    """
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}


def _serialize_tool_calls(tool_calls) -> list[dict]:
    """I convert the SDK tool_call objects into plain dicts.

    Note to myself: OVMSLLMProvider hands me OpenAI SDK objects (attribute
    access like call.function.name). But when I store the assistant turn back
    into the history and resend it, OVMS expects plain JSON dicts. If I forget
    this step, the next request is malformed and OVMS rejects it.
    """
    serialized = []
    for call in tool_calls:
        serialized.append({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        })
    return serialized


class AgentLoop:
    """One agent: an LLM, a set of tools, and the loop that connects them."""

    def __init__(self, llm: LLMProvider, tools: dict,
                 system_prompt: str | None = None, max_iterations: int = 10):
        # llm is any provider that honors the chat() contract (OVMS or a fake).
        self.llm = llm
        # tools maps a tool name to a dict with two keys:
        #   "schema"   the OpenAI tool description I send to the model (the menu)
        #   "function" the real Python callable I run when the model picks it
        # I keep both together so the menu and the hands can never drift apart.
        self.tools = tools
        self.max_iterations = max_iterations
        # Each agent owns one conversation memory.
        self.session = Session(system_prompt=system_prompt)

    def _menu(self) -> list[dict] | None:
        """The list of tool schemas I show the model. None if I have no tools."""
        menu = [tool["schema"] for tool in self.tools.values()]
        return menu or None

    def _execute(self, name: str, args: dict) -> str:
        """I run one tool by name and always return a string for the model.

        Note to myself: three things can go wrong and none of them should
        crash the agent:
          1. the model names a tool I do not have,
          2. the tool raises an exception,
          3. the tool returns something that is not a string.
        In every case I hand back a clear string so the model can read it and
        recover on the next turn.
        """
        if name not in self.tools:
            return f"Error: tool '{name}' is not available."
        try:
            result = self.tools[name]["function"](**args)
        except Exception as exc:
            return f"Error: tool '{name}' raised {type(exc).__name__}: {exc}"
        return str(result)

    def run(self, user_message: str) -> str:
        """I run the full loop for one user message and return the final text."""
        self.session.add_user(user_message)

        for _ in range(self.max_iterations):
            # BEAT 1, ASK.
            reply = self.llm.chat(self.session.messages, tools=self._menu())

            # BEAT 2, READ. If the model did not ask for tools, it answered in
            # words, so I record that answer and I am done.
            if reply["finish_reason"] != "tool_calls":
                self.session.add_assistant(content=reply["content"])
                return reply["content"]

            # The model wants one or more tools. I record its request first,
            # serialized into plain dicts, so the history stays valid.
            tool_calls = reply["tool_calls"] or []
            self.session.add_assistant(
                content=reply["content"],
                tool_calls=_serialize_tool_calls(tool_calls),
            )

            # BEAT 3 and 4, ACT and REPORT. A single reply can ask for several
            # tools at once, so I run each one and append each result before I
            # loop back and ask again.
            for call in tool_calls:
                name = call.function.name
                args = _parse_args(call.function.arguments)
                result = self._execute(name, args)
                self.session.add_tool_result(call.id, result)

        # If I fall out of the loop I hit my safety cap without a final answer.
        return (f"Error: I reached my max of {self.max_iterations} steps "
                f"without a final answer.")
