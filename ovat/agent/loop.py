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
import time

from ovat.agent.session import Session
from ovat.providers.base import LLMProvider


def _parse_args(arguments: str) -> dict | None:
    """The model sends tool arguments as a JSON string. I turn it into a dict.

    Note to myself: a no argument tool can send an empty string (fine, empty
    dict), but a small model can also send BROKEN JSON. That used to become {}
    silently, so the tool ran with missing arguments and the model never
    learned why it failed. Now broken JSON returns None and the loop tells the
    model exactly what was wrong, so it can fix its own call next turn.
    """
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None


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
        # Layer 7 (observability): after every run() this holds what happened:
        # per-turn latency, token usage (OVMS reports it on each response),
        # and every tool call with its duration. `ovat run --trace` dumps it.
        self.last_trace: dict = {}

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

        # The run trace (Layer 7). Built as the loop works; every exit path
        # goes through _finish so the totals are always filled in.
        run_started = time.monotonic()
        turns: list[dict] = []
        self.last_trace = {"engine": "native", "turns": turns, "totals": {}}

        def _total(key: str):
            """Sum a per-turn count, or None if NO turn reported one.

            Absent is not zero. A server that never sends a usage block
            would otherwise produce "prompt_tokens": 0, which reads as "this
            run used no tokens" rather than "nobody told us", and a
            benchmark built on that number would be quietly wrong.
            """
            known = [t[key] for t in turns if t[key] is not None]
            return sum(known) if known else None

        def _finish(answer: str) -> str:
            self.last_trace["totals"] = {
                "turns": len(turns),
                "latency_s": round(time.monotonic() - run_started, 3),
                "prompt_tokens": _total("prompt_tokens"),
                "completion_tokens": _total("completion_tokens"),
                "tool_calls": sum(len(t["tool_calls"]) for t in turns),
            }
            return answer

        for _ in range(self.max_iterations):
            # BEAT 1, ASK: timed, and the token usage OVMS reports per
            # response is kept instead of dropped.
            ask_started = time.monotonic()
            reply = self.llm.chat(self.session.messages, tools=self._menu())
            usage = reply.get("usage") or {}
            turn = {
                "latency_s": round(time.monotonic() - ask_started, 3),
                "finish_reason": reply["finish_reason"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "tool_calls": [],
            }
            turns.append(turn)

            # BEAT 2, READ. If the model did not ask for tools, it answered in
            # words, so I record that answer and I am done. `or ""` because a
            # server can send content=None; the CLI would print literal "None".
            if reply["finish_reason"] != "tool_calls":
                self.session.add_assistant(content=reply["content"])
                return _finish(reply["content"] or "")

            # The model wants one or more tools. Guard the malformed case
            # first: finish_reason says tool_calls but the list is empty.
            # Re-asking with unchanged history would just spin the same reply
            # until max_iterations, so surface it immediately instead.
            tool_calls = reply["tool_calls"] or []
            if not tool_calls:
                self.session.add_assistant(content=reply["content"])
                return _finish(reply["content"] or (
                    "Error: the model reported tool_calls but sent none."
                ))

            # I record its request, serialized into plain dicts, so the
            # history stays valid.
            self.session.add_assistant(
                content=reply["content"],
                tool_calls=_serialize_tool_calls(tool_calls),
            )

            # BEAT 3 and 4, ACT and REPORT. A single reply can ask for several
            # tools at once, so I run each one and append each result before I
            # loop back and ask again. Each call is timed for the trace.
            for call in tool_calls:
                name = call.function.name
                args = _parse_args(call.function.arguments)
                tool_started = time.monotonic()
                if args is None:
                    # Broken JSON from the model: report it AS the tool result
                    # so the model reads its own mistake and can retry.
                    result = (f"Error: arguments for tool '{name}' were not "
                              f"valid JSON: {call.function.arguments!r}")
                else:
                    result = self._execute(name, args)
                turn["tool_calls"].append({
                    "name": name,
                    "arguments": args,
                    "duration_s": round(time.monotonic() - tool_started, 3),
                    "result_chars": len(result),
                })
                self.session.add_tool_result(call.id, result)

        # If I fall out of the loop I hit my safety cap without a final answer.
        return _finish(f"Error: I reached my max of {self.max_iterations} "
                       f"steps without a final answer.")
