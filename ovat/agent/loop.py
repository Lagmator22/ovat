# ovat/agent/loop.py
"""Layer 3: the agent loop, the part that gives the LLM hands.

Note to myself: the model on its own can only write text. It cannot run my
Python functions. The loop is the cycle that turns its written request
(please call search_docs) into a real function call, feeds the result back,
and asks again, until the model is happy and answers in plain words.

The four beats every turn:
  1. ASK    I send the whole history plus my tool menu to the model.
  2. READ   I look at whether the reply CARRIES tool calls.
  3. ACT    If it wants tools, I run those Python functions myself.
  4. REPORT I append each result as a tool message, then loop back to 1.

Exit when a reply carries no tool calls. Deliberately not "when finish_reason
is stop": OVMS documents that NPU serving always reports "stop" even when it
has decoded a tool call, so the label is a description and the payload is the
fact. A max_iterations guard makes sure I can never spin forever.
"""
import json
import time

from ovat.agent.session import Session
from ovat.text import (looks_like_undecoded_tool_call, says_nothing,
                       strip_code_fence)
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
    # Models sometimes wrap the JSON in a markdown fence; see strip_code_fence
    # for why that is worth unwrapping rather than rejecting.
    arguments = strip_code_fence(arguments)
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
                 system_prompt: str | None = None, max_iterations: int = 10,
                 engine_name: str = "native"):
        # llm is any provider that honors the chat() contract (OVMS or a fake).
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.engine_name = engine_name
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

    @staticmethod
    def _sources_in(result) -> list[str]:
        """Source paths carried by a tool result, in order, deduplicated.

        Retrieval tools return [{"text", "source", "distance"}, ...]. The loop
        stringifies that for the model, so the paths were lost to everything
        downstream -- and the CITATION then depended on the model choosing to
        repeat them, which on live hardware it did most of the time but not
        always. Capturing them here means OVAT can state the sources itself.
        """
        if not isinstance(result, list):
            return []
        found = []
        for item in result:
            if isinstance(item, dict):
                source = item.get("source")
                if source and source not in found:
                    found.append(source)
        return found

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
            self._last_sources = []
            return f"Error: tool '{name}' raised {type(exc).__name__}: {exc}"
        self._last_sources = self._sources_in(result)
        return str(result)

    def run(self, user_message: str) -> str:
        """I run the full loop for one user message and return the final text."""
        self.session.add_user(user_message)

        # The run trace (Layer 7). Built as the loop works; every exit path
        # goes through _finish so the totals are always filled in.
        run_started = time.monotonic()
        turns: list[dict] = []
        # A one-element list so the nested _finish closure can set it without
        # a `nonlocal` declaration, matching how `turns` is shared.
        undecoded = [False]
        empty = [False]
        self.last_trace = {"engine": self.engine_name, "turns": turns, "totals": {}}

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
                # True when a tool call came back as text and was never run.
                # Without this the trace shows tool_calls: 0, which reads as
                # "the model chose not to use a tool" rather than "the request
                # could not be decoded".
                "undecoded_tool_call": undecoded[0],
                # True when the reply carried no answer once reasoning was
                # removed. Distinct from undecoded_tool_call: there the markup
                # survives, here the parser ate it.
                "empty_answer": empty[0],
                # True when any turn was CUT at model.max_tokens rather than
                # ending on its own. It travels beside undecoded_tool_call
                # because the two arrive together and mean different things:
                # the markup is a fragment because the reply was truncated,
                # not because the parser is wrong. Reading a trace without
                # this, the only visible symptom is the parser one.
                "truncated": any(t["finish_reason"] == "length"
                                 for t in turns),
                # The run did not produce a usable answer. Every "Error: ..."
                # string this loop returns is delivered AS the answer, because
                # the model has to be able to read it -- but `ovat run` then
                # printed it and exited 0, so a failed run was indistinguishable
                # from a good one to any script, CI job or benchmark. Measured
                # on an AI PC: four consecutive failures, all exit 0.
                "failed": bool(undecoded[0] or empty[0]
                               or answer.startswith("Error: ")),
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

            # BEAT 2, READ. The DECIDER is the payload, not the label.
            #
            # A server can decode a tool call perfectly and still report
            # finish_reason "stop". OVMS documents exactly that for NPU
            # serving -- "finish reason is always set to stop" -- in the same
            # demo whose own pull command passes --tool_parser hermes3. So the
            # call IS decoded and IS sitting in message.tool_calls while the
            # label says the model stopped talking.
            #
            # Keying the decision on the label meant that call was read as
            # prose and answered around: no tool ran, and it looked identical
            # to the wrong-parser failure handled below while having the
            # opposite cause. finish_reason is still recorded in the trace,
            # and still used -- but only for the malformed case just below.
            tool_calls = reply.get("tool_calls") or []

            # The label says tools, the payload carries none. Re-asking with
            # unchanged history would spin the same reply until
            # max_iterations, so surface it immediately instead.
            if not tool_calls and reply["finish_reason"] == "tool_calls":
                self.session.add_assistant(content=reply["content"])
                return _finish(reply["content"] or (
                    "Error: the model reported tool_calls but sent none."
                ))

            # Nothing was asked for, so the model answered in words: record it
            # and I am done. `or ""` because a server can send content=None;
            # the CLI would print literal "None".
            if not tool_calls:
                self.session.add_assistant(content=reply["content"])
                content = reply["content"] or ""
                # A tool call the server could not decode arrives HERE, as
                # prose, and handing it back as the answer is the worst
                # available outcome: it reads as a fluent reply while no tool
                # ran. Seen live with malformed qwen3coder markup the parser
                # correctly refused. Report it as the failure it is, and mark
                # the trace so `--trace` does not show a clean tool_calls: 0.
                #
                # The markup is deliberately NOT parsed into a real call.
                # Guessing at broken output trades a loud failure for a silent
                # wrong answer.
                if looks_like_undecoded_tool_call(content):
                    undecoded[0] = True
                    # A reply that hit the max_tokens ceiling was CUT, and a
                    # cut tool call is a fragment for exactly that reason.
                    # Blaming the parser or the KV cache here sends the reader
                    # to check two things that are both fine -- measured: a
                    # runaway generation stopped dead on completion_tokens ==
                    # max_tokens and reported "usually a tool_parser that does
                    # not match", with the parser correct and the cache at 15%.
                    # finish_reason "length" is the server saying so outright.
                    if reply["finish_reason"] == "length":
                        return _finish(
                            "Error: the reply hit the model.max_tokens "
                            "ceiling mid-tool-call, so the markup was cut off "
                            "and no tool ran. This is NOT a tool_parser or a "
                            "KV cache problem.\n"
                            "A model that generates this much before asking "
                            "for a tool is usually stuck repeating itself; "
                            "greedy decoding (temperature: 0.0) makes that "
                            "likelier. Raising model.max_tokens lets it run "
                            "longer, but the run that needs raising is "
                            "normally the run that has gone wrong.")
                    return _finish(
                        "Error: the model asked for a tool but the server "
                        "could not decode the request, so no tool ran.\n"
                        "Three causes, commonest first:\n"
                        "  1. the MODEL emitted malformed markup. Measured on "
                        "an AI PC with Qwen3.5-4B and the correct "
                        "qwen3coder parser, on a healthy server: it wrote "
                        "<parameter=city> and then closed </function> without "
                        "ever closing </parameter>. The block was complete "
                        "and finish_reason was a clean \"stop\" -- nothing was "
                        "cut off, the markup was simply wrong, and the parser "
                        "was right to refuse it. Nothing on this machine can "
                        "fix that; retry, or use a model that emits the "
                        "format cleanly.\n"
                        "  2. the wrong tool_parser for this model family "
                        "(Qwen3.5 needs qwen3coder, Qwen3 needs hermes3). "
                        "Check it once; if it is right, it is not this.\n"
                        "  3. a full STATIC KV cache. OVMS terminates a "
                        "request when no more cache can be assigned to it, "
                        "'even before reaching stopping criteria', which cuts "
                        "the markup mid-block. This applies only when "
                        "model.ovms_cache_size_gb is set: an unset cache is "
                        "dynamic, grows, and reads near 100% as its normal "
                        "state. Check `ovat telemetry` for the cache TYPE "
                        "before believing this one.\n"
                        "The raw reply is in the session history and the "
                        "trace; it is deliberately not repeated here, because "
                        "printing it is what made this look like an answer in "
                        "the first place.")
                # Third shape of the same failure, and the quietest. A parser
                # that does not match the model's format can SWALLOW the call
                # instead of passing it through: measured live with hermes3
                # forced onto Qwen3.5, the reply was reasoning, then </think>,
                # then nothing. Zero tool calls, no markup for the check above
                # to find, finish_reason a clean "stop" -- and bench scored the
                # row ok. An empty answer is the only evidence left, so it is
                # treated as evidence.
                if says_nothing(content):
                    empty[0] = True
                    return _finish(
                        "Error: the model returned no answer. Its reply "
                        "contained only reasoning, or nothing at all. This is "
                        "usually a tool_parser that does not match the model "
                        "(Qwen3.5 needs qwen3coder, Qwen3 needs hermes3), "
                        "which can swallow the reply rather than pass it on.")
                return _finish(content)

            # The model wants one or more tools.
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
                    # So a caller can cite sources without relying on the
                    # model to have repeated them in its prose.
                    "sources": getattr(self, "_last_sources", []),
                })
                self.session.add_tool_result(call.id, result)

        # If I fall out of the loop I hit my safety cap without a final answer.
        return _finish(f"Error: I reached my max of {self.max_iterations} "
                       f"steps without a final answer.")
