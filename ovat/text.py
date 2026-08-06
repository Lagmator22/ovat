# ovat/text.py
"""Reading model output: reasoning blocks, and calls that never got decoded.

This lives at the top level, not under cli/ or agent/, because BOTH need it and
neither may import the other. The agent loop has to know whether a reply says
anything; the CLI and the TUI have to know what to display. It started in
chat_screen.py, moved to cli/ui.py when the plain CLI needed it, and is here now
that loop.py does too -- one copy each time, per the single-source-of-truth
rule, because two would drift the moment a new tag spelling appeared.

Pure stdlib. Importing it costs nothing and pulls in no framework.
"""
import re

# Reasoning models narrate before they answer. Qwen3 and the DeepSeek-R1 family
# wrap that narration in <think>…</think>; some exports spell it <thinking>.
# Both spellings, any case, across newlines.
THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>",
                         re.DOTALL | re.IGNORECASE)
THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)

# Tool-call markup that arrived as CONTENT instead of being decoded. Both the
# well-formed spelling and the malformed <parameter= one seen on live hardware,
# where the model wrote <parameter=search_docs> for <function=search_docs>.
UNDECODED_TOOL_CALL = re.compile(
    r"<tool_call>|</tool_call>|<function=|</function>|<parameter=",
    re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop the reasoning, leaving the answer.

    Interesting once, clutter every time after: a single Qwen3 answer can spend
    fifteen lines deciding what the question meant before saying anything. This
    affects DISPLAY only; the session keeps the raw text, so /thinking on can
    bring it back and /save never loses it.

    Three shapes, because models produce all three:

    1. A complete <think>…</think> block.
    2. An UNCLOSED <think>: generation stopped mid-thought (Esc, or the token
       cap ran out). Everything from the opening tag on is reasoning.
    3. A DANGLING </think> with no opening tag. Qwen3's chat template inserts
       the opening tag itself, so the model emits only the terminator -- which
       is why every `ovat run` answer once carried a stray </think>. Everything
       BEFORE it is reasoning, so it goes too.
    """
    cleaned = THINK_BLOCK.sub("", text or "")
    match = THINK_OPEN.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()]
    # Case 3 runs last, so a well-formed block has already been removed and
    # cannot be mistaken for a dangling terminator.
    closing = THINK_CLOSE.search(cleaned)
    if closing:
        cleaned = cleaned[closing.end():]
    return cleaned.strip()


def looks_like_undecoded_tool_call(text: str) -> bool:
    """Did a tool call end up in the ANSWER instead of being executed?

    When OVMS has the wrong tool_parser -- or the model emits a call the right
    parser cannot read -- the markup is passed through as content. The agent
    then answers fluently, having run no tool at all, and nothing says so.

    Deliberately a plain substring check on the markers, not a parse. The input
    is by definition malformed, so anything stricter would miss the cases this
    exists to catch.
    """
    return bool(UNDECODED_TOOL_CALL.search(text or ""))


def says_nothing(text: str) -> bool:
    """Is there no ANSWER here once the reasoning is removed?

    The third shape of the same failure, and the quietest. A parser that does
    not match the model's format can SWALLOW the tool call rather than pass it
    through: measured on live hardware with tool_parser: hermes3 forced onto a
    Qwen3.5 model, the reply was reasoning, then </think>, then nothing at all.
    Zero tool calls, no markup for looks_like_undecoded_tool_call to catch, and
    `bench` scored the row ok.

    So an empty answer is not a boring edge case, it is a symptom. A model that
    genuinely has nothing to say is vanishingly rare; a pipeline that ate the
    reply is not.
    """
    return not strip_thinking(text)
