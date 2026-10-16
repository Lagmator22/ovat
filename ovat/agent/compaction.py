# ovat/agent/compaction.py
"""Compacting the model's memory: the summary prompt and the new history.

Note to myself: this mirrors Mistral Vibe's compaction (Apache-2.0):
vibe/core/prompts/compact.md + compact_system.md (the prompts),
vibe/core/compaction/context.py (render_compaction_context,
collect_prior_user_messages) and vibe/core/compaction/manager.py
(_render_transcript, the no-tools summarizer call, CompactionFailedError).
What is different here: no retry that drops the oldest round (OVAT's
failure is one plain sentence and the memory stays as it was), and a
character budget instead of a token count (there is no tokenizer to hand).

The summary is asked for as ONE message holding the whole conversation as
text, with no tools. That is Vibe's own fallback shape, and it also means a
tool call and its result can never be cut apart: both are just text.
"""
import re

from ovat.text import split_thinking

# vibe/core/prompts/compact_system.md
SYSTEM_PROMPT = """\
You are a dedicated conversation summarizer. Your only job is to read a conversation transcript and produce a faithful, structured handoff summary for another LLM that will resume the task.

Rules:
- Respond with plain text only. Never call tools. Never emit tool calls.
- Do not ask questions or request clarification.
- Preserve concrete details: file paths, identifiers, decisions made, and the next concrete step.
- Wrap the entire summary in <summary></summary> tags and output nothing outside them.
"""

# vibe/core/prompts/compact.md
REQUEST = """\
CRITICAL: Respond with text only. Do NOT call any tools. Any tool call will be rejected.

You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume this task.

Include:
- The user's current goal and any explicit constraints or preferences
- Key decisions made and their rationale
- Files touched and the current state of in-progress work (paths + one-line status)
- What remains to be done: the concrete next step
- Any data, identifiers, or references the next LLM needs to continue

Be concise and structured. One line per modified file unless a snippet is load-bearing. Do not repeat information already captured. Do not include a "Final Answer" section: the entire summary IS the handoff.

Wrap the ENTIRE summary in <summary></summary> tags and output nothing outside them:

<summary>
...your handoff summary here...
</summary>
"""

# The assistant line after the summary, so the next real question keeps the
# user / assistant order a chat template expects.
ACKNOWLEDGEMENT = "Understood. I will continue from this summary."

# How much of the person's earlier wording rides along verbatim. Vibe keeps
# 20 000 tokens; a local model has far less room.
PREVIOUS_USER_MESSAGES_MAX_CHARS = 4000

_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
# The envelope's own tags (vibe's reserved tags), escaped if the person's
# text happens to contain them.
_TAGS = ("<previous_user_messages>", "</previous_user_messages>",
         "<previous_user_message>", "</previous_user_message>")
# Start of the envelope message; also how a second compaction recognises the
# first one's envelope and does not nest it inside its own.
ENVELOPE_START = "You are continuing a trajectory after a context compaction."


class CompactionFailed(Exception):
    """The summary did not happen. `str(exc)` is a sentence for the person;
    the memory has not been touched."""


def summary_messages(memory: list[dict], instructions: str) -> list[dict]:
    """What the summarizing model is sent (vibe's fallback call)."""
    request = REQUEST
    if instructions.strip():
        request += f"\n## Additional Instructions\n{instructions.strip()}\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"{request}\n## Conversation Transcript\n"
                    f"{render_transcript(memory)}"},
    ]


def render_transcript(messages: list[dict]) -> str:
    """The conversation as text (vibe manager._render_transcript): each turn
    under its role, tool calls written out as `[calls name(args)]`."""
    blocks = []
    for message in messages:
        if message.get("role") == "system":
            continue
        parts = []
        text = (message.get("content") or "").strip()
        if text:
            parts.append(text)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(f"[calls {function.get('name')}"
                         f"({function.get('arguments')})]")
        if parts:
            blocks.append(f"### {message.get('role')}\n" + "\n".join(parts))
    return "\n\n".join(blocks)


def summary_of(reply: dict) -> str:
    """The summary out of the model's reply, or CompactionFailed.

    Vibe fails when the <summary> tags are missing. A small local model
    often forgets them, and the memory is about to be replaced, so the
    reply's own text is taken then: only an empty one fails."""
    if reply.get("tool_calls"):
        raise CompactionFailed(
            "The model answered with a tool call instead of a summary, so "
            "nothing was compacted. Try /compact again.")
    if reply.get("finish_reason") == "length":
        raise CompactionFailed(
            "The summary hit the model.max_tokens limit and was cut off, so "
            "nothing was compacted. Raise model.max_tokens and try again.")
    _, text = split_thinking(reply.get("content") or "")
    found = _SUMMARY.search(text)
    summary = (found.group(1) if found else text).strip()
    if not summary:
        raise CompactionFailed(
            "The model returned an empty summary, so nothing was compacted. "
            "Try /compact again.")
    return summary


def previous_user_messages(memory: list[dict]) -> list[str]:
    """The person's own messages, newest first into the budget (vibe
    collect_prior_user_messages), oldest first in the result. An earlier
    envelope is skipped: its summary is in the transcript the model read."""
    chosen, room = [], PREVIOUS_USER_MESSAGES_MAX_CHARS
    for message in reversed(memory):
        text = message.get("content") or ""
        if (message.get("role") != "user" or not text
                or text.startswith(ENVELOPE_START)):
            continue
        if room <= 0:
            break
        chosen.append(text[:room])
        room -= len(text)
    return list(reversed(chosen))


def envelope(memory: list[dict], summary: str) -> str:
    """The one user message that stands for everything before it
    (vibe render_compaction_context)."""
    lines = [
        ENVELOPE_START,
        "",
        "Here are some of the most recent previous user messages, preserved "
        "verbatim where possible. Treat them as prior context, not as new "
        "requests.",
        "",
        "<previous_user_messages>",
    ]
    for text in previous_user_messages(memory):
        for tag in _TAGS:
            text = text.replace(tag, tag.replace("<", "&lt;").replace(">", "&gt;"))
        lines.append(f"<previous_user_message>\n{text}\n"
                     f"</previous_user_message>")
    lines += ["</previous_user_messages>", "",
              "Here is a summary of what has happened so far:", "",
              "<compaction_summary>", summary, "</compaction_summary>"]
    return "\n".join(lines)


def compacted_memory(memory: list[dict], summary: str) -> list[dict]:
    """The new memory without the system prompt (the session puts the
    workflow's own first): the envelope as a user message, then the
    acknowledgement."""
    return [{"role": "user", "content": envelope(memory, summary)},
            {"role": "assistant", "content": ACKNOWLEDGEMENT}]
