# ovat/tools/ask_user.py
"""The ask_user tool: ask the user one question and return their answer.

Accepts a single question with optional predefined choices (a compact schema
that smaller local models fill reliably) and returns the answer as text.

Interactive responses are supplied by `bus.asker` (`tui/server/questions.py`).
Headless callers (`ovat run`, benchmark workers, and A2A servers) have no
asker attached and immediately receive an informational error instead of
blocking.
"""
from ovat.tools.project import Shown, fail

NO_PERSON = ("Error: there is no person to ask here; decide yourself or say "
             "what you need.")
MAX_CHOICES = 4                  # keep selectable options compact in prompts/UI


def ask_user_impl(question: str, choices: list | None, bus) -> str:
    question = (question or "").strip()
    if not question:
        return fail("ask_user", "the question is empty.")
    options = [str(c).strip() for c in (choices or []) if str(c).strip()]
    options = options[:MAX_CHOICES]
    asker = getattr(bus, "asker", None)
    if asker is None:
        # Not a "failed" card: nothing went wrong in the tool, the run just
        # has nobody to ask. The model reads the sentence.
        return NO_PERSON
    try:
        answer = asker(question, options)
    except Exception:
        answer = None
    shown = {"kind": "ask_user", "question": question, "choices": options}
    if answer is None:
        text = ("The user did not answer the question. Decide yourself, or "
                "say what you need in your reply.")
        return Shown(text, {**shown, "cancelled": True})
    return Shown(f"The user answered: {answer}",
                 {**shown, "answer": answer, "cancelled": False,
                  "is_other": bool(options) and answer not in options})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "Ask the person you are working for ONE question and "
                       "wait for the answer. Use it when you need a "
                       "decision or a fact only they have, not for things "
                       "you can find out yourself. Offer 2 to 4 short "
                       "choices when the answer is a pick.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "the question, in plain words"},
                "choices": {"type": "array",
                            "description": "optional short answers to pick "
                                           "from (2 to 4); the person can "
                                           "always type their own",
                            "items": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
}
