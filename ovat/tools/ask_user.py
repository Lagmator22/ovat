# ovat/tools/ask_user.py
"""The ask_user tool: put one question to the PERSON and wait for the answer.

Mirrors Mistral Vibe's ask_user_question.py (vibe/core/tools/builtins) and
its questions.py. Vibe takes a list of questions with options; OVAT takes
one question and optional choices, which is what a small local model can
fill in reliably. The reply to the model is the person's answer as text.

Who answers is `bus.asker` (agent/questions.py). Without one (`ovat run`, a
bench worker, an A2A server) the tool says so AT ONCE: a headless run must
never wait for someone who is not there.
"""
from ovat.tools.project import Shown, fail

NO_PERSON = ("Error: there is no person to ask here; decide yourself or say "
             "what you need.")
MAX_CHOICES = 4                  # Vibe recommends 2-4 options


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
