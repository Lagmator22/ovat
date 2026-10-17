# ovat/agent/questions.py
"""Asking the PERSON a question in the middle of a turn (the ask_user tool).

Note to myself: this is ApprovalBroker's sibling (permissions.py) with the
same shape on purpose. The agent's worker thread calls the broker and
BLOCKS; the UI shows the question and calls resolve() from its own thread.
The differences: the answer is text, not yes/no, and "nobody answered" is
None, which the tool turns into a sentence the model can act on.

Who may ask is decided by the EventBus: `bus.asker` is a QuestionBroker in
the agent TUI and None everywhere else (`ovat run`, a bench worker, an A2A
server). With None the tool answers AT ONCE; a headless run must never wait
for a person who is not there.

Each request has an id and is resolved exactly once. cancel_all() (Esc)
resolves everything still open as "no answer", so a cancelled turn can never
leave a worker thread waiting forever.
"""
import itertools
import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class QuestionRequest:
    id: int
    question: str
    choices: list[str]
    answer: str | None = None         # None: nobody answered (cancelled)
    _done: threading.Event = field(default_factory=threading.Event,
                                   repr=False)


class QuestionBroker:
    """`on_request(request)` runs on the WORKER thread when a question
    opens; the UI hands it to its own thread and later calls
    resolve(request.id, answer) from anywhere."""

    def __init__(self, on_request: Callable[[QuestionRequest], None]
                 | None = None, timeout: float | None = 1800.0):
        self.on_request = on_request
        self.timeout = timeout
        self._open: dict[int, QuestionRequest] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def __call__(self, question: str, choices: list[str]) -> str | None:
        """The asker signature ask_user expects. May block."""
        if self.on_request is None:
            return None
        request = QuestionRequest(next(self._ids), question, list(choices))
        with self._lock:
            self._open[request.id] = request
        try:
            self.on_request(request)
        except Exception:
            self._finish(request, None)
        # A person who walked away is "no answer", not a thread stuck forever.
        if not request._done.wait(self.timeout):
            self._finish(request, None)
        return request.answer

    def resolve(self, request_id: int, answer: str | None) -> bool:
        """Answer an open question (None = cancelled). False if it was
        already closed."""
        with self._lock:
            request = self._open.get(request_id)
        if request is None:
            return False
        return self._finish(request, answer)

    def cancel_all(self) -> None:
        """No answer for everything still open (the turn was cancelled)."""
        with self._lock:
            pending = list(self._open.values())
        for request in pending:
            self._finish(request, None)

    def pending(self) -> list[QuestionRequest]:
        with self._lock:
            return list(self._open.values())

    def _finish(self, request: QuestionRequest, answer: str | None) -> bool:
        with self._lock:
            if self._open.pop(request.id, None) is None:
                return False
        request.answer = answer
        request._done.set()
        return True
