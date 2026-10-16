# ovat/agent/permissions.py
"""Whether a tool (or a subagent) may run: allow, ask, or deny.

Note to myself: a local tool that reads my own index is one thing; a call
that sends my words to a cloud API or another machine is another. The
second kind should not happen without a person saying yes, at least the
first time. So every tool has a permission:

    allow  run it
    ask    ask the person first; with no person present, do NOT run it
    deny   never run it

"No person present" is the important case. `ovat run`, a bench worker and
an A2A server have no one to ask, so `ask` means no there, and the model
reads a sentence saying why. Only an interactive front end (the agent TUI)
sets an approver on the event bus.

The broker is the interactive half: the agent's worker thread calls
request() and BLOCKS; the UI shows a prompt and calls resolve(). Each
request has an id and is resolved exactly once; cancel_all() (Esc) resolves
everything still open as "deny", so a cancelled turn can never leave a
worker thread waiting forever.
"""
import ipaddress
import itertools
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import urlparse

from ovat.tools.project import ASK_BY_DEFAULT

ALLOW, ASK, DENY = "allow", "ask", "deny"
PERMISSIONS = (ALLOW, ASK, DENY)

# Decisions a person can give.
ONCE = "once"              # this call only
SESSION = "session"        # this tool, for the rest of the session
ALWAYS = "always"          # this tool, in this project, saved for next time
REFUSE = "deny"            # not this time


def is_loopback(url: str) -> bool:
    """True for a URL on this machine (localhost, 127.x, ::1)."""
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def default_permission(kind: str, url: str | None = None,
                       name: str | None = None) -> str:
    """The permission a tool gets when the workflow names none.

    Anything that sends data off this machine asks; everything that stays
    local is allowed, which keeps every existing workflow working as before.
    The one local exception: the coding tools that change files or run
    commands (`name` says which builtin) ask, because a model that is wrong
    about a command costs the person their files.
    """
    if kind == "builtin" and name in ASK_BY_DEFAULT:
        return ASK
    if kind == "api":
        return ASK
    if kind == "a2a":
        return ALLOW if url and is_loopback(url) else ASK
    return ALLOW


def gate(bus, name: str, arguments: dict, permission: str,
         reason: str = "") -> str | None:
    """None to go ahead, or the "Error: ..." sentence the model reads.

    `bus` is the agent's EventBus; its `approver`, when set, is called as
    approver(name, arguments, reason) -> bool and may block.
    """
    if permission == ALLOW:
        return None
    if permission == DENY:
        return (f"Error: {name} is not allowed in this workflow "
                f"(permission: deny). Answer without it.")
    approver = getattr(bus, "approver", None)
    if approver is None:
        return (f"Error: {name} needs a person's approval, and there is no "
                f"one to ask here. Run it from `ovat agent`, or set "
                f"permission: allow for it in the workflow.")
    try:
        allowed = approver(name, arguments, reason)
    except Exception:
        allowed = False
    if allowed:
        return None
    return (f"Error: the user did not allow {name} this time. Do not try "
            f"it again; answer without it, or ask the user.")


@dataclass
class ApprovalRequest:
    id: int
    tool: str
    arguments: dict
    reason: str = ""
    decision: str | None = None
    agent: str = ""
    _done: threading.Event = field(default_factory=threading.Event,
                                   repr=False)


class ApprovalBroker:
    """Asks a person, from the worker thread, through whatever UI is listening.

    `on_request(request)` is called on the WORKER thread when a request is
    opened; the UI must hand it to its own thread and later call
    resolve(request.id, decision) from anywhere.
    """

    def __init__(self, on_request: Callable[[ApprovalRequest], None]
                 | None = None, timeout: float | None = 600.0,
                 always: Iterable[str] = (),
                 on_always: Callable[[str], None] | None = None):
        self.on_request = on_request
        self.timeout = timeout
        self.auto_approve = False            # /yolo: every ask is a yes
        # Tools a person said "always allow" to, in this project: loaded from
        # .ovat/permissions.json by whoever builds an INTERACTIVE session
        # (ovat/agentic/session.py), and saved through `on_always` when one
        # more is added. Only a broker has this, so a run with no person to
        # ask never sees it: gate() still says no there.
        self._always_allowed: set[str] = set(always)
        self.on_always = on_always
        self._session_allowed: set[str] = set()
        self._open: dict[int, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def __call__(self, tool: str, arguments: dict, reason: str = "") -> bool:
        """The approver signature permissions.gate() expects."""
        if (self.auto_approve or tool in self._session_allowed
                or tool in self._always_allowed):
            return True
        if self.on_request is None:
            return False
        from ovat.agent.events import CURRENT_AGENT_NAME
        agent = CURRENT_AGENT_NAME.get("")
        request = ApprovalRequest(next(self._ids), tool, dict(arguments),
                                  reason, agent=agent)
        with self._lock:
            self._open[request.id] = request
        try:
            self.on_request(request)
        except Exception:
            self._finish(request, REFUSE)
        # A person who walked away is a no, not a thread stuck forever.
        if not request._done.wait(self.timeout):
            self._finish(request, REFUSE)
        if request.decision == SESSION:
            self._session_allowed.add(tool)
        elif request.decision == ALWAYS:
            self._always_allowed.add(tool)
            if self.on_always is not None:
                try:
                    self.on_always(tool)
                except Exception:
                    pass        # the owner of on_always tells the person
        return request.decision in (ONCE, SESSION, ALWAYS)

    def resolve(self, request_id: int, decision: str) -> bool:
        """Answer an open request. False if it was already answered."""
        with self._lock:
            request = self._open.get(request_id)
        if request is None:
            return False
        return self._finish(request, decision)

    def cancel_all(self) -> None:
        """Deny everything still open (the turn was cancelled)."""
        with self._lock:
            pending = list(self._open.values())
        for request in pending:
            self._finish(request, REFUSE)

    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return list(self._open.values())

    def forget_session_grants(self) -> None:
        self._session_allowed.clear()

    def _finish(self, request: ApprovalRequest, decision: str) -> bool:
        with self._lock:
            if self._open.pop(request.id, None) is None:
                return False
        request.decision = decision
        request._done.set()
        return True
