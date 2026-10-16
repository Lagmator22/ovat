# ovat/a2a/protocol.py
"""OVAT's own description of the A2A protocol: plain Python, no SDK.

Note to myself: the SDK represents these as protobuf messages whose class
names have already changed once. Everything outside ovat/a2a/sdk/ uses the
shapes here instead, and sdk/ translates at the border. If the SDK renames
a class, one translation function changes; nothing here does.
"""
from dataclasses import dataclass, field
from enum import Enum

# The spec version this code implements (A2A v1.0.x). The SDK adapter puts
# it on the Agent Card; a caller that speaks 0.3 is served through the
# SDK's compatibility layer.
SPEC_VERSION = "1.0"


class TaskState(str, Enum):
    """A task's state, named as the spec names them (section 3.1.1)."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


# A task in one of these will never change again.
TERMINAL = frozenset({TaskState.COMPLETED, TaskState.FAILED,
                      TaskState.CANCELED, TaskState.REJECTED})


@dataclass(frozen=True)
class RemoteSkill:
    name: str
    description: str


@dataclass(frozen=True)
class RemoteCard:
    """What another agent says about itself, reduced to what OVAT uses."""

    name: str
    description: str
    url: str
    version: str = ""
    streaming: bool = False
    skills: tuple[RemoteSkill, ...] = ()
    protocol_versions: tuple[str, ...] = ()


@dataclass
class RemoteReply:
    """How a message to another agent ended."""

    state: TaskState
    text: str
    task_id: str | None = None
    context_id: str | None = None
    # Anything the remote put in `metadata.ovat` (another OVAT's trace).
    metadata: dict = field(default_factory=dict)


class PeerError(Exception):
    """Talking to another agent failed. `kind` says how, in OVAT's words.

    The SDK raises its own exception classes; sdk/client.py maps each one
    to a kind, so callers handle these cases instead of tracking the SDK's
    class names:
      unreachable  nothing answered at that address
      auth         the token was missing or refused
      timeout      no final answer in time
      protocol     it answered, but not in a way OVAT understood
      cancelled    this side stopped waiting (Esc, or one subagent cancelled)
    """

    KINDS = ("unreachable", "auth", "timeout", "protocol", "cancelled")

    def __init__(self, kind: str, detail: str = ""):
        if kind not in self.KINDS:
            raise ValueError(f"unknown PeerError kind {kind!r}")
        super().__init__(detail or kind)
        self.kind = kind
        self.detail = detail
