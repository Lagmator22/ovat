# ovat/tui/server/shell_runs.py
"""`!command`: the person's own shell command, shown as a tool-call entry.

Note to myself: the copied TUI sends `session/shellCommand` with an
operation id and then reads history events for an entry that has that SAME
id: the entry is added running, its output arrives as appends to
`/state/outputText`, and it ends completed, failed or cancelled. The request
itself is answered when the command is over (a second request with
action "interrupt" stops it).

AgentSession.run_shell does the running: one command at a time, the lines as
ShellLine, the end as ShellDone. This file turns those into the entry. Lines
are batched (a command that prints a lot would otherwise send one
notification per line).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from ovat.agentic import session as s
from ovat.tui.server.projector import now_ms
from ovat.tui.server.rpc import RpcError
from ovat.tui.vibe.app_server._effect_models import (MANUAL_SHELL_TOOL_NAME,
                                                     ShellEffectDetail,
                                                     ShellEffectInput,
                                                     ShellEffectOutput)
from ovat.tui.vibe.app_server.models import (CancelledEffectState,
                                             CompletedEffectState,
                                             FailedEffectState,
                                             JsonPatchOperation,
                                             PublicEffectEntry,
                                             PublicEntryGenerationStatus,
                                             PublicError, RunningEffectState)
from ovat.tui.vibe.app_server.protocol import (ProtocolErrorCode,
                                               SessionShellCommandParams,
                                               SessionShellCommandResponse)
from ovat.tui.vibe.utils.tool_presentation import (EffectCallDisplay,
                                                   EffectResultDisplay)

FLUSH_SECONDS = 0.05


@dataclass
class _Op:
    id: str
    command: str
    done: asyncio.Event = field(default_factory=asyncio.Event)
    lines: list[str] = field(default_factory=list)      # every line, "\n"-ended
    sent: int = 0                                       # how many are out
    flush_scheduled: bool = False
    interrupted: bool = False
    started: float = field(default_factory=time.monotonic)


class ShellRuns:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self.active: _Op | None = None

    async def run(self, params: SessionShellCommandParams
                  ) -> SessionShellCommandResponse:
        ctx = self._ctx
        projector = ctx.require_session(params.session_id)
        if params.action == "interrupt":
            self.interrupt(params.operation_id or "")
            return SessionShellCommandResponse(last_event_id=projector.event_id)
        if self.active is not None:
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "A command is already running")
        command = (params.command or "").strip()
        op = _Op(id=params.operation_id or str(uuid4()), command=command)
        stamp = now_ms()
        projector.add_entry(PublicEffectEntry(
            id=op.id, session_id=projector.session.id, created_at=stamp,
            updated_at=stamp,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            title=MANUAL_SHELL_TOOL_NAME,
            detail=ShellEffectDetail(
                tool_name=MANUAL_SHELL_TOOL_NAME,
                input=ShellEffectInput(command=command),
                display=EffectCallDisplay(
                    summary=f"shell: {command}", verb="Running",
                    message=command, settled_verb="Ran",
                    settled_message=command, status_text="Running command")),
            state=RunningEffectState()))
        self.active = op
        if ctx.agent.run_shell(command) is None:
            self.active = None
            projector.complete_effect(op.id, FailedEffectState(
                error=PublicError(message="A command is already running"),
                display=EffectResultDisplay(
                    success=False, message="A command is already running")))
            raise RpcError(ProtocolErrorCode.CONFLICT,
                           "A command is already running")
        try:
            await op.done.wait()
        except asyncio.CancelledError:
            # The server is closing: end the command with it.
            op.interrupted = True
            cancel_shell = getattr(ctx.agent, "cancel_shell", None)
            if callable(cancel_shell):
                cancel_shell()
            else:
                ctx.agent.cancel()
            raise
        return SessionShellCommandResponse(last_event_id=projector.event_id)

    def interrupt(self, operation_id: str) -> None:
        op = self.active
        if op is None or op.id != operation_id:
            return
        op.interrupted = True
        cancel_shell = getattr(self._ctx.agent, "cancel_shell", None)
        if callable(cancel_shell):
            cancel_shell()
        else:
            self._ctx.agent.cancel()

    # ---- the session reports ---------------------------------------------------

    def on_line(self, event: s.ShellLine) -> None:
        op = self.active
        if op is None:
            return
        op.lines.append(event.text + "\n")
        if not op.flush_scheduled:
            op.flush_scheduled = True
            self._ctx.loop.call_later(FLUSH_SECONDS, self._flush, op)

    def _flush(self, op: _Op) -> None:
        op.flush_scheduled = False
        if op.sent == len(op.lines) or self._ctx.projector is None:
            return
        text, op.sent = "".join(op.lines[op.sent:]), len(op.lines)
        self._ctx.projector.patch_entry(op.id, [JsonPatchOperation(
            op="append", path="/state/outputText", value=text)])

    def on_done(self, event: s.ShellDone) -> None:
        op = self.active
        if op is None:
            return
        self._flush(op)
        output = "".join(op.lines)
        duration_ms = event.elapsed * 1000
        code = event.code
        if op.interrupted or code is None or code < 0:
            state = CancelledEffectState(
                reason="Command interrupted", output_text=output,
                duration_ms=duration_ms,
                display=EffectResultDisplay(success=False,
                                            message="Command interrupted"))
        elif code != 0:
            message = f"Command exited with status {code}"
            state = FailedEffectState(
                error=PublicError(message=message), output_text=output,
                duration_ms=duration_ms,
                display=EffectResultDisplay(success=False, message=message))
        else:
            state = CompletedEffectState(
                output=ShellEffectOutput(stdout=output, stderr="",
                                         output=output).model_dump(
                    mode="json", by_alias=True),
                output_text=output, duration_ms=duration_ms,
                display=EffectResultDisplay(
                    success=True, message=f"Ran {op.command}"))
        self._ctx.projector.complete_effect(op.id, state)
        self.active = None
        op.done.set()

    def close(self) -> None:
        op = self.active
        if op is not None:
            op.done.set()
