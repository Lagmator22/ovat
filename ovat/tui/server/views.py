# ovat/tui/server/views.py
"""The read-only pictures the copied TUI asks for at startup and on refresh.

Note to myself: `runtime/read` wants a config, the agent profiles, the
tools, the stats and a few counters; `config/read` wants the config again.
Vibe's config has about thirty fields (voice, narrator, other vendors' models...)
that mean nothing here. They are filled with plain "off" values in ONE place,
this file, so nobody has to guess later which of them OVAT really has.

What OVAT really has: a model (named by its folder), a device, an engine,
tools with permissions, subagents, and one switch, auto-approve (`/yolo`),
which shows up as the second agent profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_args

from ovat.tui.vibe.agents import AgentSafety, AgentType
from ovat.tui.vibe.app_server.config import (AudioProviderView, ConfigView,
                                             ModelConfigView,
                                             SpeechConfigView,
                                             TranscribeModelConfigView,
                                             TranscriptionConfigView,
                                             TTSModelConfigView)
from ovat.tui.vibe.app_server.models import (AgentStatsSnapshot, AgentSummary,
                                             ConnectorCounts, MCPState,
                                             SessionLogSummary, ToolSummary)
from ovat.tui.vibe.app_server.protocol import (RuntimeSnapshot)
from ovat.tui.vibe.config_values import (AudioClient, SpeechOutputFormat,
                                         TranscriptionEncoding)

DEFAULT_PROFILE = "default"
AUTO_APPROVE_PROFILE = "auto-approve"

PROFILES = (
    AgentSummary(
        name=DEFAULT_PROFILE, display_name="Default",
        description="Asks before a tool that needs permission.",
        safety=AgentSafety.NEUTRAL, agent_type=AgentType.AGENT),
    AgentSummary(
        name=AUTO_APPROVE_PROFILE, display_name="Auto-approve",
        description="Allows every tool without asking (/yolo).",
        safety=AgentSafety.YOLO, agent_type=AgentType.AGENT),
)


def profile(name: str) -> AgentSummary | None:
    return next((p for p in PROFILES if p.name == name), None)


def agent_summaries(info: dict, active: str) -> tuple[AgentSummary,
                                                      list[AgentSummary]]:
    """(the active profile, every agent). The workflow's subagents are listed
    after the two profiles, as SUBAGENT entries."""
    subagents = [
        AgentSummary(name=s["name"], display_name=s["name"],
                     description=s.get("description") or "",
                     safety=AgentSafety.NEUTRAL, agent_type=AgentType.SUBAGENT)
        for s in info.get("subagents", [])]
    return profile(active) or PROFILES[0], [*PROFILES, *subagents]


def tool_summaries(info: dict) -> list[ToolSummary]:
    # is_custom stays False: the copied app shows a deprecation notice for a
    # custom tool, and OVAT's tools are not Vibe's custom tools.
    return [ToolSummary(name=t["name"]) for t in info.get("tools", [])]


@dataclass
class StatsBook:
    """What the stats snapshot reports. The session fills it in."""
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # The last turn's own numbers; None where the engine did not report.
    last_prompt: int | None = None
    last_completion: int | None = None
    last_seconds: float = 0.0
    tool_calls: dict = field(default_factory=lambda: {
        "succeeded": 0, "failed": 0, "rejected": 0, "agreed": 0})

    def snapshot(self) -> AgentStatsSnapshot:
        completion = self.last_completion or 0
        return AgentStatsSnapshot(
            steps=self.turns,
            session_prompt_tokens=self.prompt_tokens,
            session_completion_tokens=self.completion_tokens,
            tool_calls_agreed=self.tool_calls["agreed"],
            tool_calls_rejected=self.tool_calls["rejected"],
            tool_calls_failed=self.tool_calls["failed"],
            tool_calls_succeeded=self.tool_calls["succeeded"],
            # The prompt of the last model call is the best picture of how
            # full the context is that OVAT has (it has no window size).
            context_tokens=self.last_prompt or 0,
            last_turn_prompt_tokens=self.last_prompt or 0,
            last_turn_completion_tokens=completion,
            last_turn_duration=self.last_seconds,
            tokens_per_second=(completion / self.last_seconds
                               if completion and self.last_seconds > 0
                               else 0.0))


def config_view(info: dict, workflow: str | None) -> ConfigView:
    model = info.get("model") or (_stem(workflow) or "OVAT agent")
    levels = ["off"]
    window = int(info.get("context_window") or 0)
    active = ModelConfigView(
        name=model, alias=model, thinking="off", supports_images=False,
        display_name=model, max_context_length=window or None,
        thinking_levels=levels)
    # The copied models want an audio provider even with voice off. The
    # client name is a wire value from the same type the models use; taking
    # it from the type keeps the word out of OVAT's strings.
    client = get_args(AudioClient)[0]
    provider = AudioProviderView(api_base="", api_key_env_var="",
                                 client=client)
    return ConfigView(
        active_model=active, active_model_pinned=True, images_supported=False,
        default_model_alias=model, default_agent=DEFAULT_PROFILE,
        # "auto" lets the app pick; the OVAT theme is the app's own to apply.
        theme="auto", log_level=None,
        disable_welcome_banner_animation=True, cursor_blink=True,
        show_greeting=False, autocopy_to_clipboard=False,
        file_watcher_for_autocomplete=False, ask_confirmation_on_exit=True,
        voice_mode_enabled=False, narrator_enabled=False,
        show_thinking_nodes=True, show_subagent_status_list=True,
        worktree_limit=0,
        # Nothing goes over the network from here: no update check, no
        # telemetry, no notifications.
        enable_update_checks=False, enable_notifications=False,
        enable_telemetry=False, enable_system_trust_store=False,
        experimental_enable_tab_status=False,
        experimental_enable_registry_skills=False,
        models=[active], transcribe_models=[], tts_models=[],
        transcription=TranscriptionConfigView(
            model=TranscribeModelConfigView(
                name="", sample_rate=16000,
                encoding=get_args(TranscriptionEncoding)[0], language="",
                target_streaming_delay_ms=0),
            provider=provider),
        speech=SpeechConfigView(
            model=TTSModelConfigView(
                name="", voice="",
                response_format=get_args(SpeechOutputFormat)[0]),
            provider=provider),
        validation_warnings=[])


def runtime_snapshot(info: dict, workflow: str | None, active: str,
                     stats: AgentStatsSnapshot) -> RuntimeSnapshot:
    current, agents = agent_summaries(info, active)
    window = int(info.get("context_window") or 0)
    return RuntimeSnapshot(
        config=config_view(info, workflow), active_agent=current,
        agents=agents, skills=[], tools=tool_summaries(info), stats=stats,
        context_window=window, issues=[], hooks_count=0,
        connectors=ConnectorCounts(), mcp=MCPState(),
        bypass_tool_permissions=False, experimental_harness=False)


def session_log(session_id: str, store) -> SessionLogSummary:
    """Whether conversations are saved here (`store`, a SessionStore or
    None), and whether this one already is: the app offers "resume it later"
    only when both are true."""
    if store is None:
        return SessionLogSummary(enabled=False, session_id=session_id)
    return SessionLogSummary(enabled=True, session_id=session_id,
                             persisted=store.path(session_id).is_file(),
                             path=str(store.folder))


def _stem(path: str | None) -> str:
    from pathlib import Path
    return Path(path).stem if path else ""
