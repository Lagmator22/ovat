# ovat/tui/server/methods/host.py
"""Workspace, trust, prompt preparation, and stubbed cloud-host handlers.

OVAT runs locally without a cloud account or outbound telemetry: account reads
report unsigned state, telemetry/feedback requests are no-ops, workspace trust
persists in ``~/.ovat/trusted.json``, and ``@file`` mentions attach safe text
and pasted image context locally.
"""
import os
import re
import tempfile
from pathlib import Path

from ovat.agentic import trust
from ovat.tools import project
from ovat.tui.server.rpc import Routes
from ovat.tui.vibe.app_server.models import (AccountStatus, AccountView,
                                             FileImageSource, ImageAttachment,
                                             MentionStats, PreparedPrompt,
                                             WorkspaceTrustDetails)
from ovat.tui.vibe.app_server.protocol import (AccountReadParams,
                                               AccountReadResponse,
                                               EmptyResponse,
                                               FeedbackRecordParams,
                                               FeedbackShouldShowParams,
                                               FeedbackShouldShowResponse,
                                               IdentityReadParams,
                                               IdentityReadResponse,
                                               ProviderAuthReadParams,
                                               ProviderAuthReadResponse,
                                               TelemetryRecordParams,
                                               WorkspacePromptPrepareParams,
                                               WorkspacePromptPrepareResponse,
                                               WorkspaceTrustDecisionParams,
                                               WorkspaceTrustStatusParams,
                                               WorkspaceTrustStatusResponse,
                                               WorkspaceUntrustedConfigParams,
                                               WorkspaceUntrustedConfigResponse)
from ovat.tui.vibe.app_server.provider_auth import ProviderAuthView

routes = Routes()

_MAX_MENTION_BYTES = 128 * 1024
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_MENTION_RE = re.compile(
    r"""(?:^|(?<=\s))@(?:'([^']+)'|"([^"]+)"|([^\s,;()\[\]{}<>"']+))"""
)


def _target_cwd(ctx, cwd: str | None) -> str:
    return cwd or ctx.cwd or os.getcwd()


def _is_pasted_temp_image(raw: str) -> Path | None:
    temp_dir = (Path(tempfile.gettempdir()) / "ovat-pasted-images").resolve()
    try:
        candidate = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if candidate != temp_dir and candidate.is_relative_to(temp_dir):
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_MIMES:
            return candidate
    return None


@routes.method("account/read", AccountReadParams)
async def account_read(ctx, params: AccountReadParams):
    ctx.require_session(params.session_id)
    return AccountReadResponse(account=AccountView(
        status=AccountStatus.MISSING_KEY))


@routes.method("identity/read", IdentityReadParams)
async def identity_read(ctx, params: IdentityReadParams):
    ctx.require_session(params.session_id)
    return IdentityReadResponse(identity=None)


@routes.method("providerAuth/read", ProviderAuthReadParams)
async def provider_auth_read(ctx, params: ProviderAuthReadParams):
    ctx.require_session(params.session_id)
    provider = str(ctx.info.get("provider") or "ovms")
    return ProviderAuthReadResponse(auth=ProviderAuthView(
        model_display_name=str(ctx.info.get("model") or ""),
        provider_name=provider, api_base=None))


@routes.method("feedback/shouldShow", FeedbackShouldShowParams)
async def feedback_should_show(ctx, params: FeedbackShouldShowParams):
    return FeedbackShouldShowResponse(show=False)


@routes.method("feedback/record", FeedbackRecordParams)
async def feedback_record(ctx, params: FeedbackRecordParams):
    return EmptyResponse()


@routes.method("telemetry/record", TelemetryRecordParams)
async def telemetry_record(ctx, params: TelemetryRecordParams):
    return EmptyResponse()


@routes.method("workspace/trust/status", WorkspaceTrustStatusParams)
async def trust_status(ctx, params: WorkspaceTrustStatusParams):
    cwd = _target_cwd(ctx, params.cwd)
    return WorkspaceTrustStatusResponse(
        status=trust.status(cwd),
        details=WorkspaceTrustDetails(**trust.details(cwd)),
    )


@routes.method("workspace/trust/decision", WorkspaceTrustDecisionParams)
async def trust_decision(ctx, params: WorkspaceTrustDecisionParams):
    if params.session_id is not None:
        ctx.require_session(params.session_id)
    cwd = _target_cwd(ctx, params.cwd)
    new_status = trust.decide(cwd, params.decision)
    return WorkspaceTrustStatusResponse(
        status=new_status,
        details=WorkspaceTrustDetails(**trust.details(cwd)),
    )


@routes.method("workspace/trust/untrustedConfig",
               WorkspaceUntrustedConfigParams)
async def untrusted_config(ctx, params: WorkspaceUntrustedConfigParams):
    cwd = _target_cwd(ctx, params.cwd)
    dirs, settings_path = trust.untrusted_config_dirs(cwd)
    return WorkspaceUntrustedConfigResponse(dirs=dirs, settings_path=settings_path)


@routes.method("workspace/prompt/prepare", WorkspacePromptPrepareParams)
async def prompt_prepare(ctx, params: WorkspacePromptPrepareParams):
    ctx.require_session(params.session_id)
    root = Path(ctx.cwd or os.getcwd()).resolve()
    subagents = {
        str(entry.get("name") or "")
        for entry in (ctx.info.get("subagents") or [])
        if entry.get("name")
    }
    stats = MentionStats()
    contexts: list[str] = []
    images: list[ImageAttachment] = []
    seen: set[Path] = set()

    for match in _MENTION_RE.finditer(params.message):
        quoted = match.group(1) is not None or match.group(2) is not None
        raw = match.group(1) or match.group(2) or match.group(3)
        if not raw or (not quoted and raw in subagents):
            continue

        candidates = [raw]
        if not quoted:
            stripped = raw.rstrip(".,;:!?")
            if stripped and stripped != raw:
                candidates.append(stripped)

        pasted: Path | None = None
        resolved: Path | None = None
        token_used = raw
        for candidate in candidates:
            pasted = _is_pasted_temp_image(candidate)
            if pasted is not None:
                resolved = pasted
                token_used = candidate
                break
            path_obj, _ = project.resolve_inside(candidate, root)
            if path_obj is not None and path_obj.is_file():
                resolved = path_obj
                token_used = candidate
                break

        if resolved is None or resolved in seen:
            continue

        if pasted is not None:
            try:
                size = resolved.stat().st_size
            except OSError:
                continue
            if size > _MAX_IMAGE_BYTES:
                continue
            seen.add(resolved)
            ext = resolved.suffix.lower()
            images.append(ImageAttachment(
                source=FileImageSource(path=str(resolved)),
                alias=resolved.name,
                mime_type=_IMAGE_MIMES[ext],
            ))
            stats.count += 1
            stats.context_types["image"] = stats.context_types.get("image", 0) + 1
            stats.file_extensions[ext] = stats.file_extensions.get(ext, 0) + 1
            continue

        rel = resolved.relative_to(root).as_posix()
        if project.is_sensitive_path(token_used, for_read=True) or project._names_sensitive(rel, True):
            continue

        try:
            size = resolved.stat().st_size
        except OSError:
            continue

        ext = resolved.suffix.lower()
        if ext in _IMAGE_MIMES:
            if size > _MAX_IMAGE_BYTES:
                continue
            seen.add(resolved)
            images.append(ImageAttachment(
                source=FileImageSource(path=str(resolved)),
                alias=rel,
                mime_type=_IMAGE_MIMES[ext],
            ))
            stats.count += 1
            stats.context_types["image"] = stats.context_types.get("image", 0) + 1
            stats.file_extensions[ext] = stats.file_extensions.get(ext, 0) + 1
            continue

        if size > _MAX_MENTION_BYTES:
            continue
        try:
            raw_bytes = resolved.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw_bytes:
            continue
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        seen.add(resolved)
        contexts.append(f'<context ref="{rel}">\n{content}\n</context>')
        stats.count += 1
        stats.context_types["file"] = stats.context_types.get("file", 0) + 1
        ext_key = resolved.suffix or "none"
        stats.file_extensions[ext_key] = stats.file_extensions.get(ext_key, 0) + 1

    prompt_text = (
        params.message
        if not contexts
        else f"{params.message}\n\n" + "\n\n".join(contexts)
    )
    return WorkspacePromptPrepareResponse(prompt=PreparedPrompt(
        display_text=params.message,
        prompt_text=prompt_text,
        images=images,
        mentions=stats,
    ))
