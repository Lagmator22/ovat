# ovat/tui/server/methods/host.py
"""Answers about the account, the workspace and the app's own chatter.

OVAT has no account and no cloud: nobody is signed in, the workspace is
trusted (the person started the agent here), and nothing is sent anywhere.
`telemetry/record` and `feedback/record` are accepted and dropped, so the
copied app can keep calling them without an error on every click.
"""
from ovat.tui.server.rpc import Routes
from ovat.tui.vibe.app_server.models import (AccountStatus, AccountView,
                                             MentionStats, PreparedPrompt)
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
                                               WorkspaceTrustStatusParams,
                                               WorkspaceTrustStatusResponse,
                                               WorkspaceUntrustedConfigParams,
                                               WorkspaceUntrustedConfigResponse)
from ovat.tui.vibe.app_server.provider_auth import ProviderAuthView

routes = Routes()


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
    # The base URL is left out: it can carry a host name or a port that the
    # status screen has no need to print.
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
    return WorkspaceTrustStatusResponse(status="trusted")


@routes.method("workspace/trust/untrustedConfig",
               WorkspaceUntrustedConfigParams)
async def untrusted_config(ctx, params: WorkspaceUntrustedConfigParams):
    return WorkspaceUntrustedConfigResponse()


@routes.method("workspace/prompt/prepare", WorkspacePromptPrepareParams)
async def prompt_prepare(ctx, params: WorkspacePromptPrepareParams):
    ctx.require_session(params.session_id)
    return WorkspacePromptPrepareResponse(prompt=PreparedPrompt(
        display_text=params.message, prompt_text=params.message,
        mentions=MentionStats()))
