# ovat/tui/server/methods/config.py
"""Reading the configuration, and changing it.

The model and thinking pickers and /config write the workflow file, keeping
comments and validating the result first. /theme writes the UI preferences
file, see ovat/tui/server/settings.py.
"""
import asyncio

from ovat.agentic import prefs, workflow_edit
from ovat.tui.server import settings, views
from ovat.tui.server.rpc import Routes, RpcError
from ovat.tui.vibe.app_server.protocol import (ConfigEffectiveReadParams,
                                               ConfigEffectiveReadResponse,
                                               ConfigFieldsReadParams,
                                               ConfigFieldsReadResponse,
                                               ConfigMutationResponse,
                                               ConfigReadParams,
                                               ConfigReadResponse,
                                               ConfigReloadParams,
                                               ConfigSchemaReadParams,
                                               ConfigSchemaReadResponse,
                                               ConfigWriteParams,
                                               ConfigWriteOpWire,
                                               ConfigWriteResponse,
                                               ModelConfigWriteParams,
                                               ProtocolErrorCode)

routes = Routes()

SCHEMA_VERSION = "ovat-workflow-1"


@routes.method("config/read", ConfigReadParams)
async def config_read(ctx, params: ConfigReadParams):
    if params.session_id is not None:
        ctx.require_session(params.session_id)
    return ConfigReadResponse(
        config=views.config_view(ctx.info, ctx.workflow, ctx.theme),
        skills_count=0, hooks_count=0,
        mcp_servers_total=len(ctx.info.get("mcp", ())),
        mcp_servers_enabled=sum(s["status"] != "disabled"
                                for s in ctx.info.get("mcp", ())))


@routes.method("config/effective/read", ConfigEffectiveReadParams)
async def config_effective_read(ctx, params: ConfigEffectiveReadParams):
    ctx.require_session(params.session_id)
    info = ctx.info
    # Built from what the banner already shows, so nothing secret (an MCP
    # server's env, a token) can end up in it.
    return ConfigEffectiveReadResponse(config={
        "workflow": ctx.workflow,
        "profile": ctx.profile,
        "model": {"name": info.get("model"), "provider": info.get("provider"),
                  "device": info.get("device")},
        "engine": info.get("engine"),
        "tools": [{k: t.get(k) for k in ("name", "kind", "permission")}
                  for t in info.get("tools", [])],
        "subagents": [{k: a.get(k) for k in ("name", "where")}
                      for a in info.get("subagents", [])]})


@routes.method("config/schema", ConfigSchemaReadParams)
async def config_schema(ctx, params: ConfigSchemaReadParams):
    from ovat.config.workflow import WorkflowConfig
    return ConfigSchemaReadResponse(
        config_schema_version=SCHEMA_VERSION,
        config_schema=WorkflowConfig.model_json_schema())


# Vibe's `config/fields/read` lists every setting with where each value is set
# (default, a config layer...) and the layers a write may name; the settings
# screen is drawn from it, and the /model picker asks it before opening, to
# refuse a model an administrator pinned. OVAT has no administrator layer: the
# list is the settings OVAT really has (settings.fields), and the one write
# target is the workflow file.
@routes.method("config/fields/read", ConfigFieldsReadParams)
async def config_fields(ctx, params: ConfigFieldsReadParams):
    ctx.require_session(params.session_id)
    return ConfigFieldsReadResponse(
        fields=settings.fields(ctx.info, ctx.workflow, ctx.theme,
                               ctx.theme_saved),
        targets=settings.TARGETS)


def _theme_names() -> set[str]:
    """The themes that exist: Textual's and OVAT's own, and "auto". textual
    is imported here, when a theme is written, and not before: the server
    module itself stays free of it (the app is running Textual by then)."""
    from textual.theme import BUILTIN_THEMES
    from ovat.cli.theme import OVAT_THEME
    return {views.AUTO_THEME, OVAT_THEME.name, *BUILTIN_THEMES}


# Mirrors Vibe's `config/write` (vibe/app_server/_resources.py `_config_write`):
# refused during a turn (CONFLICT); an invalid edit comes back `rejected`; a
# failure comes back in `failures`; a write that asked for it
# (`reload_runtime`) builds the agent again. Without that flag the settings
# screen writes several fields and sends one `config/reload` when it closes,
# which rebuilds once, and only if the file changed (ServerContext.reload).
#
# All or nothing: every op is planned first, and the file is written only if
# the whole result is a valid workflow. A setting Vibe has and OVAT does not
# (voice, narrator...) is answered "not used by OVAT" and changes nothing.
@routes.method("config/write", ConfigWriteParams)
async def config_write(ctx, params: ConfigWriteParams):
    ctx.require_session(params.session_id)
    if ctx.turns.busy:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "Wait for the current turn to end, or interrupt it, "
                       "before changing settings")
    if not ctx.is_ready:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "The agent is not ready yet.")
    plan = settings.plan(params.ops, ctx.workflow, _theme_names)
    plan.failures = list(dict.fromkeys(plan.failures))
    if plan.failures:
        # Nothing was written. The screen only says "could not save"; the
        # reason is said once in the transcript.
        ctx.projector.warning("; ".join(plan.failures))
        return ConfigWriteResponse(runtime=ctx.runtime(),
                                   failures=plan.failures)
    if plan.touches_workflow:
        try:
            await asyncio.to_thread(workflow_edit.apply_changes,
                                    ctx.workflow, plan.changes)
        except workflow_edit.EditError as exc:
            ctx.projector.warning(str(exc))
            return ConfigWriteResponse(runtime=ctx.runtime(), rejected=True,
                                       failures=[str(exc)])
    if plan.theme is not None:
        try:
            if plan.theme == views.AUTO_THEME:
                ctx.theme_saved = False
                prefs.update(ctx.cwd, theme=None)
            else:
                prefs.update(ctx.cwd, theme=plan.theme)
                ctx.theme_saved = True
        except OSError as exc:
            return ConfigWriteResponse(runtime=ctx.runtime(), failures=[
                f"Could not save the theme: {exc.strerror or exc}"])
        ctx.theme = plan.theme
        ctx.refresh_runtime()
    if plan.forget:
        ctx.agent.forget_picks(*plan.forget)
    if plan.changes_the_agent:
        ctx.settings_changed()
        if params.reload_runtime:
            await ctx.reload()
    return ConfigWriteResponse(runtime=ctx.runtime())


# ---- the model and thinking pickers (/model, /thinking) -------------------------

# Mirrors Vibe's model_config_write_ops in app_server/_config_write.py:
# the pickers write the same fields as the settings screen and reload once.
# An empty alias follows the workflow's default model. The rebuild keeps
# native conversation memory through ServerContext.reload/pick.
@routes.method("config/model/write", ModelConfigWriteParams)
async def config_model_write(ctx, params: ModelConfigWriteParams):
    ctx.require_session(params.session_id)
    if ctx.turns.busy:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "Wait for the current turn to end, or interrupt it, "
                       "before switching the model")
    info = ctx.info
    offered = [m["id"] for m in info.get("models") or []]
    alias = params.model_alias
    if alias and alias not in offered:
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       f"Unknown model: {alias}. Available: "
                       f"{', '.join(sorted(offered))}")
    effort = params.reasoning_effort
    if effort is not None and effort not in views.THINKING_LEVELS:
        raise RpcError(ProtocolErrorCode.INVALID_PARAMS,
                       f"Unknown thinking level: {effort}. Available: "
                       f"{', '.join(views.THINKING_LEVELS)}")
    ops = []
    if alias is not None:
        ops.append(ConfigWriteOpWire(op="set", path="/active_model", value=alias))
    if effort is not None:
        ops.append(ConfigWriteOpWire(op="set", path="/model/enable_thinking",
                                     value=effort != views.THINKING_OFF))
    remembers = ctx.agent.memory() is not None
    response = await config_write(ctx, ConfigWriteParams(
        session_id=params.session_id, ops=ops, reason="model picker",
        reload_runtime=True))
    if response.applied and ops and not remembers and ctx.projector is not None:
        ctx.projector.warning(
            "This engine keeps its own memory, so the model starts this "
            "conversation afresh after the switch. `/engine native` keeps it.")
    return response


# Vibe's `config/reload` re-reads its configuration files and rebuilds the
# agent loop with the messages kept. OVAT's configuration file is the
# workflow: it is read again, and the agent rebuilt, only when the file
# changed since the agent was built (a model pick has just rebuilt it, so the
# reload the client sends right after one must not load the model twice).
@routes.method("config/reload", ConfigReloadParams)
async def config_reload(ctx, params: ConfigReloadParams):
    ctx.require_session(params.session_id)
    if ctx.turns.busy:
        raise RpcError(ProtocolErrorCode.CONFLICT,
                       "Wait for the current turn to end, or interrupt it, "
                       "before reloading")
    if ctx.is_ready and params.reload_runtime:
        await ctx.reload()
    return ConfigMutationResponse(runtime=ctx.runtime())
