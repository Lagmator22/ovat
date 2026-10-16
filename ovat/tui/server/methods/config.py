# ovat/tui/server/methods/config.py
"""Reading the configuration, and the two things a session may change: its
model and its thinking switch. A pick lasts for this session only; writing it
back into the workflow file comes in a later phase."""
from ovat.tui.server import views
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
        config=views.config_view(ctx.info, ctx.workflow),
        skills_count=0, hooks_count=0, mcp_servers_total=0,
        mcp_servers_enabled=0)


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


# Vibe's `config/fields/read` says which settings an administrator's layer
# has fixed; the /model picker asks it before opening, to refuse a model
# the administrator pinned. OVAT has one workflow file and no such layer, so
# nothing is ever fixed: no fields, no targets.
@routes.method("config/fields/read", ConfigFieldsReadParams)
async def config_fields(ctx, params: ConfigFieldsReadParams):
    ctx.require_session(params.session_id)
    return ConfigFieldsReadResponse(fields=[], targets=[])


# ---- the model and thinking pickers (/model, /thinking) -------------------------

# Mirrors Vibe's `config/model/write` (vibe/app_server/_resources.py,
# `_model_config_write_params` + `_config_write`, and _config_write.py
# `model_config_write_ops`): the request names a model alias and/or a
# reasoning effort; an alias the configuration does not offer is
# invalidParams "Unknown model: X. Available: ..."; an EMPTY alias means
# "follow the default" (the workflow's own model here); the answer is the
# runtime that the write produced, and a write that could not be applied comes
# back as `failures`, which the client shows. Vibe applies it with a config
# patch and a reload; OVAT builds the agent again with the pick (a session
# override, see AgentSession.switch_model) and keeps the conversation.
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
    thinking = None if effort is None else effort != views.THINKING_OFF
    target = (info.get("default_model_id") if alias == ""
              else alias if alias is not None else None)
    change_model = target is not None and target != info.get("model_id")
    change_thinking = (thinking is not None
                       and thinking != bool(info.get("thinking")))
    if change_model or change_thinking:
        remembers = ctx.agent.memory() is not None
        failure = await ctx.pick(
            model=alias if change_model else None,
            enable_thinking=thinking if change_thinking else None)
        if failure is not None:
            return ConfigWriteResponse(runtime=ctx.runtime(),
                                       failures=[failure])
        if not remembers and ctx.projector is not None:
            # The same words a resumed conversation gets on this engine.
            ctx.projector.warning(
                "This engine keeps its own memory, so the model starts this "
                "conversation afresh after the switch. `/engine native` "
                "keeps it.")
    return ConfigWriteResponse(runtime=ctx.runtime())


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
