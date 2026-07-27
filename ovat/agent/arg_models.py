# ovat/agent/arg_models.py
"""Derive per-framework argument models from a tool's own SCHEMA.

Every OVAT tool ships its full contract in the SCHEMA dict co-located with its
implementation. That dict is THE contract: it carries the parameter names,
types, descriptions and defaults, and the native loop sends it to OVMS
verbatim.

Every other engine needs the same information in its own shape, and the one
thing that must never happen is a second, hand-kept registry. That mistake was
made once already: a hand-written arg-model table meant a new tool worked on
the native path and then crashed on the LangChain path until someone
remembered to update the second list. Deriving from the SCHEMA means adding a
tool is one edit, and every engine picks it up.

This module holds only pydantic and stdlib imports, so importing it costs
nothing and pulls in no framework.
"""
from pydantic import Field, create_model

# JSON-schema type names -> Python types, for deriving argument models.
_JSON_TO_PY = {"string": str, "integer": int, "number": float,
               "boolean": bool, "array": list, "object": dict}


def args_model_from_schema(name: str, schema: dict):
    """Build a pydantic model for one tool's arguments, from its SCHEMA.

    create_model is pydantic's runtime class factory: it builds the same kind
    of class a `class ...Args(BaseModel)` block would, from data instead of
    source code.
    """
    params = schema["function"].get("parameters", {})
    required = set(params.get("required", []))
    fields = {}
    for pname, spec in params.get("properties", {}).items():
        py_type = _JSON_TO_PY.get(spec.get("type"), str)
        # Required fields use `...` (pydantic's "no default, must be given");
        # optional ones take the schema's default (None if it names none).
        default = ... if pname in required else spec.get("default", None)
        fields[pname] = (py_type,
                         Field(default, description=spec.get("description", "")))
    return create_model(f"{name}_Args", **fields)


def json_schema_for_tool(schema: dict) -> dict:
    """The raw JSON Schema for a tool's parameters.

    Some SDKs (the OpenAI Agents SDK) want the JSON Schema itself rather than
    a pydantic model, so this hands back the parameters block unchanged. An
    empty schema still has to be a valid object with a properties map, or
    strict-mode validation rejects the tool before it is ever called.
    """
    params = dict(schema["function"].get("parameters") or {})
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    return params
