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
import functools
import operator
from typing import Annotated, Any, Literal

from pydantic import Field, create_model
from typing_extensions import NotRequired, Required, TypedDict

# JSON-schema type names -> Python types, for deriving argument models.
_JSON_TO_PY = {"string": str, "integer": int, "number": float,
               "boolean": bool, "array": list, "object": dict}


def _type_from_spec(prefix: str, spec: dict):
    """Turn one JSON Schema property spec into a Python type annotation."""
    if not isinstance(spec, dict):
        return Any
    enum_vals = spec.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        literals = tuple(v for v in enum_vals
                         if isinstance(v, (str, int, bool, type(None))))
        if literals:
            return Literal[literals]
    variants = spec.get("anyOf") or spec.get("oneOf")
    if isinstance(variants, list) and variants:
        sub_types = [_type_from_spec(f"{prefix}_{i}", v)
                     for i, v in enumerate(variants)]
        return functools.reduce(operator.or_, sub_types)
    raw_type = spec.get("type")
    if isinstance(raw_type, list):
        if not raw_type:
            return Any
        sub_types = [
            type(None) if t == "null"
            else _type_from_spec(prefix, {**spec, "type": t})
            for t in raw_type
        ]
        return functools.reduce(operator.or_, sub_types)
    if raw_type == "null":
        return type(None)
    if raw_type == "array":
        items = spec.get("items")
        if isinstance(items, dict) and items:
            item_type = _type_from_spec(f"{prefix}_Item", items)
            return list[item_type]
        return list
    if raw_type == "object":
        props = spec.get("properties")
        if isinstance(props, dict) and props:
            req = set(spec.get("required") or [])
            td_fields = {}
            for k, v in props.items():
                v_dict = v if isinstance(v, dict) else {}
                sub_t = _type_from_spec(f"{prefix}_{k}", v_dict)
                desc = v_dict.get("description", "")
                if k in req:
                    ann = (Annotated[sub_t, Field(description=desc)]
                           if desc else sub_t)
                    td_fields[k] = Required[ann]
                else:
                    if "default" in v_dict:
                        ann = Annotated[sub_t, Field(
                            default=v_dict["default"], description=desc)]
                    elif desc:
                        ann = Annotated[sub_t, Field(description=desc)]
                    else:
                        ann = sub_t
                    td_fields[k] = NotRequired[ann]
            return TypedDict(f"{prefix}_Obj", td_fields)
        additional = spec.get("additionalProperties")
        if isinstance(additional, dict) and additional:
            val_t = _type_from_spec(f"{prefix}_Value", additional)
            return dict[str, val_t]
        return dict
    return _JSON_TO_PY.get(raw_type, str)


def args_model_from_schema(name: str, schema: dict):
    """Build a pydantic model for one tool's arguments, from its SCHEMA.

    create_model is pydantic's runtime class factory: it builds the same kind
    of class a `class ...Args(BaseModel)` block would, from data instead of
    source code.
    """
    params = dict(schema["function"].get("parameters") or {})
    required = set(params.get("required", []))
    fields = {}
    for pname, spec in params.get("properties", {}).items():
        spec_dict = spec if isinstance(spec, dict) else {}
        py_type = _type_from_spec(f"{name}_{pname}", spec_dict)
        # Required fields use `...` (pydantic's "no default, must be given");
        # optional ones take the schema's default (None if it names none).
        is_req = pname in required
        default = ... if is_req else spec_dict.get("default", None)
        if not is_req and default is None:
            py_type = py_type | None
        fields[pname] = (
            py_type,
            Field(default, description=spec_dict.get("description", "")),
        )
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
