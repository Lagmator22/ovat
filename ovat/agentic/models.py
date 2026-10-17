# ovat/agentic/models.py
"""Which models a session can switch to (the /model picker's list).

Note to myself: the list depends on who runs the model.

- provider ovms: ONLY what the running server says it serves. OVMS 2026.4.1
  answers `GET <ovms_url>/models` (the /v3 base) with {"data": [{"id": ...}]};
  `ovat serve` already waits on the same route. Naming a model the server
  does not serve would only fail on the next question.
- provider genai: the model folders on this machine that a chat can use
  (`model_scout.CHAT_KINDS`), found the way `ovat chat` finds them.

If the list cannot be had (no server, nothing on disk), the picker still
offers the workflow's own model, and nothing is reported: not being able to
list is not an error, it only means there is one choice.
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from ovat.core import model_scout

LIST_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class ModelChoice:
    id: str        # what `model.name` is set to (a served name, or a folder)
    label: str     # what the picker shows


def _label(name: str) -> str:
    """A local model is named by its folder, as the banner shows it."""
    return os.path.basename(name.replace("\\", "/").rstrip("/")) or name


def served_models(ovms_url: str) -> list[str]:
    """The model names the server at `ovms_url` serves; [] when it cannot
    be asked."""
    try:
        # An opener with an empty ProxyHandler, as model_server.py has: a
        # plain urlopen sends localhost through HTTP_PROXY, which refuses it.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(ovms_url.rstrip("/") + "/models",
                         timeout=LIST_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [item["id"] for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")]
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        return []


def local_models(config) -> list[ModelChoice]:
    roots = ([config.model.model_repository_path]
             if config.model.model_repository_path else []) + list(
        getattr(config, "model_search_paths", None) or [])
    return [ModelChoice(m["path"], m["name"])
            for m in model_scout.find_models(extra_roots=roots)
            if m["kind"] in model_scout.CHAT_KINDS]


def choices(config) -> list[ModelChoice]:
    """The models this session can use."""
    m = config.model
    if m.provider == "ovms":
        found = [ModelChoice(name, name) for name in served_models(m.ovms_url)]
    else:
        found = local_models(config)
    here = os.path.realpath(m.name)
    present = any(c.id == m.name or os.path.realpath(c.id) == here
                  for c in found)
    # A server that lists models is the truth about what it serves. Only a
    # list that is empty (server not asked, or silent), or a local folder
    # outside the scanned ones, falls back to the workflow's own model.
    if not found or (m.provider != "ovms" and not present):
        found.insert(0, ModelChoice(m.name, _label(m.name)))
    return found
