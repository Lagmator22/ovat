# ovat/tools/todo.py
"""The todo tool: the agent's own task list for a long job.

Mirrors Mistral Vibe's todo.py (vibe/core/tools/builtins): one tool with
action "read" or "write"; "write" REPLACES the whole list; items have an id,
content, status and priority; at most 100; ids unique. The list lives in a
TodoStore that factory.build_tools makes once per built agent, so it is
session state and never touches the disk.
"""
from ovat.tools.project import Shown, fail

MAX_TODOS = 100                  # Vibe's TodoConfig.max_todos
STATUSES = ("pending", "in_progress", "completed", "cancelled")
PRIORITIES = ("low", "medium", "high")
_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]",
         "cancelled": "[-]"}


class TodoStore:
    """The list for one agent. A plain holder, like Vibe's TodoState."""

    def __init__(self) -> None:
        self.items: list[dict] = []


def _item(raw, position: int) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"todo {position} is not an object with id and content."
    todo_id = str(raw.get("id") or "").strip()
    content = str(raw.get("content") or "").strip()
    if not todo_id or not content:
        return None, f"todo {position} needs both an id and content."
    status = str(raw.get("status") or "pending").strip().lower()
    priority = str(raw.get("priority") or "medium").strip().lower()
    if status not in STATUSES:
        return None, (f"todo {todo_id} has status '{status}'. Use one of: "
                      f"{', '.join(STATUSES)}.")
    if priority not in PRIORITIES:
        return None, (f"todo {todo_id} has priority '{priority}'. Use one "
                      f"of: {', '.join(PRIORITIES)}.")
    return {"id": todo_id, "content": content, "status": status,
            "priority": priority}, None


def _render(verb: str, items: list[dict]) -> str:
    head = f"{verb} {len(items)} todos"
    if not items:
        return head
    return head + ":\n" + "\n".join(
        f"{_MARK[t['status']]} {t['id']}: {t['content']} "
        f"({t['status']}, {t['priority']})" for t in items)


def todo_impl(action: str, todos: list | None, store: TodoStore) -> str:
    action = (action or "").strip().lower()
    if action == "read":
        shown = {"kind": "todo", "action": "read", "todos": store.items}
        return Shown(_render("Retrieved", store.items), shown)
    if action != "write":
        return fail("todo", "action must be 'read' or 'write'.")
    if not isinstance(todos, list):
        return fail("todo", "action 'write' needs todos: the full list, "
                            "which replaces the old one.")
    if len(todos) > MAX_TODOS:
        return fail("todo", f"cannot store more than {MAX_TODOS} todos.")
    items = []
    for position, raw in enumerate(todos, start=1):
        item, why = _item(raw, position)
        if why:
            return fail("todo", why)
        items.append(item)
    if len({t["id"] for t in items}) != len(items):
        return fail("todo", "todo ids must be unique.")
    store.items = items
    return Shown(_render("Updated", items),
                 {"kind": "todo", "action": "write", "todos": items})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": "Keep your own task list for a job with several "
                       "steps. Use action 'write' with the FULL list to "
                       "plan or update it (it replaces the old list), and "
                       "'read' to see it. Mark one item in_progress while "
                       "you work on it and completed when it is done.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "description": "'read' to view the list, or "
                                          "'write' to replace it"},
                "todos": {
                    "type": "array",
                    "description": "for 'write': the full list. Each item "
                                   "is an object with id (stable, unique), "
                                   "content (short), status (pending, "
                                   "in_progress, completed or cancelled) "
                                   "and priority (low, medium or high)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": list(STATUSES),
                                       "default": "pending"},
                            "priority": {"type": "string", "enum": list(PRIORITIES),
                                         "default": "medium"},
                        },
                        "required": ["id", "content"],
                    }},
            },
            "required": ["action"],
        },
    },
}
