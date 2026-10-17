# ovat/tui/agents_screen.py
"""`/agents` dashboard screen in the agent TUI.

Shows configured subagents (local workflow, cloud API, remote A2A),
local A2A server status, and a live table of child agent runs with
single-child cancellation via 'x'.
"""
import time
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Static

from ovat.cli import ui


class AgentsScreen(Screen):
    """Subagents, A2A peers, server state, and active child runs."""

    SCOPED_CSS = False
    CSS = """
    #agents-view { background: $background; layout: vertical; }
    #agents-title { height: 1; width: 100%; padding: 0 1; }
    #agents-title-left { width: 1fr; }
    #agents-title-keys { width: auto; color: $text-muted; }
    #agents-serving { height: auto; padding: 0 1; margin: 0 1 1 1; border: solid $foreground 25%; }
    #agents-subagents { height: auto; padding: 0 1; margin: 0 1 1 1; border: solid $foreground 25%; }
    #agents-table-box { height: 1fr; margin: 0 1 1 1; border: solid $foreground 35%; }
    #agents-runs-table { height: 1fr; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("x", "cancel_child", "Cancel child"),
        Binding("r", "refresh_view", "Refresh"),
    ]

    def __init__(self, session=None) -> None:
        super().__init__(id="agents-view")
        self.session = session
        self._table: DataTable | None = None

    def compose(self):
        with Horizontal(id="agents-title"):
            title = Text("Agents & Peers", style="bold")
            yield Static(title, id="agents-title-left")
            keys = Text("Esc back  x cancel child  r refresh", style=ui.DIM)
            yield Static(keys, id="agents-title-keys")

        yield Static(self._serving_text(), id="agents-serving")
        yield Static(self._subagents_text(), id="agents-subagents")

        with Vertical(id="agents-table-box"):
            table = DataTable(id="agents-runs-table", cursor_type="row")
            table.zebra_stripes = True
            yield table

    def on_mount(self) -> None:
        self._table = self.query_one("#agents-runs-table", DataTable)
        self._table.add_columns("Call ID", "Agent", "Kind", "Status", "Duration", "Tokens")
        self.action_refresh_view()
        # Live: a running child's time and status move without pressing r.
        self.set_interval(1.0, self.action_refresh_view)

    def _serving_text(self) -> Text:
        serving = getattr(self.session, "serving", False) if self.session else False
        t = Text()
        t.append("A2A Server: ", style="bold")
        if serving:
            t.append("● serving", style=f"bold {ui.GREEN}")
            t.append(" (run `/serve stop` to end)")
        else:
            t.append("● not serving", style=ui.DIM)
            t.append(" (run `/serve` to offer this workflow over A2A)")
        return t

    def _subagents_text(self) -> Text:
        info = getattr(self.session, "info", {}) if self.session else {}
        subagents = info.get("subagents", [])
        tools = info.get("tools", [])
        a2a_tools = [t for t in tools if t.get("kind") == "a2a"]

        t = Text()
        t.append("Configured Subagents & Remote Peers\n", style="bold")
        if not subagents and not a2a_tools:
            t.append("  No subagents configured in this workflow (add `subagents:`).", style=ui.DIM)
            return t

        for s in subagents:
            t.append("  • ", style=ui.CYAN)
            t.append(s.get("name", "unknown"), style="bold")
            t.append(f" ({s.get('where', 'local')}): ")
            t.append(s.get("description", ""), style=ui.DIM)
            t.append("\n")

        for p in a2a_tools:
            t.append("  • ", style=ui.GREEN)
            t.append(p.get("name", "unknown"), style="bold")
            # The card as describe_agent gives it (plain values), or none
            # when the remote could not be reached at start.
            card = p.get("card") or {}
            card_desc = ""
            if card:
                card_desc = f" [{card.get('name', '')} v{card.get('version', '')}"
                skills = ", ".join(card.get("skills", []))
                card_desc += f": {skills}]" if skills else "]"
            t.append(f" (a2a remote, permission: {p.get('permission', 'ask')}){card_desc}")
            t.append("\n")
        return t

    def action_refresh_view(self) -> None:
        if self._table is None:
            return
        cursor = self._table.cursor_row
        self._table.clear()
        getter = getattr(self.session, "get_child_runs", None)
        runs = getter() if callable(getter) else []
        for r in runs:
            cid = r.get("call_id", "")
            agent = r.get("agent", "")
            kind = r.get("kind", "")
            status = r.get("status", "")
            if r.get("duration") is not None:
                dur = f"{r['duration']:.2f} s"
            elif status == "running" and r.get("started_at") is not None:
                dur = f"{time.monotonic() - r['started_at']:.0f} s so far"
            else:
                dur = "-"
            toks = str(r["tokens"]) if r.get("tokens") is not None else "-"
            status_style = ui.GREEN if status == "ok" else (ui.RED if status == "failed" else (ui.YELLOW if status == "cancelled" else ui.CYAN))
            self._table.add_row(
                Text(cid, style="bold"),
                Text(agent),
                Text(kind, style=ui.DIM),
                Text(status, style=status_style),
                Text(dur),
                Text(toks),
                key=cid,
            )
        if self._table.row_count:
            self._table.move_cursor(row=min(cursor, self._table.row_count - 1))

    def action_cancel_child(self) -> None:
        if self._table is None or self._table.row_count == 0:
            self.notify("No child runs to cancel", severity="information")
            return
        try:
            row_key, _ = self._table.coordinate_to_cell_key(self._table.cursor_coordinate)
            call_id = row_key.value if hasattr(row_key, "value") else str(row_key)
        except Exception:
            self.notify("Select a child run to cancel", severity="information")
            return

        if not self.session:
            return

        canceller = getattr(self.session, "cancel_subagent", None)
        if callable(canceller) and canceller(call_id):
            self.notify(f"Cancelled child {call_id}", severity="information")
            self.action_refresh_view()
        else:
            self.notify(f"Child {call_id} is not currently running", severity="warning")

    def action_back(self) -> None:
        self.app.pop_screen()
