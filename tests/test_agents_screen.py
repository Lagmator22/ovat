# tests/test_agents_screen.py
"""Tests for the /agents dashboard screen."""
import asyncio
import pytest

textual = pytest.importorskip("textual")
from textual.app import App
from textual.widgets import DataTable

from ovat.tui.agents_screen import AgentsScreen


class FakeSession:
    def __init__(self):
        self.serving = True
        self.info = {
            "subagents": [
                {"name": "researcher", "where": "workflow", "description": "Researches docs"},
                {"name": "remote_math", "where": "a2a", "description": "Solves math"},
            ],
            "tools": [
                {
                    "name": "remote_math",
                    "kind": "a2a",
                    "permission": "ask",
                    "card": {"name": "math-agent", "version": "1.0.0",
                             "skills": ["add", "mul"]},
                }
            ],
        }
        self.cancelled = []
        self.runs = [
            {
                "call_id": "c1",
                "agent": "researcher",
                "kind": "workflow",
                "status": "running",
                "started_at": 100.0,
                "duration": None,
                "tokens": None,
            },
            {
                "call_id": "c2",
                "agent": "remote_math",
                "kind": "a2a",
                "status": "ok",
                "started_at": 90.0,
                "duration": 1.45,
                "tokens": 250,
            },
        ]

    def get_child_runs(self):
        return list(self.runs)

    def cancel_subagent(self, call_id: str) -> bool:
        if call_id == "c1":
            self.cancelled.append(call_id)
            return True
        return False


def test_agents_screen_renders_info_and_runs():
    async def scenario():
        session = FakeSession()

        class Harness(App):
            def on_mount(self):
                self.screen_instance = AgentsScreen(session)
                self.push_screen(self.screen_instance)

        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_instance
            # 1. Verify serving text
            serving_text = screen._serving_text().plain
            assert "A2A Server:" in serving_text
            assert "serving" in serving_text

            # 2. Verify subagents text includes names and A2A card info
            subagents_text = screen._subagents_text().plain
            assert "researcher" in subagents_text
            assert "remote_math" in subagents_text
            assert "math-agent" in subagents_text
            assert "add, mul" in subagents_text

            # 3. Verify runs table populated
            table = screen.query_one(DataTable)
            assert table.row_count == 2

    asyncio.run(scenario())


def test_agents_screen_cancel_child():
    async def scenario():
        session = FakeSession()

        class Harness(App):
            def on_mount(self):
                self.screen_instance = AgentsScreen(session)
                self.push_screen(self.screen_instance)

        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_instance
            table = screen.query_one(DataTable)
            # First row is c1 (running)
            table.move_cursor(row=0, column=0)
            await pilot.press("x")
            await pilot.pause()
            assert "c1" in session.cancelled

            # Move to row 1 (c2, which is already finished ok)
            table.move_cursor(row=1, column=0)
            await pilot.press("x")
            await pilot.pause()
            # c2 was not cancelled
            assert "c2" not in session.cancelled

            # Esc pops screen
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == 1

    asyncio.run(scenario())


def test_the_card_shown_is_the_one_the_factory_read():
    """The fake above used to hand the screen a card object with string
    skills. Real info had no card at all (the factory dropped it), and a
    real RemoteCard has RemoteSkill objects, which "".join() refused."""
    from ovat.a2a.protocol import RemoteCard, RemoteSkill
    from ovat.agentic.session import describe_agent
    from ovat.config.workflow import WorkflowConfig
    card = RemoteCard("math-agent", "Solves math", "http://127.0.0.1:9100",
                      version="1.0.0", skills=(RemoteSkill("add", "adds"),
                                               RemoteSkill("mul", "times")))

    class Agent:
        tools = {"remote_math": {"kind": "a2a", "permission": "ask",
                                 "card": card},
                 "search_docs": {"kind": "builtin"}}
    info = describe_agent(WorkflowConfig(model={"name": "m"}), Agent(), "w.yml")
    session = FakeSession()
    session.info = info
    text = AgentsScreen(session)._subagents_text().plain
    assert "math-agent v1.0.0: add, mul" in text
    assert "card" not in info["tools"][1]
