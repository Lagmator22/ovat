# ovat/cli/commands.py
"""OVAT's entries in the Textual command palette (Ctrl-P / Alt-P).

The palette is Textual's own fuzzy command launcher; this adds OVAT's actions
to it. It is deliberately the home for things that do not deserve a slash
command: they are occasional, they are not part of any one screen's job, and
they apply everywhere in the app.

Two today:
  * Save a screenshot. Textual renders the live screen to SVG, so the result
    is real selectable text and vector box-drawing, not a bitmap of a
    terminal. That is what makes it useful in a report or a PR.
  * Switch the theme. Textual ships ~20; they change the app chrome, while
    OVAT's own brand colours (ui.PALETTE) stay put, because those carry
    meaning: green is a passing check, red is a failing one, and a theme must
    not be able to repaint that.

Kept in its own module rather than in tui.py so the palette can grow without
the launcher growing with it, and so importing it costs nothing at CLI time.
"""
from textual.command import DiscoveryHit, Hit, Hits, Provider


class OvatCommands(Provider):
    """OVAT's own palette entries, alongside Textual's system commands."""

    @property
    def _entries(self) -> list:
        """(title, help, callback) for everything this provider offers."""
        app = self.app
        entries = [
            ("Save screenshot",
             "write the current screen to an SVG file",
             app.action_save_ovat_screenshot),
        ]
        # One entry per installed theme, so the palette's own fuzzy search does
        # the filtering: typing "drac" finds Dracula without a submenu.
        for name in sorted(app.available_themes):
            entries.append((
                f"Theme: {name}",
                "switch the app theme (OVAT's status colours are unaffected)",
                _theme_setter(app, name),
            ))
        return entries

    async def discover(self) -> Hits:
        """What the palette lists before anything is typed."""
        # Only the screenshot: twenty themes would bury Textual's own commands
        # in the default list. They are all still one keystroke away in search.
        for title, help_text, callback in self._entries[:1]:
            yield DiscoveryHit(title, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, callback in self._entries:
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title), callback,
                          help=help_text)


def _theme_setter(app, name: str):
    """Bind one theme name to a zero-argument callback.

    A default argument, not a closure over the loop variable: closing over
    `name` would give every entry the LAST theme in the list, the classic
    late-binding trap (the same one factory._make_mcp_caller avoids).
    """
    def apply(_name: str = name) -> None:
        app.theme = _name
    return apply
