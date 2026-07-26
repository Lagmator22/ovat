# ovat/cli/commands.py
"""Shared plumbing for OVAT's command-palette providers.

Textual offers two ways in, and OVAT uses both for the reasons the docs give:

  * App.get_system_commands, for the handful of app-wide entries. That lives
    in tui.py next to the actions it exposes, because it is a few lines.
  * A command.Provider on a SCREEN, via its COMMANDS class var, for entries
    that only mean something while that screen is up. "Copy the last answer"
    has no meaning on the launcher, and listing it there is noise.

This module holds only the base class the screen providers share, so each
screen declares WHAT it offers and none of them repeat the search/discover
boilerplate.

Deliberately NOT here any more: theme switching and screenshots. Textual
already ships both as system commands ("Theme" opens a live-previewing
picker, "Screenshot" saves an SVG), and the first version of this file
reimplemented them, which put two palette entries in front of the user for
one job and gave the worse one first.
"""
from textual.command import DiscoveryHit, Hit, Hits, Provider


class ScreenCommands(Provider):
    """Base for one screen's palette entries.

    A subclass implements commands() and gets fuzzy search plus the
    empty-input discovery list for free. Entries are rebuilt on every call
    rather than cached, so a command whose wording depends on state (the
    reasoning toggle, say) always describes what it will actually do.
    """

    def commands(self) -> list:
        """(title, help, callback) for everything this screen offers."""
        raise NotImplementedError

    async def discover(self) -> Hits:
        """What the palette lists before anything is typed."""
        for title, help_text, callback in self.commands():
            yield DiscoveryHit(title, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, callback in self.commands():
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title), callback,
                          help=help_text)
