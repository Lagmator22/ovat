# ovat/tui/__init__.py
"""The agent TUI (`ovat agent`): chat-first, live tools, subagents, A2A.

A front end for ovat/agentic/session.AgentSession and nothing more: every
rule (permissions, subagents, cancel, shell) lives in the session, and this
package only draws what the session reports. Needs the [tui] extra. The
older launcher (`ovat tui`) is untouched and still the default for bare
`ovat`.

Design reference: Mistral Vibe's TUI (Apache-2.0), studied for its layout
and behaviour (banner in the transcript, a loading line above a borderless
input, folding tool summaries, an approval panel in place of the input).
The code here is OVAT's own.
"""
