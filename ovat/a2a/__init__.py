# ovat/a2a/__init__.py
"""Layer 6: Agent2Agent (A2A). Other agents can ask this one, and this one
can ask them.

The package is split so the SDK can change without the rest moving.
`a2a-sdk` shipped four releases in two weeks while this was designed, and
1.x already moved every type from pydantic to protobuf once.

    ovat/a2a/            OVAT's own A2A logic. NEVER imports `a2a`.
      protocol.py        task states, card and reply shapes, peer errors
      exposure.py        which tools a remote caller may trigger
      card.py            the Agent Card, as the plain JSON the spec defines
      policy.py          admission: size, queue, rate, hops, cycles
      tokens.py          bearer tokens and who they belong to
      audit.py           the log of every answer that left
      service.py         runs one remote task against a built agent
    ovat/a2a/sdk/        the ONLY code that imports `a2a` (the SDK)
      server.py          SDK executor + ASGI app around service.py
      client.py          a remote agent, behind protocol.py's shapes

In C++ terms: everything above sdk/ is written against an abstract
interface (service.Reporter, protocol.RemoteReply), and sdk/ is the one
concrete implementation. Upgrading the SDK, or the spec, is a change to
sdk/ alone; a test scans the source to keep it that way.

The SDK is the optional `[a2a]` extra. Nothing on the plain CLI path
imports sdk/, so `ovat --help` works without it.
"""
