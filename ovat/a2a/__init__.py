# ovat/a2a/__init__.py
"""Agent2Agent (A2A) protocol support for remote agent calls and serving.

The package isolates ``a2a-sdk`` behind OVAT-owned protocol and service types
so SDK updates stay localized:

    ovat/a2a/            OVAT A2A logic (never imports `a2a`).
      protocol.py        task states, card and reply shapes, peer errors
      exposure.py        which tools a remote caller may trigger
      card.py            Agent Card JSON generation
      policy.py          admission: size, queue, rate, hops, cycles
      tokens.py          bearer tokens and hashes
      audit.py           audit log for outgoing responses
      service.py         executes one remote task against a built agent
    ovat/a2a/sdk/        the only package that imports `a2a`
      server.py          SDK executor and ASGI app around service.py
      client.py          remote agent client implementing protocol shapes

Everything above ``sdk/`` programs against the abstract interfaces
(``service.Reporter``, ``protocol.RemoteReply``), and ``sdk/`` adapts them to
the external SDK. Source isolation is enforced by ``tests/test_a2a_isolation.py``.

``a2a-sdk`` is an optional ``[a2a]`` extra; standard CLI entry points never
import ``sdk/``.
"""
