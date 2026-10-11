# ovat/a2a/sdk/__init__.py
"""The only OVAT code allowed to import the A2A SDK (`a2a-sdk`).

Nothing is imported here at module level, so `import ovat.a2a.sdk` is free
and require_sdk() can turn a missing extra into a sentence.
"""

INSTALL_HINT = 'A2A needs the extra: pip install "ovat[a2a]"'


def require_sdk() -> None:
    """Raise RuntimeError with the install command if the SDK is absent."""
    try:
        import a2a  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc
