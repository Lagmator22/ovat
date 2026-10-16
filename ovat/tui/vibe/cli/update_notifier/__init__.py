from __future__ import annotations

from ovat.tui.vibe.cli.update_notifier.adapters.filesystem_update_cache_repository import (
    FileSystemUpdateCacheRepository,
)
from ovat.tui.vibe.cli.update_notifier.adapters.github_update_gateway import GitHubUpdateGateway
from ovat.tui.vibe.cli.update_notifier.adapters.pypi_update_gateway import PyPIUpdateGateway
from ovat.tui.vibe.cli.update_notifier.adapters.uv_tool_update_gateway import UvToolUpdateGateway
from ovat.tui.vibe.cli.update_notifier.gateway_factory import create_update_gateway
from ovat.tui.vibe.cli.update_notifier.ports.update_cache_repository import (
    UpdateCache,
    UpdateCacheRepository,
)
from ovat.tui.vibe.cli.update_notifier.ports.update_gateway import (
    DEFAULT_GATEWAY_MESSAGES,
    Update,
    UpdateGateway,
    UpdateGatewayCause,
    UpdateGatewayError,
    UpdateGatewayUnavailableError,
)
from ovat.tui.vibe.cli.update_notifier.update import (
    UpdateAvailability,
    UpdateError,
    get_pending_update_from_cache,
    get_update_if_available,
    mark_update_as_dismissed,
)
from ovat.tui.vibe.cli.update_notifier.whats_new import (
    load_whats_new_content,
    mark_version_as_seen,
    should_show_whats_new,
)

__all__ = [
    "DEFAULT_GATEWAY_MESSAGES",
    "FileSystemUpdateCacheRepository",
    "GitHubUpdateGateway",
    "PyPIUpdateGateway",
    "Update",
    "UpdateAvailability",
    "UpdateCache",
    "UpdateCacheRepository",
    "UpdateError",
    "UpdateGateway",
    "UpdateGatewayCause",
    "UpdateGatewayError",
    "UpdateGatewayUnavailableError",
    "UvToolUpdateGateway",
    "create_update_gateway",
    "get_pending_update_from_cache",
    "get_update_if_available",
    "load_whats_new_content",
    "mark_update_as_dismissed",
    "mark_version_as_seen",
    "should_show_whats_new",
]
