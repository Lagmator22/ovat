# Changed by OVAT from Mistral Vibe (commit 4ae5c59). See NOTICE.
from __future__ import annotations

from ovat.tui.vibe.app_server.models import AccountView


def plan_title(account: AccountView | None) -> str | None:
    if account is None or account.plan is None:
        return None
    return account.plan.title
