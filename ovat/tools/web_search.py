# ovat/tools/web_search.py
"""The web_search tool: a web search through the provider the workflow names.

Vibe's web_search.py talks to Mistral's own service; OVAT has no such
service, so this takes the provider from the tool's entry in workflow.yml:

    - name: web_search
      provider: searxng        # searxng | tavily | serper | brave
      base_url: http://localhost:8080   # searxng only: your instance
      api_key_env: TAVILY_API_KEY       # NAME of the variable, never the key

The key is read from the environment at call time and is never written to a
result, an error or the trace. Permission is `ask` by default: the query
leaves the machine.

Provider request and result fields checked on 2026-10-16:
- SearXNG: https://docs.searxng.org/dev/search_api.html and
  https://docs.searxng.org/dev/result_types/main/mainresult.html
  GET /search with q and format=json; JSON needs search.formats enabled.
- Tavily: https://docs.tavily.com/documentation/api-reference/endpoint/search
  POST /search with Bearer authorization, query, max_results, include_answer.
- Serper: https://serper.dev/playground, including its official page source
  /_next/static/chunks/pages/playground-7ce8960e9fe2fc99.js:
  POST https://google.serper.dev/search, X-API-KEY, q and num.
  https://serper.dev/ shows organic title, link and snippet fields.
- Brave: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
  GET /res/v1/web/search, X-Subscription-Token, q and count;
  web.results includes title, url and description.

"""
import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode, urlparse

from ovat.tools.project import Shown, fail

PROVIDERS = ("searxng", "tavily", "serper", "brave")
MAX_RESULTS = 5
TIMEOUT = 20
SNIPPET_CHARS = 500
MAX_RESPONSE_BYTES = 2_000_000
_USER_AGENT = "ovat-web-search"

_KEY_HINT = {"tavily": "TAVILY_API_KEY", "serper": "SERPER_API_KEY",
             "brave": "BRAVE_API_KEY"}
_SETUP = ("give the web_search tool a provider in workflow.yml "
          "(searxng, tavily, serper or brave); web_fetch works without a "
          "key if you know the page's address.")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect must never carry a provider's API key to another host.
    def redirect_request(self, *args, **kwargs):
        return None


def _request(method: str, url: str, headers: dict, body: dict | None) -> dict:
    """One HTTP call returning parsed JSON. The one seam tests replace."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={
        "User-Agent": _USER_AGENT, "Accept": "application/json",
        **({"Content-Type": "application/json"} if body is not None else {}),
        **headers})
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=TIMEOUT) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response is too large")
    result = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(result, dict):
        raise ValueError("response is not an object")
    return result


def _setup_error(provider: str | None, base_url: str | None,
                 api_key_env: str | None) -> tuple[str | None, str | None]:
    """(the key or "", None) when ready, or (None, the one-sentence fix)."""
    if not provider:
        return None, f"web_search is not set up; {_SETUP}"
    if provider not in PROVIDERS:
        return None, (f"web_search has an unknown provider '{provider}'. "
                      f"Use one of: {', '.join(PROVIDERS)}.")
    if provider == "searxng":
        if not base_url:
            return None, ("web_search uses searxng but has no base_url. Set "
                          "base_url to your SearXNG instance, for example "
                          "http://localhost:8080. web_fetch works without a "
                          "key.")
        try:
            parsed = urlparse(base_url)
            valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
            parsed.port             # rejects an invalid port before the request
        except ValueError:
            valid = False
        if not valid:
            return None, (f"web_search base_url '{base_url}' must start with "
                          f"http:// or https://.")
        from ovat.tools.web_fetch import check_safe_url
        blocked = check_safe_url(base_url, allow_private_lan=True)
        if blocked:
            return None, (f"web_search base_url '{base_url}' is not allowed: "
                          f"{blocked}")
        return "", None            # an instance of your own needs no key
    if not api_key_env:
        return None, (f"web_search uses {provider}, which needs an API key. "
                      f"Set api_key_env to the NAME of the environment "
                      f"variable that holds it (for example "
                      f"{_KEY_HINT[provider]}). web_fetch works without a "
                      f"key.")
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        return None, (f"web_search needs the environment variable "
                      f"{api_key_env} (your {provider} API key), and it is "
                      f"empty or not set. Set it and start again. web_fetch "
                      f"works without a key.")
    return key, None


def _call(provider: str, query: str, base_url: str | None, key: str,
          count: int) -> tuple[str, list[dict]]:
    """(answer, [{title, url, snippet}]) from one provider."""
    if provider == "searxng":
        url = (f"{base_url.rstrip('/')}/search?"
               f"{urlencode({'q': query, 'format': 'json'})}")
        data = _request("GET", url, {}, None)
        rows = [(r.get("title"), r.get("url"), r.get("content"))
                for r in data.get("results") or []]
        answer = ""
    elif provider == "tavily":
        data = _request("POST", "https://api.tavily.com/search",
                        {"Authorization": f"Bearer {key}"},
                        {"query": query, "max_results": count,
                         "include_answer": True})
        answer = data.get("answer") or ""
        rows = [(r.get("title"), r.get("url"), r.get("content"))
                for r in data.get("results") or []]
    elif provider == "serper":
        data = _request("POST", "https://google.serper.dev/search",
                        {"X-API-KEY": key}, {"q": query, "num": count})
        box = data.get("answerBox") or {}
        answer = box.get("answer") or box.get("snippet") or ""
        rows = [(r.get("title"), r.get("link"), r.get("snippet"))
                for r in data.get("organic") or []]
    else:  # brave
        url = ("https://api.search.brave.com/res/v1/web/search?"
               f"q={quote(query)}&count={count}")
        data = _request("GET", url, {"X-Subscription-Token": key}, None)
        answer = ""
        rows = [(r.get("title"), r.get("url"), r.get("description"))
                for r in (data.get("web") or {}).get("results") or []]
    sources = [{"title": str(t or u), "url": str(u),
                "snippet": " ".join(str(s or "").split())[:SNIPPET_CHARS]}
               for t, u, s in rows if u][:count]
    return str(answer or ""), sources


def web_search_impl(query: str, provider: str | None = None,
                    base_url: str | None = None,
                    api_key_env: str | None = None) -> str:
    query = (query or "").strip()
    if not query:
        return fail("web_search", "the query is empty.")
    key, why = _setup_error(provider, base_url, api_key_env)
    if why:
        return fail("web_search", why)
    try:
        answer, sources = _call(provider, query, base_url, key, MAX_RESULTS)
    except urllib.error.HTTPError as exc:
        exc.close()
        hint = ""
        if provider == "searxng" and exc.code == 403:
            hint = (" The instance may not allow format=json; enable it "
                    "under search.formats in its settings.")
        elif exc.code in (401, 403):
            hint = f" Check the key in {api_key_env}."
        return fail("web_search", f"{provider} answered HTTP {exc.code}."
                                  f"{hint}")
    except (urllib.error.URLError, TimeoutError, OSError):
        return fail("web_search", f"could not reach {provider}; check the "
                                  "connection and provider settings.")
    except (ValueError, AttributeError, TypeError):
        return fail("web_search", f"{provider} sent an answer that is not "
                                  f"readable JSON, so it could not be used.")
    lines = []
    if answer:
        lines.append(f"Answer: {answer}\n")
    for n, s in enumerate(sources, start=1):
        lines.append(f"{n}. {s['title']}\n   {s['url']}\n   {s['snippet']}"
                     .rstrip())
    text = "\n".join(lines) or f"No results for: {query}"
    return Shown(text, {
        "kind": "web_search", "query": query, "answer": answer or text,
        "sources": [{"title": s["title"], "url": s["url"]}
                    for s in sources]})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return the top results (title, "
                       "address and a short excerpt). Use web_fetch to read "
                       "a result in full.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "what to search for"},
            },
            "required": ["query"],
        },
    },
}
