# ovat/tools/web_fetch.py
"""The web_fetch tool: GET one web page and read it as text.

Mirrors Mistral Vibe's web_fetch.py (vibe/core/tools/builtins): http and
https only, a default timeout of 30 s (at most 120), a content cap of
120,000 bytes, HTML turned into readable markdown, and a redirect to
ANOTHER site is not followed (it needs its own approval, so the model is
told the new address and may call again).

Not copied: Vibe's httpx + markdownify. OVAT adds no new dependency for
this; urllib and html.parser from the standard library are enough for
"readable text of a page". The converter is plainer than markdownify's
(headings, links, lists, code, paragraphs; no tables).

Permission is `ask` by default (permissions.py): the URL and the request
leave the machine.
"""
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from ovat.tools.project import Shown, fail

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_CONTENT_BYTES = 120_000      # Vibe's max_content_bytes
MAX_REDIRECTS = 20
_VOID_SKIP = {"embed"}
# Read a little more than we keep: HTML shrinks a lot when turned to text.
MAX_DOWNLOAD_BYTES = 4_000_000
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

_SKIP = {"script", "style", "noscript", "iframe", "object", "embed", "svg",
         "template"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "nav",
          "ul", "ol", "table", "tr", "blockquote", "pre", "form", "figure",
          "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "hr"}


class _ToText(HTMLParser):
    """HTML to light markdown. Never raises on bad HTML."""

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._pre = 0
        self._links: list[str | None] = []

    def _break(self, n: int = 2) -> None:
        text = "".join(self.out)
        if not text or text.endswith("\n" * n):
            return
        self.out.append("\n" * (n - (len(text) - len(text.rstrip("\n")))))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            if tag not in _VOID_SKIP:
                self._skip += 1
            return
        if self._skip:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._break()
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self._break(1)
            self.out.append("- ")
        elif tag == "pre":
            self._break()
            self.out.append("```\n")
            self._pre += 1
        elif tag == "code" and not self._pre:
            self.out.append("`")
        elif tag == "a":
            href = (a.get("href") or "").strip()
            ok = href and not href.startswith(("#", "javascript:", "mailto:"))
            self._links.append(urljoin(self.base, href) if ok else None)
            if ok:
                self.out.append("[")
        elif tag in _BLOCK:
            self._break(1 if tag in ("br", "tr") else 2)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP:
            if tag not in _VOID_SKIP:
                self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._break(1)
            self.out.append("```")
            self._break()
        elif tag == "code" and not self._pre:
            self.out.append("`")
        elif tag == "a" and self._links:
            href = self._links.pop()
            if href:
                self.out.append(f"]({href})")
        elif tag in _BLOCK:
            self._break(2 if tag.startswith("h") or tag in (
                "p", "ul", "ol", "table", "blockquote") else 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        if self._pre:
            self.out.append(data)
            return
        text = re.sub(r"\s+", " ", data)
        if not text.strip() and (not self.out or self.out[-1].endswith(
                ("\n", " "))):
            return
        if self.out and self.out[-1].endswith(("\n", " ")):
            text = text.lstrip()
        self.out.append(text)


def html_to_text(html: str, base: str = "") -> str:
    parser = _ToText(base)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # A page the parser chokes on still gives what it read so far.
        pass
    text = "".join(parser.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if parser.title.strip() and not text.startswith("#"):
        text = f"# {' '.join(parser.title.split())}\n\n{text}".strip()
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hand every redirect back to us: the check on WHERE it goes is ours."""

    def redirect_request(self, *args, **kwargs):
        return None


def _origin(url: str) -> tuple:
    p = urlparse(url)
    return (p.scheme, (p.hostname or "").lower(),
            p.port or (443 if p.scheme == "https" else 80))


def _normalize(url: str) -> str:
    """Vibe's rule: no scheme means https."""
    url = url.strip()
    raw = url.lstrip("/") if url.startswith("//") else url
    return raw if raw.startswith(("http://", "https://")) else "https://" + raw


def _decode(data: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    for enc in ([match.group(1)] if match else []) + ["utf-8"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def web_fetch_impl(url: str, timeout: int | None = None) -> str:
    raw = (url or "").strip()
    if not raw:
        return fail("web_fetch", "the URL is empty.")
    try:
        scheme = urlparse(raw).scheme
    except ValueError:
        return fail("web_fetch", "the URL is invalid; use an http or https address.")
    if scheme and scheme not in ("http", "https") and "://" in raw:
        return fail("web_fetch", f"the URL scheme '{scheme}' is not allowed. "
                                 f"Use http or https.")
    try:
        seconds = int(timeout) if timeout else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT
    if seconds <= 0:
        return fail("web_fetch", "the timeout must be a positive number.")
    seconds = min(seconds, MAX_TIMEOUT)

    try:
        current = _normalize(raw)
        approved = _origin(current)
        if not urlparse(current).hostname:
            raise ValueError("missing host")
    except ValueError:
        return fail("web_fetch", "the URL is invalid; use an http or https address.")
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(current, headers={
            "User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;"
                      "q=0.9,*/*;q=0.8"})
        try:
            with opener.open(request, timeout=seconds) as response:
                body = response.read(MAX_DOWNLOAD_BYTES + 1)
                content_type = response.headers.get("Content-Type",
                                                    "text/plain")
                final = response.geturl()
            break
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            exc.close()
            if exc.code in (301, 302, 303, 307, 308) and location:
                try:
                    target = urljoin(current, location)
                    scheme = urlparse(target).scheme
                except ValueError:
                    return fail("web_fetch", "the page redirects to an invalid URL.")
                if scheme not in ("http", "https"):
                    return fail("web_fetch", f"the page redirects to {target}"
                                             f", which is not http or https.")
                try:
                    target_origin = _origin(target)
                except ValueError:
                    return fail("web_fetch", "the page redirects to an invalid URL.")
                if target_origin != approved:
                    return fail("web_fetch", (
                        f"the page redirects to another site: {target}. "
                        f"Call web_fetch again with that address if you "
                        f"want it."))
                current = target
                continue
            return fail("web_fetch", f"HTTP error {exc.code}: {exc.reason}")
        except TimeoutError:
            return fail("web_fetch", f"the request timed out after "
                                     f"{seconds} seconds.")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return fail("web_fetch", f"the request timed out after "
                                         f"{seconds} seconds.")
            return fail("web_fetch", f"could not fetch the URL: {exc.reason}")
        except (OSError, ValueError) as exc:
            return fail("web_fetch", f"could not fetch the URL: {exc}")
    else:
        return fail("web_fetch", f"too many redirects (more than "
                                 f"{MAX_REDIRECTS}).")

    # A NUL byte early on is the usual tell for "not text" (as in
    # read_file): checked on the bytes, since servers mislabel files.
    if b"\0" in body[:8192]:
        kind = content_type.split(";")[0].strip() or "unknown type"
        return fail("web_fetch", f"{final} is a binary file ({kind}), not a "
                                 f"page. web_fetch reads text only.")
    text = _decode(body[:MAX_DOWNLOAD_BYTES], content_type)
    if "html" in content_type.lower():
        text = html_to_text(text, final)
    encoded = text.encode("utf-8")
    cut = len(encoded) > MAX_CONTENT_BYTES or len(body) > MAX_DOWNLOAD_BYTES
    if cut:
        text = encoded[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
        text += "\n\n[Content truncated due to size limit]"
    return Shown(text, {"kind": "web_fetch", "url": final, "content": text,
                        "content_type": content_type, "was_truncated": cut})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch a web page (http or https) and return its "
                       "readable text. Use it to read documentation or a "
                       "page the user pointed you to.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string",
                        "description": "the address of the page"},
                "timeout": {"type": "integer",
                            "description": "seconds to wait (at most 120)"},
            },
            "required": ["url"],
        },
    },
}
