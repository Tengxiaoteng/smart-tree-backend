from __future__ import annotations

import re
from urllib.parse import urlparse


_FILE_PROXY_RE = re.compile(r"^/api/files/([0-9a-fA-F-]{36})$")
_FILE_PROXY_IN_TEXT_RE = re.compile(r"/api/files/([0-9a-fA-F-]{36})")
_HTTP_URL_RE = re.compile(r"https?://[^\s)\"'>]+")


def extract_file_id_from_proxy_url(url: str | None) -> str | None:
    """
    Extract file_id from a proxy URL like:
      - /api/files/<uuid>
      - https://example.com/api/files/<uuid>
    """
    if not url:
        return None
    raw = str(url).strip()
    if not raw:
        return None

    path = raw
    if raw.startswith(("http://", "https://")):
        path = urlparse(raw).path or ""

    match = _FILE_PROXY_RE.match(path)
    if not match:
        return None
    return match.group(1)


def extract_proxy_file_ids_from_text(text: str | None) -> set[str]:
    if not text:
        return set()
    return {m.group(1) for m in _FILE_PROXY_IN_TEXT_RE.finditer(text)}


def extract_http_urls_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    # Best-effort extraction for markdown/html content.
    urls = [m.group(0) for m in _HTTP_URL_RE.finditer(text)]
    return [u.rstrip(".,;:!?)]}\"'") for u in urls]
