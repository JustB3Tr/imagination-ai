from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def trusted(url: str, trusted_domains: List[str]) -> bool:
    d = get_domain(url)
    return any(d.endswith(x) for x in trusted_domains)


def web_search(
    query: str,
    *,
    max_results: int,
    max_sources: int,
    trusted_domains: List[str],
    cache: Dict[str, List[Dict]],
) -> List[Dict]:
    key = query.strip().lower()
    if key in cache:
        return cache[key]

    results: List[Dict] = []
    seen = set()
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = (r.get("href") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or "").strip()
            results.append(
                {
                    "url": url,
                    "title": title or get_domain(url) or "Untitled",
                    "snippet": snippet,
                    "trusted": trusted(url, trusted_domains),
                    "domain": get_domain(url),
                }
            )

    results.sort(key=lambda x: (not x["trusted"], x["domain"], x["title"]))
    trimmed = results[:max_sources]
    cache[key] = trimmed
    return trimmed


def fetch_page(
    url: str,
    *,
    timeout_s: int,
    max_chars: int,
    cache: Dict[str, str],
) -> str:
    if url in cache:
        return cache[url]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()

        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            cache[url] = ""
            return ""

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]
        ):
            tag.decompose()

        chunks: List[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                chunks.append(txt)

        text = " ".join(chunks)
        text = re.sub(r"\s+", " ", text).strip()[:max_chars]
        cache[url] = text
        return text
    except Exception:
        cache[url] = ""
        return ""


def fetch_parallel(urls: List[str], *, timeout_s: int, max_chars: int, cache: Dict[str, str]) -> List[str]:
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as exe:
        return list(
            exe.map(lambda u: fetch_page(u, timeout_s=timeout_s, max_chars=max_chars, cache=cache), urls)
        )


def sources_to_cards(sources: List[Dict]) -> str:
    if not sources:
        return "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>"

    cards: List[str] = []
    for s in sources:
        trust_badge = "TRUSTED" if s.get("trusted") else "OTHER"
        title = html.escape(s.get("title") or "")
        url = html.escape(s.get("url") or "")
        domain = html.escape(s.get("domain") or "")
        snippet = html.escape((s.get("snippet") or "")[:180])
        idx = html.escape(str(s.get("idx") or ""))

        cards.append(
            f"""
        <a class="source-card" href="{url}" target="_blank" rel="noopener">
          <div class="source-row">
            <div class="source-badge">[{idx}]</div>
            <div class="trust-pill {'trust-on' if s.get('trusted') else 'trust-off'}">{trust_badge}</div>
          </div>
          <div class="source-title">{title}</div>
          <div class="source-domain">{domain}</div>
          <div class="source-snippet">{snippet}</div>
        </a>
        """
        )

    return "<div class='sources-grid'>" + "\n".join(cards) + "</div>"
