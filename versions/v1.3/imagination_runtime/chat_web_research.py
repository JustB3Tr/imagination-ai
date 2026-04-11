"""
Optional web follow-ups for HTTP chat: model may end a reply with [[WEB_QUERY:...]]; we fetch
DDGS snippets and continue the conversation in-process (no extra user round-trip).

Disabled when IMAGINATION_CHAT_WEB=0. Max web fetches: IMAGINATION_CHAT_WEB_FETCHES (default 3).
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional

_WEB_OPEN = "[[WEB_QUERY:"
_WEB_RE = re.compile(r"\[\[WEB_QUERY:([^\]]+)\]\]\s*$", re.DOTALL)


def web_followups_enabled() -> bool:
    return (os.getenv("IMAGINATION_CHAT_WEB") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def max_web_fetches() -> int:
    raw = (os.getenv("IMAGINATION_CHAT_WEB_FETCHES") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(0, min(n, 8))


def inject_web_system_message(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    extra = (
        "Your general knowledge may be incomplete or outdated (training data roughly through Dec 2022). "
        "When the user asks about recent events, current figures, post-2022 facts, or anything you are "
        "not confident about, do not invent specifics. "
        "If you need up-to-date or verifiable information, after your reasoning add a single final line "
        "exactly in this form (nothing after it):\n"
        f"{_WEB_OPEN}your concise English search query]]\n"
        "Use that line only when a web lookup would materially help. If you can answer fully from memory, "
        "do not include it. Do not pretend you already searched; you will receive web excerpts in a "
        "follow-up message if you request them."
    )
    if not msgs:
        return [{"role": "system", "content": extra}]
    if msgs[0].get("role") == "system":
        merged = list(msgs)
        first = (merged[0].get("content") or "").strip()
        merged[0] = {**merged[0], "content": (extra + "\n\n" + first).strip()}
        return merged
    return [{"role": "system", "content": extra}] + list(msgs)


def strip_web_marker_for_stream(cumulative: str) -> str:
    """Hide partial or complete [[WEB_QUERY:...]] suffix from streamed display."""
    i = cumulative.find(_WEB_OPEN)
    if i == -1:
        return cumulative
    return cumulative[:i].rstrip()


def remove_web_marker(full: str) -> str:
    return _WEB_RE.sub("", (full or "").strip()).strip()


def extract_web_query(full: str) -> Optional[str]:
    m = _WEB_RE.search((full or "").strip())
    if not m:
        return None
    q = (m.group(1) or "").strip()
    return q or None


def fetch_web_context(query: str, *, max_results: int = 8) -> str:
    query = (query or "").strip()
    if not query:
        return "(empty search query)"
    lines: List[str] = []
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results), 1):
                title = (r.get("title") or "").strip()
                snippet = (r.get("body") or "").strip()
                url = (r.get("href") or "").strip()
                lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    except Exception as e:
        return f"(web search failed: {e})"
    if not lines:
        return "(no search results; try a shorter or broader query)"
    return "\n".join(lines)


BUDGET_EXHAUSTED_USER = (
    "Web search budget for this reply is exhausted. Do not request another [[WEB_QUERY:...]]. "
    "Summarize whatever was useful from prior web excerpts (if any), clearly state if there was "
    "not enough relevant or verified information for a complete answer, and do not fabricate facts."
)


def iterate_display_text_with_web(
    *,
    base_messages: List[Dict[str, str]],
    max_new_tokens: int,
    stream_native,
) -> Iterator[str]:
    """
    Yields cumulative **visible** assistant text for the UI (web marker stripped while streaming).

    ``stream_native(messages, max_new_tokens)`` must be the existing _stream_native iterator.
    """
    working = inject_web_system_message(base_messages)
    total_shown = ""
    fetches = 0
    max_f = max_web_fetches()

    while True:
        full_text = ""
        for chunk in stream_native(working, max_new_tokens):
            full_text = chunk
            disp = strip_web_marker_for_stream(full_text)
            yield total_shown + ("\n\n" if total_shown else "") + disp

        clean_assistant = remove_web_marker(full_text)
        query = extract_web_query(full_text)

        if not query:
            break

        if fetches >= max_f:
            working.append({"role": "assistant", "content": clean_assistant})
            working.append({"role": "user", "content": BUDGET_EXHAUSTED_USER})
            total_shown = total_shown + ("\n\n" if total_shown else "") + clean_assistant
            full_text2 = ""
            for chunk in stream_native(working, max_new_tokens):
                full_text2 = chunk
                disp2 = strip_web_marker_for_stream(full_text2)
                yield total_shown + ("\n\n" if total_shown else "") + disp2
            break

        fetches += 1
        ctx = fetch_web_context(query)
        pack = (
            f"Web search results for query: {query}\n\n{ctx}\n\n"
            "Use the above to answer accurately. If excerpts conflict or are insufficient, say so."
        )
        working.append({"role": "assistant", "content": clean_assistant})
        working.append({"role": "user", "content": pack})
        total_shown = total_shown + ("\n\n" if total_shown else "") + clean_assistant


def final_text_from_stream(gen: Iterator[str]) -> str:
    last = ""
    for s in gen:
        last = s
    return last
