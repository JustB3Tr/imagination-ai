"""
Thinking Path builder — generates step-by-step reasoning markdown from
search results and conversation context for display in the UI and injection into prompts.
"""
from __future__ import annotations

from typing import Any, Dict, List


def build_thinking_path(
    *,
    use_web: bool,
    reason: str,
    query: str,
    source_cards: List[Dict[str, Any]],
    conversation_turns: int,
) -> str:
    """
    Build a markdown string describing the AI's reasoning path.
    Used when web search is triggered to explain sources and synthesis.
    """
    lines: List[str] = ["### Thinking Path", ""]

    if use_web:
        lines.append(f"**Step 1 — Decision**  \nDecided to search the web because: {reason}")
        lines.append("")
        lines.append(f"**Step 2 — Search**  \nQuery: `{query}`  \nFound {len(source_cards)} sources. Evaluating relevance...")
        lines.append("")

        if source_cards:
            lines.append("**Step 3 — Source evaluation**")
            for s in source_cards:
                idx = s.get("idx", "?")
                title = s.get("title") or "Untitled"
                domain = s.get("domain") or ""
                trust_label = "trusted" if s.get("trusted") else "other"
                snippet = (s.get("snippet") or "")[:200]
                relevance_note = f"Relevance: {snippet}..." if snippet else "No excerpt available."
                lines.append(f"- **[{idx}]** {title} (`{domain}`) — *{trust_label}*  \n  {relevance_note}")
            lines.append("")
            lines.append("**Step 4 — Synthesis**  \nSynthesizing answer using the sources above and conversation context.")
        else:
            lines.append("**Step 3**  \nNo useful sources found.")
            lines.append("")
            lines.append("**Step 4**  \nAnswering from model knowledge and conversation context.")
    else:
        lines.append("**Route**  \nAnswering from model knowledge and conversation context (no web search).")
        if conversation_turns > 0:
            lines.append(f"  \nUsing {conversation_turns} prior turn(s) for context.")

    return "\n".join(lines)


def build_thinking_path_no_web(conversation_turns: int = 0) -> str:
    """Short thinking path when no web search is used."""
    lines = ["### Thinking Path", "", "**Route**  \nAnswering from model knowledge and conversation context."]
    if conversation_turns > 0:
        lines.append(f"  \nUsing {conversation_turns} prior turn(s) for context.")
    return "\n".join(lines)
