# ============================================================
# IMAGINATION v1.2 — Sleek minimal UI, model lifecycle, cloaking
#
# - Minimal dark UI: tab bar, chat, status, progress bar
# - No exposed model settings (root from env only)
# - Model lifecycle: load on demand, auto-unload after timeout
# - Response cloaking: module output rewritten by main model
# - Internal instructions + topic memory (invisible to user)
# ============================================================

from __future__ import annotations

import os
import re
import time
import warnings
from collections import OrderedDict
from threading import Lock, Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextIteratorStreamer,
)

from imagination_runtime.cache import RuntimeCache
from imagination_runtime.cloaker import build_cloak_messages
from imagination_runtime.internal_prompt import inject_internal_instructions
from imagination_runtime.model_lifecycle import (
    ModelLifecycle,
    KEY_TO_LABEL,
)
from imagination_runtime.paths import ModelPaths, resolve_root_path
from imagination_runtime.registry import TaskId, get_task_specs
from imagination_runtime.thinking import (
    build_thinking_path,
    build_thinking_path_no_web,
    build_thinking_html_open,
    build_thinking_html_collapsed,
    compose_assistant_display,
    friendly_web_decision_reason,
)
from imagination_runtime.users import (
    load_global_memory,
    load_user_memory,
    load_trusted,
    load_relevant_topics,
    save_topic_memory,
)
from imagination_runtime.training_log import append_main_model_training_turn
from imagination_runtime.web import fetch_page, get_domain, refine_search_query, sources_to_cards, web_search

warnings.filterwarnings("ignore")

SEARCH_RESULTS = 6
MAX_SOURCES = 3
MAX_CHARS_PER_SOURCE = 900
REQUEST_TIMEOUT_S = 8
# Main chat: high ceiling so replies usually finish on EOS, not a tiny max_new_tokens cap.
# Hard limits are still the model context (prompt + completion) and the ceiling below.
_MAIN_MAX_NEW_TOKENS_DEFAULT = 16384
_MAIN_MAX_NEW_TOKENS_CEILING = 131_072
# Coder/research models need a much higher cap or long programs truncate mid-token (e.g. Snake game).
# Override coder cap (trusted / private setups): IMAGINATION_CODER_MAX_NEW_TOKENS=100000
# Hard cap 250_000 so a typo does not hang forever; the real limit is the model context window.
_CODER_MAX_NEW_TOKENS_CEILING = 250_000
CODER_MAX_NEW_TOKENS = 16384
MODULE_REASONING_MAX_NEW_TOKENS = 8192
_REASONING_MAX_NEW_TOKENS_CEILING = 131_072
SHOW_TYPING_CURSOR = True
TYPING_CURSOR = "▌"

AUTO_WEB_PATTERNS = [
    r"\blatest\b", r"\bcurrent\b", r"\btoday\b", r"\bnow\b", r"\brecent\b", r"\bnews\b",
    r"\bupdate\b", r"\bupdated\b", r"\brelease date\b", r"\bprice\b", r"\bcost\b",
    r"\bhow much\b", r"\bthis week\b", r"\bthis month\b", r"\bthis year\b",
    r"\bwho won\b", r"\branking\b", r"\bstandings\b", r"\bscores?\b",
]

TRUSTED_DOMAINS_STARTER = [
    "reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org",
    "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
    "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov", "nasa.gov", "noaa.gov", "usgs.gov",
]


class LRUCache(OrderedDict):
    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max_items

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            self.popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
        return super().get(key, default)


SEARCH_CACHE = LRUCache(64)
PAGE_CACHE = LRUCache(128)
RUNTIME = RuntimeCache()
GEN_LOCKS = {"main": Lock(), "cad_coder": Lock(), "reasoning_llm": Lock()}
LIFECYCLE = ModelLifecycle(unload_callback=RUNTIME.unload_module)


def _env_positive_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return max(minimum, min(v, maximum))
    except ValueError:
        return default


def _effective_main_max_new_tokens() -> int:
    """Main Imagination tab; model stops early when it emits EOS—this is only an upper bound."""
    return _env_positive_int(
        "IMAGINATION_MAIN_MAX_NEW_TOKENS",
        _MAIN_MAX_NEW_TOKENS_DEFAULT,
        minimum=256,
        maximum=_MAIN_MAX_NEW_TOKENS_CEILING,
    )


def _effective_reasoning_max_new_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_REASONING_MAX_NEW_TOKENS",
        MODULE_REASONING_MAX_NEW_TOKENS,
        minimum=512,
        maximum=_REASONING_MAX_NEW_TOKENS_CEILING,
    )


def _effective_coder_max_new_tokens() -> int:
    return _env_positive_int(
        "IMAGINATION_CODER_MAX_NEW_TOKENS",
        CODER_MAX_NEW_TOKENS,
        minimum=360,
        maximum=_CODER_MAX_NEW_TOKENS_CEILING,
    )


def _coder_skip_cloak_for_long_output() -> bool:
    """If set, Coder streams raw output (no main-model rewrite). Use with very high token limits."""
    return (os.getenv("IMAGINATION_CODER_SKIP_CLOAK") or "").strip().lower() in ("1", "true", "yes")


# Greedy + low penalty lets coder models repeat the same block (e.g. radius += 25) until max_new_tokens.
_CODER_REPETITION_PENALTY_DEFAULT = 1.22


def _coder_extra_generate_kwargs() -> Dict[str, Any]:
    """Stronger anti-loop settings for the coder module (optional n-gram ban via env)."""
    rp = _CODER_REPETITION_PENALTY_DEFAULT
    raw = (os.getenv("IMAGINATION_CODER_REPETITION_PENALTY") or "").strip()
    if raw:
        try:
            rp = float(raw)
        except ValueError:
            pass
    rp = max(1.0, min(rp, 2.0))
    kw: Dict[str, Any] = {"repetition_penalty": rp}
    ng = (os.getenv("IMAGINATION_CODER_NO_REPEAT_NGRAM") or "").strip()
    if ng and ng.lower() not in ("0", "none", "off"):
        try:
            n = int(ng)
            if n >= 2:
                kw["no_repeat_ngram_size"] = min(n, 128)
        except ValueError:
            pass
    return kw


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def render_live_text(text: str) -> str:
    text = clean_model_text(text)
    return (text + TYPING_CURSOR if text else TYPING_CURSOR) if SHOW_TYPING_CURSOR else text


def should_auto_web(text: str, force_web: bool = False) -> Tuple[bool, str]:
    if force_web:
        return True, "manual override"
    t = (text or "").lower().strip()
    for p in AUTO_WEB_PATTERNS:
        if re.search(p, t):
            return True, f"matched pattern: {p}"
    return False, "no web trigger"


def load_main_model(root_path: str) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(root_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(root_path, device_map="auto", torch_dtype="auto")
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return tok, model


def preload_main_model(root_path: str | None = None) -> None:
    """
    Load the root LLM into RUNTIME before serving chat.
    Call from app.py (or colab_setup --launch) so the first user message is not blocked by weight load.
    """
    root = resolve_root_path(root_path)
    paths = ModelPaths(root=root)
    if RUNTIME.main_tokenizer is not None and RUNTIME.main_model is not None:
        print("[imagination] Main model already loaded.", flush=True)
        return
    print(f"[imagination] Loading main model from: {paths.main_llm}", flush=True)
    tok, mdl = load_main_model(paths.main_llm)
    RUNTIME.main_tokenizer = tok
    RUNTIME.main_model = mdl
    print("[imagination] Main model ready.", flush=True)


def _load_cad_coder(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.cad_coder, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(paths.cad_coder, device_map="auto", torch_dtype="auto")
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_reasoning_llm(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.reasoning_llm, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(paths.reasoning_llm, device_map="auto", torch_dtype="auto")
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_embeddings(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.embeddings, use_fast=True)
    model = AutoModel.from_pretrained(paths.embeddings, torch_dtype="auto")
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_reranker(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.reranker, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(paths.reranker, device_map="auto", torch_dtype="auto")
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_tiny_sd(paths: ModelPaths):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(paths.tiny_sd, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return {"pipeline": pipe}


LOADERS: Dict[str, Callable[[ModelPaths], Any]] = {
    "cad_coder": _load_cad_coder,
    "reasoning_llm": _load_reasoning_llm,
    "embeddings": _load_embeddings,
    "reranker": _load_reranker,
    "tiny_sd": _load_tiny_sd,
}


def _get_device_for_model(model_obj: Any) -> torch.device:
    if hasattr(model_obj, "hf_device_map") and getattr(model_obj, "hf_device_map", None):
        for _, dev in model_obj.hf_device_map.items():
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
    return next(model_obj.parameters()).device


def generate_stream_chat(
    *,
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    lock: Lock,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
) -> Iterable[str]:
    device = _get_device_for_model(model)
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    # Longer generations need a looser inter-chunk timeout; model still stops on EOS before max_new_tokens.
    _st = max(120.0, min(900.0, 60.0 + max_new_tokens * 0.03))
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=_st)
    gen_error = {"exc": None}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.08,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if extra_generate_kwargs:
        gen_kwargs.update({k: v for k, v in extra_generate_kwargs.items() if v is not None})

    def _run():
        try:
            with lock, torch.inference_mode():
                model.generate(
                    **model_inputs,
                    streamer=streamer,
                    **gen_kwargs,
                )
        except Exception as e:
            gen_error["exc"] = e

    t = Thread(target=_run, daemon=True)
    t.start()
    partial = ""
    for chunk in streamer:
        if gen_error["exc"] is not None:
            raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error["exc"]
        partial += chunk
        yield clean_model_text(partial)
    t.join(timeout=0.2)
    if gen_error["exc"] is not None:
        raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error["exc"]


def generate_full(
    tokenizer: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    lock: Lock,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    result = ""
    for chunk in generate_stream_chat(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=max_new_tokens,
        lock=lock,
        extra_generate_kwargs=extra_generate_kwargs,
    ):
        result = chunk
    return result


def build_system_prompt(
    root_path: str,
    global_memory: str,
    user_memory: str,
    topic_memory: str,
    retrieved_notes: str,
    thinking_path: str,
) -> str:
    base = """You are Imagination, a fast, factual assistant.

Rules:
- Answer directly in natural prose.
- Keep answers concise unless the user asks for detail.
- If retrieved web notes are present, use them for accuracy; do not quote long passages.
- Do not add a "Sources used" section, numbered source lists, or inline [1]/[2] citations—the UI shows sources separately.
- If evidence conflicts, say so briefly.
- Do not invent facts or URLs.
""".strip()

    if global_memory.strip():
        base += "\n\nGlobal behavior memory (always follow):\n" + global_memory.strip()
    if user_memory.strip():
        base += "\n\nUser-specific memory:\n" + user_memory.strip()
    if topic_memory.strip():
        base += "\n\n" + topic_memory.strip()
    if thinking_path.strip():
        base += "\n\n" + thinking_path.strip()
    if retrieved_notes.strip():
        base += (
            "\n\nRetrieved web notes (for your reasoning only; do not recite this structure or URLs in your reply):\n\n"
            f"{retrieved_notes.strip()}"
        )
    return inject_internal_instructions(base, root_path)


def build_messages(
    conversation_state: List[Dict[str, str]],
    user_msg: str,
    system_prompt: str,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for m in conversation_state or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg.strip()})
    return messages


def assemble_retrieval_from_results(
    results: List[Dict[str, Any]],
    texts: List[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build retrieved_notes string and source_cards from search results + fetched page texts."""
    source_cards: List[Dict[str, Any]] = []
    retrieved_docs: List[str] = []

    for i, r in enumerate(results):
        content = re.sub(r"\s+", " ", ((texts[i] if i < len(texts) else "") or r.get("snippet") or "")).strip()
        source_cards.append({
            "idx": str(i + 1),
            "title": r.get("title"),
            "url": r.get("url"),
            "domain": r.get("domain"),
            "trusted": bool(r.get("trusted")),
            "snippet": content[:220],
        })
        trust_label = "trusted" if r.get("trusted") else "other"
        retrieved_docs.append(
            f"[{i+1}] {r.get('title')}\nURL: {r.get('url')}\nDomain: {r.get('domain')}\nTrust: {trust_label}\nNotes: {content[:MAX_CHARS_PER_SOURCE]}"
        )

    return "\n\n".join(retrieved_docs), source_cards


def build_retrieval_bundle(
    question: str,
    trusted_domains: List[str],
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    """Batch retrieval (no per-domain UI progress). Kept for tests or tooling."""
    from imagination_runtime.web import fetch_parallel

    results = web_search(
        question,
        max_results=SEARCH_RESULTS,
        max_sources=MAX_SOURCES,
        trusted_domains=trusted_domains,
        cache=SEARCH_CACHE,
    )
    urls = [r["url"] for r in results]
    texts = fetch_parallel(urls, timeout_s=REQUEST_TIMEOUT_S, max_chars=MAX_CHARS_PER_SOURCE, cache=PAGE_CACHE)
    retrieved_notes, source_cards = assemble_retrieval_from_results(results, texts)
    trace_lines: List[str] = []
    for i, r in enumerate(results):
        content = re.sub(r"\s+", " ", ((texts[i] if i < len(texts) else "") or r.get("snippet") or "")).strip()
        trust_label = "trusted" if r.get("trusted") else "other"
        trace_lines.append(
            f"{i+1}. **{r.get('title')}**  \n   - domain: `{r.get('domain')}`  \n   - trust: **{trust_label}**  \n   - url: {r.get('url')}  \n   - snippet: {content[:220] or '(no text extracted)'}"
        )
    trace_md = "### Trace\n\n- Route: **web + model**\n" + f"- Query: `{question}`\n" + f"- Sources kept: **{len(source_cards)}**\n\n"
    trace_md += "### Source picks\n\n" + ("\n\n".join(trace_lines) if trace_lines else "_No useful sources found._")
    return retrieved_notes, sources_to_cards(source_cards), trace_md, source_cards


def _ensure_modules_loaded(
    task_id: TaskId,
    root_path: str,
) -> None:
    paths = ModelPaths(root=root_path)
    keys = LIFECYCLE.get_keys_for_task(task_id)
    keys_to_load = [k for k in keys if RUNTIME.get(k) is None]
    for key in keys_to_load:
        RUNTIME.set(key, LOADERS[key](paths))


def chat_submit(
    task_label: str,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    display_state: List[Dict[str, str]],
):
    root_path = resolve_root_path(None)
    paths = ModelPaths(root=root_path)
    trusted_domains = load_trusted(root_path, TRUSTED_DOMAINS_STARTER)

    user_msg = (user_msg or "").strip()
    conversation_state = conversation_state or []
    display_state = display_state or []
    user_id = 0

    if not user_msg:
        yield display_state, conversation_state, display_state, "Ready", "<div class='progress-bar hidden'></div>"
        return

    specs = get_task_specs()
    label_to_id = {s.label: s.id for s in specs}
    task_id = label_to_id.get(task_label, TaskId.CHAT_MAIN)

    if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
        tok, mdl = load_main_model(paths.main_llm)
        RUNTIME.main_tokenizer = tok
        RUNTIME.main_model = mdl

    working_display = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]
    if task_id == TaskId.CAD_CODER:
        max_new_tokens = _effective_coder_max_new_tokens()
    elif task_id in (TaskId.REASONING, TaskId.DEEP_RESEARCH):
        max_new_tokens = _effective_reasoning_max_new_tokens()
    else:
        max_new_tokens = _effective_main_max_new_tokens()
    use_cloaker = task_id in (TaskId.CAD_CODER, TaskId.REASONING, TaskId.DEEP_RESEARCH)
    if task_id == TaskId.CAD_CODER and _coder_skip_cloak_for_long_output():
        use_cloaker = False
    HIDDEN_PROGRESS = "<div class='progress-bar hidden'></div>"

    yield working_display, conversation_state, display_state, "Starting…", HIDDEN_PROGRESS

    keys_to_unload = LIFECYCLE.get_keys_to_unload_before_loading(task_id)
    for k in keys_to_unload:
        RUNTIME.unload_module(k)

    if task_id != TaskId.CHAT_MAIN:
        keys = LIFECYCLE.get_keys_for_task(task_id)
        keys_to_load = [k for k in keys if RUNTIME.get(k) is None]
        if keys_to_load:
            total_estimate = LIFECYCLE.get_load_time_estimate(keys_to_load)
            label = KEY_TO_LABEL.get(keys_to_load[0], keys_to_load[0])
            load_done = {"done": False, "error": None}

            def _load():
                try:
                    _ensure_modules_loaded(task_id, root_path)
                    load_done["done"] = True
                except Exception as e:
                    load_done["error"] = e

            t = Thread(target=_load, daemon=True)
            t.start()
            start = time.time()
            while not load_done["done"] and load_done["error"] is None:
                elapsed = time.time() - start
                pct = min(0.98, elapsed / total_estimate) if total_estimate > 0 else min(0.98, elapsed / 30)
                remaining = max(0, int(total_estimate - elapsed)) if total_estimate > 0 else max(0, int(30 - elapsed))
                progress_html = f"<div class='progress-bar'><div class='progress-fill' style='width:{pct*100}%'></div></div>"
                status = f"Loading {label} {int(pct * 100)}% (~{remaining}s)"
                yield working_display, conversation_state, display_state, status, progress_html
                time.sleep(0.35)
            t.join(timeout=1.0)
            if load_done["error"]:
                raise load_done["error"]
        LIFECYCLE.on_module_loaded(task_id)
    else:
        LIFECYCLE.on_switch_to_task(task_id, lambda k: RUNTIME.get(k) is not None)

    conv_turns = len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")])
    use_web, reason = should_auto_web(user_msg, force_web=False)
    source_cards: List[Dict[str, Any]] = []
    retrieved_notes = ""
    thinking_collapsed: Optional[str] = None
    collapsed_summary = "Used model knowledge"
    step_lines: List[str] = ["Deciding whether to search the web…"]
    working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
    yield working_display, conversation_state, display_state, "Gathering context…", HIDDEN_PROGRESS

    if use_web:
        step_lines.append(friendly_web_decision_reason(reason))
        q_disp = refine_search_query(user_msg).strip()
        if len(q_disp) > 100:
            q_disp = q_disp[:97] + "…"
        step_lines.append(f"Searching: “{q_disp}”")
        working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
        yield working_display, conversation_state, display_state, "Searching…", HIDDEN_PROGRESS

        results = web_search(
            user_msg,
            max_results=SEARCH_RESULTS,
            max_sources=MAX_SOURCES,
            trusted_domains=trusted_domains,
            cache=SEARCH_CACHE,
        )
        # web_search refines event-style queries internally; mirror that in the model-facing trace
        web_query_for_prompt = refine_search_query(user_msg)
        urls = [r["url"] for r in results]
        texts: List[str] = []
        for url in urls:
            dom = get_domain(url) or "source"
            step_lines.append(f"Reading {dom}…")
            working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
            yield working_display, conversation_state, display_state, "Reading sources…", HIDDEN_PROGRESS
            texts.append(
                fetch_page(url, timeout_s=REQUEST_TIMEOUT_S, max_chars=MAX_CHARS_PER_SOURCE, cache=PAGE_CACHE)
            )

        retrieved_notes, source_cards = assemble_retrieval_from_results(results, texts)
        thinking_md = build_thinking_path(
            use_web=True,
            reason=reason,
            query=web_query_for_prompt,
            source_cards=source_cards,
            conversation_turns=conv_turns,
        )
        step_lines.append("Evaluating source relevance…")
        working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
        yield working_display, conversation_state, display_state, "Evaluating sources…", HIDDEN_PROGRESS
        if not use_cloaker:
            step_lines.append("Synthesizing your answer…")
            working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
            yield working_display, conversation_state, display_state, "Preparing response…", HIDDEN_PROGRESS
        collapsed_summary = f"Searched {len(source_cards)} sources" if source_cards else "No sources retrieved"
        if not use_cloaker:
            thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))
    else:
        thinking_md = build_thinking_path_no_web(conversation_turns=conv_turns)
        step_lines.append("No web search needed — answering from conversation and model knowledge.")
        if conv_turns:
            step_lines.append(f"Using {conv_turns} earlier turn(s) as context.")
        if not use_cloaker:
            step_lines.append("Synthesizing your answer…")
        working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
        yield working_display, conversation_state, display_state, "Preparing response…", HIDDEN_PROGRESS
        collapsed_summary = "Used model knowledge"
        if not use_cloaker:
            thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))

    global_mem = load_global_memory(root_path)
    user_mem = load_user_memory(root_path, user_id) if user_id else ""
    topic_mem = load_relevant_topics(root_path, user_id)
    system_prompt = build_system_prompt(root_path, global_mem, user_mem, topic_mem, retrieved_notes, thinking_md)
    if task_id == TaskId.CAD_CODER:
        system_prompt = (
            system_prompt
            + "\n\nCoder module rules: Implement each mechanic once (one blast radius rule, one damage path). "
            "Do not emit many near-duplicate blocks that only tweak a number (e.g. repeating radius + 25). "
            "Prefer a single function, loop, or config table. End when the requested scope is complete."
        )
    messages = build_messages(conversation_state, user_msg, system_prompt)

    coder_gen_extras = _coder_extra_generate_kwargs() if task_id == TaskId.CAD_CODER else None

    if task_id == TaskId.CHAT_MAIN:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]
    elif task_id == TaskId.CAD_CODER:
        cad = RUNTIME.get("cad_coder")
        tok, mdl, lock = cad["tokenizer"], cad["model"], GEN_LOCKS["cad_coder"]
    elif task_id in (TaskId.REASONING, TaskId.DEEP_RESEARCH):
        r = RUNTIME.get("reasoning_llm")
        tok, mdl, lock = r["tokenizer"], r["model"], GEN_LOCKS["reasoning_llm"]
    elif task_id == TaskId.IMAGE_TINY:
        yield working_display, conversation_state, display_state, "Image generation not yet implemented in v1.2", HIDDEN_PROGRESS
        return
    else:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]

    if use_cloaker:
        task_label_running = next((s.label for s in specs if s.id == task_id), task_label)
        step_lines.append(f"Running {task_label_running}…")
        working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
        yield working_display, conversation_state, display_state, "Running model…", HIDDEN_PROGRESS

        module_output = generate_full(
            tokenizer=tok,
            model=mdl,
            messages=messages,
            max_new_tokens=max_new_tokens,
            lock=lock,
            extra_generate_kwargs=coder_gen_extras,
        )
        step_lines.append("Drafting your reply…")
        working_display[-1] = {"role": "assistant", "content": build_thinking_html_open(step_lines=step_lines)}
        yield working_display, conversation_state, display_state, "Drafting…", HIDDEN_PROGRESS

        step_lines.append("Synthesizing your answer…")
        thinking_collapsed = build_thinking_html_collapsed(summary=collapsed_summary, step_lines=list(step_lines))

    if thinking_collapsed is None:
        raise RuntimeError("thinking_collapsed was not built before generation")
    base_thinking = thinking_collapsed
    working_display[-1] = {
        "role": "assistant",
        "content": compose_assistant_display(base_thinking, render_live_text("")),
    }
    yield working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS

    try:
        if use_cloaker:
            cloak_messages = build_cloak_messages(
                module_output, user_msg, task_id,
                conversation_context=[m for m in conversation_state or [] if m.get("role") in ("user", "assistant")],
            )
            main_tok, main_mdl = RUNTIME.main_tokenizer, RUNTIME.main_model
            main_lock = GEN_LOCKS["main"]
            final_text = ""
            for partial in generate_stream_chat(
                tokenizer=main_tok, model=main_mdl, messages=cloak_messages,
                max_new_tokens=max_new_tokens, lock=main_lock,
            ):
                final_text = partial
                working_display[-1] = {
                    "role": "assistant",
                    "content": compose_assistant_display(base_thinking, render_live_text(partial)),
                }
                yield working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS
        else:
            final_text = ""
            for partial in generate_stream_chat(
                tokenizer=tok,
                model=mdl,
                messages=messages,
                max_new_tokens=max_new_tokens,
                lock=lock,
                extra_generate_kwargs=coder_gen_extras,
            ):
                final_text = partial
                working_display[-1] = {
                    "role": "assistant",
                    "content": compose_assistant_display(base_thinking, render_live_text(partial)),
                }
                yield working_display, conversation_state, display_state, "Writing response…", HIDDEN_PROGRESS

        final_text = clean_model_text(final_text)
        working_display[-1] = {
            "role": "assistant",
            "content": compose_assistant_display(base_thinking, final_text),
        }
        new_conv = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": final_text}]
        new_disp = display_state + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": compose_assistant_display(base_thinking, final_text)},
        ]

        save_topic_memory(root_path, user_id, user_msg, final_text)

        if task_id == TaskId.CHAT_MAIN:
            try:
                append_main_model_training_turn(
                    root_path,
                    messages=messages,
                    thinking_md=thinking_md,
                    trace_summary=collapsed_summary,
                    step_lines=step_lines,
                    thinking_collapsed_html=base_thinking,
                    use_web=use_web,
                    web_trigger_reason=reason,
                    source_cards=source_cards,
                    answer=final_text,
                    user_message=user_msg,
                )
            except Exception as log_exc:
                print(f"[imagination] training log failed (turn still saved in chat): {log_exc}", flush=True)

        task_label_short = next((s.label for s in specs if s.id == task_id), task_label)
        yield new_disp, new_conv, new_disp, f"{task_label_short} ready", HIDDEN_PROGRESS

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}
        new_conv = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        new_disp = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        yield new_disp, new_conv, new_disp, f"Error: {e}", HIDDEN_PROGRESS


def clear_all():
    return [], [], [], "Ready", "<div class='progress-bar hidden'></div>"


# ----------------------------
# UI
# ----------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg0: #07080c;
  --bg1: #0e1018;
  --surface: rgba(255,255,255,0.035);
  --surface2: rgba(255,255,255,0.055);
  --border: rgba(255,255,255,0.09);
  --border-strong: rgba(255,255,255,0.14);
  --text: rgba(248,250,255,0.97);
  --muted: rgba(165,175,205,0.72);
  --accent: #7c8ef0;
  --accent-dim: rgba(124,142,240,0.14);
  --accent-glow: rgba(124,142,240,0.22);
  --radius-lg: 18px;
  --radius-md: 12px;
  --shadow: 0 24px 64px rgba(0,0,0,0.45);
}

.gradio-container {
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
  background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(124,142,240,0.12), transparent 50%),
    linear-gradient(165deg, var(--bg1) 0%, var(--bg0) 100%) !important;
  color: var(--text);
  min-height: 100vh;
}

#shell {
  max-width: 820px;
  margin: 0 auto;
  padding: 28px 20px 48px;
}

.app-brand {
  margin-bottom: 20px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border);
}
.app-brand h1 {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0 0 4px 0;
  color: var(--text);
}
.app-brand p {
  margin: 0;
  font-size: 0.8125rem;
  color: var(--muted);
  font-weight: 400;
}

.model-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 14px 16px 18px;
  margin-bottom: 14px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(20px);
}
.model-panel-label {
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}

#task-radio, #task-radio fieldset, #task-radio .wrap {
  gap: 8px !important;
  flex-wrap: wrap !important;
}
#task-radio label {
  font-size: 0.8125rem !important;
  font-weight: 500 !important;
  padding: 8px 14px !important;
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
  background: var(--bg0) !important;
  color: var(--muted) !important;
  transition: border-color 0.15s, background 0.15s, color 0.15s !important;
}
#task-radio label:hover {
  border-color: var(--border-strong) !important;
  color: var(--text) !important;
}
#task-radio input:checked + span,
#task-radio label:has(input:checked) {
  border-color: rgba(124,142,240,0.45) !important;
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
}

#status-wrap {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
#status-wrap p,
#status p {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.8125rem;
  color: var(--muted);
  margin: 0 !important;
}
#status-wrap p::before,
#status p::before {
  content: '';
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 14px var(--accent-glow);
}

.progress-bar {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  margin: 0 0 14px 0;
  overflow: hidden;
}
.progress-bar.hidden { display: none; }
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #5a6fd4, var(--accent), #a8b4ff);
  border-radius: 2px;
  transition: width 0.25s ease;
}

.chat-wrap {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 4px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  min-height: 0;
}
#chatbox {
  min-height: 480px;
  border: none !important;
  border-radius: calc(var(--radius-lg) - 4px) !important;
  background: transparent !important;
}

/* Collapsible in-chat thinking (HTML inside assistant bubble) */
#chatbox details.thinking-block {
  margin: 0 0 12px 0;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: rgba(0,0,0,0.22);
  font-size: 0.8125rem;
  color: var(--muted);
  line-height: 1.45;
  transition: border-color 0.2s ease, background 0.2s ease;
}
#chatbox details.thinking-block[open] {
  border-color: rgba(124,142,240,0.28);
  background: rgba(124,142,240,0.06);
}
#chatbox details.thinking-block.thinking-done:not([open]) {
  margin-bottom: 10px;
  opacity: 0.92;
}
#chatbox details.thinking-block summary.thinking-summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: var(--muted);
  user-select: none;
}
#chatbox details.thinking-block summary.thinking-summary::-webkit-details-marker {
  display: none;
}
#chatbox details.thinking-block .thinking-body {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  max-height: 220px;
  overflow-y: auto;
}
#chatbox details.thinking-block .thinking-step {
  padding: 4px 0;
  border-left: 2px solid var(--border-strong);
  padding-left: 10px;
  margin-left: 2px;
}
#chatbox details.thinking-block .thinking-step:last-child {
  margin-bottom: 0;
}
#chatbox .thinking-dots {
  display: inline-flex;
  gap: 3px;
  align-items: center;
  margin-left: 4px;
}
#chatbox .thinking-dots span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--accent);
  opacity: 0.35;
  animation: thinking-pulse 1.1s ease-in-out infinite;
}
#chatbox .thinking-dots span:nth-child(2) { animation-delay: 0.15s; }
#chatbox .thinking-dots span:nth-child(3) { animation-delay: 0.3s; }
#chatbox details.thinking-block:not([open]) .thinking-dots {
  display: none;
}
@keyframes thinking-pulse {
  0%, 100% { opacity: 0.25; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.05); }
}

/* Assistant bubble accent */
#chatbox .message.bot,
#chatbox [class*="bot"] {
  border-left: 3px solid rgba(124,142,240,0.45) !important;
  padding-left: 12px !important;
  border-radius: var(--radius-md) !important;
}

.input-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
.input-row textarea {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  font-size: 0.9375rem !important;
  flex: 1;
  min-height: 48px !important;
  max-height: 140px !important;
  background: var(--bg0) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 14px !important;
}
.input-row textarea:focus {
  border-color: rgba(124,142,240,0.45) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.input-row button.primary {
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 12px 22px !important;
  border-radius: var(--radius-md) !important;
  background: linear-gradient(180deg, rgba(124,142,240,0.35), rgba(124,142,240,0.18)) !important;
  border: 1px solid rgba(124,142,240,0.4) !important;
  color: var(--text) !important;
}
.input-row button.secondary {
  font-size: 0.8125rem !important;
  padding: 12px 16px !important;
  border-radius: var(--radius-md) !important;
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
}
"""


def build_ui():
    task_specs = get_task_specs()
    task_labels = [s.label for s in task_specs]

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["DM Sans", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "monospace"],
    ).set(
        body_background_fill="transparent",
        block_background_fill="transparent",
        block_border_width="0",
        block_label_text_size="sm",
    )

    with gr.Blocks(css=CSS, theme=theme, title="Imagination v1.2") as demo:
        with gr.Column(elem_id="shell"):
            gr.HTML("""
            <header class="app-brand">
              <h1>Imagination</h1>
              <p>v1.2 · Select a model, then ask anything.</p>
            </header>
            """)

            with gr.Column(elem_classes=["model-panel"]):
                gr.HTML('<div class="model-panel-label">Model</div>')
                task = gr.Radio(
                    choices=task_labels,
                    value=task_labels[0],
                    label=None,
                    show_label=False,
                    elem_id="task-radio",
                    container=False,
                )
                progress_html = gr.HTML("<div class='progress-bar hidden'></div>")
                with gr.Row(elem_id="status-wrap"):
                    status = gr.Markdown("Ready", elem_id="status")

            with gr.Column(elem_classes=["chat-wrap"]):
                try:
                    chat = gr.Chatbot(
                        elem_id="chatbox",
                        height=520,
                        label=None,
                        show_label=False,
                        show_copy_button=True,
                        layout="bubble",
                        format="messages",
                        render_markdown=True,
                    )
                except TypeError:
                    chat = gr.Chatbot(
                        elem_id="chatbox",
                        height=520,
                        label=None,
                        show_label=False,
                        show_copy_button=True,
                        layout="bubble",
                        type="messages",
                        render_markdown=True,
                    )

            with gr.Row(elem_classes=["input-row"]):
                user = gr.Textbox(
                    placeholder="Message…",
                    show_label=False,
                    container=False,
                    scale=9,
                    lines=1,
                    max_lines=4,
                )
                send = gr.Button("Send", variant="primary", elem_classes=["primary"], scale=1)
                clear_btn = gr.Button("Clear", elem_classes=["secondary"], scale=0)

            conversation_state = gr.State([])
            display_state = gr.State([])

            def chat_inputs():
                return [task, user, conversation_state, display_state]

            send_evt = send.click(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=[chat, conversation_state, display_state, status, progress_html],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            )
            send_evt.then(lambda: "", None, user)

            user.submit(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=[chat, conversation_state, display_state, status, progress_html],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            ).then(lambda: "", None, user)

            clear_btn.click(fn=clear_all, inputs=None, outputs=[chat, conversation_state, display_state, status, progress_html])

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True, server_port=7860)
