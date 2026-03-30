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
from imagination_runtime.thinking import build_thinking_path, build_thinking_path_no_web
from imagination_runtime.users import (
    load_global_memory,
    load_user_memory,
    load_trusted,
    load_relevant_topics,
    save_topic_memory,
)
from imagination_runtime.web import fetch_parallel, sources_to_cards, web_search

warnings.filterwarnings("ignore")

SEARCH_RESULTS = 6
MAX_SOURCES = 3
MAX_CHARS_PER_SOURCE = 900
REQUEST_TIMEOUT_S = 8
DEFAULT_MAX_NEW_TOKENS = 360
WEB_MAX_NEW_TOKENS = 420
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
) -> Iterable[str]:
    device = _get_device_for_model(model)
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=90.0)
    gen_error = {"exc": None}

    def _run():
        try:
            with lock, torch.inference_mode():
                model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
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
) -> str:
    result = ""
    for chunk in generate_stream_chat(
        tokenizer=tokenizer, model=model, messages=messages,
        max_new_tokens=max_new_tokens, lock=lock,
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
- Answer directly.
- Keep answers concise unless the user asks for detail.
- If retrieved web notes are present, use them carefully.
- Cite only sources you actually used.
- If evidence conflicts, say so.
- Do not invent citations.
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
            "\n\nRetrieved web notes:\n"
            "Use these only if relevant. Cite inline as [1], [2], etc. only for sources actually used. "
            "End with a short 'Sources used' list containing only the URLs actually used.\n\n"
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


def build_retrieval_bundle(
    question: str,
    trusted_domains: List[str],
) -> Tuple[str, str, str, List[Dict[str, Any]]]:
    results = web_search(
        question,
        max_results=SEARCH_RESULTS,
        max_sources=MAX_SOURCES,
        trusted_domains=trusted_domains,
        cache=SEARCH_CACHE,
    )
    urls = [r["url"] for r in results]
    texts = fetch_parallel(urls, timeout_s=REQUEST_TIMEOUT_S, max_chars=MAX_CHARS_PER_SOURCE, cache=PAGE_CACHE)

    source_cards: List[Dict[str, Any]] = []
    trace_lines: List[str] = []
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
        trace_lines.append(
            f"{i+1}. **{r.get('title')}**  \n   - domain: `{r.get('domain')}`  \n   - trust: **{trust_label}**  \n   - url: {r.get('url')}  \n   - snippet: {content[:220] or '(no text extracted)'}"
        )
        retrieved_docs.append(
            f"[{i+1}] {r.get('title')}\nURL: {r.get('url')}\nDomain: {r.get('domain')}\nTrust: {trust_label}\nNotes: {content[:MAX_CHARS_PER_SOURCE]}"
        )

    trace_md = "### Trace\n\n- Route: **web + model**\n" + f"- Query: `{question}`\n" + f"- Sources kept: **{len(source_cards)}**\n\n"
    trace_md += "### Source picks\n\n" + ("\n\n".join(trace_lines) if trace_lines else "_No useful sources found._")
    return "\n\n".join(retrieved_docs), sources_to_cards(source_cards), trace_md, source_cards


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
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    yield working_display, conversation_state, display_state, "Starting...", "<div class='progress-bar hidden'></div>"

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
                status = f"Loading {label}... {int(pct * 100)}% (~{remaining}s)"
                yield working_display, conversation_state, display_state, status, progress_html
                time.sleep(0.35)
            t.join(timeout=1.0)
            if load_done["error"]:
                raise load_done["error"]
        LIFECYCLE.on_module_loaded(task_id)
    else:
        LIFECYCLE.on_switch_to_task(task_id, lambda k: RUNTIME.get(k) is not None)

    use_web, reason = should_auto_web(user_msg, force_web=False)
    source_cards: List[Dict[str, Any]] = []
    retrieved_notes = ""

    if use_web:
        yield working_display, conversation_state, display_state, "Searching...", "<div class='progress-bar hidden'></div>"
        retrieved_notes, _, _, source_cards = build_retrieval_bundle(user_msg, trusted_domains)
        max_new_tokens = max(max_new_tokens, WEB_MAX_NEW_TOKENS)
        thinking_md = build_thinking_path(
            use_web=True,
            reason=reason,
            query=user_msg,
            source_cards=source_cards,
            conversation_turns=len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")]),
        )
    else:
        thinking_md = build_thinking_path_no_web(
            conversation_turns=len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")])
        )

    global_mem = load_global_memory(root_path)
    user_mem = load_user_memory(root_path, user_id) if user_id else ""
    topic_mem = load_relevant_topics(root_path, user_id)
    system_prompt = build_system_prompt(root_path, global_mem, user_mem, topic_mem, retrieved_notes, thinking_md)
    messages = build_messages(conversation_state, user_msg, system_prompt)

    if task_id == TaskId.CHAT_MAIN:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]
        use_cloaker = False
    elif task_id == TaskId.CAD_CODER:
        cad = RUNTIME.get("cad_coder")
        tok, mdl, lock = cad["tokenizer"], cad["model"], GEN_LOCKS["cad_coder"]
        use_cloaker = True
    elif task_id in (TaskId.REASONING, TaskId.DEEP_RESEARCH):
        r = RUNTIME.get("reasoning_llm")
        tok, mdl, lock = r["tokenizer"], r["model"], GEN_LOCKS["reasoning_llm"]
        use_cloaker = True
    elif task_id == TaskId.IMAGE_TINY:
        yield working_display, conversation_state, display_state, "Image generation not yet implemented in v1.2", "<div class='progress-bar hidden'></div>"
        return
    else:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]
        use_cloaker = False

    working_display[-1] = {"role": "assistant", "content": render_live_text("")}
    yield working_display, conversation_state, display_state, "Generating...", "<div class='progress-bar hidden'></div>"

    try:
        if use_cloaker:
            module_output = generate_full(
                tokenizer=tok, model=mdl, messages=messages,
                max_new_tokens=max_new_tokens, lock=lock,
            )
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
                working_display[-1] = {"role": "assistant", "content": render_live_text(partial)}
                yield working_display, conversation_state, display_state, "Generating...", "<div class='progress-bar hidden'></div>"
        else:
            final_text = ""
            for partial in generate_stream_chat(
                tokenizer=tok, model=mdl, messages=messages,
                max_new_tokens=max_new_tokens, lock=lock,
            ):
                final_text = partial
                working_display[-1] = {"role": "assistant", "content": render_live_text(partial)}
                yield working_display, conversation_state, display_state, "Generating...", "<div class='progress-bar hidden'></div>"

        final_text = clean_model_text(final_text)
        working_display[-1] = {"role": "assistant", "content": final_text}
        new_conv = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": final_text}]
        new_disp = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": final_text}]

        save_topic_memory(root_path, user_id, user_msg, final_text)

        task_label_short = next((s.label for s in specs if s.id == task_id), task_label)
        yield new_disp, new_conv, new_disp, f"{task_label_short} ready", "<div class='progress-bar hidden'></div>"

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}
        new_conv = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        new_disp = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        yield new_disp, new_conv, new_disp, f"Error: {e}", "<div class='progress-bar hidden'></div>"


def clear_all():
    return [], [], [], "Ready", "<div class='progress-bar hidden'></div>"


# ----------------------------
# UI
# ----------------------------
CSS = """
:root {
  --bg: #0a0a0f;
  --bg2: #12121a;
  --border: rgba(255,255,255,0.08);
  --text: rgba(245,247,255,0.96);
  --muted: rgba(180,190,220,0.6);
  --accent: #6b7fd7;
}
.gradio-container {
  background: var(--bg) !important;
  color: var(--text);
}
#shell { max-width: 720px; margin: 0 auto; padding: 16px; }
.tab-bar {
  display: flex; gap: 6px; padding: 10px 0; margin-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.tab-pill {
  padding: 6px 14px; border-radius: 999px; font-size: 13px;
  background: transparent; color: var(--muted); border: 1px solid transparent;
  cursor: pointer; transition: 0.15s ease;
}
.tab-pill:hover { color: var(--text); }
.tab-pill.active {
  background: rgba(107,127,215,0.15); color: var(--accent);
  border-color: rgba(107,127,215,0.3);
}
.status-row {
  display: flex; justify-content: flex-end; align-items: center; gap: 8px;
  margin-bottom: 12px; font-size: 12px; color: var(--muted);
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); opacity: 0.8;
}
.progress-bar {
  height: 2px; background: var(--border); border-radius: 1px;
  margin-bottom: 12px; overflow: hidden;
}
.progress-bar.hidden { display: none; }
.progress-fill {
  height: 100%; background: linear-gradient(90deg, var(--accent), #8b9ee8);
  transition: width 0.2s ease;
}
#chatbox { min-height: 520px; }
.chat-bubble { border-radius: 16px; }
.input-row {
  display: flex; gap: 8px; margin-top: 12px;
}
.input-row textarea {
  flex: 1; min-height: 44px; max-height: 120px;
  background: var(--bg2) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important;
}
.input-row textarea:focus {
  border-color: rgba(107,127,215,0.4) !important;
  box-shadow: 0 0 0 1px rgba(107,127,215,0.2) !important;
}
.send-btn {
  padding: 10px 20px; border-radius: 12px;
  background: rgba(107,127,215,0.2) !important;
  border: 1px solid rgba(107,127,215,0.35) !important;
  color: var(--text) !important;
}
.send-btn:hover {
  background: rgba(107,127,215,0.3) !important;
}
"""


def build_ui():
    task_specs = get_task_specs()
    task_labels = [s.label for s in task_specs]

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="var(--bg)",
        block_background_fill="var(--bg2)",
        block_border_color="var(--border)",
    )

    with gr.Blocks(css=CSS, theme=theme, title="Imagination v1.2") as demo:
        with gr.Column(elem_id="shell"):
            task = gr.Radio(
                choices=task_labels,
                value=task_labels[0],
                label=None,
                show_label=False,
                elem_id="task-radio",
                container=False,
            )

            status = gr.Markdown("Ready", elem_id="status")
            progress_html = gr.HTML("<div class='progress-bar hidden'></div>")

            chat = gr.Chatbot(
                elem_id="chatbox",
                height=520,
                show_copy_button=True,
                layout="bubble",
            )

            with gr.Row(elem_classes=["input-row"]):
                user = gr.Textbox(
                    placeholder="Ask anything…",
                    show_label=False,
                    container=False,
                    scale=9,
                    lines=1,
                    max_lines=4,
                )
                send = gr.Button("Send", variant="primary", elem_classes=["send-btn"], scale=1)
                clear_btn = gr.Button("Clear", scale=0)

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
