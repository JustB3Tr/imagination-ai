# ============================================================
# IMAGINATION v1.1.2 — Thinking Path, auth, per-user memory, budget
#
# - Spruced-up UI with Thinking Path panel
# - Multi-provider auth (HF OAuth, email/password; Google/GitHub/Apple stubs)
# - Per-user + global memory
# - Compute budget indicator (~58h/mo on L4)
# - Slash commands: /code, /research, /image (default: chat)
# - 4-bit quantization for fast loading + low VRAM
# - Background pre-loading of all models at startup
# ============================================================

from __future__ import annotations

"""
Colab: !pip install -q gradio transformers accelerate safetensors bitsandbytes requests beautifulsoup4 ddgs bcrypt diffusers
Mount Drive, set IMAGINATION_ROOT, then:
%cd /content/imagination-v1.1.0/versions/v1.1.2
!python imagination_v1_1_2_colab_gradio.py
"""

import gc
import os
import re
import warnings
from collections import OrderedDict
from threading import Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from imagination_runtime.auth import login_email_password, signup_email_password, resolve_user_from_oauth_profile
from imagination_runtime.budget import remaining_hours, is_low_budget
from imagination_runtime.cache import RuntimeCache
from imagination_runtime.paths import ModelPaths, resolve_root_path
from imagination_runtime.registry import TaskId, get_task_specs
from imagination_runtime.thinking import build_thinking_path, build_thinking_path_no_web
from imagination_runtime.users import (
    load_global_memory,
    load_user_memory,
    save_user_memory,
    save_global_memory,
    load_trusted,
    trusted_sources_path,
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

SLASH_COMMANDS = {
    "/code": TaskId.CAD_CODER,
    "/cad": TaskId.CAD_CODER,
    "/research": TaskId.DEEP_RESEARCH,
    "/image": TaskId.IMAGE_TINY,
}

BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


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
PRELOAD_STATUS: Dict[str, str] = {}


def parse_slash_command(text: str) -> Tuple[TaskId, str]:
    t = text.strip()
    for cmd, task_id in SLASH_COMMANDS.items():
        if t.lower().startswith(cmd):
            rest = t[len(cmd):].strip()
            return task_id, rest
    return TaskId.CHAT_MAIN, t


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


def _use_4bit() -> bool:
    return torch.cuda.is_available()


def load_main_model(root_path: str) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(root_path, use_fast=True)
    kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if _use_4bit():
        kwargs["quantization_config"] = BNB_4BIT
    model = AutoModelForCausalLM.from_pretrained(root_path, **kwargs)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return tok, model


def _load_cad_coder(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.cad_coder, use_fast=True)
    kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if _use_4bit():
        kwargs["quantization_config"] = BNB_4BIT
    model = AutoModelForCausalLM.from_pretrained(paths.cad_coder, **kwargs)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_reasoning_llm(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.reasoning_llm, use_fast=True)
    kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if _use_4bit():
        kwargs["quantization_config"] = BNB_4BIT
    model = AutoModelForCausalLM.from_pretrained(paths.reasoning_llm, **kwargs)
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


def preload_main_model(root_path: str) -> None:
    """Pre-load only the main chat model in a background thread.
    Other models load on demand when a slash command is used, to avoid OOM."""
    paths = ModelPaths(root=root_path)

    def _load_main():
        try:
            PRELOAD_STATUS["main"] = "loading"
            print("[preload] Loading main model (4-bit)...")
            tok, mdl = load_main_model(paths.main_llm)
            RUNTIME.main_tokenizer = tok
            RUNTIME.main_model = mdl
            PRELOAD_STATUS["main"] = "ready"
            print("[preload] Main model ready.")
        except Exception as e:
            PRELOAD_STATUS["main"] = f"error: {e}"
            print(f"[preload] Main model failed: {e}")

    Thread(target=_load_main, daemon=True).start()


def ensure_modules_loaded(task_id: TaskId, root_path: str) -> List[str]:
    paths = ModelPaths(root=root_path)
    status: List[str] = []

    def ensure(key: str, loader):
        if RUNTIME.get(key) is None:
            ps = PRELOAD_STATUS.get(key, "")
            if ps == "loading":
                status.append(f"- Waiting for `{key}` (pre-loading)...")
                import time
                while PRELOAD_STATUS.get(key) == "loading":
                    time.sleep(0.5)
                if RUNTIME.get(key) is not None:
                    status.append(f"- `{key}` ready (pre-loaded).")
                    return
            status.append(f"- Loading `{key}`...")
            RUNTIME.set(key, loader(paths))
            status.append(f"- Loaded `{key}`.")

    if task_id == TaskId.CAD_CODER:
        ensure("cad_coder", _load_cad_coder)
    elif task_id == TaskId.DEEP_RESEARCH:
        ensure("embeddings", _load_embeddings)
        ensure("reranker", _load_reranker)
        ensure("reasoning_llm", _load_reasoning_llm)
    elif task_id == TaskId.IMAGE_TINY:
        ensure("tiny_sd", _load_tiny_sd)
    return status


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


def build_system_prompt(
    global_memory: str,
    user_memory: str,
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
    if thinking_path.strip():
        base += "\n\n" + thinking_path.strip()
    if retrieved_notes.strip():
        base += (
            "\n\nRetrieved web notes:\n"
            "Use these only if relevant. Cite inline as [1], [2], etc. only for sources actually used. "
            "End with a short 'Sources used' list containing only the URLs actually used.\n\n"
            f"{retrieved_notes.strip()}"
        )
    return base


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


def chat_submit(
    root_path_in: str,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    display_state: List[Dict[str, str]],
    force_web: bool,
    show_traces: bool,
    user_max_tokens: int,
    user_info: Optional[Dict[str, Any]],
):
    root_path = resolve_root_path(root_path_in)
    paths = ModelPaths(root=root_path)
    trusted_domains = load_trusted(root_path, TRUSTED_DOMAINS_STARTER)

    user_msg = (user_msg or "").strip()
    conversation_state = conversation_state or []
    display_state = display_state or []
    user_id = user_info.get("id") if user_info else None

    if not user_msg:
        empty_sources = "<div class='sources-empty'>Run a search or ask a current-events question.</div>"
        yield display_state, conversation_state, display_state, empty_sources, "### Trace\n\n_No trace yet._", "### Thinking Path\n\n_No thinking yet._", "⚠️ No question entered."
        return

    task_id, clean_msg = parse_slash_command(user_msg)
    display_msg = user_msg

    specs = get_task_specs()
    task_label = next((s.label for s in specs if s.id == task_id), "Chat (main)")

    if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
        ps = PRELOAD_STATUS.get("main", "")
        if ps == "loading":
            import time
            while PRELOAD_STATUS.get("main") == "loading":
                time.sleep(0.5)
        if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
            tok, mdl = load_main_model(paths.main_llm)
            RUNTIME.main_tokenizer = tok
            RUNTIME.main_model = mdl

    working_display = display_state + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": ""}]
    sources_html = "<div class='sources-empty'>Run a search or ask a current-events question.</div>"
    trace_parts = ["### Trace", "", f"- Task: **{task_label}**", f"- Root: `{root_path}`"]
    thinking_md = "### Thinking Path\n\n_Starting..._"

    yield working_display, conversation_state, display_state, sources_html, "\n".join(trace_parts), thinking_md, "⏳ Starting..."

    load_lines = ensure_modules_loaded(task_id, root_path)
    if load_lines:
        trace_parts.append("- Module loads:")
        trace_parts.extend([f"  {ln}" for ln in load_lines])
        yield working_display, conversation_state, display_state, sources_html, "\n".join(trace_parts), thinking_md, "📦 Modules loaded..."

    use_web, reason = should_auto_web(clean_msg, force_web=force_web)
    trace_parts.append(f"- Route: **{'web + model' if use_web else 'model only'}**")
    trace_parts.append(f"- Decision reason: **{reason}**")
    trace_parts.append(f"- Query: `{clean_msg}`")
    trace_md = "\n".join(trace_parts)
    max_new_tokens = int(user_max_tokens or DEFAULT_MAX_NEW_TOKENS)

    source_cards: List[Dict[str, Any]] = []
    retrieved_notes = ""

    if use_web:
        yield working_display, conversation_state, display_state, sources_html, trace_md, thinking_md, "🌐 Searching..."
        retrieved_notes, sources_html, trace_md_full, source_cards = build_retrieval_bundle(clean_msg, trusted_domains)
        max_new_tokens = max(max_new_tokens, WEB_MAX_NEW_TOKENS)
        trace_md = trace_md_full if show_traces else trace_md + f"\n- Sources kept: **{MAX_SOURCES} max**"
        thinking_md = build_thinking_path(
            use_web=True,
            reason=reason,
            query=clean_msg,
            source_cards=source_cards,
            conversation_turns=len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")]),
        )
        yield working_display, conversation_state, display_state, sources_html, trace_md, thinking_md, "📚 Sources ready..."
    else:
        thinking_md = build_thinking_path_no_web(
            conversation_turns=len([m for m in conversation_state or [] if m.get("role") in ("user", "assistant")])
        )

    global_mem = load_global_memory(root_path)
    user_mem = load_user_memory(root_path, user_id) if user_id else ""
    system_prompt = build_system_prompt(global_mem, user_mem, retrieved_notes, thinking_md)
    messages = build_messages(conversation_state, clean_msg, system_prompt)

    if task_id == TaskId.CAD_CODER:
        cad = RUNTIME.get("cad_coder")
        tok, mdl, lock = cad["tokenizer"], cad["model"], GEN_LOCKS["cad_coder"]
    elif task_id == TaskId.DEEP_RESEARCH:
        r = RUNTIME.get("reasoning_llm")
        tok, mdl, lock = r["tokenizer"], r["model"], GEN_LOCKS["reasoning_llm"]
    else:
        tok, mdl, lock = RUNTIME.main_tokenizer, RUNTIME.main_model, GEN_LOCKS["main"]

    working_display[-1] = {"role": "assistant", "content": render_live_text("")}
    yield working_display, conversation_state, display_state, sources_html, trace_md, thinking_md, "✍️ Generating..."

    try:
        final_text = ""
        for partial in generate_stream_chat(tokenizer=tok, model=mdl, messages=messages, max_new_tokens=max_new_tokens, lock=lock):
            final_text = partial
            working_display[-1] = {"role": "assistant", "content": render_live_text(partial)}
            yield working_display, conversation_state, display_state, sources_html, trace_md, thinking_md, "✍️ Typing..."

        final_text = clean_model_text(final_text)
        working_display[-1] = {"role": "assistant", "content": final_text}
        new_conv = conversation_state + [{"role": "user", "content": clean_msg}, {"role": "assistant", "content": final_text}]
        new_disp = display_state + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": final_text}]
        yield new_disp, new_conv, new_disp, sources_html, trace_md, thinking_md, "✓ Done"

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}
        new_conv = conversation_state + [{"role": "user", "content": clean_msg}, {"role": "assistant", "content": err_text}]
        new_disp = display_state + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": err_text}]
        yield new_disp, new_conv, new_disp, sources_html, trace_md, thinking_md, f"✖ {e}"


def clear_all():
    empty = "<div class='sources-empty'>Run a search or ask a current-events question.</div>"
    return [], [], [], empty, "### Trace\n\n_No trace yet._", "### Thinking Path\n\n_No thinking yet._", "⟡ Idle", ""


def clear_modules():
    RUNTIME.clear_modules()
    return "Cleared module caches."


# ----------------------------
# UI
# ----------------------------
CSS = """
:root{
  --bg0:#06070a; --bg1:#0b1020; --panel:rgba(255,255,255,0.065); --panel2:rgba(255,255,255,0.045);
  --border:rgba(255,255,255,0.14); --text:rgba(245,247,255,0.96); --muted:rgba(210,220,255,0.66);
  --accent:#8ea2ff; --accent2:#66e0ff; --shadow:0 16px 40px rgba(0,0,0,0.42);
}
.gradio-container{
  background: radial-gradient(1200px 800px at 10% 0%, rgba(102,224,255,0.09), transparent 45%),
    radial-gradient(1000px 700px at 100% 0%, rgba(142,162,255,0.13), transparent 48%),
    linear-gradient(180deg, var(--bg1), var(--bg0)) !important;
  color:var(--text);
}
#shell{ max-width:1200px; margin:0 auto; }
.hero{
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:18px 18px 12px 18px; margin-bottom:14px; border-radius:22px;
  background: linear-gradient(135deg, rgba(142,162,255,0.06) 0%, rgba(102,224,255,0.04) 50%, rgba(142,162,255,0.06) 100%);
  border:1px solid var(--border); box-shadow:var(--shadow); backdrop-filter: blur(12px);
  animation: shimmer 8s ease-in-out infinite;
}
@keyframes shimmer{ 0%,100%{opacity:1} 50%{opacity:.97} }
.hero-title{ font-size:30px; font-weight:800; letter-spacing:1.6px; }
.hero-sub{ color:var(--muted); font-size:13px; letter-spacing:0.5px; margin-top:4px; }
.ping{ width:12px; height:12px; border-radius:50%;
  background:linear-gradient(180deg, var(--accent2), var(--accent));
  box-shadow:0 0 22px rgba(102,224,255,0.6);
}
.budget-badge{ font-size:11px; color:var(--muted); padding:4px 10px; border-radius:999px; background:rgba(255,255,255,0.06); }
.budget-warn{ color:#ffd27a; }
.card{
  background:linear-gradient(180deg, var(--panel), var(--panel2));
  border:1px solid var(--border); border-radius:20px; box-shadow:var(--shadow);
  backdrop-filter: blur(12px); transition: box-shadow 0.2s ease;
}
.card:hover{ box-shadow:0 0 24px rgba(142,162,255,0.12); }
.section-pad{ padding:14px; }
#status{ margin:2px 2px 12px 2px; color:var(--muted); font-size:13px; }
#chatbox{ min-height:520px; }
textarea, input{
  background:rgba(8,10,16,0.72) !important; color:var(--text) !important;
  border:1px solid rgba(255,255,255,0.08) !important; border-radius:14px !important;
}
textarea:focus, input:focus{
  border-color:rgba(142,162,255,0.45) !important;
  box-shadow:0 0 0 1px rgba(142,162,255,0.24), 0 0 18px rgba(142,162,255,0.20) !important;
}
button{
  border-radius:14px !important; border:1px solid rgba(255,255,255,0.18) !important;
  background:linear-gradient(180deg, rgba(18,22,34,.92), rgba(12,15,24,.92)) !important;
  color:var(--text) !important; box-shadow:0 8px 22px rgba(0,0,0,.22);
}
button:hover{
  border-color:rgba(142,162,255,0.42) !important;
  box-shadow:0 0 18px rgba(142,162,255,0.18);
}
.sources-grid{ display:grid; grid-template-columns:1fr; gap:10px; }
.source-card{
  display:block; text-decoration:none; color:var(--text);
  padding:12px; border-radius:16px; background:rgba(10,13,20,0.70);
  border:1px solid rgba(255,255,255,0.10); transition:.16s ease;
}
.source-card:hover{
  transform:translateY(-1px); border-color:rgba(142,162,255,0.35);
  box-shadow:0 0 18px rgba(142,162,255,0.14);
}
.source-row{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.source-badge{ font-size:12px; color:var(--muted); border:1px solid rgba(255,255,255,0.14); border-radius:999px; padding:2px 8px; }
.trust-pill{ font-size:11px; font-weight:700; letter-spacing:.7px; padding:4px 8px; border-radius:999px; }
.trust-on{ color:#bffff0; background:rgba(126,243,197,0.10); border:1px solid rgba(126,243,197,0.22); }
.trust-off{ color:#ffe7ac; background:rgba(255,210,122,0.10); border:1px solid rgba(255,210,122,0.20); }
.source-title{ font-size:14px; font-weight:700; margin-bottom:6px; }
.source-domain{ color:var(--accent2); font-size:12px; margin-bottom:8px; }
.source-snippet{ color:var(--muted); font-size:12px; line-height:1.45; }
.sources-empty{ color:var(--muted); font-size:12px; padding:12px; border-radius:16px;
  border:1px dashed rgba(255,255,255,0.14); background:rgba(255,255,255,0.03);
}
.smallcap{ font-size:12px; color:var(--muted); letter-spacing:1px; margin-bottom:8px; text-transform:uppercase; }
.note{ color:var(--muted); font-size:12px; }
@media (max-width:768px){ #shell{ padding:8px; } .hero{ flex-direction:column; align-items:flex-start; } }
"""


def build_ui():
    task_specs = get_task_specs()
    root_default = os.getenv("IMAGINATION_ROOT", "/content/imagination-v1.1.0")

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")
    with gr.Blocks(css=CSS, theme=theme, title="Imagination v1.1.2") as demo:
        with gr.Column(elem_id="shell"):
            def budget_html():
                try:
                    h = remaining_hours(root_default)
                    low = is_low_budget(root_default)
                    cls = "budget-warn" if low else ""
                    return f"<span class='budget-badge {cls}'>~{h:.1f}h L4 left/mo</span>"
                except Exception:
                    return "<span class='budget-badge'>Set root path</span>"

            gr.HTML(f"""
            <div class="hero">
              <div>
                <div class="hero-title">IMAGINATION v1.1.2</div>
                <div class="hero-sub">Thinking Path &bull; per-user memory &bull; compute budget &bull; slash commands</div>
              </div>
              <div style="display:flex;align-items:center;gap:12px;">
                {budget_html()}
                <div class="ping" title="online"></div>
              </div>
            </div>
            """)

            status = gr.Markdown("⟡ Idle", elem_id="status")
            user_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Column(elem_classes=["card", "section-pad"]):
                        gr.Markdown("<div class='smallcap'>Chat</div>")
                        def _make_chatbot():
                            combos = [
                                {"elem_id": "chatbox", "height": 520, "type": "messages", "buttons": ["copy", "copy_all"], "layout": "bubble"},
                                {"elem_id": "chatbox", "height": 520, "type": "messages", "layout": "bubble"},
                                {"elem_id": "chatbox", "height": 520, "type": "messages"},
                                {"elem_id": "chatbox", "height": 520, "buttons": ["copy", "copy_all"], "layout": "bubble"},
                                {"elem_id": "chatbox", "height": 520, "layout": "bubble"},
                                {"elem_id": "chatbox", "height": 520},
                            ]
                            for kw in combos:
                                try:
                                    return gr.Chatbot(**kw)
                                except TypeError:
                                    continue
                            return gr.Chatbot(elem_id="chatbox", height=520)
                        chat = _make_chatbot()
                        user = gr.Textbox(label="Message", placeholder="Ask anything… or use /code /research /image", lines=2)
                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear")
                        with gr.Accordion("Settings", open=False):
                            root_path = gr.Textbox(label="Model root", value=root_default, info="Path to imagination-v1.1.0")
                            with gr.Row():
                                force_web = gr.Checkbox(label="Force web search", value=False)
                                show_traces = gr.Checkbox(label="Show trace details", value=True)
                            user_max_tokens = gr.Slider(160, 700, step=20, value=DEFAULT_MAX_NEW_TOKENS, label="Max new tokens")
                            clear_modules_btn = gr.Button("Unload modules")
                        gr.Markdown("<div class='note'>Commands: <b>/code</b> (CAD coder) · <b>/research</b> (deep research) · <b>/image</b> (image gen) · default: chat</div>")

                with gr.Column(scale=5):
                    with gr.Tabs():
                        with gr.Tab("Sources"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>Source cards</div>")
                                sources_html = gr.HTML("<div class='sources-empty'>Run a search or ask a current-events question.</div>")
                        with gr.Tab("Trace"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>What the app did</div>")
                                trace_md = gr.Markdown("### Trace\n\n_No trace yet._")
                        with gr.Tab("Thinking"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>AI reasoning path</div>")
                                thinking_md = gr.Markdown("### Thinking Path\n\n_No thinking yet._")
                        with gr.Tab("Memory"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>Global + user memory</div>")
                                memory_global = gr.Textbox(label="Global behavior memory (admin)", placeholder="e.g. Always cite sources. Do not...", lines=6)
                                memory_user = gr.Textbox(label="Your memory (when logged in)", placeholder="Your preferences, context...", lines=6)
                                with gr.Row():
                                    save_global_btn = gr.Button("Save global")
                                    save_user_btn = gr.Button("Save my memory")
                                gr.Markdown("<div class='note'>Global: AI always follows. User: loaded when you're signed in.</div>")

            conversation_state = gr.State([])
            display_state = gr.State([])

            def save_global(root_in, text):
                root = resolve_root_path(root_in)
                save_global_memory(root, text)
                return "Saved global memory."

            def save_user_mem(root_in, text, uinfo):
                if not uinfo or not uinfo.get("id"):
                    return "Sign in to save your memory."
                root = resolve_root_path(root_in)
                save_user_memory(root, uinfo["id"], text)
                return "Saved your memory."

            def load_memory_ui(root_in, uinfo):
                root = resolve_root_path(root_in)
                g = load_global_memory(root)
                u = load_user_memory(root, uinfo["id"]) if uinfo and uinfo.get("id") else ""
                return g, u

            save_global_btn.click(fn=lambda r, t: save_global(r, t), inputs=[root_path, memory_global], outputs=status)
            save_user_btn.click(fn=save_user_mem, inputs=[root_path, memory_user, user_state], outputs=status)
            root_path.change(fn=load_memory_ui, inputs=[root_path, user_state], outputs=[memory_global, memory_user])
            demo.load(fn=load_memory_ui, inputs=[root_path, user_state], outputs=[memory_global, memory_user])

            def chat_inputs():
                return [root_path, user, conversation_state, display_state, force_web, show_traces, user_max_tokens, user_state]

            send_evt = send.click(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=[chat, conversation_state, display_state, sources_html, trace_md, thinking_md, status],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            )
            send_evt.then(lambda: "", None, user)
            user.submit(
                fn=chat_submit,
                inputs=chat_inputs(),
                outputs=[chat, conversation_state, display_state, sources_html, trace_md, thinking_md, status],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            ).then(lambda: "", None, user)

            clear.click(fn=clear_all, inputs=None, outputs=[chat, conversation_state, display_state, sources_html, trace_md, thinking_md, status, user])
            clear_modules_btn.click(fn=clear_modules, inputs=None, outputs=status)

            _hf_ok = os.getenv("SPACE_ID") or os.getenv("HF_TOKEN") or False
            try:
                from huggingface_hub import get_token
                _hf_ok = _hf_ok or get_token() is not None
            except Exception:
                pass
            if _hf_ok:
                gr.LoginButton(value="Sign in with Hugging Face")
            else:
                gr.Markdown("<div class='note'>HF Sign-in unavailable (set HF_TOKEN or run on Spaces).</div>")

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    root = resolve_root_path(None)
    print(f"[startup] Pre-loading main model from {root} with 4-bit quantization...")
    print("[startup] Other models load on demand: /code /research /image")
    preload_main_model(root)
    demo = build_ui()
    demo.launch(share=True, debug=True)
