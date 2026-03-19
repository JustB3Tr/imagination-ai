# ============================================================
# IMAGINATION v1.1.1 — module-aware streamed chat (after)
#
# Goals:
# - Keep main LLM loaded persistently from IMAGINATION_ROOT
# - Lazy-load other module models when selected in UI dropdown
# - Preserve streamed output + typing cursor UX
# - Colab-friendly: mount/clone repo and set IMAGINATION_ROOT
# ============================================================

from __future__ import annotations

"""
Colab setup (copy into a cell if needed):

!pip install -q gradio transformers accelerate safetensors bitsandbytes requests beautifulsoup4 ddgs

# If using Drive:
from google.colab import drive
drive.mount("/content/drive")
!ln -s "/content/drive/MyDrive/imagination-v1.1.0" "/content/imagination-v1.1.0"

%cd /content/imagination-v1.1.0/versions/v1.1.1/subversions/after
!python imagination_v1_1_1_colab_gradio.py

For local runs, set:
  setx IMAGINATION_ROOT "f:\\imagination-v1.1.0"
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
    TextIteratorStreamer,
)

from imagination_runtime.cache import RuntimeCache
from imagination_runtime.paths import ModelPaths, resolve_root_path
from imagination_runtime.registry import TaskId, get_task_specs
from imagination_runtime.web import fetch_parallel, sources_to_cards, web_search

warnings.filterwarnings("ignore")


# ----------------------------
# App config
# ----------------------------
SEARCH_RESULTS = 6
MAX_SOURCES = 3
MAX_CHARS_PER_SOURCE = 900
REQUEST_TIMEOUT_S = 8

DEFAULT_MAX_NEW_TOKENS = 360
WEB_MAX_NEW_TOKENS = 420

SHOW_TYPING_CURSOR = True
TYPING_CURSOR = "▌"

AUTO_WEB_PATTERNS = [
    r"\blatest\b",
    r"\bcurrent\b",
    r"\btoday\b",
    r"\bnow\b",
    r"\brecent\b",
    r"\bnews\b",
    r"\bupdate\b",
    r"\bupdated\b",
    r"\brelease date\b",
    r"\bprice\b",
    r"\bcost\b",
    r"\bhow much\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bthis year\b",
    r"\bwho won\b",
    r"\branking\b",
    r"\bstandings\b",
    r"\bscores?\b",
]

TRUSTED_DOMAINS_STARTER = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "npr.org",
    "pbs.org",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "bloomberg.com",
    "theguardian.com",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "nasa.gov",
    "noaa.gov",
    "usgs.gov",
]


# ----------------------------
# Bounded caches (avoid RAM creep)
# ----------------------------
class LRUCache(OrderedDict):
    def __init__(self, max_items: int):
        super().__init__()
        self.max_items = max_items

    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        return super().get(key, default)

    def set(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            self.popitem(last=False)


SEARCH_CACHE: LRUCache = LRUCache(max_items=64)
PAGE_CACHE: LRUCache = LRUCache(max_items=128)


# ----------------------------
# Runtime caches + generation locks
# ----------------------------
RUNTIME = RuntimeCache()
GEN_LOCKS: Dict[str, Lock] = {
    "main": Lock(),
    "cad_coder": Lock(),
    "reasoning_llm": Lock(),
}


# ----------------------------
# Helpers
# ----------------------------
def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def render_live_text(text: str) -> str:
    text = clean_model_text(text)
    if SHOW_TYPING_CURSOR:
        return text + TYPING_CURSOR if text else TYPING_CURSOR
    return text


def should_auto_web(text: str, force_web: bool = False) -> Tuple[bool, str]:
    if force_web:
        return True, "manual override"
    t = (text or "").lower().strip()
    for p in AUTO_WEB_PATTERNS:
        if re.search(p, t):
            return True, f"matched pattern: {p}"
    return False, "no web trigger"


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------
# Model loading (main + modules)
# ----------------------------
def load_main_model(root_path: str) -> Tuple[Any, Any]:
    """
    Load the always-on main model from the repo root.
    """
    tok = AutoTokenizer.from_pretrained(root_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        root_path,
        device_map="auto",
        torch_dtype="auto",
    )
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return tok, model


def _load_cad_coder(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.cad_coder, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        paths.cad_coder,
        device_map="auto",
        torch_dtype="auto",
    )
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_reasoning_llm(paths: ModelPaths):
    tok = AutoTokenizer.from_pretrained(paths.reasoning_llm, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        paths.reasoning_llm,
        device_map="auto",
        torch_dtype="auto",
    )
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
    model = AutoModelForSequenceClassification.from_pretrained(
        paths.reranker,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    return {"tokenizer": tok, "model": model}


def _load_tiny_sd(paths: ModelPaths):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(paths.tiny_sd, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return {"pipeline": pipe}


def ensure_modules_loaded(task_id: TaskId, root_path: str) -> List[str]:
    """
    Load (and cache) any module objects needed for the selected task.
    Returns a list of status lines.
    """
    paths = ModelPaths(root=root_path)
    status: List[str] = []

    def ensure(key: str, loader):
        if RUNTIME.get(key) is None:
            status.append(f"- Loading `{key}`...")
            obj = loader(paths)
            RUNTIME.set(key, obj)
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


# ----------------------------
# Generation (streaming)
# ----------------------------
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
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=90.0,
    )

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


def build_system_prompt(memory_text: str, retrieved_notes: str) -> str:
    base = """
You are Imagination, a fast, factual assistant.

Rules:
- Answer directly.
- Keep answers concise unless the user asks for detail.
- If retrieved web notes are present, use them carefully.
- Cite only sources you actually used.
- If evidence conflicts, say so.
- Do not invent citations.
""".strip()

    if memory_text.strip():
        base += "\n\nPersistent memory:\n" + memory_text.strip()

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
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in conversation_state or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg.strip()})
    return messages


# ----------------------------
# Memory (per-root-path)
# ----------------------------
def memory_paths(root_path: str) -> Tuple[str, str]:
    base_dir = os.path.join(root_path, "temp", "imagination1")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "memory.txt"), os.path.join(base_dir, "trusted_sources.txt")


def load_memory(memory_file: str) -> str:
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def save_memory(memory_file: str, text: str) -> None:
    with open(memory_file, "w", encoding="utf-8") as f:
        f.write((text or "").strip())


def load_trusted(trusted_file: str) -> List[str]:
    if not os.path.exists(trusted_file):
        with open(trusted_file, "w", encoding="utf-8") as f:
            f.write("\n".join(TRUSTED_DOMAINS_STARTER) + "\n")
    with open(trusted_file, "r", encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


def render_memory_md(memory_text: str) -> str:
    if memory_text.strip():
        return "### Saved memory\n\n" + "\n".join(
            f"- {line}" for line in memory_text.splitlines() if line.strip()
        )
    return "### Saved memory\n\n_No persistent memory yet._"


# ----------------------------
# Retrieval bundle (web notes + cards + trace)
# ----------------------------
def build_retrieval_bundle(question: str, trusted_domains: List[str]) -> Tuple[str, str, str]:
    results = web_search(
        question,
        max_results=SEARCH_RESULTS,
        max_sources=MAX_SOURCES,
        trusted_domains=trusted_domains,
        cache=SEARCH_CACHE,
    )
    urls = [r["url"] for r in results]
    texts = fetch_parallel(
        urls,
        timeout_s=REQUEST_TIMEOUT_S,
        max_chars=MAX_CHARS_PER_SOURCE,
        cache=PAGE_CACHE,
    )

    source_cards: List[Dict[str, Any]] = []
    trace_lines: List[str] = []
    retrieved_docs: List[str] = []

    for i, r in enumerate(results):
        content = (texts[i] if i < len(texts) else "") or r.get("snippet") or ""
        content = re.sub(r"\s+", " ", content).strip()

        source_cards.append(
            {
                "idx": str(i + 1),
                "title": r.get("title"),
                "url": r.get("url"),
                "domain": r.get("domain"),
                "trusted": bool(r.get("trusted")),
                "snippet": content[:220],
            }
        )

        trust_label = "trusted" if r.get("trusted") else "other"
        trace_lines.append(
            f"{i+1}. **{r.get('title')}**  \n"
            f"   - domain: `{r.get('domain')}`  \n"
            f"   - trust: **{trust_label}**  \n"
            f"   - url: {r.get('url')}  \n"
            f"   - snippet: {content[:220] or '(no text extracted)'}"
        )

        retrieved_docs.append(
            f"[{i+1}] {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"Domain: {r.get('domain')}\n"
            f"Trust: {trust_label}\n"
            f"Notes: {content[:MAX_CHARS_PER_SOURCE]}"
        )

    trace_md = "### Trace\n\n"
    trace_md += "- Route: **web + model**\n"
    trace_md += f"- Query: `{question}`\n"
    trace_md += f"- Sources kept: **{len(source_cards)}**\n\n"
    trace_md += "### Source picks\n\n" + ("\n\n".join(trace_lines) if trace_lines else "_No useful sources found._")

    return "\n\n".join(retrieved_docs), sources_to_cards(source_cards), trace_md


# ----------------------------
# Chat logic
# ----------------------------
def chat_submit(
    task_label: str,
    root_path_in: str,
    user_msg: str,
    conversation_state: List[Dict[str, str]],
    display_state: List[Dict[str, str]],
    force_web: bool,
    show_traces: bool,
    user_max_tokens: int,
):
    root_path = resolve_root_path(root_path_in)
    paths = ModelPaths(root=root_path)
    memory_file, trusted_file = memory_paths(root_path)
    trusted_domains = load_trusted(trusted_file)

    user_msg = (user_msg or "").strip()
    conversation_state = conversation_state or []
    display_state = display_state or []

    if not user_msg:
        yield (
            display_state,
            conversation_state,
            display_state,
            "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>",
            "### Trace\n\n_No trace yet._",
            "⚠️ No question entered.",
        )
        return

    # Map label -> TaskId
    specs = get_task_specs()
    label_to_id = {s.label: s.id for s in specs}
    task_id = label_to_id.get(task_label, TaskId.CHAT_MAIN)

    # Ensure main model is loaded
    if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
        tok, mdl = load_main_model(paths.main_llm)
        RUNTIME.main_tokenizer = tok
        RUNTIME.main_model = mdl

    # Stable working chat: one user bubble + one assistant bubble
    working_display = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]
    sources_html = "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>"

    trace_parts: List[str] = ["### Trace", "", f"- Task: **{task_label}**", f"- Root: `{root_path}`"]
    yield working_display, conversation_state, display_state, sources_html, "\n".join(trace_parts), "⏳ Starting..."

    # Lazy load module models for this task
    load_lines = ensure_modules_loaded(task_id, root_path)
    if load_lines:
        trace_parts.append("- Module loads:")
        trace_parts.extend([f"  {ln}" for ln in load_lines])
        yield working_display, conversation_state, display_state, sources_html, "\n".join(trace_parts), "📦 Modules loaded..."

    # Web routing
    use_web, reason = should_auto_web(user_msg, force_web=force_web)
    trace_parts.append(f"- Route: **{'web + model' if use_web else 'model only'}**")
    trace_parts.append(f"- Decision reason: **{reason}**")
    trace_parts.append(f"- Query: `{user_msg}`")

    retrieved_notes = ""
    trace_md = "\n".join(trace_parts)
    max_new_tokens = int(user_max_tokens or DEFAULT_MAX_NEW_TOKENS)

    if use_web:
        yield working_display, conversation_state, display_state, sources_html, trace_md, "🌐 Searching..."
        retrieved_notes, sources_html, trace_md_full = build_retrieval_bundle(user_msg, trusted_domains)
        max_new_tokens = max(max_new_tokens, WEB_MAX_NEW_TOKENS)
        trace_md = trace_md_full if show_traces else trace_md + f"\n- Sources kept: **{MAX_SOURCES} max**"
        yield working_display, conversation_state, display_state, sources_html, trace_md, "📚 Sources ready..."

    memory_text = load_memory(memory_file)
    system_prompt = build_system_prompt(memory_text, retrieved_notes)
    messages = build_messages(conversation_state, user_msg, system_prompt)

    # Choose model for generation
    if task_id == TaskId.CAD_CODER:
        cad = RUNTIME.get("cad_coder")
        tok = cad["tokenizer"]
        mdl = cad["model"]
        lock = GEN_LOCKS["cad_coder"]
    elif task_id == TaskId.DEEP_RESEARCH:
        # For now, deep research uses the reasoning LLM for answering.
        reasoning = RUNTIME.get("reasoning_llm")
        tok = reasoning["tokenizer"]
        mdl = reasoning["model"]
        lock = GEN_LOCKS["reasoning_llm"]
    else:
        tok = RUNTIME.main_tokenizer
        mdl = RUNTIME.main_model
        lock = GEN_LOCKS["main"]

    final_text = ""
    working_display[-1] = {"role": "assistant", "content": render_live_text("")}
    yield working_display, conversation_state, display_state, sources_html, trace_md, "✍️ Generating..."

    try:
        for partial in generate_stream_chat(
            tokenizer=tok,
            model=mdl,
            messages=messages,
            max_new_tokens=max_new_tokens,
            lock=lock,
        ):
            final_text = partial
            working_display[-1] = {"role": "assistant", "content": render_live_text(partial)}
            yield working_display, conversation_state, display_state, sources_html, trace_md, "✍️ Typing..."

        final_text = clean_model_text(final_text)
        working_display[-1] = {"role": "assistant", "content": final_text}

        new_conversation_state = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": final_text}]
        new_display_state = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": final_text}]
        yield new_display_state, new_conversation_state, new_display_state, sources_html, trace_md, "✓ Done"

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}
        new_conversation_state = conversation_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        new_display_state = display_state + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": err_text}]
        yield new_display_state, new_conversation_state, new_display_state, sources_html, trace_md, f"✖ {e}"


def clear_all():
    return [], [], [], "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>", "### Trace\n\n_No trace yet._", "⟡ Idle", ""


def clear_modules():
    RUNTIME.clear_modules()
    return "Cleared module caches (main model kept loaded)."


# ----------------------------
# UI styling (same look as before)
# ----------------------------
css = """
:root{
  --bg0:#06070a;
  --bg1:#0b1020;
  --panel:rgba(255,255,255,0.065);
  --panel2:rgba(255,255,255,0.045);
  --border:rgba(255,255,255,0.14);
  --border2:rgba(255,255,255,0.08);
  --text:rgba(245,247,255,0.96);
  --muted:rgba(210,220,255,0.66);
  --accent:#8ea2ff;
  --accent2:#66e0ff;
  --shadow:0 16px 40px rgba(0,0,0,0.42);
}

.gradio-container{
  background:
    radial-gradient(1200px 800px at 10% 0%, rgba(102,224,255,0.09), transparent 45%),
    radial-gradient(1000px 700px at 100% 0%, rgba(142,162,255,0.13), transparent 48%),
    linear-gradient(180deg, var(--bg1), var(--bg0)) !important;
  color:var(--text);
}

#shell{ max-width:1200px; margin:0 auto; }

.hero{
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:18px 18px 12px 18px; margin-bottom:14px; border-radius:22px;
  background:linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  border:1px solid var(--border); box-shadow:var(--shadow); backdrop-filter: blur(12px);
}
.hero-title{ font-size:30px; font-weight:800; letter-spacing:1.6px; }
.hero-sub{ color:var(--muted); font-size:13px; letter-spacing:0.5px; margin-top:4px; }
.ping{ width:12px; height:12px; border-radius:50%;
  background:linear-gradient(180deg, var(--accent2), var(--accent));
  box-shadow:0 0 22px rgba(102,224,255,0.6);
}
.card{ background:linear-gradient(180deg, var(--panel), var(--panel2));
  border:1px solid var(--border); border-radius:20px; box-shadow:var(--shadow);
  backdrop-filter: blur(12px);
}
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
.source-badge{
  font-size:12px; color:var(--muted); border:1px solid rgba(255,255,255,0.14);
  border-radius:999px; padding:2px 8px;
}
.trust-pill{ font-size:11px; font-weight:700; letter-spacing:.7px; padding:4px 8px; border-radius:999px; }
.trust-on{ color:#bffff0; background:rgba(126,243,197,0.10); border:1px solid rgba(126,243,197,0.22); }
.trust-off{ color:#ffe7ac; background:rgba(255,210,122,0.10); border:1px solid rgba(255,210,122,0.20); }
.source-title{ font-size:14px; font-weight:700; margin-bottom:6px; }
.source-domain{ color:var(--accent2); font-size:12px; margin-bottom:8px; }
.source-snippet{ color:var(--muted); font-size:12px; line-height:1.45; }
.sources-empty{
  color:var(--muted); font-size:12px; padding:12px; border-radius:16px;
  border:1px dashed rgba(255,255,255,0.14); background:rgba(255,255,255,0.03);
}
.smallcap{ font-size:12px; color:var(--muted); letter-spacing:1px; margin-bottom:8px; text-transform:uppercase; }
.note{ color:var(--muted); font-size:12px; }
"""


def build_ui():
    task_specs = get_task_specs()
    task_labels = [s.label for s in task_specs]

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")
    with gr.Blocks(css=css, theme=theme, title="Imagination v1.1.1") as demo:
        with gr.Column(elem_id="shell"):
            gr.HTML(
                """
            <div class="hero">
              <div>
                <div class="hero-title">IMAGINATION v1.1.1</div>
                <div class="hero-sub">main model stays loaded • dropdown tasks lazy-load modules • optional web lookup • source cards • trace panel</div>
              </div>
              <div class="ping" title="online"></div>
            </div>
            """
            )

            status = gr.Markdown("⟡ Idle", elem_id="status")

            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Column(elem_classes=["card", "section-pad"]):
                        gr.Markdown("<div class='smallcap'>Chat</div>")

                        task = gr.Dropdown(
                            choices=task_labels,
                            value=task_labels[0],
                            label="Task",
                            info="Main stays loaded. Other models load only when selected.",
                        )

                        root_path = gr.Textbox(
                            label="Model root path",
                            value=os.getenv("IMAGINATION_ROOT", "/content/imagination-v1.1.0"),
                            info="Set to where your `imagination-v1.1.0` folder is mounted (Colab or local).",
                        )

                        chat = gr.Chatbot(elem_id="chatbox", height=520, buttons=["copy", "copy_all"], layout="bubble")
                        user = gr.Textbox(
                            label="Message",
                            placeholder="Ask anything… current events will auto-route through search when needed.",
                            lines=3,
                        )

                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear")
                            clear_modules_btn = gr.Button("Unload module models")

                        with gr.Row():
                            force_web = gr.Checkbox(label="Force web search", value=False)
                            show_traces = gr.Checkbox(label="Show trace details", value=True)

                        user_max_tokens = gr.Slider(
                            minimum=160,
                            maximum=700,
                            step=20,
                            value=DEFAULT_MAX_NEW_TOKENS,
                            label="Max new tokens",
                        )

                        gr.Markdown("<div class='note'>Tip: lower token count = faster answers.</div>")

                with gr.Column(scale=5):
                    with gr.Tabs():
                        with gr.Tab("Sources"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>Source cards</div>")
                                sources_html = gr.HTML("<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>")

                        with gr.Tab("Trace"):
                            with gr.Column(elem_classes=["card", "section-pad"]):
                                gr.Markdown("<div class='smallcap'>What the app did</div>")
                                trace_md = gr.Markdown("### Trace\n\n_No trace yet._")

            conversation_state = gr.State([])
            display_state = gr.State([])

            send_evt = send.click(
                fn=chat_submit,
                inputs=[task, root_path, user, conversation_state, display_state, force_web, show_traces, user_max_tokens],
                outputs=[chat, conversation_state, display_state, sources_html, trace_md, status],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            )
            send_evt.then(lambda: "", None, user)

            enter_evt = user.submit(
                fn=chat_submit,
                inputs=[task, root_path, user, conversation_state, display_state, force_web, show_traces, user_max_tokens],
                outputs=[chat, conversation_state, display_state, sources_html, trace_md, status],
                concurrency_limit=1,
                concurrency_id="chat_gpu",
            )
            enter_evt.then(lambda: "", None, user)

            clear.click(
                fn=clear_all,
                inputs=None,
                outputs=[chat, conversation_state, display_state, sources_html, trace_md, status, user],
            )

            clear_modules_btn.click(fn=clear_modules, inputs=None, outputs=status)

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True, debug=True)

