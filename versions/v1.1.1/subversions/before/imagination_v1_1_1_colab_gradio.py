# ============================================================
# IMAGINATION v1.1.1 — safer streamed chat + live typing effect
# Assumes `tokenizer` and `model` are ALREADY loaded above.
# ============================================================

import os
import re
import html
import time
import warnings
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import torch
import requests
import gradio as gr

from bs4 import BeautifulSoup
from ddgs import DDGS
from transformers import TextIteratorStreamer

warnings.filterwarnings("ignore")

# ----------------------------
# Safety/defaults for tokenizer
# ----------------------------
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Config
# ----------------------------
BASE_DIR = "/content/drive/MyDrive/imagination1"
MEMORY_FILE = os.path.join(BASE_DIR, "memory.txt")
TRUSTED_FILE = os.path.join(BASE_DIR, "trusted_sources.txt")

SEARCH_RESULTS = 6
MAX_SOURCES = 3
MAX_CHARS_PER_SOURCE = 900
REQUEST_TIMEOUT = 8

DEFAULT_MAX_NEW_TOKENS = 360
WEB_MAX_NEW_TOKENS = 420

# Live typing behavior
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

os.makedirs(BASE_DIR, exist_ok=True)

# ----------------------------
# Persistent memory
# ----------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def save_memory(text):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write((text or "").strip())


MEMORY = load_memory()


def render_memory_md():
    global MEMORY
    if MEMORY.strip():
        return "### Saved memory\n\n" + "\n".join(
            f"- {line}" for line in MEMORY.splitlines() if line.strip()
        )
    return "### Saved memory\n\n_No persistent memory yet._"


def add_memory(note):
    global MEMORY
    note = (note or "").strip()
    if not note:
        return render_memory_md(), "", "⚠️ No memory text entered."
    MEMORY = (MEMORY + "\n" + note).strip() if MEMORY else note
    save_memory(MEMORY)
    return render_memory_md(), "", "💾 Memory saved."


def clear_memory():
    global MEMORY
    MEMORY = ""
    save_memory(MEMORY)
    return render_memory_md(), "", "🧹 Memory cleared."


# ----------------------------
# Trusted sources
# ----------------------------
def load_trusted():
    if not os.path.exists(TRUSTED_FILE):
        starter = [
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
            "fifa.com",
            "uefa.com",
            "mlb.com",
            "nba.com",
            "nfl.com",
            "nhl.com",
            "olympics.com",
            "espn.com",
            "grammy.com",
            "billboard.com",
            "sec.gov",
            "federalreserve.gov",
            "treasury.gov",
            "worldbank.org",
            "imf.org",
            "oecd.org",
        ]
        with open(TRUSTED_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(starter) + "\n")

    with open(TRUSTED_FILE, "r", encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]


TRUSTED_DOMAINS = load_trusted()


def get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def trusted(url):
    d = get_domain(url)
    return any(d.endswith(x) for x in TRUSTED_DOMAINS)


# ----------------------------
# Tiny caches
# ----------------------------
SEARCH_CACHE = {}
PAGE_CACHE = {}


# ----------------------------
# Search + fetch
# ----------------------------
def web_search(query):
    key = query.strip().lower()
    if key in SEARCH_CACHE:
        return SEARCH_CACHE[key]

    results = []
    seen = set()

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=SEARCH_RESULTS):
            url = (r.get("href") or "").strip()
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or "").strip()

            if not url or url in seen:
                continue
            seen.add(url)

            results.append(
                {
                    "url": url,
                    "title": title or get_domain(url) or "Untitled",
                    "snippet": snippet,
                    "trusted": trusted(url),
                    "domain": get_domain(url),
                }
            )

    results.sort(key=lambda x: (not x["trusted"], x["domain"], x["title"]))
    trimmed = results[:MAX_SOURCES]
    SEARCH_CACHE[key] = trimmed
    return trimmed


def fetch_page(url):
    if url in PAGE_CACHE:
        return PAGE_CACHE[url]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            PAGE_CACHE[url] = ""
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "noscript",
                "svg",
            ]
        ):
            tag.decompose()

        chunks = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                chunks.append(txt)

        text = " ".join(chunks)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:MAX_CHARS_PER_SOURCE]

        PAGE_CACHE[url] = text
        return text
    except Exception:
        PAGE_CACHE[url] = ""
        return ""


def fetch_parallel(urls):
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as exe:
        return list(exe.map(fetch_page, urls))


# ----------------------------
# Helpers
# ----------------------------
def default_sources_html():
    return "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>"


def default_trace_md():
    return "### Trace\n\n_No trace yet._"


def sources_to_cards(sources):
    if not sources:
        return default_sources_html()

    cards = []
    for s in sources:
        trust_badge = "TRUSTED" if s["trusted"] else "OTHER"
        title = html.escape(s["title"])
        url = html.escape(s["url"])
        domain = html.escape(s["domain"])
        snippet = html.escape((s.get("snippet") or "")[:180])

        cards.append(
            f"""
        <a class="source-card" href="{url}" target="_blank" rel="noopener">
          <div class="source-row">
            <div class="source-badge">[{s["idx"]}]</div>
            <div class="trust-pill {'trust-on' if s['trusted'] else 'trust-off'}">{trust_badge}</div>
          </div>
          <div class="source-title">{title}</div>
          <div class="source-domain">{domain}</div>
          <div class="source-snippet">{snippet}</div>
        </a>
        """
        )

    return "<div class='sources-grid'>" + "\n".join(cards) + "</div>"


def should_auto_web(text, force_web=False):
    if force_web:
        return True, "manual override"

    t = (text or "").lower().strip()
    for p in AUTO_WEB_PATTERNS:
        if re.search(p, t):
            return True, f"matched pattern: {p}"
    return False, "no web trigger"


def build_system_prompt():
    base = """
You are Imagination, a fast, factual assistant.

Rules:
- Answer directly.
- Keep answers concise unless the user asks for detail.
- If retrieved web notes are present, use them carefully.
- Cite only sources you actually used.
- If evidence conflicts, say so.
- Do not invent citations.
- Do not ask follow-up questions unless absolutely necessary.
""".strip()

    if MEMORY.strip():
        return base + "\n\nPersistent memory:\n" + MEMORY.strip()
    return base


def clean_model_text(text):
    text = (text or "").strip()
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def render_live_text(text):
    text = clean_model_text(text)
    if SHOW_TYPING_CURSOR:
        return text + TYPING_CURSOR if text else TYPING_CURSOR
    return text


def get_model_input_device():
    if hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, str) and dev not in ("cpu", "disk"):
                return torch.device(dev)
    return next(model.parameters()).device


def build_model_messages(conversation_state, user_msg, retrieved_notes=""):
    system_prompt = build_system_prompt().strip()

    if retrieved_notes.strip():
        system_prompt += (
            "\n\nRetrieved web notes:\n"
            "Use these only if relevant. Cite inline as [1], [2], etc. only for sources actually used. "
            "End with a short 'Sources used' list containing only the URLs actually used.\n\n"
            f"{retrieved_notes}"
        )

    messages = [{"role": "system", "content": system_prompt}]

    for m in conversation_state or []:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_msg.strip()})
    return messages


# ----------------------------
# Streaming generation
# ----------------------------
def generate_stream(messages, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    device = get_model_input_device()
    model.eval()

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
        timeout=60.0,
    )

    gen_error = {"exc": None}

    def _run_generation():
        try:
            with torch.inference_mode():
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

    thread = Thread(target=_run_generation, daemon=True)
    thread.start()

    partial = ""

    try:
        for chunk in streamer:
            if gen_error["exc"] is not None:
                raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error[
                    "exc"
                ]

            partial += chunk
            yield clean_model_text(partial)

    except Exception:
        if gen_error["exc"] is not None:
            raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error[
                "exc"
            ]
        raise

    finally:
        thread.join(timeout=0.2)

    if gen_error["exc"] is not None:
        raise RuntimeError(f"Generation failed: {gen_error['exc']}") from gen_error["exc"]


# ----------------------------
# RAG helpers
# ----------------------------
def build_retrieval_bundle(question):
    results = web_search(question)
    urls = [r["url"] for r in results]
    texts = fetch_parallel(urls) if urls else []

    source_cards = []
    trace_lines = []
    retrieved_docs = []

    for i, r in enumerate(results):
        content = (texts[i] if i < len(texts) else "") or r["snippet"] or ""
        content = re.sub(r"\s+", " ", content).strip()

        source_cards.append(
            {
                "idx": str(i + 1),
                "title": r["title"],
                "url": r["url"],
                "domain": r["domain"],
                "trusted": r["trusted"],
                "snippet": content[:220],
            }
        )

        trust_label = "trusted" if r["trusted"] else "other"
        trace_lines.append(
            f"{i+1}. **{r['title']}**  \n"
            f"   - domain: `{r['domain']}`  \n"
            f"   - trust: **{trust_label}**  \n"
            f"   - url: {r['url']}  \n"
            f"   - snippet: {content[:220] or '(no text extracted)'}"
        )

        retrieved_docs.append(
            f"[{i+1}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Domain: {r['domain']}\n"
            f"Trust: {trust_label}\n"
            f"Notes: {content[:MAX_CHARS_PER_SOURCE]}"
        )

    trace_md = "### Trace\n\n"
    trace_md += f"- Route: **web + model**\n"
    trace_md += f"- Query: `{question}`\n"
    trace_md += f"- Sources kept: **{len(source_cards)}**\n\n"
    if trace_lines:
        trace_md += "### Source picks\n\n" + "\n\n".join(trace_lines)
    else:
        trace_md += "_No useful sources found._"

    return "\n\n".join(retrieved_docs), sources_to_cards(source_cards), trace_md, source_cards


# ----------------------------
# Chat logic
# ----------------------------
def chat_submit(
    user_msg,
    conversation_state,
    display_state,
    force_web,
    show_traces,
    user_max_tokens,
):
    user_msg = (user_msg or "").strip()
    conversation_state = conversation_state or []
    display_state = display_state or []

    if not user_msg:
        yield (
            display_state,
            conversation_state,
            display_state,
            default_sources_html(),
            default_trace_md(),
            "⚠️ No question entered.",
        )
        return

    # Stable working chat: one user bubble + one assistant bubble
    working_display = display_state + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]

    sources_html = default_sources_html()
    trace_parts = []

    yield (
        working_display,
        conversation_state,
        display_state,
        sources_html,
        default_trace_md(),
        "⏳ Starting...",
    )

    use_web, reason = should_auto_web(user_msg, force_web=force_web)
    trace_parts.append("### Trace")
    trace_parts.append("")
    trace_parts.append(f"- Route: **{'web + model' if use_web else 'model only'}**")
    trace_parts.append(f"- Decision reason: **{reason}**")
    trace_parts.append(f"- Query: `{user_msg}`")

    retrieved_notes = ""
    max_new_tokens = int(user_max_tokens or DEFAULT_MAX_NEW_TOKENS)

    try:
        if use_web:
            yield (
                working_display,
                conversation_state,
                display_state,
                sources_html,
                "\n".join(trace_parts),
                "🌐 Searching...",
            )

            retrieved_notes, sources_html, trace_md_full, _ = build_retrieval_bundle(
                user_msg
            )
            max_new_tokens = max(max_new_tokens, WEB_MAX_NEW_TOKENS)

            if show_traces:
                trace_md = trace_md_full
            else:
                trace_md = "\n".join(
                    trace_parts + ["", f"- Sources kept: **{MAX_SOURCES} max**"]
                )

            yield (
                working_display,
                conversation_state,
                display_state,
                sources_html,
                trace_md,
                "📚 Sources ready...",
            )
        else:
            trace_md = "\n".join(trace_parts)
            yield (
                working_display,
                conversation_state,
                display_state,
                sources_html,
                trace_md,
                "🧠 Model only...",
            )

        messages = build_model_messages(conversation_state, user_msg, retrieved_notes)

        final_text = ""
        working_display[-1] = {"role": "assistant", "content": render_live_text("")}
        yield (
            working_display,
            conversation_state,
            display_state,
            sources_html,
            trace_md,
            "✍️ Generating...",
        )

        for partial in generate_stream(messages, max_new_tokens=max_new_tokens):
            final_text = partial
            working_display[-1] = {
                "role": "assistant",
                "content": render_live_text(partial),
            }
            yield (
                working_display,
                conversation_state,
                display_state,
                sources_html,
                trace_md,
                "✍️ Typing...",
            )

        final_text = clean_model_text(final_text)
        working_display[-1] = {"role": "assistant", "content": final_text}

        new_conversation_state = conversation_state + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": final_text},
        ]

        new_display_state = display_state + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": final_text},
        ]

        yield (
            new_display_state,
            new_conversation_state,
            new_display_state,
            sources_html,
            trace_md,
            "✓ Done",
        )

    except Exception as e:
        err_text = f"Error: {e}"
        working_display[-1] = {"role": "assistant", "content": err_text}

        new_conversation_state = conversation_state + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": err_text},
        ]

        new_display_state = display_state + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": err_text},
        ]

        yield (
            new_display_state,
            new_conversation_state,
            new_display_state,
            sources_html,
            trace_md if "trace_md" in locals() else default_trace_md(),
            f"✖ {e}",
        )


def clear_all():
    return [], [], [], default_sources_html(), default_trace_md(), "⟡ Idle", ""


# ----------------------------
# UI styling
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
  --good:#7ef3c5;
  --warn:#ffd27a;
  --shadow:0 16px 40px rgba(0,0,0,0.42);
}

.gradio-container{
  background:
    radial-gradient(1200px 800px at 10% 0%, rgba(102,224,255,0.09), transparent 45%),
    radial-gradient(1000px 700px at 100% 0%, rgba(142,162,255,0.13), transparent 48%),
    linear-gradient(180deg, var(--bg1), var(--bg0)) !important;
  color:var(--text);
}

#shell{
  max-width:1200px;
  margin:0 auto;
}

.hero{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding:18px 18px 12px 18px;
  margin-bottom:14px;
  border-radius:22px;
  background:linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  border:1px solid var(--border);
  box-shadow:var(--shadow);
  backdrop-filter: blur(12px);
}

.hero-title{
  font-size:30px;
  font-weight:800;
  letter-spacing:1.6px;
}

.hero-sub{
  color:var(--muted);
  font-size:13px;
  letter-spacing:0.5px;
  margin-top:4px;
}

.ping{
  width:12px;
  height:12px;
  border-radius:50%;
  background:linear-gradient(180deg, var(--accent2), var(--accent));
  box-shadow:0 0 22px rgba(102,224,255,0.6);
  animation:pulse 1.8s ease-in-out infinite;
}
@keyframes pulse{
  0%{transform:scale(0.95);opacity:.7}
  50%{transform:scale(1.28);opacity:1}
  100%{transform:scale(0.95);opacity:.7}
}

.card{
  background:linear-gradient(180deg, var(--panel), var(--panel2));
  border:1px solid var(--border);
  border-radius:20px;
  box-shadow:var(--shadow);
  backdrop-filter: blur(12px);
}

.section-pad{ padding:14px; }

#status{
  margin:2px 2px 12px 2px;
  color:var(--muted);
  font-size:13px;
}

#chatbox{
  min-height:520px;
}

textarea, input{
  background:rgba(8,10,16,0.72) !important;
  color:var(--text) !important;
  border:1px solid var(--border2) !important;
  border-radius:14px !important;
}

textarea:focus, input:focus{
  border-color:rgba(142,162,255,0.45) !important;
  box-shadow:0 0 0 1px rgba(142,162,255,0.24), 0 0 18px rgba(142,162,255,0.20) !important;
}

button{
  border-radius:14px !important;
  border:1px solid rgba(255,255,255,0.18) !important;
  background:linear-gradient(180deg, rgba(18,22,34,.92), rgba(12,15,24,.92)) !important;
  color:var(--text) !important;
  box-shadow:0 8px 22px rgba(0,0,0,.22);
}

button:hover{
  border-color:rgba(142,162,255,0.42) !important;
  box-shadow:0 0 18px rgba(142,162,255,0.18);
}

.pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border:1px solid var(--border2);
  border-radius:999px;
  padding:6px 10px;
  color:var(--muted);
  font-size:12px;
  background:rgba(255,255,255,0.04);
}

.sources-grid{
  display:grid;
  grid-template-columns:1fr;
  gap:10px;
}
.source-card{
  display:block;
  text-decoration:none;
  color:var(--text);
  padding:12px;
  border-radius:16px;
  background:rgba(10,13,20,0.70);
  border:1px solid rgba(255,255,255,0.10);
  transition:.16s ease;
}
.source-card:hover{
  transform:translateY(-1px);
  border-color:rgba(142,162,255,0.35);
  box-shadow:0 0 18px rgba(142,162,255,0.14);
}
.source-row{
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:8px;
}
.source-badge{
  font-size:12px;
  color:var(--muted);
  border:1px solid rgba(255,255,255,0.14);
  border-radius:999px;
  padding:2px 8px;
}
.trust-pill{
  font-size:11px;
  font-weight:700;
  letter-spacing:.7px;
  padding:4px 8px;
  border-radius:999px;
}
.trust-on{
  color:#bffff0;
  background:rgba(126,243,197,0.10);
  border:1px solid rgba(126,243,197,0.22);
}
.trust-off{
  color:#ffe7ac;
  background:rgba(255,210,122,0.10);
  border:1px solid rgba(255,210,122,0.20);
}
.source-title{
  font-size:14px;
  font-weight:700;
  margin-bottom:6px;
}
.source-domain{
  color:var(--accent2);
  font-size:12px;
  margin-bottom:8px;
}
.source-snippet{
  color:var(--muted);
  font-size:12px;
  line-height:1.45;
}

.sources-empty{
  color:var(--muted);
  font-size:12px;
  padding:12px;
  border-radius:16px;
  border:1px dashed rgba(255,255,255,0.14);
  background:rgba(255,255,255,0.03);
}

.smallcap{
  font-size:12px;
  color:var(--muted);
  letter-spacing:1px;
  margin-bottom:8px;
  text-transform:uppercase;
}

.note{
  color:var(--muted);
  font-size:12px;
}
"""


# ----------------------------
# UI
# ----------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(css=css, theme=theme, title="Imagination v1.1.1") as demo:
    with gr.Column(elem_id="shell"):
        gr.HTML(
            """
        <div class="hero">
          <div>
            <div class="hero-title">IMAGINATION v1.1.1</div>
            <div class="hero-sub">live token streaming • live typing effect • optional web lookup • source cards • trace panel</div>
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
                    chat = gr.Chatbot(
                        elem_id="chatbox",
                        height=520,
                        buttons=["copy", "copy_all"],
                        layout="bubble",
                    )

                    user = gr.Textbox(
                        label="Message",
                        placeholder="Ask anything… current events will auto-route through search when needed.",
                        lines=3,
                    )

                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear")

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

                    gr.Markdown(
                        "<div class='note'>Tip: lower token count = faster answers. Most normal replies feel good around 280–420.</div>"
                    )

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Sources"):
                        with gr.Column(elem_classes=["card", "section-pad"]):
                            gr.Markdown("<div class='smallcap'>Source cards</div>")
                            sources_html = gr.HTML(default_sources_html())

                    with gr.Tab("Trace"):
                        with gr.Column(elem_classes=["card", "section-pad"]):
                            gr.Markdown("<div class='smallcap'>What the app did</div>")
                            trace_md = gr.Markdown(default_trace_md())

                    with gr.Tab("Memory"):
                        with gr.Column(elem_classes=["card", "section-pad"]):
                            gr.Markdown("<div class='smallcap'>Persistent memory</div>")
                            memory_view = gr.Markdown(render_memory_md())
                            memory_input = gr.Textbox(
                                label="Add memory",
                                placeholder="Example: Prefer official sources. Keep answers concise. Cite URLs when web is used.",
                                lines=5,
                            )
                            with gr.Row():
                                save_mem_btn = gr.Button("Save memory")
                                clear_mem_btn = gr.Button("Clear memory")

        conversation_state = gr.State([])
        display_state = gr.State([])

        send_evt = send.click(
            fn=chat_submit,
            inputs=[
                user,
                conversation_state,
                display_state,
                force_web,
                show_traces,
                user_max_tokens,
            ],
            outputs=[chat, conversation_state, display_state, sources_html, trace_md, status],
            concurrency_limit=1,
            concurrency_id="chat_gpu",
        )
        send_evt.then(lambda: "", None, user)

        enter_evt = user.submit(
            fn=chat_submit,
            inputs=[
                user,
                conversation_state,
                display_state,
                force_web,
                show_traces,
                user_max_tokens,
            ],
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

        save_mem_btn.click(
            fn=add_memory,
            inputs=[memory_input],
            outputs=[memory_view, memory_input, status],
        )

        clear_mem_btn.click(
            fn=clear_memory,
            inputs=None,
            outputs=[memory_view, memory_input, status],
        )

demo.queue(default_concurrency_limit=1)
demo.launch(share=True, debug=True)

