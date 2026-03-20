"""AI Contextual UI — a Gradio chat app whose panels adapt in real time."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from engine import (
    AIResponse,
    PanelPayload,
    badges_html,
    detect_contexts,
    get_ai_response,
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg0: #06070a;
  --bg1: #0b1020;
  --panel: rgba(255,255,255,0.055);
  --panel2: rgba(255,255,255,0.035);
  --border: rgba(255,255,255,0.12);
  --text: rgba(245,247,255,0.96);
  --muted: rgba(210,220,255,0.62);
  --accent: #8ea2ff;
  --accent2: #66e0ff;
  --shadow: 0 12px 32px rgba(0,0,0,0.38);
}

/* Overall container */
.gradio-container {
  background:
    radial-gradient(1200px 800px at 10% 0%, rgba(102,224,255,0.07), transparent 45%),
    radial-gradient(1000px 700px at 100% 0%, rgba(142,162,255,0.10), transparent 48%),
    linear-gradient(180deg, var(--bg1), var(--bg0)) !important;
  color: var(--text);
}

#app-shell { max-width: 1280px; margin: 0 auto; }

/* Hero */
.hero-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 16px 20px 12px; margin-bottom: 10px; border-radius: 20px;
  background: linear-gradient(135deg,
    rgba(142,162,255,0.06) 0%,
    rgba(102,224,255,0.04) 50%,
    rgba(142,162,255,0.06) 100%);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
}
.hero-title {
  font-size: 26px; font-weight: 800; letter-spacing: 1.4px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
.ping {
  width: 10px; height: 10px; border-radius: 50%;
  background: linear-gradient(180deg, var(--accent2), var(--accent));
  box-shadow: 0 0 18px rgba(102,224,255,0.55);
  animation: pulse-ping 2s ease-in-out infinite;
}
@keyframes pulse-ping {
  0%,100% { box-shadow: 0 0 18px rgba(102,224,255,0.55); }
  50% { box-shadow: 0 0 28px rgba(102,224,255,0.9); }
}

/* Cards */
.card {
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border: 1px solid var(--border); border-radius: 18px;
  box-shadow: var(--shadow); backdrop-filter: blur(12px);
}
.pad { padding: 14px; }

/* Status */
#status-bar { color: var(--muted); font-size: 12px; margin: 0 4px 8px; }

/* Inputs */
textarea, input[type=text], input[type=password] {
  background: rgba(8,10,16,0.72) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
}
textarea:focus, input:focus {
  border-color: rgba(142,162,255,0.45) !important;
  box-shadow: 0 0 0 1px rgba(142,162,255,0.22), 0 0 16px rgba(142,162,255,0.18) !important;
}

/* Buttons */
button {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  background: linear-gradient(180deg, rgba(18,22,34,.92), rgba(12,15,24,.92)) !important;
  color: var(--text) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,.2);
}
button:hover {
  border-color: rgba(142,162,255,0.4) !important;
  box-shadow: 0 0 16px rgba(142,162,255,0.16);
}

/* Panel headers */
.panel-header {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 700; letter-spacing: 0.8px;
  text-transform: uppercase; margin-bottom: 8px;
  padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.panel-header .dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}

/* Panel animations */
.ctx-panel { animation: panel-in 0.35s ease; }
@keyframes panel-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Chatbox */
#chatbox { min-height: 480px; }

/* Welcome panel */
.welcome-content {
  text-align: center; padding: 32px 16px;
}
.welcome-content h3 {
  font-size: 18px; margin-bottom: 12px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.welcome-content p { color: var(--muted); font-size: 13px; line-height: 1.7; }
.welcome-content code {
  background: rgba(142,162,255,0.1); padding: 2px 6px; border-radius: 6px;
  font-size: 12px; color: var(--accent);
}

/* Smallcap */
.smallcap {
  font-size: 11px; color: var(--muted); letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 6px;
}
.note { color: var(--muted); font-size: 11px; }

/* Settings row */
.settings-row { display: flex; gap: 8px; align-items: center; }

@media (max-width: 768px) {
  #app-shell { padding: 6px; }
  .hero-bar { flex-direction: column; align-items: flex-start; }
}
"""

# ---------------------------------------------------------------------------
# Panel header helpers
# ---------------------------------------------------------------------------

_PANEL_META = {
    "code":     {"icon": "\U0001f4bb", "label": "Code",     "color": "#8ea2ff"},
    "data":     {"icon": "\U0001f4ca", "label": "Data",     "color": "#66e0ff"},
    "terminal": {"icon": "\u2328\ufe0f",  "label": "Terminal", "color": "#7efcc3"},
    "preview":  {"icon": "\U0001f4dd", "label": "Preview",  "color": "#ffd27a"},
    "chart":    {"icon": "\U0001f4c8", "label": "Chart",    "color": "#c08aff"},
}


def _panel_hdr(key: str) -> str:
    m = _PANEL_META.get(key, {"icon": "", "label": key, "color": "#888"})
    return (
        f"<div class='panel-header'>"
        f"<span class='dot' style='background:{m['color']}'></span>"
        f"{m['icon']} {m['label']}</div>"
    )


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_contextual_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")

    with gr.Blocks(title="AI Contextual UI") as demo:
        # ---- state ----
        chat_state = gr.State([])  # full conversation [{role, content}, ...]

        with gr.Column(elem_id="app-shell"):
            # ---- hero ----
            gr.HTML(
                "<div class='hero-bar'>"
                "  <div>"
                "    <div class='hero-title'>AI CONTEXTUAL UI</div>"
                "    <div class='hero-sub'>Talk to an AI — the interface adapts to your conversation in real time</div>"
                "  </div>"
                "  <div style='display:flex;align-items:center;gap:10px;'>"
                "    <span style='font-size:11px;color:var(--muted);'>adaptive panels</span>"
                "    <div class='ping'></div>"
                "  </div>"
                "</div>"
            )

            # ---- context badges ----
            context_html = gr.HTML(
                value="<div style='min-height:28px;'></div>",
                elem_id="context-badges",
            )

            status = gr.Markdown("Idle", elem_id="status-bar")

            # ---- main row ----
            with gr.Row():
                # ========== LEFT: chat ==========
                with gr.Column(scale=6):
                    with gr.Column(elem_classes=["card", "pad"]):
                        gr.HTML("<div class='smallcap'>Chat</div>")

                        try:
                            chat = gr.Chatbot(
                                elem_id="chatbox",
                                height=480,
                                type="messages",
                                layout="bubble",
                            )
                        except TypeError:
                            try:
                                chat = gr.Chatbot(
                                    elem_id="chatbox",
                                    height=480,
                                    type="messages",
                                )
                            except TypeError:
                                chat = gr.Chatbot(elem_id="chatbox", height=480)

                        msg_box = gr.Textbox(
                            label="Message",
                            placeholder="Ask anything — the UI will adapt to your topic…",
                            lines=2,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Accordion("Settings", open=False):
                        api_key_box = gr.Textbox(
                            label="OpenAI API Key (optional)",
                            placeholder="sk-…  Leave blank for built-in demo mode",
                            type="password",
                        )
                        gr.HTML(
                            "<div class='note'>"
                            "Built-in mode uses templates to demonstrate contextual panels. "
                            "Add an OpenAI key for full conversational AI."
                            "</div>"
                        )

                # ========== RIGHT: dynamic panels ==========
                with gr.Column(scale=6):
                    # -- Welcome (default) --
                    with gr.Column(
                        visible=True,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as welcome_panel:
                        gr.HTML(
                            "<div class='welcome-content'>"
                            "<h3>Welcome to AI Contextual UI</h3>"
                            "<p>Start a conversation and watch the panels transform.<br>"
                            "Try prompts like:</p>"
                            "<p>"
                            "<code>write a Python sort function</code><br>"
                            "<code>show me some sales data</code><br>"
                            "<code>make a bar chart</code><br>"
                            "<code>how do I set up Docker?</code><br>"
                            "<code>draft a README</code><br>"
                            "<code>explain the quadratic formula</code>"
                            "</p>"
                            "</div>"
                        )

                    # -- Code panel --
                    with gr.Column(
                        visible=False,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as code_panel:
                        gr.HTML(_panel_hdr("code"))
                        code_display = gr.Code(
                            label="",
                            language="python",
                            lines=16,
                            interactive=False,
                        )

                    # -- Data panel --
                    with gr.Column(
                        visible=False,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as data_panel:
                        gr.HTML(_panel_hdr("data"))
                        data_display = gr.Dataframe(
                            label="",
                            interactive=False,
                            wrap=True,
                        )

                    # -- Terminal panel --
                    with gr.Column(
                        visible=False,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as terminal_panel:
                        gr.HTML(_panel_hdr("terminal"))
                        terminal_display = gr.Code(
                            label="",
                            language="shell",
                            lines=12,
                            interactive=False,
                        )

                    # -- Preview panel (docs / math) --
                    with gr.Column(
                        visible=False,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as preview_panel:
                        gr.HTML(_panel_hdr("preview"))
                        preview_display = gr.Markdown("")

                    # -- Chart panel --
                    with gr.Column(
                        visible=False,
                        elem_classes=["card", "pad", "ctx-panel"],
                    ) as chart_panel:
                        gr.HTML(_panel_hdr("chart"))
                        chart_display = gr.Plot(label="")

            # ----------------------------------------------------------------
            # Event handlers
            # ----------------------------------------------------------------

            def _vis(show: bool):
                """Return a Gradio update that toggles visibility."""
                return gr.update(visible=show)

            def chat_submit(
                user_msg: str,
                history: List[Dict[str, str]],
                api_key: str,
            ):
                user_msg = (user_msg or "").strip()
                if not user_msg:
                    yield (
                        history,            # chat display (no change)
                        history,            # state
                        gr.update(),        # badges
                        _vis(True),         # welcome
                        _vis(False),        # code panel
                        gr.update(),        # code content
                        _vis(False),        # data panel
                        gr.update(),        # data content
                        _vis(False),        # terminal panel
                        gr.update(),        # terminal content
                        _vis(False),        # preview panel
                        gr.update(),        # preview content
                        _vis(False),        # chart panel
                        gr.update(),        # chart content
                        "No message entered.",
                    )
                    return

                # Build display history with typing indicator
                display = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "Thinking\u2026"},
                ]
                contexts = detect_contexts(user_msg, history)
                badges = badges_html(contexts)

                yield (
                    display, history, badges,
                    gr.update(),  # welcome — don't change yet
                    gr.update(), gr.update(),  # code
                    gr.update(), gr.update(),  # data
                    gr.update(), gr.update(),  # terminal
                    gr.update(), gr.update(),  # preview
                    gr.update(), gr.update(),  # chart
                    "Thinking\u2026",
                )

                # Get AI response
                response = get_ai_response(user_msg, history, api_key or "")

                new_history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": response.text},
                ]

                panels = response.panels
                has_any_panel = bool(panels)

                # Prepare code output
                code_p = panels.get("code")
                code_vis = code_p is not None and code_p.visible
                code_val = code_p.content if code_p else ""
                code_lang = (code_p.language if code_p else "python") or "python"

                # Prepare data output
                data_p = panels.get("data")
                data_vis = data_p is not None and data_p.visible
                if data_vis and isinstance(data_p.content, dict):
                    headers = data_p.content.get("headers", [])
                    rows = data_p.content.get("rows", [])
                    data_val = gr.update(value=rows, headers=headers)
                else:
                    data_val = gr.update()

                # Prepare terminal output
                term_p = panels.get("terminal")
                term_vis = term_p is not None and term_p.visible
                term_val = term_p.content if term_p else ""

                # Prepare preview output
                prev_p = panels.get("preview")
                prev_vis = prev_p is not None and prev_p.visible
                prev_val = prev_p.content if prev_p else ""

                # Prepare chart output
                chart_p = panels.get("chart")
                chart_vis = chart_p is not None and chart_p.visible
                chart_val = chart_p.content if chart_p else None

                ctx_label = ", ".join(response.detected_contexts) if response.detected_contexts else "general"

                yield (
                    new_history,
                    new_history,
                    badges,
                    _vis(not has_any_panel),  # welcome (hide if panels active)
                    _vis(code_vis),
                    gr.update(value=code_val, language=code_lang) if code_vis else gr.update(),
                    _vis(data_vis),
                    data_val,
                    _vis(term_vis),
                    gr.update(value=term_val) if term_vis else gr.update(),
                    _vis(prev_vis),
                    gr.update(value=prev_val) if prev_vis else gr.update(),
                    _vis(chart_vis),
                    gr.update(value=chart_val) if chart_vis else gr.update(),
                    f"Done — context: {ctx_label}",
                )

            def clear_all():
                return (
                    [],                 # chat
                    [],                 # state
                    "<div style='min-height:28px;'></div>",  # badges
                    _vis(True),         # welcome
                    _vis(False), gr.update(),  # code
                    _vis(False), gr.update(),  # data
                    _vis(False), gr.update(),  # terminal
                    _vis(False), gr.update(),  # preview
                    _vis(False), gr.update(),  # chart
                    "Idle",
                    "",                 # clear textbox
                )

            _outputs = [
                chat,           # 0
                chat_state,     # 1
                context_html,   # 2
                welcome_panel,  # 3
                code_panel,     # 4
                code_display,   # 5
                data_panel,     # 6
                data_display,   # 7
                terminal_panel, # 8
                terminal_display, # 9
                preview_panel,  # 10
                preview_display, # 11
                chart_panel,    # 12
                chart_display,  # 13
                status,         # 14
            ]

            send_evt = send_btn.click(
                fn=chat_submit,
                inputs=[msg_box, chat_state, api_key_box],
                outputs=_outputs,
            )
            send_evt.then(fn=lambda: "", inputs=None, outputs=msg_box)

            msg_box.submit(
                fn=chat_submit,
                inputs=[msg_box, chat_state, api_key_box],
                outputs=_outputs,
            ).then(fn=lambda: "", inputs=None, outputs=msg_box)

            clear_btn.click(
                fn=clear_all,
                inputs=None,
                outputs=_outputs + [msg_box],
            )

    demo.queue(default_concurrency_limit=2)
    return demo


_THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")


def launch_contextual_ui(**kwargs):
    """Build and launch with Gradio 6+ css/theme in launch()."""
    app = build_contextual_ui()
    defaults = dict(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7861")),
        share=False,
        css=CSS,
        theme=_THEME,
    )
    defaults.update(kwargs)
    # css/theme in launch() is Gradio 6+ only; strip if unsupported
    try:
        app.launch(**defaults)
    except TypeError:
        defaults.pop("css", None)
        defaults.pop("theme", None)
        app.launch(**defaults)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    launch_contextual_ui()
