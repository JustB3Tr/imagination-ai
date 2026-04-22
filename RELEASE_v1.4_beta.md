# Imagination **v1.4 beta** — release notes

> **GitHub-style summary** of the main capabilities, changes, and bug fixes from the **v1.3** baseline through the current stack (monorepo `imagination-ai`, patch line through **`1.3.3`** on `versions/v1.3/VERSION`, plus ongoing UI submodule work).  
> Use this text as the body of a **GitHub Release** (tag suggestion: `v1.4.0-beta.1` or keep shipping patches as `v1.3.N` until you cut a formal `v1.4` tree per `AGENTS.md`).

**Scope:** `versions/v1.3/` (backend + embedded Next), **`v0-imagination-ui/`** (primary Next UI submodule), vision / CLIP projector wiring.

---

## Highlights

- **Agentic “Composer” loop** — session-backed tools (`read_file`, `write_file`, `run_shell`, `web_search`, optional **`capture_ui`**), NDJSON streaming to the UI, workspace tree, terminal + diff + summary cards, and diff apply / capture proxy routes.
- **Hardened agent runtime** — stricter tool contracts, more robust JSON/tool parsing, shell argv handling, guardrails when the model omits required fields, and clearer failure / stop behavior during long loops.
- **Chat UX** — live agent trace streaming, final-answer surfacing in the trace, backend status affordances, **IndexedDB persistence** for agent session memory (workspace / trace / diffs), and **composer chrome only for `/agent`** so normal chat and `/research` stay visually unchanged.
- **Vision (CLIP + projector)** — smarter resolution of `vision_cpt_out_real` / `IMAGINATION_VISION_PROJECTOR_DIR` so bundles next to weights **or** under the repo checkout (e.g. Google Drive layout) load without fragile env-only setup.

---

## New capabilities

| Area | What shipped |
|------|----------------|
| **Backend agent** | `agent_loop.py` + FastAPI routes: `POST /api/chat/agent`, `GET /api/agent/workspace`, `POST /api/agent/apply`, capture artifact fetch. NDJSON event types: session, workspace tree, thought, tool call/result, diff preview, media, summary, final, error. |
| **Tools** | Staged writes with diff proposals and apply path; shell with self-correction messages on failure; web search; optional Playwright-based UI capture (deps in `versions/v1.3/requirements.txt`). |
| **Frontend (Next)** | Proxy routes under `app/api/chat/agent` and `app/api/agent/*`; file tree, terminal cards, diff preview + Apply, media artifacts, summary report; math renderer support for fenced `terminal` blocks. |
| **Primary UI (`v0-imagination-ui`)** | Same agent streaming model as above; submodule kept in sync with embedded frontend where applicable. |
| **Persistence** | Agent chat memory persisted locally (IndexedDB) so workspace/trace/diff state survives refresh when supported by the UI build. |

---

## Bug fixes & reliability

- **Agent loop:** Normalize tool names; validate **`write_file`** requires **`content`**; nudge / recover when **`write_file`** is rejected; reduce “invalid JSON object” class failures via more robust parsing and validation; fail-streak / loop hygiene to avoid runaway iterations.
- **Shell:** **`run_shell`** accepts argv-style arrays / list-like commands in addition to string commands.
- **Streaming / tokens:** Floor / clamp behavior for agent **`max_new_tokens`**; larger truncation budget for tool results so traces stay usable without blowing context as easily.
- **Chat stack:** Avoid routing conflicts between legacy chat and **`/api/chat/agent`**; improved streaming yield between NDJSON lines so the UI paints incrementally.
- **CLIP / projector:** Resolve bundle under **LLM folder**, **`IMAGINATION_ROOT`**, and **checkout-inferred repo root** so `vision_cpt_out_real` (with `projector.pt` + `attach_vision_multimodal_meta.json`) is found on Colab, local Drive, or split weight layouts. Explicit **`IMAGINATION_USE_CLIP_PROJECTOR=0`** still forces native HF VLM or text-only path.

---

## Developer & ops notes

- **Docs:** `versions/v1.3/AGENTIC_COMPOSER_README.md` — implementation map for the agentic engine (backend + frontend file list, protocol sketch).
- **Versioning:** `AGENTS.md` still recommends **`v1.3.N`** patch bumps for routine changes; this **`v1.4 beta`** document is a **marketing / release-notes umbrella** for “everything since v1.3 GA” until you open a dedicated **`versions/v1.4/`** line if needed.
- **Frontend workflow:** Prefer local clone on **`C:\dev\v0-imagination-ui`** for `npm` builds; push **`v0-imagination-ui`**, then bump the submodule pointer in **`imagination-ai`** when shipping.

---

## Configuration cheat sheet

| Variable | Role |
|----------|------|
| `IMAGINATION_USE_CLIP_PROJECTOR` | Set `0` / `false` to disable CLIP+bundle path. |
| `IMAGINATION_VISION_PROJECTOR_DIR` | Absolute bundle dir, or relative to `IMAGINATION_ROOT` / model folder. |
| `IMAGINATION_ROOT` | Content root for modules and default vision bundle search. |
| `SKIP_PRELOAD` | API server: skip eager model load (see `api_server.py`). |

---

## Known limitations (beta)

- **`capture_ui`** depends on Playwright and environment allowlists; treat as optional in restricted hosts.
- Agent quality still depends on the base model and JSON adherence; backend should be on **`main`** (or equivalent) to pick up parser and tool fixes.

---

*Prepared for paste into **GitHub → Releases → Draft a new release** — title: **Imagination v1.4 beta**.*
