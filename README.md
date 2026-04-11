# imagination-ai

Multimodal **Imagination** model stack: Gradio apps, Colab/FastAPI backends, training tooling, and an optional Next.js chat UI. This repo holds **versioned runtimes** under `versions/` plus shared `modules/` (weights and assets are not committed—see below).

## Repository layout

| Path | Role |
|------|------|
| `versions/v1.1.2/` | Gradio app entry `app.py`, runtime `imagination_runtime/` |
| `versions/v1.2.1/` | Gradio + optional QLoRA training (`train_imagination.py`) |
| `versions/v1.3/` | Colab/FastAPI backend (`colab_backend.py`, `imagination_runtime/chat_http.py`) |
| `v0-imagination-ui/` | Next.js frontend (separate git history; submodule-style nested repo) |
| `modules/` | Expected on-disk model trees (CAD, reasoning, research, image, …) |
| `.github/workflows/` | CI (e.g. `validate-v112.yml`) |

Agent-oriented notes for Cursor and automation live in [`AGENTS.md`](AGENTS.md).

## Requirements

- **Python 3.10+** for app code under `versions/`
- **Node.js 20+** (or pnpm/npm) only if you work on `v0-imagination-ui/`

## Environment: `IMAGINATION_ROOT`

Runtimes expect **`IMAGINATION_ROOT`** to point at a directory that contains the expected `modules/` layout (and `temp/` as needed). Without real weights you can use a minimal fake tree for import/UI smoke tests—see [`AGENTS.md`](AGENTS.md) and version-specific READMEs.

## Quick starts (by version)

### Gradio (v1.1.2)

```bash
cd versions/v1.1.2
pip install -r requirements.txt
export IMAGINATION_ROOT=/path/to/your/root
python app.py
```

### Gradio + training hooks (v1.2.1)

```bash
cd versions/v1.2.1
pip install -r requirements.txt
export IMAGINATION_ROOT=/path/to/your/root
python app.py
```

### FastAPI / Colab (v1.3)

From `versions/v1.3/` (after dependencies and `IMAGINATION_ROOT` are set), see `colab_backend.py` and `imagination_runtime/chat_http.py`. Typical flow: run Uvicorn on the FastAPI app and point the Next UI at the public URL via `NEXT_PUBLIC_API_URL`.

### Next.js UI (`v0-imagination-ui`)

```bash
cd v0-imagination-ui
pnpm install   # or npm install
cp .env.example .env.local   # if present; set NEXT_PUBLIC_API_URL to your backend origin
pnpm dev
```

## Contributing & license

- [Contributing](CONTRIBUTING.md)
- [License](LICENSE.md)

## Disclaimer

Imagination is a research/engineering stack. Model outputs can be wrong or unsafe; deploy with appropriate safeguards for your use case.
