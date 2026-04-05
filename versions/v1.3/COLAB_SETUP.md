# Imagination v1.3 — Google Colab (full notebook setup)

## What you need on Drive

1. The full repo under **Google Drive**, e.g. `My Drive/imagination-v1.1.0/`, including:
   - `modules/` and root model files expected by Imagination
   - `versions/v1.3/` (this app, `colab_setup.py`, `app.py`, `server.py`, `colab_fullstack_orchestrator.py`, `front&back/frontend/`, …)

2. Colab: **Runtime → Change runtime type → GPU** (T4 / L4 / A100, etc.).

3. **`drive.mount()` must run in a normal Python cell**, not inside `!python …` subprocesses (those cannot open the auth popup).

---

## Path constants (adjust if your folder name differs)

| Meaning | Default on Colab |
|--------|-------------------|
| Repo on Drive | `/content/drive/MyDrive/imagination-v1.1.0` |
| Fast workspace (symlink or copy target) | `/content/imagination-v1.1.0` |
| App directory | `/content/imagination-v1.1.0/versions/v1.3` |

Override with env vars (see below) if your layout differs.

---

## Cell 1 — Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Complete the browser sign-in when prompted.

---

## Cell 2 — Environment variables (optional)

Set these **in the same notebook kernel** before any `!python` setup/orchestrator cells.

```python
import os

# --- Drive / content paths ---
os.environ["IMAGINATION_DRIVE_PATH"] = "/content/drive/MyDrive/imagination-v1.1.0"
os.environ["IMAGINATION_ROOT"] = "/content/imagination-v1.1.0"

# Copy repo to /content for faster I/O (uses more disk); omit for symlink only
# os.environ["IMAGINATION_COPY"] = "1"

# Skip pip if you already installed requirements this session
# os.environ["SKIP_PIP"] = "1"

# --- Imagination server (Gradio + FastAPI) ---
os.environ["PORT"] = "7860"          # AI backend (app.py)
# os.environ["SKIP_PRELOAD"] = "1"   # load main model on first message instead
# os.environ["GRADIO_SHARE"] = "true"  # public *.gradio.live for port 7860 (optional)

# --- Full stack only (orchestrator): Flask + Next ---
os.environ["FLASK_BRIDGE_PORT"] = "5000"
os.environ["NEXT_PORT"] = "3000"
# os.environ["AI_READY_TIMEOUT_S"] = "7200"   # max wait for preload + /api/health
# os.environ["COLAB_STACK_LOG_DIR"] = "/tmp/imagination_colab_stack_logs"
```

---

## Cell 3 — Python dependencies + Drive → `/content` link

Runs `colab_setup.py`: `pip install -r versions/v1.3/requirements.txt`, mounts Drive if needed, symlinks (or copies) the repo into `IMAGINATION_ROOT`, sets `IMAGINATION_ROOT` for child processes.

```python
!python "/content/drive/MyDrive/imagination-v1.1.0/versions/v1.3/colab_setup.py"
```

Or:

```python
%cd /content/drive/MyDrive/imagination-v1.1.0/versions/v1.3
!python colab_setup.py
```

---

## Choose one path after setup

### A) Gradio + FastAPI only (single process, simplest)

Main model preloads **before** the server listens on `PORT`.

```python
%cd /content/imagination-v1.1.0/versions/v1.3
import os
os.environ.setdefault("IMAGINATION_ROOT", "/content/imagination-v1.1.0")
!python app.py
```

Use the printed **local** URL or **`*.gradio.live`** if `GRADIO_SHARE` is enabled. In Colab you can also use the **ports** UI for `7860`.

**One cell after mount** (setup + then `app.py` in a subprocess — same as running `app.py` yourself after Cell 3):

```python
!python "/content/drive/MyDrive/imagination-v1.1.0/versions/v1.3/colab_setup.py" --launch
```

---

### B) Full stack: AI (`app.py`) + Flask (`server.py`) + Next.js

Requires **Node.js** and **`npm install`** in the frontend once per runtime (or after `package.json` changes).

#### Cell 4a — Install Node (if `!node -v` fails)

```python
!apt-get update -qq && apt-get install -qq -y nodejs npm
!node -v && npm -v
```

#### Cell 4b — Frontend dependencies (**required** for the orchestrator)

Without this, Next will fail with **`sh: 1: next: not found`** (no local `node_modules/.bin/next`).

```python
%cd "/content/imagination-v1.1.0/versions/v1.3/front&back/frontend"
!npm install
```

#### Cell 5 — Orchestrator (one blocking cell)

Starts **AI → Flask → Next** in order, waits until **`/api/health`** on the AI port succeeds, prints Colab **`eval_js` proxy URL** for Next when available, streams **Next.js** logs until you interrupt.

```python
%cd /content/imagination-v1.1.0/versions/v1.3
import os
os.environ.setdefault("IMAGINATION_ROOT", "/content/imagination-v1.1.0")
!python colab_fullstack_orchestrator.py
```

**Interrupt the cell** (stop button) to shut down all three processes.

**Logs:**

- AI (stdout/stderr): `/tmp/imagination_colab_stack_logs/ai_backend.log`
- Flask: `/tmp/imagination_colab_stack_logs/flask_bridge.log`
- Next: streamed in the notebook

**URLs (typical):**

| Service | Port | Notes |
|--------|------|--------|
| Imagination (Gradio + FastAPI) | `7860` | Colab “open port” or `GRADIO_SHARE` |
| Flask bridge | `5000` | `GET /health`, `GET /api/health`, proxy ` /api/upstream/...` |
| Next dev | `3000` | Orchestrator prints `google.colab.kernel.proxyPort(3000)` when it works |

Point the Next app at the Flask bridge (e.g. `NEXT_PUBLIC_*` or server routes) as you wire chat APIs.

---

## Environment variable reference

| Variable | Purpose |
|----------|---------|
| `IMAGINATION_DRIVE_PATH` | Repo path **on Drive** (source for symlink/copy). |
| `IMAGINATION_ROOT` | Repo path used at runtime (usually `/content/imagination-v1.1.0`). |
| `IMAGINATION_COPY` | `1` = copy Drive → `/content`; default = symlink. |
| `SKIP_PIP` | `1` = skip `pip install -r requirements.txt` in `colab_setup.py`. |
| `PORT` | AI backend HTTP port (`app.py`), default `7860`. |
| `SKIP_PRELOAD` | `1` = defer main model load until first use. |
| `GRADIO_SHARE` | `true`/`false` — public Gradio tunnel for AI port (see `app.py`). |
| `FLASK_BRIDGE_PORT` | Flask bridge (default `5000`). |
| `NEXT_PORT` | Next dev server (default `3000`). |
| `IMAGINATION_AI_BACKEND_URL` | Flask → AI base URL (default `http://127.0.0.1:7860`). |
| `AI_READY_TIMEOUT_S` | Max seconds to wait for AI `/api/health` (orchestrator). |
| `AI_WAIT_STATUS_INTERVAL_S` | Seconds between heartbeat prints + log tail while waiting (default `30`). |
| `COLAB_STACK_LOG_DIR` | Directory for AI/Flask log files (orchestrator). |

Other tuning (tokens, coder, training log, etc.) matches `colab_setup.py` docstring and `app.py` / runtime env docs.

---

## Syncing edits from your PC

See [DRIVE_SYNC.md](DRIVE_SYNC.md) (Drive for desktop, rclone, or Git).

---

## `next: not found` when the orchestrator starts Next

You skipped **`npm install`** in `front&back/frontend`, so there is no `node_modules/.bin/next`. Run **Cell 4b**, then run the orchestrator again.

The orchestrator now calls that binary **directly** (instead of `npm run dev`) so Colab’s shell does not need `next` on the global PATH.

---

## Orchestrator looks stuck after “AI log: …”

That step **only waits** until `http://127.0.0.1:7860/api/health` responds. **`app.py` preloads the whole main model first**, which often takes **15–45+ minutes** on Colab before any port opens — the notebook used to print nothing during that wait.

Updated `colab_fullstack_orchestrator.py` prints a **heartbeat every 30s** and **tails `ai_backend.log`**. If `app.py` **crashes**, you should see **exit code + log tail** immediately.

To watch the log in another cell while the orchestrator runs:  
`!tail -f /tmp/imagination_colab_stack_logs/ai_backend.log`

---

## Notes

- **Torch**: Colab images usually include `torch`; `requirements.txt` still pins it for reproducibility.
- **Session lifetime**: Stopping the runtime or closing Colab stops all servers; ngrok/Gradio links from past runs will not stay valid.
- **Model lifecycle** (on-demand modules) is enforced inside the **Gradio app** via `imagination_runtime/model_lifecycle.py`, not in Flask; the bridge is HTTP only.
