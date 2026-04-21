# AGENTS.md

## Cursor Cloud specific instructions

This is a Python/Gradio AI application ("Imagination v1.1.2"). The primary runnable code lives in `versions/v1.1.2/`.

### Key paths

- **App entrypoint**: `versions/v1.1.2/app.py` (calls `build_ui()` from `imagination_v1_1_2_colab_gradio.py`)
- **Runtime library**: `versions/v1.1.2/imagination_runtime/`
- **Requirements**: `versions/v1.1.2/requirements.txt`
- **CI workflow**: `.github/workflows/validate-v112.yml`

### Running locally without model weights

The app requires an `IMAGINATION_ROOT` env var pointing to a directory with the expected module subdirectories. For local dev without actual model weights, create a fake root:

```bash
FAKE_ROOT="/workspace/.ci-fake-root"
mkdir -p "$FAKE_ROOT/modules/cad" "$FAKE_ROOT/modules/reasoning" \
         "$FAKE_ROOT/modules/research/embeddings" "$FAKE_ROOT/modules/research/reranker" \
         "$FAKE_ROOT/modules/image" "$FAKE_ROOT/temp"
export IMAGINATION_ROOT="$FAKE_ROOT"
```

### OAuth/LoginButton gotcha

The Gradio `LoginButton` (HuggingFace OAuth) requires a valid HF token for local dev mocking. Without `HF_TOKEN` set and authenticated via `huggingface-cli login`, `build_ui()` will fail with `ValueError: Your machine must be logged in to HF`. To work around this without a real token, patch `gradio.oauth._get_mocked_oauth_info` before calling `build_ui()` — see the CI-like smoke test approach below.

### Smoke tests (same as CI)

From `versions/v1.1.2/`:

```bash
# Validate runtime imports
python3 -c "
from imagination_runtime.paths import resolve_root_path, ModelPaths
from imagination_runtime.registry import get_task_specs, TaskId
from imagination_runtime.thinking import build_thinking_path, build_thinking_path_no_web
from imagination_runtime.budget import remaining_hours, is_low_budget
from imagination_runtime.auth import login_email_password
from imagination_runtime.users import load_global_memory, load_user_memory
root = resolve_root_path(None)
specs = get_task_specs()
assert len(specs) >= 4
print('Runtime modules OK')
"
```

### Launching the Gradio UI

From `versions/v1.1.2/`, with `IMAGINATION_ROOT` set:

```bash
python3 app.py
```

Chat inference will fail without real model weights (expected). The UI itself loads and is fully interactive.

### Dependencies

Install with `pip install -r versions/v1.1.2/requirements.txt` plus `pip install "gradio[oauth]"` for the OAuth extras (`itsdangerous`, `authlib`) needed by the `LoginButton`.

### Imagination v1.2.1 (Gradio + optional QLoRA training)

- **App entrypoint**: [`versions/v1.2.1/app.py`](versions/v1.2.1/app.py) (imports `build_ui` / `preload_main_model` from [`imagination_v1_2_1.py`](versions/v1.2.1/imagination_v1_2_1.py))
- **Runtime**: [`versions/v1.2.1/imagination_runtime/`](versions/v1.2.1/imagination_runtime/)
- **Requirements**: [`versions/v1.2.1/requirements.txt`](versions/v1.2.1/requirements.txt) (includes `peft`, `trl`, `datasets` for training)
- **Training script**: [`versions/v1.2.1/train_imagination.py`](versions/v1.2.1/train_imagination.py) — QLoRA on `final_dataset.jsonl` (or chat `messages` JSONL); `--model_path` should point at the same tree as `IMAGINATION_ROOT`
- **Training exports**: turns append to `IMAGINATION_ROOT/temp/training_exports/training_turns.jsonl` (schema `imagination_turn_v2`) unless `IMAGINATION_TRAINING_LOG=0`

Run the UI from `versions/v1.2.1/` with `IMAGINATION_ROOT` set: `python app.py`

### v0-imagination-ui (Next.js frontend, separate repo)

The Next app for Imagination UI lives in [`v0-imagination-ui/`](v0-imagination-ui/) and pushes to `https://github.com/JustB3Tr/v0-imagination-ui` (remote `origin` inside that folder).

**Builds and npm on Google Drive:** The monorepo path under `G:\My Drive\...` is Google Drive. `npm install` / `next build` there often fails (tar checksum/write errors). **For agents (and humans) testing a frontend build:** clone or copy `v0-imagination-ui` to a **local disk** such as `C:\dev\v0-imagination-ui` and run `npm install` and `next build` / `next dev` there instead.
