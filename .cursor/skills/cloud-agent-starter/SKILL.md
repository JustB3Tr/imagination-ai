# Cloud Agent Starter Skill: Imagination codebase

## When to use this skill
Use this skill when a Cloud agent needs to run, smoke-test, or debug this repo quickly (especially `versions/v1.1.2`).

## First 5 minutes: practical setup
1. From repo root (`/workspace`), install app deps for active runtime:
   - `cd versions/v1.1.2`
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
2. Set model root:
   - Real models: `export IMAGINATION_ROOT=/workspace`
   - Mocked root for smoke tests: `export IMAGINATION_ROOT=/tmp/imagination-fake-root`
3. Create temp dirs if mocking:
   - `mkdir -p "$IMAGINATION_ROOT"/modules/{cad,reasoning,research/embeddings,research/reranker,image} "$IMAGINATION_ROOT"/temp`
4. Start app (Spaces-style entrypoint):
   - `python app.py`

## Environment toggles and quick mocks (feature-flag-like behavior)
- `IMAGINATION_ROOT`: controls model/data root resolution in runtime path helpers.
- Hugging Face sign-in button visibility in `v1.1.2` UI is gated by any of:
  - `SPACE_ID` env var present, or
  - `HF_TOKEN` env var present, or
  - cached HF token from `huggingface_hub.get_token()`.
- Quick auth UI toggle checks:
  - Show sign-in button: `export HF_TOKEN=dummy`
  - Hide sign-in button (best effort): `unset HF_TOKEN SPACE_ID`
- Email/password auth helpers exist in runtime, but the default `v1.1.2` UI primarily exposes Hugging Face login controls.
- Google/GitHub/Apple auth providers are declared as stubs in runtime auth metadata and are not wired as active login flows.

---

## Codebase area: `versions/v1.1.2` (primary app/runtime)

### What lives here
- Main Gradio app + runtime modules (paths, auth, budget, web, user memory).
- Main runnable entrypoints:
  - `versions/v1.1.2/app.py`
  - `versions/v1.1.2/imagination_v1_1_2_colab_gradio.py`

### Workflow A: CI-style smoke test (no real model weights)
Run from `versions/v1.1.2`:
1. `export IMAGINATION_ROOT=/tmp/imagination-fake-root`
2. `mkdir -p "$IMAGINATION_ROOT"/modules/{cad,reasoning,research/embeddings,research/reranker,image} "$IMAGINATION_ROOT"/temp`
3. `python -c "from imagination_runtime.paths import resolve_root_path, ModelPaths; print(resolve_root_path(None)); print(ModelPaths(resolve_root_path(None)).cad_coder)"`
4. `python -c "from imagination_runtime.registry import get_task_specs; print(len(get_task_specs()))"`
5. `python -c "from imagination_v1_1_2_colab_gradio import build_ui; d = build_ui(); print(type(d).__name__)"`

Pass criteria:
- Imports succeed.
- Task specs return at least the four core tasks.
- `build_ui()` returns a Gradio Blocks object.

### Workflow B: app launch + manual functional checks
Run from `versions/v1.1.2`:
1. Set real root if weights exist: `export IMAGINATION_ROOT=/workspace`
2. Start app: `python app.py`
3. Open local URL and validate:
   - Send a normal chat message.
   - Send `/code ...`, `/research ...`, `/image ...` messages.
   - Toggle `Force web search` and confirm sources card behavior changes.
   - Toggle `Show trace details` and confirm trace panel updates.

Pass criteria:
- UI boots.
- Commands route to expected task modes.
- Sources/trace/thinking panes update without frontend errors.

### Workflow C: login/auth behavior check
1. `export HF_TOKEN=dummy` then run `python app.py` and confirm HF sign-in control appears.
2. `unset HF_TOKEN SPACE_ID` then rerun and confirm fallback note appears.

Pass criteria:
- Login control visibility matches env toggle.

---

## Codebase area: `versions/v1.1.1/subversions/after` (legacy runtime snapshot)

### What lives here
- Earlier refactored Gradio app with lazy module loading and similar command flow.
- Useful when comparing regressions against `v1.1.2`.

### Test workflow
Run from `versions/v1.1.1/subversions/after`:
1. `python -m pip install --upgrade pip`
2. `pip install -r requirements.txt`
3. `export IMAGINATION_ROOT=/workspace`
4. `python -c "from imagination_v1_1_1_colab_gradio import build_ui; d = build_ui(); print(type(d).__name__)"`
5. Optional full launch: `python app.py`

Pass criteria:
- Legacy UI builds; optional launch starts on configured port.

---

## Codebase area: repo-root deep research prototypes

### What lives here
- `deep_research_colab_skeleton.py`: pipeline scaffold with many explicit stubs.
- `deep_research_gradio_ui.py`: UI wrapper that streams workflow status.

### Test workflow
Run from repo root:
1. `python -m pip install --upgrade pip`
2. `pip install gradio`
3. `python -c "from deep_research_colab_skeleton import chunk_documents, FetchedPage, ResearchConfig; pages=[FetchedPage(url='u', title='t', text='abc '*800)]; print(len(chunk_documents(pages, ResearchConfig())))"`
4. `python -c "import deep_research_gradio_ui as ui; print(hasattr(ui, 'launch'))"`

Pass criteria:
- Deterministic helper functions run.
- UI module imports and exposes launch entrypoint.

Note:
- Most model-loading/research pipeline functions in the skeleton intentionally raise `NotImplementedError`; this is expected unless you implement the stubs.

---

## Codebase area: `.github/workflows` cloud validation

### What lives here
- `validate-v112.yml` mirrors a safe smoke-test path for cloud CI.

### Test workflow
- Before pushing, run the same local smoke commands from the `v1.1.2` Workflow A section.
- On push/PR, confirm GitHub Actions `Validate v1.1.2` passes.

Pass criteria:
- Local smoke checks pass first.
- CI run passes with no runtime import/build regressions.

---

## How to update this skill when new runbook knowledge appears
Keep updates short and operational:
1. Add the new trick under the relevant codebase area, not in a generic dump.
2. Record four things only:
   - Symptom/error signature.
   - Exact command(s) that reproduce.
   - Exact fix/workaround.
   - Verification command and expected output signal.
3. Prefer replacing obsolete steps instead of appending contradictory ones.
4. If a trick is broadly reusable (for all areas), also add one line to the `First 5 minutes` section.
