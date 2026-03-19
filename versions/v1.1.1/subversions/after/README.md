# v1.1.1 — subversion (after)

This folder contains the **refactored** Google Colab + Gradio app:
- Keeps the **main model** loaded persistently
- Lazy-loads **module models** on demand (dropdown task selection)
- Includes bounded caches and safer streaming
- Includes Hugging Face Spaces entrypoint files for a **stable tester URL**

## Google Colab quick start

### Option A: Mount Google Drive (recommended if your models are in Drive)
1. Put `imagination-v1.1.0/` on Drive.
2. In Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content
!ln -s "/content/drive/MyDrive/imagination-v1.1.0" "/content/imagination-v1.1.0"
```

### Option B: Upload / zip (good for sharing a snapshot)
- Upload a zip of `imagination-v1.1.0` and unzip to `/content/imagination-v1.1.0`.

### Install deps + run

```python
!pip install -q gradio transformers accelerate safetensors bitsandbytes requests beautifulsoup4 ddgs
%cd /content/imagination-v1.1.0/versions/v1.1.1/subversions/after
!python imagination_v1_1_1_colab_gradio.py
```

If you mounted the repo somewhere else, set:
- `IMAGINATION_ROOT` env var, or
- the **Model root path** textbox in the UI.

