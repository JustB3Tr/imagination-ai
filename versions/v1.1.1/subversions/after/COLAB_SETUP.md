# Colab setup (GPU) for Imagination v1.1.1

This runbook assumes your project folder exists as `imagination-v1.1.0` and contains:
- `versions/v1.1.1/subversions/after/imagination_v1_1_1_colab_gradio.py`
- `config.json`, `tokenizer.json`, and `model-0000*-of-00002.safetensors` at the repo root
- `modules/` folder with the finetuned sub-models

## 1) Create a notebook
1. Open [Colab](https://colab.research.google.com)
2. Create a new notebook
3. Select GPU: `Runtime -> Change runtime type -> GPU -> Save`

## 2) Cell: install dependencies
```bash
!pip install -q gradio transformers accelerate safetensors bitsandbytes requests beautifulsoup4 ddgs diffusers
```

## 3) Cell: mount Drive + expose repo at `/content/imagination-v1.1.0`
Use Option A or B.

### Option A (recommended): symlink from Drive
```python
from google.colab import drive
drive.mount("/content/drive")

import os
repo_drive_path = "/content/drive/MyDrive/imagination-v1.1.0"

# Create /content/imagination-v1.1.0 -> Drive-backed folder
if os.path.islink("/content/imagination-v1.1.0") or os.path.exists("/content/imagination-v1.1.0"):
    pass
else:
    !ln -s "$repo_drive_path" "/content/imagination-v1.1.0"
```

### Option B: copy to Colab local disk (faster reads)
```python
from google.colab import drive
drive.mount("/content/drive")

!rm -rf /content/imagination-v1.1.0
!cp -r "/content/drive/MyDrive/imagination-v1.1.0" "/content/imagination-v1.1.0"
```

## 4) Cell: set ROOT path (optional)
Your UI has a “Model root path” textbox and also uses `IMAGINATION_ROOT` env var.

```python
import os
os.environ["IMAGINATION_ROOT"] = "/content/imagination-v1.1.0"
```

## 5) Cell: launch the Gradio app
```bash
%cd /content/imagination-v1.1.0/versions/v1.1.1/subversions/after
!python imagination_v1_1_1_colab_gradio.py
```

## Notes
- The script keeps the **main model** loaded and lazily loads module models when you pick a task from the dropdown.
- If you see CUDA/bitsandbytes issues, you can remove `bitsandbytes` from the install cell and retry (the main path works without it for many setups).

