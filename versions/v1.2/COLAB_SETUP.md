# Imagination v1.2 — Google Colab

## Before you start

1. Put the full repo on **Google Drive** (e.g. `My Drive/imagination-v1.1.0/`), including model weights under `modules/` and the root model files.
2. Colab: **Runtime → Change runtime type → GPU**.

## Important: mount Drive in a **Python** cell first

`drive.mount()` does **not** work inside `!python colab_setup.py` (that runs a subprocess with no notebook kernel). Mount in a normal code cell, then run setup with `!python`.

## Three cells (recommended)

### Cell 1 — mount Drive only

```python
from google.colab import drive
drive.mount("/content/drive")
```

Complete the browser auth when prompted.

### Cell 2 — setup (deps + symlink)

```python
!python "/content/drive/MyDrive/imagination-v1.1.0/versions/v1.2/colab_setup.py"
```

(Adjust the path if your Drive folder name differs.)

Or:

```python
%cd /content/drive/MyDrive/imagination-v1.1.0/versions/v1.2
!python colab_setup.py
```

The script will:

- `pip install` everything from `requirements.txt`
- Mount Drive (if not already)
- Symlink `MyDrive/imagination-v1.1.0` → `/content/imagination-v1.1.0` (unless it already exists)
- Set `IMAGINATION_ROOT` for this process

**Optional env vars** (set in the cell before `!python` if needed):

```python
import os
os.environ["IMAGINATION_DRIVE_PATH"] = "/content/drive/MyDrive/imagination-v1.1.0"
os.environ["IMAGINATION_ROOT"] = "/content/imagination-v1.1.0"
os.environ["IMAGINATION_COPY"] = "1"   # copy to /content instead of symlink (faster I/O, more disk)
os.environ["SKIP_PIP"] = "1"           # skip reinstall if you already installed deps
```

### Cell 3 — launch UI

```python
%cd /content/imagination-v1.1.0/versions/v1.2
import os
os.environ["IMAGINATION_ROOT"] = "/content/imagination-v1.1.0"
!python app.py
```

Use the **Gradio public URL** printed in the output to open the UI.

## Notes

- **Stable URL**: Gradio’s share link changes each new run. Local Colab URL is always `http://localhost:7860` inside the notebook’s proxy.
- **Torch**: Colab images usually include `torch`; `requirements.txt` lists it for completeness.
