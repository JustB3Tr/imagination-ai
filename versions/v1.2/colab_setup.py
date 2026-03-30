#!/usr/bin/env python3
"""
Imagination v1.2 — Google Colab one-shot setup.

Colab (recommended — two steps):

    # Cell 1 — run as Python (not !python), then authenticate:
    from google.colab import drive
    drive.mount("/content/drive")

    # Cell 2 — setup (subprocess cannot call drive.mount):
    !python colab_setup.py

Then either:

    !python app.py

Or one-shot (setup + load weights + Gradio in the **same** process — recommended if you want
everything in one cell after mounting Drive):

    !python colab_setup.py --launch

Note: `!python colab_setup.py` and `!python app.py` are **different processes** — weights loaded
in setup alone are **not** visible to app.py. Use `--launch` or rely on **app.py** preloading
(which runs before the server starts).

If Drive is already mounted, you can run only `!python colab_setup.py`.

Optional environment variables (set before running, or edit defaults below):

    IMAGINATION_DRIVE_PATH  Path to repo on Drive (default: /content/drive/MyDrive/imagination-v1.1.0)
    IMAGINATION_ROOT        Content path for the repo (default: /content/imagination-v1.1.0)
    IMAGINATION_COPY        Set to "1" to copy Drive folder to /content instead of symlink (faster I/O, more disk)
    SKIP_PIP                Set to "1" to skip dependency install
    --launch                After setup: preload main LLM + start Gradio (same process as setup)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _pip_install() -> None:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if req.is_file():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        )
    else:
        pkgs = [
            "gradio",
            "transformers",
            "accelerate",
            "safetensors",
            "bitsandbytes",
            "requests",
            "beautifulsoup4",
            "ddgs",
            "diffusers",
            "bcrypt",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _drive_already_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def _can_use_colab_drive_mount() -> bool:
    """drive.mount() only works in the notebook kernel, not under !python subprocess."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def _mount_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


def _ensure_drive_mounted_colab() -> None:
    if _drive_already_mounted():
        print("[setup] Google Drive already mounted — skipping drive.mount()")
        return
    if _can_use_colab_drive_mount():
        print("[setup] Mounting Google Drive…")
        _mount_drive()
        return
    print(
        "[setup] ERROR: Google Drive is not mounted, and this script was started with !python …\n"
        "        Colab only allows drive.mount() from a normal notebook cell.\n\n"
        "        Fix — run this in the cell ABOVE, then run colab_setup again:\n\n"
        "            from google.colab import drive\n"
        "            drive.mount('/content/drive')\n\n"
        "        Then:\n"
        "            !python colab_setup.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _ensure_content_repo(drive_src: str, content_dst: str, copy: bool) -> None:
    drive_path = Path(drive_src)
    dst = Path(content_dst)

    if dst.resolve() == drive_path.resolve():
        return

    if dst.is_symlink():
        print(f"[setup] Using existing symlink: {dst}")
        return
    if dst.is_dir():
        try:
            if any(dst.iterdir()):
                print(f"[setup] Using existing folder: {dst}")
                return
        except OSError:
            pass
        shutil.rmtree(dst, ignore_errors=True)
    elif dst.exists():
        dst.unlink()

    if not drive_path.is_dir():
        raise FileNotFoundError(
            f"Repo not found on Drive: {drive_src}\n"
            "Upload your full imagination-v1.1.0 folder (with models) to that path, "
            "or set IMAGINATION_DRIVE_PATH to the correct folder."
        )

    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)

    if copy:
        print(f"[setup] Copying (this may take a while)…\n  {drive_src}\n  -> {content_dst}")
        shutil.copytree(drive_src, content_dst, dirs_exist_ok=True)
    else:
        print(f"[setup] Symlinking:\n  {drive_src}\n  -> {content_dst}")
        os.symlink(drive_src, content_dst, target_is_directory=True)


def main() -> None:
    print("Imagination v1.2 — Colab setup\n")

    if os.environ.get("SKIP_PIP", "").strip().lower() in ("1", "true", "yes"):
        print("[setup] Skipping pip (SKIP_PIP set)")
    else:
        print("[setup] Installing Python dependencies…")
        _pip_install()
        print("[setup] Dependencies OK")

    drive_src = os.environ.get(
        "IMAGINATION_DRIVE_PATH",
        "/content/drive/MyDrive/imagination-v1.1.0",
    ).rstrip("/\\")

    content_root = os.environ.get(
        "IMAGINATION_ROOT",
        "/content/imagination-v1.1.0",
    ).rstrip("/\\")

    copy_mode = os.environ.get("IMAGINATION_COPY", "").strip().lower() in ("1", "true", "yes")

    if _in_colab():
        print("[setup] Colab detected — checking Google Drive…")
        _ensure_drive_mounted_colab()
        _ensure_content_repo(drive_src, content_root, copy=copy_mode)
    else:
        print("[setup] Not running in Colab — skipping Drive mount and symlink.")
        root = Path(content_root)
        if not root.is_dir():
            print(
                f"[setup] Warning: IMAGINATION_ROOT is not a directory: {content_root}\n"
                "Set IMAGINATION_ROOT to your local repo path if needed."
            )

    os.environ["IMAGINATION_ROOT"] = content_root

    v12 = Path(content_root) / "versions" / "v1.2"
    if not v12.is_dir():
        print(f"[setup] Warning: expected app folder missing: {v12}")

    print()
    print("=" * 56)
    print("  Setup finished.")
    print(f"  IMAGINATION_ROOT = {content_root}")
    print()
    print("  Next cell — start the UI:")
    print(f"    %cd {v12}")
    print("    !python app.py")
    print("=" * 56)


if __name__ == "__main__":
    main()
